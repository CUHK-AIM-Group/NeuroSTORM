import os
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

_early_parser = ArgumentParser(add_help=False)
_early_parser.add_argument("--gpu_ids", type=str, default=None)
_early_args, _ = _early_parser.parse_known_args()
if _early_args.gpu_ids is not None and "LOCAL_RANK" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_args.gpu_ids

# Silence noisy third-party warnings before any heavy imports so that the
# `FutureWarning`s raised when mamba_ssm loads its @custom_fwd/@custom_bwd
# decorators are suppressed at import time.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.custom_(fwd|bwd).*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Checkpoint directory .* exists and is not empty.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*No device id is provided via `init_process_group`.*",
    category=UserWarning,
)
# Tensor Cores hint from Lightning; address it explicitly below.
warnings.filterwarnings(
    "ignore",
    message=r".*You are using a CUDA device .* that has Tensor Cores.*",
)
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")

# Tag DDP-spawned workers so setup-time prints only happen in the parent
# process. Lightning's ddp strategy sets LOCAL_RANK in the child env before
# re-exec'ing main.py, so its presence is a reliable worker signal.
if "LOCAL_RANK" in os.environ:
    os.environ["NEUROSTORM_IS_WORKER"] = "1"

import torch
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# Silence Lightning's INFO-level chatter: "GPU available ... / TPU available
# ... / Initializing distributed / LOCAL_RANK: X - CUDA_VISIBLE_DEVICES ..."
# and the per-rank "Global seed set to X" messages.
import logging as _logging
_logging.getLogger("pytorch_lightning").setLevel(_logging.WARNING)
_logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(_logging.WARNING)
_logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(_logging.WARNING)
_logging.getLogger("lightning_fabric").setLevel(_logging.WARNING)
_logging.getLogger("lightning_fabric.utilities.seed").setLevel(_logging.WARNING)

# Honour Lightning's Tensor Core hint once so we don't pay the precision
# warning on every rank.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

import neptune
from datasets.data_module import fMRIDataModule
from utils.parser import str2bool
from models.lightning_model import LightningModel
from huggingface_hub import hf_hub_download


def cli_main():

    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1234, type=int, help="random seeds. recommend aligning this argument with data split number to control randomness")
    parser.add_argument("--dataset_name", type=str, default="HCP1200",
                        help="Dataset name. Supported: HCP1200, HCPA, HCPD, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, NSD, BOLD5000, MOVIE, TransDiag. "
                             "In --pretraining mode a comma-separated list (e.g. 'HCP1200,HCPA,HCPD,ABCD,UKB') is accepted; --image_path must then list the same number of entries.")
    parser.add_argument("--downstream_task_id", type=int, default="1", help="downstream task id")
    parser.add_argument("--downstream_task_type", type=str, default="classification", help="select either classification or regression according to your downstream task")
    parser.add_argument("--task_name", type=str, default="sex", help="specify the task name")
    parser.add_argument("--loggername", default="tensorboard", type=str, help="A name of logger")
    parser.add_argument("--project_name", default="default", type=str, help="A name of project")
    parser.add_argument("--auto_resume", action='store_true', help="Whether to find the last checkpoint and resume the training")
    parser.add_argument("--resume_ckpt_path", type=str, help="A path to previous checkpoint. Use when you want to continue the training from the previous checkpoints")
    parser.add_argument("--load_model_path", type=str, help="A path to the pre-trained model weight file (.pth)")
    parser.add_argument("--test_only", action='store_true', help="Whether to test the checkpoints (model weights)")
    parser.add_argument("--validate_only", action='store_true', help="Run trainer.validate on the ckpt and exit (covers both val + test loaders)")
    parser.add_argument("--test_ckpt_path", type=str, help="A path to the previous checkpoint that intends to evaluate (--test_only should be True)")
    parser.add_argument("--print_flops", action='store_true', help="Whether to print the number of FLOPs")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, uses all available GPUs")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use. If not specified, uses all available GPUs or those specified by --gpu_ids")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory (bypasses category_dir/project_name logic)")

    # Set dataset
    Dataset = fMRIDataModule

    # add two additional arguments
    parser = LightningModel.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Handle GPU selection
    # Priority: --gpu_ids > --num_gpus > CUDA_VISIBLE_DEVICES > all GPUs
    if args.gpu_ids is not None:
        num_gpus = len(args.gpu_ids.split(','))
    elif args.num_gpus is not None:
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        num_gpus = torch.cuda.device_count()

    #override parameters
    max_epochs = args.max_epochs
    num_nodes = args.num_nodes
    devices = num_gpus if num_gpus > 0 else None
    project_name = args.project_name
    image_path = args.image_path

    if args.model == "neurostorm":
        category_dir = "neurostorm"
    elif args.model in ["swift", "tff", "band"]:
        category_dir = "volume-based"
    elif args.model in ("brainjepa", "brainlm"):
        category_dir = "roi-based"
    elif args.model in ["braingnn", "lggnn", "ibgnn"]:
        category_dir = "graph-based"
    elif args.model in ["bnt", "combraintf", "brainnetcnn"]:
        category_dir = "fc-based"
    else:
        category_dir = "other"

    if args.output_dir is not None:
        setattr(args, "default_root_dir", args.output_dir)
    else:
        setattr(args, "default_root_dir", os.path.join('nas', 'output', category_dir, args.project_name))

    resume_ckpt_path = None if args.resume_ckpt_path is None else args.resume_ckpt_path
    if args.resume_ckpt_path is None and args.auto_resume:
        ckpt_candidate = os.path.join('output', category_dir, args.project_name, 'last.ckpt')
        if os.path.exists(ckpt_candidate):
            resume_ckpt_path = ckpt_candidate
    setattr(args, "resume_ckpt_path", resume_ckpt_path)

    if args.resume_ckpt_path is not None:
        # resume previous experiment
        from utils.neptune_utils import get_prev_args
        args = get_prev_args(resume_ckpt_path, args)
        exp_id = None
        # override max_epochs if you hope to prolong the training
        args.project_name = project_name
        args.max_epochs = max_epochs
        args.num_nodes = num_nodes
        args.devices = torch.cuda.device_count()
        args.image_path = image_path
    else:
        exp_id = None

    # ------------ data -------------
    data_module = Dataset(**vars(args))
    pl.seed_everything(args.seed)

    if args.task_name == 'fmri_reid':
        args.num_classes = data_module.hparams.num_classes
        print(f'ReID task: num_classes set to {args.num_classes}')

    # ------------ logger -------------
    if args.loggername == "tensorboard":
        dirpath = args.default_root_dir
        logger = TensorBoardLogger(dirpath)
    elif args.loggername == "neptune":
        API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
        run = neptune.init(api_token=API_KEY, project=args.project_name, capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False, run=exp_id)
        
        if exp_id == None:
            setattr(args, "id", run.fetch()['sys']['id'])

        logger = NeptuneLogger(run=run, log_model_checkpoints=False)
        dirpath = os.path.join(args.default_root_dir, logger.version)
    else:
        raise Exception("Wrong logger name.")

    # ------------ callbacks -------------
    # callback for pretraining task
    if args.pretraining:
        if getattr(args, 'use_contrastive', False):
            # Contrastive SSL: valid_loss on the InfoNCE objective is uninformative
            # (in-batch negatives at eval time collapse the metric), so we monitor
            # train_loss instead and save at every train epoch end.
            checkpoint_callback = ModelCheckpoint(
                dirpath=dirpath,
                monitor="train_loss",
                filename="checkpt-{epoch:02d}-{train_loss:.4f}",
                save_last=True,
                mode="min",
                save_on_train_epoch_end=True,
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=dirpath,
                monitor="valid_loss",
                filename="checkpt-{epoch:02d}-{valid_loss:.2f}",
                save_last=True,
                mode="min",
                save_on_train_epoch_end=False,
            )
    # callback for classification task
    elif args.downstream_task_type == "classification":
        if args.task_name == 'fmri_reid':
            checkpoint_callback = ModelCheckpoint(
                dirpath=dirpath,
                monitor="valid_reid_top1",
                filename="checkpt-{epoch:02d}-{valid_reid_top1:.4f}",
                save_last=True,
                mode="max",
                save_on_train_epoch_end=False,
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=dirpath,
                monitor="valid_acc",
                filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
                save_last=True,
                mode="max",
                save_on_train_epoch_end=False,
            )
    # callback for regression task
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_mse",
            filename="checkpt-{epoch:02d}-{valid_mse:.2f}",
            save_last=True,
            mode="min",
            save_on_train_epoch_end=False,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    from utils._step_timer import maybe_attach as _maybe_attach_step_timer
    _maybe_attach_step_timer(callbacks)

    # ------------ trainer -------------
    # Determine accelerator and devices
    if torch.cuda.is_available() and num_gpus > 0:
        accelerator = 'gpu'
        trainer_devices = num_gpus
        strategy = 'ddp' if num_gpus > 1 else None
    else:
        accelerator = 'cpu'
        trainer_devices = None
        strategy = None

    use_custom_sampler = getattr(args, 'reweighting_strategy', None) is not None

    # Contrastive SSL: skip val/test loops entirely (the InfoNCE eval metric is
    # uninformative — see ckpt callback above). MAE still benefits from val/test.
    skip_eval = bool(getattr(args, 'pretraining', False) and getattr(args, 'use_contrastive', False))
    extra_trainer_kwargs = {}
    if skip_eval:
        extra_trainer_kwargs.update(limit_val_batches=0, limit_test_batches=0, num_sanity_val_steps=0)

    if args.grad_clip:
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
            track_grad_norm=-1,
            accelerator=accelerator,
            devices=trainer_devices,
            strategy=strategy,
            replace_sampler_ddp=not use_custom_sampler,
            **extra_trainer_kwargs,
        )
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            check_val_every_n_epoch=1,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=trainer_devices,
            strategy=strategy,
            replace_sampler_ddp=not use_custom_sampler,
            **extra_trainer_kwargs,
        )

    # ------------ model -------------
    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        try:
            _peek = torch.load(args.load_model_path, map_location="cpu", weights_only=False)
            _hp = _peek.get("hyper_parameters", {})
            _ckpt_tpt = _hp.get("tpt_strategy", "none")
            if _ckpt_tpt and _ckpt_tpt != "none" and getattr(args, "tpt_strategy", "none") == "none":
                args.tpt_strategy = _ckpt_tpt
                args.prompt_len = int(_hp.get("prompt_len", args.prompt_len))
                print(f"[TPT auto-load] inherited from ckpt: tpt_strategy={args.tpt_strategy}, prompt_len={args.prompt_len}")
            del _peek, _hp
        except Exception as e:
            print(f"[TPT auto-load] skipped (could not peek ckpt hparams): {e}")

    model = LightningModel(data_module = data_module, **vars(args))

    path = None
    if args.load_model_path is not None:
        if os.path.exists(args.load_model_path):
            print(f'loading model from {args.load_model_path}')
            path = args.load_model_path
        else:
            if args.model in ('brainjepa', 'band', 'brainlm', 'tff'):
                raise FileNotFoundError(f"Provide a local framework checkpoint: {args.load_model_path}")
            print('cannot find the ckpt file. try to download model from huggingface')
            repo_id = "zxcvb20001/NeuroSTORM"
            if args.model == 'neurostorm':
                filename = "neurostorm/{}".format(os.path.basename(args.load_model_path))
            elif args.model in ['swift']:
                filename = "volume-based/{}/{}".format(args.model, os.path.basename(args.load_model_path))
            
            try:
                path = hf_hub_download(repo_id=repo_id, filename=filename)
            except:
                print('train from scratch')
    
    if path is not None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if 'model.' in k: #transformer-related layers
                new_state_dict[k.removeprefix("model.")] = v
        model.model.load_state_dict(new_state_dict, strict=args.model in ('brainjepa', 'band', 'brainlm', 'tff'))

    if getattr(args, 'use_strd', False):
        if args.use_mae and args.model == 'neurostorm':
            print(f'[STRD] enabled for MAE: l_spat={args.strd_l_spat}, l_temp={args.strd_l_temp}')
        else:
            print('[STRD] WARNING: --use_strd requires --use_mae and --model neurostorm; silently disabled')

    if args.tpt_strategy != 'none' and getattr(args, 'pretraining', False):
        print(f"[TPT] WARNING: --tpt_strategy={args.tpt_strategy} ignored during pretraining")
    elif args.tpt_strategy != 'none':
        from models.peft import apply_tpt
        apply_tpt(model, args.tpt_strategy)

    # ------------ run -------------
    if args.validate_only:
        trainer.validate(model, datamodule=data_module, ckpt_path=args.test_ckpt_path)
    elif args.test_only:
        trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path) # dataloaders=data_module
    else:
        if args.resume_ckpt_path is None:
            # New run
            print(f'[diag] FIT_START fresh max_epochs={args.max_epochs}', flush=True)
            trainer.fit(model, datamodule=data_module)
        else:
            # Resume existing run
            print(f'[diag] FIT_START resume_from={args.resume_ckpt_path} max_epochs={args.max_epochs}', flush=True)
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
