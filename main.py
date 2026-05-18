import os
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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
    parser.add_argument("--test_ckpt_path", type=str, help="A path to the previous checkpoint that intends to evaluate (--test_only should be True)")
    parser.add_argument("--print_flops", action='store_true', help="Whether to print the number of FLOPs")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, uses all available GPUs")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use. If not specified, uses all available GPUs or those specified by --gpu_ids")

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
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
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
    elif args.model in ["swift", "tff"]:
        category_dir = "volume-based"
    elif args.model in ["braingnn", "lggnn", "ibgnn"]:
        category_dir = "graph-based"
    elif args.model in ["bnt", "combraintf", "brainnetcnn"]:
        category_dir = "fc-based"
    else:
        category_dir = "other"
    setattr(args, "default_root_dir", os.path.join('output', category_dir, args.project_name))

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

    use_custom_sampler = getattr(args, 'sampling_strategy', None) is not None

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
        )

    # ------------ model -------------
    model = LightningModel(data_module = data_module, **vars(args))

    path = None
    if args.load_model_path is not None:
        if os.path.exists(args.load_model_path):
            print(f'loading model from {args.load_model_path}')
            path = args.load_model_path
        else:
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
        ckpt = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if 'model.' in k: #transformer-related layers
                new_state_dict[k.removeprefix("model.")] = v
        model.model.load_state_dict(new_state_dict, strict=False)

    if getattr(args, 'use_strd', False):
        if args.use_mae and args.model == 'neurostorm':
            print(f'[STRD] enabled for MAE: l_spat={args.strd_l_spat}, l_temp={args.strd_l_temp}')
        else:
            print('[STRD] WARNING: --use_strd requires --use_mae and --model neurostorm; silently disabled')

    if args.use_prompt_tuning and getattr(args, 'pretraining', False):
        print('[TPT] WARNING: --use_prompt_tuning is ignored during pretraining')

    if args.use_prompt_tuning and not getattr(args, 'pretraining', False):
        # Task-specific Prompt Tuning: freeze backbone except per-block prompts;
        # the output_head remains fully trainable (it's a separate module under `model`).
        model.model.eval()
        n_prompt, n_frozen = 0, 0
        for name, param in model.model.named_parameters():
            if 'prompt' in name:
                param.requires_grad = True
                n_prompt += param.numel()
            else:
                param.requires_grad = False
                n_frozen += param.numel()

        # Also count output_head (trainable by default; lives outside model.model)
        n_head = 0
        if getattr(model, 'output_head', None) is not None:
            n_head = sum(p.numel() for p in model.output_head.parameters() if p.requires_grad)

        n_trainable_total = n_prompt + n_head
        n_full = n_prompt + n_frozen + n_head  # full-model parameter count
        pct = (100.0 * n_trainable_total / n_full) if n_full > 0 else 0.0
        print(f'[TPT] prompt params:   {n_prompt:>12,}')
        print(f'[TPT] head params:     {n_head:>12,}')
        print(f'[TPT] trainable total: {n_trainable_total:>12,}')
        print(f'[TPT] frozen backbone: {n_frozen:>12,}')
        print(f'[TPT] full model:      {n_full:>12,}')
        print(f'[TPT] trainable ratio: {pct:.2f}%  (vs. full fine-tuning)')
        if n_prompt == 0:
            print('[TPT] WARNING: no parameters with "prompt" in name found. '
                  'Did you set --prompt_len > 0 with --model neurostorm?')

    # ------------ run -------------
    if args.test_only:
        trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path) # dataloaders=data_module
    else:
        if args.resume_ckpt_path is None:
            # New run
            trainer.fit(model, datamodule=data_module)
        else:
            # Resume existing run
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
