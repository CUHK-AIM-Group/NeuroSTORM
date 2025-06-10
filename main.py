import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import neptune
from utils.data_module import fMRIDataModule
from utils.parser import str2bool
from models.lightning_model import LightningModel
from huggingface_hub import hf_hub_download


def cli_main():

    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1234, type=int, help="random seeds. recommend aligning this argument with data split number to control randomness")
    parser.add_argument("--dataset_name", type=str, choices=["HCP1200", "ABCD", "UKB", "Cobre", "ADHD200", "HCPA", "HCPD", "UCLA", "HCPEP", "HCPTASK", "GOD", "NSD", "BOLD5000", "MOVIE", "TransDiag"], default="HCP1200")
    parser.add_argument("--downstream_task_id", type=int, default="1", help="downstream task id")
    parser.add_argument("--downstream_task_type", type=str, default="classification", help="select either classification or regression according to your downstream task")
    parser.add_argument("--task_name", type=str, default="sex", help="specify the task name")
    parser.add_argument("--loggername", default="default", type=str, help="A name of logger")
    parser.add_argument("--project_name", default="default", type=str, help="A name of project")
    parser.add_argument("--auto_resume", action='store_true', help="Whether to find the last checkpoint and resume the training")
    parser.add_argument("--resume_ckpt_path", type=str, help="A path to previous checkpoint. Use when you want to continue the training from the previous checkpoints")
    parser.add_argument("--load_model_path", type=str, help="A path to the pre-trained model weight file (.pth)")
    parser.add_argument("--test_only", action='store_true', help="specify when you want to test the checkpoints (model weights)")
    parser.add_argument("--test_ckpt_path", type=str, help="A path to the previous checkpoint that intends to evaluate (--test_only should be True)")
    parser.add_argument("--freeze_feature_extractor", action='store_true', help="Whether to freeze the feature extractor (for evaluating the pre-trained weight)")
    parser.add_argument("--print_flops", action='store_true', help="Whether to print the number of FLOPs")

    # Set dataset
    Dataset = fMRIDataModule

    # add two additional arguments
    parser = LightningModel.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    #override parameters
    max_epochs = args.max_epochs
    num_nodes = args.num_nodes
    devices = torch.cuda.device_count()
    project_name = args.project_name
    image_path = args.image_path

    if args.model == "neurostorm":
        category_dir = "neurostorm"
    elif args.model in ["swift", "tff"]:
        category_dir = "volume-based"
    elif args.model in ["braingnn", "bnt"]:
        category_dir = "roi-based"
    setattr(args, "default_root_dir", os.path.join('output', category_dir, args.project_name))

    resume_ckpt_path = None if args.resume_ckpt_path is None else args.resume_ckpt_path
    if args.resume_ckpt_path is None and args.auto_resume:
        resume_ckpt_path = os.path.join('output', category_dir, args.project_name, 'last.ckpt')
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
        )
    # callback for classification task
    elif args.downstream_task_type == "classification":
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_acc",
            filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
            mode="max",
        )
    # callback for regression task
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_mse",
            filename="checkpt-{epoch:02d}-{valid_mse:.2f}",
            save_last=True,
            mode="min",
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    # ------------ trainer -------------
    if args.grad_clip:
        print('using gradient clipping')
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
            track_grad_norm=-1,
        )
    else:
        print('not using gradient clipping')
        print(args)
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            check_val_every_n_epoch=1,
            callbacks=callbacks
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
            repo_id = "zxcvb20001/fMRI-GPT"
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

    if args.freeze_feature_extractor:
        # layers are frozen by using eval()
        model.model.eval()
        # freeze params
        for name, param in model.model.named_parameters():
            if 'output_head' not in name: # unfreeze only output head
                param.requires_grad = False
                print(f'freezing layer {name}')

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
