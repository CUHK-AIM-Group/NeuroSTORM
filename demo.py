import argparse
import os
import sys
import torch
import pytorch_lightning as pl

from utils.data_module import fMRIDataModule
from models.lightning_model import LightningModel
from utils.parser import str2bool


SUPPORTED_TASKS = ("age", "gender", "phenotype")


def _coerce_precision(val):
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return val


def _load_hparams(ckpt_path: str) -> dict:
    state = torch.load(ckpt_path, map_location="cpu")
    hparams = state.get("hyper_parameters")
    if hparams is None:
        raise ValueError("Checkpoint does not contain hyper_parameters. Please provide a Lightning checkpoint.")
    return hparams


def _task_overrides(args, base_hparams):
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(f"Only tasks {SUPPORTED_TASKS} are supported.")

    if base_hparams.get("dataset_name") and base_hparams["dataset_name"] != "HCP1200":
        raise ValueError("This demo only supports the HCP-YA (HCP1200) dataset.")

    task_cfg = {
        "gender": {
            "task_name": "sex",
            "downstream_task_id": 1,
            "downstream_task_type": "classification",
            "num_classes": 2,
        },
        "age": {
            "task_name": "age",
            "downstream_task_id": 1,
            "downstream_task_type": "regression",
            "num_classes": 1,
            "label_scaling_method": args.label_scaling_method or base_hparams.get("label_scaling_method", "standardization"),
        },
    }

    if args.task == "phenotype":
        if not args.phenotype_name:
            raise ValueError("--phenotype_name is required when task is 'phenotype'.")
        task_cfg["phenotype"] = {
            "task_name": args.phenotype_name,
            "downstream_task_id": 2,
            "downstream_task_type": args.phenotype_type,
            "num_classes": args.num_classes if args.phenotype_type == "classification" else 1,
            "label_scaling_method": args.label_scaling_method or base_hparams.get("label_scaling_method", "standardization"),
        }

    merged = {
        "dataset_name": "HCP1200",
        "pretraining": False,
        "use_contrastive": False,
        "use_mae": False,
        "test_only": True,
        "freeze_feature_extractor": args.freeze_feature_extractor,
        **task_cfg[args.task],
    }
    return merged


def _merge_hparams(args, base_hparams):
    merged = dict(base_hparams)

    merged.update(_task_overrides(args, base_hparams))

    if args.image_path:
        merged["image_path"] = args.image_path

    if args.dataset_split_num is not None:
        merged["dataset_split_num"] = args.dataset_split_num

    merged["batch_size"] = args.batch_size or base_hparams.get("batch_size", 4)
    merged["eval_batch_size"] = args.eval_batch_size or base_hparams.get("eval_batch_size", merged["batch_size"])
    merged["num_workers"] = args.num_workers or base_hparams.get("num_workers", 8)
    merged["with_voxel_norm"] = args.with_voxel_norm if args.with_voxel_norm is not None else base_hparams.get("with_voxel_norm", False)

    merged.setdefault("train_split", 0.9)
    merged.setdefault("val_split", 0.1)
    merged.setdefault("sequence_length", base_hparams.get("sequence_length", 20))
    merged.setdefault("stride_between_seq", base_hparams.get("stride_between_seq", 1))
    merged.setdefault("stride_within_seq", base_hparams.get("stride_within_seq", 1))
    merged.setdefault("img_size", base_hparams.get("img_size", [96, 96, 96, 20]))

    return merged


def _build_trainer(args, base_hparams):
    precision = _coerce_precision(args.precision) or base_hparams.get("precision", 32)
    devices = args.devices or base_hparams.get("devices", "auto")
    accelerator = args.accelerator or base_hparams.get("accelerator", "auto")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
    )
    
    return trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Inference-only demo for HCP-YA tasks")
    parser.add_argument("--ckpt_path", required=True, help="Path to a trained Lightning checkpoint (.ckpt)")
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS, help="Task to evaluate")
    parser.add_argument("--image_path", default=None, help="Root path to the preprocessed HCP-YA data (overrides checkpoint)")
    parser.add_argument("--dataset_split_num", type=int, default=None, help="Split id used during training; defaults to checkpoint value")
    parser.add_argument("--batch_size", type=int, default=None, help="Eval batch size; defaults to checkpoint value")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Eval batch size for validation/test; defaults to batch_size or checkpoint")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers; defaults to checkpoint value")
    parser.add_argument("--with_voxel_norm", type=str2bool, default=None, help="Whether to use voxel norm; overrides checkpoint when set")
    parser.add_argument("--freeze_feature_extractor", action="store_true", help="Freeze transformer backbone during inference")
    parser.add_argument("--devices", default=None, help="Devices for inference (e.g., 1, 'auto', '0,1')")
    parser.add_argument("--accelerator", default=None, help="Lightning accelerator, defaults to 'auto'")
    parser.add_argument("--precision", default=None, help="Precision setting, e.g., 16, 32, or 'bf16' (defaults to checkpoint)")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for determinism")

    # Phenotype-specific
    parser.add_argument("--phenotype_name", default=None, help="Column name in metadata for phenotype prediction")
    parser.add_argument("--phenotype_type", choices=["classification", "regression"], default="classification", help="Loss type for phenotype prediction")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes when phenotype_type is classification")
    parser.add_argument("--label_scaling_method", choices=["standardization", "minmax"], default=None, help="Label scaling for regression tasks")

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt_path}")

    base_hparams = _load_hparams(args.ckpt_path)
    merged_hparams = _merge_hparams(args, base_hparams)

    if merged_hparams.get("dataset_name") != "HCP1200":
        raise ValueError("Only the HCP-YA (HCP1200) dataset is supported in this demo.")

    data_module = fMRIDataModule(**merged_hparams)

    model = LightningModel.load_from_checkpoint(
        args.ckpt_path,
        data_module=data_module,
        **merged_hparams,
    )

    trainer = _build_trainer(args, base_hparams)
    trainer.test(model, dataloaders=data_module.test_dataloader())

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Inference failed: {exc}")
        sys.exit(1)
