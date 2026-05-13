"""
Unified inference demo for NeuroSTORM.

Supports two modes:
  1. Single-file mode: run inference on a single preprocessed fMRI subject folder.
  2. Dataset mode: evaluate on a full dataset test split via Lightning Trainer.

Usage:
    # Single fMRI subject
    python demo.py \
        --mode single \
        --ckpt_path /path/to/model.ckpt \
        --fmri_path /path/to/subject/folder \
        --task age

    # Full dataset evaluation
    python demo.py \
        --mode dataset \
        --ckpt_path /path/to/model.ckpt \
        --task age \
        --image_path /path/to/preprocessed/data
"""

import argparse
import os
import sys
import math
import torch
import numpy as np
import pytorch_lightning as pl

from models.lightning_model import LightningModel
from utils.data_module import fMRIDataModule
from utils.parser import str2bool
from datasets.fmri_datasets import pad_to_96, resize_volume


SUPPORTED_TASKS = ("age", "gender", "phenotype")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

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
        raise ValueError("Checkpoint does not contain hyper_parameters.")
    return hparams


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

def _task_config(task, args, base_hparams):
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Task must be one of {SUPPORTED_TASKS}")

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
            "label_scaling_method": getattr(args, "label_scaling_method", None)
                or base_hparams.get("label_scaling_method", "standardization"),
        },
    }

    if task == "phenotype":
        if not args.phenotype_name:
            raise ValueError("--phenotype_name is required when task is 'phenotype'")
        task_cfg["phenotype"] = {
            "task_name": args.phenotype_name,
            "downstream_task_id": 2,
            "downstream_task_type": args.phenotype_type,
            "num_classes": args.num_classes if args.phenotype_type == "classification" else 1,
            "label_scaling_method": getattr(args, "label_scaling_method", None)
                or base_hparams.get("label_scaling_method", "standardization"),
        }

    return task_cfg[task]


# ---------------------------------------------------------------------------
# Single-file helpers
# ---------------------------------------------------------------------------

def _load_single_subject(subject_path, sequence_length, stride_within_seq=1):
    """Load a clip of ``sequence_length`` frames from ``subject_path``.

    Supports both storage layouts:
      * new int8 blob: ``subject_path/data.pt`` (mmap-based partial read)
      * legacy per-frame: ``subject_path/frame_*.pt`` (float16)

    Returns a float32 tensor of shape [1, H, W, D, T].
    """
    blob_path = os.path.join(subject_path, "data.pt")
    need = sequence_length * stride_within_seq

    if os.path.isfile(blob_path):
        blob = torch.load(blob_path, mmap=True, weights_only=True)
        frames = blob['frames']                   # int8 [T, H, W, D]
        scale = float(blob['scale'])
        num_frames = int(blob['num_frames'])
        if num_frames < need:
            raise ValueError(f"Not enough frames: have {num_frames}, need at least {need}")
        clip = frames[0:need:stride_within_seq].to(torch.float32).mul_(scale)
        return clip.permute(1, 2, 3, 0).unsqueeze(0)

    # legacy per-frame format
    frame_files = [f for f in os.listdir(subject_path)
                   if f.startswith("frame_") and f.endswith(".pt")]
    num_frames = len(frame_files)
    if num_frames == 0:
        raise FileNotFoundError(f"No data.pt and no frame_*.pt found in {subject_path}")
    if num_frames < need:
        raise ValueError(f"Not enough frames: found {num_frames}, need at least {need}")
    parts = []
    for i in range(0, need, stride_within_seq):
        frame = torch.load(os.path.join(subject_path, f"frame_{i}.pt"),
                           weights_only=True).to(torch.float32)
        parts.append(frame)
    clip = torch.cat(parts, dim=3)       # [H, W, D, T]
    return clip.unsqueeze(0)             # [1, H, W, D, T]


# ---------------------------------------------------------------------------
# Mode: single
# ---------------------------------------------------------------------------

def run_single(args):
    if not os.path.isdir(args.fmri_path):
        raise NotADirectoryError(f"fMRI path must be a directory: {args.fmri_path}")

    print(f"Loading checkpoint from {args.ckpt_path}...")
    base_hparams = _load_hparams(args.ckpt_path)
    task_cfg = _task_config(args.task, args, base_hparams)

    merged = {**base_hparams, **task_cfg}
    merged.update(test_only=True, pretraining=False, use_contrastive=False, use_mae=False)

    sequence_length = args.sequence_length or base_hparams.get("sequence_length", 20)
    stride_within_seq = args.stride_within_seq or base_hparams.get("stride_within_seq", 1)
    img_size = base_hparams.get("img_size", [96, 96, 96, 20])

    print(f"Loading fMRI data from {args.fmri_path}...")
    volume = _load_single_subject(args.fmri_path, sequence_length, stride_within_seq)
    volume = pad_to_96(volume)
    volume = resize_volume(volume, img_size)

    device = torch.device(args.device)
    volume = volume.to(device)

    print("Loading model...")
    model = LightningModel.load_from_checkpoint(
        args.ckpt_path, data_module=None, **merged
    )
    model = model.to(device)
    model.eval()

    print(f"Running inference for task: {args.task}...")
    with torch.no_grad():
        output = model(volume)

        if task_cfg["downstream_task_type"] == "classification":
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

            print("\n" + "=" * 50)
            print("RESULTS")
            print("=" * 50)
            print(f"Task: {args.task}")
            print(f"Predicted class: {pred_class}")
            print(f"Confidence: {confidence:.4f}")
            print(f"All probabilities: {probs[0].cpu().numpy()}")
            if args.task == "gender":
                print(f"Predicted gender: {'Male' if pred_class == 1 else 'Female'}")
        else:
            pred_value = output.item()

            print("\n" + "=" * 50)
            print("RESULTS")
            print("=" * 50)
            print(f"Task: {args.task}")
            print(f"Predicted value: {pred_value:.4f}")
            if args.task == "age":
                print(f"Predicted age: {pred_value:.1f} years")

    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Mode: dataset
# ---------------------------------------------------------------------------

def run_dataset(args):
    base_hparams = _load_hparams(args.ckpt_path)
    task_cfg = _task_config(args.task, args, base_hparams)

    merged = dict(base_hparams)
    merged.update(task_cfg)
    merged.update(
        pretraining=False,
        use_contrastive=False,
        use_mae=False,
        test_only=True,
    )

    if args.image_path:
        merged["image_path"] = args.image_path
    if args.dataset_split_num is not None:
        merged["dataset_split_num"] = args.dataset_split_num

    merged["batch_size"] = args.batch_size or base_hparams.get("batch_size", 4)
    merged["eval_batch_size"] = args.eval_batch_size or base_hparams.get("eval_batch_size", merged["batch_size"])
    merged["num_workers"] = args.num_workers or base_hparams.get("num_workers", 8)
    merged["with_voxel_norm"] = (
        args.with_voxel_norm if args.with_voxel_norm is not None
        else base_hparams.get("with_voxel_norm", False)
    )

    merged.setdefault("train_split", 0.9)
    merged.setdefault("val_split", 0.1)
    merged.setdefault("sequence_length", base_hparams.get("sequence_length", 20))
    merged.setdefault("stride_between_seq", base_hparams.get("stride_between_seq", 1))
    merged.setdefault("stride_within_seq", base_hparams.get("stride_within_seq", 1))
    merged.setdefault("img_size", base_hparams.get("img_size", [96, 96, 96, 20]))

    data_module = fMRIDataModule(**merged)

    model = LightningModel.load_from_checkpoint(
        args.ckpt_path, data_module=data_module, **merged
    )

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
    trainer.test(model, dataloaders=data_module.test_dataloader())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuroSTORM inference demo (single file or dataset evaluation)"
    )
    parser.add_argument("--mode", required=True, choices=["single", "dataset"],
                        help="'single' for one fMRI subject, 'dataset' for full test-set evaluation")
    parser.add_argument("--ckpt_path", required=True, help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS, help="Task to perform")

    # --- single mode ---
    single = parser.add_argument_group("single-file mode")
    single.add_argument("--fmri_path", default=None,
                        help="Path to subject folder containing data.pt or frame_*.pt")
    single.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (single mode only)")
    single.add_argument("--sequence_length", type=int, default=None,
                        help="Sequence length (defaults to checkpoint value)")
    single.add_argument("--stride_within_seq", type=int, default=None,
                        help="Stride within sequence (defaults to checkpoint value)")

    # --- dataset mode ---
    ds = parser.add_argument_group("dataset mode")
    ds.add_argument("--image_path", default=None,
                    help="Root path to preprocessed data (overrides checkpoint)")
    ds.add_argument("--dataset_split_num", type=int, default=None,
                    help="Split id; defaults to checkpoint value")
    ds.add_argument("--batch_size", type=int, default=None)
    ds.add_argument("--eval_batch_size", type=int, default=None)
    ds.add_argument("--num_workers", type=int, default=None)
    ds.add_argument("--with_voxel_norm", type=str2bool, default=None)
    ds.add_argument("--devices", default=None)
    ds.add_argument("--accelerator", default=None)
    ds.add_argument("--precision", default=None)

    # --- shared ---
    parser.add_argument("--seed", type=int, default=1234)

    # --- phenotype-specific ---
    pheno = parser.add_argument_group("phenotype task")
    pheno.add_argument("--phenotype_name", default=None)
    pheno.add_argument("--phenotype_type", choices=["classification", "regression"],
                       default="classification")
    pheno.add_argument("--num_classes", type=int, default=2)
    pheno.add_argument("--label_scaling_method", choices=["standardization", "minmax"],
                       default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    if args.mode == "single":
        if not args.fmri_path:
            raise ValueError("--fmri_path is required for single mode")
        run_single(args)
    else:
        run_dataset(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
