"""
Unified inference demo for NeuroSTORM.

Supports all five NeuroSTORM benchmark tasks in two modes:
  1. Single-file mode: run inference on a single preprocessed fMRI subject folder.
  2. Dataset mode: evaluate on a full dataset test split via Lightning Trainer.

Task 1 contains two prediction targets (age and gender), so the CLI exposes six
task names across the five benchmark categories.

Single mode runs the model over ALL windows of the scan (sliding window with the
same spacing as the training/evaluation Dataset) and averages the window outputs,
matching the full evaluation pipeline. For regression checkpoints trained with
label normalization (e.g. the Task 1 age checkpoint uses standardization), the
printed prediction is inverse-transformed back to the original label scale.

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
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from models.lightning_model import LightningModel
from datasets.data_module import fMRIDataModule
from utils.parser import str2bool
from datasets.fmri_datasets import BaseDataset, pad_to_96, resize_volume


SUPPORTED_TASKS = (
    "age", "gender",       # Task 1
    "phenotype",           # Task 2
    "diagnosis",           # Task 3
    "retrieval",           # Task 4
    "state",               # Task 5
)

STATE_CLASS_NAMES = (
    "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR",
    "RELATIONAL", "SOCIAL", "WM",
)

# Label normalization statistics of released checkpoints, keyed by
# (dataset_name, task_name). Checkpoints fitted with label scaling do not store
# the scaler; these values are recomputed from each checkpoint's training split.
#   ("HCP1200", "age"): mean/std of the 864 training subjects of split_fixed_1
#   (HCP-YA precise-age metadata), used by neurostorm_hcpya_age.ckpt.
KNOWN_LABEL_STATS = {
    ("HCP1200", "age"): {"standardization": (28.759259, 3.681461)},
}


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
        "diagnosis": {
            "task_name": "diagnosis",
            "downstream_task_id": 3,
            "downstream_task_type": "classification",
            "num_classes": int(base_hparams.get("num_classes", 2)),
        },
        "retrieval": {
            "task_name": "fmri_reid",
            "downstream_task_id": 4,
            "downstream_task_type": "classification",
            "num_classes": int(base_hparams.get("num_classes", 2)),
        },
        "state": {
            "task_name": base_hparams.get("task_name", "state_classification"),
            "downstream_task_id": 5,
            "downstream_task_type": "classification",
            "num_classes": int(base_hparams.get("num_classes", 7)),
        },
    }

    if task == "phenotype":
        phenotype_name = args.phenotype_name or base_hparams.get("task_name")
        if not phenotype_name:
            raise ValueError(
                "--phenotype_name is required when it is not stored in the checkpoint"
            )
        phenotype_type = (
            args.phenotype_type
            or base_hparams.get("downstream_task_type", "classification")
        )
        num_classes = args.num_classes or base_hparams.get("num_classes", 2)
        task_cfg["phenotype"] = {
            "task_name": phenotype_name,
            "downstream_task_id": 2,
            "downstream_task_type": phenotype_type,
            "num_classes": int(num_classes) if phenotype_type == "classification" else 1,
            "label_scaling_method": getattr(args, "label_scaling_method", None)
                or base_hparams.get("label_scaling_method", "standardization"),
        }

    return task_cfg[task]


def _classification_probabilities(output, num_classes):
    """Return one-sample class probabilities for binary or multiclass heads."""
    flat_output = output.reshape(-1)
    if num_classes == 2 and flat_output.numel() == 1:
        # NeuroSTORM binary checkpoints use one BCEWithLogits output, not two
        # softmax logits.
        positive_prob = torch.sigmoid(flat_output[0])
        return torch.stack((1.0 - positive_prob, positive_prob))

    logits = output.reshape(-1, num_classes)[0]
    return torch.softmax(logits, dim=0)


def _class_name(task, pred_class, base_hparams):
    if task == "gender":
        return ("Female", "Male")[pred_class]
    if task == "state" and pred_class < len(STATE_CLASS_NAMES):
        return STATE_CLASS_NAMES[pred_class]
    if task == "diagnosis" and base_hparams.get("dataset_name") == "ABIDE":
        return ("Control", "ASD")[pred_class]
    return None


def _label_stats(args, base_hparams, task_cfg):
    """Resolve label-scaling statistics for inverse transformation.

    Priority: CLI arguments > checkpoint hyper_parameters > KNOWN_LABEL_STATS
    (released-checkpoint defaults). Returns ``(method, stats, source)`` where
    ``stats`` is ``(mean, std)`` for standardization or ``(min, max)`` for
    minmax, or ``None`` when the task needs no inverse transform.
    """
    if task_cfg["downstream_task_type"] != "regression":
        return None
    method = task_cfg.get("label_scaling_method", "standardization")

    if method == "standardization":
        mean = getattr(args, "label_mean", None)
        std = getattr(args, "label_std", None)
        source = "command line"
        if mean is None or std is None:
            mean = base_hparams.get("label_mean")
            std = base_hparams.get("label_std")
            source = "checkpoint"
        if mean is None or std is None:
            known = KNOWN_LABEL_STATS.get(
                (base_hparams.get("dataset_name"), task_cfg["task_name"]), {}
            ).get("standardization")
            if known is not None:
                mean, std = known
                source = "released-checkpoint defaults (train split of the released model)"
        if mean is None or std is None:
            return (method, None, None)
        return (method, (float(mean), float(std)), source)

    if method == "minmax":
        vmin = getattr(args, "label_min", None)
        vmax = getattr(args, "label_max", None)
        source = "command line"
        if vmin is None or vmax is None:
            vmin = base_hparams.get("label_min")
            vmax = base_hparams.get("label_max")
            source = "checkpoint"
        if vmin is None or vmax is None:
            return (method, None, None)
        return (method, (float(vmin), float(vmax)), source)

    return None


def _inverse_transform(value, stats_entry):
    """Map a raw network output back to the original label scale."""
    method, stats, _ = stats_entry
    if stats is None:
        return None
    if method == "standardization":
        mean, std = stats
        return value * std + mean
    if method == "minmax":
        vmin, vmax = stats
        return value * (vmax - vmin) + vmin
    return None


# ---------------------------------------------------------------------------
# Single-file helpers
# ---------------------------------------------------------------------------

def _window_starts(num_frames, sample_duration, window_step, max_windows=None):
    """Window start indices, matching the training/evaluation Dataset layout."""
    if num_frames < sample_duration:
        raise ValueError(
            f"Not enough frames: have {num_frames}, need at least {sample_duration}"
        )
    starts = list(range(0, num_frames - sample_duration + 1, window_step))
    if max_windows is not None:
        starts = starts[:max_windows]
    return starts


def _load_subject_windows(subject_path, sequence_length, stride_within_seq=1,
                          stride_between_seq=20, max_windows=None):
    """Load all evaluation windows of a subject's scan.

    Windows cover the whole scan: each window spans
    ``sequence_length * stride_within_seq`` frames and consecutive windows start
    ``stride_between_seq * sequence_length * stride_within_seq`` frames apart
    (the same spacing as the training/evaluation Dataset). Window outputs should
    be averaged to obtain a subject-level prediction.

    Supports both storage layouts:
      * new int8 blob: ``subject_path/data.pt`` (mmap-based partial read)
      * legacy per-frame: ``subject_path/frame_*.pt`` (float16)

    Returns ``(windows, starts)`` with ``windows`` a float32 tensor of shape
    [n_windows, H, W, D, T].
    """
    blob_path = os.path.join(subject_path, "data.pt")
    sample_duration = sequence_length * stride_within_seq
    window_step = max(round(stride_between_seq * sample_duration), 1)

    if os.path.isfile(blob_path):
        blob = BaseDataset._load_blob_file(blob_path, mmap=True)
        frames = blob['frames']                   # int8 [T, H, W, D]
        scale = float(blob['scale'])
        num_frames = int(blob['num_frames'])
        starts = _window_starts(num_frames, sample_duration, window_step, max_windows)
        clips = [
            frames[s:s + sample_duration:stride_within_seq].to(torch.float32).mul_(scale)
            for s in starts
        ]
        windows = torch.stack(clips).permute(0, 2, 3, 4, 1)   # [B, H, W, D, T]
        return windows, starts

    # legacy per-frame format
    frame_files = [f for f in os.listdir(subject_path)
                   if f.startswith("frame_") and f.endswith(".pt")]
    num_frames = len(frame_files)
    if num_frames == 0:
        raise FileNotFoundError(f"No data.pt and no frame_*.pt found in {subject_path}")
    starts = _window_starts(num_frames, sample_duration, window_step, max_windows)

    frame_cache = {}
    clips = []
    for s in starts:
        parts = []
        for i in range(s, s + sample_duration, stride_within_seq):
            if i not in frame_cache:
                frame_cache[i] = torch.load(
                    os.path.join(subject_path, f"frame_{i}.pt"), weights_only=True
                )
            parts.append(frame_cache[i].to(torch.float32))
        clips.append(torch.cat(parts, dim=3))                 # [H, W, D, T]
    windows = torch.stack(clips)                              # [B, H, W, D, T]
    return windows, starts


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
    stride_between_seq = args.stride_between_seq or base_hparams.get("stride_between_seq", 20)
    img_size = base_hparams.get("img_size", [96, 96, 96, 20])

    print(f"Loading fMRI data from {args.fmri_path}...")
    windows, starts = _load_subject_windows(
        args.fmri_path, sequence_length, stride_within_seq,
        stride_between_seq, max_windows=args.max_windows,
    )
    print(f"Whole-scan inference: {len(starts)} windows "
          f"(window length {sequence_length * stride_within_seq} frames, "
          f"step {max(round(stride_between_seq * sequence_length * stride_within_seq), 1)} frames)")
    windows = pad_to_96(windows)
    windows = resize_volume(windows, img_size)
    windows = windows.unsqueeze(1)  # [B, C=1, H, W, D, T]

    device = torch.device(args.device)

    print("Loading model...")
    model = LightningModel.load_from_checkpoint(
        args.ckpt_path, data_module=None, **merged
    )
    model = model.to(device)
    model.eval()

    print(f"Running inference for task: {args.task}...")
    batch_size = max(int(args.window_batch_size), 1)
    outputs = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start:start + batch_size].to(device)
            if args.task == "retrieval":
                feature = model.model(batch)
                if isinstance(feature, tuple):
                    feature = feature[0]

                embedding = None
                if hasattr(model.output_head, "forward_with_features"):
                    _, embedding = model.output_head.forward_with_features(feature)
                if embedding is None:
                    embedding = feature.flatten(start_dim=2).mean(dim=2)
                outputs.append(F.normalize(embedding, p=2, dim=1).cpu())
            else:
                outputs.append(model(batch).cpu())

    if args.task == "retrieval":
        # Subject-level embedding: mean of per-window embeddings, re-normalized.
        embedding = F.normalize(torch.cat(outputs).mean(dim=0, keepdim=True), p=2, dim=1)

        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        print("Task: retrieval")
        print(f"Windows averaged: {len(starts)}")
        print(f"Embedding shape: {tuple(embedding.shape)}")
        print(f"L2 norm: {embedding.norm(dim=1).item():.4f}")
        print(f"Embedding: {embedding[0].numpy()}")
        print("=" * 50 + "\n")
        return

    # Subject-level prediction: average window logits/values, matching the
    # subject aggregation of the full evaluation pipeline.
    output = torch.cat(outputs).mean(dim=0, keepdim=True)

    if task_cfg["downstream_task_type"] == "classification":
        probs = _classification_probabilities(output, task_cfg["num_classes"])
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(f"Task: {args.task}")
        print(f"Windows averaged: {len(starts)}")
        print(f"Predicted class: {pred_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All probabilities: {probs.numpy()}")
        class_name = _class_name(args.task, pred_class, base_hparams)
        if class_name is not None:
            print(f"Predicted label: {class_name}")
    else:
        raw_value = output.item()
        stats_entry = _label_stats(args, base_hparams, task_cfg)
        pred_value = _inverse_transform(raw_value, stats_entry) if stats_entry else None

        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(f"Task: {args.task}")
        print(f"Windows averaged: {len(starts)}")
        if stats_entry is not None and stats_entry[1] is not None:
            method, stats, source = stats_entry
            print(f"Raw model output ({method}-scaled): {raw_value:.4f}")
            if method == "standardization":
                print(f"Inverse standardized with mean={stats[0]:.4f}, std={stats[1]:.4f} [{source}]")
            else:
                print(f"Inverse minmax with min={stats[0]:.4f}, max={stats[1]:.4f} [{source}]")
            if args.task == "age":
                print(f"Predicted age: {pred_value:.1f} years")
            else:
                print(f"Predicted value: {pred_value:.4f} (original label scale)")
        else:
            method = stats_entry[0] if stats_entry else task_cfg.get("label_scaling_method")
            print(f"Raw model output ({method}-scaled): {raw_value:.4f}")
            print("NOTE: this checkpoint was trained with label normalization and the")
            print("      scaler statistics are not stored in the checkpoint, so the value")
            print("      above is NOT on the original label scale.")
            print("      Standardization: label = output * std + mean")
            print("      Provide --label_mean/--label_std (or --label_min/--label_max for")
            print("      minmax) to convert, e.g. the released HCP-YA age checkpoint uses")
            print("      mean=28.7593, std=3.6815.")
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
    single.add_argument("--stride_between_seq", type=int, default=None,
                        help="Spacing between evaluation windows, in units of the window "
                             "length (defaults to checkpoint value); windows cover the whole scan")
    single.add_argument("--max_windows", type=int, default=None,
                        help="Cap the number of windows (default: use the whole scan)")
    single.add_argument("--window_batch_size", type=int, default=8,
                        help="Windows per forward pass (single mode only)")
    single.add_argument("--label_mean", type=float, default=None,
                        help="Train-set label mean for inverse standardization "
                             "(overrides checkpoint/released defaults)")
    single.add_argument("--label_std", type=float, default=None,
                        help="Train-set label std for inverse standardization "
                             "(overrides checkpoint/released defaults)")
    single.add_argument("--label_min", type=float, default=None,
                        help="Train-set label min for inverse minmax scaling")
    single.add_argument("--label_max", type=float, default=None,
                        help="Train-set label max for inverse minmax scaling")

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
                       default=None, help="Defaults to the checkpoint value")
    pheno.add_argument("--num_classes", type=int, default=None,
                       help="Defaults to the checkpoint value")
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
