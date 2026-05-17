#!/usr/bin/env python
"""Pre-compute frame counts for all subjects in a dataset directory.

Writes a frame_counts.json mapping subject_name -> num_frames.
This avoids slow per-subject torch.load calls over NFS at training time.

Usage:
    python datasets/compute_frame_counts.py /data/cwang/remote/fmri/hcpya_preprocessed
    python datasets/compute_frame_counts.py /data/cwang/remote/fmri/hcpa_processed /data/cwang/remote/fmri/hcpd_preprocessed
"""
import json
import os
import sys
from pathlib import Path

import torch


def count_frames_for_subject(subject_path):
    blob_path = os.path.join(subject_path, "data.pt")
    if os.path.isfile(blob_path):
        try:
            peek = torch.load(blob_path, mmap=True, weights_only=True)
        except RuntimeError:
            try:
                peek = torch.load(blob_path, mmap=False, weights_only=True)
            except Exception:
                return 0
        n = int(peek["num_frames"])
        del peek
        return n
    frames = [f for f in os.listdir(subject_path) if f.startswith("frame_") and f.endswith(".pt")]
    return len(frames)


def compute_dataset(dataset_root):
    dataset_root = os.path.abspath(dataset_root)
    img_dir = os.path.join(dataset_root, "img")
    if not os.path.isdir(img_dir):
        print(f"ERROR: {img_dir} not found")
        return

    subjects = sorted(os.listdir(img_dir))
    print(f"Scanning {dataset_root}: {len(subjects)} subjects...")

    counts = {}
    skipped = 0
    for i, subj in enumerate(subjects):
        subj_path = os.path.join(img_dir, subj)
        if not os.path.isdir(subj_path):
            continue
        n = count_frames_for_subject(subj_path)
        if n == 0:
            skipped += 1
        else:
            counts[subj] = n
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(subjects)} done ({skipped} skipped)")

    out_path = os.path.join(dataset_root, "frame_counts.json")
    with open(out_path, "w") as f:
        json.dump(counts, f)
    print(f"Done: {len(counts)} subjects, {skipped} skipped -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <dataset_root> [dataset_root2 ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        compute_dataset(path)
