#!/usr/bin/env bash
# Examples for running inference with demo.py.
#
# demo.py supports all five benchmark tasks in two modes:
#   --mode single   : one preprocessed fMRI subject folder (data.pt)
#   --mode dataset  : full dataset test split (uses image_path)
#
# Task 1 has separate age and gender targets. The complete --task choices are:
# age, gender, phenotype, diagnosis, retrieval, and state.
#
# Fill in the placeholder paths before running.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_PY="${ROOT_DIR}/demo.py"

# ---------------------------------------------------------------------------
# Paths — edit these for your environment
# ---------------------------------------------------------------------------
CKPT_PATH="/path/to/model.ckpt"              # checkpoint for the chosen task
FMRI_PATH="/path/to/subject/folder"          # single-subject folder (data.pt)
IMAGE_PATH="/path/to/preprocessed/data"      # dataset root for --mode dataset

# ---------------------------------------------------------------------------
# Single-subject inference
# ---------------------------------------------------------------------------

# Age regression
python "${DEMO_PY}" \
    --mode single \
    --ckpt_path "${CKPT_PATH}" \
    --fmri_path "${FMRI_PATH}" \
    --task age \
    --device cuda \
    --label_scaling_method standardization

# Gender classification
# python "${DEMO_PY}" \
#     --mode single \
#     --ckpt_path "${CKPT_PATH}" \
#     --fmri_path "${FMRI_PATH}" \
#     --task gender \
#     --device cuda

# Phenotype regression
# python "${DEMO_PY}" \
#     --mode single \
#     --ckpt_path "${CKPT_PATH}" \
#     --fmri_path "${FMRI_PATH}" \
#     --task phenotype \
#     --phenotype_name CogTotalComp_Unadj \
#     --phenotype_type regression \
#     --device cuda \
#     --label_scaling_method standardization

# Disease diagnosis
# python "${DEMO_PY}" \
#     --mode single \
#     --ckpt_path "${CKPT_PATH}" \
#     --fmri_path "${FMRI_PATH}" \
#     --task diagnosis \
#     --device cuda

# fMRI retrieval embedding
# python "${DEMO_PY}" \
#     --mode single \
#     --ckpt_path "${CKPT_PATH}" \
#     --fmri_path "${FMRI_PATH}" \
#     --task retrieval \
#     --device cuda

# Task-fMRI state classification
# python "${DEMO_PY}" \
#     --mode single \
#     --ckpt_path "${CKPT_PATH}" \
#     --fmri_path "${FMRI_PATH}" \
#     --task state \
#     --device cuda

# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

# Age regression on the full test split
# python "${DEMO_PY}" \
#     --mode dataset \
#     --ckpt_path "${CKPT_PATH}" \
#     --task age \
#     --image_path "${IMAGE_PATH}" \
#     --accelerator gpu \
#     --devices 1 \
#     --precision 32 \
#     --label_scaling_method standardization
