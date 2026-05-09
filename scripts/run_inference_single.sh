#!/usr/bin/env bash
# Example script for running NeuroSTORM inference demo

set -euo pipefail

# TODO: Update these paths for your environment
CKPT_PATH="/path/to/model.ckpt"
FMRI_PATH="/path/to/subject/folder"  # folder containing frame_*.pt or frames.h5
IMAGE_PATH="/path/to/preprocessed/data"

# =============================================
# Single-file mode: one subject at a time
# =============================================

# Example 1: Age prediction (single subject)
echo "Running age prediction (single)..."
python demo.py \
  --mode single \
  --ckpt_path "${CKPT_PATH}" \
  --fmri_path "${FMRI_PATH}" \
  --task age \
  --device cuda

# Example 2: Gender classification (single subject)
echo "Running gender classification (single)..."
python demo.py \
  --mode single \
  --ckpt_path "${CKPT_PATH}" \
  --fmri_path "${FMRI_PATH}" \
  --task gender \
  --device cuda

# Example 3: Phenotype prediction (single subject, regression)
echo "Running phenotype prediction (single)..."
python demo.py \
  --mode single \
  --ckpt_path "${CKPT_PATH}" \
  --fmri_path "${FMRI_PATH}" \
  --task phenotype \
  --phenotype_name "CogTotalComp_Unadj" \
  --phenotype_type regression \
  --device cuda

# =============================================
# Dataset mode: full test-set evaluation
# =============================================

# Example 4: Age regression on full test split
echo "Running age evaluation (dataset)..."
python demo.py \
  --mode dataset \
  --ckpt_path "${CKPT_PATH}" \
  --task age \
  --image_path "${IMAGE_PATH}"
