#!/usr/bin/env bash
# Simple examples for running inference-only demo.py on HCP-YA tasks.
# Fill in DATA_ROOT and checkpoint paths before running.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_PY="${ROOT_DIR}/demo.py"

# TODO: update these paths for your environment
DATA_ROOT="/path/to/HCP1200"            # e.g., /data/hcp1200-preprocessed
CKPT_GENDER="/path/to/gender.ckpt"      # checkpoint trained for gender
CKPT_AGE="/path/to/age.ckpt"            # checkpoint trained for age
CKPT_PHENO="/path/to/phenotype.ckpt"    # checkpoint trained for phenotype

# Gender classification
python "${DEMO_PY}" \
  --ckpt_path "${CKPT_GENDER}" \
  --task gender \
  --image_path "${DATA_ROOT}" \
  --devices 1 \
  --precision 32

# Age regression
python "${DEMO_PY}" \
  --ckpt_path "${CKPT_AGE}" \
  --task age \
  --image_path "${DATA_ROOT}" \
  --devices 1 \
  --precision 32 \
  --label_scaling_method standardization

# Phenotype prediction
python "${DEMO_PY}" \
  --ckpt_path "${CKPT_PHENO}" \
  --task phenotype \
  --phenotype_name some_regression_column \
  --phenotype_type regression \
  --image_path "${DATA_ROOT}" \
  --devices 1 \
  --precision 32 \
  --label_scaling_method standardization
