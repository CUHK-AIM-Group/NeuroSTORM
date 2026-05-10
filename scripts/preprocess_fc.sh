#!/bin/bash
# Compute ROI time series + functional connectivity matrices for FC / graph
# based models (BNT, BrainGNN, LG-GNN, Com-BrainTF, IBGNN, BrainNetCNN).
#
# This produces:
#   <image_path>/roi/<atlas>/                # per-subject ROI time series
#   <image_path>/fc/<atlas>/<fc_type>/       # FC matrices
#
# Usage:
#   bash scripts/preprocess_fc.sh <dataset> [atlas] [fc_types...]
#
# Examples:
#   bash scripts/preprocess_fc.sh hcp1200
#   bash scripts/preprocess_fc.sh hcp1200 cc200 correlation partial_correlation
#   bash scripts/preprocess_fc.sh adhd200 aal correlation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs/datasets"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/preprocess_fc.sh <dataset> [atlas] [fc_types...]"
    echo "Available datasets:"
    ls "${CONFIG_DIR}" 2>/dev/null | sed 's/.yaml$//'
    exit 1
fi

DATASET="$1"
ATLAS="${2:-cc200}"
shift 1 || true
if [[ $# -gt 1 ]]; then
    shift 1
    FC_TYPES=("$@")
else
    FC_TYPES=(correlation partial_correlation)
fi

DATASET_CONFIG="${CONFIG_DIR}/${DATASET}.yaml"
if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "Error: dataset config not found: ${DATASET_CONFIG}"
    exit 1
fi

# Pull image_path / dataset_name out of the yaml (no external parser dependency).
yaml_get() {
    grep -E "^[[:space:]]*$2:" "$1" 2>/dev/null | head -1 \
        | sed 's/^[^:]*:[[:space:]]*//' | sed 's/^"\(.*\)"$/\1/' | sed 's/[[:space:]]*$//'
}

DATASET_NAME="$(yaml_get "${DATASET_CONFIG}" dataset_name)"
IMAGE_PATH="$(yaml_get "${DATASET_CONFIG}" image_path)"
if [[ -z "${IMAGE_PATH}" ]]; then
    echo "Error: image_path missing in ${DATASET_CONFIG}"
    exit 1
fi

NUM_PROCESSES="${NUM_PROCESSES:-8}"

echo "=============================================="
echo " FC / graph preprocessing"
echo "=============================================="
echo " Dataset:     ${DATASET_NAME}"
echo " Image path:  ${IMAGE_PATH}"
echo " Atlas:       ${ATLAS}"
echo " FC types:    ${FC_TYPES[*]}"
echo " Processes:   ${NUM_PROCESSES}"
echo "=============================================="

# Step 1: ROI time series from raw fMRI nii files.
# Expects <IMAGE_PATH>/img/<dataset_name>/<subject>/*.nii.gz layout (see the
# generate_roi_data_from_nii.py docstring for details).
echo "Step 1: ROI time series -> ${IMAGE_PATH}/roi/${ATLAS}"
python "${ROOT_DIR}/datasets/generate_roi_data_from_nii.py" \
    --atlas_names "${ATLAS}" \
    --dataset_names "${DATASET_NAME}" \
    --fmri_dir "${IMAGE_PATH}/img" \
    --atlas_dir "${ROOT_DIR}/datasets/atlas" \
    --output_dir "${IMAGE_PATH}/roi/${ATLAS}" \
    --num_processes "${NUM_PROCESSES}"

# Step 2: FC matrices.
echo "Step 2: FC matrices -> ${IMAGE_PATH}/fc/${ATLAS}"
python "${ROOT_DIR}/datasets/compute_fc.py" \
    --roi_dir "${IMAGE_PATH}/roi/${ATLAS}" \
    --output_dir "${IMAGE_PATH}/fc/${ATLAS}" \
    --atlas_name "${ATLAS}" \
    --fc_types "${FC_TYPES[@]}" \
    --num_processes "${NUM_PROCESSES}"

echo "Done. Train FC / graph models via:"
echo "  bash scripts/run_experiment.sh --model <bnt|braingnn|...> --dataset ${DATASET} --task <taskN> --mode <finetune|train_scratch>"
