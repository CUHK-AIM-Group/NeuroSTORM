#!/bin/bash
# Compute ROI time series + functional connectivity matrices for FC / graph
# based models (BNT, BrainGNN, LG-GNN, Com-BrainTF, IBGNN, BrainNetCNN).
#
# This produces:
#   <output_dir>/roi/<atlas>/                # per-subject ROI time series
#   <output_dir>/fc/<atlas>/<fc_type>/       # FC matrices
#
# Usage:
#   bash scripts/preprocess_fc.sh <dataset> [--format blob|nii] [--atlas ATLAS] [--fc_types TYPE...]
#
# Examples:
#   bash scripts/preprocess_fc.sh hcp1200
#   bash scripts/preprocess_fc.sh hcp1200 --format blob --atlas cc200
#   bash scripts/preprocess_fc.sh adhd200 --format nii --atlas aal3 --fc_types correlation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs/datasets"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/preprocess_fc.sh <dataset> [--format blob|nii] [--atlas ATLAS] [--fc_types TYPE...]"
    echo "Available datasets:"
    ls "${CONFIG_DIR}" 2>/dev/null | sed 's/.yaml$//'
    exit 1
fi

DATASET="$1"; shift
FORMAT="blob"
ATLAS="cc200"
FC_TYPES=(correlation partial_correlation)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --format)  FORMAT="$2"; shift 2 ;;
        --atlas)   ATLAS="$2"; shift 2 ;;
        --fc_types) shift; FC_TYPES=(); while [[ $# -gt 0 && ! "$1" == --* ]]; do FC_TYPES+=("$1"); shift; done ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

DATASET_CONFIG="${CONFIG_DIR}/${DATASET}.yaml"
if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "Error: dataset config not found: ${DATASET_CONFIG}"
    exit 1
fi

yaml_get() {
    grep -E "^[[:space:]]*$2:" "$1" 2>/dev/null | head -1 \
        | sed 's/^[^:]*:[[:space:]]*//' | sed 's/^"\(.*\)"$/\1/' | sed 's/[[:space:]]*$//'
}

IMAGE_PATH="$(yaml_get "${DATASET_CONFIG}" image_path)"
if [[ -z "${IMAGE_PATH}" ]]; then
    echo "Error: image_path missing in ${DATASET_CONFIG}"
    exit 1
fi

NUM_PROCESSES="${NUM_PROCESSES:-4}"

echo "=============================================="
echo " FC / graph preprocessing"
echo "=============================================="
echo " Dataset:     ${DATASET}"
echo " Image path:  ${IMAGE_PATH}"
echo " Format:      ${FORMAT}"
echo " Atlas:       ${ATLAS}"
echo " FC types:    ${FC_TYPES[*]}"
echo " Processes:   ${NUM_PROCESSES}"
echo "=============================================="

python "${ROOT_DIR}/datasets/compute_roi_fc.py" \
    --input_dir "${IMAGE_PATH}/img" \
    --input_format "${FORMAT}" \
    --output_dir "${IMAGE_PATH}" \
    --atlas_names "${ATLAS}" \
    --fc_types "${FC_TYPES[@]}" \
    --num_processes "${NUM_PROCESSES}"

echo "Done. Train FC / graph models via:"
echo "  bash scripts/run_experiment.sh --model <bnt|braingnn|...> --dataset ${DATASET} --task <taskN> --mode <finetune|train_scratch>"
