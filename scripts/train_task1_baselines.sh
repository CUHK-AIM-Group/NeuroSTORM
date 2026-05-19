#!/bin/bash
# Train baseline models (BrainNetCNN, BNT, BrainGNN) on sex and age tasks
# Dataset: HCP-YA (hcpya_preprocessed)
# GPUs: 4-7 (one per experiment, 4 parallel)
#
# Usage:
#   bash scripts/train_task1_baselines.sh          # run all 6 experiments
#   bash scripts/train_task1_baselines.sh sex      # run only sex classification (3 experiments)
#   bash scripts/train_task1_baselines.sh age      # run only age regression (3 experiments)

set -euo pipefail

PYTHON="/home/cwang/anaconda3/envs/neurostorm/bin/python"
DATA_PATH="/data/cwang/remote/fmri/hcpya_preprocessed"
OUTPUT_BASE="output/task1"
TASK_FILTER="${1:-all}"

export NCCL_P2P_DISABLE=1

run_experiment() {
    local gpu="$1"
    local name="$2"
    shift 2
    local output_dir="${OUTPUT_BASE}/${name}"

    echo "[GPU ${gpu}] Starting: ${name}"
    CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} main.py \
        --output_dir "${output_dir}" \
        --project_name "${name}" \
        --loggername tensorboard \
        --dataset_name HCP1200 \
        --image_path "${DATA_PATH}" \
        --num_nodes 1 \
        --seed 1234 \
        "$@" 2>&1 | tee "${output_dir}.log" &
}

# ============================================================
# Sex Classification
# ============================================================
run_sex() {
    # BrainNetCNN - sex (atlas: cc200, num_rois=190)
    run_experiment 4 "brainnetcnn_hcpya_sex" \
        --model brainnetcnn \
        --data_type fc_bnt \
        --atlas_name cc200 \
        --fc_type correlation \
        --num_rois 190 \
        --downstream_task_id 1 \
        --downstream_task_type classification \
        --task_name sex \
        --num_classes 2 \
        --learning_rate 1e-3 \
        --weight_decay 5e-4 \
        --dropout 0.5 \
        --e2e_channels 32 \
        --e2n_channels 64 \
        --n2g_channels 256 \
        --fc_channels \
        --optimizer Adam \
        --batch_size 16 \
        --eval_batch_size 32 \
        --max_epochs 50 \
        --num_workers 8

    # BNT - sex (atlas: schaefer_200_7net, num_rois=200)
    run_experiment 5 "bnt_hcpya_sex" \
        --model bnt \
        --data_type fc_bnt \
        --atlas_name schaefer_200_7net \
        --fc_type correlation \
        --num_rois 200 \
        --downstream_task_id 1 \
        --downstream_task_type classification \
        --task_name sex \
        --num_classes 2 \
        --learning_rate 5e-4 \
        --weight_decay 1e-4 \
        --dropout 0.3 \
        --hidden_size 1024 \
        --nhead 4 \
        --optimizer Adam \
        --use_scheduler \
        --batch_size 16 \
        --eval_batch_size 32 \
        --max_epochs 100 \
        --num_workers 8

    # BrainGNN - sex (atlas: schaefer_100_7net, num_rois=100)
    run_experiment 6 "braingnn_hcpya_sex" \
        --model braingnn \
        --data_type fc_graph \
        --atlas_name schaefer_100_7net \
        --fc_type partial_correlation \
        --num_rois 100 \
        --downstream_task_id 1 \
        --downstream_task_type classification \
        --task_name sex \
        --num_classes 2 \
        --learning_rate 1e-3 \
        --weight_decay 1e-3 \
        --dropout 0.5 \
        --pooling_ratio 0.5 \
        --num_communities 16 \
        --optimizer SGD \
        --momentum 0.9 \
        --batch_size 16 \
        --eval_batch_size 32 \
        --max_epochs 50 \
        --num_workers 8
}

# ============================================================
# Age Regression
# ============================================================
run_age() {
    # BrainGNN - age (atlas: schaefer_100_7net, num_rois=100)
    run_experiment 4 "braingnn_hcpya_age" \
        --model braingnn \
        --data_type fc_graph \
        --atlas_name schaefer_100_7net \
        --fc_type partial_correlation \
        --num_rois 100 \
        --downstream_task_id 1 \
        --downstream_task_type regression \
        --task_name age \
        --num_classes 1 \
        --learning_rate 1e-3 \
        --weight_decay 1e-3 \
        --dropout 0.5 \
        --pooling_ratio 0.5 \
        --num_communities 16 \
        --optimizer SGD \
        --momentum 0.9 \
        --batch_size 16 \
        --eval_batch_size 32 \
        --max_epochs 50 \
        --num_workers 8

    # BNT - age (atlas: basc_122, num_rois=122)
    run_experiment 5 "bnt_hcpya_age" \
        --model bnt \
        --data_type fc_bnt \
        --atlas_name basc_122 \
        --fc_type correlation \
        --num_rois 122 \
        --downstream_task_id 1 \
        --downstream_task_type regression \
        --task_name age \
        --num_classes 1 \
        --learning_rate 5e-4 \
        --weight_decay 1e-4 \
        --dropout 0.3 \
        --hidden_size 1024 \
        --nhead 4 \
        --batch_size 16 \
        --eval_batch_size 32 \
        --max_epochs 50 \
        --num_workers 8

    # BrainNetCNN - age (atlas: cc200, num_rois=190)
    run_experiment 6 "brainnetcnn_hcpya_age" \
        --model brainnetcnn \
        --data_type fc_bnt \
        --atlas_name cc200 \
        --fc_type correlation \
        --num_rois 190 \
        --downstream_task_id 1 \
        --downstream_task_type regression \
        --task_name age \
        --num_classes 1 \
        --learning_rate 1e-3 \
        --weight_decay 5e-4 \
        --dropout 0.5 \
        --e2e_channels 32 \
        --e2n_channels 64 \
        --n2g_channels 256 \
        --batch_size 16 \
        --eval_batch_size 32 \
        --max_epochs 50 \
        --num_workers 8
}

# ============================================================
# Main
# ============================================================
case "$TASK_FILTER" in
    sex)
        echo "=== Running Sex Classification (3 experiments on GPUs 4-6) ==="
        run_sex
        ;;
    age)
        echo "=== Running Age Regression (3 experiments on GPUs 4-6) ==="
        run_age
        ;;
    all)
        echo "=== Running Sex Classification first (GPUs 4-6), then Age Regression ==="
        run_sex
        echo "Waiting for sex classification to finish..."
        wait
        echo "=== Sex classification done. Starting Age Regression (GPUs 4-6) ==="
        run_age
        ;;
    *)
        echo "Usage: $0 [sex|age|all]"
        exit 1
        ;;
esac

echo "Waiting for all experiments to finish..."
wait
echo "=== All experiments completed ==="
