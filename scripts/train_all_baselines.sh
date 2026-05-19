#!/bin/bash
# Comprehensive parameter search: 6 models x 2 tasks on HCP-YA.
# Models: brainnetcnn, bnt, braingnn, combraintf, ibgnn, lggnn
# Tasks:  sex (classification), age (regression)
#
# Runs all sex experiments in parallel on 6 GPUs, then all age experiments.
#
# Usage:
#   bash scripts/train_all_baselines.sh [sex|age|all]
set -uo pipefail

PYTHON="/home/cwang/anaconda3/envs/neurostorm/bin/python"
DATA_PATH="/data/cwang/remote/fmri/hcpya_preprocessed"
OUTPUT_BASE="output/task1"
TASK_FILTER="${1:-all}"

mkdir -p "${OUTPUT_BASE}"
export NCCL_P2P_DISABLE=1

# GPUs to use (one per experiment, parallel)
GPUS=(0 1 2 3 5 7)

run() {
    local gpu="$1"
    local name="$2"
    shift 2
    local outdir="${OUTPUT_BASE}/${name}"
    rm -rf "${outdir}"
    echo "[GPU ${gpu}] Starting: ${name}"
    CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} main.py \
        --output_dir "${outdir}" \
        --project_name "${name}" \
        --loggername tensorboard \
        --dataset_name HCP1200 \
        --image_path "${DATA_PATH}" \
        --num_nodes 1 \
        --seed 1234 \
        "$@" > "${outdir}.log" 2>&1 &
}

# ============================================================
# SEX CLASSIFICATION
# ============================================================
run_sex() {
    # BrainNetCNN (cc200, 190 ROI)
    run ${GPUS[0]} "brainnetcnn_hcpya_sex" \
        --model brainnetcnn --data_type fc_bnt \
        --atlas_name cc200 --fc_type correlation --num_rois 190 \
        --downstream_task_id 1 --downstream_task_type classification \
        --task_name sex --num_classes 2 \
        --learning_rate 1e-3 --weight_decay 5e-4 --dropout 0.5 \
        --e2e_channels 32 --e2n_channels 64 --n2g_channels 256 \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8

    # BNT (schaefer_200_7net, 200 ROI)
    run ${GPUS[1]} "bnt_hcpya_sex" \
        --model bnt --data_type fc_bnt \
        --atlas_name schaefer_200_7net --fc_type correlation --num_rois 200 \
        --downstream_task_id 1 --downstream_task_type classification \
        --task_name sex --num_classes 2 \
        --learning_rate 1e-4 --weight_decay 1e-4 --dropout 0.1 \
        --pos_embed_dim 8 --hidden_size 1024 \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 100 --num_workers 8

    # BrainGNN (schaefer_100_7net, 100 ROI, partial_correlation)
    run ${GPUS[2]} "braingnn_hcpya_sex" \
        --model braingnn --data_type fc_graph \
        --atlas_name schaefer_100_7net --fc_type partial_correlation --num_rois 100 \
        --downstream_task_id 1 --downstream_task_type classification \
        --task_name sex --num_classes 2 \
        --learning_rate 1e-3 --weight_decay 1e-3 --dropout 0.5 \
        --pooling_ratio 0.5 --num_communities 16 \
        --optimizer SGD --momentum 0.9 \
        --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8

    # ComBrainTF (cc200, 190 ROI)
    run ${GPUS[3]} "combraintf_hcpya_sex" \
        --model combraintf --data_type fc_bnt \
        --atlas_name cc200 --fc_type correlation --num_rois 190 \
        --downstream_task_id 1 --downstream_task_type classification \
        --task_name sex --num_classes 2 \
        --learning_rate 5e-4 --weight_decay 1e-4 --dropout 0.1 \
        --d_model 128 --nhead 4 --num_layers 3 --dim_feedforward 512 \
        --num_communities 10 \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8

    # IBGNN (cc200, 190 ROI, partial_correlation)
    run ${GPUS[4]} "ibgnn_hcpya_sex" \
        --model ibgnn --data_type fc_graph \
        --atlas_name cc200 --fc_type partial_correlation --num_rois 190 \
        --downstream_task_id 1 --downstream_task_type classification \
        --task_name sex --num_classes 2 \
        --learning_rate 5e-4 --weight_decay 5e-4 --dropout 0.5 \
        --hidden_dims 128 64 \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8

    # LG-GNN (cc200, 190 ROI)
    run ${GPUS[5]} "lggnn_hcpya_sex" \
        --model lggnn --data_type fc_bnt \
        --atlas_name cc200 --fc_type correlation --num_rois 190 \
        --downstream_task_id 1 --downstream_task_type classification \
        --task_name sex --num_classes 2 \
        --learning_rate 1e-3 --weight_decay 5e-4 --dropout 0.5 \
        --hidden_dims 128 64 --k_neighbors 10 \
        --learn_graph True --graph_metric cosine \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8
}

# ============================================================
# AGE REGRESSION
# ============================================================
run_age() {
    # BrainNetCNN (cc200)
    run ${GPUS[0]} "brainnetcnn_hcpya_age" \
        --model brainnetcnn --data_type fc_bnt \
        --atlas_name cc200 --fc_type correlation --num_rois 190 \
        --downstream_task_id 1 --downstream_task_type regression \
        --task_name age --num_classes 1 \
        --learning_rate 1e-3 --weight_decay 5e-4 --dropout 0.5 \
        --e2e_channels 32 --e2n_channels 64 --n2g_channels 256 \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8

    # BNT (basc_122)
    run ${GPUS[1]} "bnt_hcpya_age" \
        --model bnt --data_type fc_bnt \
        --atlas_name basc_122 --fc_type correlation --num_rois 122 \
        --downstream_task_id 1 --downstream_task_type regression \
        --task_name age --num_classes 1 \
        --learning_rate 1e-4 --weight_decay 1e-4 --dropout 0.1 \
        --pos_embed_dim 8 --hidden_size 1024 \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8

    # BrainGNN (schaefer_100_7net, partial_correlation)
    run ${GPUS[2]} "braingnn_hcpya_age" \
        --model braingnn --data_type fc_graph \
        --atlas_name schaefer_100_7net --fc_type partial_correlation --num_rois 100 \
        --downstream_task_id 1 --downstream_task_type regression \
        --task_name age --num_classes 1 \
        --learning_rate 1e-3 --weight_decay 1e-3 --dropout 0.5 \
        --pooling_ratio 0.5 --num_communities 16 \
        --optimizer SGD --momentum 0.9 \
        --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8

    # ComBrainTF (cc200)
    run ${GPUS[3]} "combraintf_hcpya_age" \
        --model combraintf --data_type fc_bnt \
        --atlas_name cc200 --fc_type correlation --num_rois 190 \
        --downstream_task_id 1 --downstream_task_type regression \
        --task_name age --num_classes 1 \
        --learning_rate 5e-4 --weight_decay 1e-4 --dropout 0.1 \
        --d_model 128 --nhead 4 --num_layers 3 --dim_feedforward 512 \
        --num_communities 10 \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8

    # IBGNN (cc200, partial_correlation)
    run ${GPUS[4]} "ibgnn_hcpya_age" \
        --model ibgnn --data_type fc_graph \
        --atlas_name cc200 --fc_type partial_correlation --num_rois 190 \
        --downstream_task_id 1 --downstream_task_type regression \
        --task_name age --num_classes 1 \
        --learning_rate 5e-4 --weight_decay 5e-4 --dropout 0.5 \
        --hidden_dims 128 64 \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8

    # LG-GNN (cc200)
    run ${GPUS[5]} "lggnn_hcpya_age" \
        --model lggnn --data_type fc_bnt \
        --atlas_name cc200 --fc_type correlation --num_rois 190 \
        --downstream_task_id 1 --downstream_task_type regression \
        --task_name age --num_classes 1 \
        --learning_rate 1e-3 --weight_decay 5e-4 --dropout 0.5 \
        --hidden_dims 128 64 --k_neighbors 10 \
        --learn_graph True --graph_metric cosine \
        --optimizer Adam --batch_size 16 --eval_batch_size 32 \
        --max_epochs 50 --num_workers 8
}

case "$TASK_FILTER" in
    sex)
        echo "=== Sex classification (6 models, parallel on GPUs ${GPUS[*]}) ==="
        run_sex
        wait
        echo "=== Sex done ==="
        ;;
    age)
        echo "=== Age regression (6 models, parallel on GPUs ${GPUS[*]}) ==="
        run_age
        wait
        echo "=== Age done ==="
        ;;
    all)
        echo "=== Phase 1: Sex (parallel on GPUs ${GPUS[*]}) ==="
        run_sex
        wait
        echo "=== Phase 1 done. Phase 2: Age (parallel on GPUs ${GPUS[*]}) ==="
        run_age
        wait
        echo "=== All experiments done ==="
        ;;
    *)
        echo "Usage: $0 [sex|age|all]"
        exit 1
        ;;
esac
