#!/bin/bash
# Test trained NeuroSTORM model on ADNI dataset for AD vs CN classification
# Usage: bash scripts/adni_downstream/test_neurostorm_adni_ad_cn.sh [checkpoint_path] [batch_size]
#
# Example:
#   bash scripts/adni_downstream/test_neurostorm_adni_ad_cn.sh ./output/neurostorm/adni_ft_neurostorm_ad_cn/checkpoints/best.ckpt 8

# Set default parameters
checkpoint_path="${1:-./output/neurostorm/adni_ft_neurostorm_ad_cn/checkpoints/best.ckpt}"
batch_size="${2:-8}"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

# Project name for logging
project_name="adni_test_neurostorm_ad_cn"

# IMPORTANT: Update this path to point to your ADNI split files directory
ADNI_SPLIT_DIR="/mnt/dataset4/DATASETS/fsl_fmri/adni_split"

echo "=================================================="
echo "ADNI AD vs CN Classification - Testing Script"
echo "=================================================="
echo "Checkpoint: $checkpoint_path"
echo "Batch size: $batch_size"
echo "ADNI split directory: $ADNI_SPLIT_DIR"
echo "Project name: $project_name"
echo "=================================================="

python main.py \
  --test_only \
  --test_ckpt_path "$checkpoint_path" \
  --accelerator gpu \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --clf_head_version v2 \
  --dataset_name ADNI \
  --image_path "$ADNI_SPLIT_DIR" \
  --batch_size "$batch_size" \
  --eval_batch_size "$batch_size" \
  --num_workers 4 \
  --project_name "$project_name" \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_id 3 \
  --downstream_task_type "classification" \
  --num_classes 2 \
  --task_name "diagnosis" \
  --dataset_split_num 1 \
  --seed 42 \
  --model neurostorm \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --num_heads 3 6 12 24 \
  --sequence_length 20 \
  --img_size 96 96 96 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --patch_size 4 4 4 1 \
  --stride_between_seq 1.0 \
  --stride_within_seq 1 \
  --with_voxel_norm False \
  --shuffle_time_sequence False

echo "=================================================="
echo "Testing completed!"
echo "Results saved to: ./output/neurostorm/$project_name"
echo "=================================================="
