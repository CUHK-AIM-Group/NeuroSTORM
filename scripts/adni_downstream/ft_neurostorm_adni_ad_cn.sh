#!/bin/bash
# Fine-tuning NeuroSTORM on ADNI dataset for AD vs CN classification
# Usage: bash scripts/adni_downstream/ft_neurostorm_adni_ad_cn.sh [batch_size] [learning_rate] [pretrained_model_path]
#
# Example:
#   bash scripts/adni_downstream/ft_neurostorm_adni_ad_cn.sh 4 5e-5 ./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt

# Set default parameters
batch_size="${1:-4}"
learning_rate="${2:-5e-5}"
pretrained_model_path="${3:-./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt}"

# GPU configuration
# Adjust CUDA_VISIBLE_DEVICES based on your available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

# Project name for logging and checkpoints
project_name="adni_ft_neurostorm_ad_cn_lr${learning_rate}_bs${batch_size}"

# IMPORTANT: Update this path to point to your ADNI split files directory
# The directory should contain:
#   - adni_ad_mni_train.txt
#   - adni_ad_mni_val.txt
#   - adni_ad_mni_test.txt
ADNI_SPLIT_DIR="/mnt/dataset4/DATASETS/fsl_fmri/adni_split"

echo "=================================================="
echo "ADNI AD vs CN Classification - Fine-tuning Script"
echo "=================================================="
echo "Batch size: $batch_size"
echo "Learning rate: $learning_rate"
echo "Pretrained model: $pretrained_model_path"
echo "ADNI split directory: $ADNI_SPLIT_DIR"
echo "Project name: $project_name"
echo "=================================================="

python main.py \
  --accelerator gpu \
  --max_epochs 50 \
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
  --limit_training_samples 1.0 \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_id 3 \
  --downstream_task_type "classification" \
  --num_classes 2 \
  --task_name "diagnosis" \
  --dataset_split_num 1 \
  --seed 42 \
  --learning_rate "$learning_rate" \
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
  --shuffle_time_sequence False \
  --load_model_path "$pretrained_model_path"

echo "=================================================="
echo "Training completed!"
echo "Results saved to: ./output/neurostorm/$project_name"
echo "=================================================="
