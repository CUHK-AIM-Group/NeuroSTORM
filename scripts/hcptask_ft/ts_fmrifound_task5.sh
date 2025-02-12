#!/bin/bash
# bash scripts/hcp_ft/ts_abcd2hcp_mamba.sh batch_size

batch_size="12"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  batch_size=$1
fi

# export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

# Construct project_name using score_name
project_name="hcp_ts_state_classification_train1.0_mamba"

python project/main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --strategy ddp \
  --loggername tensorboard \
  --clf_head_version v2 \
  --dataset_name HCPTASK \
  --image_path ./data/HCPTASK_MNI_to_TRs_minmax \
  --batch_size "$batch_size" \
  --num_workers "$batch_size" \
  --project_name "$project_name" \
  --limit_training_samples 1.0 \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task classification \
  --downstream_task_type classification \
  --num_classes 7 \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 5e-5 \
  --model swin4d_ver7 \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --img_size 96 96 96 20 \
  --use_mamba
