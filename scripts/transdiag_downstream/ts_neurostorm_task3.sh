#!/bin/bash
# bash scripts/adhd200_downstream/ts_neurostorm_task3.sh batch_size

# Set default task_name
batch_size="12"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  batch_size=$1
fi


# We will use all aviailable GPUs, and automatically set the same batch size for each GPU
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export MASTER_PORT=29501

# Construct project_name using task_name
project_name="transdiag_ts_neurostorm_task3_dx_train1.0"


python main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --clf_head_version v1 \
  --dataset_name TransDiag \
  --image_path ./data/TRANS_preprocessed \
  --batch_size "$batch_size" \
  --num_workers "$batch_size" \
  --project_name "$project_name" \
  --limit_training_samples 1.0 \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_id 3 \
  --downstream_task_type "classification" \
  --num_classes 6 \
  --task_name "diagnosis" \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 1e-3 \
  --model neurostorm \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 40 \
  --img_size 96 96 96 40 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --train_split 0.8 --val_split 0.2
