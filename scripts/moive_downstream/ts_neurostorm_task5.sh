#!/bin/bash
# bash scripts/hcptask_downstream/ts_neurostorm_task5.sh batch_size

batch_size="12"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  score_name=$1
fi
if [ ! -z "$2" ]; then
  batch_size=$2
fi

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export MASTER_PORT=29500

# Construct project_name using score_name
project_name="movie_ts_neurostorm_task5_train1.0"

python main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --num_nodes 2 \
  --strategy ddp \
  --loggername tensorboard \
  --clf_head_version v1 \
  --dataset_name MOVIE \
  --image_path ./data/MOVIE_preprocessed \
  --batch_size "$batch_size" \
  --num_workers "$batch_size" \
  --project_name "$project_name" \
  --limit_training_samples 1.0 \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_id 5 \
  --downstream_task_type classification \
  --num_classes 2 \
  --task_name "diagnosis" \
  --dataset_split_num 2 \
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
