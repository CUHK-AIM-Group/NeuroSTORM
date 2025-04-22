#!/bin/bash
# bash scripts/adhd200_downstream/ts_fmrifound_task3.sh batch_size

# Set default score_name
score_name="cgi01_cgi_si"
master_port=29500
batch_size="12"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  score_name=$1
fi
if [ ! -z "$2" ]; then
  master_port=$2
fi
if [ ! -z "$3" ]; then
  batch_size=$3
fi

# We will use all aviailable GPUs, and automatically set the same batch size for each GPU
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export MASTER_PORT=$master_port

# Construct project_name using task_name
project_name="transdiag_ts_fmrifound_task2_${score_name}_train1.0"


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
  --downstream_task_id 2 \
  --downstream_task_type "regression" \
  --task_name "${score_name}" \
  --dataset_split_num 2 \
  --seed 1 \
  --learning_rate 1e-3 \
  --model fmrifound \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 40 \
  --img_size 96 96 96 40 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --train_split 0.8 --val_split 0.2
