#!/bin/bash
# bash scripts/hcp_pretrain/pt_contrastive_swift.sh batch_size

batch_size="12"

if [ ! -z "$1" ]; then
  batch_size=$1
fi

# We will use all aviailable GPUs, and automatically set the same batch size for each GPU
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

# Construct project_name
project_name="hcp_pt_swift_mae0.5"

python main.py \
  --max_epochs 30 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --batch_size "$batch_size" \
  --num_workers "$batch_size" \
  --project_name "$project_name" \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_type "classification" \
  --pretraining \
  --use_contrastive \
  --contrastive_type 1 \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 5e-5 \
  --model swift \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 20 \
  --img_size 96 96 96 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4
