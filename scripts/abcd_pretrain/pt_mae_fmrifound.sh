#!/bin/bash
# bash scripts/abcd_pretrain/pt_mae_fmrifound.sh batch_size

batch_size="8"

if [ ! -z "$1" ]; then
  batch_size=$1
fi

# We will use all aviailable GPUs, and automatically set the same batch size for each GPU
export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1

# Construct project_name
project_name="abcd_pt_fmrifound_mae0.5"

python main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --dataset_name ABCD \
  --image_path ./data/ABCD_MNI_to_TRs_minmax \
  --batch_size "$batch_size" \
  --num_workers "$batch_size" \
  --project_name "$project_name" \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_type "classification" \
  --pretraining \
  --use_mae \
  --spatial_mask window \
  --time_mask random \
  --mask_ratio 0.5 \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 5e-5 \
  --model fmrifound \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 20 \
  --img_size 96 96 96 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4
