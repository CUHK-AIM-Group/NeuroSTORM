#!/bin/bash
# Fine-tune on HCP1200 for re-identification (classification)
# bash scripts/hcp_downstream/ft_fmrifound_task4.sh [batch_size]

batch_size="12"

if [ ! -z "$1" ]; then
  batch_size=$1
fi

# Single-GPU by default
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

project_name="hcp_ft_fmrifound_task4_reid_train1.0"

python main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --clf_head_version v1 \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --batch_size "$batch_size" \
  --num_workers "$batch_size" \
  --project_name "$project_name" \
  --limit_training_samples 1.0 \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_id 4 \
  --downstream_task_type classification \
  --task_name fmri_reid \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 5e-5 \
  --model neurostorm \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 20 \
  --img_size 96 96 96 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --load_model_path ./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt
