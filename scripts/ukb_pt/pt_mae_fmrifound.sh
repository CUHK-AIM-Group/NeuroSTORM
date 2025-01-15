#!/bin/bash
# bash sample_scripts/ukb/pt_mae_fmrifound.sh score_name

# Set default score_name
mask_ratio="0.5"

# Override with the first argument if provided
if [ ! -z "$1" ]; then
  mask_ratio=$1
fi

# export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

# Construct project_name using score_name
project_name="ukb_mae_fmrifound_ratio${mask_ratio}"

python project/main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --loggername tensorboard \
  --clf_head_version v3 \
  --dataset_name UKB \
  --image_path ./data/UKB_MNI_to_TRs_minmax \
  --batch_size 4 \
  --num_workers 4 \
  --project_name "$project_name" \
  --limit_training_samples 1.0 \
  --last_layer_full_MSA True \
  --downstream_task sex \
  --pretraining \
  --use_mae \
  --spatial_mask window \
  --time_mask random \
  --mask_ratio "$mask_ratio" \
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
