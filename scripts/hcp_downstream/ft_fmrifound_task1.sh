#!/bin/bash
# bash scripts/hcp_downstream/ft_fmrifound_task1.sh task_name batch_size

# Set default task_name
task_name="sex"
batch_size="12"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  task_name=$1
fi

if [ "$task_name" = "sex" ]; then
    downstream_task_type="classification"
else
    downstream_task_type="regression"
fi

if [ ! -z "$2" ]; then
  batch_size=$2
fi

# We will use all aviailable GPUs, and automatically set the same batch size for each GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1

# Construct project_name using task_name
project_name="hcp_ts_fmrifound_task1_${task_name}_train1.0"

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
  --downstream_task_id 1 \
  --downstream_task_type "$downstream_task_type" \
  --task_name "$task_name" \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 5e-5 \
  --model fmrifound \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 20 \
  --img_size 96 96 96 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --load_model_path ./output/fmrifound/pt_fmrifound_mae_ratio0.5.ckpt
