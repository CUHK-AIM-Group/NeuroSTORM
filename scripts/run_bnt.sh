#!/bin/bash
# Example script for training BrainNetworkTransformer (BNT) on functional connectivity data

# BNT uses correlation matrices as input, where each row represents a node's connectivity pattern

# Step 1: Generate ROI time series (if not already done)
echo "Step 1: Generating ROI time series..."
python datasets/generate_roi_data_from_nii.py \
  --atlas_names cc200 \
  --dataset_names HCP1200 \
  --fmri_dir ./data/HCP1200/img \
  --atlas_dir ./datasets/atlas \
  --output_dir ./data/HCP1200/roi \
  --num_processes 8

# Step 2: Compute functional connectivity matrices (correlation for BNT)
echo "Step 2: Computing functional connectivity matrices..."
python datasets/compute_fc.py \
  --roi_dir ./data/HCP1200/roi/cc200 \
  --output_dir ./data/HCP1200/fc/cc200 \
  --atlas_name cc200 \
  --fc_types correlation \
  --num_processes 8

# Step 3: Train BNT for classification (e.g., gender)
echo "Step 3: Training BNT for gender classification..."
python main.py \
  --gpu_ids 0 \
  --max_epochs 100 \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200 \
  --data_type fc_bnt \
  --atlas_name cc200 \
  --fc_type correlation \
  --model bnt \
  --num_rois 200 \
  --pos_encoding identity \
  --pos_embed_dim 32 \
  --pooling_sizes 100 50 25 \
  --do_pooling True True False \
  --hidden_size 1024 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --downstream_task_type classification \
  --task_name sex \
  --num_classes 2 \
  --dataset_split_num 1

# Step 4: Train BNT for regression (e.g., age)
echo "Step 4: Training BNT for age regression..."
python main.py \
  --gpu_ids 0 \
  --max_epochs 100 \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200 \
  --data_type fc_bnt \
  --atlas_name cc200 \
  --fc_type correlation \
  --model bnt \
  --num_rois 200 \
  --pos_encoding identity \
  --pos_embed_dim 32 \
  --pooling_sizes 100 50 25 \
  --do_pooling True True False \
  --hidden_size 1024 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --downstream_task_type regression \
  --task_name age \
  --num_classes 1 \
  --dataset_split_num 1 \
  --label_scaling_method standardization

# Step 5: Train BNT with different pooling strategy (no pooling)
echo "Step 5: Training BNT without pooling..."
python main.py \
  --gpu_ids 0 \
  --max_epochs 100 \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200 \
  --data_type fc_bnt \
  --atlas_name cc200 \
  --fc_type correlation \
  --model bnt \
  --num_rois 200 \
  --pos_encoding identity \
  --pos_embed_dim 32 \
  --pooling_sizes 200 200 200 \
  --do_pooling False False False \
  --hidden_size 1024 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --downstream_task_type classification \
  --task_name sex \
  --num_classes 2 \
  --dataset_split_num 1
