#!/bin/bash
# Example script for training BrainGNN on functional connectivity data

# Step 1: Generate ROI time series from preprocessed fMRI data
echo "Step 1: Generating ROI time series..."
python datasets/generate_roi_data_from_nii.py \
  --atlas_names cc200 \
  --dataset_names HCP1200 \
  --fmri_dir ./data/HCP1200/img \
  --atlas_dir ./datasets/atlas \
  --output_dir ./data/HCP1200/roi \
  --num_processes 8

# Step 2: Compute functional connectivity matrices
echo "Step 2: Computing functional connectivity matrices..."
python datasets/compute_fc.py \
  --roi_dir ./data/HCP1200/roi/cc200 \
  --output_dir ./data/HCP1200/fc/cc200 \
  --atlas_name cc200 \
  --fc_types correlation partial_correlation \
  --num_processes 8

# Step 3: Train BrainGNN for classification (e.g., gender)
echo "Step 3: Training BrainGNN for gender classification..."
python main.py \
  --gpu_ids 0 \
  --max_epochs 100 \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200 \
  --data_type fc_graph \
  --atlas_name cc200 \
  --fc_type partial_correlation \
  --model braingnn \
  --num_rois 200 \
  --pooling_ratio 0.5 \
  --num_communities 8 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --downstream_task_type classification \
  --task_name sex \
  --num_classes 2 \
  --dataset_split_num 1

# Step 4: Train BrainGNN for regression (e.g., age)
echo "Step 4: Training BrainGNN for age regression..."
python main.py \
  --gpu_ids 0 \
  --max_epochs 100 \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200 \
  --data_type fc_graph \
  --atlas_name cc200 \
  --fc_type partial_correlation \
  --model braingnn \
  --num_rois 200 \
  --pooling_ratio 0.5 \
  --num_communities 8 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --downstream_task_type regression \
  --task_name age \
  --num_classes 1 \
  --dataset_split_num 1 \
  --label_scaling_method standardization
