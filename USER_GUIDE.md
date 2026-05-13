# NeuroSTORM User Guide

Complete guide for training, fine-tuning, and using NeuroSTORM models.

---

## Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Quick Start & Demo](#2-quick-start--demo)
3. [Training Models](#3-training-models)
4. [Fine-tuning](#4-fine-tuning)
5. [Advanced Usage](#5-advanced-usage)
6. [Model-Specific Guides](#6-model-specific-guides)

---

## 1. Data Preparation

### 1.1 Data Downloading

We provide data download scripts for HCP-YA, including rfMRI, tfMRI, T1, and T2. Please register for an account on the official [HCP-YA project website](https://humanconnectome.org/study/hcp-young-adult/overview).

```bash
cd ./scripts/dataset_download
python download_HCP_1200_rfMRI.py --id your_aws_id --key your_aws_key --out_dir hcp_ya --cpu_worker 1
python download_HCP_1200_tfMRI.py --id your_aws_id --key your_aws_key --out_dir hcp_ya --cpu_worker 1
python download_HCP_1200_t1t2.py --id your_aws_id --key your_aws_key --out_dir hcp_ya --cpu_worker 1
```

### 1.2 Data Pre-processing

First, ensure that you have applied a primary processing pipeline (FSL, fMRIPrep, or HCP pipeline) and that your data has been aligned into MNI152 space.

#### Brain Extraction (Optional)

```bash
cd ./datasets
bash brain_extraction.sh /path/to/your/dataset /path/to/output/dataset
```

#### Volume Pre-processing

Each subject's rfMRI is spatially resampled to 2 mm iso, temporally resampled to TR=0.8 s,
center-cropped to 96³, z-normalized, **symmetrically quantized to int8 (plus one per-subject
`scale`)**, and stored as a single `data.pt` per subject (`[T, H, W, D]` layout for
mmap-friendly clip reads).

```bash
cd NeuroSTORM/datasets
python preprocessing_volume.py \
  --dataset_name hcp \
  --load_root ./data/hcp \
  --save_root ./processed_data/hcp \
  --num_processes 8
```

Output: `./processed_data/hcp/img/<subject_id>/data.pt` — a dict with
`{'frames': int8[T, 96, 96, 96], 'scale': float, 'num_frames': int}`.
Dequantize at load time with `frames.to(torch.float32) * scale`.

> **Legacy format compatibility**: the loader also accepts data preprocessed
> by earlier versions (per-frame `frame_*.pt` float16 files). If a subject
> directory contains `data.pt`, the new format is used; otherwise the loader
> falls back to `frame_*.pt`. You do not need to re-run preprocessing on
> existing datasets.

### 1.3 Converting 4D Volume to 2D ROIs

For graph-based models (BrainGNN, LG-GNN, IBGNN, BNT, Com-BrainTF, BrainNetCNN):

```bash
cd NeuroSTORM/datasets
python generate_roi_data_from_nii.py \
  --atlas_names cc200 \
  --dataset_names hcp \
  --output_dir ./processed_data \
  --num_processes 32
```

### 1.4 Computing Functional Connectivity

For graph-based models:

```bash
cd NeuroSTORM/datasets
python compute_fc.py \
  --roi_dir ./processed_data/roi/cc200 \
  --output_dir ./processed_data/fc/cc200 \
  --atlas_name cc200 \
  --fc_types correlation partial_correlation \
  --num_processes 8
```

---

## 2. Quick Start & Demo

### 2.1 Single File Inference

Run inference on a single preprocessed fMRI subject:

```bash
# Age prediction
python demo.py \
  --mode single \
  --ckpt_path ./pretrained_models/age.ckpt \
  --fmri_path ./data/HCP1200_MNI_to_TRs_minmax/img/100206 \
  --task age

# Gender classification
python demo.py \
  --mode single \
  --ckpt_path ./pretrained_models/gender.ckpt \
  --fmri_path ./data/HCP1200_MNI_to_TRs_minmax/img/100206 \
  --task gender

# Phenotype prediction
python demo.py \
  --mode single \
  --ckpt_path ./pretrained_models/phenotype.ckpt \
  --fmri_path ./data/HCP1200_MNI_to_TRs_minmax/img/100206 \
  --task phenotype \
  --phenotype_name "CogTotalComp_Unadj" \
  --phenotype_type regression
```

### 2.2 Batch Inference on Test Set

Evaluate on a full dataset test split:

```bash
python demo.py \
  --mode dataset \
  --ckpt_path /path/to/model.ckpt \
  --task age \
  --image_path /path/to/preprocessed/data
```

Or use the provided script:

```bash
sh scripts/run_demo.sh
```

---

## 3. Training Models

### 3.1 Pre-training NeuroSTORM

**MAE Pre-training:**

```bash
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --model neurostorm \
  --pretraining \
  --use_mae \
  --mask_ratio 0.75 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --max_epochs 100
```

**Contrastive Pre-training (SwiFT):**

```bash
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --model swift \
  --pretraining \
  --use_contrastive \
  --contrastive_type 3 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --max_epochs 100
```

### 3.2 Fine-tuning for Downstream Tasks

**Classification (Gender):**

```bash
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --model neurostorm \
  --load_model_path ./pretrained_models/neurostorm_mae.pth \
  --downstream_task_type classification \
  --task_name sex \
  --num_classes 2 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --max_epochs 50
```

**Regression (Age):**

```bash
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --model neurostorm \
  --load_model_path ./pretrained_models/neurostorm_mae.pth \
  --downstream_task_type regression \
  --task_name age \
  --num_classes 1 \
  --label_scaling_method standardization \
  --batch_size 32 \
  --learning_rate 0.001 \
  --max_epochs 50
```

### 3.3 Training Graph-Based Models

**BrainGNN:**

```bash
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --data_type fc_graph \
  --atlas_name cc200 \
  --fc_type partial_correlation \
  --model braingnn \
  --num_rois 200 \
  --downstream_task_type classification \
  --task_name sex \
  --num_classes 2 \
  --batch_size 32
```

**BrainNetworkTransformer (BNT):**

```bash
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --data_type fc_bnt \
  --atlas_name cc200 \
  --model bnt \
  --num_rois 200 \
  --pooling_sizes 100 50 25 \
  --do_pooling True True False \
  --downstream_task_type classification \
  --task_name sex \
  --num_classes 2
```

**BrainNetCNN:**

```bash
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --data_type fc_bnt \
  --atlas_name cc200 \
  --model brainnetcnn \
  --num_rois 200 \
  --downstream_task_type classification \
  --task_name sex \
  --num_classes 2
```

---

## 4. Fine-tuning

### 4.1 Loading Pre-trained Weights

```bash
python main.py \
  --load_model_path ./pretrained_models/neurostorm_mae.pth \
  --use_prompt_tuning \              # freeze backbone, train per-block prompts + head
  --prompt_len 50 \                  # k = 50 prompt tokens per block (default)
  ...
```

`--use_prompt_tuning` enables Task-specific Prompt Tuning (TPT, NeuroSTORM only). The
backbone is frozen and only learnable prompts (one set per Swin block) plus the output
head are trained. Console prints the trainable / full-model parameter ratio at startup.

### 4.2 Resume Training

```bash
python main.py \
  --resume_ckpt_path ./checkpoints/last.ckpt \
  ...
```

---

## 5. Advanced Usage

### 5.1 Multi-GPU Training

```bash
python main.py \
  --accelerator gpu \
  --devices 4 \
  --strategy ddp \
  ...
```

### 5.2 Custom Learning Rate Schedule

```bash
python main.py \
  --use_scheduler \
  --optimizer AdamW \
  --learning_rate 0.001 \
  --weight_decay 0.01 \
  --milestones 50 100 \
  ...
```

### 5.3 Data Augmentation

```bash
python main.py \
  --augment_during_training \
  --augment_only_affine \  # or --augment_only_intensity
  ...
```

### 5.4 Logging

```bash
python main.py \
  --loggername tensorboard \
  --project_name my_project \
  ...
```

