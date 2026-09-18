# NeuroSTORM User Guide

Complete guide for training, fine-tuning, and using NeuroSTORM models.

---

## Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Quick Start & Demo](#2-quick-start--demo)
3. [Training Models](#3-training-models)
4. [Parameter-efficient Tuning and Resuming](#4-parameter-efficient-tuning-and-resuming)
5. [Advanced Usage](#5-advanced-usage)

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

### 1.3 Computing ROI Time Series and Functional Connectivity

FC- and graph-based models use ROI time series or functional-connectivity (FC)
matrices. The same command generates both from the preprocessed `data.pt`
blobs:

```bash
python datasets/compute_roi_fc.py \
  --input_dir ./processed_data/hcp/img \
  --input_format blob \
  --atlas_names cc200 \
  --fc_types correlation partial_correlation \
  --output_dir ./processed_data/hcp \
  --num_processes 32
```

For raw NIfTI input, use `--input_format nii` and point `--input_dir` to the
directory containing `.nii` or `.nii.gz` files. Outputs follow this layout:

```text
<output_dir>/roi/<atlas>/
<output_dir>/fc/<atlas>/<fc_type>/
```

---

## 2. Quick Start & Demo

`demo.py` supports all five benchmark categories. Task 1 exposes separate
`age` and `gender` targets, so the available task names are `age`, `gender`,
`phenotype`, `diagnosis`, `retrieval`, and `state`.

| Benchmark | `--task` value | Output |
|---|---|---|
| Task 1: Age and gender prediction | `age` or `gender` | Scalar age or binary class |
| Task 2: Phenotype prediction | `phenotype` | Regression value or class |
| Task 3: Disease diagnosis | `diagnosis` | Diagnostic class |
| Task 4: fMRI retrieval | `retrieval` | L2-normalized embedding or dataset Rank-1/mAP |
| Task 5: Task-state classification | `state` | Cognitive-state class |

Download the available checkpoints from
[Hugging Face](https://huggingface.co/zxcvb20001/NeuroSTORM). The commands below
assume the downloaded `task*` folders are stored under `./checkpoints`.
Released downstream weights currently cover Tasks 1-3. The Task 4 retrieval
and Task 5 state examples require compatible user-trained checkpoints.

Single-subject inference runs the model over all windows of the scan (with the
same window spacing as the training/evaluation pipeline) and averages the window
outputs into a subject-level prediction. For regression checkpoints trained with
label standardization (such as the released Task 1 age checkpoint), the printed
prediction is automatically inverse-transformed to the original label scale, and
the raw standardized output is shown alongside it. The released age checkpoint
uses its train-split statistics (mean = 28.7593 years, std = 3.6815 years),
which are built into the demo. For checkpoints you fine-tune yourself, pass
`--label_mean`/`--label_std` (or `--label_min`/`--label_max` for minmax scaling)
computed from your own training split; without them the demo prints the raw
standardized value with a warning instead of a converted label.

### 2.1 Single File Inference

Run inference on a single preprocessed fMRI subject:

```bash
# Age prediction
python demo.py \
  --mode single \
  --ckpt_path ./checkpoints/task1/neurostorm_hcpya_age.ckpt \
  --fmri_path ./data/HCP1200_MNI_to_TRs_minmax/img/100206 \
  --task age

# Gender classification
python demo.py \
  --mode single \
  --ckpt_path ./checkpoints/task1/neurostorm_hcpya_sex.ckpt \
  --fmri_path ./data/HCP1200_MNI_to_TRs_minmax/img/100206 \
  --task gender

# Phenotype prediction
python demo.py \
  --mode single \
  --ckpt_path ./checkpoints/task2/neurostorm_hcpya_cogtotalcomp_ageadj.ckpt \
  --fmri_path ./data/HCP1200_MNI_to_TRs_minmax/img/100206 \
  --task phenotype \
  --phenotype_name "CogTotalComp_AgeAdj" \
  --phenotype_type regression

# Disease diagnosis (for example, ABIDE ASD vs. control)
python demo.py \
  --mode single \
  --ckpt_path ./checkpoints/task3/neurostorm_abide_diagnosis.ckpt \
  --fmri_path /path/to/abide_preprocessed/img/subject_id \
  --task diagnosis

# fMRI retrieval embedding
python demo.py \
  --mode single \
  --ckpt_path /path/to/retrieval.ckpt \
  --fmri_path /path/to/preprocessed/img/subject_id \
  --task retrieval

# Task-fMRI state classification
python demo.py \
  --mode single \
  --ckpt_path /path/to/state_classification.ckpt \
  --fmri_path /path/to/HCPTASK_preprocessed/img/subject_task \
  --task state
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
  --mask_ratio 0.5 \
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

The recommended interface is the YAML-backed experiment runner. It keeps model,
dataset, and task settings together:

```bash
# Full fine-tuning on HCP-YA sex classification
bash scripts/run_experiment.sh \
  --model neurostorm \
  --dataset hcp1200 \
  --task task1 \
  --task_name sex \
  --mode finetune \
  --load_model_path ./checkpoints/pretraining/pt_neurostorm_mae_5ds.ckpt
```

The equivalent direct `main.py` commands are shown below.

**Classification (Gender):**

```bash
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --model neurostorm \
  --load_model_path ./checkpoints/pretraining/pt_neurostorm_mae_5ds.ckpt \
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
  --load_model_path ./checkpoints/pretraining/pt_neurostorm_mae_5ds.ckpt \
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

## 4. Parameter-efficient Tuning and Resuming

### 4.1 Task-specific Parameter-efficient Tuning

```bash
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --model neurostorm \
  --load_model_path ./checkpoints/pretraining/pt_neurostorm_mae_5ds.ckpt \
  --downstream_task_type classification \
  --task_name sex \
  --num_classes 2 \
  --tpt_strategy prompt \
  --prompt_len 50
```

`--tpt_strategy` supports `none` (full fine-tuning), `prompt`, `ln`, `linear`,
and `prompt_ln`. The `prompt` strategy trains per-block prompt tokens and the
task head while freezing the backbone; `--prompt_len` controls the number of
prompt tokens per block.

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
  --augment_only_affine \
  ...
```

### 5.4 Logging

```bash
python main.py \
  --loggername tensorboard \
  --project_name my_project \
  ...
```
