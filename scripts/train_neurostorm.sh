#!/bin/bash
# NeuroSTORM Training Script
# Uses run_experiment.sh with YAML configs from scripts/configs/

# =============================================================================
# Single GPU Pretraining (from scratch)
# =============================================================================

# Pretrain on GPU 0
# bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --mode pretrain --gpus 0

# =============================================================================
# Multi-GPU Pretraining (DDP)
# =============================================================================

# Pretrain on GPUs 0,1,2,3
bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --mode pretrain --gpus 0,1,2,3

# =============================================================================
# Multi-Dataset Pretraining (pretrain only)
# =============================================================================
# Pass a comma-separated list of dataset configs; the datamodule loads each
# dataset's img/ directory, skips metadata/label parsing (MAE/contrastive don't
# need labels), and concatenates the samples into one training set.

# Joint pretraining on HCP1200 + HCPA + HCPD + ABCD + UKB (all five pretrain datasets)
# bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200,hcpa,hcpd,abcd,ukb --mode pretrain --gpus 0,1,2,3

# =============================================================================
# Resume Pretraining from Checkpoint
# =============================================================================

# bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --mode pretrain --gpus 0,1,2,3 --load_model_path output/neurostorm/xxx/last.ckpt

# =============================================================================
# Finetune on downstream tasks (requires pretrained checkpoint)
# =============================================================================

# Finetune on gender classification (task1)
# bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --task task1 --mode finetune --gpus 0 --load_model_path output/neurostorm/xxx/last.ckpt

# Finetune on regression task (task2)
# bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --task task2 --mode finetune --gpus 0 --load_model_path output/neurostorm/xxx/last.ckpt

# =============================================================================
# Train from scratch (no checkpoint needed)
# =============================================================================

# bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --task task1 --mode train_scratch --gpus 0

# =============================================================================
# Override batch size / epochs
# =============================================================================

# bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --mode pretrain --gpus 0,1 --batch_size 8 --max_epochs 50

# =============================================================================
# Dry run (print command without executing)
# =============================================================================

# bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --mode pretrain --gpus 0,1,2,3 --dry_run
