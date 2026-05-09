#!/bin/bash
# Example: Train baseline models from scratch on downstream tasks
# FC-based and graph-based models that don't use pretraining.

# BrainGNN on HCP1200 age regression
bash scripts/run_experiment.sh \
    --model braingnn \
    --dataset hcp1200 \
    --task task1 \
    --mode train_scratch

# BNT on HCP1200 age regression
bash scripts/run_experiment.sh \
    --model bnt \
    --dataset hcp1200 \
    --task task1 \
    --mode train_scratch

# LG-GNN on ADHD200 diagnosis
bash scripts/run_experiment.sh \
    --model lggnn \
    --dataset adhd200 \
    --task task3 \
    --mode train_scratch

# ComBrainTF on Cobre diagnosis
bash scripts/run_experiment.sh \
    --model combraintf \
    --dataset cobre \
    --task task3 \
    --mode train_scratch

# IBGNN on UCLA diagnosis
bash scripts/run_experiment.sh \
    --model ibgnn \
    --dataset ucla \
    --task task3 \
    --mode train_scratch

# BrainNetCNN on HCP1200
bash scripts/run_experiment.sh \
    --model brainnetcnn \
    --dataset hcp1200 \
    --task task1 \
    --mode train_scratch

# NeuroSTORM trained from scratch (no pretrained weights)
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset hcp1200 \
    --task task1 \
    --mode train_scratch

# Dry run: see the full command without executing
bash scripts/run_experiment.sh \
    --model braingnn \
    --dataset hcp1200 \
    --task task1 \
    --mode train_scratch \
    --dry_run
