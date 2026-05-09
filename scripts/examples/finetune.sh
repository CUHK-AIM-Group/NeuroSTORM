#!/bin/bash
# Example: Finetune pretrained models on downstream tasks
# These commands load pretrained weights and finetune on classification/regression.

# NeuroSTORM on HCP1200 sex classification
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset hcp1200 \
    --task task1 \
    --mode finetune

# NeuroSTORM on HCP1200 age regression (override task_name)
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset hcp1200 \
    --task task1 \
    --mode finetune \
    --task_name age

# NeuroSTORM on ADHD200 diagnosis classification
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset adhd200 \
    --task task3 \
    --mode finetune

# NeuroSTORM on Cobre diagnosis (4-class)
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset cobre \
    --task task3 \
    --mode finetune

# NeuroSTORM on UCLA diagnosis (4-class)
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset ucla \
    --task task3 \
    --mode finetune

# NeuroSTORM on HCPTASK state classification (7-class)
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset hcptask \
    --task task5 \
    --mode finetune

# Multi-GPU finetuning with custom batch size
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset hcp1200 \
    --task task1 \
    --mode finetune \
    --gpus 0,1,2,3 \
    --batch_size 8
