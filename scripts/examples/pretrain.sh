#!/bin/bash
# Example: Pretrain NeuroSTORM on different datasets
# These commands demonstrate MAE pretraining on all supported pretrain datasets.

# HCP1200 pretraining (default batch=12)
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset hcp1200 \
    --mode pretrain

# ABCD pretraining (smaller batch due to memory)
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset abcd \
    --mode pretrain

# UKB pretraining
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset ukb \
    --mode pretrain

# SwiFT contrastive pretraining on HCP1200
bash scripts/run_experiment.sh \
    --model swift \
    --dataset hcp1200 \
    --mode pretrain \
    --gpus 0,1

# Joint pretraining across multiple datasets (pretrain only).
# Pass a comma-separated dataset list; each dataset must have a config under
# scripts/configs/datasets/. Their img/ directories are scanned and the
# resulting samples are concatenated into a single training set. The five
# pretrain-capable datasets today are HCP1200, HCPA, HCPD, ABCD, and UKB.
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset hcp1200,hcpa,hcpd,abcd,ukb \
    --mode pretrain \
    --gpus 0,1,2,3
