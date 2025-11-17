# ADNI Dataset Fine-tuning for NeuroSTORM

This directory contains scripts for fine-tuning NeuroSTORM on the ADNI dataset for Alzheimer's Disease (AD) vs Cognitively Normal (CN) classification.

## Dataset Structure

Your ADNI data should be organized as follows:

```
/mnt/dataset4/DATASETS/fsl_fmri/adni_split/
├── adni_ad_mni_train.txt    # Training split (list of file paths)
├── adni_ad_mni_val.txt      # Validation split (list of file paths)
└── adni_ad_mni_test.txt     # Test split (list of file paths)
```

Each `.txt` file should contain one file path per line, pointing to preprocessed fMRI `.nii.gz` files. The labels (AD or CN) are automatically extracted from the directory structure in the file paths (e.g., `/ad/` or `/cn/`).

**Example line in split file:**
```
/mnt/dataset4/DATASETS/fsl_fmri/ADNI(all)/mnispace/cn/ADNI_sub-035S6730_ses-01_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz_cn_zscore.nii.gz
```

## Scripts

### 1. Fine-tuning Script (`ft_neurostorm_adni_ad_cn.sh`)

Fine-tunes a pretrained NeuroSTORM model on the ADNI dataset.

**Usage:**
```bash
bash scripts/adni_downstream/ft_neurostorm_adni_ad_cn.sh [batch_size] [learning_rate] [pretrained_model_path]
```

**Parameters:**
- `batch_size` (default: 4): Batch size per GPU
- `learning_rate` (default: 5e-5): Learning rate for fine-tuning
- `pretrained_model_path` (default: `./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt`): Path to pretrained model checkpoint

**Example:**
```bash
# Use default parameters
bash scripts/adni_downstream/ft_neurostorm_adni_ad_cn.sh

# Custom batch size and learning rate
bash scripts/adni_downstream/ft_neurostorm_adni_ad_cn.sh 8 1e-4

# Specify all parameters
bash scripts/adni_downstream/ft_neurostorm_adni_ad_cn.sh 4 5e-5 ./output/neurostorm/my_pretrained_model.ckpt
```

**Important Configuration:**
Before running, update the `ADNI_SPLIT_DIR` variable in the script to point to your actual data location:
```bash
ADNI_SPLIT_DIR="/path/to/your/adni_split"
```

### 2. Testing Script (`test_neurostorm_adni_ad_cn.sh`)

Evaluates a trained model on the ADNI test set.

**Usage:**
```bash
bash scripts/adni_downstream/test_neurostorm_adni_ad_cn.sh [checkpoint_path] [batch_size]
```

**Parameters:**
- `checkpoint_path` (default: `./output/neurostorm/adni_ft_neurostorm_ad_cn/checkpoints/best.ckpt`): Path to trained model checkpoint
- `batch_size` (default: 8): Batch size for testing

**Example:**
```bash
# Use default checkpoint
bash scripts/adni_downstream/test_neurostorm_adni_ad_cn.sh

# Custom checkpoint and batch size
bash scripts/adni_downstream/test_neurostorm_adni_ad_cn.sh ./output/neurostorm/my_model/checkpoints/best.ckpt 16
```

## Model Configuration

The scripts use the following NeuroSTORM configuration:
- **Architecture**: NeuroSTORM with Swin Transformer + Mamba SSM
- **Depths**: [2, 2, 6, 2] (4 stages)
- **Embedding dimension**: 36
- **Number of heads**: [3, 6, 12, 24]
- **Input size**: 96×96×96×20 (spatial + temporal)
- **Window size**: 4×4×4×4
- **Patch size**: 4×4×4×1
- **Classification head**: v2 (multi-layer MLP)
- **Number of classes**: 2 (AD vs CN)

## Training Details

- **Max epochs**: 50
- **Optimizer**: AdamW (default in NeuroSTORM)
- **Learning rate**: 5e-5 (default, adjustable)
- **Batch size**: 4 per GPU (default, adjustable)
- **Strategy**: DDP (Distributed Data Parallel)
- **Logger**: TensorBoard

## Output

Training outputs are saved to:
```
./output/neurostorm/adni_ft_neurostorm_ad_cn_lr{lr}_bs{bs}/
├── checkpoints/
│   ├── best.ckpt          # Best model checkpoint
│   └── last.ckpt          # Last epoch checkpoint
├── tensorboard/           # TensorBoard logs
└── hparams.yaml          # Hyperparameters used
```

## Monitoring Training

You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir ./output/neurostorm/adni_ft_neurostorm_ad_cn_lr5e-5_bs4/tensorboard
```

Then open http://localhost:6006 in your browser.

## GPU Configuration

The scripts are configured to use GPUs by default. Adjust the `CUDA_VISIBLE_DEVICES` environment variable in the scripts to control which GPUs to use:

```bash
# Use GPUs 0,1,2,3 (training script default)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Use only GPU 0 (testing script default)
export CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size
```bash
bash scripts/adni_downstream/ft_neurostorm_adni_ad_cn.sh 2
```

### Issue: File not found errors
**Solution**: Verify your data paths:
1. Check that split files exist at `$ADNI_SPLIT_DIR`
2. Verify that `.nii.gz` files referenced in split files are accessible
3. Update `ADNI_SPLIT_DIR` in the scripts to match your actual data location

### Issue: NCCL errors in multi-GPU training
**Solution**: The scripts already set `NCCL_P2P_DISABLE=1`. If issues persist, try using fewer GPUs or running on a single GPU.

## Notes

1. **Data Format**: The ADNI dataset loader expects preprocessed fMRI data in NIfTI format (`.nii.gz`). The data should already be:
   - Registered to MNI space
   - Motion corrected
   - Z-score normalized (as indicated by `_zscore.nii.gz` suffix)

2. **Label Extraction**: Labels are automatically extracted from the file path:
   - Files in `/ad/` directory → Label 1 (Alzheimer's Disease)
   - Files in `/cn/` directory → Label 0 (Cognitively Normal)

3. **Pretrained Model**: Download or train a pretrained NeuroSTORM model first. The default path assumes you have a model at:
   ```
   ./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt
   ```

4. **Time Dimension**: The scripts expect fMRI data with at least 20 time points. If your data has fewer time points, adjust the `--sequence_length` parameter.

## Citation

If you use this code, please cite:

```bibtex
@article{neurostorm2024,
  title={NeuroSTORM: A Brain Foundation Model for fMRI Analysis},
  author={...},
  journal={...},
  year={2024}
}
```
