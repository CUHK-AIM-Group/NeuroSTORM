<div align="center">    
 
# NeuroSTORM: Towards a general-purpose foundation model for fMRI analysis

</div>

<div align="center">
  <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a>
  <a href='https://www.nature.com/articles/s41551-026-01666-y'><img src='https://img.shields.io/badge/Paper-Nature_BME-red'></a>  &nbsp;
  <a href='https://cuhk-aim-group.github.io/NeuroSTORM/'><img src='https://img.shields.io/badge/Project-NeuroSTORM-green'></a> &nbsp;
  <a href='https://github.com/CUHK-AIM-Group/NeuroSTORM'><img src="https://img.shields.io/badge/GitHub-NeuroSTORM-9E95B7?logo=github"></a> &nbsp; 
  <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-NeuroSTORM-blue'></a> &nbsp; 
</div>


## Introduction

This repo provides a platform that covers all aspects involved in using deep learning for fMRI analysis. It is moderately encapsulated, highly customizable, and supports most common tasks and methods out of the box. 

This platform is proposed in our paper *Towards a General-Purpose Foundation Model for fMRI Analysis*. NeuroSTORM is a pretrained fMRI foundation model developed by the AIM group for fMRI analysis. You can run the pre-training and fine-tuning of NeuroSTORM in this repo. Specifically, our code provides the following:

- **Multiple Input Modalities**: Support for voxel-based (4D), ROI time series (2D), and functional connectivity (2D) data
- Preprocessing tools for fMRI volumes. You can use the tools to process fMRI volumes in MNI152 space into a unified 4D Volume (for models like NeuroSTORM), 2D time series data (for models like BNT), and 2D Functional Correlation Matrix (for models like BrainGNN).
- Trainer for pre-training, including the MAE-based mechanism proposed in NeuroSTORM and the contrastive learning approach in SwiFT.
- Trainer for fine-tuning, including both fully learnable parameters and Task-specific Prompt Learning as proposed in NeuroSTORM.
- A comprehensive fMRI benchmark, including five tasks: Age and Gender Prediction, Phenotype Prediction, Disease Diagnosis, fMRI Retrieval, and Task fMRI State Classification.
- Implementations of NeuroSTORM, SwiFT, BrainGNN, BrainNetworkTransformer (BNT), LG-GNN, Com-BrainTF, IBGNN, BrainNetCNN, and other commonly used fMRI analysis models.
- Customization options for all stages. You can quickly add custom preprocessing procedures, pre-training methods, fine-tuning strategies, new downstream tasks, and implement other models on the platform.

We welcome community contributions! Feel free to submit a PR to add support for your model or dataset.

## 🚀 Updates
* __[2026.05.08]__: Added BrainGNN, BNT, LG-GNN, Com-BrainTF, BrainNetCNN and IBGNN support. Framework now supports voxel (4D), ROI (2D), and FC (2D) inputs with dual-format (PT/H5) data loading.
* __[2026.03.24]__: Our paper has been accepted by [Nature Biomedical Engineering](https://www.nature.com/articles/s41551-026-01666-y).
* __[2025.12.09]__: Release demo code, including automated data and model downloads. Performed age regression, gender classification, and phenotype prediction on sample data. Release the code for all benchmark tasks (task4).
* __[2025.06.10]__: Release the [project website](https://cuhk-aim-group.github.io/NeuroSTORM/). Welcome to visit!
* __[2025.02.13]__: Release the code of NeuroSTORM model, (volume&ROI) data pre-processing, and benchmark (task1&2&3&5)
  

## 1. Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/CUHK-AIM-Group/NeuroSTORM.git
cd NeuroSTORM

# Create conda environment
conda create -n neurostorm python=3.11
conda activate neurostorm

# Set environment variables (auto-detects paths)
source ./set_env.sh

# Install dependencies
pip install -r requirements.txt

# Install mamba-ssm (for NeuroSTORM only, optional)
bash scripts/install_mamba.sh
```

**For detailed installation instructions, troubleshooting, and alternative methods, see [INSTALLATION.md](INSTALLATION.md).**

### Docker (Alternative)

```bash
docker build -t neurostorm:latest .
docker run --gpus all -it --rm -v $(pwd):/workspace neurostorm:latest
```

### Testing

NeuroSTORM includes a comprehensive test suite with automatic CI/CD.

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run only unit tests
make test-unit

# Run CI checks locally
make ci
```

**CI Status**: Tests automatically run on every push via GitHub Actions. See [tests/README.md](tests/README.md) for details.


## 2. Basic Usage

### Pre-training

```bash
# MAE pre-training on NeuroSTORM
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200 \
  --model neurostorm \
  --pretraining \
  --use_mae \
  --mask_ratio 0.75 \
  --batch_size 16 \
  --max_epochs 100
```

### Fine-tuning

```bash
# Gender classification
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200 \
  --model neurostorm \
  --load_model_path ./pretrained_models/neurostorm_mae.pth \
  --downstream_task_type classification \
  --task_name sex \
  --num_classes 2 \
  --batch_size 32 \
  --max_epochs 50

# Age regression
python main.py \
  --dataset_name HCP1200 \
  --image_path ./data/HCP1200 \
  --model neurostorm \
  --downstream_task_type regression \
  --task_name age \
  --num_classes 1 \
  --batch_size 32 \
  --max_epochs 50
```

**For detailed usage, data preparation, and advanced options, see [USER_GUIDE.md](USER_GUIDE.md).**


## 3. Project Structure

Our directory structure looks like this:

```
├── datasets                           <- tools and dataset class
│   ├── atlas                          <- examples of brain atlas
│   ├── preprocessing_volume.py        <- remove background, z-normalization, save as pt or h5 files
│   ├── compute_stats_and_mask.py      <- compute statistics and save as pt or h5 files
│   ├── generate_roi_data_from_nii.py  <- extract ROI time series from volumetric fMRI
│   ├── compute_fc.py                  <- compute functional connectivity matrices
│   ├── compute_atlas_map.py           <- compute atlas map for masking
│   ├── brain_extraction.sh            <- brain extraction with FSL BET
│   ├── fmri_datasets.py               <- voxel-based dataset loaders
│   └── roi_datasets.py                <- ROI and FC dataset loaders
│
├── models                 
│   ├── heads                          <- task heads
│   │   ├── base.py                    <- BaseHead base class and head registry
│   │   ├── cls_head.py                <- for classification tasks
│   │   ├── emb_head.py                <- for contrastive learning
│   │   └── reg_head.py                <- for regression tasks
│   ├── load_model.py                  <- load any backbone or head network
│   ├── patchembedding.py              <- patch embedding for vision transformers
│   ├── neurostorm.py                  <- NeuroSTORM
│   ├── swift.py                       <- SwiFT
│   ├── braingnn.py                    <- BrainGNN
│   ├── bnt.py                         <- BrainNetworkTransformer
│   ├── lggnn.py                       <- LG-GNN
│   ├── combraintf.py                  <- Com-BrainTF
│   ├── ibgnn.py                       <- IBGNN
│   ├── brainnetcnn.py                 <- BrainNetCNN
│   └── lightning_model.py             <- the basic lightning model class
│
├── utils                              <- utility modules
│   ├── data_module.py                 <- PyTorch Lightning data module
│   ├── parser.py                      <- argument parsing utilities
│   ├── losses.py                      <- loss functions
│   ├── metrics.py                     <- evaluation metrics
│   ├── lr_scheduler.py                <- learning rate schedulers
│   └── seed_creation.py               <- dataset split creation
│
├── tests                              <- test suite
│   ├── test_model_loading.py          <- model import and forward pass tests
│   ├── test_dual_format.py            <- PT/H5 dual format tests
│   └── test_atlas_masking.py          <- atlas masking tests
│
├── scripts                            <- training and utility scripts
│   ├── hcp_pretrain                   <- pre-training scripts for HCP
│   ├── hcp_downstream                 <- fine-tuning scripts for HCP
│   ├── dataset_download               <- dataset download scripts
│   ├── install_mamba.sh               <- automatic mamba-ssm installer
│   ├── run_demo.sh                    <- inference demo
│   ├── run_braingnn.sh                <- BrainGNN training example
│   └── run_bnt.sh                     <- BNT training example
│
├── docs                               <- project website
├── pretrained_models                   <- pre-trained model checkpoints
├── .github/workflows/ci.yml           <- GitHub Actions CI configuration
│ 
├── main.py                            <- training entry point
├── demo.py                            <- unified inference (single file & dataset)
├── set_env.sh                         <- environment variable setup
├── Dockerfile                         <- Docker image configuration
├── Makefile                           <- test and dev commands
├── requirements.txt                   <- Python dependencies
├── pytest.ini                         <- test configuration
├── INSTALLATION.md                    <- detailed installation guide
├── USER_GUIDE.md                      <- complete user guide
├── LICENSE                            <- MIT license
└── README.md
```

<br>


## 4. Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete guide for data preparation, training, and fine-tuning
- **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation instructions

## 5. Citation

If you use NeuroSTORM in your research, please cite:

```bibtex
@article{wang2026towards,
  title={Towards a general-purpose foundation model for functional MRI analysis},
  author={Wang, Cheng and Jiang, Yu and Peng, Zhihao and Li, Chenxin and Bang, Chang-bae and Zhao, Lin and Fu, Wanyi and Lv, Jinglei and Sepulcre, Jorge and Yang, Carl and others},
  journal={Nature Biomedical Engineering},
  pages={1--12},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

## 5. Acknowledgments

We gratefully acknowledge the following projects and their authors:

**Preprocessing Pipelines:**
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs) - FMRIB Software Library
- [fMRIPrep](https://fmriprep.org/en/stable/) - fMRI preprocessing pipeline
- [HCP Pipelines](https://github.com/Washington-University/HCPpipelines) - Human Connectome Project pipelines

**Models:**
- [SwiFT](https://github.com/Transconnectome/SwiFT) - Swin 4D fMRI Transformer
- [BrainGNN](https://github.com/LifangHe/BrainGNN_Pytorch) - Interpretable Brain Graph Neural Network
- [BrainNetworkTransformer](https://github.com/Wayfear/BrainNetworkTransformer) - Brain Network Transformer
- [LG-GNN](https://github.com/cnuzh/LG-GNN) - Learnable Graph GNN
- [Com-BrainTF](https://github.com/ubc-tea/Com-BrainTF) - Community-aware Brain Transformer
- [IBGNN](https://github.com/HennyJie/IBGNN) - Interpretable Brain Graph Neural Network
- [BrainNetCNN](https://github.com/nicofarr/brainnetcnn) - Convolutional Neural Networks for Brain Networks

**Libraries:**
- [MONAI](https://monai.io/) - Medical Open Network for AI
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph Neural Network Library
- [Mamba](https://github.com/state-spaces/mamba) - Linear-Time Sequence Modeling
- [MindEyeV2](https://github.com/MedARC-AI/MindEyeV2) - fMRI-to-Image Reconstruction
