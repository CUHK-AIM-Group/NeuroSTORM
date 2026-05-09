# NeuroSTORM Installation Guide

Complete installation guide for NeuroSTORM platform with automatic environment detection.

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/CUHK-AIM-Group/NeuroSTORM.git
cd NeuroSTORM

# 2. Create conda environment
conda create -n neurostorm python=3.11
conda activate neurostorm

# 3. Set environment variables (auto-detects paths)
source ./set_env.sh

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install torch-scatter and torch-sparse (for graph-based models)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# 6. Install mamba-ssm (for NeuroSTORM only)
bash scripts/install_mamba.sh
```

---

## Detailed Installation

### Step 1: System Requirements

**Recommended:**
- Python: 3.11
- CUDA: 12.8
- RAM: 32GB+
- GPU: NVIDIA A100, RTX 4090, or H100

### Step 2: Create Conda Environment

```bash
# Create environment
conda create -n neurostorm python=3.11
conda activate neurostorm

# Verify Python version
python --version  # Should show Python 3.11.x
```

### Step 3: Set Environment Variables

The `set_env.sh` script automatically detects your conda and CUDA paths:

```bash
source ./set_env.sh
```

**What it does:**
- Auto-detects conda environment path from `$CONDA_PREFIX`
- Auto-detects CUDA installation (tries `/usr/local/cuda-12.8`, `/usr/local/cuda`, `/opt/cuda`)
- Sets up GCC-11 for mamba-ssm compilation
- Configures CUDA architecture for your GPU

**Manual Configuration (if auto-detection fails):**

```bash
# Set conda environment path
export CONDA_ENV_PATH=/path/to/anaconda3/envs/neurostorm

# Set CUDA home
export CUDA_HOME=/usr/local/cuda-12.8

# Set compilers
export CC=gcc-11
export CXX=g++-11

# Set CUDA architecture (adjust for your GPU)
export TORCH_CUDA_ARCH_LIST="12.0"  # Blackwell
# export TORCH_CUDA_ARCH_LIST="9.0"   # H100
# export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090
# export TORCH_CUDA_ARCH_LIST="8.6"   # RTX 3090
# export TORCH_CUDA_ARCH_LIST="8.0"   # A100
```

### Step 4: Install GCC (Linux only, optional)

For mamba-ssm compilation, you need GCC 11:

```bash
# Install via conda
conda install gcc_impl_linux-64=11.2.0 gxx_linux-64=11.2.0 ninja

# Create symlinks
ln -s ${CONDA_ENV_PATH}/libexec/gcc/x86_64-conda-linux-gnu/11.2.0/gcc ${CONDA_ENV_PATH}/bin/gcc
```

**Or install system-wide:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install gcc-11 g++-11 build-essential

# Set as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

### Step 5: Install Core Dependencies

```bash
# PyTorch (CUDA 12.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Fix setuptools for pytorch-lightning compatibility
pip install "setuptools<81"

# PyTorch Lightning and utilities
pip install pytorch-lightning==1.9.4
pip install tensorboard tensorboardX tqdm ipdb nvitop

# Pin transformers for mamba-ssm compatibility
pip install "transformers<=4.39.3"

# Medical imaging libraries
pip install monai nibabel nilearn numpy scipy
```

**Verify PyTorch installation:**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Step 6: Install Model-Specific Dependencies

#### For Graph-Based Models (BrainGNN, LG-GNN, IBGNN)

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

**Verify installation:**

```bash
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"
```

#### For FC-Based Models (All FC models)

```bash
pip install scikit-learn pandas h5py deepdish
```

### Step 7: Install Mamba-SSM (for NeuroSTORM only)

**Option 1: Automatic installation (recommended)**

```bash
bash scripts/install_mamba.sh
```

**Option 2: Manual installation**

```bash
# Install causal-conv1d
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.5.0.post8
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-cache-dir --no-build-isolation .
cd ..

# Install mamba
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v2.2.2
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-cache-dir --no-build-isolation .
cd ..
```

**Verify installation:**

```bash
python -c "from mamba_ssm import Mamba; print('Mamba-SSM installed successfully!')"
```

---

## Docker Installation

For a hassle-free installation:

```bash
# Build image
docker build -t neurostorm:latest .

# Run container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  --shm-size=8g \
  neurostorm:latest

# Inside container
cd /workspace
python main.py --help
```

---

## GPU Configuration

NeuroSTORM supports single-GPU and multi-GPU training.

### Single GPU Training

By default, NeuroSTORM uses a single GPU:

```bash
python main.py --dataset_name HCP1200 --model neurostorm --pretraining --batch_size 4
```

### Specify GPU ID

To use a specific GPU:

```bash
# Use GPU 0
python main.py --dataset_name HCP1200 --model neurostorm --pretraining --batch_size 4 --gpu_ids 0

# Use GPU 2
python main.py --dataset_name HCP1200 --model neurostorm --pretraining --batch_size 4 --gpu_ids 2
```

### Multi-GPU Training

To use multiple GPUs with Distributed Data Parallel (DDP):

```bash
# Use GPUs 0, 1, 2
python main.py --dataset_name HCP1200 --model neurostorm --pretraining --batch_size 4 --gpu_ids 0,1,2

# Use first 4 GPUs
python main.py --dataset_name HCP1200 --model neurostorm --pretraining --batch_size 4 --num_gpus 4

# Use all available GPUs (default)
python main.py --dataset_name HCP1200 --model neurostorm --pretraining --batch_size 4
```

**Note**: When using multi-GPU training, the effective batch size is `batch_size * num_gpus`. You may need to adjust the learning rate accordingly.

---

## Verification

After installation, verify everything works:

```bash
# Test PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Test PyTorch Geometric
python -c "import torch_geometric; print('PyG OK')"

# Test Mamba-SSM
python -c "from mamba_ssm import Mamba; print('Mamba OK')"

# Test NeuroSTORM
python -c "from models.neurostorm import NeuroSTORM; print('NeuroSTORM OK')"

# Run quick test
python main.py --help
```
