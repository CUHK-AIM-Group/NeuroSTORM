#!/bin/bash
# Automatic Mamba-SSM Installation Script for NeuroSTORM
# This script installs causal-conv1d and mamba-ssm with proper version control

set -e  # Exit on error

echo "=========================================="
echo "Mamba-SSM Installation Script"
echo "=========================================="
echo ""

# Check if conda environment is activated
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: Conda environment not activated."
    echo "Please run: conda activate neurostorm"
    exit 1
fi

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Warning: nvcc not found. CUDA may not be properly installed."
    echo "Please set CUDA_HOME environment variable."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set default CUDA_HOME if not set
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda-12.8" ]; then
        export CUDA_HOME="/usr/local/cuda-12.8"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        echo "Error: CUDA_HOME not set and CUDA not found in default locations."
        echo "Please set CUDA_HOME manually: export CUDA_HOME=/path/to/cuda"
        exit 1
    fi
fi

echo "Using CUDA_HOME: $CUDA_HOME"

# Set compilers
export CC=gcc-11
export CXX=g++-11

# Check GCC version
if ! command -v gcc-11 &> /dev/null; then
    echo "Warning: gcc-11 not found. Trying default gcc..."
    export CC=gcc
    export CXX=g++
fi

echo "Using CC: $CC"
echo "Using CXX: $CXX"

# Detect GPU architecture
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "Detected GPU: $GPU_NAME"

    # Set CUDA architecture based on GPU
    if [[ $GPU_NAME == *"H100"* ]]; then
        CUDA_ARCH="9.0"
    elif [[ $GPU_NAME == *"4090"* ]] || [[ $GPU_NAME == *"4080"* ]]; then
        CUDA_ARCH="8.9"
    elif [[ $GPU_NAME == *"A100"* ]]; then
        CUDA_ARCH="8.0"
    elif [[ $GPU_NAME == *"3090"* ]] || [[ $GPU_NAME == *"3080"* ]]; then
        CUDA_ARCH="8.6"
    elif [[ $GPU_NAME == *"V100"* ]]; then
        CUDA_ARCH="7.0"
    else
        CUDA_ARCH="8.0"  # Default
        echo "Warning: Unknown GPU. Using default CUDA architecture: $CUDA_ARCH"
    fi
else
    CUDA_ARCH="8.0"  # Default
    echo "Warning: nvidia-smi not found. Using default CUDA architecture: $CUDA_ARCH"
fi

# Allow user override
if [ -n "$TORCH_CUDA_ARCH_LIST" ]; then
    CUDA_ARCH="$TORCH_CUDA_ARCH_LIST"
    echo "Using user-specified CUDA architecture: $CUDA_ARCH"
else
    export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
    echo "Using detected CUDA architecture: $CUDA_ARCH"
fi

echo ""
echo "=========================================="
echo "Installing causal-conv1d..."
echo "=========================================="

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Clone and install causal-conv1d
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.5.0.post8

echo "Compiling causal-conv1d (this may take a few minutes)..."
TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" pip install --no-cache-dir --no-build-isolation .

if [ $? -eq 0 ]; then
    echo "✓ causal-conv1d installed successfully!"
else
    echo "✗ causal-conv1d installation failed!"
    exit 1
fi

cd "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Installing mamba-ssm..."
echo "=========================================="

# Clone and install mamba
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v2.2.2

echo "Compiling mamba-ssm (this may take a few minutes)..."
TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" pip install --no-cache-dir --no-build-isolation .

if [ $? -eq 0 ]; then
    echo "✓ mamba-ssm installed successfully!"
else
    echo "✗ mamba-ssm installation failed!"
    exit 1
fi

# Clean up
cd ~
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

# Test import
python -c "from mamba_ssm import Mamba; print('✓ Mamba-SSM import successful!')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "You can now use NeuroSTORM with Mamba-SSM support."
else
    echo ""
    echo "=========================================="
    echo "Installation completed with warnings"
    echo "=========================================="
    echo ""
    echo "Mamba-SSM was installed but import test failed."
    echo "This may be normal if you're not in the correct environment."
    echo "Please verify by running:"
    echo "  python -c 'from mamba_ssm import Mamba'"
fi

echo ""
echo "Installation log saved to: /tmp/mamba_install.log"
