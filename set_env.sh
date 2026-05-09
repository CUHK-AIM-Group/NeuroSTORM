#!/bin/bash
# NeuroSTORM Environment Setup Script
# This script sets up environment variables for GCC and CUDA

# Auto-detect conda environment path
if [ -z "$CONDA_PREFIX" ]; then
    echo "Warning: Conda environment not activated. Please run 'conda activate neurostorm' first."
    CONDA_ENV_PATH="${CONDA_PREFIX:-$(conda info --base)/envs/neurostorm}"
else
    CONDA_ENV_PATH="$CONDA_PREFIX"
fi

# Auto-detect CUDA installation (prefer CUDA 12.8 even if CUDA_HOME is preset)
if [ -d "/usr/local/cuda-12.8" ]; then
    if [ -n "$CUDA_HOME" ] && [ "$CUDA_HOME" != "/usr/local/cuda-12.8" ]; then
        echo "Overriding existing CUDA_HOME ($CUDA_HOME) with /usr/local/cuda-12.8"
    fi
    CUDA_HOME="/usr/local/cuda-12.8"
elif [ -z "$CUDA_HOME" ]; then
    # Fall back to other common CUDA installation paths
    if [ -d "/usr/local/cuda" ]; then
        CUDA_HOME="/usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        CUDA_HOME="/opt/cuda"
    else
        echo "Warning: CUDA installation not found. Please set CUDA_HOME manually."
        CUDA_HOME="/usr/local/cuda-12.8"  # Default fallback
    fi
fi

echo "Setting up NeuroSTORM environment..."
echo "Conda environment: $CONDA_ENV_PATH"
echo "CUDA home: $CUDA_HOME"

# Set GCC paths (for conda-installed GCC)
if [ -d "$CONDA_ENV_PATH/x86_64-conda-linux-gnu/lib64" ]; then
    export LD_LIBRARY_PATH="$CONDA_ENV_PATH/x86_64-conda-linux-gnu/lib64:$LD_LIBRARY_PATH"
fi

# Set CUDA paths
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64:$LIBRARY_PATH"

# Set compiler environment variables for mamba-ssm compilation
export CC=gcc-11
export CXX=g++-11

# Set CUDA architecture for compilation (adjust based on your GPU)
# Common values: 8.0 (A100), 8.6 (RTX 3090), 8.9 (RTX 4090), 9.0 (H100), 12.0 (Blackwell)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

echo "Environment variables set successfully!"
echo ""
echo "To verify your setup:"
echo "  - GCC version: gcc --version"
echo "  - CUDA version: nvcc --version"
echo "  - Python version: python --version"
echo ""
echo "You can now proceed with installing dependencies."
