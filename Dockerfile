# 1. Use official CUDA 12.8 image with cuDNN and Ubuntu 24.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 2. Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    gcc-11 g++-11 build-essential \
    ninja-build \
    git wget curl ca-certificates \
    python3 python3-pip python3-dev python3-setuptools python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set gcc-11/g++-11 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

ENV CC=gcc-11
ENV CXX=g++-11
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# 4. Upgrade pip
# RUN pip install --upgrade pip

# 5. Install PyTorch and CUDA-related packages
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 6. Install tensorboard and other utilities
RUN pip install tensorboard tensorboardX tqdm ipdb nvitop monai

# 7. Install scientific and ML packages
RUN pip install pytorch-lightning==1.9.4 neptune nibabel nilearn numpy

# 8. Install PyTorch Geometric and dependencies (for graph-based models)
RUN pip install torch-geometric && \
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# 9. Install additional dependencies for FC computation and data processing
RUN pip install scikit-learn pandas h5py deepdish

# 10. Clone and install causal-conv1d
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git /opt/causal-conv1d \
 && cd /opt/causal-conv1d && git checkout v1.5.0.post8 \
 && TORCH_CUDA_ARCH_LIST="12.0" pip install --no-cache-dir --no-build-isolation -e .

# 11. Clone and install mamba (State-Spaces)
RUN git clone https://github.com/state-spaces/mamba.git /opt/mamba \
 && cd /opt/mamba && git checkout v2.2.2 \
 && TORCH_CUDA_ARCH_LIST="12.0" pip install --no-cache-dir --no-build-isolation -e .

# 12. (Optional) Create a non-root user for better security
ARG USERNAME=app
ARG UID=1001
RUN useradd -m -u ${UID} ${USERNAME}
USER ${USERNAME}

WORKDIR /workspace
CMD ["/bin/bash"]
