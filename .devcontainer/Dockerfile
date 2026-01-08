# -----------------------------
# Base image: CUDA 12.8 + cuDNN
# (Works with PyTorch 2.9.1 GPU wheels)
# -----------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# -----------------------------
# System setup
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    git wget curl nano \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# -----------------------------
# Install pip + requirements
# -----------------------------
RUN pip install --upgrade pip

# Install PyTorch 2.9.1 with CUDA 12.8
RUN pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy only requirements first (cache-friendly)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# mount /workspace at runtime
WORKDIR /workspace

# -----------------------------
# Default command
# -----------------------------
CMD ["bash"]

