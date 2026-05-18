# docker/Dockerfile
# Grand Challenge compatible Docker image for ISLES'26
# Base: PyTorch with CUDA 12.1 (Ampere / Hopper compatible)

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml

# Copy project
COPY . /opt/ml/isles26
WORKDIR /opt/ml/isles26

# Install Python deps (no dev extras)
RUN pip install --no-cache-dir -e "." \
    && pip install --no-cache-dir hd-bet  # skull-stripping

# Copy checkpoints (built separately, then COPY-ed in CI)
# In practice: docker build --build-arg CKPT_DIR=outputs/checkpoints ...
ARG CKPT_DIR=outputs/checkpoints
COPY ${CKPT_DIR} /opt/ml/checkpoints

# Grand Challenge entrypoint
ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output

ENTRYPOINT ["python", "docker/process.py"]
