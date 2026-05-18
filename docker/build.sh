#!/usr/bin/env bash
# docker/build.sh — Build and test the Grand Challenge Docker image
set -euo pipefail

IMAGE_NAME="isles26-submission"
IMAGE_TAG="${1:-latest}"
CKPT_DIR="${CKPT_DIR:-outputs/checkpoints}"

echo "==> Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build \
    --build-arg CKPT_DIR="${CKPT_DIR}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f docker/Dockerfile \
    .

echo "==> Smoke test — running container with mock input"
mkdir -p /tmp/gc_test/input/images/t1-brain-mri
mkdir -p /tmp/gc_test/output/images/stroke-lesion-segmentation

# Verify the container starts without error on empty input (will error on missing
# image, which is expected — we just check the Python environment is intact)
docker run --rm \
    -v /tmp/gc_test/input:/input:ro \
    -v /tmp/gc_test/output:/output \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    python -c "import torch; import monai; import torchio; print('OK')"

echo "==> Image ready: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "    Export with: docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > ${IMAGE_NAME}.tar.gz"
