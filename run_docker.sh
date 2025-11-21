#!/bin/bash
# -----------------------------
# Simple script to build and run the GPU container
# -----------------------------

# Image name
IMAGE_NAME="neural-net-gpu"

# Check if image exists, build if not
if [[ "$(sudo docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    sudo docker build -t $IMAGE_NAME -f .devcontainer/Dockerfile .
else
    echo "Docker image $IMAGE_NAME already exists."
fi

# Run docker container
sudo docker run --rm --gpus all -it \
    -v "$(pwd)":/workspace \
    $IMAGE_NAME \
    bash

