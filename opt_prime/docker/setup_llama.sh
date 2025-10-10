#!/usr/bin/env bash

# === Set up environment ===
DOCKER_IMAGE_NAME="pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel"
CONTAINER_NAME="llama1b"
WORKSPACE_DIR="~/workspace/aicomp"
CONTAINER_DIR="/workspace/aicomp"

# === Build Docker image if it doesn't exist ===
# ~ extended image
WORKSPACE_DIR="${WORKSPACE_DIR/#\~/$HOME}"

# Check if workspace directory exists
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Workspace directory $WORKSPACE_DIR does not exist. Please create it and try again."
    exit 1
fi

# Check if the Docker image already exists
echo "Checking for Docker image $DOCKER_IMAGE_NAME..."
if [[ "$(sudo docker images -q $DOCKER_IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Docker image $DOCKER_IMAGE_NAME not found. Building the image..."
    sudo docker pull $DOCKER_IMAGE_NAME
else
    echo "Docker image $DOCKER_IMAGE_NAME already exists."
fi

# === Run Docker container ===
# Check if a container with the same name is already running
if [ "$(sudo docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "A container named $CONTAINER_NAME is already running. Please stop it before starting a new one."
    exit 1
fi

# Making docker container
echo "Starting Docker container $CONTAINER_NAME..."
sudo docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    --ipc=host \
    --network=host \
    --mount type=bind,source="$WORKSPACE_DIR",target="$CONTAINER_DIR" \
    -w $CONTAINER_DIR \
    "$DOCKER_IMAGE_NAME" \
    bash -lc "tail -f /dev/null"

# === Install necessary packages inside the container ===
echo "APT update and install necessary packages..."
sudo docker exec -u root "$CONTAINER_NAME" bash -lc '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends \
        net-tools iproute2 openssh-client vim wget git
    if dpkg -s inetutils-ping >/dev/null 2>&1; then
        if apt-cache show inetutils-ping >/dev/null 2>&1; then
            apt-get install -y inetutils-ping;
        else
            apt-get install -y iputils-ping;
        fi
    fi
    apt-get clean
    rm -rf /var/lib/apt/lists/*
'
# === Install Python packages ===
echo "Installing Python packages..."
sudo docker exec "$CONTAINER_NAME" bash -lc '
    python -m pip3 install --upgrade pip
    python -m pip3 install \
        "transformers=4.46.2" datasets evaluate sentencepiece \
        accelerate bitsandbytes \
        pandas numpy scipy scikit-learn \
        psutil
'

# === Setup Git configuration ===
echo "Setting up Git configuration..."
GIT_USER_NAME="ieungyeong"
GIT_USER_EMAIL="anhanmu78@gmail.com"
sudo docker exec "$CONTAINER_NAME" bash -lc "
    git config --global user.name \"$GIT_USER_NAME\"
    git config --global user.email \"$GIT_USER_EMAIL\"
    git config --global --add safe.directory \"$CONTAINER_DIR\"
" 

echo "Setup complete."
echo "Name of the Docker container: $CONTAINER_NAME"
echo "MOUNT INFO: $WORKSPACE_DIR -> $CONTAINER_DIR"
echo "To access the container, use: docker exec -it $CONTAINER_NAME bash"