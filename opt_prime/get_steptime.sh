#!/bin/bash

CONTAINER_NAME="optimus-timelog"
DOCKER_IMAGE_NAME="optimus-timelog"
WORKSPACE_DIR="${HOME}/workspace/aicomp"
CONTAINER_DIR="/workspace/aicomp"

# delete existing container if it exists
if [ "$(sudo docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    echo "==> Removing existing container: $CONTAINER_NAME"
    sudo docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
    sudo docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi  

# run a new container
echo "==> Starting new container: $CONTAINER_NAME"
sudo docker run -d --name $CONTAINER_NAME \
                --gpus all \
                --ipc=host \
                --network=host \
                --mount type=bind,source="$WORKSPACE_DIR",target="$CONTAINER_DIR" \
                --mount type=bind,source="/home/ieg95/.ssh",target="/root/.ssh",readonly \
                -w $CONTAINER_DIR \
                "$DOCKER_IMAGE_NAME" \
                bash -lc "tail -f /dev/null"
# run the training script inside the container in order to get the steptime
echo "==> Running llama training script inside the container..."
sudo docker exec -it $CONTAINER_NAME bash -lc "cd /workspace/aicomp/opt_prime && bash get_steptime_run_llama.sh"


                