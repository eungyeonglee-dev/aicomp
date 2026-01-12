#!/bin/bash

CONTAINER_NAME="etri_test_container"
CONTAINER_IMAGE="etri_test_image:latest"
CONTAINER_WORKSPACE_DIR="/workspace/aicomp"

# 1=============== Remove existing container ===============
CID=$(sudo docker ps -q -f name="^/${CONTAINER_NAME}$")

if sudo docker ps -q --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' exists."
    echo "Removing '$CONTAINER_NAME' container."
    # 2>/dev/null: suppress error messages
    sudo docker stop $CONTAINER_NAME 2>/dev/null

    echo "Removing '$CONTAINER_NAME' container."
    sudo docker rm $CONTAINER_NAME

    echo "Container '$CONTAINER_NAME' removed successfully."
fi

echo "Creating new container '$CONTAINER_NAME'."

# 2=============== Clean up port ===============
echo "Checking port range 29500-29509"
for port in $(seq 29500 29509); do
    if sudo lsof -i :$port >/dev/null 2>&1; then
        echo "Port $port is already in use."
        echo "Cleaning up port $port"
        sudo lsof -i :$port | xargs -r sudo kill -9
        echo "Port $port cleaned up."
        sleep 2
    else
        echo "Port $port is not in use."
    fi
done

# 3=============== Run container ===============
sudo docker run -d --gpus all -i -t --name $CONTAINER_NAME \
            -v ${HOME}/workspace/aicomp:$CONTAINER_WORKSPACE_DIR \
            --ipc=host \
            --network=host \
            -w $CONTAINER_WORKSPACE_DIR \
            -e LLAMA_ACCESS_TOKEN=$LLAMA_ACCESS_TOKEN \
            $CONTAINER_IMAGE bash -lc "tail -f /dev/null"
echo "Container '$CONTAINER_NAME' created."

# install gpustat
echo "Installing gpustat."
sudo docker exec -it $CONTAINER_NAME pip install gpustat
echo "Gpustat installed successfully."

# 4=============== Run model ===============
MODEL_SIZE=$1
NODE_RANK=$2
MASTER_ADDR=$3
NNODES=$4
NPROC_PER_NODE=$5
USE_CACHE=$6
PP_SIZE=$7
TP_SIZE=$8
DP_SIZE=$9


if [ $MODEL_SIZE -eq 70 ]; then
    MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
fi

echo "Run $CONTAINER_NAME"
sudo docker exec -it $CONTAINER_NAME \
                /bin/bash -lc 'cd /workspace/aicomp/opt_prime/tutoruslabs && LOGFILE=./results/$(date +%Y%m%d%H%M%S).log; \
                GPULOGFILE=./results/$(date +%Y%m%d%H%M%S)_gpustats.log; MEMLOGFILE=./results/$(date +%Y%m%d%H%M%S)_memstats.log; \
                MODEL_NAME='$MODEL_NAME'; NODE_RANK='$NODE_RANK'; MASTER_ADDR='$MASTER_ADDR'; NNODES='$NNODES'; NPROC_PER_NODE='$NPROC_PER_NODE'; USE_CACHE='$USE_CACHE'; PP_SIZE='$PP_SIZE'; TP_SIZE='$TP_SIZE'; DP_SIZE='$DP_SIZE'; \
                (while true; do echo "===== $(date "+%F %T") ====="; gpustat --no-color || true; echo; sleep 1; done) >> "$GPULOGFILE" 2>&1 & GPUSTAT_PID=$!; \
                (while true; do echo "===== $(date "+%F %T") ====="; free -h || true; echo; sleep 1; done) >> "$MEMLOGFILE" 2>&1 & MEMSTAT_PID=$!; \
                trap "kill $GPUSTAT_PID $MEMSTAT_PID 2>/dev/null || true; wait $GPUSTAT_PID $MEMSTAT_PID 2>/dev/null || true" EXIT INT TERM; \
                bash ./run_rdzv_70b.sh "$MODEL_NAME" "$NODE_RANK" "$MASTER_ADDR" "$NNODES" "$NPROC_PER_NODE" "$USE_CACHE" "$PP_SIZE" "$TP_SIZE" "$DP_SIZE" > "$LOGFILE" 2>&1;'                
