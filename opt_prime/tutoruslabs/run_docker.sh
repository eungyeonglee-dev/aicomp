#!/bin/bash

CONTAINER_NAME="etri_test_container"
CONTAINER_IMAGE="etri_test_image:latest"
CONTAINER_WORKSPACE_DIR="/workspace/aicomp"

# 1=============== Remove container ===============
echo "===> Removing '$CONTAINER_NAME' container."
# 2>/dev/null: suppress error messages
sudo docker rm -f $CONTAINER_NAME 2>/dev/null || true
sleep 2
echo "===> Container '$CONTAINER_NAME' removed successfully."


echo "===> Creating new container '$CONTAINER_NAME'."

# 2=============== Clean up port ===============
echo "===> Checking port range 29500-29509"
for port in $(seq 29500 29509); do
    if sudo lsof -i :$port >/dev/null 2>&1; then
        echo "===> Port $port is already in use."
        echo "===> Cleaning up port $port"
        sudo kill -9 $(sudo lsof -t -i :$port) 2>/dev/null || true
        echo "===> Port $port cleaned up."
        sleep 2
    else
        echo "===> Port $port is not in use."
    fi
done

# 3=============== Run container ===============
HOSTNAME=$(hostname)
case "$HOSTNAME" in
    "s1")
        IB_IFNAME="ibp194s0"
        ETH_IFNAME="enp34s0f0"
        ;;
    "s5")
        IB_IFNAME="ibp194s0"
        ETH_IFNAME="enp34s0f0"
        ;;
    "s6")
        IB_IFNAME="ibp194s0"
        ETH_IFNAME="enp33s0f0"
        ;;
    "s8")
        IB_IFNAME="ibp194s0"
        ETH_IFNAME="enp35s0f0np0"
        ;;
    *)
        echo "===> Unknown hostname: $HOSTNAME"
        echo "===> Please check the hostname and set the IB_IFNAME and ETH_IFNAME."
        exit 1
    ;;
esac    
sudo docker run -d --gpus all --name $CONTAINER_NAME \
            -v ${HOME}/workspace/aicomp:$CONTAINER_WORKSPACE_DIR \
            --ipc=host \
            --network=host \
            --cap-add=NET_ADMIN \
            -w $CONTAINER_WORKSPACE_DIR \
            -e LLAMA_ACCESS_TOKEN=$LLAMA_ACCESS_TOKEN \
            -e NCCL_DEBUG=INFO \
            -e NCCL_DEBUG_SUBSYS=ALL \
            -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
            -e TORCH_SHOW_CPP_STACKTRACES=1 \
            -e NCCL_SOCKET_IFNAME=$ETH_IFNAME \
            -e GLOO_SOCKET_IFNAME=$ETH_IFNAME \
            -e NCCL_IB_DISABLE=1 \
            -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
            -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=18000 \
            -e HF_DATASETS_OFFLINE=1 \
            -e HF_HUB_OFFLINE=1 \
            -e NCCL_SOCKET_FAMILY=AF_INET \
            -e GLOO_SOCKET_FAMILY=AF_INET \
            -e PYTHON_PREFER_IPV4=1 \
            --entrypoint /bin/bash \
            $CONTAINER_IMAGE -c "tail -f /dev/null"
echo "===> Container '$CONTAINER_NAME' created."

# install gpustat
echo "===> Installing gpustat."
sudo docker exec -it $CONTAINER_NAME pip install gpustat
echo "===> Gpustat installed successfully."

# infiniband deactivation
echo "===> Deactivating network without ethernet interface."
sudo docker exec $CONTAINER_NAME ip link set $IB_IFNAME down 2>/dev/null || true
sudo docker exec $CONTAINER_NAME ip link set docker0 down 2>/dev/null || true
case "$HOSTNAME" in
    "s6")
        sudo docker exec $CONTAINER_NAME ip link set br-76f0280ffba5 down 2>/dev/null || true
        ;;
    *)
        ;;
esac

echo "===> Network Interfaces after disabling IB"
sudo docker exec $CONTAINER_NAME hostname -I

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

echo "===> Run $CONTAINER_NAME"
sudo docker exec -it $CONTAINER_NAME \
                /bin/bash -lc 'cd /workspace/aicomp/opt_prime/tutoruslabs && LOGFILE=./results/$(date +%Y%m%d%H%M%S).log; \
                GPULOGFILE=./results/$(date +%Y%m%d%H%M%S)_gpustats.log; MEMLOGFILE=./results/$(date +%Y%m%d%H%M%S)_memstats.log; \
                MODEL_NAME='$MODEL_NAME'; NODE_RANK='$NODE_RANK'; MASTER_ADDR='$MASTER_ADDR'; NNODES='$NNODES'; NPROC_PER_NODE='$NPROC_PER_NODE'; USE_CACHE='$USE_CACHE'; PP_SIZE='$PP_SIZE'; TP_SIZE='$TP_SIZE'; DP_SIZE='$DP_SIZE'; \
                (while true; do echo "===== $(date "+%F %T") ====="; gpustat --no-color || true; echo; sleep 1; done) >> "$GPULOGFILE" 2>&1 & GPUSTAT_PID=$!; \
                (while true; do echo "===== $(date "+%F %T") ====="; free -h || true; echo; sleep 1; done) >> "$MEMLOGFILE" 2>&1 & MEMSTAT_PID=$!; \
                trap "kill $GPUSTAT_PID $MEMSTAT_PID 2>/dev/null || true; wait $GPUSTAT_PID $MEMSTAT_PID 2>/dev/null || true" EXIT INT TERM; \
                bash ./run_rdzv_70b.sh "$MODEL_NAME" "$NODE_RANK" "$MASTER_ADDR" "$NNODES" "$NPROC_PER_NODE" "$USE_CACHE" "$PP_SIZE" "$TP_SIZE" "$DP_SIZE" > "$LOGFILE" 2>&1;'                
