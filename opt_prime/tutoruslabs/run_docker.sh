#!/bin/bash
#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# Docker-based Training / Layer Profiling Script
#
# Usage:
#   Training Mode:
#     ./run_docker.sh <MODEL_SIZE> <NODE_RANK> <MASTER_ADDR> <NNODES> <NPROC> <USE_CACHE> <PP> <TP> <DP>
#
#   Profile Mode (Hook-based):
#     ./run_docker.sh <MODEL_SIZE> <NODE_RANK> <MASTER_ADDR> <NNODES> <NPROC> <USE_CACHE> <PP> <TP> <DP> profile
#
#   Profile Mode (FX Interpreter-based):
#     ./run_docker.sh <MODEL_SIZE> <NODE_RANK> <MASTER_ADDR> <NNODES> <NPROC> <USE_CACHE> <PP> <TP> <DP> profile_fx
#
#   MODEL_SIZE: 1B, 3B, 70B or full model name (e.g., meta-llama/Llama-3.2-1B)
#
# Examples:
#   ./run_docker.sh 70B 0 127.0.0.1 1 2 True 2 1 1 profile      # Hook-based profile
#   ./run_docker.sh 1B 0 127.0.0.1 1 2 True 2 1 1 profile_fx    # FX Interpreter profile
#

CONTAINER_NAME="etri_test_container"
CONTAINER_IMAGE="etri_test_image:latest"
CONTAINER_WORKSPACE_DIR="/workspace/aicomp"

# ============================================================================
# 1. Remove existing container
# ============================================================================
echo "===> Removing '$CONTAINER_NAME' container."
docker rm -f $CONTAINER_NAME 2>/dev/null || true
sleep 2
echo "===> Container removed."

echo "===> Creating new container '$CONTAINER_NAME'."

# ============================================================================
# 2. Clean up ports
# ============================================================================
echo "===> Checking port range 29500-29509"
for port in $(seq 29500 29509); do
    if lsof -i :$port >/dev/null 2>&1; then
        echo "===> Cleaning up port $port"
        kill -9 $(lsof -t -i :$port) 2>/dev/null || true
        sleep 1
    fi
done

# ============================================================================
# 3. Run container
# ============================================================================
docker run -d --gpus all --name $CONTAINER_NAME \
    -v ${HOME}/workspace/aicomp:$CONTAINER_WORKSPACE_DIR \
    --ipc=host \
    --network=host \
    -w $CONTAINER_WORKSPACE_DIR \
    -e LLAMA_ACCESS_TOKEN=$LLAMA_ACCESS_TOKEN \
    --entrypoint /bin/bash \
    $CONTAINER_IMAGE -c "tail -f /dev/null"
echo "===> Container '$CONTAINER_NAME' created."

# Install gpustat
echo "===> Installing gpustat."
docker exec $CONTAINER_NAME pip install gpustat -q
echo "===> Gpustat installed."

echo "===> Network interfaces:"
docker exec $CONTAINER_NAME hostname -I

# ============================================================================
# 4. Parse arguments (with defaults)
# ============================================================================
MODEL_SIZE=$1
NODE_RANK=$2
MASTER_ADDR=$3
NNODES=$4
NPROC_PER_NODE=$5
USE_CACHE=$6
PP_SIZE=$7
TP_SIZE=$8
DP_SIZE=$9

# Set MODEL_NAME based on MODEL_SIZE
case "$MODEL_SIZE" in
    70|70B|70b)
        MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
        ;;
    1|1B|1b)
        MODEL_NAME="meta-llama/Llama-3.2-1B"
        ;;
    3|3B|3b)
        MODEL_NAME="meta-llama/Llama-3.2-3B"
        ;;
    *)
        # Assume it's a full model name if not a number
        if [[ "$MODEL_SIZE" == *"/"* ]]; then
            MODEL_NAME="$MODEL_SIZE"
        else
            echo "Unknown model size: $MODEL_SIZE"
            echo "Supported: 1B, 3B, 70B or full model name (e.g., meta-llama/Llama-3.2-1B)"
            exit 1
        fi
        ;;
esac

# ============================================================================
# 5. Run model (Training or Profile)
# ============================================================================
echo ""
echo "================================================="
echo " Mode: profile"
echo " Model: $MODEL_NAME"
echo " PP=$PP_SIZE, TP=$TP_SIZE, DP=$DP_SIZE"
echo "================================================="

# Create log file names
TIMESTAMP=$(date +%Y%m%d%H%M%S)
LOGFILE="./results/${TIMESTAMP}.log"
GPULOGFILE="./results/${TIMESTAMP}_gpustats.log"
MEMLOGFILE="./results/${TIMESTAMP}_memstats.log"

echo "===> Log files:"
echo "     LOGFILE: $LOGFILE"
echo "     GPULOGFILE: $GPULOGFILE"
echo "     MEMLOGFILE: $MEMLOGFILE"

echo "===> Running $CONTAINER_NAME"

docker exec -it $CONTAINER_NAME \
    /bin/bash -lc "cd /workspace/aicomp/opt_prime/tutoruslabs && \
    LOGFILE='$LOGFILE'; GPULOGFILE='$GPULOGFILE'; MEMLOGFILE='$MEMLOGFILE'; \
    MODEL_NAME='$MODEL_NAME'; NODE_RANK='$NODE_RANK'; MASTER_ADDR='$MASTER_ADDR'; \
    NNODES='$NNODES'; NPROC_PER_NODE='$NPROC_PER_NODE'; USE_CACHE='$USE_CACHE'; \
    PP_SIZE='$PP_SIZE'; TP_SIZE='$TP_SIZE'; DP_SIZE='$DP_SIZE'; \
    PROFILE_ARG='$PROFILE_ARG'; \
    (while true; do echo \"===== \$(date '+%F %T') =====\"; gpustat --no-color || true; echo; sleep 1; done) >> \"\$GPULOGFILE\" 2>&1 & GPUSTAT_PID=\$!; \
    (while true; do echo \"===== \$(date '+%F %T') =====\"; free -h || true; echo; sleep 1; done) >> \"\$MEMLOGFILE\" 2>&1 & MEMSTAT_PID=\$!; \
    trap \"kill \$GPUSTAT_PID \$MEMSTAT_PID 2>/dev/null || true\" EXIT INT TERM; \
    bash ./run_rdzv_70b.sh \"\$MODEL_NAME\" \"\$NODE_RANK\" \"\$MASTER_ADDR\" \"\$NNODES\" \"\$NPROC_PER_NODE\" \"\$USE_CACHE\" \"\$PP_SIZE\" \"\$TP_SIZE\" \"\$DP_SIZE\" 2>&1 | tee \"\$LOGFILE\""

echo ""
echo "================================================="
echo " Completed!"
echo " Check logs in results/ directory"
echo "================================================="

