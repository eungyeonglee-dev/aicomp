#!/bin/bash
#
# Layer profiling script for Llama-3.2-1B with PP=2, TP=1, DP=1
# This configuration matches the tutoruslabs profiling for comparison
#

CONTAINER_NAME="etri_test_container"

# Check if container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "Container $CONTAINER_NAME is not running. Starting it..."
    # Use existing container or create new one
    if docker ps -a | grep -q $CONTAINER_NAME; then
        docker start $CONTAINER_NAME
    else
        echo "Container not found. Please run run_docker.sh first."
        exit 1
    fi
fi

# Configuration
PP=2
TP=1
DP=1
MBS=1
GBS=2
PROFILE_STEP=30

LOG_DIR=./_logs
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR

EXT="_layer-prof_GBS${GBS}_MBS${MBS}_PP${PP}_TP${TP}_DP${DP}"
OUT_LOG=${LOG_DIR}/${DATE}${EXT}_log.txt
GPU_LOG=${LOG_DIR}/${DATE}_gpustat${EXT}.log

echo "========================================"
echo " Layer Profiling: Llama-3.2-1B"
echo " PP=$PP, TP=$TP, DP=$DP"
echo " MBS=$MBS, GBS=$GBS"
echo " Profile Steps: $PROFILE_STEP"
echo " Log: $OUT_LOG"
echo "========================================"

# Run profiling
docker exec $CONTAINER_NAME bash -lc "cd /workspace/aicomp/opt_prime && \
(while true; do echo \"===== \$(date '+%F %T') =====\"; gpustat --no-color || true; echo; sleep 1; done) >> '$GPU_LOG' 2>&1 & GPUSTAT_PID=\$!; \
trap \"kill \$GPUSTAT_PID 2>/dev/null || true\" EXIT INT TERM; \
torchrun --standalone --nproc_per_node=2 --nnodes=1 --master_port=29500 examples/pp_train_llama4.py \
--access-token \${HF_ACCESS_TOKEN} \
--pp-degree $PP --tp-degree $TP --dp-degree $DP \
--micro-batch-size $MBS --batch-size $GBS \
--pipeline-parallel-schedule 1f1b \
--profile-cut True --profile-step $PROFILE_STEP \
2>&1 | tee '$OUT_LOG'"

echo "========================================"
echo " Profiling Complete"
echo " Log saved to: $OUT_LOG"
echo "========================================"

