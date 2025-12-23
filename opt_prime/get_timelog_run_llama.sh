# !/bin/bash

# NCCL debug settings
# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1 # Disable InfiniBand
# export NCCL_P2P_LEVEL=NVL # Use NVLink for P2P communication
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export OMP_NUM_THREADS=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

LOG_DIR="./_logs"
mkdir -p $LOG_DIR
# DATE="20251029_055110"
# DATE=$(date +%Y%m%d_%H%M%S)
DATE=$1
FAILED=false
TRYNUMBER=1
MBS=1
TP=1
DP=4
PP=2
GBS=32
PROFILE_MODE=0 # 0: no profile, 1: nvtx, 2: torch profiler
PROFILE_CUT="True" # whether to cut off after profiling steps
PROFILE_STEP=60 # profiling steps
LOG_LEVEL=2

TRAIN_ARGS="--access-token $HF_ACCESS_TOKEN
            --pp-degree $PP
            --tp-degree $TP
            --dp-degree $DP
            --micro-batch-size $MBS
            --batch-size $GBS
            --profile-mode $PROFILE_MODE
            --profile-cut $PROFILE_CUT
            --profile-step $PROFILE_STEP
            --log-level $LOG_LEVEL"

echo "=== Logging llama model training"
START_TIME=$(date +%s)
PROFILE_CMD=""

if [ "$PROFILE_MODE" -eq 1 ]; then
    PROFILE_CMD="nsys profile -t cuda,nvtx -c nvtx --capture-range-end=stop -f true --delay=50 --duration=20 --stats=true -o "nsys-${DATE}""
else
    PROFILE_CMD=""
fi


$PROFILE_CMD torchrun --standalone \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_port=$MASTER_PORT \
    examples/pp_train_llama4.py $TRAIN_ARGS > "${LOG_DIR}/${DATE}_log_llama4.txt" 2>&1

END_TIME=$(date +%s)
STEPTIME=$((END_TIME - START_TIME))
echo "=== Training Attempt: $i Completed in $STEPTIME seconds ==="


