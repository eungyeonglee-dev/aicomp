#!/bin/bash

# run_benchmark.sh
cd /workspace/aicomp/opt_prime || exit

LOG_DIR="./_logs"
mkdir -p "$LOG_DIR"

# PP DP TP 구성
# CONFIGS=("8 1 1" "4 2 1" "4 1 2" "2 4 1" "2 2 2" "2 1 4")
CONFIGS=("2 1 4")

for CONF in "${CONFIGS[@]}"; do
    read -r PP DP TP <<< "$CONF"
    MBS=1
    GBS=$DP
    DATE=$(date +%Y%m%d_%H%M%S)
    OUT_LOG="${LOG_DIR}/${DATE}_MBS${MBS}_PP${PP}_TP${TP}_DP${DP}_GAS1_send-recv_log.txt"

    echo "Running: PP=$PP DP=$DP TP=$TP..."

    torchrun --standalone --nproc_per_node=8 --nnodes=1 --master_port=29500 \
        examples/pp_train_llama4.py \
        --access-token "${HF_ACCESS_TOKEN}" \
        --pp-degree "$PP" \
        --tp-degree "$TP" \
        --dp-degree "$DP" \
        --micro-batch-size "$MBS" \
        --batch-size "$GBS" \
        --pipeline-parallel-schedule 1f1b \
        --profile-cut True \
        --profile-step 20 > "$OUT_LOG" 2>&1
done