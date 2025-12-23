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
TRYNUMBER=3
HF_ACCESS_TOKEN={your_huggingface_token_here}
MBS=1
TP=1
DP=1
PP=8
GBS=32
PROFILE_MODE=0 # 0: no profile, 1: nvtx, 2: torch profiler
PROFILE_CUT="True" # whether to cut off after profiling steps
PROFILE_STEP=60 # profiling steps

TRAIN_ARGS="--access-token $HF_ACCESS_TOKEN
            --pp-degree $PP
            --tp-degree $TP
            --dp-degree $DP
            --micro-batch-size $MBS
            --batch-size $GBS
            --profile-mode $PROFILE_MODE
            --profile-cut $PROFILE_CUT
            --profile-step $PROFILE_STEP"

for i in $(seq 1 $TRYNUMBER)
do
    echo "=== Training Attempt: $i ==="
    START_TIME=$(date +%s)
    torchrun --standalone \
        --nproc_per_node=8 --nnodes=1 --node_rank=0 \
        --master_port=$MASTER_PORT \
        examples/pp_train_llama4.py $TRAIN_ARGS > "${LOG_DIR}/${DATE}_log_llama4_${i}.txt" 2>&1 &&
    ./get_time.sh "${LOG_DIR}/${DATE}_log_llama4_${i}.txt" > "${LOG_DIR}/${DATE}_steptime_log_llama4_${i}.txt" 2>&1 &&

    # status=$?
    # if [ $status -ne 0 ]; then
    #     echo "Training Attempt: $i failed with status $status"
    #     FAILED=true
    #     mv "${LOG_DIR}/${DATE}_log_llama4_${i}.txt" "${LOG_DIR}/${DATE}_log_llama4_${i}_failed.txt"
    #     break
    # fi
    END_TIME=$(date +%s)
    STEPTIME=$((END_TIME - START_TIME))
    echo "=== Training Attempt: $i Completed in $STEPTIME seconds ==="
done

# # If any attempt failed, set FAILED to true
# if [[ $FAILED == true ]]; then
#     echo "=== Some training attempts failed. Please check the logs in $LOG_DIR ==="
#     exit 1
# fi

files=(
    "${LOG_DIR}/${DATE}_log_llama4_1_avg.txt"
    "${LOG_DIR}/${DATE}_log_llama4_2_avg.txt"
    "${LOG_DIR}/${DATE}_log_llama4_3_avg.txt"
)

echo "=== Collecting Average Steptime Results ===" > "${LOG_DIR}/${DATE}_log_llama4_overall_avg.txt"
sum=0
count=0
for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        { read line1;} < "$file"
        line1=$(python -c "print(f'{float(${line1}/1000):.3f}')")
        echo $line1 >> "${LOG_DIR}/${DATE}_log_llama4_overall_avg.txt"
        sum=$(python -c "print(${sum} + ${line1})")
        count=$((count + 1))
    else
        echo "File $file does not exist."
    fi
done

if [[ $count -gt 0 ]]; then
    avg=$(python -c "print(f'{float(${sum} / ${count}):.3f}')")
    echo $avg >> "${LOG_DIR}/${DATE}_log_llama4_overall_avg.txt"
else
    echo "No valid average files found to compute overall average."
fi

# echo "=== Average Steptime over $TRYNUMBER attempts: $avg ms/batch ===" >> "${LOG_DIR}/${DATE}_log_llama4_overall_avg.txt"
