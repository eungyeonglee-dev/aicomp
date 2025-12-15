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
PROFILE_MODE=0 # 0: no profile, 1: nvtx, 2: torch profiler
PROFILE_CUT="True" # whether to cut off after profiling steps
PROFILE_STEP=60 # profiling steps

# Define experiment configurations: MBS TP DP PP GBS
# Format: "MBS:TP:DP:PP:GBS"
experiment_configs=(
    "2:1:4:2:32"
    "16:1:2:4:32"
    # Add more configurations as needed
    # "MBS:TP:DP:PP:GBS"
)

# Process each experiment configuration
for config_idx in "${!experiment_configs[@]}"; do
    IFS=':' read -r MBS TP DP PP GBS <<< "${experiment_configs[$config_idx]}"
    
    config_name="MBS${MBS}_TP${TP}_DP${DP}_PP${PP}_GBS${GBS}"
    echo "=========================================="
    echo "=== Experiment Config: $config_name ==="
    echo "=========================================="
    
    TRAIN_ARGS="--access-token $HF_ACCESS_TOKEN
                --pp-degree $PP
                --tp-degree $TP
                --dp-degree $DP
                --micro-batch-size $MBS
                --batch-size $GBS
                --profile-mode $PROFILE_MODE
                --profile-cut $PROFILE_CUT
                --profile-step $PROFILE_STEP"
    
    # Run TRYNUMBER times for this configuration
    for i in $(seq 1 $TRYNUMBER)
    do
        echo "=== Training Attempt: $i/$TRYNUMBER (Config: $config_name) ==="
        START_TIME=$(date +%s)
        torchrun --standalone \
            --nproc_per_node=8 --nnodes=1 --node_rank=0 \
            --master_port=$MASTER_PORT \
            examples/pp_train_llama4.py $TRAIN_ARGS > "${LOG_DIR}/${DATE}_${config_name}_log_llama4_${i}.txt" 2>&1 &&
        ./get_time.sh "${LOG_DIR}/${DATE}_${config_name}_log_llama4_${i}.txt" > "${LOG_DIR}/${DATE}_${config_name}_steptime_log_llama4_${i}.txt" 2>&1 &&
        
        # status=$?
        # if [ $status -ne 0 ]; then
        #     echo "Training Attempt: $i failed with status $status"
        #     FAILED=true
        #     mv "${LOG_DIR}/${DATE}_${config_name}_log_llama4_${i}.txt" "${LOG_DIR}/${DATE}_${config_name}_log_llama4_${i}_failed.txt"
        #     break
        # fi
        END_TIME=$(date +%s)
        STEPTIME=$((END_TIME - START_TIME))
        echo "=== Training Attempt: $i Completed in $STEPTIME seconds ==="
    done
    
    # Collect average for this configuration
    files=()
    for i in $(seq 1 $TRYNUMBER); do
        files+=("${LOG_DIR}/${DATE}_${config_name}_log_llama4_${i}_avg.txt")
    done
    
    avg_file="${LOG_DIR}/${DATE}_${config_name}_overall_avg.txt"
    echo "=== Collecting Average Steptime Results for $config_name ===" > "$avg_file"
    sum=0
    count=0
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            { read line1;} < "$file"
            line1=$(python -c "print(f'{float(${line1}/1000):.3f}')")
            echo "$line1" >> "$avg_file"
            sum=$(python -c "print(${sum} + ${line1})")
            count=$((count + 1))
        else
            echo "File $file does not exist." >> "$avg_file"
        fi
    done
    
    if [[ $count -gt 0 ]]; then
        avg=$(python -c "print(f'{float(${sum} / ${count}):.3f}')")
        echo "$avg" >> "$avg_file"
        echo "=== Config $config_name: Average Steptime = $avg seconds ==="
    else
        echo "No valid average files found to compute overall average." >> "$avg_file"
    fi
    
    echo ""
done

# Create summary file with all configurations
summary_file="${LOG_DIR}/${DATE}_all_configs_summary.txt"
echo "=== Summary of All Experiment Configurations ===" > "$summary_file"
echo "" >> "$summary_file"

for config_idx in "${!experiment_configs[@]}"; do
    IFS=':' read -r MBS TP DP PP GBS <<< "${experiment_configs[$config_idx]}"
    config_name="MBS${MBS}_TP${TP}_DP${DP}_PP${PP}_GBS${GBS}"
    avg_file="${LOG_DIR}/${DATE}_${config_name}_overall_avg.txt"
    
    if [[ -f "$avg_file" ]]; then
        # Get the last line (average value)
        avg=$(tail -n 1 "$avg_file")
        echo "Config: $config_name | Average Steptime: $avg seconds" >> "$summary_file"
    fi
done

echo ""
echo "=== All Experiments Completed ==="
echo "=== Summary saved to: $summary_file ==="

# echo "=== Average Steptime over $TRYNUMBER attempts: $avg ms/batch ===" >> "${LOG_DIR}/${DATE}_log_llama4_overall_avg.txt"
