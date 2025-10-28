# !/bin/bash

# NCCL debug settings
# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1 # Disable InfiniBand
# export NCCL_P2P_LEVEL=NVL # Use NVLink for P2P communication
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export OMP_NUM_THREADS=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

HF_ACCESS_TOKEN=$1
LOG_DIR="./_logs"
mkdir -p $LOG_DIR
DATE=$(date +%Y%m%d_%H%M%S)

# Run the training script with torchrun for 8 GPUs and log output
# get time for batch size 1
# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
#     --master_port=$MASTER_PORT \
#     /workspace/aicomp/compiler_fx/llama_2d_local4fasop_prof.py ${HF_ACCESS_TOKEN} > "${LOG_DIR}/${DATE}_log_swsok_batchsize1.txt" 2>&1
# get steptime
# torchrun --standalone \
#     --nproc_per_node=8 --nnodes=1 --node_rank=0 \
#     --master_port=$MASTER_PORT \
#     examples/pp_train_llama4.py ${HF_ACCESS_TOKEN} > "${LOG_DIR}/${DATE}_log_llama5.txt" 2>&1 &&
#     ./get_time.sh "${LOG_DIR}/${DATE}_log_llama5.txt" 

# nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn \
#              --capture-range=nvtx --capture-range-end=stop \
#              --force-overwrite=true --delay=100 --duration=20 --stats=true \
#              --output=nsys-rank_${DATE} \

# torchrun --standalone \
#     --nproc_per_node=8 --nnodes=1 --node_rank=0 \
#     --master_port=$MASTER_PORT \
#     examples/pp_train_llama4.py ${HF_ACCESS_TOKEN} > "${LOG_DIR}/${DATE}_log_llama4.txt" 2>&1

nsys profile --trace=cuda,nvtx \
             --capture-range=nvtx --capture-range-end=stop \
             --force-overwrite=true --delay=100 --duration=30 --stats=true \
             --output=nsys_${DATE}_PP4DP2TP1 \
torchrun --standalone \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_port=$MASTER_PORT \
    examples/pp_train_llama4.py ${HF_ACCESS_TOKEN} > "${LOG_DIR}/${DATE}_log_llama4.txt" 2>&1


