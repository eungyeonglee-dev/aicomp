# !/bin/bash

# NCCL debug settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1 # Disable InfiniBand
# export NCCL_P2P_LEVEL=NVL # Use NVLink for P2P communication
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr="host" --master_port=12345 \
    examples/pp_train_llama4.py <hf_token>

