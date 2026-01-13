#!/bin/bash

echo "=========================================="
echo "      Training Node Cleanup Script"
echo "=========================================="

# 1. Find and kill GPU processes found by nvidia-smi (PID based)
echo "[INFO] Killing processes found by nvidia-smi..."
# Query nvidia-smi to extract PIDs (excluding headers)
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

if [ -z "$PIDS" ]; then
    echo "[INFO] No GPU processes found by nvidia-smi."
else
    # If PIDs are found, kill all processes with kill -9
    echo "[INFO] Found PIDs: $PIDS"
    echo "$PIDS" | xargs -r sudo kill -9
    echo "[INFO] Killed successfully."
fi
echo ""

# 2. Find and kill python processes
echo "[INFO] Killing all python processes..."
PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep python | awk -F',' '{print $1}' | tr -d ' ')

if [ -z "$PIDS" ]; then
    echo "[INFO] No python processes found."
else
    echo "[INFO] Killing the following python processes:"
    echo "$PIDS"
    for PID in $PIDS; do
        kill -9 $PID 2>/dev/null $ echo "[OK] Process with PID: $PID killed."
        echo "[INFO] Killed process with PID: $PID"
    done
fi

# 3. Find and kill processes occupying all GPU devices (/dev/nvidiaX) (fuser)
echo "[INFO] Cleaning up processes on all /dev/nvidia devices..."
# Automatically detect the number of GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)

if [ -z "$GPU_COUNT" ]; then
    echo "[INFO] Nvidia driver not detected or no GPUs found."
else
    # Loop from 0 to GPU_COUNT-1
    for (( i=0; i<GPU_COUNT; i++ ))
    do
        echo "[INFO] Checking /dev/nvidia$i..."
        # -k: kill, -v: verbose, -s: silent(minimize messages)
        # 2>/dev/null: Error messages from processes that don't exist are suppressed
        sudo fuser -k -v /dev/nvidia$i 2>/dev/null
    done
    echo "[INFO] Done cleaning /dev/nvidia devices."
fi
echo "=========================================="
nvidia-smi

# 4. Kill processes occupying port 29500
TARGET_PORT=29500
echo "[INFO] Killing process on port $TARGET_PORT..."
# Find PIDs using lsof (-t: PID only, -i: check port)
PORT_PID=$(sudo lsof -t -i:$TARGET_PORT)

if [ -z "$PORT_PID" ]; then
    echo "[INFO] Port $TARGET_PORT is free."
else
    echo "[INFO] Found PID $PORT_PID on port $TARGET_PORT. Killing..."
    sudo kill -9 $PORT_PID
    echo "[INFO] Killed."
fi
echo ""

# 5. Check the status of shared memory (/dev/shm)
echo "[INFO] Checking Shared Memory (/dev/shm)..."
df -h /dev/shm
ls -lh /dev/shm/torch_* 2>/dev/null

echo "[INFO] If you remove torch_* files, do it manually."
echo "[INFO] sudo rm -rf /dev/shm/torch_*"
echo ""

echo "=========================================="
echo "           Cleanup Complete"
echo "=========================================="

echo "[INFO] If you want to kill all processes, run the following command:"
echo "[INFO] sudo pkill -9 python"
echo "[INFO] sudo pkill -9 torch"
echo "[INFO] sudo pkill -9 nccl"
echo "[INFO] sudo pkill -9 rdma"
echo "[INFO] sudo pkill -9 mpi"