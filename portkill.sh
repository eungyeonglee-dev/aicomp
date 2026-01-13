#!/bin/bash

# Kill processes occupying port 29500
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