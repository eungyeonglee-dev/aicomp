#!/bin/bash

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