#!/bin/bash

DATE=$1
echo "gpu utils" > ./_logs/${DATE}_gpu_log.txt

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S')" >> ./_logs/${DATE}_gpu_log.txt
    gpustat --no-color >> ./_logs/${DATE}_gpu_log.txt
    echo "====================================================" >> ./_logs/${DATE}_gpu_log.txt
    sleep 1
done