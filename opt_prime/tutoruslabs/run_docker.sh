#!/bin/bash

CONTAINER_NAME="etri_test_container"
CONTAINER_IMAGE="etri_test_image:latest"
CONTAINER_WORKSPACE_DIR="/workspace/aicomp"

# Required args
# if [ $# -lt 4 ]; then
#   echo "Usage: $0 <MODEL_NAME> <NODE_RANK> <MASTER_ADDR> [NNODES] [NPROC_PER_NODE] [USE_CACHE]"
#   exit 1
# fi

# if [ "$(sudo docker ps -q -f name=$CONTAINER_NAME)" ]; then
#   sudo docker stop $CONTAINER_NAME
#   sudo docker rm $CONTAINER_NAME
# fi

# sudo docker run --gpus all -i -t --name $CONTAINER_NAME \
#                 -v ${HOME}/workspace/aicomp:$CONTAINER_WORKSPACE_DIR \
#                 --ipc=host \
#                 --network=host \
#                 -w $CONTAINER_WORKSPACE_DIR \
#                 -e LLAMA_ACCESS_TOKEN=$LLAMA_ACCESS_TOKEN \
#                 $CONTAINER_IMAGE \
#                 bash -lc "tail -f /dev/null"


# sudo docker exec -it $CONTAINER_NAME \
                # /bin/bash -lc 'cd /workspace/aicomp/opt_prime/opt_prime/tutoruslabs && LOGFILE=./results/$(date +%Y%m%d%H%M%S).log; GPULOGFILE=./results/$(date +%Y%m%d%H%M%S)_gpustats.log; (while true; do echo "===== $(date "+%F %T") ====="; gpustat --no-color || true; echo; sleep 1; done) >> "$GPULOGFILE" 2>&1 & GPUSTAT_PID=$!; trap "kill $GPUSTAT_PID 2>/dev/null || true; wait $GPUSTAT_PID 2>/dev/null || true" EXIT INT TERM; bash ./run_rdzv_70b.sh $1 $2 $3 $4 $5 $6 > "$LOGFILE" 2>&1;' 

MODEL_SIZE=$1
NODE_RANK=$2
MASTER_ADDR=$3
NNODES=$4
NPROC_PER_NODE=$5
USE_CACHE=$6

# PP/TP/DP 값을 명령줄 인자 또는 환경변수로 받기
# 명령줄 인자: $7=PP, $8=TP, $9=DP
# 환경변수: PP_SIZE, TP_SIZE, DP_SIZE
PP_SIZE=$7
TP_SIZE=$8
DP_SIZE=$9

MBS=$10
GBS=$11

if [ $MODEL_SIZE -eq 70 ]; then
    MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
fi

sudo docker exec -it $CONTAINER_NAME \
                /bin/bash -lc 'cd /workspace/aicomp/opt_prime/tutoruslabs && LOGFILE=./results/$(date +%Y%m%d%H%M%S).log; \
                GPULOGFILE=./results/$(date +%Y%m%d%H%M%S)_gpustats.log; MEMLOGFILE=./results/$(date +%Y%m%d%H%M%S)_memstats.log; \
                MODEL_NAME='$MODEL_NAME'; NODE_RANK='$NODE_RANK'; MASTER_ADDR='$MASTER_ADDR'; NNODES='$NNODES'; NPROC_PER_NODE='$NPROC_PER_NODE'; USE_CACHE='$USE_CACHE'; PP_SIZE='$PP_SIZE'; TP_SIZE='$TP_SIZE'; DP_SIZE='$DP_SIZE'; \
                (while true; do echo "===== $(date "+%F %T") ====="; gpustat --no-color || true; echo; sleep 1; done) >> "$GPULOGFILE" 2>&1 & GPUSTAT_PID=$!; \
                (while true; do echo "===== $(date "+%F %T") ====="; free -h || true; echo; sleep 1; done) >> "$MEMLOGFILE" 2>&1 & MEMSTAT_PID=$!; \
                trap "kill $GPUSTAT_PID $MEMSTAT_PID 2>/dev/null || true; wait $GPUSTAT_PID $MEMSTAT_PID 2>/dev/null || true" EXIT INT TERM; \
                bash ./run_rdzv_70b.sh "$MODEL_NAME" "$NODE_RANK" "$MASTER_ADDR" "$NNODES" "$NPROC_PER_NODE" "$USE_CACHE" "$PP_SIZE" "$TP_SIZE" "$DP_SIZE" "$MBS" "$GBS" > "$LOGFILE" 2>&1;'                
