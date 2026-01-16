#!/bin/bash
#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# Distributed Training / Layer Profiling Script for Llama 70B
#
# Usage:
#   Training Mode:
#     ./run_rdzv_70b.sh <MODEL_NAME> <NODE_RANK> <MASTER_ADDR> <NNODES> <NPROC> <USE_CACHE> <PP> <TP> <DP>
#
#   Profile Mode:
#     ./run_rdzv_70b.sh <MODEL_NAME> <NODE_RANK> <MASTER_ADDR> <NNODES> <NPROC> <USE_CACHE> <PP> <TP> <DP> profile
#

# export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_BLOCKING_WAIT
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=18000

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

############################################
# Arguments
############################################
MODEL_NAME="${1:-meta-llama/Llama-3.3-70B-Instruct}"
NODE_RANK="${2:-0}"
MASTER_ADDR="${3:-127.0.0.1}"
NNODES="${4:-1}"
NPROC_PER_NODE="${5:-8}"
USE_CACHE="${6:-True}"
PP_SIZE="${7:-${PP_SIZE:-2}}"
TP_SIZE="${8:-${TP_SIZE:-1}}"
DP_SIZE="${9:-${DP_SIZE:-1}}"

WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
MAX_TIME=18000

# Batch settings
BATCH_SIZES=(128 256 512)
MICRO_BATCH_SIZES=(16 32 64)

# Profile settings (when PROFILE_MODE="profile")
PROFILE_STEPS=10
PROFILE_WARMUP=50
NUM_HIDDEN_LAYERS=8

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
MODEL_FILENAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)

# Result file (master only)
RESULT_FILEPATH="$RESULT_DIR/${MODEL_FILENAME}.csv"
if [ "$NODE_RANK" -eq 0 ] && [ ! -f "$RESULT_FILEPATH" ]; then
  echo "batch_size,micro_batch_size,pp_size,tp_size,dp_size,training_time(sec)" > "$RESULT_FILEPATH"
fi

############################################
# Helper Functions
############################################
status_from_exit() {
  case "$1" in
    0)  echo "" ;;
    10) echo "OOM ERROR" ;;
    20) echo "DIST ERROR" ;;
    30) echo "EXCEPTION" ;;
    40) echo "PEER FAILED" ;;
    41) echo "FINALIZE ERROR" ;;
    50) echo "TIMEOUT" ;;
    60) echo "#MB < PP(1F1B DEADLOCK)" ;;
    *)  echo "FAIL($1)" ;;
  esac
}

############################################
# Validate PP/TP/DP
############################################
COMBINATIONS=()

if [ -n "$PP_SIZE" ] && [ -n "$TP_SIZE" ] && [ -n "$DP_SIZE" ]; then
  if [ $((PP_SIZE * TP_SIZE * DP_SIZE)) -eq $WORLD_SIZE ]; then
    COMBINATIONS+=("$PP_SIZE $TP_SIZE $DP_SIZE")
    echo "Using specified PP/TP/DP: PP=$PP_SIZE, TP=$TP_SIZE, DP=$DP_SIZE"
  else
    echo "ERROR: PP($PP_SIZE) * TP($TP_SIZE) * DP($DP_SIZE) != WORLD_SIZE($WORLD_SIZE)"
    exit 1
  fi
else
  echo "Auto-generating PP/TP/DP combinations..."
  for ((PP=2; PP<=WORLD_SIZE; PP*=2)); do
    for ((TP=1; TP<=WORLD_SIZE; TP*=2)); do
      for ((DP=1; DP<=WORLD_SIZE; DP*=2)); do
        if [ $((PP * TP * DP)) -eq $WORLD_SIZE ]; then
          COMBINATIONS+=("$PP $TP $DP")
        fi
      done
    done
  done
fi

# Sort by PP desc
if [ ${#COMBINATIONS[@]} -gt 1 ]; then
  mapfile -t COMBINATIONS < <(printf '%s\n' "${COMBINATIONS[@]}" | sort -k1,1nr -k2,2nr -k3,3nr)
fi

echo "================================================="
echo " Mode: profile"
echo " Model: $MODEL_NAME"
echo " World Size: $WORLD_SIZE (${NNODES} nodes x ${NPROC_PER_NODE} GPUs)"
echo " PP/TP/DP combinations:"
for COMBO in "${COMBINATIONS[@]}"; do
  read PP TP DP <<<"$COMBO"
  echo "   PP=$PP, TP=$TP, DP=$DP"
done
echo "================================================="

############################################
# Main Loop
############################################
COUNTER=0
NUM_PAIRS=${#BATCH_SIZES[@]}

for ((i=0; i<NUM_PAIRS; i++)); do
  BATCH=${BATCH_SIZES[$i]}
  MICRO_BATCH=${MICRO_BATCH_SIZES[$i]}

    if [ $MICRO_BATCH -gt $BATCH ]; then
      echo ">>> Skip: MICRO_BATCH($MICRO_BATCH) > BATCH($BATCH)"
      continue
    fi

    for COMBO in "${COMBINATIONS[@]}"; do
      read PP TP DP <<<"$COMBO"

      RUN_ID="${MODEL_FILENAME}-${BATCH}-${MICRO_BATCH}-${PP}-${TP}-${DP}"
      
      COUNTER=$((COUNTER+1))
      RDZV_PORT=$((29500 + (COUNTER % 200)))
      RDZV_TIMEOUT=18000

      echo "================================================="
      echo "RUN_ID            : $RUN_ID"
      echo "Mode              : profile"
      echo "Batch/Micro       : $BATCH / $MICRO_BATCH"
      echo "PP/TP/DP          : $PP / $TP / $DP"
      echo "RDZV              : ${MASTER_ADDR}:${RDZV_PORT}"
      echo "================================================="

      ROLE=$([ "$NODE_RANK" -eq 0 ] && echo "master" || echo "worker")

      if [ "$ROLE" = "worker" ]; then
        echo "Waiting for master rendezvous..."
        while ! nc -z "$MASTER_ADDR" "$RDZV_PORT"; do
          sleep 3
          echo "Still waiting..."
        done
      fi

      # Build command args
      PROFILE_ARGS=""
      PROFILE_ARGS="--profile_steps $PROFILE_STEPS --profile_warmup_steps $PROFILE_WARMUP --num_hidden_layers $NUM_HIDDEN_LAYERS"

      SECONDS=0
      timeout ${MAX_TIME}s env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${RDZV_PORT}" \
        --rdzv_id="${RUN_ID}" \
        --rdzv_conf="timeout=${RDZV_TIMEOUT}" \
        --max_restarts=0 \
        pp_train_llama_70b.py \
          --llama_access_token "$LLAMA_TOKEN" \
          --model_name "$MODEL_NAME" \
          --batch_size $BATCH \
          --micro_batch_size $MICRO_BATCH \
          --pp_size $PP \
          --tp_size $TP \
          --dp_size $DP \
          --run_id "$RUN_ID" \
          --use_cache $USE_CACHE \
          $PROFILE_ARGS

      EXIT_CODE=$?
      [ "$EXIT_CODE" -eq 124 ] && EXIT_CODE=50

      sleep 1

      # Read exit code from file
      EXIT_LOG="tmp/exitcode_${RUN_ID}.txt"
      ELAPSED_SEC=""
      if [ -f "$EXIT_LOG" ]; then
        EXIT_LINE=$(cat "$EXIT_LOG")
        if [[ "$EXIT_LINE" == *,* ]]; then
          EXIT_CODE="${EXIT_LINE%%,*}"
          ELAPSED_SEC="${EXIT_LINE##*,}"
        else
          EXIT_CODE="$EXIT_LINE"
        fi
      fi

      [ "$EXIT_CODE" -eq 0 ] && [ -z "$ELAPSED_SEC" ] && ELAPSED_SEC=$SECONDS

      # Cleanup
      pkill -9 -f "torchrun" 2>/dev/null || true
      pkill -9 -f "pp_train_llama" 2>/dev/null || true
      sleep 2
      fuser -v /dev/nvidia* -k 2>/dev/null || true

      # Record result (master only)
      if [ "$NODE_RANK" -eq 0 ]; then
        if [ "$EXIT_CODE" -eq 0 ]; then
          echo "${BATCH},${MICRO_BATCH},${PP},${TP},${DP},${ELAPSED_SEC}" >> "$RESULT_FILEPATH"
          echo "SUCCESS → ${ELAPSED_SEC}s"
        else
          STATUS_STR=$(status_from_exit "$EXIT_CODE")
          echo "${BATCH},${MICRO_BATCH},${PP},${TP},${DP},${STATUS_STR}" >> "$RESULT_FILEPATH"
          echo "FAILED (exit=$EXIT_CODE) → '${STATUS_STR}'"
        fi
      fi

      sleep 5
    done
  done

# for BATCH in "${BATCH_SIZES[@]}"; do
#   for MICRO_BATCH in "${MICRO_BATCH_SIZES[@]}"; do

#     if [ $MICRO_BATCH -gt $BATCH ]; then
#       echo ">>> Skip: MICRO_BATCH($MICRO_BATCH) > BATCH($BATCH)"
#       continue
#     fi

#     for COMBO in "${COMBINATIONS[@]}"; do
#       read PP TP DP <<<"$COMBO"

#       RUN_ID="${MODEL_FILENAME}-${BATCH}-${MICRO_BATCH}-${PP}-${TP}-${DP}"
      
#       COUNTER=$((COUNTER+1))
#       RDZV_PORT=$((29500 + (COUNTER % 200)))
#       RDZV_TIMEOUT=18000

#       echo "================================================="
#       echo "RUN_ID            : $RUN_ID"
#       echo "Mode              : profile"
#       echo "Batch/Micro       : $BATCH / $MICRO_BATCH"
#       echo "PP/TP/DP          : $PP / $TP / $DP"
#       echo "RDZV              : ${MASTER_ADDR}:${RDZV_PORT}"
#       echo "================================================="

#       ROLE=$([ "$NODE_RANK" -eq 0 ] && echo "master" || echo "worker")

#       if [ "$ROLE" = "worker" ]; then
#         echo "Waiting for master rendezvous..."
#         while ! nc -z "$MASTER_ADDR" "$RDZV_PORT"; do
#           sleep 3
#           echo "Still waiting..."
#         done
#       fi

#       # Build command args
#       PROFILE_ARGS=""
#       PROFILE_ARGS="--profile_steps $PROFILE_STEPS --profile_warmup_steps $PROFILE_WARMUP --num_hidden_layers $NUM_HIDDEN_LAYERS"

#       SECONDS=0
#       timeout ${MAX_TIME}s env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#       torchrun \
#         --nproc_per_node=$NPROC_PER_NODE \
#         --nnodes=$NNODES \
#         --node_rank=$NODE_RANK \
#         --rdzv_backend=c10d \
#         --rdzv_endpoint="${MASTER_ADDR}:${RDZV_PORT}" \
#         --rdzv_id="${RUN_ID}" \
#         --rdzv_conf="timeout=${RDZV_TIMEOUT}" \
#         --max_restarts=0 \
#         pp_train_llama_70b.py \
#           --llama_access_token "$LLAMA_TOKEN" \
#           --model_name "$MODEL_NAME" \
#           --batch_size $BATCH \
#           --micro_batch_size $MICRO_BATCH \
#           --pp_size $PP \
#           --tp_size $TP \
#           --dp_size $DP \
#           --run_id "$RUN_ID" \
#           --use_cache $USE_CACHE \
#           $PROFILE_ARGS

#       EXIT_CODE=$?
#       [ "$EXIT_CODE" -eq 124 ] && EXIT_CODE=50

#       sleep 1

#       # Read exit code from file
#       EXIT_LOG="tmp/exitcode_${RUN_ID}.txt"
#       ELAPSED_SEC=""
#       if [ -f "$EXIT_LOG" ]; then
#         EXIT_LINE=$(cat "$EXIT_LOG")
#         if [[ "$EXIT_LINE" == *,* ]]; then
#           EXIT_CODE="${EXIT_LINE%%,*}"
#           ELAPSED_SEC="${EXIT_LINE##*,}"
#         else
#           EXIT_CODE="$EXIT_LINE"
#         fi
#       fi

#       [ "$EXIT_CODE" -eq 0 ] && [ -z "$ELAPSED_SEC" ] && ELAPSED_SEC=$SECONDS

#       # Cleanup
#       pkill -9 -f "torchrun" 2>/dev/null || true
#       pkill -9 -f "pp_train_llama" 2>/dev/null || true
#       sleep 2
#       fuser -v /dev/nvidia* -k 2>/dev/null || true

#       # Record result (master only)
#       if [ "$NODE_RANK" -eq 0 ]; then
#         if [ "$EXIT_CODE" -eq 0 ]; then
#           echo "${BATCH},${MICRO_BATCH},${PP},${TP},${DP},${ELAPSED_SEC}" >> "$RESULT_FILEPATH"
#           echo "SUCCESS → ${ELAPSED_SEC}s"
#         else
#           STATUS_STR=$(status_from_exit "$EXIT_CODE")
#           echo "${BATCH},${MICRO_BATCH},${PP},${TP},${DP},${STATUS_STR}" >> "$RESULT_FILEPATH"
#           echo "FAILED (exit=$EXIT_CODE) → '${STATUS_STR}'"
#         fi
#       fi

#       sleep 5
#     done
#   done
# done

echo ""
echo "================================================="
echo " Completed!"
echo "================================================="
