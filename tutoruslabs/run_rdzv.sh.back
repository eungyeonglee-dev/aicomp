#!/bin/bash

############################################
# User params
############################################
# models
# "meta-llama/Llama-3.2-1B"
# "meta-llama/Llama-3.2-3B"
# "meta-llama/Llama-3.1-8B-Instruct"
# "meta-llama/Llama-2-13b-chat-hf"
# "meta-llama/Llama-3.3-70B-Instruct"

# Required args
if [ $# -lt 4 ]; then
  echo "Usage: $0 <MODEL_NAME> <LLAMA_TOKEN> <NODE_RANK> <MASTER_ADDR> [NNODES] [NPROC_PER_NODE] [RDZV_PORT]"
  echo "Example: $0 meta-llama/Llama-3.1-8B-Instruct hf_xxxxx 0 10.0.0.11 8 8 29501"
  exit 1
fi

MODEL_NAME="$1"
LLAMA_TOKEN="$2"
NODE_RANK="${3}"
MASTER_ADDR="${4}"
NNODES="${5:-8}"
NPROC_PER_NODE="${6:-8}"
RDZV_PORT="${7:-29501}"

############################################
# Derived params
############################################
WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))

BATCH_SIZES=(32 64 128 256 512 1024 2048 4096)
MICRO_BATCH_SIZES=(4 8 16 32 64 128 256 512 1024 2048)

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
MODEL_FILENAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)
RESULT_FILEPATH="$RESULT_DIR/${MODEL_FILENAME}.csv"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

############################################
# NCCL / network sanity (optional but helpful)
############################################
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_SOCKET_IFNAME=enp34s0f0
# export GLOO_SOCKET_IFNAME=enp34s0f0
# export NCCL_IB_DISABLE=1

############################################
# Generate PP/TP/DP combinations (PP*TP*DP == WORLD_SIZE)
############################################
COMBINATIONS=()
for ((PP=2; PP<=WORLD_SIZE; PP*=2)); do
  for ((TP=1; TP<=WORLD_SIZE; TP*=2)); do
    for ((DP=1; DP<=WORLD_SIZE; DP*=2)); do
      if [ $((PP * TP * DP)) -eq $WORLD_SIZE ]; then
        COMBINATIONS+=("$PP $TP $DP")
      fi
    done
  done
done

echo "======== Generated PP/TP/DP combinations (WORLD_SIZE=$WORLD_SIZE) ========"
for COMBO in "${COMBINATIONS[@]}"; do
  read PP TP DP <<<"$COMBO"
  echo "PP=$PP, TP=$TP, DP=$DP"
done
echo "=========================================================================="

############################################
# Helpers
############################################
cleanup() {
  echo "[NODE $NODE_RANK] Caught signal, cleaning up..."
  pkill -P $$ || true
}
trap cleanup INT TERM

wait_master_tcp() {
  local host="$1" port="$2"
  if ping -c 1 -W 1 "$host" >/dev/null 2>&1; then
    echo "[NODE $NODE_RANK] master host reachable: $host"
  else
    echo "[NODE $NODE_RANK] WARN: cannot ping $host (will rely on elastic retry)"
  fi
}

dedup_csv_if_exists() {
  if [ -f "$RESULT_FILEPATH" ]; then
    ( head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -u ) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"
    echo ">>> Deduped: $RESULT_FILEPATH"
  fi
}

# OOM 감지: 흔한 에러 문구들 매칭
is_oom() {
  local file="$1"
  grep -qiE \
    "out of memory|CUDA.*out of memory|CUBLAS_STATUS_ALLOC_FAILED|cudnn.*alloc|std::bad_alloc|terminate called after throwing an instance of 'std::bad_alloc'|RuntimeError:.*(out of memory|OOM)" \
    "$file"
}

############################################
# Main loop
############################################
wait_master_tcp "$MASTER_ADDR" "$RDZV_PORT"

MAX_RETRY=5
RDZV_TIMEOUT=900   # seconds

for COMBO in "${COMBINATIONS[@]}"; do
  read PP TP DP <<<"$COMBO"

  for BATCH in "${BATCH_SIZES[@]}"; do
    for MICRO_BATCH in "${MICRO_BATCH_SIZES[@]}"; do

      if [ $MICRO_BATCH -ge $BATCH ]; then
        echo ">>> Skip: batch=$BATCH, micro_batch=$MICRO_BATCH (MICRO>=BATCH)"
        continue
      fi

      RUN_ID_BASE="${MODEL_FILENAME}-${BATCH}-${MICRO_BATCH}-${PP}-${TP}-${DP}"
      echo "================================================="
      echo "RUN_ID            : $RUN_ID_BASE"
      echo "Model             : $MODEL_NAME"
      echo "Batch/Micro       : $BATCH / $MICRO_BATCH"
      echo "PP/TP/DP          : $PP / $TP / $DP"
      echo "Nodes x GPUs/node : $NNODES x $NPROC_PER_NODE (WORLD_SIZE=$WORLD_SIZE)"
      echo "RDZV              : c10d ${MASTER_ADDR}:${RDZV_PORT} (timeout=${RDZV_TIMEOUT}s)"
      echo "================================================="

      attempt=1
      aborted_on_oom=0

      while [ $attempt -le $MAX_RETRY ]; do
        RUN_ID="${RUN_ID_BASE}-try${attempt}"
        LOG_FILE="$LOG_DIR/${RUN_ID}.log"

        echo "[RUN_ID=$RUN_ID][NODE $NODE_RANK] Attempt $attempt/$MAX_RETRY"

        # 약간의 랜덤 백오프로 초기 충돌 감소
        sleep $((RANDOM % 3))

        # torchrun 출력 로그 저장, 실제 종료코드는 PIPESTATUS[0]로 획득
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE-1))) torchrun \
          --nproc_per_node="${NPROC_PER_NODE}" \
          --nnodes="${NNODES}" \
          --node_rank="${NODE_RANK}" \
          --master_addr="${MASTER_ADDR}" \
          --master_port=29500 \
          #--rdzv_backend=c10d \
          #--rdzv_endpoint="${MASTER_ADDR}:${RDZV_PORT}" \
          #--rdzv_id="${RUN_ID}" \
          #--rdzv_conf "timeout=${RDZV_TIMEOUT}" \
          pp_train_llama.py \
            --llama_access_token "$LLAMA_TOKEN" \
            --model_name "$MODEL_NAME" \
            --batch_size $BATCH \
            --micro_batch_size $MICRO_BATCH \
            --pp_size $PP \
            --tp_size $TP \
            --dp_size $DP \
          2>&1 | tee "$LOG_FILE"
        exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
          echo "[RUN_ID=$RUN_ID] SUCCESS"
          break
        else
          # OOM이면 재시도하지 않고 다음 스텝으로 즉시 이동
          if is_oom "$LOG_FILE"; then
            echo "[RUN_ID=$RUN_ID] DETECTED OOM. Skipping retries and moving to next step."
            aborted_on_oom=1
            break
          fi
          echo "[RUN_ID=$RUN_ID] FAIL(exit=$exit_code). Backoff & retry..."
          sleep $((RANDOM % 10 + 5))
        fi

        attempt=$((attempt+1))
      done

      if [ $aborted_on_oom -eq 1 ]; then
        echo "[RUN_ID=$RUN_ID] OOM encountered -> proceed to next configuration."
      elif [ $attempt -gt $MAX_RETRY ]; then
        echo "[RUN_ID=$RUN_ID] GAVE UP after $MAX_RETRY attempts."
      fi

      echo ">>> Done: $RUN_ID"
      dedup_csv_if_exists
      sleep 5
    done
  done
done

# 최종 정렬 (있을 때만)
if [ -f "$RESULT_FILEPATH" ]; then
  (head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -t',' -k1,1n -k2,2n) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"
  echo ">>> Final sort: $RESULT_FILEPATH"
fi

echo "=== ALL DONE (NODE $NODE_RANK) ==="
