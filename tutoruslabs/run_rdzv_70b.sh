#!/bin/bash

export NCCL_DEBUG=ERROR
# 피어가 죽으면 다른 랭크도 통신 에러로 즉시 터지도록
export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_BLOCKING_WAIT=0
#export TORCH_DIST_INIT_BARRIER=1

unset NCCL_BLOCKING_WAIT
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=18000

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

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
  echo "Usage: $0 <MODEL_NAME> <LLAMA_TOKEN> <NODE_RANK> <MASTER_ADDR> [NNODES] [NPROC_PER_NODE]"
  echo "Example: $0 meta-llama/Llama-3.1-8B-Instruct hf_xxxxx 0 10.0.0.11 8 8"
  exit 1
fi

MODEL_NAME="$1"
LLAMA_TOKEN="$2"
NODE_RANK="${3}"
MASTER_ADDR="${4}"
NNODES="${5:-8}"
NPROC_PER_NODE="${6:-8}"

WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
MAX_TIME=18000

BATCH_SIZES=(64 32 16)
MICRO_BATCH_SIZES=(1 4 8 16 32 64)

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
MODEL_FILENAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)

# 마스터 전용 결과 파일
RESULT_FILEPATH="$RESULT_DIR/${MODEL_FILENAME}.csv"
if [ "$NODE_RANK" -eq 0 ] && [ ! -f "$RESULT_FILEPATH" ]; then
  echo "batch_size,micro_batch_size,pp_size,tp_size,dp_size,training_time(sec)" > "$RESULT_FILEPATH"
fi

status_from_exit() {
  case "$1" in
    0)  echo "" ;;                 # 성공 시엔 숫자 시간 기록이 들어감
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

# PP/TP/DP 조합 생성
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
# 정렬: PP desc, TP desc, DP desc
mapfile -t COMBINATIONS < <(
  printf '%s\n' "${COMBINATIONS[@]}" | sort -k1,1nr -k2,2nr -k3,3nr
)

echo "======== Generated PP/TP/DP combinations ========"
for COMBO in "${COMBINATIONS[@]}"; do
  read PP TP DP <<<"$COMBO"
  echo "PP=$PP, TP=$TP, DP=$DP"
done
echo "================================================="

COUNTER=0

# 모델 학습
for BATCH in "${BATCH_SIZES[@]}"; do
  for MICRO_BATCH in "${MICRO_BATCH_SIZES[@]}"; do

    if [ $MICRO_BATCH -ge $BATCH ]; then
      echo ">>> Skip: batch=$BATCH, micro_batch=$MICRO_BATCH (MICRO_BATCH >= BATCH)"
      continue
    fi

    NUM_MB=$(( BATCH / MICRO_BATCH ))

    for COMBO in "${COMBINATIONS[@]}"; do
      read PP TP DP <<<"$COMBO"

      # Deadlock 조건만 샘플링: num_mb < pp_size 인 경우만 실험
      if [ "$NUM_MB" -ge "$PP" ]; then
        echo ">>> Skip: batch=$BATCH, micro_batch=$MICRO_BATCH, PP=$PP (num_mb=${NUM_MB} >= PP → 1F1B 가드 조건 아님.)"
        continue
      fi

      RUN_ID="${MODEL_FILENAME}-${BATCH}-${MICRO_BATCH}-${PP}-${TP}-${DP}"
      
      COUNTER=$((COUNTER+1))
      RDZV_PORT=$((29500 + (COUNTER % 200)))

      RDZV_TIMEOUT=18000

      echo "================================================="
      echo "RUN_ID            : $RUN_ID"
      echo "Model             : $MODEL_NAME"
      echo "Batch/Micro       : $BATCH / $MICRO_BATCH"
      echo "PP/TP/DP          : $PP / $TP / $DP"
      echo "Nodes x GPUs/node : $NNODES x $NPROC_PER_NODE (WORLD_SIZE=$WORLD_SIZE)"
      echo "RDZV              : c10d ${MASTER_ADDR}:${RDZV_PORT} (timeout=${RDZV_TIMEOUT}s)"
      echo "================================================="

      ROLE=$([ "$NODE_RANK" -eq 0 ] && echo "master" || echo "worker")
      echo "[$ROLE] RUN_ID=$RUN_ID  batch/micro=$BATCH/$MICRO_BATCH  PP/TP/DP=$PP/$TP/$DP"

      if [ "$ROLE" = "worker" ]; then
        echo "Waiting for master rendezvous (${MASTER_ADDR}:${RDZV_PORT})..."
        while ! nc -z "$MASTER_ADDR" "$RDZV_PORT"; do
          sleep 3
          echo "Still waiting for master..."
        done
      fi

      SECONDS=0 # for fallback
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
          --run_id "$RUN_ID"
      EXIT_CODE=$?
      if [ "$EXIT_CODE" -eq 124 ]; then
        EXIT_CODE=50
      fi

      sleep 1

      # tmp/exitcode_<RUN_ID>.txt 에서 EXIT_CODE와 elapsed_time(sec) 같이 읽기
      EXIT_LOG=$(ls tmp/exitcode_${RUN_ID}.txt 2>/dev/null | tail -n 1)
      ELAPSED_SEC=""
      if [ -f "$EXIT_LOG" ]; then
        EXIT_LINE=$(cat "$EXIT_LOG")
        # "0,123.456" 형식이면 EXIT_CODE, ELAPSED_SEC 분리
        if [[ "$EXIT_LINE" == *,* ]]; then
          EXIT_CODE="${EXIT_LINE%%,*}"
          ELAPSED_SEC="${EXIT_LINE##*,}"
        else
          # "10" 같이 코드만 있을 때
          EXIT_CODE="$EXIT_LINE"
        fi
      fi

      # 혹시 성공인데 ELAPSED_SEC가 비어 있으면 SECONDS로 fallback
      if [ "$EXIT_CODE" -eq 0 ] && [ -z "$ELAPSED_SEC" ]; then
        ELAPSED_SEC=$SECONDS
      fi

      pkill -9 -f "torchrun" || true
      pkill -9 -f "pp_train_llama.py" || true
      pkill -9 -f "python" || true
      sleep 2
      fuser -v /dev/nvidia* -k 2>/dev/null || true

      if [ "$NODE_RANK" -eq 0 ]; then
        if [ $EXIT_CODE -eq 0 ]; then
          # 성공: 경과시간 숫자 기록
          echo "${BATCH},${MICRO_BATCH},${PP},${TP},${DP},${ELAPSED_SEC}" >> "$RESULT_FILEPATH"
          echo "SUCCESS → recorded ${ELAPSED_SEC}s"
          #echo "--- END ---"
          #break 3
        else
          # 실패: 상태 문자열 기록
          STATUS_STR=$(status_from_exit "$EXIT_CODE")
          echo "${BATCH},${MICRO_BATCH},${PP},${TP},${DP},${STATUS_STR}" >> "$RESULT_FILEPATH"
          echo "FAILED (exit=$EXIT_CODE) → recorded '${STATUS_STR}'"
        fi

        # 중복 제거(헤더 유지)
        #( head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -u ) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"
      fi

      sleep 5
    done
  done
done

# 결과 파일 정렬
#if [ "$NODE_RANK" -eq 0 ]; then
#  (head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -t',' -k1,1n -k2,2n -k3,3n -k4,4n -k5,5n) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"
#  echo ">> Master wrote results to: $RESULT_FILEPATH"
#fi
