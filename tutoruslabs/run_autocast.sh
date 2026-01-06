#!/bin/bash

# models
# "meta-llama/Llama-3.2-1B"
# "meta-llama/Llama-3.2-3B"
# "meta-llama/Llama-3.1-8B-Instruct"
# "meta-llama/Llama-2-13b-chat-hf"

MODEL_NAME="meta-llama/Llama-3.2-1B"
LLAMA_TOKEN=""

BATCH_SIZES=(32 64 128 256 512 1024 2048 4096)
MICRO_BATCH_SIZES=(4 8 16 32 64 128 256 512 1024 2048)
WORLD_SIZE=4

RESULT_DIR="results"
MODEL_FILENAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)
RESULT_FILEPATH="$RESULT_DIR/${MODEL_FILENAME}_autocast.csv"


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

echo "======== Generated PP/TP/DP combinations ========"
for COMBO in "${COMBINATIONS[@]}"; do
  read PP TP DP <<<"$COMBO"
  echo "PP=$PP, TP=$TP, DP=$DP"
done
echo "================================================="

# 모델 학습
for BATCH in "${BATCH_SIZES[@]}"; do
  for MICRO_BATCH in "${MICRO_BATCH_SIZES[@]}"; do

    if [ $MICRO_BATCH -ge $BATCH ]; then
      echo ">>> Skip: batch=$BATCH, micro_batch=$MICRO_BATCH (MICRO_BATCH >= BATCH)"
      continue
    fi

    for COMBO in "${COMBINATIONS[@]}"; do
      read PP TP DP <<<"$COMBO"
      
      echo "================================================="
      echo " Model             : $MODEL_NAME"
      echo " Batch size        : $BATCH"
      echo " Micro batch size  : $MICRO_BATCH"
      echo " PP size           : $PP"
      echo " TP size           : $TP"
      echo " DP size           : $DP"
      echo " World size        : $WORLD_SIZE"
      echo "================================================="

      CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=$WORLD_SIZE \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=29500 \
        pp_train_llama_autocast.py \
          --llama_access_token "$LLAMA_TOKEN" \
          --model_name "$MODEL_NAME" \
          --batch_size $BATCH \
          --micro_batch_size $MICRO_BATCH \
          --pp_size $PP \
          --tp_size $TP \
          --dp_size $DP

      # $? 변수로 종료 상태 코드 확인
      if [ $? -eq 0 ]; then
        echo "SUCCESS: pp_train_llama.py completed successfully."
      else
        echo "FAILED: pp_train_llama.py failed. Exiting script."
      fi

      echo ">>> Done: batch=$BATCH, micro_batch=$MICRO_BATCH, pp=$PP, tp=$TP, dp=$DP"
      echo ""

      # 결과 파일에서 중복 라인 제거 (헤더 유지)
      ( head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -u ) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"
      echo ">>> Duplicate lines removed from $RESULT_FILEPATH"

      sleep 10
    done
  done
done

# 결과 파일 정렬
(head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -t',' -k1,1n -k2,2n) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"