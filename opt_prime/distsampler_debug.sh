docker exec optimus-prime bash -lc 'cd /workspace/aicomp/opt_prime && PP=2; TP=1; DP=4; MBS=1; GBS=32;  \
EXT="_distsampler-debug_MBS${MBS}_PP${PP}_TP${TP}_DP${DP}_GBS${GBS}";
LOG_DIR=./_logs; DATE=$(date +%Y%m%d_%H%M%S); mkdir -p $LOG_DIR; OUT_LOG=${LOG_DIR}/${DATE}${EXT}_log.txt; \
GPU_LOG=${LOG_DIR}/${DATE}_gpustat${EXT}.log; (while true; do echo "===== $(date "+%F %T") ====="; gpustat --no-color || true; echo; sleep 1; done) >> "$GPU_LOG" 2>&1 & GPUSTAT_PID=$!; \
trap "kill $GPUSTAT_PID 2>/dev/null || true; wait $GPUSTAT_PID 2>/dev/null || true" EXIT INT TERM;  echo "Run nsys capture DATE=$DATE"; \
torchrun --standalone --nproc_per_node=8 --nnodes=1 --master_port=29500 examples/pp_train_llama4.py --access-token ${HF_ACCESS_TOKEN} --pp-degree $PP --tp-degree $TP --dp-degree $DP \
--micro-batch-size $MBS --batch-size $GBS \
  --debug-dataset True --debug-dataset-k 1 \
  --debug-batch True --debug-batch-steps 1 \
  --debug-batch-raw True --debug-batch-raw-k 2 \
  --profile-cut True --profile-step 1 > "$OUT_LOG" 2>&1; \
echo "done"'