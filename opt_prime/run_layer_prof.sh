docker exec optimus-prime bash -lc 'cd /workspace/aicomp/opt_prime \
&& LOG_DIR=./_logs; DATE=$(date +%Y%m%d_%H%M%S); mkdir -p $LOG_DIR; \
PP=2; TP=1; DP=4; MBS=1; GBS=32; EXT="_ir-layer-prof_GBS${GBS}_MBS${MBS}_PP${PP}_TP${TP}_DP${DP}";\
OUT_LOG=${LOG_DIR}/${DATE}${EXT}_log.txt; GPU_LOG=${LOG_DIR}/${DATE}_gpustat${EXT}.log; \
(while true; do echo "===== $(date "+%F %T") ====="; gpustat --no-color || true; echo; sleep 1; done) >> "$GPU_LOG" 2>&1 & GPUSTAT_PID=$!; trap "kill $GPUSTAT_PID 2>/dev/null || true; \
wait $GPUSTAT_PID 2>/dev/null || true" EXIT INT TERM;  \
echo "Run ir layer profiling DATE=$DATE"; \
torchrun --standalone --nproc_per_node=8 --nnodes=1 --master_port=29500 examples/pp_train_llama4.py \
--access-token ${HF_ACCESS_TOKEN} --pp-degree $PP --tp-degree $TP --dp-degree $DP --micro-batch-size $MBS --batch-size $GBS \
--pipeline-parallel-schedule 1f1b --profile-cut True --profile-step 20  > "$OUT_LOG" 2>&1; echo "done"'