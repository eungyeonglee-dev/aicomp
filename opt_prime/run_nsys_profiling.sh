cd /home/ieg95/workspace && docker exec optimus-timelog bash -lc \
'cd /workspace/aicomp/opt_prime \
&& MBS=1; PP=4; TP=2; DP=1; GBS=32; \
EXT="_MBS${MBS}_PP${PP}_TP${TP}_DP${DP}_GBS${GBS}"; 
LOG_DIR=./_logs; NSYS_DIR=./_nsys_log; DATE=$(date +%Y%m%d_%H%M%S); mkdir -p $LOG_DIR; \
NSYS_OUT=${LOG_DIR}/nsys_${DATE}${EXT}; \
echo "Running nsys DATE=$DATE"; \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop -o ${NSYS_OUT} \
torchrun --standalone --nproc_per_node=8 --nnodes=1 --master_port=29500 \
examples/pp_train_llama4.py --pp-degree $PP --tp-degree $TP --dp-degree $DP --micro-batch-size $MBS --batch-size $GBS \
--profile-cut True --profile-step 3 --profile-start-step 50 > ${LOG_DIR}/${DATE}${EXT}_nsys_log.txt 2>&1; \
echo "nsys finished"; ls -1 ${NSYS_OUT}.nsys-rep ${NSYS_OUT}.qdstrm 2>/dev/null || true; nsys stats --report nvtxsum ${NSYS_OUT}.nsys-rep | head -n 60 || true; tail -n 40 ${LOG_DIR}/${DATE}${EXT}_nsys_log.txt'