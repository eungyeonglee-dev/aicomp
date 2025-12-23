cd /home/ieg95/workspace && docker exec optimus-timelog bash -lc 'cd /workspace/aicomp/opt_prime 
&& LOG_DIR=./_logs; DATE=$(date +%Y%m%d_%H%M%S); mkdir -p $LOG_DIR; 
echo "Running quick log-only check DATE=$DATE"; 
OPTPRIME_FX_LAYER_TIMING=1 OPTPRIME_FX_LAYER_START_STEP=51 OPTPRIME_FX_LAYER_END_STEP=60 
torchrun --standalone --nproc_per_node=8 --nnodes=1 --master_port=29500 examples/pp_train_llama.py --pp-degree 2 
--tp-degree 1 --dp-degree 4 --micro-batch-size 8 --batch-size 32 
--profile-mode 0 --profile-cut True --profile-step 60 --log-level 0 
--layer-timing True --layer-timing-mode fx --layer-timing-start-step 51 --layer-timing-end-step 60 --emit-nvtx False --nsys-capture False 
> ${LOG_DIR}/${DATE}_component_log_check.txt 2>&1; echo "done"; grep -n "=== FX Component fwd timing" -A5 ${LOG_DIR}/${DATE}_component_log_check.txt | head -n 60'