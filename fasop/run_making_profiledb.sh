#!/bin/bash
# in docker container

# for TP in 1 2 4; do
#     torchrun --nproc_per_node=$TP --nnodes=1 --node_rank=0 \
#              --master_port=29500 \
#              /workspace/aicomp/compiler_fx/llama_2d_local4fasop_prof.py {HUGGINGFACE_TOKEN} | tee "/workspace/aicomp/fasop/output.out"
#     python3 /workspace/aicomp/fasop/_06_get_layer_time.py | grep mean | awk '{print $2}' | tee tp$TP-fasop.txt
# done

{
    read layer_mean_1
    read emb_mean_1
    read post_mean_1
} < tp1-fasop.txt

{
    read layer_mean_2
    read emb_mean_2
    read post_mean_2
} < tp2-fasop.txt

{
    read layer_mean_4
    read emb_mean_4
    read post_mean_4
} < tp4-fasop.txt

# SURFIX="llama-3.2-1B-$(date +%Y-%m-%d)"
python3 /workspace/aicomp/fasop/profile.py --model_name llama \
                                           --gpu_type "A40" \
                                           --transformer_type de \
                                           --decoder_embedding_time $emb_mean_1 $emb_mean_2 $emb_mean_4 \
                                           --decoder_time $layer_mean_1 $layer_mean_2 $layer_mean_4 \
                                           --decoder_post_process_time $post_mean_1 $post_mean_2 $post_mean_4 \
                                           --de_layer_num 16

