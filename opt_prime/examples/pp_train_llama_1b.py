#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
# Usage: torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> 
#                 --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama4.py <llama_access_token>
#
# *** This program was tested with torch 2.5.0 and transformers 4.46.2.
#     The version of transformers used must be consistent across all machines used for testing ***
#


import torch
import torch.nn as nn


import torch.distributed as dist
import argparse
import datetime
import logging
import os
import sys
import math
import time
from packaging import version

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal

logging.basicConfig(level=logging.ERROR)

# Parse arguments
parser = argparse.ArgumentParser()
parser.description = 'Llama training with pipeline parallelism using Optimus-p (with per-layer timing)'
parser.add_argument('--access-token', type=str, default=None, help='access token for Llama from Hugging Face')
parser.add_argument('--pp-degree', type=int, default=4, help='pipeline parallelism degree')
parser.add_argument('--tp-degree', type=int, default=1, help='tensor parallelism degree')
parser.add_argument('--dp-degree', type=int, default=2, help='data parallelism degree')
parser.add_argument('--micro-batch-size', type=int, default=16, help='micro batch size')
parser.add_argument('--batch-size', type=int, default=32, help='global batch size')
parser.add_argument('--pipeline-parallel-schedule', type=str, default="1f1b", help='"1f1b" or "gpipe"')
parser.add_argument('--profile_start_step', type=int, default=51, help='step to start profiling')
parser.add_argument('--profile_end_step', type=int, default=60, help='step to end profiling')
parser.add_argument('--nsys-capture', type=bool, default=True, help='Call torch.cuda.profiler start/stop for steps [layer-timing-start-step, layer-timing-end-step] so nsys can capture only that window with --capture-range=cudaProfilerApi.')
args = parser.parse_args()

def print0(message):
    if int(os.environ["RANK"]) == 0:
        print(message)


#
# This program needs 'access token' for Llama. First, obtain your access token for Llama !!!
#
access_token=args.access_token
if access_token is None:
    access_token = os.getenv('HF_ACCESS_TOKEN')

if access_token is None:
    raise ValueError("HF_ACCESS_TOKEN environment variable is not set."
                    "       [Usage:] torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <llama_access_token>")


#
# This program needs torch version 2.3.1 or higher !!!
#
required_version = "2.3.1"
current_version = torch.__version__

if version.parse(current_version) >= version.parse(required_version):
    print(f"[rank:{int(os.environ['RANK'])}] torch version 2.3.1 or higher --> OK")
else:
    print(f"[rank:{int(os.environ['RANK'])}] current torch version is {current_version}.")
    raise ValueError('This program needs torch version 2.3.1 or higher.')

#
# This program needs transformers version 4.46.2 or higher !!!
#
required_tf_version = "4.46.2"
current_tf_version = transformers.__version__

if version.parse(current_tf_version) >= version.parse(required_tf_version):
    print(f"[rank:{int(os.environ['RANK'])}] transformers version 4.46.2 or higher --> OK")
else:
    print(f"[rank:{int(os.environ['RANK'])}] current transformers version is {current_tf_version}.")
    raise ValueError('This program needs transformers version 4.46.2 or higher.')


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token, use_cache=False)

# Extract arguments
pp_degree = args.pp_degree
tp_degree = args.tp_degree
dp_degree = args.dp_degree
micro_batch_size = args.micro_batch_size
batch_size = args.batch_size
pp_schedule = args.pipeline_parallel_schedule
profile_start_step = args.profile_start_step
profile_end_step = args.profile_end_step
nsys_capture = args.nsys_capture
if profile_start_step > profile_end_step:
    print(f"[profile] start_step {profile_start_step} > end_step {profile_end_step}, disabling profiling.")
    profile_start_step = 0
    profile_end_step = 0
    nsys_capture = False
gas = int(batch_size / (micro_batch_size * dp_degree))
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

batch_size = batch_size
num_mb = gas

print0('> Total parameters in model: {:,}'.format(get_total_params(model)))
print0('> World size: {}'.format(world_size))
print0('> GAS: {}'.format(gas))
print0('> GBS: {}'.format(batch_size))
print0('> MBS: {}'.format(micro_batch_size))
print0('> TP: {}'.format(tp_degree))
print0('> DP: {}'.format(dp_degree))
print0('> PP: {}'.format(pp_degree))


optimus_p = Optimus_p(model, num_mb, use_gpu=True, 
                      activation_ckpt=False, force_free_mem=True, display_mem=True, 
                      swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze=IR_Anal.SEQUENTIAL,
                      pp_size=pp_degree, tp_size=tp_degree, dp_size=dp_degree)

print(f"> rank={rank}, optimus_p.rank={optimus_p.get_rank()}")

optimus_p.train()

optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimus_p.optimizer, 1.0, gamma=0.95)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=4)
# dataloader = optimus_p.prepare_dataloader(datasets, batch_size)
data_size=len(dataloader.dataset)
nbatches = len(dataloader)

print0(f"> data_size={data_size}")
print0(f"> nbatches={nbatches}")


epochs = 1 # The number of epochs

def train():

    optimus_p.train() # turn on the train mode

    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        # 1. Start profiling with nsys
        if i == profile_start_step and nsys_capture:
            torch.cuda.profiler.start()
        
        # 2. NVTX mark
        if i >= profile_start_step and i <= profile_end_step and nsys_capture:
            torch.cuda.nvtx.range_push(f"step_{i}")
        
        data, labels = None, None

        # prepare input and label
        if optimus_p.is_first_stage():
            tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
            data, labels = tokens.input_ids, tokens.input_ids

        labels = optimus_p.move_labels2last_stage(labels)

        optimus_p.optimizer.zero_grad()

        #optimus_p.run(data, labels)
        #optimus_p.run(data, labels, mode="gpipe")
        optimus_p.run(data, labels, mode=pp_schedule)

        if optimus_p.is_last_stage():
            loss = optimus_p.get_loss() 
        else:
            loss = None

        torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
        optimus_p.optimizer.step()

        if optimus_p.is_last_stage():
            loss = sum(loss) / optimus_p.num_mb
            total_loss += loss
            log_interval = 10
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('[rank:{}] | epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        optimus_p.get_rank(), epoch, i, nbatches, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))

                total_loss = 0
                start_time = time.time()
        
        if nsys_capture:
            torch.cuda.synchronize()
            # 3. Stop profiling with nsys
            if i >= profile_start_step and i < profile_end_step:
                torch.cuda.nvtx.range_pop()

            # 4. NVTX mark
            if i == profile_end_step:
                torch.cuda.profiler.stop()
                print(f"> rank={rank}, step_{i} completed")
                break

if optimus_p.get_rank() == 0:
    tick = time.time()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    scheduler.step()

if optimus_p.get_rank() == 0:
    tock = time.time()
    elapsed_time = tock - tick

    print('Time elapsed: %.3f sec ' % (elapsed_time))

print(f"[rank:{optimus_p.get_rank()}, run completed ...")

