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
import datetime
import logging
import argparse
import os
import sys
import math
import time
import re
import nvtx
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from packaging import version

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal
from opt_prime import get_timers

logging.basicConfig(level=logging.ERROR)


# This program needs 'access token' for Llama. First, obtain your access token for Llama !!!

if len(sys.argv) > 1:
    os.environ['LLAMA_ACCESS_TOKEN'] = sys.argv[1]

access_token = os.getenv('LLAMA_ACCESS_TOKEN')
if access_token is None:
    raise ValueError("LLAMA_ACCESS_TOKEN environment variable is not set."
                    "       [Usage:] torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <hf_access_token>")

# Parse arguments
parser = argparse.ArgumentParser()
parser.description = 'Llama training with pipeline parallelism using Optimus-p'
parser.add_argument('--access-token', type=str, default=access_token, help='access token for Llama from Hugging Face')
parser.add_argument('--pp-degree', type=int, default=4, help='pipeline parallelism degree')
parser.add_argument('--tp-degree', type=int, default=1, help='tensor parallelism degree')
parser.add_argument('--dp-degree', type=int, default=2, help='data parallelism degree')
parser.add_argument('--micro-batch-size', type=int, default=16, help='micro batch size')
parser.add_argument('--batch-size', type=int, default=32, help='global batch size')
parser.add_argument('--profile-mode', type=str, default="0", help='"0": normal, "1": nvtx, "2": pytorch profiler')
parser.add_argument('--profile-cut', type=bool, default=False, help='"False": normal training, "True": profile only a few steps')
parser.add_argument('--profile-step', type=int, default=20, help='the number of steps to profile')
parser.add_argument('--log-level', type=int, default=0, help='log-level')
parser.add_argument('--pipeline-parallel-schedule', type=str, default="1f1b", help='"1f1b" or "gpipe"')

args = parser.parse_args()

# Extract arguments
pp_degree = args.pp_degree
tp_degree = args.tp_degree
dp_degree = args.dp_degree
micro_batch_size = args.micro_batch_size
batch_size = args.batch_size # 32 = dp(2) * micro_batch_size(1) * #iterations(16)
gas = int(batch_size / (micro_batch_size * dp_degree)) # gas means gradient accumulation steps. It equals the number of micro-batches in one batch.
profile_mode = args.profile_mode
profile_cut = args.profile_cut
profile_step = args.profile_step
rank=int(os.environ['RANK'])
world_size=int(os.environ['WORLD_SIZE'])
pp_schedule = args.pipeline_parallel_schedule

# This program needs torch version 2.3.1 or higher !!!

required_version = "2.3.1"
current_version = torch.__version__

if version.parse(current_version) >= version.parse(required_version):
    print(f"[rank:{rank}] torch version 2.3.1 or higher --> OK")
else:
    print(f"[rank:{rank}] current torch version is {current_version}.")
    raise ValueError('This program needs torch version 2.3.1 or higher.')

# This program needs transformers version 4.46.2 or higher !!!

required_tf_version = "4.46.2"
current_tf_version = transformers.__version__

if version.parse(current_tf_version) >= version.parse(required_tf_version):
    print(f"[rank:{rank}] transformers version 4.46.2 or higher --> OK")
else:
    print(f"[rank:{rank}] current transformers version is {current_tf_version}.")
    raise ValueError('This program needs transformers version 4.46.2 or higher.')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token, use_cache=False)

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

def print0(message):
    if rank == 0:
        print(message)

ir_analyze = IR_Anal.SEQUENTIAL
print0('> Total parameters in model: {:,}'.format(get_total_params(model)))
print0('> World size: {}'.format(world_size)) # world size
print0('> IR: {}'.format(ir_analyze)) # IR analysis mode
print0('> GAS: {}'.format(gas)) # gradient accumulation steps(equals the number of micro-batch)
print0('> GBS: {}'.format(batch_size))
print0('> MBS: {}'.format(micro_batch_size)) # micro batch size
print0('> TP: {}'.format(tp_degree)) # tensor parallelism degree
print0('> DP: {}'.format(dp_degree)) # data parallelism degree
print0('> PP: {}'.format(pp_degree)) # pipeline parallelism degree

timers = get_timers(log_level=args.log_level)

optimus_p = Optimus_p(model, mbsize=gas, use_gpu=True, \
             pp_size=pp_degree , dp_size=dp_degree, tp_size=tp_degree, \
             activation_ckpt=False, \
             force_free_mem=True, display_mem=False, \
             swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze=ir_analyze, \
             timers=timers, profile_mode=profile_mode)

optimus_rank = optimus_p.get_rank()
print(f" rank={optimus_rank} ...")

optimus_p.train()

if tp_degree > 1:
    # When using tensor parallelism, it is recommended to use AdamW optimizer from transformers
    optimus_p.optimizer = transformers.AdamW(optimus_p.parameters(), lr=3e-5, foreach=False)
else:
    optimus_p.optimizer = transformers.AdamW(optimus_p.parameters(), lr=3e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimus_p.optimizer, 1.0, gamma=0.95)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
# dataloader = DataLoader(datasets, batch_size=batch_size)
dataloader = optimus_p.prepare_dataloader(datasets, batch_size)
data_size=len(dataloader.dataset)
print0(f"data_size={data_size}")
nbatches = len(dataloader)
print0(f"nbatches={nbatches}")

epochs = 1 # The number of epochs
def train():
    timers_to_log = [
    'tokenizer',
    'IR-preparing',
    'prepare-labels',
    'free-memory',
    'forward-backward']
    a_timers_to_log = []
    optimus_p.train() # turn on the train mode
    total_loss = 0
    start_time = time.time()
    # timers('training-time',log_level=1).start()
    # print0('> training started')
    if profile_mode == "1":
        for i, batch in enumerate(dataloader):
            with torch.cuda.nvtx.range(f"rank-{optimus_p.get_rank()}_batch-{i}"):
                data, labels = None, None
                # prepare input and label(only in the first stage)
                with torch.cuda.nvtx.range("prep.tokenizer"):
                    if optimus_p.is_first_stage():
                        tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
                        data, labels = tokens.input_ids, tokens.input_ids
                # move data to the last stage
                with torch.cuda.nvtx.range("comm.prepare_labels"):
                    labels = optimus_p.move_labels2last_stage(labels)
                # fw/bw
                with torch.cuda.nvtx.range("fwdbwd"):
                    optimus_p.optimizer.zero_grad()
                    with torch.cuda.nvtx.range("optimus_p.run"):
                        optimus_p.run(data, labels, mode=pp_schedule)
            
                # get loss (only in the last stage)
                if optimus_p.is_last_stage():
                    with torch.cuda.nvtx.range("loss.get_loss"):
                        loss = optimus_p.get_loss() 
                else:
                    loss = None
                # optimizer
                with torch.cuda.nvtx.range("optim.step"):
                    if tp_degree > 1:
                        pass
                    else:
                        torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
                    optimus_p.optimizer.step()
            
                # logging
                if optimus_p.is_last_stage():
                    loss = sum(loss) / optimus_p.mbsize
                    total_loss += loss
                    log_interval = 1
                    if i % log_interval == 0 and i > 0:
                        cur_loss = total_loss / log_interval
                        elapsed = time.time() - start_time
                        print('[RANK {:1d}] | epoch {:3d} | {:5d}/{:5d} batches | '
                            'lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(int(optimus_p.get_rank()),
                                epoch, i, nbatches, scheduler.get_lr()[0],
                                elapsed * 1000 / log_interval,
                                cur_loss, math.exp(cur_loss)))
                        total_loss = 0
                        start_time = time.time()
            if profile_cut and i > profile_step:
                break
    elif profile_mode == "0":
        timers_to_log = timers_to_log + ['dp-gradient-allreduce', 'optimizer-step']
        for i, batch in enumerate(dataloader):
            data, labels = None, None
            # prepare input and label
            if optimus_p.is_first_stage():
                # print0('> tokenizer started')
                timers('tokenizer', log_level=1).start()
                tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
                data, labels = tokens.input_ids, tokens.input_ids
                timers('tokenizer').stop()
            
            # print0('> move_labels2last_stage started')
            labels = optimus_p.move_labels2last_stage(labels)
            
            optimus_p.optimizer.zero_grad()

            # print0('> run started')
            optimus_p.run(data, labels, mode=pp_schedule)

            if optimus_p.is_last_stage():
                loss = optimus_p.get_loss() 
            else:
                loss = None
                
            if tp_degree > 1:
                pass
            else:
                torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5) # if tp > 1, don't use this line
            
            # print0('> optimizer-step started')
            timers('optimizer-step',log_level=1).start()
            optimus_p.optimizer.step()
            timers('optimizer-step').stop()

            if optimus_p.is_last_stage():
                loss = sum(loss) / optimus_p.mbsize
                total_loss += loss
                log_interval = 1
                if i % log_interval == 0 and i > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    # print(f'[RANK {optimus_p.get_rank()}] world_size={world_size}, pp_degree={pp_degree}')
                    print(f"rank [{optimus_p.get_rank() % int(world_size/pp_degree)}]")
                    if optimus_p.get_rank() % int(world_size/pp_degree) == 0:
                        print('| epoch {:3d} | {:5d}/{:5d} batches | '
                            'lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                                epoch, i, nbatches, scheduler.get_lr()[0],
                                elapsed * 1000 / log_interval,
                                cur_loss, math.exp(cur_loss)))

                    total_loss = 0
                    start_time = time.time()
            
            timers.log(timers_to_log)

            if args.log_level == 2:
                all_timers = timers.get_timer_names()
                # Filter timers that start with prefix and end with a number
                prefixes = ['pre-forward-','forward-recv-', 'forward-','post-forward-','forward-send-', 
                            'pre-backward-','backward-recv-','backward-','post-backward-','backward-send-']
                dynamic_timers = [name for name in all_timers
                                if any(name.startswith(prefix) and 
                                        re.match(r'^' + re.escape(prefix) + r'\d+$', name) 
                                        for prefix in prefixes)]
                
                # print0(f"dynamic_timers: {dynamic_timers}")
                a_timers_to_log = timers_to_log + dynamic_timers + ['dp-gradient-allreduce', 'optimizer-step']
            else:
                pass

            timers.log(a_timers_to_log)
            if profile_cut and i > profile_step:
                break
    elif profile_mode == "2":
        now=datetime.datetime.now()
        date=now.strftime("%Y%m%d%H%M%S")
        prof_schedule=torch.profiler.schedule(wait=1, warmup=2, active=1, repeat=1)
        trace_tb=(tensorboard_trace_handler(f"./_tensorlog/{date}_rank_{optimus_p.get_rank()}") if optimus_p.get_rank() == 0 else None)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=prof_schedule,
                     on_trace_ready=trace_tb,
                     record_shapes=False,
                     profile_memory=False,
                     with_stack=False) as prof:
            for i, batch in enumerate(dataloader):
                data, labels = None, None
                # prepare input and label(only in the first stage)
                if optimus_p.is_first_stage():
                    tokens = tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
                    data, labels = tokens.input_ids, tokens.input_ids
                # move data to the last stage
                labels = optimus_p.move_labels2last_stage(labels)
                # fw/bw
                optimus_p.optimizer.zero_grad()
                optimus_p.run(data, labels, mode=pp_schedule)
                # get loss (only in the last stage)
                if optimus_p.is_last_stage():
                    loss = optimus_p.get_loss() 
                else:
                    loss = None
                # optimizer
                if tp_degree > 1:
                    pass
                else:
                    torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
                optimus_p.optimizer.step()
                # logging
                if optimus_p.is_last_stage():
                    loss = sum(loss) / optimus_p.mbsize
                    total_loss += loss
                    log_interval = 1
                    if i % log_interval == 0 and i > 0:
                        cur_loss = total_loss / log_interval
                        elapsed = time.time() - start_time
                        print('[RANK {:1d}] | epoch {:3d} | {:5d}/{:5d} batches | '
                            'lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(int(optimus_p.get_rank()),
                                epoch, i, nbatches, scheduler.get_lr()[0],
                                elapsed * 1000 / log_interval,
                                cur_loss, math.exp(cur_loss)))
                        total_loss = 0
                        start_time = time.time()
                prof.step()
                if profile_cut and i > profile_step:
                    break
    else:
        assert False, f">> profile mode: {profile_mode}. It is unknown profile mode. Please set profile_mode to '0', '1', or '2'."
    
    
    
    # timers('training-time').stop()
        
for epoch in range(1, epochs + 1):
    train()
    scheduler.step()

print(f"[rank:{optimus_rank}, run completed ...")

