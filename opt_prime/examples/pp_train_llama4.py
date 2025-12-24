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
import argparse
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

def print0(message):
    if int(os.environ['RANK']) == 0:
        print(message)

#
# This program needs 'access token' for Llama. First, obtain your access token for Llama !!!
#
if len(sys.argv) > 1:
    os.environ['HF_ACCESS_TOKEN'] = sys.argv[1]
access_token = os.getenv('HF_ACCESS_TOKEN')
if access_token is None:
    raise ValueError("HF_ACCESS_TOKEN environment variable is not set."
                    "       [Usage:] torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <llama_access_token>")
#
# Parse arguments
#
parser = argparse.ArgumentParser()
parser.description = 'Llama training with pipeline parallelism using Optimus-p'
parser.add_argument('--access-token', type=str, default=access_token, help='access token for Llama from Hugging Face')
parser.add_argument('--pp-degree', type=int, default=4, help='pipeline parallelism degree')
parser.add_argument('--tp-degree', type=int, default=1, help='tensor parallelism degree')
parser.add_argument('--dp-degree', type=int, default=2, help='data parallelism degree')
parser.add_argument('--micro-batch-size', type=int, default=16, help='micro batch size')
parser.add_argument('--batch-size', type=int, default=32, help='global batch size')
parser.add_argument('--profile-cut', type=bool, default=False, help='"False": normal training, "True": profile only a few steps')
parser.add_argument('--profile-step', type=int, default=20, help='the number of steps to profile')
parser.add_argument('--log-level', type=int, default=0, help='log-level')
parser.add_argument('--pipeline-parallel-schedule', type=str, default="1f1b", help='"1f1b" or "gpipe"')

args = parser.parse_args()

#
# Extract arguments
#
pp_degree = args.pp_degree
tp_degree = args.tp_degree
dp_degree = args.dp_degree
micro_batch_size = args.micro_batch_size
batch_size = args.batch_size # 32 = dp(2) * micro_batch_size(1) * #iterations(16)
profile_cut = args.profile_cut
profile_step = args.profile_step
gas = int(batch_size / (micro_batch_size * dp_degree)) # gas means gradient accumulation steps. It equals the number of micro-batches in one batch.
rank=int(os.environ['RANK'])
world_size=int(os.environ['WORLD_SIZE'])
pp_schedule = args.pipeline_parallel_schedule
ir_analyze = IR_Anal.SEQUENTIAL
epochs = 1

#
# This program needs torch version 2.3.1 or higher !!!
#
required_version = "2.3.1"
current_version = torch.__version__

if version.parse(current_version) >= version.parse(required_version):
    print0(f"[rank:{int(os.environ['RANK'])}] torch version 2.3.1 or higher --> OK")
else:
    print0(f"[rank:{int(os.environ['RANK'])}] current torch version is {current_version}.")
    raise ValueError('This program needs torch version 2.3.1 or higher.')

#
# This program needs transformers version 4.46.2 or higher !!!
#
required_tf_version = "4.46.2"
current_tf_version = transformers.__version__

if version.parse(current_tf_version) >= version.parse(required_tf_version):
    print0(f"[rank:{int(os.environ['RANK'])}] transformers version 4.46.2 or higher --> OK")
else:
    print0(f"[rank:{int(os.environ['RANK'])}] current transformers version is {current_tf_version}.")
    raise ValueError('This program needs transformers version 4.46.2 or higher.')

# Tokenizer setting
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model parameters
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token, use_cache=False)
def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

# Print model information
print0('> Total parameters in model: {:,}'.format(get_total_params(model)))
print0('> World size: {}'.format(world_size)) # world size
print0('> IR: {}'.format(ir_analyze)) # IR analysis mode
print0('> GAS: {}'.format(gas)) # gradient accumulation steps(equals the number of micro-batch)
print0('> GBS: {}'.format(batch_size))
print0('> MBS: {}'.format(micro_batch_size)) # micro batch size
print0('> TP: {}'.format(tp_degree)) # tensor parallelism degree
print0('> DP: {}'.format(dp_degree)) # data parallelism degree
print0('> PP: {}'.format(pp_degree)) # pipeline parallelism degree

# Optimus-p instance and setting
optimus_p = Optimus_p(model, gas, 
                      use_gpu=True, activation_ckpt=False, force_free_mem=True, display_mem=True, 
                      swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze=IR_Anal.SEQUENTIAL,
                      pp_size=pp_degree, tp_size=tp_degree, dp_size=dp_degree)
print0(f" rank={optimus_p.get_rank()} ...")
optimus_p.train()
# Optimizer setting
optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)
# Scheduler setting
scheduler = torch.optim.lr_scheduler.StepLR(optimus_p.optimizer, 1.0, gamma=0.95)
# Dataset setting
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
dataloader = optimus_p.prepare_dataloader(datasets, batch_size)
data_size=len(dataloader.dataset)
nbatches = len(dataloader)
print0(f"data_size={data_size}")
print0(f"nbatches={nbatches}")

def train():

    optimus_p.train() # turn on the train mode

    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        print0(f"===step start {i+1}===")
        data, labels = None, None

        # prepare input and label
        print0(f"===> prepare input and label")
        if optimus_p.is_first_stage():
            print0(f"===> tokenizer")
            tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
            data, labels = tokens.input_ids, tokens.input_ids
        print0(f"===> move labels to last stage")
        labels = optimus_p.move_labels2last_stage(labels)

        optimus_p.optimizer.zero_grad()

        #optimus_p.run(data, labels)
        #optimus_p.run(data, labels, mode="gpipe")
        print0(f"===> run")
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
            log_interval = 1
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, nbatches, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))

                total_loss = 0
                start_time = time.time()
        print0(f"===step end {i+1}===")
        if profile_cut and i == profile_step:
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

