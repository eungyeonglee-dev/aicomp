#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#
# Usage: torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank>
#                 --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <llama_access_token>
#
# This version adds per-layer forward/backward timing that does not depend on log level.
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
from packaging import version

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import load_dataset
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal
from opt_prime import get_timers

logging.basicConfig(level=logging.ERROR)


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


def attach_layer_timers(model: AutoModelForCausalLM, enable: bool = True):
    """Attach CUDA-event timers to each Llama decoder layer to measure per-layer forward/backward."""
    if not enable:
        return None, []
    if not torch.cuda.is_available():
        print("Layer timing disabled: CUDA is not available.")
        return None, []

    decoder_layers = getattr(model, "model", model).layers
    num_layers = len(decoder_layers)
    stats = {
        "forward": [0.0 for _ in range(num_layers)],
        "backward": [0.0 for _ in range(num_layers)],
        "forward_count": [0 for _ in range(num_layers)],
        "backward_count": [0 for _ in range(num_layers)],
    }
    handles = []

    def make_fwd_pre(idx):
        def hook(module, inputs):
            module._lt_fwd_start = torch.cuda.Event(enable_timing=True)
            module._lt_fwd_start.record()
        return hook

    def make_fwd_post(idx):
        def hook(module, inputs, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            if hasattr(module, "_lt_fwd_start"):
                elapsed = module._lt_fwd_start.elapsed_time(end) / 1000.0
                stats["forward"][idx] += elapsed
                stats["forward_count"][idx] += 1
        return hook

    def make_bwd_pre(idx):
        def hook(module, grad_inputs):
            module._lt_bwd_start = torch.cuda.Event(enable_timing=True)
            module._lt_bwd_start.record()
        return hook

    def make_bwd_post(idx):
        def hook(module, grad_inputs, grad_outputs):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            if hasattr(module, "_lt_bwd_start"):
                elapsed = module._lt_bwd_start.elapsed_time(end) / 1000.0
                stats["backward"][idx] += elapsed
                stats["backward_count"][idx] += 1
        return hook

    for idx, layer in enumerate(decoder_layers):
        handles.append(layer.register_forward_pre_hook(make_fwd_pre(idx)))
        handles.append(layer.register_forward_hook(make_fwd_post(idx)))
        handles.append(layer.register_full_backward_pre_hook(make_bwd_pre(idx)))
        handles.append(layer.register_full_backward_hook(make_bwd_post(idx)))

    return stats, handles


def log_layer_timers(stats, rank: int):
    if stats is None:
        return
    if rank != 0:
        return

    print("=== Layer timing (seconds) ===")
    for idx, (ft, bt, fc, bc) in enumerate(zip(
        stats["forward"], stats["backward"], stats["forward_count"], stats["backward_count"]
    )):
        f_avg = ft / fc if fc else 0.0
        b_avg = bt / bc if bc else 0.0
        print(f"layer{idx:02d}: fwd_total={ft:.4f}s avg={f_avg:.4f}s over {fc} calls | "
              f"bwd_total={bt:.4f}s avg={b_avg:.4f}s over {bc} calls")


# Parse arguments
parser = argparse.ArgumentParser()
parser.description = 'Llama training with pipeline parallelism using Optimus-p (with per-layer timing)'
parser.add_argument('--access-token', type=str, default=None, help='access token for Llama from Hugging Face')
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
parser.add_argument('--layer-timing', type=bool, default=True, help='Enable per-layer timing (always collected, independent of log level)')

args = parser.parse_args()

# Access token handling
if len(sys.argv) > 1 and args.access_token is None:
    os.environ['LLAMA_ACCESS_TOKEN'] = sys.argv[1]
if args.access_token is None:
    args.access_token = os.getenv('LLAMA_ACCESS_TOKEN')

access_token = args.access_token
if access_token is None:
    raise ValueError("LLAMA_ACCESS_TOKEN environment variable is not set."
                    "       [Usage:] torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <hf_access_token>")

# Extract arguments
pp_degree = args.pp_degree
tp_degree = args.tp_degree
dp_degree = args.dp_degree
micro_batch_size = args.micro_batch_size
batch_size = args.batch_size
gas = int(batch_size / (micro_batch_size * dp_degree))
profile_mode = args.profile_mode
profile_cut = args.profile_cut
profile_step = args.profile_step
pp_schedule = args.pipeline_parallel_schedule
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# Version checks
required_version = "2.3.1"
current_version = torch.__version__

if version.parse(current_version) >= version.parse(required_version):
    print(f"[rank:{rank}] torch version 2.3.1 or higher --> OK")
else:
    print(f"[rank:{rank}] current torch version is {current_version}.")
    raise ValueError('This program needs torch version 2.3.1 or higher.')

required_tf_version = "4.46.2"
import transformers
current_tf_version = transformers.__version__

if version.parse(current_tf_version) >= version.parse(required_tf_version):
    print(f"[rank:{rank}] transformers version 4.46.2 or higher --> OK")
else:
    print(f"[rank:{rank}] current transformers version is {current_tf_version}.")
    raise ValueError('This program needs transformers version 4.46.2 or higher.')

# Tokenizer / model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token, use_cache=False)

def print0(message):
    if rank == 0:
        print(message)

print0('> Total parameters in model: {:,}'.format(get_total_params(model)))
print0('> World size: {}'.format(world_size))
print0('> GAS: {}'.format(gas))
print0('> GBS: {}'.format(batch_size))
print0('> MBS: {}'.format(micro_batch_size))
print0('> TP: {}'.format(tp_degree))
print0('> DP: {}'.format(dp_degree))
print0('> PP: {}'.format(pp_degree))

# Timers and Optimus
timers = get_timers(log_level=args.log_level)
layer_stats, layer_handles = attach_layer_timers(model, enable=args.layer_timing)

optimus_p = Optimus_p(model, mbsize=gas, use_gpu=True,
             pp_size=pp_degree, dp_size=dp_degree, tp_size=tp_degree,
             activation_ckpt=False,
             force_free_mem=True, display_mem=False,
             swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze=IR_Anal.SEQUENTIAL,
             timers=timers, profile_mode=profile_mode)

optimus_rank = optimus_p.get_rank()
print(f" rank={optimus_rank} ...")

optimus_p.train()

if tp_degree > 1:
    optimus_p.optimizer = transformers.AdamW(optimus_p.parameters(), lr=3e-5, foreach=False)
else:
    optimus_p.optimizer = transformers.AdamW(optimus_p.parameters(), lr=3e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimus_p.optimizer, 1.0, gamma=0.95)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(record) for record in datasets if len(str(record)) < 500]
dataloader = DataLoader(datasets, batch_size=batch_size)
data_size=len(dataloader.dataset)
print0(f"data_size={data_size}")
nbatches = len(dataloader)
print0(f"nbatches={nbatches}")

epochs = 1

def train():
    optimus_p.train()
    total_loss = 0
    start_time = time.time()

    if profile_mode == "0":
        for i, batch in enumerate(dataloader):
            data, labels = None, None
            if optimus_p.is_first_stage():
                tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
                data, labels = tokens.input_ids, tokens.input_ids

            labels = optimus_p.move_labels2last_stage(labels)

            optimus_p.optimizer.zero_grad()
            optimus_p.run(data, labels, mode=pp_schedule)

            if optimus_p.is_last_stage():
                loss = optimus_p.get_loss()
            else:
                loss = None

            if tp_degree <= 1:
                torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)

            optimus_p.optimizer.step()

            if optimus_p.is_last_stage():
                loss = sum(loss) / optimus_p.mbsize
                total_loss += loss
                log_interval = 10
                if i % log_interval == 0 and i > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    print(f'| epoch {epoch:3d} | {i:5d}/{nbatches:5d} batches | '
                        f'lr {scheduler.get_lr()[0]:02.2f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | '
                        f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
                    total_loss = 0
                    start_time = time.time()

            if profile_cut and i > profile_step:
                break
    else:
        # Keep other profile modes unchanged: they are not the focus for layer timing
        print0(f"Profile mode {profile_mode} is not specialized for layer timing; using normal loop.")
        for i, batch in enumerate(dataloader):
            data, labels = None, None
            if optimus_p.is_first_stage():
                tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
                data, labels = tokens.input_ids, tokens.input_ids
            labels = optimus_p.move_labels2last_stage(labels)
            optimus_p.optimizer.zero_grad()
            optimus_p.run(data, labels, mode=pp_schedule)
            if optimus_p.is_last_stage():
                loss = optimus_p.get_loss()
                loss = sum(loss) / optimus_p.mbsize
            if tp_degree <= 1:
                torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
            optimus_p.optimizer.step()
            if profile_cut and i > profile_step:
                break

for epoch in range(1, epochs + 1):
    train()
    scheduler.step()

log_layer_timers(layer_stats, rank)

print(f"[rank:{optimus_rank}, run completed ...")