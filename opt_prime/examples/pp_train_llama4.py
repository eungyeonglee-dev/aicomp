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
import hashlib
from packaging import version

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal

logging.basicConfig(level=logging.ERROR)


# This program needs 'access token' for Llama. First, obtain your access token for Llama !!!

if len(sys.argv) > 1:
    os.environ['HF_ACCESS_TOKEN'] = sys.argv[1]

access_token = os.getenv('HF_ACCESS_TOKEN')
if access_token is None:
    raise ValueError("HF_ACCESS_TOKEN environment variable is not set."
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
parser.add_argument('--debug-dataset', type=bool, default=False, help='Print proof that each rank holds the same dataset list in memory')
parser.add_argument('--debug-dataset-k', type=int, default=3, help='How many dataset samples to print per rank for --debug-dataset')
parser.add_argument('--debug-dataset-hash-k', type=int, default=200, help='How many dataset samples to hash per rank for --debug-dataset')
parser.add_argument('--debug-batch', type=bool, default=False, help='Print proof of per-rank batch size and effective global batch (stage0 only)')
parser.add_argument('--debug-batch-steps', type=int, default=1, help='How many steps to print for --debug-batch (starting from step 0)')
parser.add_argument('--debug-batch-raw', type=bool, default=False, help='Print proof of raw batch CONTENTS returned by DataLoader after sharding (shows dp sharding)')
parser.add_argument('--debug-batch-raw-k', type=int, default=2, help='How many raw samples from the batch to print per rank for --debug-batch-raw')

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
debug_dataset = args.debug_dataset
debug_dataset_k = args.debug_dataset_k
debug_dataset_hash_k = args.debug_dataset_hash_k
debug_batch = args.debug_batch
debug_batch_steps = args.debug_batch_steps
debug_batch_raw = args.debug_batch_raw
debug_batch_raw_k = args.debug_batch_raw_k
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

optimus_p = Optimus_p(model, num_mb=gas, use_gpu=True, \
             pp_size=pp_degree , dp_size=dp_degree, tp_size=tp_degree, \
             activation_ckpt=False, \
             force_free_mem=True, display_mem=False, \
             swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze=ir_analyze)

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

# ---------------------------------------------------------------------------
# Debug/proof: show that each process holds the same full dataset list in memory
# (this is normal in many training scripts; DP sharding is done by the sampler).
# ---------------------------------------------------------------------------
def _dataset_digest(items):
    h = hashlib.sha1()
    for s in items:
        # normalize to bytes; include separator to avoid ambiguity
        h.update(s.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()

if debug_dataset and dist.is_initialized():
    k = max(0, int(debug_dataset_k))
    hk = max(0, int(debug_dataset_hash_k))
    snap = datasets[:k]
    digest = _dataset_digest(datasets[:hk])
    payload = {
        "rank": rank,
        "world_size": world_size,
        "len_dataset": len(datasets),
        "hash_first_k": hk,
        "sha1": digest,
        "samples": snap,
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)
    if rank == 0:
        print("===== DEBUG_DATASET: per-rank dataset snapshot proof =====")
        for p in gathered:
            print(
                f"[rank:{p['rank']}] len={p['len_dataset']} sha1(first {p['hash_first_k']})={p['sha1']}"
            )
            for i, s in enumerate(p["samples"]):
                print(f"  sample[{i}]: {repr(s)[:200]}")
        print("===== /DEBUG_DATASET =====")

epochs = 1 # The number of epochs
def train():
    optimus_p.train() # turn on the train mode

    total_loss = 0
    start_time = time.time()    
    debug_batch_printed = 0
    debug_batch_raw_printed = 0
    for i, batch in enumerate(dataloader):
        data, labels = None, None

        # -------------------------------------------------------------------
        # Debug/proof: per-rank batch size and effective global batch
        # - local per-rank batch = len(batch) from DataLoader
        # - only first pipeline stage uses batch to build input_ids
        #   so effective global batch used = sum local_bs over stage 0 ranks.
        # -------------------------------------------------------------------
        if debug_batch and debug_batch_printed < int(debug_batch_steps) and dist.is_initialized():
            local_bs = len(batch) if hasattr(batch, "__len__") else None
            payload = {
                "rank": rank,
                "stage": getattr(optimus_p.tpl, "stage", None),
                "is_first_stage": bool(optimus_p.is_first_stage()),
                "local_bs": local_bs,
            }
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, payload)
            if rank == 0:
                first_stage_ranks = set(optimus_p.tpl.stage2rank[0])
                eff_global_bs = 0
                stage_sums = {}
                for p in gathered:
                    st = p["stage"]
                    lbs = p["local_bs"] if p["local_bs"] is not None else 0
                    stage_sums[st] = stage_sums.get(st, 0) + lbs
                    if p["rank"] in first_stage_ranks and p["local_bs"] is not None:
                        eff_global_bs += p["local_bs"]

                print("===== DEBUG_BATCH: per-rank batch size proof =====")
                print(f"[step:{i}] expected GBS(arg --batch-size)={batch_size} dp={dp_degree} pp={pp_degree} tp={tp_degree}")
                print(f"[step:{i}] effective_global_batch_used_by_stage0(sum over stage0 ranks)={eff_global_bs}")
                print(f"[step:{i}] per-stage sum of len(batch) (note: non-stage0 ranks iterate the loader but do not use it)={stage_sums}")
                for p in gathered:
                    print(
                        f"[rank:{p['rank']}] stage={p['stage']} first_stage={p['is_first_stage']} local_bs(len(batch))={p['local_bs']}"
                    )
                print("===== /DEBUG_BATCH =====")
            debug_batch_printed += 1

        # -------------------------------------------------------------------
        # Debug/proof: raw batch CONTENTS after prepare_dataloader()+sampler sharding
        # This is what you want to "see with eyes":
        #  - batch is a list[str] returned by the DataLoader iterator
        #  - if DP sharding works, different dp ranks (within the same PP stage)
        #    will typically see different strings at step 0.
        #
        # NOTE: In this codebase, *all* ranks iterate the DataLoader, even non-first PP stages.
        # Those ranks will also show batches, but they are not used for tokenization.
        # If stage0 and stage1 have the same dp_rank mapping, they can show identical batches.
        # -------------------------------------------------------------------
        if debug_batch_raw and debug_batch_raw_printed < int(debug_batch_steps) and dist.is_initialized():
            k = max(0, int(debug_batch_raw_k))
            raw_samples = [str(x) for x in batch[:k]]
            raw_digest = _dataset_digest([str(x) for x in batch])
            payload = {
                "rank": rank,
                "stage": getattr(optimus_p.tpl, "stage", None),
                "is_first_stage": bool(optimus_p.is_first_stage()),
                "local_bs": len(batch) if hasattr(batch, "__len__") else None,
                "sha1_batch": raw_digest,
                "samples": raw_samples,
            }
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, payload)
            if rank == 0:
                print("===== DEBUG_BATCH_RAW: raw batch contents after sampler =====")
                print(f"[step:{i}] expected GBS={batch_size} dp={dp_degree} pp={pp_degree} tp={tp_degree}")
                for p in gathered:
                    print(
                        f"[rank:{p['rank']}] stage={p['stage']} first_stage={p['is_first_stage']} "
                        f"local_bs={p['local_bs']} sha1(batch)={p['sha1_batch']}"
                    )
                    for j, s in enumerate(p["samples"]):
                        print(f"  sample[{j}]: {repr(s)[:200]}")
                print("===== /DEBUG_BATCH_RAW =====")
            debug_batch_raw_printed += 1
        # prepare input and label
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
            
        if tp_degree > 1:
            pass
        else:
            torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5) # if tp > 1, don't use this line
        
        optimus_p.optimizer.step()

        if optimus_p.is_last_stage():
            loss = sum(loss) / optimus_p.num_mb
            total_loss += loss
            log_interval = 1
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                if optimus_p.get_rank() % int(world_size/pp_degree) == 0:
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                        'lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, i, nbatches, scheduler.get_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss, math.exp(cur_loss)))

                total_loss = 0
                start_time = time.time()
        if profile_cut and i > profile_step:
            break

for epoch in range(1, epochs + 1):
    train()
    scheduler.step()

print(f"[rank:{optimus_rank}, run completed ...")
