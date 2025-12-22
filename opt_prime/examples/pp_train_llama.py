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
from opt_prime.IR import IR_Anal, fx_layer_timer_set_step, fx_layer_timer_get, wrap_decoder_layers_for_timing
from opt_prime import get_timers

logging.basicConfig(level=logging.ERROR)


def _install_layer_bwd_nvtx_param_hooks(submod: torch.nn.Module, get_step, start_step: int, end_step: int):
    """Install param grad hooks to create per-layer backward NVTX ranges.
    This approximates per-layer backward time by bracketing the period where all params of a layer receive grads.
    """
    if not torch.cuda.is_available():
        return []

    import re
    pat = re.compile(r"model_layers_(\d+)_")

    layer_param_names = {}
    for name, p in submod.named_parameters(recurse=True):
        m = pat.search(name)
        if m:
            idx = int(m.group(1))
            layer_param_names.setdefault(idx, []).append(name)

    # Per-layer counters (auto-reset) so each microbatch backward gets its own push/pop.
    state = {}
    for idx, names in layer_param_names.items():
        state[idx] = {"count": len(names), "remaining": len(names), "started": False}

    handles = []
    for name, p in submod.named_parameters(recurse=True):
        m = pat.search(name)
        if not m:
            continue
        idx = int(m.group(1))
        st = state[idx]

        def make_hook(layer_idx, st_ref):
            def hook(grad):
                step = get_step()
                if not (start_step <= step <= end_step):
                    return grad
                # Start range at the first grad of this layer in a microbatch.
                if st_ref["remaining"] == st_ref["count"] and not st_ref["started"]:
                    torch.cuda.nvtx.range_push(f"layer{layer_idx}_bwd")
                    st_ref["started"] = True
                st_ref["remaining"] -= 1
                if st_ref["remaining"] == 0 and st_ref["started"]:
                    torch.cuda.nvtx.range_pop()
                    st_ref["started"] = False
                    st_ref["remaining"] = st_ref["count"]
                return grad
            return hook

        handles.append(p.register_hook(make_hook(idx, st)))

    return handles


def _install_component_bwd_nvtx_param_hooks(submod: torch.nn.Module, get_step, start_step: int, end_step: int):
    """Backward NVTX ranges for embed/lm_head based on parameter grad hooks."""
    if not torch.cuda.is_available():
        return []

    def group_key(param_name: str):
        if "embed_tokens" in param_name or "model_embed_tokens" in param_name:
            return "embed"
        if "lm_head" in param_name:
            return "lm_head"
        return None

    groups = {}
    for name, p in submod.named_parameters(recurse=True):
        k = group_key(name)
        if k is None:
            continue
        groups.setdefault(k, []).append((name, p))

    state = {k: {"count": len(v), "remaining": len(v), "started": False} for k, v in groups.items()}
    handles = []

    for k, params in groups.items():
        st = state[k]
        tag = f"{k}_bwd"

        for _, p in params:
            def make_hook(tag_name, st_ref):
                def hook(grad):
                    step = get_step()
                    if not (start_step <= step <= end_step):
                        return grad
                    if st_ref["remaining"] == st_ref["count"] and not st_ref["started"]:
                        torch.cuda.nvtx.range_push(tag_name)
                        st_ref["started"] = True
                    st_ref["remaining"] -= 1
                    if st_ref["remaining"] == 0 and st_ref["started"]:
                        torch.cuda.nvtx.range_pop()
                        st_ref["started"] = False
                        st_ref["remaining"] = st_ref["count"]
                    return grad
                return hook

            handles.append(p.register_hook(make_hook(tag, st)))

    return handles


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


def attach_layer_timers(model: torch.nn.Module, enable: bool = True,
                       start_step: int = 1, end_step: int = 1):
    """Attach CUDA-event timers to each Llama decoder layer to measure per-layer forward/backward.
       Only steps in [start_step, end_step] (inclusive, 1-based) are recorded."""
    if not enable:
        return None, [], []

    if not torch.cuda.is_available():
        print("Layer timing disabled: CUDA is not available.")
        return None, [], []


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
                step = getattr(module, "_lt_step", None)
                if step is not None and start_step <= step <= end_step:
                    elapsed_ms = module._lt_fwd_start.elapsed_time(end)
                    stats["forward"][idx] += elapsed_ms
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
                step = getattr(module, "_lt_step", None)
                if step is not None and start_step <= step <= end_step:
                    elapsed_ms = module._lt_bwd_start.elapsed_time(end)
                    stats["backward"][idx] += elapsed_ms
                    stats["backward_count"][idx] += 1
        return hook

    for idx, layer in enumerate(decoder_layers):
        handles.append(layer.register_forward_pre_hook(make_fwd_pre(idx)))
        handles.append(layer.register_forward_hook(make_fwd_post(idx)))
        handles.append(layer.register_full_backward_pre_hook(make_bwd_pre(idx)))
        handles.append(layer.register_full_backward_hook(make_bwd_post(idx)))

        return stats, handles, decoder_layers

def _set_layer_step(layers, step: int):
    """Propagate the current global step to each hooked decoder layer so hooks can read it."""
    for layer in layers:
        layer._lt_step = step

def log_layer_timers(stats, rank: int, start_step: int, end_step: int):
    if stats is None:
        return
    if rank != 0:
        return

    print(f"=== Layer timing (ms) for steps {start_step}-{end_step} ===")
    for idx, (ft, bt, fc, bc) in enumerate(zip(
        stats["forward"], stats["backward"], stats["forward_count"], stats["backward_count"]
    )):
        f_avg = ft / fc if fc else 0.0
        b_avg = bt / bc if bc else 0.0
        print(f"layer{idx:02d}: fwd_total={ft:.4f}ms avg={f_avg:.4f}ms over {fc} calls | "
              f"bwd_total={bt:.4f}ms avg={b_avg:.4f}ms over {bc} calls")


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
parser.add_argument('--layer-timing-start-step', type=int, default=51, help='1-based inclusive start step for layer timing')
parser.add_argument('--layer-timing-end-step', type=int, default=60, help='1-based inclusive end step for layer timing')
parser.add_argument('--layer-timing-mode', type=str, default="fx", help='"fx": instrument FX graph using name patterns + NVTX, "wrap": wrap LlamaDecoderLayer.forward (may be bypassed by FX).')
# NOTE: torch.autograd.profiler.emit_nvtx() emits NVTX ranges for *every* autograd/ATen op.
# This looks like "kernel-level push/pop spam" in nsys. For layer-block NVTX only, keep this False
# and rely on FX-inserted NVTX ranges (layer{idx}_fwd / embed_fwd / lm_head_fwd).
parser.add_argument('--emit-nvtx', type=bool, default=False, help='Enable autograd emit_nvtx() (per-op NVTX; noisy). Turn off for layer-block NVTX only.')
parser.add_argument('--nsys-capture', type=bool, default=True, help='Call torch.cuda.profiler start/stop for steps [layer-timing-start-step, layer-timing-end-step] so nsys can capture only that window with --capture-range=cudaProfilerApi.')

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
layer_start = args.layer_timing_start_step
layer_end = args.layer_timing_end_step
if layer_start > layer_end:
    print(f"[layer-timing] start_step {layer_start} > end_step {layer_end}, disabling layer timing.")
    args.layer_timing = False
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

optimus_p = Optimus_p(model, mbsize=gas, use_gpu=True,
             pp_size=pp_degree, dp_size=dp_degree, tp_size=tp_degree,
             activation_ckpt=False,
             force_free_mem=True, display_mem=False,
             swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze=IR_Anal.SEQUENTIAL,
             timers=timers, profile_mode=profile_mode)

# Attach layer timers after pipeline setup so we hook the actual submodules used in this rank.
layer_timing_target = None
try:
    if hasattr(optimus_p, "ir") and hasattr(optimus_p.ir, "model_ir") and optimus_p.ir.model_ir:
        layer_timing_target = optimus_p.ir.model_ir[0]
except Exception:
    layer_timing_target = None
if layer_timing_target is None:
    layer_timing_target = model
layer_stats = None
layer_modules = []
layer_handles = []
instrumented_layers = 0

if args.layer_timing and args.layer_timing_mode == "wrap":
    layer_stats, layer_modules = wrap_decoder_layers_for_timing(layer_timing_target, start_step=layer_start, end_step=layer_end)
    instrumented_layers = len(layer_modules)
    if rank == 0 and (layer_stats is not None):
        print(f"[layer-timing][wrap] wrapped layers: {instrumented_layers}, step window [{layer_start}, {layer_end}]")
elif args.layer_timing and args.layer_timing_mode == "fx":
    # instrument FX graph (NVTX + CUDA-event timers)
    # FX instrumentation happens inside Optimus_p init (before clean_module_memory) when env is set.
    if rank == 0:
        print("[layer-timing][fx] note: set OPTPRIME_FX_LAYER_TIMING=1 (and *_START_STEP/*_END_STEP) to enable FX instrumentation.")       

optimus_p.train()

# Use torch.optim.AdamW for compatibility across transformers versions.
try:
    optimus_p.optimizer = torch.optim.AdamW(optimus_p.parameters(), lr=3e-5, foreach=False)
except TypeError:
    optimus_p.optimizer = torch.optim.AdamW(optimus_p.parameters(), lr=3e-5)

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
    global_step = 0
    # Install backward NVTX hooks once (they self-gate by step)
    bwd_nvtx_step = {"val": 0}
    bwd_hooks = []
    if args.emit_nvtx:
        try:
            bwd_hooks = _install_layer_bwd_nvtx_param_hooks(
                optimus_p.run_info.submod,
                get_step=lambda: bwd_nvtx_step["val"],
                start_step=layer_start,
                end_step=layer_end,
            )
            comp_hooks = _install_component_bwd_nvtx_param_hooks(
                optimus_p.run_info.submod,
                get_step=lambda: bwd_nvtx_step["val"],
                start_step=layer_start,
                end_step=layer_end,
            )
            bwd_hooks.extend(comp_hooks)
            print0(f"[nvtx-bwd] installed param hooks: {len(bwd_hooks)} (layers + embed/lm_head)")
        except Exception as e:
            print0(f"[nvtx-bwd] failed to install hooks: {e}")
    if profile_mode == "0":
        for i, batch in enumerate(dataloader):
            data, labels = None, None
            global_step += 1
            bwd_nvtx_step["val"] = global_step
            if optimus_p.is_first_stage():
                tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
                data, labels = tokens.input_ids, tokens.input_ids

            labels = optimus_p.move_labels2last_stage(labels)

            optimus_p.optimizer.zero_grad()
            _set_layer_step(layer_modules, global_step)
            fx_layer_timer_set_step(global_step)

            # nsys capture window using CUDA profiler API (works with nsys --capture-range=cudaProfilerApi)
            if args.nsys_capture and global_step == layer_start:
                try:
                    torch.cuda.profiler.start()
                except Exception:
                    pass

            # Enable NVTX ranges for autograd ops (includes backward) in the measurement window
            if args.emit_nvtx and (layer_start <= global_step <= layer_end):
                with torch.autograd.profiler.emit_nvtx():
                    optimus_p.run(data, labels, mode=pp_schedule)
            else:
                optimus_p.run(data, labels, mode=pp_schedule)

            if args.nsys_capture and global_step == layer_end + 1:
                try:
                    torch.cuda.profiler.stop()
                except Exception:
                    pass

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
            global_step += 1
            if optimus_p.is_first_stage():
                tokens =  tokenizer(batch, padding=True, truncation=True, max_length=1024,return_tensors="pt")
                data, labels = tokens.input_ids, tokens.input_ids
            labels = optimus_p.move_labels2last_stage(labels)
            optimus_p.optimizer.zero_grad()
            _set_layer_step(layer_modules, global_step)     
            fx_layer_timer_set_step(global_step)

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

log_layer_timers(layer_stats, rank, layer_start, layer_end)
fx_stats = fx_layer_timer_get()
if rank == 0:
    if fx_stats is not None:
        print(f"=== FX Layer timing (ms) for steps {layer_start}-{layer_end} ===")
        for idx, (ft, bt, fc, bc) in enumerate(zip(
            fx_stats["forward"], fx_stats["backward"], fx_stats["forward_count"], fx_stats["backward_count"]
        )):
            f_avg = ft / fc if fc else 0.0
            b_avg = bt / bc if bc else 0.0
            print(f"fx-layer{idx:02d}: fwd_total={ft:.4f}ms avg={f_avg:.4f}ms over {fc} calls | "
                  f"bwd_total={bt:.4f}ms avg={b_avg:.4f}ms over {bc} calls")

    else:
        print("=== FX Layer timing: (no stats collected) ===")

# Component forward timing: print from ranks that actually execute the component (no collectives).
if fx_stats is not None:
    comps = fx_stats.get("components") or {}
    if comps:
        lines = []
        for name in ("embed", "lm_head"):
            if name not in comps:
                continue
            ft = float(comps[name].get("forward", 0.0))
            fc = int(comps[name].get("forward_count", 0))
            if fc <= 0:
                continue
            f_avg = ft / fc
            lines.append(f"fx-{name}: fwd_total={ft:.4f}ms avg={f_avg:.4f}ms over {fc} calls")
        if lines:
            print(f"=== FX Component fwd timing (ms) [rank{rank}] ===")
            for ln in lines:
                print(ln)

# FX layer timing (per-rank, parsable): print only nonzero layers from each rank.
if fx_stats is not None:
    try:
        fwd = fx_stats.get("forward") or []
        fcnt = fx_stats.get("forward_count") or []
        bwd = fx_stats.get("backward") or []
        bcnt = fx_stats.get("backward_count") or []
        any_nonzero = any((c > 0) for c in fcnt)
        if any_nonzero:
            print(f"=== FX Layer timing (ms) [rank{rank}] steps {layer_start}-{layer_end} ===")
            for idx in range(min(len(fwd), len(fcnt), len(bwd), len(bcnt))):
                fc = int(fcnt[idx])
                bc = int(bcnt[idx])
                if fc <= 0 and bc <= 0:
                    continue
                ft = float(fwd[idx])
                bt = float(bwd[idx])
                f_avg = ft / fc if fc else 0.0
                b_avg = bt / bc if bc else 0.0
                print(
                    f"fxr-layer{idx:02d}: fwd_total={ft:.4f}ms avg={f_avg:.4f}ms over {fc} calls | "
                    f"bwd_total={bt:.4f}ms avg={b_avg:.4f}ms over {bc} calls"
                )
    except Exception:
        pass