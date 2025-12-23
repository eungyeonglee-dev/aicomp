"""
NVTX-only layer profiling for Llama-3.2-1B (forward, transformer layers only).

- Adds NVTX ranges: layer{idx}_fwd around each decoder block forward.
- Avoids torch.autograd.profiler.emit_nvtx() to prevent per-op NVTX spam.
- Uses CUDA profiler start/stop so nsys can capture only a step window.

Env:
  - LLAMA_ACCESS_TOKEN (required unless model is cached)
  - HF_TRANSFORMERS_SRC (e.g., /workspace/transformers-4.46.2/src)
  - DP_SIZE, TP_SIZE (2D only; PP not supported)
  - LLAMA_MBS (micro-batch size), default 1
  - LLAMA_NVTX_START_STEP / LLAMA_NVTX_END_STEP, default 51/60
  - LLAMA_PROFILE_CUT (1/0), default 1 (stop after end_step+1)
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn


# ---- local transformers source path (optional) ----
_src = os.getenv("HF_TRANSFORMERS_SRC", "/workspace/transformers-4.46.2/src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed._tensor import Replicate  # noqa: E402
from torch.distributed.tensor.parallel import (  # noqa: E402
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn.parallel import DistributedDataParallel  # noqa: E402


def _nvtx_push(tag: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(tag)


def _nvtx_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _unwrap_ddp(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


def _install_layer_fwd_nvtx(model: torch.nn.Module, get_step, start_step: int, end_step: int):
    base = _unwrap_ddp(model)
    layers = getattr(getattr(base, "model", base), "layers", None)
    if layers is None:
        if int(os.environ.get("RANK", "0")) == 0:
            print("[nvtx] could not find base.model.layers; skip")
        return

    for idx, layer in enumerate(layers):
        orig = layer.forward

        def make_wrapped(layer_idx, orig_fwd):
            def wrapped(*args, **kwargs):
                step = int(get_step())
                if start_step <= step <= end_step:
                    _nvtx_push(f"layer{layer_idx}_fwd")
                    try:
                        return orig_fwd(*args, **kwargs)
                    finally:
                        _nvtx_pop()
                return orig_fwd(*args, **kwargs)

            return wrapped

        layer.forward = make_wrapped(idx, orig)

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[nvtx] installed transformer-layer NVTX (count={len(layers)}) steps [{start_step},{end_step}]")


class Runner2D:
    def __init__(self, model: nn.Module, *, dp_size: int, tp_size: int, local_rank: int, rank: int, world_size: int):
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self._step = 0

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        self.device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        self.dp_mesh = self.device_mesh["dp"]
        self.tp_mesh = self.device_mesh["tp"]

        if rank == 0:
            print(f"[2d] dp={dp_size} tp={tp_size} world={world_size}")

        if tp_size == 1 and dp_size == 1:
            self.model = model.to(self.device)
        else:
            # TP for embed/lm_head + per-layer TP plan
            self.model = parallelize_module(
                model.to(device="cuda"),
                self.tp_mesh,
                {
                    "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
                    "lm_head": ColwiseParallel(output_layouts=Replicate()),
                },
            )
            for layer_id, block in enumerate(self.model.model.layers):
                layer_tp_plan = {
                    f"model.layers.{layer_id}.self_attn.q_proj": ColwiseParallel(),
                    f"model.layers.{layer_id}.self_attn.k_proj": ColwiseParallel(),
                    f"model.layers.{layer_id}.self_attn.v_proj": ColwiseParallel(),
                    f"model.layers.{layer_id}.self_attn.o_proj": RowwiseParallel(),
                    f"model.layers.{layer_id}.mlp.gate_proj": ColwiseParallel(),
                    f"model.layers.{layer_id}.mlp.down_proj": RowwiseParallel(),
                    f"model.layers.{layer_id}.mlp.up_proj": ColwiseParallel(),
                }
                attn = block.self_attn
                attn.num_heads = attn.num_heads // self.tp_mesh.size()
                attn.num_key_value_heads = attn.num_key_value_heads // self.tp_mesh.size()
                parallelize_module(block, device_mesh=self.tp_mesh, parallelize_plan=layer_tp_plan)

            self.model = DistributedDataParallel(self.model, find_unused_parameters=True, device_mesh=self.dp_mesh)

        self.criterion = nn.CrossEntropyLoss().cuda(rank)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

    def run_step(self, input_ids: torch.Tensor, labels: torch.Tensor, step: int) -> torch.Tensor:
        self._step = int(step)
        self.optimizer.zero_grad(set_to_none=True)
        out = self.model(input_ids)
        logits = out.logits.view(-1, out.logits.size(-1))
        lab = labels.view(-1)
        loss = self.criterion(logits, lab)
        loss.backward()
        self.optimizer.step()
        return loss


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    token = os.getenv("LLAMA_ACCESS_TOKEN")
    if token is None:
        raise ValueError("LLAMA_ACCESS_TOKEN is not set")

    dp_size = int(os.getenv("DP_SIZE", "1"))
    tp_size = int(os.getenv("TP_SIZE", str(world_size)))
    if dp_size * tp_size != world_size:
        raise ValueError(f"DP_SIZE*TP_SIZE must equal WORLD_SIZE. Got {dp_size}*{tp_size}!={world_size}")

    mbs = int(os.getenv("LLAMA_MBS", "1"))
    start_step = int(os.getenv("LLAMA_NVTX_START_STEP", "51"))
    end_step = int(os.getenv("LLAMA_NVTX_END_STEP", "60"))
    profile_cut = os.getenv("LLAMA_PROFILE_CUT", "1") == "1"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=token, use_cache=False)

    runner = Runner2D(model, dp_size=dp_size, tp_size=tp_size, local_rank=local_rank, rank=rank, world_size=world_size)
    _install_layer_fwd_nvtx(runner.model, get_step=lambda: runner._step, start_step=start_step, end_step=end_step)

    ds = load_dataset("squad").data["train"]["context"]
    ds = [str(x) for x in ds if len(str(x)) < 500]
    loader = DataLoader(ds, batch_size=mbs)

    if rank == 0:
        print(f"[run] mbs={mbs} steps [{start_step},{end_step}] profile_cut={profile_cut}")

    total_loss = 0.0
    t0 = time.time()
    for i, batch in enumerate(loader):
        step = i + 1
        toks = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        input_ids = toks.input_ids.to(device=f"cuda:{local_rank}")
        labels = toks.input_ids.to(device=f"cuda:{local_rank}")

        # nsys capture window via cudaProfilerApi
        if step == start_step:
            try:
                torch.cuda.profiler.start()
            except Exception:
                pass

        loss = runner.run_step(input_ids, labels, step=step)

        if step == end_step + 1:
            try:
                torch.cuda.profiler.stop()
            except Exception:
                pass

        if local_rank == 0:
            total_loss += float(loss)
            if step % 10 == 0:
                elapsed = time.time() - t0
                print(f"| step {step:4d} | ms/batch {elapsed*1000/10:7.2f} | loss {total_loss/10:6.3f} | ppl {math.exp(total_loss/10):8.2f}")
                total_loss = 0.0
                t0 = time.time()

        if profile_cut and step >= (end_step + 1):
            break


if __name__ == "__main__":
    main()


