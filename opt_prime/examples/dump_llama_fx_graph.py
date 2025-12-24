#!/usr/bin/env python3
"""
Dump Llama model FX graph / node list WITHOUT loading weights (empty/meta init).

Why:
  - Llama-70B cannot be loaded "fully" on each rank just to build IR.
  - For graph inspection and PP partition planning, we only need the module structure and shapes from config.

Requires:
  - transformers
  - accelerate (for init_empty_weights)

Example (PP=8, dump nodes):
  export LLAMA_ACCESS_TOKEN=...
  python3 examples/dump_llama_fx_graph.py \
    --model-id meta-llama/Llama-3.1-70B \
    --pp-degree 8 \
    --split-method simple \
    --out _logs/llama70b_pp8_fx_nodes.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import torch

from transformers import AutoConfig, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.IR import IR  # noqa: E402


class _DummyTpl:
    def __init__(self, rank: int = 0):
        self.rank = rank


class _DummyOptimus:
    def __init__(self):
        self.tpl = _DummyTpl(rank=0)
        self.model2type: Dict[str, int] = {"hf": 50, "sy": 51, "vt": 52}


def _get_token(args) -> str | None:
    if args.access_token:
        return args.access_token
    # keep compatibility with existing scripts
    return os.getenv("LLAMA_ACCESS_TOKEN") or os.getenv("HF_ACCESS_TOKEN")


def _build_empty_model(model_id: str, token: str | None, trust_remote_code: bool) -> torch.nn.Module:
    cfg = AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
    # reduce side-effects
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    try:
        from accelerate import init_empty_weights  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "accelerate is required for empty-weight instantiation. Install: pip install accelerate"
        ) from e

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return model


def main() -> int:
    p = argparse.ArgumentParser(description="Dump Llama FX graph nodes without loading weights.")
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--access-token", type=str, default=None)
    p.add_argument("--pp-degree", type=int, default=8)
    p.add_argument("--split-method", type=str, default="simple", choices=["simple", "llama-tp-split"])
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--include-node-args", action="store_true", help="Also print node.args (can be huge).")
    args = p.parse_args()

    # IR.py uses os.environ["RANK"] for some prints; make offline run safe.
    os.environ.setdefault("RANK", "0")

    token = _get_token(args)
    model = _build_empty_model(args.model_id, token=token, trust_remote_code=args.trust_remote_code)

    optimus = _DummyOptimus()
    ir = IR(model, optimus)
    ir.retrieve_IR(model)
    ir.split_IR(model, args.split_method, num_stage=args.pp_degree)

    root = ir.model_ir[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"model_id={args.model_id}\n")
        f.write(f"pp_degree={args.pp_degree}\n")
        f.write(f"split_method={args.split_method}\n")
        f.write("---- FX nodes ----\n")

        for n in root.graph.nodes:
            if args.include_node_args:
                f.write(
                    f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}, n.args:{n.args}, n.all_input_nodes:{n.all_input_nodes}\n"
                )
            else:
                f.write(f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}\n")

        f.write("---- submodules ----\n")
        for name, m in root.named_children():
            f.write(f"{name}: {m.__class__.__name__}\n")

    print(f"[done] wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


