#!/usr/bin/env python3
"""
Dump (split) model structure similar to opt_prime/_logs/llama1B_pp*_modelgraph.txt, but without training.

Key goal: support very large Llama models (e.g., 70B) by instantiating with empty weights (meta device),
so we can trace/split and print module structure without loading full checkpoints into RAM/VRAM.

Example:
  OPTPRIME_PRINT_IR=0 \
  python3 examples/dump_modelgraph_llama.py \
    --model-id meta-llama/Llama-3.1-70B \
    --pp-degree 2 \
    --split-method simple \
    --out _logs/llama70b_pp2_model_only.txt \
    --dump model
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pprint import pformat
from typing import Any, Dict

from transformers import AutoConfig, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.IR import IR  # noqa: E402


@dataclass
class _DummyTpl:
    rank: int = 0


@dataclass
class _DummyOptimus:
    tpl: _DummyTpl = _DummyTpl()
    model2type: Dict[str, int] = None  # filled in __post_init__

    def __post_init__(self):
        if self.model2type is None:
            # keep consistent with Optimus_p
            self.model2type = {"hf": 50, "sy": 51, "vt": 52}

    def get_rank(self) -> int:
        return self.tpl.rank


def _get_token(args) -> str | None:
    if args.access_token:
        return args.access_token
    return os.getenv("LLAMA_ACCESS_TOKEN")


def _load_llama_empty(model_id: str, token: str | None, trust_remote_code: bool = False):
    """Instantiate model on meta device (no real weights) using accelerate.init_empty_weights if available."""
    cfg = AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
    # Make sure we don't build cache-related blocks unnecessarily
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    try:
        from accelerate import init_empty_weights  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This dumper requires `accelerate` for empty-weight instantiation. "
            "Install it (pip install accelerate) or use a smaller model."
        ) from e

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    # Some models reference this attribute at runtime.
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return model


def main() -> int:
    p = argparse.ArgumentParser(description="Dump Llama model structure / split GraphModule children (no training).")
    p.add_argument("--model-id", type=str, required=True, help="HF model id or local path (e.g., meta-llama/Llama-3.1-70B)")
    p.add_argument("--access-token", type=str, default=None, help="HF access token (or use env LLAMA_ACCESS_TOKEN)")
    p.add_argument("--pp-degree", type=int, default=2, help="Number of pipeline stages to split into")
    p.add_argument("--split-method", type=str, default="simple", choices=["simple", "llama-tp-split"], help="IR split method")
    p.add_argument("--dump", type=str, default="children", choices=["children", "model", "fx_only"], help="What to dump")
    p.add_argument("--out", type=str, required=True, help="Output text file path")
    p.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers")
    args = p.parse_args()

    # Ensure IR print guards won't crash in offline mode.
    os.environ.setdefault("RANK", "0")

    token = _get_token(args)
    model = _load_llama_empty(args.model_id, token=token, trust_remote_code=args.trust_remote_code)

    dummy = _DummyOptimus()
    ir = IR(model, dummy)
    ir.retrieve_IR(model)
    ir.split_IR(model, args.split_method, num_stage=args.pp_degree)

    root = ir.model_ir[0]

    if args.dump == "children":
        payload: Any = list(root.named_children())
    elif args.dump == "model":
        # Print only the "model" submodule (this is the big part users usually want).
        payload = root.get_submodule("model")
    else:  # fx_only
        payload = [(n.op, n.name, str(n.target)) for n in root.graph.nodes]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(pformat(payload, width=200))
        f.write("\n")

    print(f"[done] wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


