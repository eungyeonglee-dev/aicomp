#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# Usage: torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> 
#                 --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama_70b.py [options]
#
# *** This program was tested with torch 2.5.0 and transformers 4.46.2.
#     The version of transformers used must be consistent across all machines used for testing ***
#
# Modes:
#   - Training Mode (default): Run distributed training
#   - Profile Mode (--profile_mode): Measure layer-wise execution time
#
import torch
import torch.nn as nn
import torch.distributed as dist
import datetime
import logging
import os
import sys
import math
import time
import json
from collections import defaultdict
from packaging import version

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal, LayerProfileInterpreter
from opt_prime.utils import ts, log

logging.basicConfig(level=logging.ERROR)


# ============================================================================
# Argument Parser
# ============================================================================
import argparse

parser = argparse.ArgumentParser(description="Llama Training/Profiling with Pipeline Parallelism")

# Model settings
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                    help="Model name: meta-llama/Llama-3.3-70B-Instruct or meta-llama/Llama-3.2-1B")
parser.add_argument("--num_hidden_layers", type=int, default=None, 
                    help="Number of transformer layers (None=use model default, or specify for lightweight)")
parser.add_argument("--use_cache", type=bool, default=False)
parser.add_argument("--llama_access_token", type=str, default=None)

# Training settings
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--micro_batch_size", type=int, default=1)
parser.add_argument("--pp_size", type=int, default=2)
parser.add_argument("--tp_size", type=int, default=1)
parser.add_argument("--dp_size", type=int, default=1)
parser.add_argument("--run_id", type=str, default="default")

# Profile mode settings
parser.add_argument("--profile_mode", action="store_true", help="Enable layer profiling mode")
parser.add_argument("--profile_steps", type=int, default=15, help="Total profiling steps")
parser.add_argument("--profile_warmup_steps", type=int, default=10, help="Warmup steps before measurement")
parser.add_argument("--profile_start_node", type=str, default="model_layers_0_self_attn_q_proj")
parser.add_argument("--profile_end_node", type=str, default="model_layers_0_mlp_down_proj")
parser.add_argument("--profile_output", type=str, default="", help="JSON output file for profile results")

args, unknown = parser.parse_known_args()


# ============================================================================
# Directories & Result Files
# ============================================================================
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)


# ============================================================================
# Model Configurations (from HuggingFace official configs)
# ============================================================================
# These configs match the official HuggingFace model configurations exactly

MODEL_CONFIGS = {
    # Llama-3.3-70B-Instruct (https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
    "meta-llama/Llama-3.3-70B-Instruct": {
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 80,  # Full model has 80 layers
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "head_dim": 128,  # hidden_size / num_attention_heads = 8192/64
    },
    # Llama-3.2-1B (https://huggingface.co/meta-llama/Llama-3.2-1B)
    "meta-llama/Llama-3.2-1B": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 16,  # Full model has 16 layers
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "head_dim": 64,  # hidden_size / num_attention_heads = 2048/32
    },
    # Llama-3.2-3B (https://huggingface.co/meta-llama/Llama-3.2-3B)
    "meta-llama/Llama-3.2-3B": {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "num_hidden_layers": 28,  # Full model has 28 layers
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "head_dim": 128,  # hidden_size / num_attention_heads = 3072/24
    },
}

# Aliases for convenience
MODEL_CONFIGS["70B"] = MODEL_CONFIGS["meta-llama/Llama-3.3-70B-Instruct"]
MODEL_CONFIGS["1B"] = MODEL_CONFIGS["meta-llama/Llama-3.2-1B"]
MODEL_CONFIGS["3B"] = MODEL_CONFIGS["meta-llama/Llama-3.2-3B"]


def get_model_config(model_name: str) -> dict:
    """Get model configuration by name or alias"""
    # Check direct match
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Check if it's a partial match (e.g., "70B" in "Llama-3.3-70B-Instruct")
    model_name_lower = model_name.lower()
    for key in MODEL_CONFIGS:
        if model_name_lower in key.lower() or key.lower() in model_name_lower:
            return MODEL_CONFIGS[key]
    
    # Default to 70B if not found
    print(f"[WARNING] Unknown model '{model_name}', defaulting to Llama-3.3-70B-Instruct config")
    return MODEL_CONFIGS["meta-llama/Llama-3.3-70B-Instruct"]


# ============================================================================
# Utility Functions
# ============================================================================
def get_total_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def save_exit_code(exit_code: int, run_id: str, elapsed_time: float = None):
    """Save exit code (rank 0 only)"""
    if os.environ.get("RANK", "0") != "0":
        return
    try:
        log_path = f"tmp/exitcode_{run_id}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            if exit_code == 0 and elapsed_time is not None:
                f.write(f"{exit_code},{elapsed_time:.3f}")
            else:
                f.write(str(exit_code))
        print(f"[{ts()}][rank:0] EXIT_CODE {exit_code} saved to {log_path}")
    except Exception as e:
        print(f"[{ts()}][rank:0] Failed to save EXIT_CODE: {e}")


def save_profile_result(result: dict, output_path: str = ""):
    """Save profile results to JSON file"""
    if os.environ.get("RANK", "0") != "0":
        return
    
    if not output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{RESULT_DIR}/profile_{timestamp}.json"
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[{ts()}][rank:0] Profile results saved to {output_path}")
    except Exception as e:
        print(f"[{ts()}][rank:0] Failed to save profile results: {e}")


def gather_and_print_combined_profile(block_profiler, warmup_steps: int, world_size: int, rank: int, gloo_group=None):
    """
    Gather profile results from all ranks and print combined summary.
    
    PP=2 with 2 layers:
      - Stage 0 (rank 0): embedding + layer_0
      - Stage 1 (rank 1): layer_1 + lm_head
    
    Combined output shows: embedding, avg(layer_0, layer_1), lm_head
    """
    import pickle
    
    # Get local summary
    local_summary = block_profiler.get_summary(warmup_steps)
    
    # Use dist.all_gather_object for simple object gathering (works with gloo)
    all_summaries_list = [None] * world_size
    local_data = {
        'rank': rank,
        'stage': block_profiler.stage,
        'summary': local_summary
    }
    
    # all_gather_object uses gloo backend internally if available
    try:
        dist.all_gather_object(all_summaries_list, local_data, group=gloo_group)
    except Exception as e:
        print(f"[rank:{rank}] gather error: {e}")
        # Fallback: just use local data
        if rank == 0:
            all_summaries_list = [local_data]
        else:
            return None
    
    if rank == 0:
        # Process gathered data
        all_summaries = {}
        for data in all_summaries_list:
            if data is not None:
                all_summaries[data['rank']] = {
                    'stage': data['stage'],
                    'summary': data['summary']
                }
        
        # Combine results
        combined = {
            'embedding': {'mean_ms': 0, 'min_ms': float('inf'), 'max_ms': 0, 'count': 0},
            'layers': [],  # List of layer times
            'lm_head': {'mean_ms': 0, 'min_ms': float('inf'), 'max_ms': 0, 'count': 0}
        }
        
        for rank_id, rank_data in all_summaries.items():
            summary = rank_data['summary']
            for key, stats in summary.items():
                if key == 'embedding':
                    combined['embedding']['mean_ms'] = stats['mean_ms']
                    combined['embedding']['median_ms'] = stats.get('median_ms', stats['mean_ms'])
                    combined['embedding']['min_ms'] = stats['min_ms']
                    combined['embedding']['max_ms'] = stats['max_ms']
                    combined['embedding']['count'] = 1
                elif key == 'lm_head':
                    combined['lm_head']['mean_ms'] = stats['mean_ms']
                    combined['lm_head']['median_ms'] = stats.get('median_ms', stats['mean_ms'])
                    combined['lm_head']['min_ms'] = stats['min_ms']
                    combined['lm_head']['max_ms'] = stats['max_ms']
                    combined['lm_head']['count'] = 1
                elif key.startswith('layer_'):
                    combined['layers'].append({
                        'name': key,
                        'mean_ms': stats['mean_ms'],
                        'median_ms': stats.get('median_ms', stats['mean_ms']),
                        'min_ms': stats['min_ms'],
                        'max_ms': stats['max_ms'],
                        'std_ms': stats.get('std_ms', 0)
                    })
        
        # Sort layers by index
        combined['layers'].sort(key=lambda x: int(x['name'].split('_')[1]))
        
        # Calculate layer average (use median for more stable values)
        if combined['layers']:
            layer_median_total = sum(l.get('median_ms', l['mean_ms']) for l in combined['layers'])
            layer_mean_total = sum(l['mean_ms'] for l in combined['layers'])
            layer_avg = layer_median_total / len(combined['layers'])
        else:
            layer_median_total = 0
            layer_mean_total = 0
            layer_avg = 0
        
        # Use median values for total (more robust to outliers)
        emb_val = combined['embedding'].get('median_ms', combined['embedding']['mean_ms'])
        lm_val = combined['lm_head'].get('median_ms', combined['lm_head']['mean_ms'])
        total_time = emb_val + layer_median_total + lm_val
        
        # Print combined summary
        print(f"\n{'='*90}")
        print(f" COMBINED PROFILE (All Stages/Ranks)")
        print(f" PP={world_size} stages, {len(combined['layers'])} transformer layers")
        print(f" Values: per-forward-pass average")
        print(f"{'='*90}")
        
        print(f"\n[Combined Summary - Per Forward Pass (Median values for stability)]")
        print(f"{'Component':<25} {'Median(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'%':<8}")
        print(f"{'-'*75}")
        
        # Embedding
        if combined['embedding']['count'] > 0:
            emb = combined['embedding']
            median_val = emb.get('median_ms', emb['mean_ms'])
            pct = (median_val / total_time * 100) if total_time > 0 else 0
            print(f"{'Embedding':<25} {median_val:<12.4f} {emb['min_ms']:<12.4f} {emb['max_ms']:<12.4f} {pct:<8.2f}")
        
        # Transformer Layers (average)
        if combined['layers']:
            pct = (layer_median_total / total_time * 100) if total_time > 0 else 0
            print(f"{'Transformer (total)':<25} {layer_median_total:<12.4f} {'-':<12} {'-':<12} {pct:<8.2f}")
            print(f"{'  └ Avg per layer':<25} {layer_avg:<12.4f}")
        
        # LM Head
        if combined['lm_head']['count'] > 0:
            lm = combined['lm_head']
            median_val = lm.get('median_ms', lm['mean_ms'])
            pct = (median_val / total_time * 100) if total_time > 0 else 0
            print(f"{'LM Head':<25} {median_val:<12.4f} {lm['min_ms']:<12.4f} {lm['max_ms']:<12.4f} {pct:<8.2f}")
        
        print(f"{'-'*75}")
        print(f"{'TOTAL':<25} {total_time:<12.4f}")
        
        # Per-layer breakdown
        if combined['layers']:
            print(f"\n[Per-Layer Breakdown]")
            print(f"{'Layer':<15} {'Median(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Mean(ms)':<12} {'%':<8}")
            print(f"{'-'*75}")
            for layer in combined['layers']:
                median_val = layer.get('median_ms', layer['mean_ms'])
                pct = (median_val / total_time * 100) if total_time > 0 else 0
                print(f"{layer['name']:<15} {median_val:<12.4f} {layer['min_ms']:<12.4f} {layer['max_ms']:<12.4f} {layer['mean_ms']:<12.4f} {pct:<8.2f}")
        
        print(f"\n{'='*90}\n")
        
        return combined
    
    return None


# ============================================================================
# Layer Block Profiler (Measures entire layer block: q_proj start -> down_proj end)
# ============================================================================
class LayerBlockProfiler:
    """
    Measures transformer layer block time from q_proj (start) to down_proj (end).
    
    Layer N timing:
      START: model_layers_N_self_attn_q_proj (pre_hook)
      END:   model_layers_N_mlp_down_proj (post_hook)
    
    NOTE: Times are accumulated per-step (matching LayerProfiler behavior).
          With num_mb micro-batches per step, this measures total time across all micro-batches.
    """
    
    def __init__(self, submod, device, rank, stage, num_mb: int = 1):
        self.submod = submod
        self.device = device
        self.rank = rank
        self.stage = stage
        self.num_mb = num_mb  # micro-batches per step
        
        # Per-step accumulated times (list of step totals)
        self.step_times = defaultdict(list)  # component -> list of step totals
        
        # Current step accumulator (sum of all micro-batches in current step)
        self.current_step_times = defaultdict(float)
        
        # Timing state - store event pairs for deferred timing
        self.current_layer_start = {}  # layer_N -> start_event
        self.embedding_start = None
        self.lm_head_start = None
        
        # Deferred event pairs (will be processed at step_end)
        self.pending_events = []  # [(key, start_event, end_event), ...]
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on layer boundary modules"""
        import re
        
        for name, module in self.submod.named_modules():
            name_lower = name.lower()
            
            # Embedding: start and end on same module
            if 'embed_tokens' in name_lower or name_lower == 'model_embed_tokens':
                self._register_embedding_hooks(name, module)
            
            # LM Head: start and end on same module
            elif 'lm_head' in name_lower:
                self._register_lm_head_hooks(name, module)
            
            # Layer START: q_proj
            elif 'self_attn_q_proj' in name_lower or 'self_attn.q_proj' in name_lower:
                match = re.search(r'layers?[_.]?(\d+)', name_lower)
                if match:
                    layer_idx = int(match.group(1))
                    self._register_layer_start_hook(name, module, layer_idx)
            
            # Layer END: mlp_down_proj
            elif 'mlp_down_proj' in name_lower or 'mlp.down_proj' in name_lower:
                match = re.search(r'layers?[_.]?(\d+)', name_lower)
                if match:
                    layer_idx = int(match.group(1))
                    self._register_layer_end_hook(name, module, layer_idx)
        
        if self.rank == 0:
            layers_found = set()
            for name, _ in self.submod.named_modules():
                match = re.search(r'layers?[_.]?(\d+)', name.lower())
                if match:
                    layers_found.add(int(match.group(1)))
            print(f"[LayerBlockProfiler] Found {len(layers_found)} transformer layers: {sorted(layers_found)}")
            print(f"[LayerBlockProfiler] num_mb={self.num_mb} (times will be per-forward-pass average)")
    
    def _register_embedding_hooks(self, name, module):
        """Register pre/post hooks for embedding"""
        profiler = self
        
        def pre_hook(mod, inp):
            profiler.embedding_start = torch.cuda.Event(enable_timing=True)
            profiler.embedding_start.record()
        
        def post_hook(mod, inp, out):
            if profiler.embedding_start:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                # Defer timing - no sync during forward pass
                profiler.pending_events.append(('embedding', profiler.embedding_start, end_event))
                profiler.embedding_start = None
        
        self.hooks.append(module.register_forward_pre_hook(pre_hook))
        self.hooks.append(module.register_forward_hook(post_hook))
    
    def _register_lm_head_hooks(self, name, module):
        """Register pre/post hooks for lm_head"""
        profiler = self
        
        def pre_hook(mod, inp):
            profiler.lm_head_start = torch.cuda.Event(enable_timing=True)
            profiler.lm_head_start.record()
        
        def post_hook(mod, inp, out):
            if profiler.lm_head_start:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                # Defer timing - no sync during forward pass
                profiler.pending_events.append(('lm_head', profiler.lm_head_start, end_event))
                profiler.lm_head_start = None
        
        self.hooks.append(module.register_forward_pre_hook(pre_hook))
        self.hooks.append(module.register_forward_hook(post_hook))
    
    def _register_layer_start_hook(self, name, module, layer_idx):
        """Register pre_hook on q_proj (layer START)"""
        profiler = self
        layer_key = f"layer_{layer_idx}"
        
        def pre_hook(mod, inp):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            profiler.current_layer_start[layer_key] = start_event
        
        self.hooks.append(module.register_forward_pre_hook(pre_hook))
        if self.rank == 0:
            print(f"[LayerBlockProfiler] {layer_key} START hook: {name}")
    
    def _register_layer_end_hook(self, name, module, layer_idx):
        """Register post_hook on down_proj (layer END)"""
        profiler = self
        layer_key = f"layer_{layer_idx}"
        
        def post_hook(mod, inp, out):
            if layer_key in profiler.current_layer_start:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                # Defer timing - no sync during forward pass
                profiler.pending_events.append((layer_key, profiler.current_layer_start[layer_key], end_event))
                del profiler.current_layer_start[layer_key]
        
        self.hooks.append(module.register_forward_hook(post_hook))
        if self.rank == 0:
            print(f"[LayerBlockProfiler] {layer_key} END hook: {name}")
    
    def step_end(self):
        """Called at end of each training step to finalize measurements"""
        # Sync GPU and process all pending events
        torch.cuda.synchronize()
        
        for key, start_event, end_event in self.pending_events:
            elapsed = start_event.elapsed_time(end_event)
            self.current_step_times[key] += elapsed
        self.pending_events.clear()
        
        # Move current step times to history
        for key, value in self.current_step_times.items():
            self.step_times[key].append(value)
        self.current_step_times.clear()
    
    def get_summary(self, warmup_steps: int = 0) -> dict:
        """Get timing summary (excluding warmup, per-forward-pass averages)"""
        summary = {}
        
        for key, times in self.step_times.items():
            measured = times[warmup_steps:] if len(times) > warmup_steps else times
            if measured:
                # Divide by num_mb to get per-forward-pass time
                per_fwd_times = [t / self.num_mb for t in measured]
                sorted_times = sorted(per_fwd_times)
                mean_val = sum(per_fwd_times) / len(per_fwd_times)
                median_val = sorted_times[len(sorted_times) // 2]
                summary[key] = {
                    'mean_ms': mean_val,
                    'median_ms': median_val,  # P50 - more robust to outliers
                    'min_ms': min(per_fwd_times),
                    'max_ms': max(per_fwd_times),
                    'std_ms': (sum((t - mean_val)**2 for t in per_fwd_times) / len(per_fwd_times)) ** 0.5 if len(per_fwd_times) > 1 else 0,
                    'measured_steps': len(measured)
                }
        
        return summary
    
    def print_summary(self, warmup_steps: int = 0):
        """Print layer block timing summary"""
        if self.rank != 0:
            return
        
        summary = self.get_summary(warmup_steps)
        # Use median for total time calculation (more robust)
        total_time = sum(v.get('median_ms', v['mean_ms']) for v in summary.values())
        
        # Categorize components
        transformer_layers = {k: v for k, v in summary.items() if k.startswith('layer_')}
        transformer_total = sum(v['mean_ms'] for v in transformer_layers.values())
        
        measured_steps = list(summary.values())[0].get('measured_steps', 0) if summary else 0
        
        print(f"\n{'='*90}")
        print(f" Layer Block Profile (Stage {self.stage}, Rank {self.rank})")
        print(f" Measures: q_proj (start) -> down_proj (end) per layer [END-TO-END]")
        print(f" Warmup: {warmup_steps} steps, Measured: {measured_steps} steps")
        print(f" Values: per-forward-pass average (num_mb={self.num_mb})")
        print(f"{'='*90}")
        
        # Main summary
        print(f"\n[Summary - Per Forward Pass]")
        print(f"{'Component':<20} {'Median(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Mean(ms)':<12} {'%':<8}")
        print(f"{'-'*80}")
        
        # Embedding
        if 'embedding' in summary:
            s = summary['embedding']
            pct = (s['median_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"{'Embedding':<20} {s['median_ms']:<12.4f} {s['min_ms']:<12.4f} {s['max_ms']:<12.4f} {s['mean_ms']:<12.4f} {pct:<8.2f}")
        
        # Transformer total (use median sum)
        if transformer_layers:
            trans_median_total = sum(v.get('median_ms', v['mean_ms']) for v in transformer_layers.values())
            trans_pct = (trans_median_total / total_time * 100) if total_time > 0 else 0
            print(f"{'Transformer (total)':<20} {trans_median_total:<12.4f} {'-':<12} {'-':<12} {transformer_total:<12.4f} {trans_pct:<8.2f}")
        
        # LM Head
        if 'lm_head' in summary:
            s = summary['lm_head']
            pct = (s['median_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"{'LM Head':<20} {s['median_ms']:<12.4f} {s['min_ms']:<12.4f} {s['max_ms']:<12.4f} {s['mean_ms']:<12.4f} {pct:<8.2f}")
        
        print(f"{'-'*80}")
        print(f"{'TOTAL':<20} {total_time:<12.4f}")
        
        # Per-layer breakdown
        if transformer_layers:
            print(f"\n[Per-Layer Breakdown (q_proj -> down_proj, END-TO-END)]")
            print(f"{'Layer':<15} {'Mean(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Std(ms)':<12} {'%':<8}")
            print(f"{'-'*75}")
            
            sorted_layers = sorted(transformer_layers.items(), key=lambda x: int(x[0].split('_')[1]))
            for layer_key, stats in sorted_layers:
                pct = (stats['mean_ms'] / total_time * 100) if total_time > 0 else 0
                print(f"{layer_key:<15} {stats['mean_ms']:<12.4f} {stats['min_ms']:<12.4f} {stats['max_ms']:<12.4f} {stats['std_ms']:<12.4f} {pct:<8.2f}")
            
            if len(transformer_layers) > 0:
                avg_layer_time = transformer_total / len(transformer_layers)
                print(f"{'-'*75}")
                print(f"{'Avg per layer':<15} {avg_layer_time:<12.4f}")
        
        print(f"\n{'='*90}\n")
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


# ============================================================================
# Layer Profiler (Hook-based, works during forward/backward)
# ============================================================================
class LayerProfiler:
    """Hook-based layer profiler for measuring ALL module execution times during training"""
    
    # Module types to profile (all leaf modules)
    PROFILE_MODULE_TYPES = (
        nn.Linear,
        nn.Embedding,
        nn.LayerNorm,
        nn.RMSNorm if hasattr(nn, 'RMSNorm') else type(None),
        nn.SiLU,
        nn.GELU,
        nn.ReLU,
        nn.Tanh,
        nn.Sigmoid,
        nn.Softmax,
        nn.Dropout,
        nn.Conv1d,
        nn.Conv2d,
    )
    
    def __init__(self, submod, device, rank, stage, num_mb: int = 1):
        self.submod = submod
        self.device = device
        self.rank = rank
        self.stage = stage
        self.num_mb = num_mb  # micro-batches per step
        
        self.step_times = defaultdict(list)
        self.current_step_times = defaultdict(float)
        self.hooks = []
        self.module_info = {}  # Store module type info
        self._register_hooks()
    
    def _get_category(self, name: str, module: nn.Module) -> str:
        """Determine category from module name and type"""
        name_lower = name.lower()
        module_type = type(module).__name__
        
        # By module type
        if isinstance(module, nn.Embedding):
            return 'embedding'
        elif isinstance(module, (nn.SiLU, nn.GELU, nn.ReLU, nn.Tanh, nn.Sigmoid)):
            return 'activation'
        elif 'norm' in module_type.lower() or 'norm' in name_lower:
            if 'input_layernorm' in name_lower or 'post_attention' in name_lower:
                return 'layer_norm'
            elif 'model_norm' in name_lower or (not 'layer' in name_lower):
                return 'final_norm'
            return 'layer_norm'
        elif isinstance(module, nn.Dropout):
            return 'dropout'
        elif isinstance(module, nn.Softmax):
            return 'softmax'
        
        # By name pattern
        if 'embed' in name_lower:
            return 'embedding'
        elif 'lm_head' in name_lower:
            return 'lm_head'
        elif 'q_proj' in name_lower or 'k_proj' in name_lower or 'v_proj' in name_lower:
            return 'attn_qkv_proj'
        elif 'o_proj' in name_lower:
            return 'attn_o_proj'
        elif 'gate_proj' in name_lower or 'up_proj' in name_lower:
            return 'mlp_gate_up'
        elif 'down_proj' in name_lower:
            return 'mlp_down'
        elif 'mlp' in name_lower:
            return 'mlp_other'
        elif 'attn' in name_lower or 'attention' in name_lower:
            return 'attention_other'
        elif 'layers' in name_lower or 'layer' in name_lower:
            return 'transformer_other'
        else:
            return 'other'
    
    def _get_op_type(self, module: nn.Module) -> str:
        """Get operation type string"""
        return type(module).__name__
    
    def _is_leaf_module(self, module: nn.Module) -> bool:
        """Check if module is a leaf (no children or specific types)"""
        children = list(module.children())
        if len(children) == 0:
            return True
        # Also profile containers that have their own forward logic
        if isinstance(module, self.PROFILE_MODULE_TYPES):
            return True
        return False
    
    def _register_hooks(self):
        """Register forward hooks for ALL leaf modules"""
        for name, module in self.submod.named_modules():
            # Skip empty name (root module)
            if not name:
                continue
            
            # Profile leaf modules and specific types
            if self._is_leaf_module(module):
                category = self._get_category(name, module)
                op_type = self._get_op_type(module)
                
                # Store module info
                self.module_info[name] = {
                    'type': op_type,
                    'category': category
                }
                
                hook = self._make_hook(name, category, op_type)
                handle = module.register_forward_pre_hook(hook.pre_hook)
                self.hooks.append(handle)
                handle = module.register_forward_hook(hook.post_hook)
                self.hooks.append(handle)
        
        if self.rank == 0:
            print(f"[LayerProfiler] Registered {len(self.hooks)//2} modules for profiling")
    
    def _make_hook(self, name: str, category: str, op_type: str):
        """Create timing hooks"""
        class TimingHook:
            def __init__(hook_self, profiler, name, category, op_type):
                hook_self.profiler = profiler
                hook_self.name = name
                hook_self.category = category
                hook_self.op_type = op_type
                hook_self.start_event = None
                hook_self.end_event = None
            
            def pre_hook(hook_self, module, input):
                hook_self.start_event = torch.cuda.Event(enable_timing=True)
                hook_self.end_event = torch.cuda.Event(enable_timing=True)
                hook_self.start_event.record()
            
            def post_hook(hook_self, module, input, output):
                if hook_self.start_event:
                    hook_self.end_event.record()
                    torch.cuda.synchronize()
                    elapsed = hook_self.start_event.elapsed_time(hook_self.end_event)
                    hook_self.profiler.current_step_times[hook_self.name] += elapsed
                    hook_self.profiler.current_step_times[f"category_{hook_self.category}"] += elapsed
                    hook_self.profiler.current_step_times[f"op_type_{hook_self.op_type}"] += elapsed
        
        return TimingHook(self, name, category, op_type)
    
    def step_end(self):
        """Called at end of each training step"""
        for key, value in self.current_step_times.items():
            self.step_times[key].append(value)
        self.current_step_times.clear()
    
    def get_summary(self, warmup_steps: int = 0) -> dict:
        """Get timing summary (excluding warmup, per-forward-pass averages)"""
        summary = {}
        for key, times in self.step_times.items():
            measured_times = times[warmup_steps:] if len(times) > warmup_steps else times
            if measured_times:
                # Divide by num_mb to get per-forward-pass time
                per_fwd_times = [t / self.num_mb for t in measured_times]
                mean_val = sum(per_fwd_times) / len(per_fwd_times)
                summary[key] = {
                    'mean_ms': mean_val,
                    'min_ms': min(per_fwd_times),
                    'max_ms': max(per_fwd_times),
                    'total_ms': sum(per_fwd_times),
                    'count': len(measured_times)
                }
        return summary
    
    def _get_block_name(self, module_name: str) -> str:
        """Get block name: embedding / layer_N / final_norm / lm_head"""
        name_lower = module_name.lower()
        
        if 'embed' in name_lower:
            return 'embedding'
        elif 'lm_head' in name_lower:
            return 'lm_head'
        elif 'model_norm' in name_lower or (name_lower == 'norm'):
            return 'final_norm'
        else:
            # Extract layer number: model_layers_0_xxx -> layer_0
            import re
            match = re.search(r'layers?[_.]?(\d+)', name_lower)
            if match:
                return f"layer_{match.group(1)}"
        return 'other'
    
    def print_summary(self, warmup_steps: int = 0):
        """Print timing summary grouped by: Embedding / Transformer Layer Blocks / LM Head"""
        if self.rank != 0:
            return
        
        summary = self.get_summary(warmup_steps)
        
        # Separate modules only (exclude category_ and op_type_ prefixes)
        modules = {k: v for k, v in summary.items() 
                   if not k.startswith('category_') and not k.startswith('op_type_')}
        
        # Calculate total time
        total_time = sum(v['mean_ms'] for v in modules.values())
        
        # Group by block (embedding / layer_N / final_norm / lm_head)
        block_times = defaultdict(lambda: {'mean_ms': 0.0, 'min_ms': float('inf'), 'max_ms': 0.0, 'modules': []})
        
        for name, stats in modules.items():
            block = self._get_block_name(name)
            block_times[block]['mean_ms'] += stats['mean_ms']
            block_times[block]['min_ms'] = min(block_times[block]['min_ms'], stats['min_ms'])
            block_times[block]['max_ms'] = max(block_times[block]['max_ms'], stats['max_ms'])
            block_times[block]['modules'].append((name, stats))
        
        # Separate into categories
        embedding_blocks = {k: v for k, v in block_times.items() if k == 'embedding'}
        layer_blocks = {k: v for k, v in block_times.items() if k.startswith('layer_')}
        final_norm_blocks = {k: v for k, v in block_times.items() if k == 'final_norm'}
        lm_head_blocks = {k: v for k, v in block_times.items() if k == 'lm_head'}
        other_blocks = {k: v for k, v in block_times.items() 
                        if k not in embedding_blocks and k not in layer_blocks 
                        and k not in final_norm_blocks and k not in lm_head_blocks}
        
        # Calculate transformer layers total
        transformer_total = sum(v['mean_ms'] for v in layer_blocks.values())
        transformer_total += sum(v['mean_ms'] for v in final_norm_blocks.values())
        
        measured_steps = len(list(self.step_times.values())[0]) - warmup_steps if self.step_times else 0
        
        print(f"\n{'='*90}")
        print(f" Layer Profile Summary (Stage {self.stage}, Rank {self.rank})")
        print(f" Warmup: {warmup_steps} steps, Measured: {measured_steps} steps")
        print(f" Total Modules Profiled: {len(modules)}")
        print(f"{'='*90}")
        
        # ===== Main Summary: Embedding / Transformer / LM Head =====
        print(f"\n[Main Components]")
        print(f"{'Component':<30} {'Mean(ms)':<12} {'%':<10} {'Modules':<8}")
        print(f"{'-'*65}")
        
        # Embedding
        emb_time = sum(v['mean_ms'] for v in embedding_blocks.values())
        emb_modules = sum(len(v['modules']) for v in embedding_blocks.values())
        emb_pct = (emb_time / total_time * 100) if total_time > 0 else 0
        print(f"{'Embedding':<30} {emb_time:<12.4f} {emb_pct:<10.2f} {emb_modules:<8}")
        
        # Transformer Layers (sum of all layer_N + final_norm)
        trans_modules = sum(len(v['modules']) for v in layer_blocks.values())
        trans_modules += sum(len(v['modules']) for v in final_norm_blocks.values())
        trans_pct = (transformer_total / total_time * 100) if total_time > 0 else 0
        print(f"{'Transformer Layers':<30} {transformer_total:<12.4f} {trans_pct:<10.2f} {trans_modules:<8}")
        
        # LM Head
        lm_time = sum(v['mean_ms'] for v in lm_head_blocks.values())
        lm_modules = sum(len(v['modules']) for v in lm_head_blocks.values())
        lm_pct = (lm_time / total_time * 100) if total_time > 0 else 0
        print(f"{'LM Head':<30} {lm_time:<12.4f} {lm_pct:<10.2f} {lm_modules:<8}")
        
        # Other (if any)
        if other_blocks:
            other_time = sum(v['mean_ms'] for v in other_blocks.values())
            other_modules = sum(len(v['modules']) for v in other_blocks.values())
            other_pct = (other_time / total_time * 100) if total_time > 0 else 0
            print(f"{'Other':<30} {other_time:<12.4f} {other_pct:<10.2f} {other_modules:<8}")
        
        print(f"{'-'*65}")
        print(f"{'TOTAL':<30} {total_time:<12.4f} {'100.00':<10} {len(modules):<8}")
        
        # ===== Transformer Layer Breakdown =====
        if layer_blocks or final_norm_blocks:
            print(f"\n[Transformer Layer Breakdown]")
            print(f"{'Layer':<20} {'Mean(ms)':<12} {'%':<10} {'Modules':<8}")
            print(f"{'-'*55}")
            
            # Sort layers by number
            sorted_layers = sorted(layer_blocks.items(), key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else 0)
            for layer_name, stats in sorted_layers:
                pct = (stats['mean_ms'] / total_time * 100) if total_time > 0 else 0
                print(f"{layer_name:<20} {stats['mean_ms']:<12.4f} {pct:<10.2f} {len(stats['modules']):<8}")
            
            # Final norm
            for block_name, stats in final_norm_blocks.items():
                pct = (stats['mean_ms'] / total_time * 100) if total_time > 0 else 0
                print(f"{block_name:<20} {stats['mean_ms']:<12.4f} {pct:<10.2f} {len(stats['modules']):<8}")
        
        # ===== Per-Layer Module Details =====
        print(f"\n[Module Details by Layer]")
        print(f"{'-'*90}")
        
        # Embedding details
        if embedding_blocks:
            print(f"\n  >> Embedding")
            for block_name, stats in embedding_blocks.items():
                for mod_name, mod_stats in sorted(stats['modules'], key=lambda x: x[1]['mean_ms'], reverse=True):
                    op_type = self.module_info.get(mod_name, {}).get('type', 'unknown')
                    pct = (mod_stats['mean_ms'] / total_time * 100) if total_time > 0 else 0
                    print(f"     {mod_name:<50} {op_type:<12} {mod_stats['mean_ms']:<10.4f} ms ({pct:.2f}%)")
        
        # Layer details
        sorted_layers = sorted(layer_blocks.items(), key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else 0)
        for layer_name, stats in sorted_layers:
            print(f"\n  >> {layer_name}")
            for mod_name, mod_stats in sorted(stats['modules'], key=lambda x: x[1]['mean_ms'], reverse=True):
                op_type = self.module_info.get(mod_name, {}).get('type', 'unknown')
                pct = (mod_stats['mean_ms'] / total_time * 100) if total_time > 0 else 0
                print(f"     {mod_name:<50} {op_type:<12} {mod_stats['mean_ms']:<10.4f} ms ({pct:.2f}%)")
        
        # Final norm details
        if final_norm_blocks:
            print(f"\n  >> Final Norm")
            for block_name, stats in final_norm_blocks.items():
                for mod_name, mod_stats in sorted(stats['modules'], key=lambda x: x[1]['mean_ms'], reverse=True):
                    op_type = self.module_info.get(mod_name, {}).get('type', 'unknown')
                    pct = (mod_stats['mean_ms'] / total_time * 100) if total_time > 0 else 0
                    print(f"     {mod_name:<50} {op_type:<12} {mod_stats['mean_ms']:<10.4f} ms ({pct:.2f}%)")
        
        # LM Head details
        if lm_head_blocks:
            print(f"\n  >> LM Head")
            for block_name, stats in lm_head_blocks.items():
                for mod_name, mod_stats in sorted(stats['modules'], key=lambda x: x[1]['mean_ms'], reverse=True):
                    op_type = self.module_info.get(mod_name, {}).get('type', 'unknown')
                    pct = (mod_stats['mean_ms'] / total_time * 100) if total_time > 0 else 0
                    print(f"     {mod_name:<50} {op_type:<12} {mod_stats['mean_ms']:<10.4f} ms ({pct:.2f}%)")
        
        print(f"\n{'='*90}\n")
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


# ============================================================================
# Main Execution
# ============================================================================
EXIT_CODE = 0
ELAPSED_TIME = None

# Initialize distributed
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
master_addr = os.getenv("MASTER_ADDR")
master_port = os.getenv("MASTER_PORT")

timeout = datetime.timedelta(hours=1)
init_method = f"tcp://{master_addr}:{master_port}"

print(f"[{ts()}] rank:{rank}, world_size:{world_size}, init_method:{init_method}")

dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=init_method, timeout=timeout)
group_gloo = dist.new_group(backend="gloo", timeout=timeout)
store = dist.distributed_c10d._get_default_store()
if store is not None:
    store.set_timeout(timeout)


try:
    # Version checks
    required_torch = "2.3.1"
    required_tf = "4.46.2"
    
    if version.parse(torch.__version__) < version.parse(required_torch):
        raise ValueError(f'Requires torch >= {required_torch}, got {torch.__version__}')
    if version.parse(transformers.__version__) < version.parse(required_tf):
        raise ValueError(f'Requires transformers >= {required_tf}, got {transformers.__version__}')
    
    log(f"[rank:{rank}] torch={torch.__version__}, transformers={transformers.__version__} ✓")

    # ========================================================================
    # Tokenizer Setup
    # ========================================================================
    use_cache = args.use_cache
    access_token = args.llama_access_token
    
    if use_cache:
        # Convert model name to cache directory format
        # e.g., "meta-llama/Llama-3.3-70B-Instruct" -> "models--meta-llama--Llama-3.3-70B-Instruct"
        cache_dir_name = "models--" + args.model_name.replace("/", "--")
        model_path = f"/root/.cache/huggingface/hub/{cache_dir_name}/snapshots/"
        
        if os.path.exists(model_path):
            snapshot_id = os.listdir(model_path)
            model_path = os.path.join(model_path, snapshot_id[0])
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            if rank == 0:
                log(f'> Tokenizer loaded from cache: {model_path}')
        else:
            # Fallback: try loading from HuggingFace with token
            if access_token is None:
                raise ValueError(f"Cache not found at {model_path} and LLAMA_ACCESS_TOKEN not provided")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token)
            if rank == 0:
                log(f'> Tokenizer loaded from HuggingFace: {args.model_name}')
    else:
        if access_token is None:
            raise ValueError("LLAMA_ACCESS_TOKEN required when use_cache=False")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token)
        if rank == 0:
            log(f'> Tokenizer loaded from HuggingFace: {args.model_name}')
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # ========================================================================
    # Model Setup (Dynamic config based on model_name)
    # ========================================================================
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    num_mb = batch_size // (micro_batch_size * args.dp_size)
    
    # Get model configuration based on model_name
    model_cfg = get_model_config(args.model_name)
    
    # Determine number of layers (use args if specified, else model default)
    if args.num_hidden_layers is not None:
        num_layers = args.num_hidden_layers
    else:
        num_layers = model_cfg["num_hidden_layers"]
    
    # Determine model size string for logging
    if "70B" in args.model_name or "70b" in args.model_name:
        model_size_str = "70B"
    elif "3B" in args.model_name or "3b" in args.model_name:
        model_size_str = "3B"
    elif "1B" in args.model_name or "1b" in args.model_name:
        model_size_str = "1B"
    else:
        model_size_str = "Unknown"
    
    log('===> Model loading...') if local_rank == 0 else None
    
    if local_rank == 0:
        log(f'> Model: {args.model_name} ({model_size_str} architecture)')
        log(f'> Config: hidden_size={model_cfg["hidden_size"]}, '
            f'intermediate_size={model_cfg["intermediate_size"]}, '
            f'num_attention_heads={model_cfg["num_attention_heads"]}, '
            f'num_key_value_heads={model_cfg["num_key_value_heads"]}')
        log(f'> Using {num_layers} layers (full model: {model_cfg["num_hidden_layers"]} layers)')

    for i in range(local_world_size):
        if local_rank == i:
            # Llama 3.x vocab_size is 128256
            # Using tokenizer.vocab_size can cause mismatch with actual token IDs
            actual_vocab_size = model_cfg["vocab_size"]
            tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', actual_vocab_size)
            if tokenizer_vocab_size > actual_vocab_size:
                actual_vocab_size = tokenizer_vocab_size
            
            if local_rank == 0:
                log(f'> Tokenizer vocab_size: {tokenizer_vocab_size}, Using: {actual_vocab_size}')
            
            # Create LlamaConfig from model configuration
            config = LlamaConfig(
                vocab_size=actual_vocab_size,
                hidden_size=model_cfg["hidden_size"],
                intermediate_size=model_cfg["intermediate_size"],
                num_hidden_layers=num_layers,
                num_attention_heads=model_cfg["num_attention_heads"],
                num_key_value_heads=model_cfg["num_key_value_heads"],
                max_position_embeddings=model_cfg["max_position_embeddings"],
                rms_norm_eps=model_cfg["rms_norm_eps"],
                rope_theta=model_cfg["rope_theta"],
                use_cache=False,
                tie_word_embeddings=False,
            )
            model = LlamaForCausalLM(config)

            if local_rank == 0:
                mode_str = "PROFILE" if args.profile_mode else "TRAIN"
                log(f'> [{mode_str} MODE] {num_layers}-layer Llama {model_size_str} model')
                log(f'> Total parameters: {get_total_params(model):,}')
                log(f'> PP={args.pp_size}, TP={args.tp_size}, DP={args.dp_size}')
                log(f'> GBS={batch_size}, MBS={micro_batch_size}, #MB={num_mb}')

            optimus_p = Optimus_p(
                model, num_mb,
                use_gpu=True,
                pp_size=args.pp_size, tp_size=args.tp_size, dp_size=args.dp_size,
                activation_ckpt=False, force_free_mem=True, display_mem=True,
                swap_opt_in_fwdbwd=False, swap_model_in_optstep=False,
                ir_analyze=IR_Anal.PARALLEL, pre_barrier=group_gloo
            )
            log(f"[rank:{optimus_p.get_rank()}] Optimus_p initialized")

        if local_rank > i:
            log(f"[local_rank:{local_rank}] Waiting for rank {i}...")
            dist.barrier(group=group_gloo)
            log(f"[local_rank:{local_rank}] Rank {i} finished")
    
    log('===> Model loading completed') if local_rank == 0 else None
    optimus_p.train()

    # ========================================================================
    # PROFILE MODE
    # ========================================================================
    if args.profile_mode:
        log(f"[{ts()}] ========== PROFILE MODE ==========") if rank == 0 else None
        
        # Setup profilers
        # 1. LayerProfiler: measures individual module times (sum of all ops)
        profiler = LayerProfiler(
            optimus_p.run_info.submod,
            optimus_p.run_info.device,
            rank, 
            optimus_p.tpl.stage,
            num_mb=num_mb  # Pass num_mb for per-forward-pass normalization
        )
        
        # 2. LayerBlockProfiler: measures layer block times (q_proj start -> down_proj end, END-TO-END)
        block_profiler = LayerBlockProfiler(
            optimus_p.run_info.submod,
            optimus_p.run_info.device,
            rank,
            optimus_p.tpl.stage,
            num_mb=num_mb  # Pass num_mb for per-forward-pass normalization
        )
        
        # Use optimizer to ensure realistic timing
        if args.tp_size > 1:
            optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5, foreach=False)
        else:
            optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)
        
        # Load dataset
        datasets = load_dataset("squad").data["train"]["context"]
        datasets = [str(record) for record in datasets if len(str(record)) < 500]
        dataloader = optimus_p.prepare_dataloader(datasets, batch_size)
        
        log(f"[rank:{rank}] Profile: {args.profile_steps} steps ({args.profile_warmup_steps} warmup)")
        
        # Profile loop
        tick = time.time()
        step_count = 0
        
        for batch in dataloader:
            if step_count >= args.profile_steps:
                break
            
            data, labels = None, None
            if optimus_p.is_first_stage():
                tokens = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
                data, labels = tokens.input_ids, tokens.input_ids
            
            labels = optimus_p.move_labels2last_stage(labels)
            optimus_p.optimizer.zero_grad()
            
            # Forward + Backward
            optimus_p.run(data, labels, mode="1f1b")
            
            if args.tp_size == 1:
                torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
            
            optimus_p.optimizer.step()
            profiler.step_end()
            block_profiler.step_end()  # Finalize layer block times for this step
            
            step_count += 1
            if rank == 0 and step_count % 5 == 0:
                log(f"[rank:0] Profile step {step_count}/{args.profile_steps}")
        
        tock = time.time()
        
        # Synchronize all ranks before printing
        dist.barrier()
        
        # Print per-rank results (for debugging)
        block_profiler.print_summary(warmup_steps=args.profile_warmup_steps)
        
        # Synchronize again
        dist.barrier()
        
        # Gather and print combined results from all ranks
        combined_result = gather_and_print_combined_profile(
            block_profiler, 
            args.profile_warmup_steps, 
            world_size, 
            rank,
            gloo_group=group_gloo
        )
        
        # Save results (rank 0 only)
        if rank == 0:
            profile_result = {
                'config': {
                    'model_name': args.model_name,
                    'model_size': model_size_str,
                    'hidden_size': model_cfg["hidden_size"],
                    'intermediate_size': model_cfg["intermediate_size"],
                    'num_attention_heads': model_cfg["num_attention_heads"],
                    'num_key_value_heads': model_cfg["num_key_value_heads"],
                    'num_layers': num_layers,
                    'num_layers_full': model_cfg["num_hidden_layers"],
                    'batch_size': batch_size,
                    'micro_batch_size': micro_batch_size,
                    'pp_size': args.pp_size,
                    'tp_size': args.tp_size,
                    'dp_size': args.dp_size,
                    'profile_steps': args.profile_steps,
                    'warmup_steps': args.profile_warmup_steps,
                    'total_time_sec': tock - tick,
                },
                'combined_timing': combined_result,
                'per_rank_timing': {
                    f'rank_{rank}': block_profiler.get_summary(args.profile_warmup_steps)
                }
            }
            
            output_path = args.profile_output if args.profile_output else f"{RESULT_DIR}/profile_{args.run_id}.json"
            save_profile_result(profile_result, output_path)
            ELAPSED_TIME = tock - tick
        
        profiler.remove_hooks()
        block_profiler.remove_hooks()
        EXIT_CODE = 0

    # ========================================================================
    # TRAINING MODE
    # ========================================================================
    else:
        log(f"[{ts()}] ========== TRAINING MODE ==========") if rank == 0 else None
        
        # Optimizer setup
        if args.tp_size > 1:
            optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5, foreach=False)
        else:
            optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)

        scheduler = torch.optim.lr_scheduler.StepLR(optimus_p.optimizer, 1.0, gamma=0.95)
        
        # Dataset
        datasets = load_dataset("squad").data["train"]["context"]
        datasets = [str(record) for record in datasets if len(str(record)) < 500]
        dataloader = optimus_p.prepare_dataloader(datasets, batch_size)
        data_size = len(dataloader.dataset)
        nbatches = len(dataloader)
        
        log(f"[rank:{optimus_p.get_rank()}] data_size={data_size}, nbatches={nbatches}")

        epochs = 1

        def train():
            optimus_p.train()
            total_loss = 0
            start_time = time.time()

            for i, batch in enumerate(dataloader):
                data, labels = None, None
                
                if optimus_p.is_first_stage():
                    tokens = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
                    data, labels = tokens.input_ids, tokens.input_ids

                labels = optimus_p.move_labels2last_stage(labels)
                optimus_p.optimizer.zero_grad()
                optimus_p.run(data, labels, mode="1f1b")

                if optimus_p.is_last_stage():
                    loss = optimus_p.get_loss()
                else:
                    loss = None

                if args.tp_size == 1:
                    torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)

                optimus_p.optimizer.step()

                if optimus_p.is_last_stage():
                    loss = sum(loss) / optimus_p.mbsize
                    total_loss += loss
                    log_interval = 1
                    if i % log_interval == 0 and i > 0:
                        cur_loss = total_loss / log_interval
                        elapsed = time.time() - start_time
                        if optimus_p.get_rank() % int(world_size/args.pp_size) == 0:
                            print(f"===== {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
                            print(f'| epoch {epoch:3d} | {i:5d}/{nbatches:5d} batches | '
                                  f'lr {scheduler.get_lr()[0]:02.2f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | '
                                  f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
                        total_loss = 0
                        start_time = time.time()

        if optimus_p.get_rank() == 0:
            tick = time.time()

        for epoch in range(1, epochs + 1):
            train()
            scheduler.step()

        if optimus_p.get_rank() == 0:
            tock = time.time()
            ELAPSED_TIME = tock - tick
            print(f'Time elapsed: {ELAPSED_TIME:.3f} sec')
            EXIT_CODE = 0


except torch.cuda.OutOfMemoryError as e:
    print(f"[{ts()}] ERROR: OOM - {e}")
    EXIT_CODE = 10

except dist.DistBackendError as e:
    print(f"[{ts()}] ERROR: Distributed communication failed - {e}")
    EXIT_CODE = 20

except Exception as e:
    print(f"[{ts()}] ERROR: {e}")
    import traceback
    traceback.print_exc()
    EXIT_CODE = 30

finally:
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            print(f"[rank:{os.environ.get('RANK','?')}] process group destroyed.")
    except Exception as e:
        print(f"[rank:{os.environ.get('RANK','?')}] cleanup failed: {e}")
        if EXIT_CODE == 0:
            EXIT_CODE = 40

print(f">>> EXIT_CODE: {EXIT_CODE}, ELAPSED_TIME: {ELAPSED_TIME}")
save_exit_code(EXIT_CODE, args.run_id, ELAPSED_TIME)
sys.exit(EXIT_CODE)
