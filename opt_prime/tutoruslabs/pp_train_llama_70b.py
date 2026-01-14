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
parser.add_argument("--profile_fx", action="store_true", help="Use FX Interpreter-based profiling (node-level)")
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
# FX Interpreter-based Layer Profiler (precise node-level timing)
# ============================================================================
class FXLayerProfiler:
    """
    FX Interpreter 기반 레이어 프로파일러.
    
    정확한 경계 기반 측정:
    - Embedding: model_embed_tokens ~ to_2 (pow_1 직전까지)
    - Transformer Layer N: pow_(N*2+1) ~ 다음 layer q_proj 직전
    - LM Head: lm_head 노드
    
    FX graph를 직접 순회하며 각 노드의 실행 시간을 측정하고
    지정된 경계에 따라 그룹화합니다.
    """
    
    def __init__(self, submod, device, rank, stage, num_mb: int = 1):
        self.submod = submod
        self.device = device
        self.rank = rank
        self.stage = stage
        self.num_mb = num_mb
        
        # Results storage
        self.step_times = defaultdict(list)  # component -> list of step totals
        self.current_step_times = defaultdict(float)
        self.node_times = defaultdict(list)  # node_name -> list of times
        
        # Graph analysis
        self.graph = submod.graph
        self.node_to_component = {}  # node_name -> component (embedding/layer_N/lm_head)
        self._analyze_graph()
        
        if rank == 0:
            print(f"[FXLayerProfiler] Initialized with {len(self.node_to_component)} nodes")
            print(f"[FXLayerProfiler] Components: {set(self.node_to_component.values())}")
    
    def _analyze_graph(self):
        """
        FX Graph를 분석하여 각 노드의 소속 컴포넌트를 결정합니다.
        
        Boundaries (based on log analysis):
        - Embedding: from model_embed_tokens until we hit pow_1 (RMSNorm start)
        - Transformer Layer N: from pow_(2N+1) until next layer's q_proj
        - LM Head: lm_head node
        """
        import re
        
        current_component = None
        embedding_ended = False
        current_layer = -1
        pow_count = 0
        
        # First pass: identify all layer-related nodes
        layer_q_proj_nodes = {}  # layer_idx -> q_proj node name
        for node in self.graph.nodes:
            if 'self_attn_q_proj' in node.name or 'self_attn.q_proj' in str(node.target):
                match = re.search(r'layers?[_.]?(\d+)', node.name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in layer_q_proj_nodes:
                        layer_q_proj_nodes[layer_idx] = node.name
        
        # Second pass: assign components
        for node in self.graph.nodes:
            node_name = node.name
            
            # Skip placeholder and output nodes
            if node.op in ['placeholder', 'output']:
                continue
            
            # Check for embedding
            if 'embed_tokens' in node_name:
                current_component = 'embedding'
                embedding_ended = False
            
            # Check for lm_head
            elif 'lm_head' in node_name:
                current_component = 'lm_head'
            
            # Check for pow (RMSNorm start marker)
            elif node.op == 'call_method' and node.name.startswith('pow'):
                pow_count += 1
                
                if pow_count == 1 and not embedding_ended:
                    # First pow marks end of embedding, start of layer 0
                    embedding_ended = True
                    current_layer = 0
                    current_component = f'layer_{current_layer}'
                elif pow_count > 1:
                    # Every 2nd pow marks a new layer's input_layernorm
                    # Actually: pow_1 = layer_0 input_norm, pow_2 = layer_0 post_attn_norm
                    #           pow_3 = layer_1 input_norm, pow_4 = layer_1 post_attn_norm
                    # So layer N starts at pow_(2N+1)
                    new_layer = (pow_count - 1) // 2
                    if new_layer != current_layer:
                        current_layer = new_layer
                        current_component = f'layer_{current_layer}'
            
            # Check for layer q_proj to mark layer transition more accurately
            elif node_name in layer_q_proj_nodes.values():
                match = re.search(r'layers?[_.]?(\d+)', node_name)
                if match:
                    layer_idx = int(match.group(1))
                    current_layer = layer_idx
                    current_component = f'layer_{layer_idx}'
            
            # Use layer info from node name if available
            elif current_component is None:
                if 'model_layers' in node_name or 'layers' in str(node.target):
                    match = re.search(r'layers?[_.]?(\d+)', node_name)
                    if match:
                        layer_idx = int(match.group(1))
                        current_component = f'layer_{layer_idx}'
            
            # Assign component
            if current_component:
                self.node_to_component[node_name] = current_component
    
    def run_with_timing(self, input_data):
        """
        FX Graph를 실행하며 각 노드의 시간을 측정합니다.
        
        측정 방법:
        - 시작 전 한 번 동기화 (clean state)
        - 각 노드: start_event.record() → execute → end_event.record() → end_event.synchronize()
        - end_event.synchronize()는 해당 노드의 연산만 대기 (다른 연산 기다리지 않음)
        """
        from torch.fx.interpreter import Interpreter
        
        profiler = self
        
        class TimingInterpreter(Interpreter):
            def __init__(inner_self, gm):
                super().__init__(gm)
                inner_self.first_node = True
            
            def run_node(inner_self, n):
                if n.op in ['placeholder', 'output']:
                    return super().run_node(n)
                
                # 첫 노드 전에만 전체 동기화 (clean state)
                if inner_self.first_node:
                    torch.cuda.synchronize()
                    inner_self.first_node = False
                
                # Record start event
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
                # Execute node
                result = super().run_node(n)
                
                # Record end event and sync only this event
                end_event.record()
                end_event.synchronize()  # 이 노드의 연산만 대기
                
                elapsed_ms = start_event.elapsed_time(end_event)
                
                # Store node time
                profiler.node_times[n.name].append(elapsed_ms)
                
                # Accumulate to component
                component = profiler.node_to_component.get(n.name)
                if component:
                    profiler.current_step_times[component] += elapsed_ms
                
                return result
        
        interpreter = TimingInterpreter(self.submod)
        return interpreter.run(input_data)
    
    def step_end(self):
        """Called at end of each training step"""
        for key, value in self.current_step_times.items():
            self.step_times[key].append(value)
        self.current_step_times.clear()
    
    def get_summary(self, warmup_steps: int = 0) -> dict:
        """Get timing summary (excluding warmup, per-forward-pass averages)"""
        summary = {}
        
        for key, times in self.step_times.items():
            measured = times[warmup_steps:] if len(times) > warmup_steps else times
            if measured:
                per_fwd_times = [t / self.num_mb for t in measured]
                sorted_times = sorted(per_fwd_times)
                mean_val = sum(per_fwd_times) / len(per_fwd_times)
                median_val = sorted_times[len(sorted_times) // 2]
                summary[key] = {
                    'mean_ms': mean_val,
                    'median_ms': median_val,
                    'min_ms': min(per_fwd_times),
                    'max_ms': max(per_fwd_times),
                    'std_ms': (sum((t - mean_val)**2 for t in per_fwd_times) / len(per_fwd_times)) ** 0.5 if len(per_fwd_times) > 1 else 0,
                    'measured_steps': len(measured)
                }
        
        return summary
    
    def get_node_summary(self, warmup_steps: int = 0, top_n: int = 50) -> dict:
        """Get per-node timing summary"""
        summary = {}
        for node_name, times in self.node_times.items():
            measured = times[warmup_steps:] if len(times) > warmup_steps else times
            if measured:
                mean_val = sum(measured) / len(measured)
                component = self.node_to_component.get(node_name, 'unknown')
                summary[node_name] = {
                    'mean_ms': mean_val,
                    'component': component,
                    'calls': len(measured)
                }
        
        # Sort by time
        sorted_summary = dict(sorted(summary.items(), key=lambda x: x[1]['mean_ms'], reverse=True))
        return dict(list(sorted_summary.items())[:top_n])
    
    def print_summary(self, warmup_steps: int = 0):
        """Print layer timing summary"""
        if self.rank != 0:
            return
        
        summary = self.get_summary(warmup_steps)
        total_time = sum(v.get('median_ms', v['mean_ms']) for v in summary.values())
        
        # Categorize
        embedding = {k: v for k, v in summary.items() if k == 'embedding'}
        layers = {k: v for k, v in summary.items() if k.startswith('layer_')}
        lm_head = {k: v for k, v in summary.items() if k == 'lm_head'}
        
        layer_total = sum(v.get('median_ms', v['mean_ms']) for v in layers.values())
        measured_steps = list(summary.values())[0].get('measured_steps', 0) if summary else 0
        
        print(f"\n{'='*90}")
        print(f" FX Interpreter Layer Profile (Stage {self.stage}, Rank {self.rank})")
        print(f" Measures: ALL FX nodes grouped by component")
        print(f" Warmup: {warmup_steps} steps, Measured: {measured_steps} steps")
        print(f" Values: per-forward-pass average (num_mb={self.num_mb})")
        print(f"{'='*90}")
        
        print(f"\n[Summary - Per Forward Pass]")
        print(f"{'Component':<20} {'Median(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Mean(ms)':<12} {'%':<8}")
        print(f"{'-'*80}")
        
        # Embedding
        if embedding:
            s = embedding['embedding']
            pct = (s['median_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"{'Embedding':<20} {s['median_ms']:<12.4f} {s['min_ms']:<12.4f} {s['max_ms']:<12.4f} {s['mean_ms']:<12.4f} {pct:<8.2f}")
        
        # Transformer total
        if layers:
            trans_pct = (layer_total / total_time * 100) if total_time > 0 else 0
            print(f"{'Transformer (total)':<20} {layer_total:<12.4f} {'-':<12} {'-':<12} {sum(v['mean_ms'] for v in layers.values()):<12.4f} {trans_pct:<8.2f}")
        
        # LM Head
        if lm_head:
            s = lm_head['lm_head']
            pct = (s['median_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"{'LM Head':<20} {s['median_ms']:<12.4f} {s['min_ms']:<12.4f} {s['max_ms']:<12.4f} {s['mean_ms']:<12.4f} {pct:<8.2f}")
        
        print(f"{'-'*80}")
        print(f"{'TOTAL':<20} {total_time:<12.4f}")
        
        # Per-layer breakdown
        if layers:
            print(f"\n[Per-Layer Breakdown (FX node-level timing)]")
            print(f"{'Layer':<15} {'Median(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Mean(ms)':<12} {'%':<8}")
            print(f"{'-'*75}")
            
            sorted_layers = sorted(layers.items(), key=lambda x: int(x[0].split('_')[1]))
            for layer_key, stats in sorted_layers:
                pct = (stats['median_ms'] / total_time * 100) if total_time > 0 else 0
                print(f"{layer_key:<15} {stats['median_ms']:<12.4f} {stats['min_ms']:<12.4f} {stats['max_ms']:<12.4f} {stats['mean_ms']:<12.4f} {pct:<8.2f}")
            
            if len(layers) > 0:
                avg_layer = layer_total / len(layers)
                print(f"{'-'*75}")
                print(f"{'Avg per layer':<15} {avg_layer:<12.4f}")
        
        print(f"\n{'='*90}\n")
    
    def print_node_breakdown(self, component: str, warmup_steps: int = 0, top_n: int = 20):
        """Print top nodes for a specific component"""
        if self.rank != 0:
            return
        
        print(f"\n[{component} Node Breakdown (Top {top_n})]")
        print(f"{'Node Name':<50} {'Mean(ms)':<12} {'Calls':<8}")
        print(f"{'-'*75}")
        
        nodes = []
        for node_name, times in self.node_times.items():
            if self.node_to_component.get(node_name) == component:
                measured = times[warmup_steps:] if len(times) > warmup_steps else times
                if measured:
                    mean_val = sum(measured) / len(measured)
                    nodes.append((node_name, mean_val, len(measured)))
        
        # Sort by time descending
        nodes.sort(key=lambda x: x[1], reverse=True)
        
        for name, mean_ms, calls in nodes[:top_n]:
            print(f"{name[:50]:<50} {mean_ms:<12.4f} {calls:<8}")
        
        print()


# ============================================================================
# Layer Block Profiler (FULL decoder layer: embedding_end → next_q_proj_start)
# ============================================================================
class LayerBlockProfiler:
    """
    Measures FULL transformer decoder layer time using boundary-based approach.
    
    FX Graph에서 layernorm/residual은 call_function이므로 직접 hook 불가.
    대신 q_proj를 경계로 사용하여 FULL 레이어 시간을 측정:
    
    Layer 0:
      START: embedding post_hook (embedding_end)
      END:   layer_1 q_proj pre_hook (또는 lm_head pre_hook if single layer)
      포함: input_layernorm + attn + 1st_residual + post_attn_norm + mlp + 2nd_residual
    
    Layer N (N>0, not last):
      START: layer_N q_proj pre_hook
      END:   layer_(N+1) q_proj pre_hook
      
    Last Layer:
      START: layer_N q_proj pre_hook
      END:   lm_head pre_hook
      포함: 동일 + model_norm (final norm before lm_head)
    
    NOTE: Times are accumulated per-step.
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
        
        # Timing state for boundary-based measurement
        self.embedding_start = None
        self.embedding_end_event = None  # Layer 0 start boundary
        self.layer_q_proj_events = {}    # layer_idx -> q_proj start event
        self.lm_head_start_event = None  # Last layer end boundary
        
        # Track layer indices
        self.layer_indices = set()
        self.min_layer_idx = None  # First layer (starts from embedding_end)
        self.max_layer_idx = None  # Last layer (ends at lm_head_start)
        
        # Deferred event pairs (will be processed at step_end)
        self.pending_events = []  # [(key, start_event, end_event), ...]
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for boundary-based FULL layer timing"""
        import re
        
        # First pass: find all layer indices
        for name, module in self.submod.named_modules():
            name_lower = name.lower()
            if 'self_attn_q_proj' in name_lower or 'self_attn.q_proj' in name_lower:
                match = re.search(r'layers?[_.]?(\d+)', name_lower)
                if match:
                    self.layer_indices.add(int(match.group(1)))
        
        if self.layer_indices:
            self.min_layer_idx = min(self.layer_indices)
            self.max_layer_idx = max(self.layer_indices)
        
        # Second pass: register hooks
        registered_q_proj = set()
        
        for name, module in self.submod.named_modules():
            name_lower = name.lower()
            
            # Embedding: record end as layer_0 start boundary
            if 'embed_tokens' in name_lower or name_lower == 'model_embed_tokens':
                self._register_embedding_hooks(name, module)
            
            # LM Head: pre_hook records last layer end boundary
            elif 'lm_head' in name_lower:
                self._register_lm_head_hooks(name, module)
            
            # q_proj: records layer boundaries
            elif 'self_attn_q_proj' in name_lower or 'self_attn.q_proj' in name_lower:
                match = re.search(r'layers?[_.]?(\d+)', name_lower)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in registered_q_proj:
                        self._register_q_proj_hook(name, module, layer_idx)
                        registered_q_proj.add(layer_idx)
            
            # down_proj: fallback for single-layer stages (PP scenario)
            elif 'mlp_down_proj' in name_lower or 'mlp.down_proj' in name_lower:
                match = re.search(r'layers?[_.]?(\d+)', name_lower)
                if match:
                    layer_idx = int(match.group(1))
                    # Only register if this is the max layer and it equals min (single layer stage)
                    if self.max_layer_idx is not None and layer_idx == self.max_layer_idx:
                        if self.min_layer_idx == self.max_layer_idx:
                            self._register_down_proj_fallback(name, module, layer_idx)
        
        if self.rank == 0:
            print(f"[LayerBlockProfiler] FULL LAYER PROFILING (boundary-based)")
            print(f"[LayerBlockProfiler] Layer 0: embedding_end → next_q_proj_start")
            print(f"[LayerBlockProfiler] Layer N: q_proj_start → next_q_proj_start")  
            print(f"[LayerBlockProfiler] Last Layer: q_proj_start → lm_head_start")
            print(f"[LayerBlockProfiler] Includes: input_layernorm + attn + 1st_residual + post_attn_norm + mlp + 2nd_residual")
            print(f"[LayerBlockProfiler] Registered layers: {sorted(self.layer_indices)}")
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
                profiler.pending_events.append(('embedding', profiler.embedding_start, end_event))
                profiler.embedding_end_event = end_event
                profiler.embedding_start = None
        
        self.hooks.append(module.register_forward_pre_hook(pre_hook))
        self.hooks.append(module.register_forward_hook(post_hook))
        if self.rank == 0:
            print(f"[LayerBlockProfiler] Embedding hooks: {name}")
    
    def _register_lm_head_hooks(self, name, module):
        """Register pre/post hooks for lm_head (pre_hook serves as last layer end)"""
        profiler = self
        
        def pre_hook(mod, inp):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            profiler.lm_head_start_event = start_event
            
            if profiler.max_layer_idx is not None:
                last_layer_key = f"layer_{profiler.max_layer_idx}"
                last_start = profiler.layer_q_proj_events.get(profiler.max_layer_idx)
                if last_start is not None:
                    profiler.pending_events.append((last_layer_key, last_start, start_event))
        
        def post_hook(mod, inp, out):
            if profiler.lm_head_start_event:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                profiler.pending_events.append(('lm_head', profiler.lm_head_start_event, end_event))
                profiler.lm_head_start_event = None
        
        self.hooks.append(module.register_forward_pre_hook(pre_hook))
        self.hooks.append(module.register_forward_hook(post_hook))
        if self.rank == 0:
            print(f"[LayerBlockProfiler] LM Head hooks: {name}")
    
    def _register_q_proj_hook(self, name, module, layer_idx):
        """Register pre_hook on q_proj for boundary-based layer timing."""
        profiler = self
        is_first_layer = (layer_idx == self.min_layer_idx)
        is_last_layer = (layer_idx == self.max_layer_idx)
        
        def pre_hook(mod, inp):
            current_event = torch.cuda.Event(enable_timing=True)
            current_event.record()
            
            # For first layer: use embedding_end as start
            if is_first_layer and profiler.embedding_end_event is not None:
                profiler.layer_q_proj_events[layer_idx] = profiler.embedding_end_event
            
            # For non-first layers: measure previous layer
            if not is_first_layer:
                prev_layer_idx = layer_idx - 1
                prev_layer_key = f"layer_{prev_layer_idx}"
                prev_start = profiler.layer_q_proj_events.get(prev_layer_idx)
                if prev_start is not None:
                    profiler.pending_events.append((prev_layer_key, prev_start, current_event))
            
            profiler.layer_q_proj_events[layer_idx] = current_event
        
        self.hooks.append(module.register_forward_pre_hook(pre_hook))
        if self.rank == 0:
            layer_key = f"layer_{layer_idx}"
            if is_first_layer:
                print(f"[LayerBlockProfiler] {layer_key}: embedding_end → next ({name})")
            else:
                print(f"[LayerBlockProfiler] {layer_key}: prev_q_proj → {name}")
        
        # For last layer on this stage: also register down_proj fallback
        # (in case there's no next q_proj on the same stage, PP scenario)
        if is_last_layer and self.min_layer_idx == self.max_layer_idx:
            # Single layer stage: need down_proj fallback
            self._need_down_proj_fallback = True
    
    def _register_down_proj_fallback(self, name, module, layer_idx):
        """Fallback for last layer when next q_proj is on different stage."""
        profiler = self
        layer_key = f"layer_{layer_idx}"
        
        def post_hook(mod, inp, out):
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            
            start = profiler.layer_q_proj_events.get(layer_idx)
            if start is not None:
                profiler.pending_events.append((layer_key, start, end_event))
        
        self.hooks.append(module.register_forward_hook(post_hook))
        if self.rank == 0:
            print(f"[LayerBlockProfiler] {layer_key}: fallback down_proj end ({name})")
    
    def step_end(self):
        """Called at end of each training step to finalize measurements"""
        torch.cuda.synchronize()
        
        for key, start_event, end_event in self.pending_events:
            elapsed = start_event.elapsed_time(end_event)
            self.current_step_times[key] += elapsed
        self.pending_events.clear()
        
        self.embedding_end_event = None
        self.lm_head_start_event = None
        self.layer_q_proj_events.clear()
        
        for key, value in self.current_step_times.items():
            self.step_times[key].append(value)
        self.current_step_times.clear()
    
    def get_summary(self, warmup_steps: int = 0) -> dict:
        """Get timing summary (excluding warmup, per-forward-pass averages)"""
        summary = {}
        
        for key, times in self.step_times.items():
            measured = times[warmup_steps:] if len(times) > warmup_steps else times
            if measured:
                per_fwd_times = [t / self.num_mb for t in measured]
                sorted_times = sorted(per_fwd_times)
                mean_val = sum(per_fwd_times) / len(per_fwd_times)
                median_val = sorted_times[len(sorted_times) // 2]
                summary[key] = {
                    'mean_ms': mean_val,
                    'median_ms': median_val,
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
        total_time = sum(v.get('median_ms', v['mean_ms']) for v in summary.values())
        transformer_layers = {k: v for k, v in summary.items() if k.startswith('layer_')}
        transformer_total = sum(v['mean_ms'] for v in transformer_layers.values())
        measured_steps = list(summary.values())[0].get('measured_steps', 0) if summary else 0
        
        print(f"\n{'='*90}")
        print(f" Layer Block Profile (Stage {self.stage}, Rank {self.rank})")
        print(f" Measures: FULL DecoderLayer (embedding_end → next_q_proj or lm_head)")
        print(f" Includes: input_layernorm + attn + 1st_res + post_attn_norm + mlp + 2nd_res")
        print(f" Warmup: {warmup_steps} steps, Measured: {measured_steps} steps")
        print(f" Values: per-forward-pass average (num_mb={self.num_mb})")
        print(f"{'='*90}")
        
        print(f"\n[Summary - Per Forward Pass]")
        print(f"{'Component':<20} {'Median(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Mean(ms)':<12} {'%':<8}")
        print(f"{'-'*80}")
        
        if 'embedding' in summary:
            s = summary['embedding']
            pct = (s['median_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"{'Embedding':<20} {s['median_ms']:<12.4f} {s['min_ms']:<12.4f} {s['max_ms']:<12.4f} {s['mean_ms']:<12.4f} {pct:<8.2f}")
        
        if transformer_layers:
            trans_median_total = sum(v.get('median_ms', v['mean_ms']) for v in transformer_layers.values())
            trans_pct = (trans_median_total / total_time * 100) if total_time > 0 else 0
            print(f"{'Transformer (total)':<20} {trans_median_total:<12.4f} {'-':<12} {'-':<12} {transformer_total:<12.4f} {trans_pct:<8.2f}")
        
        if 'lm_head' in summary:
            s = summary['lm_head']
            pct = (s['median_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"{'LM Head':<20} {s['median_ms']:<12.4f} {s['min_ms']:<12.4f} {s['max_ms']:<12.4f} {s['mean_ms']:<12.4f} {pct:<8.2f}")
        
        print(f"{'-'*80}")
        print(f"{'TOTAL':<20} {total_time:<12.4f}")
        
        if transformer_layers:
            print(f"\n[Per-Layer Breakdown (FULL: input_layernorm → 2nd_residual)]")
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
        
        # ================================================================
        # Simple boundary-based profiling (start/end only, all ranks)
        # Uses LayerBlockProfiler which measures:
        #   - Embedding: embed_tokens module
        #   - Layer N: from embedding_end/prev_layer_end to next_layer_start/lm_head_start  
        #   - LM Head: lm_head module
        # ================================================================
        if args.profile_fx:
            log(f"[{ts()}] Using Simple Boundary Profiling (start/end only)") if rank == 0 else None
            
            # LayerBlockProfiler measures FULL layer time via boundary hooks
            block_profiler = LayerBlockProfiler(
                optimus_p.run_info.submod,
                optimus_p.run_info.device,
                rank,
                optimus_p.tpl.stage,
                num_mb=num_mb
            )
            
            # Use optimizer for realistic timing
            if args.tp_size > 1:
                optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5, foreach=False)
            else:
                optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5)
            
            # Load dataset
            datasets = load_dataset("squad").data["train"]["context"]
            datasets = [str(record) for record in datasets if len(str(record)) < 500]
            dataloader = optimus_p.prepare_dataloader(datasets, batch_size)
            
            log(f"[rank:{rank}] Simple Profile: {args.profile_steps} steps ({args.profile_warmup_steps} warmup)")
            
            # Profile loop (actual training execution)
            tick = time.time()
            step_count = 0
            
            for batch in dataloader:
                if step_count >= args.profile_steps:
                    break
                
                data, labels = None, None
                if optimus_p.is_first_stage():
                    tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    data, labels = tokens.input_ids, tokens.input_ids
                
                labels = optimus_p.move_labels2last_stage(labels)
                optimus_p.optimizer.zero_grad()
                
                # Forward + Backward (actual execution for realistic timing)
                optimus_p.run(data, labels, mode="1f1b")
                
                if args.tp_size == 1:
                    torch.nn.utils.clip_grad_norm_(optimus_p.parameters(), 0.5)
                
                optimus_p.optimizer.step()
                block_profiler.step_end()
                
                step_count += 1
                if rank == 0 and step_count % 5 == 0:
                    log(f"[rank:0] Simple Profile step {step_count}/{args.profile_steps}")
            
            tock = time.time()
            
            # Synchronize all ranks
            dist.barrier()
            
            # Print per-rank summary
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
                        'num_layers': num_layers,
                        'batch_size': batch_size,
                        'micro_batch_size': micro_batch_size,
                        'pp_size': args.pp_size,
                        'tp_size': args.tp_size,
                        'dp_size': args.dp_size,
                        'profile_steps': args.profile_steps,
                        'warmup_steps': args.profile_warmup_steps,
                        'total_time_sec': tock - tick,
                        'method': 'simple_boundary (start/end only)'
                    },
                    'combined_timing': combined_result,
                }
                
                output_path = args.profile_output if args.profile_output else f"{RESULT_DIR}/profile_{args.run_id}.json"
                save_profile_result(profile_result, output_path)
                ELAPSED_TIME = tock - tick
            
            block_profiler.remove_hooks()
            EXIT_CODE = 0
        
        # ================================================================
        # Hook-based profiling (during normal training execution)
        # ================================================================
        else:
            # Setup profilers
            # 1. LayerProfiler: measures individual module times (sum of all ops)
            profiler = LayerProfiler(
                optimus_p.run_info.submod,
                optimus_p.run_info.device,
                rank, 
                optimus_p.tpl.stage,
                num_mb=num_mb  # Pass num_mb for per-forward-pass normalization
            )
            
            # 2. LayerBlockProfiler: measures layer block times (boundary-based, END-TO-END)
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
    # TRAINING MODE (when profile_mode is False)
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
        
        def train_epoch():
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
                
                if optimus_p.is_last_stage() and loss is not None:
                    loss_val = sum(loss) / optimus_p.mbsize
                    total_loss += loss_val
                    log_interval = 10
                    if i % log_interval == 0 and i > 0:
                        cur_loss = total_loss / log_interval
                        elapsed = time.time() - start_time
                        if optimus_p.get_rank() % int(world_size / args.pp_size) == 0:
                            print(f"===== {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
                            print(f'| epoch {epoch:3d} | {i:5d}/{nbatches:5d} batches | '
                                  f'lr {scheduler.get_lr()[0]:02.2f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | '
                                  f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
                        total_loss = 0
                        start_time = time.time()
        
        if optimus_p.get_rank() == 0:
            tick = time.time()
        
        for epoch in range(1, epochs + 1):
            train_epoch()
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
