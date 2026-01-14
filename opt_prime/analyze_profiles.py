#!/usr/bin/env python3
"""
Analyze layer profile numpy files and compare with tutoruslabs results.
"""
import numpy as np
import os
import glob

PROFILE_DIR = "_profiles"

def load_and_print_profile(filepath):
    """Load and print layer profile from numpy file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    data = np.load(filepath)
    filename = os.path.basename(filepath)
    
    print(f"\n{'='*60}")
    print(f" Profile: {filename}")
    print(f"{'='*60}")
    print(f" Array shape: {data.shape}")
    print(f" Total entries: {len(data)}")
    
    # Format: [embed, layer0, layer1, ..., layerN-1, lm_head]
    embed_time = data[0]
    lm_head_time = data[-1]
    layer_times = data[1:-1]
    
    print(f"\n Component           Time (ms)")
    print(f" {'-'*40}")
    print(f" Embedding           {embed_time:.4f}")
    print(f" Transformer (sum)   {layer_times.sum():.4f}")
    print(f"   - Avg per layer   {layer_times.mean():.4f}")
    print(f"   - Min             {layer_times.min():.4f}")
    print(f"   - Max             {layer_times.max():.4f}")
    print(f" LM Head             {lm_head_time:.4f}")
    print(f" {'-'*40}")
    print(f" TOTAL               {data.sum():.4f}")
    
    # Print per-layer breakdown
    print(f"\n Layer Breakdown:")
    for i, t in enumerate(layer_times):
        print(f"   Layer {i:2d}: {t:.4f} ms")
    
    return {
        'embed': embed_time,
        'layers': layer_times,
        'lm_head': lm_head_time,
        'total': data.sum()
    }


def main():
    print("="*60)
    print(" Llama-3.2-1B Layer Profile Analysis")
    print("="*60)
    
    # Find all profile files
    profile_files = glob.glob(os.path.join(PROFILE_DIR, "*.npy"))
    
    if not profile_files:
        print(f"No profile files found in {PROFILE_DIR}/")
        return
    
    print(f"\nFound {len(profile_files)} profile files:")
    for f in sorted(profile_files):
        print(f"  - {os.path.basename(f)}")
    
    # Load and analyze each profile
    all_results = {}
    for filepath in sorted(profile_files):
        result = load_and_print_profile(filepath)
        if result:
            all_results[os.path.basename(filepath)] = result
    
    # Print comparison with tutoruslabs results
    print(f"\n{'='*60}")
    print(" Comparison with tutoruslabs Profiling Results")
    print(" (Llama-3.2-1B, PP=2, TP=1, 2 layers)")
    print("="*60)
    print(f"""
 tutoruslabs Results (Median values):
   - Embedding:    0.25 ms
   - Transformer:  3.29 ms (per layer)
   - LM Head:      4.45 ms
   
 opt_prime Results (from this analysis):
   - See above per-stage profiles
   
 NOTE: The difference may be due to:
   1. Different TP setting (TP=2 vs TP=1)
   2. Different PP setting (PP=4 vs PP=2)  
   3. Different measurement methodology
   4. Different number of layers (16 vs 2)
""")


if __name__ == "__main__":
    main()

