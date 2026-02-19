"""
Diagnostic script to analyze numerical precision differences
between full sequence and incremental inference.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diffulex_edge.model.fast_dllm_v2_edge import (
    FastdLLMV2Edge,
    FastdLLMV2EdgeConfig,
)
from diffulex_edge.model.kv_cache import KVCacheConfig, create_kv_caches


def analyze_precision():
    """Analyze numerical differences."""
    print("=" * 70)
    print("Numerical Precision Analysis")
    print("=" * 70)
    
    config = FastdLLMV2EdgeConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    
    model = FastdLLMV2Edge(config)
    model.eval()
    
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    full_ids = torch.randint(0, config.vocab_size, (1, 6))
    
    print(f"\nTest sequence: {full_ids}")
    
    # Full sequence inference
    with torch.no_grad():
        full_logits, _ = model(full_ids)
    
    # Incremental inference
    kv_config = KVCacheConfig(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=config.max_position_embeddings,
    )
    kv_cache = create_kv_caches(kv_config, batch_size=1)
    
    with torch.no_grad():
        # Prefill first 5
        prefill_logits, prefill_kv = model(
            full_ids[:, :5],
            kv_cache=kv_cache.get_cache_tensor(),
            start_pos=0,
        )
        
        # Update cache
        full_cache = kv_cache.get_cache_tensor()
        full_cache[:, :, :, :, :5, :] = prefill_kv
        
        # Decode 6th
        decode_logits, _ = model(
            full_ids[:, 5:6],
            kv_cache=full_cache,
            start_pos=5,
        )
    
    # Compare 6th position
    full_6th = full_logits[:, 5:6, :]
    decode_6th = decode_logits
    
    print("\n" + "-" * 70)
    print("Comparison Results")
    print("-" * 70)
    
    # Basic stats
    diff = (full_6th - decode_6th).abs()
    
    print(f"\nFull sequence logits (6th token):")
    print(f"  Mean: {full_6th.mean():.6f}")
    print(f"  Std:  {full_6th.std():.6f}")
    print(f"  Min:  {full_6th.min():.6f}")
    print(f"  Max:  {full_6th.max():.6f}")
    
    print(f"\nIncremental logits (6th token):")
    print(f"  Mean: {decode_6th.mean():.6f}")
    print(f"  Std:  {decode_6th.std():.6f}")
    print(f"  Min:  {decode_6th.min():.6f}")
    print(f"  Max:  {decode_6th.max():.6f}")
    
    print(f"\nDifference statistics:")
    print(f"  Max absolute error: {diff.max():.6e}")
    print(f"  Mean absolute error: {diff.mean():.6e}")
    print(f"  Median absolute error: {diff.median():.6e}")
    print(f"  Std of error: {diff.std():.6e}")
    
    # Relative error
    relative_error = diff / (full_6th.abs() + 1e-8)
    print(f"\nRelative error:")
    print(f"  Max relative error: {relative_error.max():.6e}")
    print(f"  Mean relative error: {relative_error.mean():.6e}")
    
    # Check if error is within reasonable bounds for float32
    print(f"\nFloat32 machine epsilon: {torch.finfo(torch.float32).eps:.6e}")
    print(f"Error relative to epsilon: {diff.max() / torch.finfo(torch.float32).eps:.1f}x")
    
    # Top-k analysis
    print("\n" + "-" * 70)
    print("Top-k Predictions Comparison")
    print("-" * 70)
    
    for k in [1, 3, 5, 10]:
        full_topk = full_6th.topk(k, dim=-1).indices
        decode_topk = decode_6th.topk(k, dim=-1).indices
        
        overlap = (full_topk == decode_topk).any(dim=-1).float().mean()
        print(f"  Top-{k} overlap: {overlap * 100:.1f}%")
    
    # Sample some specific values
    print("\n" + "-" * 70)
    print("Sample Values (first 10 logits)")
    print("-" * 70)
    print(f"{'Index':<8} {'Full':<15} {'Incremental':<15} {'Diff':<15}")
    for i in range(min(10, full_6th.shape[-1])):
        f = full_6th[0, 0, i].item()
        d = decode_6th[0, 0, i].item()
        diff_val = abs(f - d)
        print(f"{i:<8} {f:<15.6f} {d:<15.6f} {diff_val:<15.6e}")
    
    # Determine if error is acceptable
    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)
    
    max_err = diff.max().item()
    eps = torch.finfo(torch.float32).eps
    
    if max_err < 1e-4:
        print(f"✅ EXCELLENT: Max error {max_err:.6e} is very small")
    elif max_err < 1e-3:
        print(f"✅ GOOD: Max error {max_err:.6e} is acceptable")
    elif max_err < 1e-2:
        print(f"⚠️  MODERATE: Max error {max_err:.6e} - may need investigation")
    else:
        print(f"❌ HIGH: Max error {max_err:.6e} - potential bug")
    
    print(f"\nCompared to float32 epsilon ({eps:.6e}):")
    ratio = max_err / eps
    if ratio < 100:
        print(f"  Error is {ratio:.1f}x epsilon - likely normal FP precision")
    elif ratio < 10000:
        print(f"  Error is {ratio:.1f}x epsilon - moderate, acceptable for this use case")
    else:
        print(f"  Error is {ratio:.1f}x epsilon - investigate for potential bug")


def compare_prefill_consistency():
    """Compare prefill with cache vs without cache."""
    print("\n\n" + "=" * 70)
    print("Prefill Consistency Test")
    print("=" * 70)
    
    config = FastdLLMV2EdgeConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    
    model = FastdLLMV2Edge(config)
    model.eval()
    
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # Without cache
    with torch.no_grad():
        logits_no_cache, _ = model(input_ids)
    
    # With cache (start_pos=0, no previous cache)
    kv_config = KVCacheConfig(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=config.max_position_embeddings,
    )
    kv_cache = create_kv_caches(kv_config, batch_size=1)
    
    with torch.no_grad():
        logits_with_cache, _ = model(
            input_ids,
            kv_cache=kv_cache.get_cache_tensor(),
            start_pos=0,
        )
    
    diff = (logits_no_cache - logits_with_cache).abs()
    
    print(f"\nPrefill with empty cache vs no cache:")
    print(f"  Max error: {diff.max():.6e}")
    print(f"  Mean error: {diff.mean():.6e}")
    
    if diff.max() < 1e-5:
        print(f"  ✅ EXCELLENT: Essentially identical")
    elif diff.max() < 1e-4:
        print(f"  ✅ GOOD: Very small difference")
    else:
        print(f"  ⚠️  NOTICEABLE: Difference may accumulate")


if __name__ == "__main__":
    analyze_precision()
    compare_prefill_consistency()
