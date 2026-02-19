"""
Debug attention behavior in incremental mode.
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_sdpa_causal():
    """Test SDPA causal behavior with cached keys."""
    print("=" * 70)
    print("SDPA Causal Mask Test")
    print("=" * 70)
    
    batch = 1
    num_heads = 2
    head_dim = 32
    
    # Simulate: 5 cached positions + 1 new position
    q_len = 1
    kv_len = 6  # 5 cached + 1 new
    
    q = torch.randn(batch, num_heads, q_len, head_dim)
    k_full = torch.randn(batch, num_heads, kv_len, head_dim)
    v_full = torch.randn(batch, num_heads, kv_len, head_dim)
    
    print(f"\nQuery shape: {q.shape} (position 5)")
    print(f"Key/Value shape: {k_full.shape} (positions 0-5)")
    
    # Test with is_causal=True
    out_causal = F.scaled_dot_product_attention(
        q, k_full, v_full, is_causal=True
    )
    
    # Test with explicit mask
    # For query at position 5, it should attend to keys at positions 0-5
    mask = torch.zeros(q_len, kv_len, dtype=torch.bool)
    # No masking needed if query is at the last position
    
    out_masked = F.scaled_dot_product_attention(
        q, k_full, v_full, attn_mask=mask, is_causal=False
    )
    
    print(f"\nOutput with is_causal=True: {out_causal[0, 0, 0, :5]}")
    print(f"Output with explicit mask: {out_masked[0, 0, 0, :5]}")
    
    diff = (out_causal - out_masked).abs().max()
    print(f"\nDifference: {diff:.6e}")


def test_attention_equivalence():
    """Test if attention produces same output with different KV ordering."""
    print("\n" + "=" * 70)
    print("Attention Equivalence Test")
    print("=" * 70)
    
    from diffulex_edge.model.fast_dllm_v2_edge import AttentionEdge, RotaryEmbedding, FastdLLMV2EdgeConfig
    
    config = FastdLLMV2EdgeConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    
    attn = AttentionEdge(
        hidden_size=64,
        num_heads=2,
        num_kv_heads=2,
        head_dim=32,
    )
    attn.eval()
    
    rope = RotaryEmbedding(dim=32, max_position_embeddings=128)
    
    # Method 1: Full sequence (positions 0-5)
    hidden_full = torch.randn(1, 6, 64)
    positions_full = torch.arange(6).unsqueeze(0)
    
    with torch.no_grad():
        out_full, _, _ = attn(
            hidden_full, positions_full, rope,
            k_cache=None, v_cache=None, start_pos=0
        )
    
    # Method 2: Incremental (first 5, then 1)
    hidden_5 = hidden_full[:, :5, :]
    positions_5 = torch.arange(5).unsqueeze(0)
    
    # Create dummy cache
    k_cache = torch.zeros(1, 2, 128, 32)
    v_cache = torch.zeros(1, 2, 128, 32)
    
    with torch.no_grad():
        out_5, new_k, new_v = attn(
            hidden_5, positions_5, rope,
            k_cache=k_cache, v_cache=v_cache, start_pos=0
        )
        
        # Update cache
        k_cache[:, :, :5, :] = new_k
        v_cache[:, :, :5, :] = new_v
        
        # Decode 6th
        hidden_6 = hidden_full[:, 5:6, :]
        positions_6 = torch.tensor([[5]])
        
        out_6, _, _ = attn(
            hidden_6, positions_6, rope,
            k_cache=k_cache, v_cache=v_cache, start_pos=5
        )
    
    print(f"\nFull sequence output (position 5): {out_full[0, 5, :5]}")
    print(f"Incremental output (position 5): {out_6[0, 0, :5]}")
    
    diff = (out_full[0, 5:6, :] - out_6).abs()
    print(f"\nDifference: max={diff.max():.6e}, mean={diff.mean():.6e}")
    
    if diff.max() < 1e-4:
        print("[PASS] Attention outputs match (within FP32 precision)")
    elif diff.max() < 1e-3:
        print("[WARN] Small difference (acceptable)")
    else:
        print("[FAIL] Significant difference - investigate")


def debug_forward_pass():
    """Debug full forward pass differences."""
    print("\n" + "=" * 70)
    print("Full Forward Pass Debug")
    print("=" * 70)
    
    from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
    from diffulex_edge.model.kv_cache import KVCacheConfig, create_kv_caches
    
    config = FastdLLMV2EdgeConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,  # Single layer for debugging
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    
    torch.manual_seed(42)
    model = FastdLLMV2Edge(config)
    model.eval()
    
    input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]])
    
    # Full pass
    with torch.no_grad():
        logits_full, _ = model(input_ids)
    
    # Split pass
    kv_config = KVCacheConfig(
        num_layers=1, num_kv_heads=2, head_dim=32, max_seq_len=128
    )
    kv_cache = create_kv_caches(kv_config, batch_size=1)
    
    with torch.no_grad():
        # First 5
        logits_5, kv_5 = model(
            input_ids[:, :5],
            kv_cache=kv_cache.get_cache_tensor(),
            start_pos=0
        )
        
        # Update cache
        full_cache = kv_cache.get_cache_tensor()
        full_cache[:, :, :, :, :5, :] = kv_5
        
        # 6th
        logits_6, _ = model(
            input_ids[:, 5:6],
            kv_cache=full_cache,
            start_pos=5
        )
    
    print(f"\nFull logits (pos 5): {logits_full[0, 5, :10]}")
    print(f"Split logits (pos 5): {logits_6[0, 0, :10]}")
    
    # Check intermediate values
    print("\nDebugging intermediate values...")
    
    # Compare embeddings
    emb_full = model.embed_tokens(input_ids)
    emb_5 = model.embed_tokens(input_ids[:, :5])
    emb_6 = model.embed_tokens(input_ids[:, 5:6])
    
    print(f"\nEmbedding [0,0,:5] full: {emb_full[0, 0, :5]}")
    print(f"Embedding [0,0,:5] split-5: {emb_5[0, 0, :5]}")
    print(f"Embedding [0,0,:5] split-6: {emb_6[0, 0, :5]}")
    
    diff_emb = (emb_full[0, 0, :5] - emb_5[0, 0, :5]).abs().max()
    print(f"Embedding diff (should be 0): {diff_emb:.6e}")


if __name__ == "__main__":
    test_sdpa_causal()
    test_attention_equivalence()
    debug_forward_pass()
