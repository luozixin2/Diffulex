"""Test that forward_export matches forward output.

This script verifies numerical equivalence between:
1. forward() - dynamic KV cache list (for Python inference)
2. forward_export() - fixed-shape KV cache tensor (for ExecuTorch export)
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from diffulex_edge.model.sdar_edge import SDAREdge, SDAREdgeConfig


def create_test_config():
    """Create a small config for testing."""
    return SDAREdgeConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        head_dim=32,
        diffusion_block_size=4,
    )


def kv_list_to_tensor(kv_list, max_seq_len):
    """Convert dynamic KV cache list to fixed-shape tensor.
    
    Args:
        kv_list: List of (k, v) tuples per layer
        max_seq_len: Maximum sequence length for the tensor
        
    Returns:
        kv_tensor: [num_layers, 2, batch, num_kv_heads, max_seq_len, head_dim]
        cache_len: number of valid tokens in the cache
    """
    if kv_list is None or len(kv_list) == 0:
        return None, 0
    
    num_layers = len(kv_list)
    batch, num_kv_heads, seq_len, head_dim = kv_list[0][0].shape
    
    kv_tensor = torch.zeros(num_layers, 2, batch, num_kv_heads, max_seq_len, head_dim,
                           dtype=kv_list[0][0].dtype, device=kv_list[0][0].device)
    
    for i, (k, v) in enumerate(kv_list):
        actual_len = min(seq_len, max_seq_len)
        kv_tensor[i, 0, :, :, :actual_len, :] = k[:, :, :actual_len, :]
        kv_tensor[i, 1, :, :, :actual_len, :] = v[:, :, :actual_len, :]
    
    return kv_tensor, seq_len


def compare_tensors(name, t1, t2, atol=1e-5, rtol=1e-5):
    """Compare two tensors and print results."""
    max_diff = torch.max(torch.abs(t1 - t2)).item()
    mean_diff = torch.mean(torch.abs(t1 - t2)).item()
    
    is_close = torch.allclose(t1, t2, atol=atol, rtol=rtol)
    status = "✓ PASS" if is_close else "✗ FAIL"
    
    print(f"  {name}: {status}")
    print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    
    return is_close


def test_forward_match():
    """Test that forward and forward_export produce identical outputs."""
    print("=" * 60)
    print("Testing forward_export vs forward equivalence")
    print("=" * 60)
    
    # Create model
    config = create_test_config()
    model = SDAREdge(config)
    model.eval()
    
    print(f"\nModel config:")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Heads: {config.num_attention_heads} / KV heads: {config.num_key_value_heads}")
    print(f"  Head dim: {config.head_dim}")
    
    # Test parameters
    batch_size = 2
    seq_len = 4
    max_seq_len = 16
    
    # Create test inputs
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    print(f"\nTest input:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Max cache length: {max_seq_len}")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Positions shape: {positions.shape}")
    
    # Test 1: Forward without cache
    print("\n" + "-" * 40)
    print("Test 1: Forward without cache (cache_len=0)")
    print("-" * 40)
    
    # Empty KV cache list for forward
    empty_kv_list = []
    
    with torch.no_grad():
        logits_forward, kv_list_out = model.forward(input_ids, positions, kv_cache=empty_kv_list, max_seq_len=max_seq_len)
    
    # Empty KV cache tensor for forward_export
    kv_cache_empty = torch.zeros(
        config.num_hidden_layers, 2, batch_size, config.num_key_value_heads, 
        max_seq_len, config.head_dim
    )
    
    with torch.no_grad():
        logits_export, kv_cache_out = model.forward_export(
            input_ids, positions, kv_cache_empty, cache_len=0
        )
    
    # Compare logits
    match1 = compare_tensors("Logits", logits_forward, logits_export, atol=1e-5, rtol=1e-5)
    
    # Compare top-1 tokens
    top1_forward = torch.argmax(logits_forward, dim=-1)
    top1_export = torch.argmax(logits_export, dim=-1)
    top1_match = torch.equal(top1_forward, top1_export)
    print(f"  Top-1 tokens match: {'✓ YES' if top1_match else '✗ NO'}")
    
    # Test 2: Forward with pre-filled cache
    print("\n" + "-" * 40)
    print("Test 2: Forward with pre-filled cache")
    print("-" * 40)
    
    # First, run one forward to get cache (using cache_len=0)
    torch.manual_seed(123)
    input_ids_first = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    positions_first = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        _, kv_list_prefill = model.forward(input_ids_first, positions_first, kv_cache=[], max_seq_len=max_seq_len)
    
    # Convert to tensor format
    kv_cache_prefill, cache_len = kv_list_to_tensor(kv_list_prefill, max_seq_len)
    print(f"  Prefill cache_len: {cache_len}")
    
    # New tokens to generate
    new_seq_len = 2
    input_ids_second = torch.randint(0, config.vocab_size, (batch_size, new_seq_len))
    positions_second = torch.arange(seq_len, seq_len + new_seq_len).unsqueeze(0).expand(batch_size, -1)
    
    print(f"  New tokens: {new_seq_len}")
    print(f"  Positions: {positions_second[0].tolist()}")
    
    # Forward with cache
    with torch.no_grad():
        logits_forward2, kv_list2 = model.forward(
            input_ids_second, positions_second, kv_cache=kv_list_prefill, max_seq_len=max_seq_len
        )
    
    # Forward export with cache
    with torch.no_grad():
        logits_export2, kv_cache_out2 = model.forward_export(
            input_ids_second, positions_second, kv_cache_prefill, cache_len=cache_len
        )
    
    # Compare
    match2 = compare_tensors("Logits", logits_forward2, logits_export2, atol=1e-5, rtol=1e-5)
    
    # Compare top-1 tokens
    top1_forward2 = torch.argmax(logits_forward2, dim=-1)
    top1_export2 = torch.argmax(logits_export2, dim=-1)
    top1_match2 = torch.equal(top1_forward2, top1_export2)
    print(f"  Top-1 tokens match: {'✓ YES' if top1_match2 else '✗ NO'}")
    
    # Test 3: Multi-step generation consistency
    print("\n" + "-" * 40)
    print("Test 3: Multi-step generation consistency")
    print("-" * 40)
    
    num_steps = 5
    tokens_match_all = True
    logits_match_all = True
    
    # Reset
    kv_list_dyn = []
    kv_cache_tensor = torch.zeros(
        config.num_hidden_layers, 2, batch_size, config.num_key_value_heads, 
        max_seq_len, config.head_dim
    )
    cache_len_tensor = 0
    
    prompt = torch.randint(0, config.vocab_size, (batch_size, 1))
    current_pos = 0
    
    for step in range(num_steps):
        positions_step = torch.tensor([[current_pos]], dtype=torch.long).expand(batch_size, 1)
        
        # Dynamic forward
        with torch.no_grad():
            logits_dyn, kv_list_dyn = model.forward(
                prompt, positions_step, kv_cache=kv_list_dyn, max_seq_len=max_seq_len
            )
        
        # Export forward
        with torch.no_grad():
            logits_exp, kv_cache_tensor = model.forward_export(
                prompt, positions_step, kv_cache_tensor, cache_len=cache_len_tensor
            )
        
        # Compare
        step_logits_match = torch.allclose(logits_dyn, logits_exp, atol=1e-4, rtol=1e-4)
        token_dyn = torch.argmax(logits_dyn[:, -1, :], dim=-1)
        token_exp = torch.argmax(logits_exp[:, -1, :], dim=-1)
        step_token_match = torch.equal(token_dyn, token_exp)
        
        if not step_logits_match:
            logits_match_all = False
        if not step_token_match:
            tokens_match_all = False
            
        status = "✓ PASS" if (step_logits_match and step_token_match) else "✗ FAIL"
        print(f"  Step {step + 1}: {status} (logits: {step_logits_match}, tokens: {step_token_match})")
        
        # Update for next step
        prompt = token_dyn.unsqueeze(1)
        current_pos += 1
        cache_len_tensor = min(cache_len_tensor + 1, max_seq_len)
        
        # Update dynamic cache
        kv_list_dyn = kv_list_dyn
    
    # Final summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_pass = match1 and match2 and logits_match_all and tokens_match_all
    if all_pass:
        print("✓ All tests PASSED - forward_export matches forward!")
    else:
        print("✗ Some tests FAILED")
        print(f"  Test 1 (no cache): {'PASS' if match1 else 'FAIL'}")
        print(f"  Test 2 (with cache): {'PASS' if match2 else 'FAIL'}")
        print(f"  Test 3 (multi-step): {'PASS' if (logits_match_all and tokens_match_all) else 'FAIL'}")
    
    return all_pass


if __name__ == "__main__":
    success = test_forward_match()
    sys.exit(0 if success else 1)
