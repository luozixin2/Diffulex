"""Lightweight test for DreamEdge numerical alignment.

Uses smaller sequence lengths for faster CPU testing.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_small_model_alignment():
    """Test alignment with a small synthetic model."""
    print("Testing alignment with small synthetic model...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    # Create small config for fast testing
    config = DreamEdgeConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=True,
        mask_token_id=998,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
    )
    
    # Create two models with same weights
    torch.manual_seed(42)
    model1 = DreamEdge(config)
    model1.eval()
    
    torch.manual_seed(42)
    model2 = DreamEdge(config)
    model2.eval()
    
    # Test with same input
    torch.manual_seed(123)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    
    with torch.no_grad():
        logits1, _ = model1(input_ids)
        logits2, _ = model2(input_ids)
    
    # Should be identical
    max_diff = (logits1 - logits2).abs().max().item()
    print(f"  Max diff between identical models: {max_diff:.10f}")
    
    assert max_diff < 1e-6, f"Identical models should produce identical outputs, got {max_diff}"
    print("  ✓ Model outputs are deterministic")
    
    return True


def test_bidirectional_attention():
    """Test that attention is truly bidirectional (non-causal).
    
    In bidirectional attention, masking position j should affect output at position i
    for all i, j (unlike causal attention where position i can only see positions <= i).
    """
    print("Testing bidirectional attention...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    config = DreamEdgeConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=1024,
        attention_bias=True,
        pad_token_id=0,
    )
    
    torch.manual_seed(42)
    model = DreamEdge(config)
    model.eval()
    
    torch.manual_seed(123)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    
    # Get baseline output
    with torch.no_grad():
        logits_baseline, _ = model(input_ids)
    
    # Test: mask position 6, check effect on position 2
    # In bidirectional attention, this should affect position 2
    # In causal attention, position 2 cannot see position 6, so no effect
    attention_mask = torch.ones(1, 8)
    attention_mask[0, 6] = 0
    
    with torch.no_grad():
        logits_masked, _ = model(input_ids, attention_mask=attention_mask)
    
    # Position 2 should be affected
    diff_at_2 = (logits_baseline[0, 2] - logits_masked[0, 2]).abs().max().item()
    
    # Position 6 itself should definitely be affected
    diff_at_6 = (logits_baseline[0, 6] - logits_masked[0, 6]).abs().max().item()
    
    print(f"  Diff at position 2 when masking position 6: {diff_at_2:.6f}")
    print(f"  Diff at position 6 (masked position): {diff_at_6:.6f}")
    
    # In bidirectional attention, both should be affected
    assert diff_at_2 > 1e-3, f"Bidirectional attention: position 2 should be affected by masking position 6, got {diff_at_2}"
    assert diff_at_6 > 1e-3, f"Masked position should be affected, got {diff_at_6}"
    
    print("  ✓ Bidirectional attention verified")
    return True


def test_attention_mask_effect():
    """Test that attention mask actually affects output."""
    print("Testing attention mask effect...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    config = DreamEdgeConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=1024,
        attention_bias=True,
        pad_token_id=0,
    )
    
    torch.manual_seed(42)
    model = DreamEdge(config)
    model.eval()
    
    torch.manual_seed(123)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    
    # Without mask
    with torch.no_grad():
        logits_no_mask, _ = model(input_ids)
    
    # With mask (mask out last 3 positions of second sequence)
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    attention_mask[1, 5:] = 0
    
    with torch.no_grad():
        logits_with_mask, _ = model(input_ids, attention_mask=attention_mask)
    
    # First sequence should be identical (no masking)
    diff_first = (logits_no_mask[0] - logits_with_mask[0]).abs().max().item()
    print(f"  Diff for unmasked sequence: {diff_first:.6f}")
    
    # Second sequence should be different (masked)
    diff_second = (logits_no_mask[1] - logits_with_mask[1]).abs().max().item()
    print(f"  Diff for masked sequence: {diff_second:.6f}")
    
    # The first sequence should be nearly identical
    assert diff_first < 1e-5, f"Unmasked sequence should be identical, got {diff_first}"
    
    # The second sequence should be different
    assert diff_second > 1e-3, f"Masked sequence should differ, got {diff_second}"
    
    print("  ✓ Attention mask works correctly")
    return True


def test_model_save_load():
    """Test that model can be saved and loaded correctly."""
    print("Testing model save/load...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    import tempfile
    
    config = DreamEdgeConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=1024,
        attention_bias=True,
        pad_token_id=0,
    )
    
    torch.manual_seed(42)
    model = DreamEdge(config)
    model.eval()
    
    # Test input
    torch.manual_seed(123)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    
    with torch.no_grad():
        original_logits, _ = model(input_ids)
    
    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    try:
        torch.save(model.state_dict(), temp_path)
        
        # Create new model and load
        model2 = DreamEdge(config)
        model2.load_state_dict(torch.load(temp_path, weights_only=True))
        model2.eval()
        
        with torch.no_grad():
            loaded_logits, _ = model2(input_ids)
        
        max_diff = (original_logits - loaded_logits).abs().max().item()
        print(f"  Max diff after save/load: {max_diff:.10f}")
        
        assert max_diff < 1e-6, f"Save/load should preserve weights, got diff {max_diff}"
        print("  ✓ Model save/load works correctly")
        return True
        
    finally:
        import os
        os.unlink(temp_path)


def test_model_structure():
    """Test model structure matches expected Dream architecture."""
    print("Testing model structure...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    config = DreamEdgeConfig(
        vocab_size=152064,
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        intermediate_size=18944,
        max_position_embeddings=131072,
        rope_theta=1000000.0,
        attention_bias=True,
        mask_token_id=151666,
        pad_token_id=151643,
    )
    
    model = DreamEdge(config)
    
    # Check layer count
    assert len(model.layers) == 28, f"Expected 28 layers, got {len(model.layers)}"
    
    # Check attention heads
    layer = model.layers[0]
    assert layer.self_attn.num_heads == 28, "Expected 28 attention heads"
    assert layer.self_attn.num_kv_heads == 4, "Expected 4 KV heads (GQA)"
    assert layer.self_attn.num_kv_groups == 7, "Expected 7x grouping (28/4)"
    
    # Check bias configuration
    assert layer.self_attn.q_proj.bias is not None, "Q proj should have bias"
    assert layer.self_attn.k_proj.bias is not None, "K proj should have bias"
    assert layer.self_attn.v_proj.bias is not None, "V proj should have bias"
    assert layer.self_attn.o_proj.bias is None, "O proj should not have bias"
    
    # Check non-causal attention
    assert layer.self_attn.is_causal == False, "Dream uses non-causal attention"
    
    # Check embeddings
    assert model.embed_tokens.num_embeddings == 152064
    assert model.embed_tokens.embedding_dim == 3584
    assert model.lm_head.out_features == 152064
    assert model.lm_head.in_features == 3584
    
    print("  ✓ Model structure is correct")
    print(f"    - Layers: {len(model.layers)}")
    print(f"    - Hidden size: {config.hidden_size}")
    print(f"    - Attention heads: {layer.self_attn.num_heads}")
    print(f"    - KV heads: {layer.self_attn.num_kv_heads}")
    print(f"    - Intermediate size: {config.intermediate_size}")
    print(f"    - Vocab size: {config.vocab_size}")
    print(f"    - Causal attention: {layer.self_attn.is_causal}")
    
    return True


def test_rope_application():
    """Test that RoPE is applied correctly."""
    print("Testing RoPE application...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    from diffulex_edge.components import RotaryEmbedding
    
    # Test RoPE directly
    head_dim = 64
    max_seq_len = 128
    
    rope = RotaryEmbedding(head_dim, max_position_embeddings=max_seq_len, base=10000.0)
    
    # Create sample Q and K tensors
    batch_size = 1
    num_heads = 4
    seq_len = 4
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Apply RoPE at different positions
    position_ids_1 = torch.tensor([[0, 1, 2, 3]])
    position_ids_2 = torch.tensor([[4, 5, 6, 7]])
    
    q_rot_1, k_rot_1 = rope(position_ids_1, q, k)
    q_rot_2, k_rot_2 = rope(position_ids_2, q, k)
    
    # Same relative positions should have different absolute encodings
    diff = (q_rot_1 - q_rot_2).abs().mean().item()
    print(f"  Mean diff of Q with different position IDs: {diff:.6f}")
    
    assert diff > 0.1, f"RoPE should produce different encodings for different positions, got {diff}"
    
    # Test that RoPE module exists in model
    config = DreamEdgeConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        attention_bias=True,
        pad_token_id=0,
    )
    
    model = DreamEdge(config)
    assert hasattr(model.layers[0].self_attn, 'rotary_emb'), "Model should have rotary_emb"
    
    print("  ✓ RoPE is applied correctly")
    
    return True


def run_all_tests():
    """Run all lightweight tests."""
    print("=" * 60)
    print("DreamEdge Lightweight Alignment Tests")
    print("=" * 60)
    
    tests = [
        test_small_model_alignment,
        test_bidirectional_attention,
        test_attention_mask_effect,
        test_rope_application,
        test_model_save_load,
        test_model_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print()
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
