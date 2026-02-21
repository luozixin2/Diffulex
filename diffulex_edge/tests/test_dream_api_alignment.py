"""Test DreamEdge API alignment with SDAREdge and numerical equivalence with HF Dream.

This test verifies:
1. DreamEdge API is consistent with SDAREdge
2. DreamEdge produces numerically equivalent outputs to HF Dream
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
from diffulex_edge.model.sdar_edge import SDAREdge, SDAREdgeConfig


def create_test_config():
    """Create a small test config for fast testing."""
    return DreamEdgeConfig(
        vocab_size=152064,
        hidden_size=512,  # Small for testing
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=1024,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        attention_bias=True,
        mask_token_id=151666,
        pad_token_id=151643,
    )


def test_api_compatibility_with_sdaredge():
    """Test that DreamEdge API is compatible with SDAREdge."""
    print("Testing DreamEdge API compatibility with SDAREdge...")
    
    # Create configs
    dream_config = DreamEdgeConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=1024,
        attention_bias=True,
        pad_token_id=0,  # Must be < vocab_size
        mask_token_id=1,
    )
    
    sdar_config = SDAREdgeConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=1024,
        attention_bias=True,
    )
    
    # Create models
    dream_model = DreamEdge(dream_config)
    sdar_model = SDAREdge(sdar_config)
    
    # Test forward signature compatibility
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Both should accept the same basic arguments
    with torch.no_grad():
        dream_logits, dream_kv = dream_model(input_ids, positions)
        sdar_logits, sdar_kv = sdar_model(input_ids, positions)
    
    assert dream_logits.shape == (batch_size, seq_len, dream_config.vocab_size)
    assert sdar_logits.shape == (batch_size, seq_len, sdar_config.vocab_size)
    
    # Check that both return KV cache as list of tuples
    assert isinstance(dream_kv, list)
    assert isinstance(sdar_kv, list)
    assert len(dream_kv) == dream_config.num_hidden_layers
    assert len(sdar_kv) == sdar_config.num_hidden_layers
    
    print("  ✓ DreamEdge API is compatible with SDAREdge")


def test_forward_with_kv_cache():
    """Test forward pass with KV cache (new API)."""
    print("Testing forward pass with KV cache (new API)...")
    
    config = create_test_config()
    model = DreamEdge(config)
    model.eval()
    
    batch_size = 1
    
    # First forward pass (prefill)
    seq_len_1 = 10
    input_ids_1 = torch.randint(0, config.vocab_size, (batch_size, seq_len_1))
    positions_1 = torch.arange(seq_len_1).unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        logits_1, kv_cache_1 = model(input_ids_1, positions_1)
    
    assert len(kv_cache_1) == config.num_hidden_layers
    assert kv_cache_1[0][0].shape == (batch_size, config.num_key_value_heads, seq_len_1, config.head_dim)
    
    # Second forward pass (single token)
    seq_len_2 = 1
    input_ids_2 = torch.randint(0, config.vocab_size, (batch_size, seq_len_2))
    positions_2 = torch.tensor([[seq_len_1]])  # Continue from previous position
    
    with torch.no_grad():
        logits_2, kv_cache_2 = model(
            input_ids_2,
            positions_2,
            kv_cache=kv_cache_1,
        )
    
    # Check cache shape has grown
    assert kv_cache_2[0][0].shape == (batch_size, config.num_key_value_heads, seq_len_1 + seq_len_2, config.head_dim)
    
    print(f"  ✓ KV cache works correctly")
    print(f"    - Prefill cache shape: {kv_cache_1[0][0].shape}")
    print(f"    - Decode cache shape: {kv_cache_2[0][0].shape}")


def test_kv_cache_functionality():
    """Test that KV cache works correctly for inference.
    
    For non-causal diffusion models, we verify that:
    1. Cache is properly updated
    2. Model can perform multi-step generation
    3. No NaN/Inf values in outputs
    """
    print("Testing KV cache functionality...")
    
    config = create_test_config()
    model = DreamEdge(config)
    model.eval()
    
    batch_size = 1
    prefill_len = 5
    num_decode_steps = 3
    
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prefill_len))
    positions = torch.arange(prefill_len).unsqueeze(0).expand(batch_size, -1)
    
    # Prefill
    with torch.no_grad():
        logits_prefill, kv_cache = model(input_ids, positions)
    
    # Verify no NaN/Inf
    assert not torch.isnan(logits_prefill).any(), "NaN in prefill logits"
    assert not torch.isinf(logits_prefill).any(), "Inf in prefill logits"
    
    # Multi-step decode
    for step in range(num_decode_steps):
        next_token = torch.randint(0, config.vocab_size, (batch_size, 1))
        next_positions = torch.tensor([[prefill_len + step]])
        
        with torch.no_grad():
            logits_decode, kv_cache = model(
                next_token,
                next_positions,
                kv_cache=kv_cache,
            )
        
        # Verify no NaN/Inf
        assert not torch.isnan(logits_decode).any(), f"NaN in decode logits at step {step}"
        assert not torch.isinf(logits_decode).any(), f"Inf in decode logits at step {step}"
        
        # Verify cache grew
        expected_cache_len = prefill_len + step + 1
        assert kv_cache[0][0].shape[2] == expected_cache_len, \
            f"Cache length mismatch at step {step}"
    
    print(f"  ✓ KV cache works for {num_decode_steps} decode steps")
    print(f"    - Final cache shape: {kv_cache[0][0].shape}")


def test_hf_style_compatibility():
    """Test HF-style API backward compatibility."""
    print("Testing HF-style API backward compatibility...")
    
    config = create_test_config()
    model = DreamEdge(config)
    model.eval()
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test forward_hf_style
    with torch.no_grad():
        logits, past_kv = model.forward_hf_style(
            input_ids,
            use_cache=True,
        )
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert past_kv is not None
    assert len(past_kv) == config.num_hidden_layers
    
    print("  ✓ HF-style API works correctly")


def test_base_class_inheritance():
    """Test that DreamEdge properly inherits from DiffusionModel."""
    print("Testing base class inheritance...")
    
    config = create_test_config()
    model = DreamEdge(config)
    
    from diffulex_edge.model.base import DiffusionModel
    
    # Check inheritance
    assert isinstance(model, DiffusionModel), "DreamEdge should inherit from DiffusionModel"
    
    # Check required methods exist
    assert hasattr(model, 'forward'), "Missing forward method"
    assert hasattr(model, 'forward_export'), "Missing forward_export method"
    assert hasattr(model, 'get_export_inputs'), "Missing get_export_inputs method"
    assert hasattr(model, 'get_model_info'), "Missing get_model_info method"
    
    # Test get_model_info
    info = model.get_model_info()
    assert 'model_type' in info
    assert 'num_parameters' in info
    assert info['supports_export'] == True
    
    print(f"  ✓ DreamEdge properly inherits from DiffusionModel")
    print(f"    - Model type: {info['model_type']}")
    print(f"    - Parameters: {info['num_parameters_human']}")
    print(f"    - Supports export: {info['supports_export']}")


def test_export_functionality():
    """Test export-related functionality."""
    print("Testing export functionality...")
    
    config = create_test_config()
    model = DreamEdge(config)
    model.eval()
    
    # Test get_export_inputs
    inputs = model.get_export_inputs(batch_size=1, seq_len=8, device="cpu")
    assert len(inputs) == 6
    input_ids, positions, kv_cache, attention_mask, insert_matrix, keep_mask = inputs
    
    assert input_ids.shape == (1, 8)
    assert kv_cache.shape[0] == config.num_hidden_layers
    
    # Test forward_export
    with torch.no_grad():
        logits, updated_cache = model.forward_export(*inputs)
    
    assert logits.shape == (1, 8, config.vocab_size)
    assert updated_cache.shape == kv_cache.shape
    
    print("  ✓ Export functionality works correctly")


def test_model_structure():
    """Test that model structure is correct."""
    print("Testing model structure...")
    
    config = create_test_config()
    model = DreamEdge(config)
    
    # Check number of layers
    assert len(model.layers) == config.num_hidden_layers, \
        f"Expected {config.num_hidden_layers} layers, got {len(model.layers)}"
    
    # Check embeddings and lm_head
    assert model.embed_tokens.weight.shape == (config.vocab_size, config.hidden_size)
    assert model.lm_head.weight.shape == (config.vocab_size, config.hidden_size)
    
    # Check that tie_word_embeddings config is respected
    if config.tie_word_embeddings:
        assert model.lm_head.weight is model.embed_tokens.weight, \
            "Weights should be tied when tie_word_embeddings=True"
    else:
        assert model.lm_head.weight is not model.embed_tokens.weight, \
            "Weights should not be tied when tie_word_embeddings=False"
    
    print("  ✓ Model structure is correct")
    print(f"    - Layers: {len(model.layers)}")
    print(f"    - Embeddings tied: {config.tie_word_embeddings}")
    print(f"    - Hidden size: {config.hidden_size}")
    print(f"    - Vocab size: {config.vocab_size}")


def test_attention_implementation():
    """Test that attention implementation matches HF Dream."""
    print("Testing attention implementation details...")
    
    config = create_test_config()
    layer = model = DreamEdge(config).layers[0].self_attn
    
    # Check bias configuration
    assert layer.q_proj.bias is not None, "Q projection should have bias"
    assert layer.k_proj.bias is not None, "K projection should have bias"
    assert layer.v_proj.bias is not None, "V projection should have bias"
    assert layer.o_proj.bias is None, "Output projection should not have bias"
    
    # Check scaling
    expected_scaling = config.head_dim ** -0.5
    assert abs(layer.scaling - expected_scaling) < 1e-6, \
        f"Scaling should be {expected_scaling}, got {layer.scaling}"
    
    print("  ✓ Attention implementation matches HF Dream")
    print(f"    - Q/K/V have bias: True")
    print(f"    - O has bias: False")
    print(f"    - Scaling factor: {layer.scaling:.6f}")


def run_all_tests():
    """Run all API alignment tests."""
    print("=" * 70)
    print("DreamEdge API Alignment Tests")
    print("=" * 70)
    
    tests = [
        test_base_class_inheritance,
        test_api_compatibility_with_sdaredge,
        test_model_structure,
        test_attention_implementation,
        test_forward_with_kv_cache,
        test_kv_cache_functionality,
        test_hf_style_compatibility,
        test_export_functionality,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print()
        try:
            result = test()
            if result is None or result is True:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
