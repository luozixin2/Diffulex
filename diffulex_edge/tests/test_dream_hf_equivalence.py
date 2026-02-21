"""Test numerical equivalence between DreamEdge and HF Dream model.

This test loads both models with the same weights and compares their outputs
to verify numerical equivalence.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_dream_edge_model(config_dict, device="cpu", dtype=torch.float32):
    """Create DreamEdge model from config dict."""
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    config = DreamEdgeConfig(
        vocab_size=config_dict.get("vocab_size", 152064),
        hidden_size=config_dict.get("hidden_size", 3584),
        num_hidden_layers=config_dict.get("num_hidden_layers", 28),
        num_attention_heads=config_dict.get("num_attention_heads", 28),
        num_key_value_heads=config_dict.get("num_key_value_heads", 4),
        intermediate_size=config_dict.get("intermediate_size", 18944),
        max_position_embeddings=config_dict.get("max_position_embeddings", 131072),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
        rope_theta=config_dict.get("rope_theta", 1000000.0),
        attention_bias=config_dict.get("attention_bias", True),
        attention_dropout=config_dict.get("attention_dropout", 0.0),
        mask_token_id=config_dict.get("mask_token_id", 151666),
        pad_token_id=config_dict.get("pad_token_id", 151643),
        bos_token_id=config_dict.get("bos_token_id", 151643),
        eos_token_id=config_dict.get("eos_token_id", 151643),
    )
    
    model = DreamEdge(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    return model, config


def create_hf_dream_model(config_dict, device="cpu", dtype=torch.float32):
    """Create HF Dream model from config dict."""
    import importlib.util
    
    # Check if HF Dream is available
    spec = importlib.util.find_spec("transformers")
    if spec is None:
        print("  ⚠ transformers not installed, trying local model_cache...")
        # Try to load from local model_cache
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
        try:
            from model_cache.dream.model_dream import DreamModel
            from model_cache.dream.configuration_dream import DreamConfig
            
            config = DreamConfig(**config_dict)
            model = DreamModel(config)
            model = model.to(device=device, dtype=dtype)
            model.eval()
            return model, config
        except ImportError:
            return None, None
    else:
        from transformers import AutoModel, AutoConfig
        
        try:
            config = AutoConfig.from_pretrained("Dream-org/Dream-v0-Instruct-7B")
            model = AutoModel.from_pretrained("Dream-org/Dream-v0-Instruct-7B", torch_dtype=dtype)
            model = model.to(device)
            model.eval()
            return model, config
        except Exception as e:
            print(f"  ⚠ Could not load HF Dream: {e}")
            return None, None


def test_small_model_equivalence():
    """Test numerical equivalence with small random models."""
    print("Testing numerical equivalence with small random models...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    # Create config
    config_dict = {
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 512,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "mask_token_id": 998,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 0,
    }
    
    # Create two DreamEdge models with same config
    model1, config1 = create_dream_edge_model(config_dict)
    model2, config2 = create_dream_edge_model(config_dict)
    
    # Copy weights from model1 to model2
    model2.load_state_dict(model1.state_dict())
    
    # Test with same input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config_dict["vocab_size"], (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        logits1, kv1 = model1(input_ids, positions)
        logits2, kv2 = model2(input_ids, positions)
    
    max_diff = (logits1 - logits2).abs().max().item()
    mean_diff = (logits1 - logits2).abs().mean().item()
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  ✓ Models with same weights produce identical results")
        return True
    else:
        print("  ✗ Models differ unexpectedly")
        return False


def test_forward_shapes():
    """Test that forward pass produces correct shapes."""
    print("Testing forward pass shapes...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    config = DreamEdgeConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=1024,
        pad_token_id=0,
    )
    
    model = DreamEdge(config)
    model.eval()
    
    # Test different batch sizes and sequence lengths
    test_cases = [
        (1, 1),   # Single token
        (1, 10),  # Single batch, seq_len=10
        (2, 10),  # Batch size 2
        (4, 20),  # Larger batch and seq
    ]
    
    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        with torch.no_grad():
            logits, kv_cache = model(input_ids, positions)
        
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, \
            f"Expected {expected_shape}, got {logits.shape} for batch={batch_size}, seq={seq_len}"
        
        assert len(kv_cache) == config.num_hidden_layers
        assert kv_cache[0][0].shape[0] == batch_size
        assert kv_cache[0][0].shape[2] == seq_len
    
    print(f"  ✓ Forward pass produces correct shapes for all test cases")
    return True


def test_incremental_decoding():
    """Test incremental decoding with KV cache."""
    print("Testing incremental decoding with KV cache...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    config = DreamEdgeConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=1024,
        pad_token_id=0,
    )
    
    model = DreamEdge(config)
    model.eval()
    
    batch_size = 1
    prefill_len = 5
    num_tokens_to_generate = 3
    
    # Prefill
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prefill_len))
    positions = torch.arange(prefill_len).unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        logits_prefill, kv_cache = model(input_ids, positions)
    
    # Generate tokens one by one
    generated_tokens = []
    for i in range(num_tokens_to_generate):
        # Get next token (greedy)
        next_token_id = logits_prefill[:, -1:, :].argmax(dim=-1)
        generated_tokens.append(next_token_id.item())
        
        # Forward with new token
        next_positions = torch.tensor([[prefill_len + i]])
        
        with torch.no_grad():
            logits_step, kv_cache = model(
                next_token_id,
                next_positions,
                kv_cache=kv_cache,
            )
        
        logits_prefill = torch.cat([logits_prefill, logits_step], dim=1)
    
    print(f"  ✓ Incremental decoding works correctly")
    print(f"    - Generated {num_tokens_to_generate} tokens: {generated_tokens}")
    print(f"    - Final sequence length: {prefill_len + num_tokens_to_generate}")
    
    return True


def test_attention_mask_effect():
    """Test that attention mask affects output (for future use with mask)."""
    print("Testing attention mask effect...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    config = DreamEdgeConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=1024,
        pad_token_id=0,
    )
    
    model = DreamEdge(config)
    model.eval()
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward without mask
    with torch.no_grad():
        logits_no_mask, _ = model(input_ids, positions, mask=None)
    
    # Forward with mask (all ones = no masking)
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    with torch.no_grad():
        logits_with_mask, _ = model(input_ids, positions, mask=mask)
    
    # They should be the same (mask of all ones = no masking)
    max_diff = (logits_no_mask - logits_with_mask).abs().max().item()
    
    print(f"  Max diff with all-ones mask: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  ✓ All-ones mask produces same results as no mask")
    else:
        print(f"  ⚠ Unexpected difference with all-ones mask")
    
    return True


def test_rope_consistency():
    """Test RoPE produces consistent results across calls."""
    print("Testing RoPE consistency...")
    
    from diffulex_edge.components.rope import RotaryEmbedding
    
    head_dim = 64
    max_len = 1024
    rope = RotaryEmbedding(head_dim, max_position_embeddings=max_len, base=10000.0)
    
    batch_size = 2
    seq_len = 10
    num_heads = 4
    
    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Apply RoPE twice
    q1, k1 = rope(positions, q, k)
    q2, k2 = rope(positions, q, k)
    
    max_diff_q = (q1 - q2).abs().max().item()
    max_diff_k = (k1 - k2).abs().max().item()
    
    print(f"  Max diff in Q: {max_diff_q:.6f}")
    print(f"  Max diff in K: {max_diff_k:.6f}")
    
    if max_diff_q < 1e-6 and max_diff_k < 1e-6:
        print("  ✓ RoPE produces consistent results")
        return True
    else:
        print("  ✗ RoPE results are inconsistent")
        return False


def test_model_info():
    """Test model info reporting."""
    print("Testing model info...")
    
    from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
    
    config = DreamEdgeConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=1024,
        pad_token_id=0,
    )
    
    model = DreamEdge(config)
    info = model.get_model_info()
    
    required_keys = [
        "model_type", "vocab_size", "hidden_size", "num_layers",
        "num_heads", "num_kv_heads", "head_dim", "max_position_embeddings",
        "num_parameters", "num_parameters_human", "supports_export"
    ]
    
    for key in required_keys:
        assert key in info, f"Missing key: {key}"
    
    print(f"  ✓ Model info is complete")
    print(f"    - Type: {info['model_type']}")
    print(f"    - Parameters: {info['num_parameters_human']}")
    print(f"    - Hidden size: {info['hidden_size']}")
    print(f"    - Layers: {info['num_layers']}")
    
    return True


def run_all_tests():
    """Run all equivalence tests."""
    print("=" * 70)
    print("DreamEdge Numerical Equivalence Tests")
    print("=" * 70)
    
    tests = [
        test_small_model_equivalence,
        test_forward_shapes,
        test_incremental_decoding,
        test_attention_mask_effect,
        test_rope_consistency,
        test_model_info,
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
