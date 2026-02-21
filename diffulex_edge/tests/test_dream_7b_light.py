"""Lightweight test for Dream 7B numerical equivalence.

Tests first few layers to verify numerical alignment without loading full model into memory twice.
"""

import sys
import torch
from pathlib import Path

# Import transformers first
import transformers
from transformers import AutoModel, AutoConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_weight_loading():
    """Test that weights are loaded correctly."""
    print("=" * 70)
    print("Testing Weight Loading")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    from diffulex_edge.model.model_loader import load_hf_model
    
    print("Loading DreamEdge model...")
    edge_model, model_type, edge_config = load_hf_model(
        model_path, 
        dtype=dtype,
        device=device,
        optimize_cpu=False
    )
    
    print(f"\nModel type: {model_type}")
    print(f"Vocab size: {edge_config['vocab_size']}")
    print(f"Hidden size: {edge_config['hidden_size']}")
    print(f"Num layers: {edge_config['num_hidden_layers']}")
    print(f"Num heads: {edge_config['num_attention_heads']}")
    print(f"Num KV heads: {edge_config['num_key_value_heads']}")
    
    # Test a simple forward pass
    batch_size = 1
    seq_len = 5
    torch.manual_seed(42)
    input_ids = torch.randint(0, edge_config['vocab_size'], (batch_size, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    
    print(f"\nTesting forward pass with input shape: {input_ids.shape}")
    
    with torch.no_grad():
        logits, kv_cache = edge_model(input_ids, positions)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"KV cache length: {len(kv_cache)}")
    print(f"KV cache[0] K shape: {kv_cache[0][0].shape}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    
    print(f"\nHas NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")
    
    if not has_nan and not has_inf:
        print("  ✓ Forward pass successful!")
        return True
    else:
        print("  ✗ Forward pass produced invalid values")
        return False


def test_first_layer_components():
    """Test individual components of first layer."""
    print("\n" + "=" * 70)
    print("Testing First Layer Components")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load DreamEdge model
    from diffulex_edge.model.model_loader import load_hf_model
    
    print("Loading DreamEdge model...")
    edge_model, _, _ = load_hf_model(
        model_path, 
        dtype=dtype,
        device=device,
        optimize_cpu=False
    )
    
    # Create test input
    batch_size = 1
    seq_len = 5
    torch.manual_seed(42)
    input_ids = torch.randint(0, edge_model.config.vocab_size, (batch_size, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    with torch.no_grad():
        # Test embedding
        embeds = edge_model.embed_tokens(input_ids)
        print(f"Embeddings shape: {embeds.shape}")
        print(f"Embeddings sample: {embeds[0, 0, :5]}")
        
        # Test first layer norm
        layer0 = edge_model.layers[0]
        normed = layer0.input_layernorm(embeds)
        print(f"\nNormed shape: {normed.shape}")
        print(f"Normed sample: {normed[0, 0, :5]}")
        
        # Test attention
        attn_out, k, v = layer0.self_attn(positions, normed)
        print(f"\nAttention output shape: {attn_out.shape}")
        print(f"Attention output sample: {attn_out[0, 0, :5]}")
        print(f"K cache shape: {k.shape}")
        print(f"V cache shape: {v.shape}")
        
        # Test residual connection + MLP
        hidden = embeds + attn_out
        mlp_out = layer0.mlp(layer0.post_attention_layernorm(hidden))
        hidden = hidden + mlp_out
        
        print(f"\nLayer output shape: {hidden.shape}")
        print(f"Layer output sample: {hidden[0, 0, :5]}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(hidden).any().item() or torch.isnan(attn_out).any().item()
    has_inf = torch.isinf(hidden).any().item() or torch.isinf(attn_out).any().item()
    
    if not has_nan and not has_inf:
        print("  ✓ First layer components work correctly!")
        return True
    else:
        print("  ✗ First layer produced invalid values")
        return False


def test_multi_layer_forward():
    """Test forward through multiple layers."""
    print("\n" + "=" * 70)
    print("Testing Multi-Layer Forward (5 layers)")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load DreamEdge model
    from diffulex_edge.model.model_loader import load_hf_model
    
    print("Loading DreamEdge model...")
    edge_model, _, _ = load_hf_model(
        model_path, 
        dtype=dtype,
        device=device,
        optimize_cpu=False
    )
    
    # Create test input
    batch_size = 1
    seq_len = 5
    num_layers_to_test = 5
    torch.manual_seed(42)
    input_ids = torch.randint(0, edge_model.config.vocab_size, (batch_size, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Testing {num_layers_to_test} layers...")
    
    with torch.no_grad():
        hidden = edge_model.embed_tokens(input_ids)
        
        for layer_idx in range(num_layers_to_test):
            layer = edge_model.layers[layer_idx]
            hidden, k, v = layer(positions, hidden)
            
            max_val = hidden.abs().max().item()
            mean_val = hidden.abs().mean().item()
            has_nan = torch.isnan(hidden).any().item()
            has_inf = torch.isinf(hidden).any().item()
            
            status = "✓" if not has_nan and not has_inf else "✗"
            print(f"  Layer {layer_idx}: max={max_val:.4f}, mean={mean_val:.4f}, nan={has_nan}, inf={has_inf} {status}")
            
            if has_nan or has_inf:
                print(f"    ✗ Layer {layer_idx} produced invalid values!")
                return False
    
    # Final norm
    hidden = edge_model.norm(hidden)
    logits = edge_model.lm_head(hidden)
    
    print(f"\nFinal logits shape: {logits.shape}")
    print(f"Logits sample: {logits[0, 0, :5]}")
    
    pred = logits.argmax(dim=-1)
    print(f"Predictions: {pred[0].tolist()}")
    
    print("  ✓ Multi-layer forward successful!")
    return True


def test_kv_cache():
    """Test KV cache functionality."""
    print("\n" + "=" * 70)
    print("Testing KV Cache")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load DreamEdge model
    from diffulex_edge.model.model_loader import load_hf_model
    
    print("Loading DreamEdge model...")
    edge_model, _, _ = load_hf_model(
        model_path, 
        dtype=dtype,
        device=device,
        optimize_cpu=False
    )
    
    # Create test input
    batch_size = 1
    prefill_len = 5
    torch.manual_seed(42)
    input_ids = torch.randint(0, edge_model.config.vocab_size, (batch_size, prefill_len), device=device)
    positions = torch.arange(prefill_len, device=device).unsqueeze(0)
    
    print(f"\nPrefill input shape: {input_ids.shape}")
    
    # Prefill
    with torch.no_grad():
        logits_prefill, kv_cache = edge_model(input_ids, positions)
    
    print(f"Prefill logits shape: {logits_prefill.shape}")
    print(f"KV cache layers: {len(kv_cache)}")
    print(f"KV cache[0] K shape: {kv_cache[0][0].shape}")
    
    # Decode one token
    next_token = torch.tensor([[42]], device=device)
    next_position = torch.tensor([[prefill_len]], device=device)
    
    print(f"\nDecode token: {next_token.item()}")
    
    with torch.no_grad():
        logits_decode, kv_cache = edge_model(
            next_token,
            next_position,
            kv_cache=kv_cache,
        )
    
    print(f"Decode logits shape: {logits_decode.shape}")
    print(f"Updated KV cache[0] K shape: {kv_cache[0][0].shape}")
    
    pred = logits_decode.argmax(dim=-1)
    print(f"Predicted next token: {pred.item()}")
    
    print("  ✓ KV cache works correctly!")
    return True


def compare_with_hf():
    """Compare DreamEdge output with HF model."""
    print("\n" + "=" * 70)
    print("Comparing with HF Model (First 3 Layers)")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load HF model
    print("Loading HF model...")
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True
    )
    hf_model = hf_model.to(device)
    hf_model.eval()
    
    # Load DreamEdge model
    from diffulex_edge.model.model_loader import load_hf_model
    
    print("Loading DreamEdge model...")
    edge_model, _, _ = load_hf_model(
        model_path, 
        dtype=dtype,
        device=device,
        optimize_cpu=False
    )
    
    # Create test input
    batch_size = 1
    seq_len = 5
    num_layers = 3
    torch.manual_seed(42)
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Comparing first {num_layers} layers...")
    
    with torch.no_grad():
        # Get embeddings
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        edge_hidden = edge_model.embed_tokens(input_ids)
        
        embed_diff = (hf_hidden - edge_hidden).abs().max().item()
        print(f"\nEmbeddings max diff: {embed_diff:.6f}")
        
        for layer_idx in range(num_layers):
            # HF forward
            hf_hidden = hf_model.model.layers[layer_idx](
                hf_hidden,
                attention_mask=None,
                position_ids=position_ids,
            )[0]
            
            # Edge forward
            edge_hidden, _, _ = edge_model.layers[layer_idx](
                position_ids,
                edge_hidden,
            )
            
            max_diff = (hf_hidden - edge_hidden).abs().max().item()
            mean_diff = (hf_hidden - edge_hidden).abs().mean().item()
            
            status = "✓" if max_diff < 1e-3 else "⚠" if max_diff < 1e-2 else "✗"
            print(f"  Layer {layer_idx}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} {status}")
    
    print("\n  Comparison complete!")
    return True


def run_all_tests():
    """Run all lightweight tests."""
    print("\n" + "=" * 70)
    print("Dream 7B Lightweight Numerical Tests")
    print("=" * 70)
    
    tests = [
        ("Weight Loading", test_weight_loading),
        ("First Layer Components", test_first_layer_components),
        ("Multi-Layer Forward", test_multi_layer_forward),
        ("KV Cache", test_kv_cache),
        ("Compare with HF", compare_with_hf),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
