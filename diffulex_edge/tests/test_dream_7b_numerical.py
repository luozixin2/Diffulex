"""Test numerical equivalence between DreamEdge and HF Dream 7B model.

This test loads the actual 7B model weights and compares layer-by-layer outputs
to verify numerical equivalence.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Import transformers first to avoid circular import issues
import transformers
from transformers import AutoModel, AutoConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_hf_dream_model(model_path: str, device="cpu", dtype=torch.float32):
    """Load HF Dream model using AutoModel."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True
    )
    model = model.to(device)
    model.eval()
    
    return model, config


def load_dream_edge_model(model_path: str, device="cpu", dtype=torch.float32):
    """Load DreamEdge model with HF weights."""
    from diffulex_edge.model.model_loader import load_hf_model
    
    model, model_type, config = load_hf_model(
        model_path, 
        dtype=dtype,
        device=device,
        optimize_cpu=False
    )
    
    return model, config


def test_embedding_layer():
    """Test embedding layer equivalence."""
    print("=" * 70)
    print("Testing Embedding Layer")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load models
    print("Loading HF Dream model...")
    hf_model, hf_config = load_hf_dream_model(model_path, device, dtype)
    
    print("Loading DreamEdge model...")
    edge_model, edge_config = load_dream_edge_model(model_path, device, dtype)
    
    # Create test input
    batch_size = 1
    seq_len = 10
    torch.manual_seed(42)
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Get embeddings
    with torch.no_grad():
        hf_embeds = hf_model.embed_tokens(input_ids)
        edge_embeds = edge_model.embed_tokens(input_ids)
    
    print(f"HF embeddings shape: {hf_embeds.shape}")
    print(f"Edge embeddings shape: {edge_embeds.shape}")
    
    # Compare
    max_diff = (hf_embeds - edge_embeds).abs().max().item()
    mean_diff = (hf_embeds - edge_embeds).abs().mean().item()
    
    print(f"\nEmbedding comparison:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  ✓ Embedding layer matches perfectly!")
        return True
    elif max_diff < 1e-3:
        print("  ⚠ Embedding layer has minor differences")
        return True
    else:
        print(f"  ✗ Embedding layer differs significantly")
        return False


def test_layer_norm():
    """Test LayerNorm equivalence."""
    print("\n" + "=" * 70)
    print("Testing LayerNorm")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load models
    hf_model, hf_config = load_hf_dream_model(model_path, device, dtype)
    edge_model, edge_config = load_dream_edge_model(model_path, device, dtype)
    
    # Create test input
    batch_size = 1
    seq_len = 10
    torch.manual_seed(42)
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)
    
    # Get embeddings
    with torch.no_grad():
        hf_embeds = hf_model.embed_tokens(input_ids)
        edge_embeds = edge_model.embed_tokens(input_ids)
        
        # Apply first layer norm
        hf_normed = hf_model.layers[0].input_layernorm(hf_embeds)
        edge_normed = edge_model.layers[0].input_layernorm(edge_embeds)
    
    print(f"HF normed shape: {hf_normed.shape}")
    print(f"Edge normed shape: {edge_normed.shape}")
    
    # Compare
    max_diff = (hf_normed - edge_normed).abs().max().item()
    mean_diff = (hf_normed - edge_normed).abs().mean().item()
    
    print(f"\nLayerNorm comparison:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  ✓ LayerNorm matches perfectly!")
        return True
    elif max_diff < 1e-3:
        print("  ⚠ LayerNorm has minor differences")
        return True
    else:
        print(f"  ✗ LayerNorm differs significantly")
        return False


def test_attention_first_layer():
    """Test first layer attention equivalence."""
    print("\n" + "=" * 70)
    print("Testing First Layer Attention")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load models
    hf_model, hf_config = load_hf_dream_model(model_path, device, dtype)
    edge_model, edge_config = load_dream_edge_model(model_path, device, dtype)
    
    # Create test input
    batch_size = 1
    seq_len = 10
    torch.manual_seed(42)
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Get embeddings and apply first layer norm
    with torch.no_grad():
        hf_embeds = hf_model.embed_tokens(input_ids)
        edge_embeds = edge_model.embed_tokens(input_ids)
        
        hf_normed = hf_model.layers[0].input_layernorm(hf_embeds)
        edge_normed = edge_model.layers[0].input_layernorm(edge_embeds)
        
        print(f"HF normed sample: {hf_normed[0, 0, :5]}")
        print(f"Edge normed sample: {edge_normed[0, 0, :5]}")
        
        # Apply attention
        # HF Dream
        hf_attn_out = hf_model.layers[0].self_attn(
            hf_normed,
            attention_mask=None,
            position_ids=position_ids,
        )[0]
        
        # DreamEdge
        positions = position_ids
        edge_attn_out, _, _ = edge_model.layers[0].self_attn(
            positions,
            edge_normed,
        )
    
    print(f"\nHF attention output shape: {hf_attn_out.shape}")
    print(f"Edge attention output shape: {edge_attn_out.shape}")
    
    print(f"HF attention sample: {hf_attn_out[0, 0, :5]}")
    print(f"Edge attention sample: {edge_attn_out[0, 0, :5]}")
    
    # Compare
    max_diff = (hf_attn_out - edge_attn_out).abs().max().item()
    mean_diff = (hf_attn_out - edge_attn_out).abs().mean().item()
    
    print(f"\nAttention comparison:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  ✓ Attention matches perfectly!")
        return True
    elif max_diff < 1e-3:
        print("  ⚠ Attention has minor differences")
        return True
    else:
        print(f"  ✗ Attention differs significantly")
        return False


def test_full_first_layer():
    """Test full first layer equivalence."""
    print("\n" + "=" * 70)
    print("Testing Full First Layer")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load models
    hf_model, hf_config = load_hf_dream_model(model_path, device, dtype)
    edge_model, edge_config = load_dream_edge_model(model_path, device, dtype)
    
    # Create test input
    batch_size = 1
    seq_len = 10
    torch.manual_seed(42)
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward through first layer
    with torch.no_grad():
        # HF Dream
        hf_embeds = hf_model.embed_tokens(input_ids)
        hf_hidden = hf_model.layers[0](
            hf_embeds,
            attention_mask=None,
            position_ids=position_ids,
        )[0]
        
        # DreamEdge
        edge_embeds = edge_model.embed_tokens(input_ids)
        positions = position_ids
        edge_hidden, _, _ = edge_model.layers[0](
            positions,
            edge_embeds,
        )
    
    print(f"HF layer output shape: {hf_hidden.shape}")
    print(f"Edge layer output shape: {edge_hidden.shape}")
    
    print(f"HF layer output sample: {hf_hidden[0, 0, :5]}")
    print(f"Edge layer output sample: {edge_hidden[0, 0, :5]}")
    
    # Compare
    max_diff = (hf_hidden - edge_hidden).abs().max().item()
    mean_diff = (hf_hidden - edge_hidden).abs().mean().item()
    
    print(f"\nLayer comparison:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  ✓ First layer matches perfectly!")
        return True
    elif max_diff < 1e-3:
        print("  ⚠ First layer has minor differences")
        return True
    else:
        print(f"  ✗ First layer differs significantly")
        return False


def test_all_layers():
    """Test all layers sequentially."""
    print("\n" + "=" * 70)
    print("Testing All 28 Layers")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load models
    print("Loading models...")
    hf_model, hf_config = load_hf_dream_model(model_path, device, dtype)
    edge_model, edge_config = load_dream_edge_model(model_path, device, dtype)
    
    # Create test input
    batch_size = 1
    seq_len = 5  # Use smaller sequence for faster testing
    torch.manual_seed(42)
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Testing {len(hf_model.model.layers)} layers...")
    
    all_pass = True
    
    with torch.no_grad():
        # Get embeddings
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        edge_hidden = edge_model.embed_tokens(input_ids)
        
        for layer_idx in range(len(hf_model.model.layers)):
            # HF Dream layer
            hf_hidden = hf_model.model.layers[layer_idx](
                hf_hidden,
                attention_mask=None,
                position_ids=position_ids,
            )[0]
            
            # DreamEdge layer
            positions = position_ids
            edge_hidden, _, _ = edge_model.layers[layer_idx](
                positions,
                edge_hidden,
            )
            
            # Compare
            max_diff = (hf_hidden - edge_hidden).abs().max().item()
            mean_diff = (hf_hidden - edge_hidden).abs().mean().item()
            
            status = "✓" if max_diff < 1e-3 else "✗"
            print(f"  Layer {layer_idx:2d}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} {status}")
            
            if max_diff >= 1e-3:
                all_pass = False
    
    print(f"\n{'=' * 70}")
    if all_pass:
        print("  ✓ All layers match!")
    else:
        print("  ⚠ Some layers have differences")
    
    return all_pass


def test_full_model_output():
    """Test full model output equivalence."""
    print("\n" + "=" * 70)
    print("Testing Full Model Output")
    print("=" * 70)
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    device = "cpu"
    dtype = torch.float32
    
    # Load models
    print("Loading models...")
    hf_model, hf_config = load_hf_dream_model(model_path, device, dtype)
    
    # For DreamEdge, we need to use the forward method
    from diffulex_edge.model.model_loader import load_hf_model
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
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        # HF Dream
        hf_logits = hf_model(input_ids).logits
        
        # DreamEdge
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        edge_logits, _ = edge_model(input_ids, positions)
    
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"Edge logits shape: {edge_logits.shape}")
    
    print(f"HF logits sample: {hf_logits[0, 0, :5]}")
    print(f"Edge logits sample: {edge_logits[0, 0, :5]}")
    
    # Compare
    max_diff = (hf_logits - edge_logits).abs().max().item()
    mean_diff = (hf_logits - edge_logits).abs().mean().item()
    
    print(f"\nFull model comparison:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    # Compare predictions
    hf_pred = hf_logits.argmax(dim=-1)
    edge_pred = edge_logits.argmax(dim=-1)
    match_rate = (hf_pred == edge_pred).float().mean().item()
    
    print(f"  Prediction match rate: {match_rate * 100:.2f}%")
    
    if max_diff < 1e-5:
        print("  ✓ Full model matches perfectly!")
        return True
    elif max_diff < 1e-3:
        print("  ⚠ Full model has minor differences")
        return True
    elif match_rate > 0.95:
        print("  ⚠ Outputs differ but predictions match well")
        return True
    else:
        print(f"  ✗ Full model differs significantly")
        return False


def run_all_tests():
    """Run all numerical equivalence tests."""
    print("\n" + "=" * 70)
    print("Dream 7B Numerical Equivalence Tests")
    print("=" * 70)
    
    tests = [
        ("Embedding Layer", test_embedding_layer),
        ("LayerNorm", test_layer_norm),
        ("First Layer Attention", test_attention_first_layer),
        ("Full First Layer", test_full_first_layer),
        ("All 28 Layers", test_all_layers),
        ("Full Model Output", test_full_model_output),
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
