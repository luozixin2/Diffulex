"""Test numerical alignment between DreamEdge and HF Dream model.

This test loads both models with the same weights and compares their outputs
to verify numerical equivalence.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_hf_dream_model_direct(model_path: str, device="cpu"):
    """Load HF Dream model directly using the model files."""
    import json
    
    # Load config
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Import and load using transformers
    sys.path.insert(0, model_path)
    from configuration_dream import DreamConfig
    from modeling_dream import DreamModel
    
    dream_config = DreamConfig.from_pretrained(model_path)
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    
    return model, dream_config


def test_output_alignment():
    """Test that DreamEdge produces same output as HF Dream."""
    print("Testing numerical alignment between DreamEdge and HF Dream...")
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    
    if not Path(model_path).exists():
        print(f"  ⚠ Model path does not exist, skipping")
        return
    
    try:
        # Load DreamEdge model
        print("  Loading DreamEdge model...")
        from diffulex_edge.model.model_loader import load_hf_model
        edge_model, _, _ = load_hf_model(model_path, dtype=torch.float32)
        edge_model.eval()
        
        # Load HF Dream model
        print("  Loading HF Dream model...")
        hf_model, hf_config = load_hf_dream_model_direct(model_path)
        hf_model.eval()
        
        # Create test input
        batch_size = 1
        seq_len = 10
        vocab_size = hf_config.vocab_size
        
        torch.manual_seed(42)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        print(f"  Input shape: {input_ids.shape}")
        
        # Forward through DreamEdge
        with torch.no_grad():
            edge_logits, _ = edge_model(input_ids)
        
        # Forward through HF Dream
        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits
        
        print(f"  DreamEdge logits shape: {edge_logits.shape}")
        print(f"  HF Dream logits shape: {hf_logits.shape}")
        
        # Compare outputs
        max_diff = (edge_logits - hf_logits).abs().max().item()
        mean_diff = (edge_logits - hf_logits).abs().mean().item()
        
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        # Check if outputs are close enough
        # Using a relatively loose tolerance due to potential implementation differences
        if max_diff < 1e-3:
            print("  ✓ Outputs are well aligned!")
        elif max_diff < 1e-2:
            print("  ⚠ Outputs have minor differences (acceptable)")
        else:
            print(f"  ✗ Outputs differ significantly (max_diff={max_diff:.4f})")
            
        # Also compare argmax predictions
        edge_pred = edge_logits.argmax(dim=-1)
        hf_pred = hf_logits.argmax(dim=-1)
        match_rate = (edge_pred == hf_pred).float().mean().item()
        print(f"  Prediction match rate: {match_rate*100:.2f}%")
        
        return max_diff < 1e-2
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kv_cache_alignment():
    """Test that KV cache produces same results in incremental decoding."""
    print("Testing KV cache alignment...")
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    
    if not Path(model_path).exists():
        print(f"  ⚠ Model path does not exist, skipping")
        return
    
    try:
        from diffulex_edge.model.model_loader import load_hf_model
        
        # Load DreamEdge model
        print("  Loading DreamEdge model...")
        model, _, config = load_hf_model(model_path, dtype=torch.float32)
        model.eval()
        
        # Create test input
        batch_size = 1
        prefill_len = 5
        vocab_size = config['vocab_size']
        
        torch.manual_seed(42)
        input_ids = torch.randint(0, vocab_size, (batch_size, prefill_len))
        
        # Prefill
        with torch.no_grad():
            logits_prefill, past_kv = model(input_ids, use_cache=True)
        
        # Decode one token
        next_token = torch.randint(0, vocab_size, (batch_size, 1))
        position_ids = torch.tensor([[prefill_len]])
        
        with torch.no_grad():
            logits_decode, past_kv_2 = model(
                next_token,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True
            )
        
        # Compare with full forward
        full_input = torch.cat([input_ids, next_token], dim=1)
        with torch.no_grad():
            logits_full, _ = model(full_input, use_cache=False)
        
        # The last token of full forward should match decode forward
        logits_full_last = logits_full[:, -1:, :]
        max_diff = (logits_decode - logits_full_last).abs().max().item()
        
        print(f"  Prefill length: {prefill_len}")
        print(f"  Max diff between KV decode and full forward: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            print("  ✓ KV cache produces identical results!")
            return True
        elif max_diff < 1e-4:
            print("  ⚠ Minor differences in KV cache (acceptable)")
            return True
        else:
            print(f"  ✗ KV cache differs significantly")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_mask_handling():
    """Test attention mask handling in both models."""
    print("Testing attention mask handling...")
    
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    
    if not Path(model_path).exists():
        print(f"  ⚠ Model path does not exist, skipping")
        return
    
    try:
        from diffulex_edge.model.model_loader import load_hf_model
        
        # Load DreamEdge model
        print("  Loading DreamEdge model...")
        model, _, config = load_hf_model(model_path, dtype=torch.float32)
        model.eval()
        
        # Create test input with padding
        batch_size = 2
        seq_len = 10
        vocab_size = config['vocab_size']
        pad_token_id = config.get('pad_token_id', 151643)
        
        torch.manual_seed(42)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Pad the second sequence
        input_ids[1, 5:] = pad_token_id
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[1, 5:] = 0
        
        # Forward with mask
        with torch.no_grad():
            logits_masked, _ = model(input_ids, attention_mask=attention_mask)
        
        # Forward without mask
        with torch.no_grad():
            logits_no_mask, _ = model(input_ids)
        
        # The outputs should be different
        max_diff = (logits_masked - logits_no_mask).abs().max().item()
        print(f"  Max diff with/without mask: {max_diff:.6f}")
        
        if max_diff > 0:
            print("  ✓ Attention mask affects output correctly")
            return True
        else:
            print("  ✗ Attention mask has no effect")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all numerical alignment tests."""
    print("=" * 70)
    print("DreamEdge Numerical Alignment Tests")
    print("=" * 70)
    
    tests = [
        test_output_alignment,
        test_kv_cache_alignment,
        test_attention_mask_handling,
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
            print(f"  ✗ Test failed with exception: {e}")
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
