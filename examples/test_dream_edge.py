"""Example script for loading and using DreamEdge model.

This demonstrates how to load the Dream 7B model and run inference.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from diffulex_edge.model.model_loader import load_hf_model


def main():
    # Model path
    model_path = "/root/autodl-tmp/Dream-v0-Instruct-7B"
    
    print("=" * 60)
    print("DreamEdge Model Loading and Inference Example")
    print("=" * 60)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\nModel not found at {model_path}")
        print("Please download the model first.")
        return
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    print("(This may take a while for 7B parameters...)")
    
    try:
        model, model_type, config = load_hf_model(
            model_path, 
            dtype=torch.float16,  # Use FP16 for faster inference
            device="cpu",
            optimize_cpu=True,     # Enable CPU optimizations
        )
        print(f"✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Print model info
    print("\n" + "=" * 60)
    print("Model Configuration")
    print("=" * 60)
    print(f"Model type: {model_type}")
    print(f"Vocab size: {config['vocab_size']}")
    print(f"Hidden size: {config['hidden_size']}")
    print(f"Num layers: {config['num_hidden_layers']}")
    print(f"Num attention heads: {config['num_attention_heads']}")
    print(f"Num KV heads: {config['num_key_value_heads']}")
    print(f"Intermediate size: {config['intermediate_size']}")
    print(f"Max position embeddings: {config['max_position_embeddings']}")
    print(f"RoPE theta: {config['rope_theta']}")
    
    # Test inference
    print("\n" + "=" * 60)
    print("Inference Test")
    print("=" * 60)
    
    # Create sample input (random tokens for testing)
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 10
    
    # Special tokens
    mask_token_id = config.get('mask_token_id', 151666)
    pad_token_id = config.get('pad_token_id', 151643)
    
    # Create input with some mask tokens (simulating diffusion input)
    input_ids = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long)
    input_ids[0, :3] = torch.tensor([100, 200, 300])  # Some context tokens
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids[0].tolist()}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        logits, _ = model(input_ids)
    
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    
    # Check output validity
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    
    print(f"\nOutput validation:")
    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")
    
    # Show top predictions for a few positions
    print(f"\nTop 3 predictions for each position:")
    for i in range(min(5, seq_len)):
        top_k = torch.topk(logits[0, i], k=3)
        tokens = top_k.indices.tolist()
        probs = torch.softmax(top_k.values, dim=0).tolist()
        print(f"  Position {i}: tokens={tokens}, probs={[f'{p:.3f}' for p in probs]}")
    
    print("\n" + "=" * 60)
    print("✓ Inference test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
