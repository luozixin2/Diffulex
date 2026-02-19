#!/usr/bin/env python3
"""
End-to-end example of DiffuLex Edge inference.

This example demonstrates:
1. Creating a model
2. Running inference with PyTorch
3. Exporting to ExecuTorch format
4. Running inference with the exported model (on supported platforms)

Usage:
    python edge_inference_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import tempfile

# Import DiffuLex Edge modules
from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.model.kv_cache import KVCache, KVCacheConfig
from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
from diffulex_edge.export import DiffuLexExporter, ExportConfig, BackendType, QuantizationType


def create_test_model():
    """Create a small test model."""
    config = FastdLLMV2EdgeConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,  # GQA
        intermediate_size=1024,
        max_position_embeddings=512,
    )
    model = FastdLLMV2Edge(config)
    model.eval()
    return model, config


def demo_pytorch_inference(model, config):
    """Demonstrate PyTorch inference."""
    print("\n" + "=" * 60)
    print("Demo 1: PyTorch Inference")
    print("=" * 60)
    
    # Create inference engine
    engine = InferenceEngine.from_model(model, device="cpu")
    
    # Simulate a prompt (token IDs)
    prompt_tokens = [1, 2, 3, 4, 5]  # Example token IDs
    
    # Generate
    gen_config = GenerationConfig(
        max_new_tokens=10,
        temperature=1.0,
        top_k=50,
        eos_token_id=999,  # Example EOS token
    )
    
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Generating up to {gen_config.max_new_tokens} tokens...")
    
    generated = engine.generate(prompt_tokens, gen_config)
    print(f"Generated tokens: {generated}")
    print(f"Total tokens: {len(prompt_tokens) + len(generated)}")


def demo_kv_cache_usage(model, config):
    """Demonstrate KV cache usage."""
    print("\n" + "=" * 60)
    print("Demo 2: KV Cache Usage")
    print("=" * 60)
    
    # Create KV cache
    cache_config = KVCacheConfig(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=128,
        dtype=torch.float32,
    )
    
    kv_cache = KVCache(cache_config, batch_size=1)
    print(f"KV Cache shape: {kv_cache.cache.shape}")
    print(f"  - Layers: {cache_config.num_layers}")
    print(f"  - KV heads: {cache_config.num_kv_heads}")
    print(f"  - Max seq len: {cache_config.max_seq_len}")
    
    # Prefill
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    positions = torch.arange(10).unsqueeze(0)
    
    with torch.no_grad():
        logits, updated_kv = model(
            input_ids,
            positions=positions,
            kv_cache=kv_cache.get_cache_tensor(),
            start_pos=0,
        )
    
    print(f"\nPrefill:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Updated KV shape: {updated_kv.shape}")
    
    # Decode step
    next_token = torch.argmax(logits[0, -1]).item()
    input_ids = torch.tensor([[next_token]])
    positions = torch.tensor([[10]])
    
    # Update cache with prefill results
    kv_cache.cache = updated_kv
    
    with torch.no_grad():
        logits, updated_kv = model(
            input_ids,
            positions=positions,
            kv_cache=kv_cache.get_cache_tensor(),
            start_pos=10,
        )
    
    print(f"\nDecode step:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")


def demo_export(model, config):
    """Demonstrate model export."""
    print("\n" + "=" * 60)
    print("Demo 3: Model Export")
    print("=" * 60)
    
    # Create example inputs
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    positions = torch.arange(16).unsqueeze(0)
    
    # Create KV cache for export
    kv_cache = torch.zeros(
        config.num_hidden_layers, 2, 1,
        config.num_key_value_heads, config.max_position_embeddings, config.head_dim
    )
    
    example_inputs = (input_ids, positions, None, kv_cache, 0)
    
    # Try different backends
    backends = [
        (BackendType.REFERENCE, "Reference"),
        (BackendType.XNNPACK, "XNNPACK"),
    ]
    
    for backend_type, name in backends:
        print(f"\nTrying {name} backend...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pte"
            
            export_config = ExportConfig(
                output_path=output_path,
                backend=backend_type,
                quantization=QuantizationType.NONE,
            )
            
            exporter = DiffuLexExporter(export_config)
            result = exporter.export(model, example_inputs)
            
            if result.success:
                print(f"  [OK] Success!")
                print(f"     File: {result.output_path}")
                print(f"     Size: {result.file_size_mb:.2f} MB")
                print(f"     Time: {result.compilation_time_sec:.2f}s")
            else:
                error_msg = result.error_message[:100] if result.error_message else "Unknown error"
                print(f"  [FAIL] {error_msg}")


def demo_quantization(model, config):
    """Demonstrate quantization."""
    print("\n" + "=" * 60)
    print("Demo 4: Quantization")
    print("=" * 60)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    positions = torch.arange(16).unsqueeze(0)
    example_inputs = (input_ids, positions)
    
    quant_modes = [
        (QuantizationType.NONE, "FP32 (No quantization)"),
        (QuantizationType.DYNAMIC_INT8, "Dynamic INT8"),
        (QuantizationType.WEIGHT_ONLY_INT8, "Weight-only INT8"),
    ]
    
    for quant_type, name in quant_modes:
        print(f"\n{name}:")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pte"
            
            export_config = ExportConfig(
                output_path=output_path,
                backend=BackendType.REFERENCE,
                quantization=quant_type,
            )
            
            exporter = DiffuLexExporter(export_config)
            result = exporter.export(model, example_inputs)
            
            if result.success:
                print(f"  [OK] Success - Size: {result.file_size_mb:.2f} MB")
            else:
                error_msg = result.error_message[:80] if result.error_message else "Unknown error"
                print(f"  [WARN] {error_msg}...")


def main():
    """Run all demos."""
    print("=" * 60)
    print("DiffuLex Edge - End-to-End Example")
    print("=" * 60)
    
    # Create model
    print("\nCreating test model...")
    model, config = create_test_model()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {num_params:,} parameters ({num_params / 1e6:.2f}M)")
    
    # Run demos
    demo_pytorch_inference(model, config)
    demo_kv_cache_usage(model, config)
    demo_export(model, config)
    demo_quantization(model, config)
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
