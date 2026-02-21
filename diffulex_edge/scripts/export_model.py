#!/usr/bin/env python3
"""Export DiffuLex models to ExecuTorch .pte format.

Examples:
    # Export from HuggingFace safetensors (recommended)
    python export_model.py /path/to/sdar-1.7b -o model.pte
    
    # Export with specific backend and quantization
    python export_model.py /path/to/model --backend xnnpack --quantization fp16 -o model.pte
    
    # Export a demo model for testing
    python export_model.py --demo --model-type fast_dllm_v2 -o demo.pte
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffulex_edge.export import ExportConfig, BackendType, QuantizationType, DiffuLexExporter
from diffulex_edge.model import load_hf_model, MODEL_REGISTRY
from diffulex_edge.quant.real_weight_quant import get_model_size_info


def create_demo_model(model_type: str):
    """Create a demo model for testing."""
    from diffulex_edge.model import (
        FastdLLMV2EdgeConfig, FastdLLMV2Edge,
        DreamEdgeConfig, DreamEdge,
        LLaDAEdgeConfig, LLaDAEdge,
        SDAREdgeConfig, SDAREdge,
    )

    configs = {
        "fast_dllm_v2": (FastdLLMV2EdgeConfig, FastdLLMV2Edge, {
            "vocab_size": 32000, "hidden_size": 768, "num_hidden_layers": 12,
            "num_attention_heads": 12, "intermediate_size": 2048,
            "max_position_embeddings": 2048,
        }),
        "dream": (DreamEdgeConfig, DreamEdge, {
            "vocab_size": 32000, "hidden_size": 768, "num_hidden_layers": 12,
            "num_attention_heads": 12, "intermediate_size": 2048,
            "max_position_embeddings": 2048,
        }),
        "llada": (LLaDAEdgeConfig, LLaDAEdge, {
            "vocab_size": 126464, "hidden_size": 768, "num_hidden_layers": 12,
            "num_attention_heads": 12, "intermediate_size": 2048,
            "max_position_embeddings": 2048, "mask_token_id": 126336,
        }),
        "sdar": (SDAREdgeConfig, SDAREdge, {
            "vocab_size": 151936, "hidden_size": 2048, "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8,
            "intermediate_size": 6144, "max_position_embeddings": 4096,
        }),
        "sdar_tiny": (SDAREdgeConfig, SDAREdge, {
            "vocab_size": 1000, "hidden_size": 256, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "intermediate_size": 512, "max_position_embeddings": 512,
        }),
    }

    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}")

    config_class, model_class, config_dict = configs[model_type]
    config = config_class(**config_dict)
    model = model_class(config)
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Export DiffuLex models to ExecuTorch")
    
    # Model source
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to HuggingFace model directory (with config.json and .safetensors)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Export a demo model instead of loading from HF",
    )
    parser.add_argument(
        "--model-type",
        choices=["fast_dllm_v2", "dream", "llada", "sdar", "sdar_tiny"],
        help="Model type for demo export",
    )
    
    # Output
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output .pte file path",
    )
    
    # Export options
    parser.add_argument(
        "--backend",
        choices=["reference", "xnnpack", "coreml", "qnn"],
        default="xnnpack",
        help="Target backend (default: xnnpack)",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "fp16", "dynamic_int8", "static_int8", "weight_only_int8", "int4"],
        default="fp16",
        help="Quantization type (default: fp16)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for example inputs (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for example inputs (default: 1)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length for KV cache (default: 2048)",
    )
    
    args = parser.parse_args()
    
    if not args.demo and not args.model_path:
        parser.error("Must specify either --demo or model_path")
    
    if args.demo and not args.model_type:
        parser.error("Must specify --model-type when using --demo")
    
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DiffuLex ExecuTorch Export")
    print("=" * 60)
    
    # Load or create model
    if args.demo:
        print(f"\n[1/3] Creating demo model: {args.model_type}")
        model, config = create_demo_model(args.model_type)
        print(f"      Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    else:
        model_path = Path(args.model_path)
        
        print(f"\n[1/3] Loading model from: {model_path}")
        
        dtype = torch.float32 if args.quantization == "none" else torch.float16
        
        try:
            model, model_type, hf_config = load_hf_model(str(model_path), dtype=dtype)
            
            # Create a simple config object for vocab_size access
            class SimpleConfig:
                def __init__(self, vocab_size):
                    self.vocab_size = vocab_size
            
            config = SimpleConfig(hf_config.get("vocab_size", 32000))
            print(f"      Type: {model_type}")
            print(f"      Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Prepare export
    print(f"\n[2/3] Configuring export:")
    print(f"      Backend: {args.backend}")
    print(f"      Quantization: {args.quantization}")
    print(f"      Output: {output_path}")
    
    # Map string to QuantizationType
    quant_map = {
        "none": QuantizationType.NONE,
        "fp16": QuantizationType.FP16,
        "dynamic_int8": QuantizationType.DYNAMIC_INT8,
        "static_int8": QuantizationType.STATIC_INT8,
        "weight_only_int8": QuantizationType.WEIGHT_ONLY_INT8,
        "int4": QuantizationType.INT4,
    }
    
    # Create example inputs for export
    # New architecture: use model's get_export_inputs method if available
    if hasattr(model, 'get_export_inputs'):
        print(f"      Using model's get_export_inputs method")
        example_inputs = model.get_export_inputs(
            batch_size=args.batch_size,
            seq_len=args.max_seq_len,
        )
        if hasattr(model, 'forward_export'):
            print(f"      Detected forward_export method, using Block Diffusion mask format")
        print(f"      Input shapes: {[t.shape for t in example_inputs]}")
    elif hasattr(model, 'forward_export'):
        # Legacy: SDAREdge style with manual input creation
        print(f"      Detected forward_export method (legacy path)")
        
        model_config = model.config
        batch_size = args.batch_size
        block_size = getattr(model_config, 'diffusion_block_size', 4)
        max_seq_len = args.max_seq_len
        num_kv_heads = getattr(model_config, 'num_key_value_heads', 8)
        head_dim = getattr(model_config, 'head_dim', None) or (model_config.hidden_size // model_config.num_attention_heads)
        num_layers = model_config.num_hidden_layers
        
        input_ids = torch.zeros(batch_size, block_size, dtype=torch.long)
        positions = torch.arange(block_size, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        kv_cache = torch.zeros(num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim, dtype=torch.float32)
        attention_mask = torch.zeros(num_layers, batch_size, 1, block_size, max_seq_len + block_size, dtype=torch.float32)
        attention_mask[:, :, :, :, 0:max_seq_len] = -10000.0
        insert_matrix = torch.zeros(num_layers, batch_size, 1, max_seq_len, block_size, dtype=torch.float32)
        for i in range(block_size):
            insert_matrix[:, :, :, i, i] = 1.0
        keep_mask = torch.ones(num_layers, batch_size, 1, max_seq_len, 1, dtype=torch.float32)
        keep_mask[:, :, :, 0:block_size, :] = 0.0
        
        example_inputs = (input_ids, positions, kv_cache, attention_mask, insert_matrix, keep_mask)
        print(f"      Input shapes: input_ids={input_ids.shape}, positions={positions.shape}, kv_cache={kv_cache.shape}")
    else:
        # Standard format: forward(input_ids)
        vocab_size = config.vocab_size
        example_inputs = (torch.randint(0, vocab_size, (args.batch_size, args.seq_len)),)
    
    export_config = ExportConfig(
        output_path=output_path,
        backend=BackendType(args.backend),
        quantization=QuantizationType(args.quantization),
        memory_planning="greedy",
        max_seq_len=args.max_seq_len,
    )
    
    # Export
    print(f"\n[3/3] Exporting to ExecuTorch (this may take a while)...")
    exporter = DiffuLexExporter(export_config)
    result = exporter.export(model, example_inputs)
    
    if result.success:
        print(f"\n      Success!")
        print(f"      File: {result.output_path}")
        print(f"      Size: {result.file_size_mb:.2f} MB")
        print(f"      Time: {result.compilation_time_sec:.2f}s")
    else:
        print(f"\n      Failed: {result.error_message}")
        sys.exit(1)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
