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
        choices=["fast_dllm_v2", "dream", "llada", "sdar"],
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
        choices=["none", "fp16", "dynamic_int8"],
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
    print(f"\n[3/3] Exporting to ExecuTorch...")
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
