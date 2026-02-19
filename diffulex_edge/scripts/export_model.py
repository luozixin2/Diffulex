#!/usr/bin/env python3
"""Command-line tool to export DiffuLex models to ExecuTorch format.

Usage:
    python -m diffulex_edge.scripts.export_model \
        --output model.pte \
        --hidden-size 512 \
        --num-layers 4 \
        --quantization dynamic_int8
"""

import argparse
import dataclasses
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.export import ExportConfig, BackendType, QuantizationType, DiffuLexExporter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export DiffuLex model to ExecuTorch .pte format"
    )
    
    # Model configuration
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="model.pte",
        help="Output path for .pte file"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads (for GQA, defaults to num-heads)"
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=None,
        help="FFN intermediate dimension (default: 4 * hidden-size)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Attention head dimension (default: hidden-size // num-heads)"
    )
    
    # Export options
    parser.add_argument(
        "--backend",
        type=str,
        default="reference",
        choices=["reference", "xnnpack", "coreml", "qnn", "mps"],
        help="Target backend"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="none",
        choices=["none", "dynamic_int8", "static_int8", "weight_only_int8", "fp16"],
        help="Quantization type"
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV cache"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to load"
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to calibration data (for static quantization)"
    )
    
    # Input configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for example inputs"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for example inputs"
    )
    
    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()
    
    print("=" * 60)
    print("DiffuLex Edge Model Export")
    print("=" * 60)
    
    # Create model config
    model_config = FastdLLMV2EdgeConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads or args.num_heads,
        head_dim=args.head_dim or (args.hidden_size // args.num_heads),
        intermediate_size=args.intermediate_size or (4 * args.hidden_size),
        max_seq_len=args.max_seq_len,
    )
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Heads: {model_config.num_heads}")
    print(f"  KV Heads: {model_config.num_kv_heads}")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Max seq len: {model_config.max_seq_len}")
    
    # Create export config
    export_config = ExportConfig(
        output_path=Path(args.output),
        backend=BackendType(args.backend),
        quantization=QuantizationType(args.quantization),
        use_kv_cache=not args.no_kv_cache,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        calibration_data_path=Path(args.calibration_data) if args.calibration_data else None,
        **dataclasses.asdict(model_config),
    )
    
    print(f"\nExport Configuration:")
    print(f"  Output: {export_config.output_path}")
    print(f"  Backend: {export_config.backend.value}")
    print(f"  Quantization: {export_config.quantization.value}")
    print(f"  KV Cache: {export_config.use_kv_cache}")
    
    # Create model
    print(f"\nCreating model...")
    model = FastdLLMV2Edge(model_config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Create example inputs
    print(f"\nPreparing example inputs (batch={args.batch_size}, seq={args.seq_len})...")
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    positions = torch.arange(args.seq_len).unsqueeze(0).expand(args.batch_size, -1)
    
    if export_config.use_kv_cache:
        # Create KV cache for example
        kv_cache = torch.zeros(
            model_config.num_layers, 2, args.batch_size,
            model_config.num_kv_heads, model_config.max_seq_len, model_config.head_dim
        )
        example_inputs = (input_ids, positions, None, kv_cache, 0)
    else:
        example_inputs = (input_ids, positions, None, None, 0)
    
    # Export
    print(f"\nStarting export...")
    exporter = DiffuLexExporter(export_config)
    result = exporter.export(model, example_inputs)
    
    # Print result
    print("\n" + "=" * 60)
    if result.success:
        print("Export Successful!")
        print(f"  Output: {result.output_path}")
        print(f"  File size: {result.file_size_mb:.2f} MB")
        print(f"  Time: {result.compilation_time_sec:.2f}s")
        
        # Calculate compression
        param_size_mb = num_params * 4 / (1024 * 1024)  # FP32
        print(f"  Compression ratio: {param_size_mb / result.file_size_mb:.2f}x")
    else:
        print("Export Failed!")
        print(f"  Error: {result.error_message}")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
