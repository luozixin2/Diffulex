#!/usr/bin/env python3
"""离线量化脚本：将模型权重量化为 vLLM 标准 GPTQ/AWQ 格式

支持两种量化格式（对齐 vLLM 权重格式）：
- GPTQ: qweight/qzeros 为 int32 packed，scales 为 fp16，g_idx 可选（常见 desc_act=False 时为空）
- GPTQ_MARLIN: 导出 Marlin-ready 的 GPTQ 权重布局（qweight 已 repack，scales 已 permute，zp 为空）
- AWQ : qweight/qzeros 为 int32 packed，scales 为 fp16

使用方法:
    python -m diffulex.utils.quantization.quantize_model \
        --model-path /path/to/model \
        --output-path /path/to/output \
        --quant-format gptq_marlin \
        --group-size 128 \
        --bits 4
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import save_file

# Import model loading utilities
import sys
from pathlib import Path as PathLib

# Add project root to path
_REPO_ROOT = PathLib(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoConfig, AutoModelForCausalLM
from safetensors import safe_open
from glob import glob


def _require_vllm():
    try:
        from vllm.scalar_type import scalar_types  # type: ignore
        from vllm.model_executor.layers.quantization.utils.quant_utils import (  # type: ignore
            awq_pack,
            gptq_pack,
            pack_cols,
            quantize_weights,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "离线 GPTQ/AWQ 打包已切换到 vLLM 标准格式，需要可 import 的 vLLM。"
        ) from e
    return scalar_types, quantize_weights, gptq_pack, awq_pack, pack_cols


def _require_vllm_marlin():
    # Marlin 预处理依赖 CUDA custom ops
    try:
        from vllm import _custom_ops as ops  # type: ignore
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
            marlin_permute_scales,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "导出 gptq_marlin 格式需要可 import 的 vLLM Marlin（含 CUDA custom ops）。"
        ) from e
    return ops, marlin_permute_scales


def _quantize_to_vllm_gptq(
    weight: torch.Tensor, *, group_size: int, bits: int, use_v2_format: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize and pack weights into vLLM GPTQ checkpoint format.

    Input:
      weight: fp32 [N, K] (PyTorch Linear weight)
    Output (vLLM format):
      qweight: int32 [K/pack, N]
      qzeros : int32 [K/group, N/pack]   (GPTQ v1 stores (zeros - 1); v2 stores zeros)
      scales : fp16  [K/group, N]
      g_idx  : int32 empty tensor (desc_act=False)
    """
    scalar_types, quantize_weights, gptq_pack, _, pack_cols = _require_vllm()
    # vLLM GPTQConfig mentions 2/3/4/8, but the standard vLLM int32 packing
    # used by `gptq_pack/pack_cols` requires 32 % bits == 0.
    # So we support 2/4/8 here; 3-bit would need a different packing scheme.
    if bits not in (2, 4, 8):
        raise ValueError(
            f"GPTQ bits 仅支持 2/4/8（vLLM 标准 int32 pack 要求 32%bits==0），当前 bits={bits}"
        )

    # vLLM operates on (K, N)
    w = weight.T.contiguous()
    size_k, size_n = w.shape
    group_size_norm = size_k if group_size == -1 else group_size
    if group_size_norm <= 0 or size_k % group_size_norm != 0:
        raise ValueError(f"Invalid group_size={group_size} for in_features={size_k}")

    if bits == 2:
        quant_type = scalar_types.uint2b2
    elif bits == 4:
        quant_type = scalar_types.uint4b8
    else:  # bits == 8
        quant_type = scalar_types.uint8b128

    _, w_q, w_s, _ = quantize_weights(w, quant_type, group_size_norm, zero_points=False)

    pack_factor = 32 // bits
    qweight = gptq_pack(w_q, bits, size_k, size_n).contiguous()  # [K/pack, N]

    num_groups = size_k // group_size_norm
    zeros = torch.full(
        (num_groups, size_n),
        int(getattr(quant_type, "bias", 0)),
        dtype=torch.int32,
        device=w.device,
    )
    # GPTQ v1 stores zeros-1 in the checkpoint.
    zeros_to_store = zeros if use_v2_format else (zeros - 1)
    qzeros = pack_cols(zeros_to_store, bits, num_groups, size_n).contiguous()  # [K/group, N/pack]

    scales = w_s.to(torch.float16).contiguous()  # [K/group, N]
    g_idx = torch.empty((0,), dtype=torch.int32, device=w.device)
    return qweight, qzeros, scales, g_idx


def _quantize_to_vllm_gptq_marlin(
    weight: torch.Tensor, *, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weights and export marlin-ready GPTQ layout.

    该导出格式对齐 vLLM `MarlinLinearKernel.process_weights_after_loading` 的结果：
    - qweight: 已执行 `gptq_marlin_repack`
    - scales : 已执行 `marlin_permute_scales`
    - qzeros : 置空（Marlin GPTQ symmetric 路径不使用 runtime zp）
    - g_idx  : 空（desc_act=False）

    注意：需要在 CUDA 上执行（`gptq_marlin_repack` 为 CUDA op）。
    """
    if weight.device.type != "cuda":
        raise ValueError("gptq_marlin 导出需要 device=cuda（Marlin repack 为 CUDA op）")

    ops, marlin_permute_scales = _require_vllm_marlin()

    # 先按 vLLM 标准 GPTQ（symmetric, zero_points=False）量化并打包
    qweight, _qzeros, scales, g_idx = _quantize_to_vllm_gptq(
        weight, group_size=group_size, bits=bits, use_v2_format=False
    )

    # vLLM GPTQ packing 的 shape 基于 w=(K,N)；这里 size_k=in_features, size_n=out_features
    size_k = weight.shape[1]
    size_n = weight.shape[0]
    group_size_norm = size_k if group_size == -1 else group_size

    # desc_act=False 时 perm 为空
    empty_perm = torch.empty((0,), dtype=torch.int32, device=weight.device)

    marlin_qweight = ops.gptq_marlin_repack(
        qweight.contiguous(),
        perm=empty_perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=bits,
        is_a_8bit=False,
    ).contiguous()

    marlin_scales = marlin_permute_scales(
        scales.contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=group_size_norm,
        is_a_8bit=False,
    ).contiguous()

    # Marlin GPTQ symmetric 不使用 runtime zero points，导出空 qzeros 保持一致性
    marlin_qzeros = torch.empty((0,), dtype=torch.int32, device=weight.device)
    marlin_g_idx = g_idx  # already empty

    return marlin_qweight, marlin_qzeros, marlin_scales, marlin_g_idx


def _quantize_to_vllm_awq(
    weight: torch.Tensor, *, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize and pack weights into vLLM AWQ checkpoint format.

    Input:
      weight: fp32 [N, K]
    Output (vLLM format):
      qweight: int32 [K, N/pack]
      qzeros : int32 [K/group, N/pack]
      scales : fp16  [K/group, N]
    """
    scalar_types, quantize_weights, _, awq_pack, _ = _require_vllm()
    if bits != 4:
        raise ValueError(f"AWQ 目前仅支持 4-bit，当前 bits={bits}")

    w = weight.T.contiguous()
    size_k, size_n = w.shape
    group_size_norm = size_k if group_size == -1 else group_size
    if group_size_norm <= 0 or size_k % group_size_norm != 0:
        raise ValueError(f"Invalid group_size={group_size} for in_features={size_k}")

    quant_type = scalar_types.uint4
    _, w_q, w_s, w_zp = quantize_weights(w, quant_type, group_size_norm, zero_points=True)
    if w_zp is None:
        raise RuntimeError("AWQ zero_points=True 但未生成 zero points，vLLM 量化返回异常。")

    qweight = awq_pack(w_q, bits, size_k, size_n).contiguous()  # [K, N/pack]
    num_groups = size_k // group_size_norm
    qzeros = awq_pack(w_zp.to(torch.int32), bits, num_groups, size_n).contiguous()  # [K/group, N/pack]
    scales = w_s.to(torch.float16).contiguous()  # [K/group, N]
    return qweight, qzeros, scales


def quantize_model(
    model_path: str,
    output_path: str,
    quant_format: str = "gptq",
    group_size: int = 128,
    bits: int = 4,
    target_modules: Optional[list[str]] = None,
    device: str = "cpu",
) -> None:
    """Quantize model weights to GPTQ/AWQ format.
    
    Args:
        model_path: Path to input model directory (containing safetensors files)
        output_path: Path to output directory (will create if not exists)
        quant_format: "gptq" or "awq"
        group_size: Group size for quantization (default: 128)
        bits: Number of bits per weight (default: 4)
        target_modules: List of module name patterns to quantize (e.g., ["q_proj", "k_proj"]).
                       If None, quantizes all linear layers.
        device: Device to use for quantization ("cpu" or "cuda")
    """
    if quant_format not in ["gptq", "gptq_marlin", "awq"]:
        raise ValueError(
            f"Unsupported quant_format: {quant_format}. Must be 'gptq', 'gptq_marlin' or 'awq'"
        )
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model weights from safetensors files
    safetensors_files = list(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    
    print(f"Found {len(safetensors_files)} safetensors files")
    
    # Collect all weight names
    all_weight_keys = []
    for file in safetensors_files:
        with safe_open(file, "pt", device) as f:
            all_weight_keys.extend(f.keys())
    
    # Filter to linear layer weights only (exclude biases and non-linear layers)
    linear_weight_keys = []
    for key in all_weight_keys:
        # Skip biases, layer norms, embeddings, etc.
        # Note: lm_head is excluded because ParallelLMHead doesn't support offline quantization yet
        if any(skip in key for skip in [".bias", ".norm", ".embed", ".lm_head"]):
            continue
        # Only process weight parameters
        if not key.endswith(".weight"):
            continue
        # Check if target_modules filter applies
        if target_modules:
            if not any(target in key for target in target_modules):
                continue
        linear_weight_keys.append(key)
    
    print(f"Found {len(linear_weight_keys)} linear layer weights to quantize")
    
    # Quantize each linear layer
    quantized_weights = {}
    metadata = {
        "quant_format": quant_format,
        "group_size": group_size,
        "bits": bits,
        "quantized_modules": [],
    }
    
    for key in tqdm(linear_weight_keys, desc="Quantizing weights"):
        # Load weight from safetensors
        weight = None
        source_file = None
        for file in safetensors_files:
            with safe_open(file, "pt", device) as f:
                if key in f.keys():
                    weight = f.get_tensor(key)
                    source_file = file
                    break
        
        if weight is None:
            print(f"Warning: Could not load weight for {key}")
            continue
        
        # Skip if weight is not 2D (not a linear layer weight)
        if weight.dim() != 2:
            print(f"Skipping {key}: not a 2D weight (shape: {weight.shape})")
            continue
        
        out_features, in_features = weight.shape
        
        # Convert to float32 for quantization
        weight_fp32 = weight.to(torch.float32).to(device)
        
        # Quantize
        prefix = key[:-7]  # Remove ".weight"
        if quant_format == "gptq":
            qweight, qzeros, scales, g_idx = _quantize_to_vllm_gptq(
                weight_fp32, group_size=group_size, bits=bits, use_v2_format=False
            )
        elif quant_format == "gptq_marlin":
            qweight, qzeros, scales, g_idx = _quantize_to_vllm_gptq_marlin(
                weight_fp32, group_size=group_size, bits=bits
            )
            quantized_weights[f"{prefix}.qweight"] = qweight.cpu()
            quantized_weights[f"{prefix}.qzeros"] = qzeros.cpu()
            quantized_weights[f"{prefix}.scales"] = scales.cpu()
            # Keep g_idx key for compatibility (often empty when desc_act=False).
            quantized_weights[f"{prefix}.g_idx"] = g_idx.cpu()
        else:  # awq
            qweight, qzeros, scales = _quantize_to_vllm_awq(
                weight_fp32, group_size=group_size, bits=bits
            )
            quantized_weights[f"{prefix}.qweight"] = qweight.cpu()
            quantized_weights[f"{prefix}.qzeros"] = qzeros.cpu()
            quantized_weights[f"{prefix}.scales"] = scales.cpu()
        
        metadata["quantized_modules"].append({
            "name": prefix,
            "out_features": int(out_features),
            "in_features": int(in_features),
            "group_size": group_size,
            "bits": bits,
        })
        
        # Clear GPU cache if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Copy all model files (config, tokenizer, etc.) to output directory
    import shutil
    print(f"\nCopying model files to {output_path}...")
    model_path_obj = Path(model_path)
    
    # First, copy original safetensors files (for non-quantized layers like lm_head, embeddings, etc.)
    print("  Copying original safetensors files (for non-quantized layers)...")
    for file in model_path_obj.glob("*.safetensors"):
        dest_file = output_path / file.name
        shutil.copy2(file, dest_file)
        print(f"    Copied {file.name}")
    
    # Copy other non-safetensors files
    for file in model_path_obj.iterdir():
        if file.is_file() and not file.name.endswith('.safetensors'):
            dest_file = output_path / file.name
            shutil.copy2(file, dest_file)
            print(f"  Copied {file.name}")
    
    # Save quantized weights to safetensors (this will add quantized weights to the directory)
    output_file = output_path / f"model_quantized_{quant_format}.safetensors"
    print(f"\nSaving quantized weights to {output_file}...")
    save_file(quantized_weights, output_file)
    
    # Save metadata
    metadata_file = output_path / f"quantization_metadata_{quant_format}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # vLLM GPTQ/GPTQ-Marlin 会读取 quantize_config.json
    # - gptq_marlin: 需要 sym/desc_act 等字段用于识别并选择 Marlin kernel
    if quant_format == "gptq_marlin":
        quantize_cfg = {
            "bits": int(bits),
            "group_size": int(group_size),
            "desc_act": False,
            "sym": True,
            "lm_head": False,
            "checkpoint_format": "gptq_marlin",
        }
        with open(output_path / "quantize_config.json", "w") as f:
            json.dump(quantize_cfg, f, indent=2)
    
    print(f"\n✓ Quantization complete!")
    print(f"  - Quantized {len(metadata['quantized_modules'])} modules")
    print(f"  - Output directory: {output_path}")
    print(f"  - Quantized weights file: {output_file}")
    print(f"  - Metadata file: {metadata_file}")
    print(f"\n  You can now use this directory directly as model path:")
    print(f"    --model-path {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="离线量化模型权重为 GPTQ/AWQ 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, required=True, help="输入模型路径")
    parser.add_argument("--output-path", type=str, required=True, help="输出路径")
    parser.add_argument(
        "--quant-format",
        type=str,
        choices=["gptq", "gptq_marlin", "awq"],
        default="gptq",
        help="量化格式: gptq / gptq_marlin / awq",
    )
    parser.add_argument("--group-size", type=int, default=128, help="量化组大小 (默认: 128)")
    parser.add_argument("--bits", type=int, default=4, help="每个权重的位数 (默认: 4)")
    parser.add_argument("--target-modules", type=str, help="要量化的模块名称模式（逗号分隔），例如: q_proj,k_proj,v_proj")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="量化设备 (默认: cpu)")
    
    args = parser.parse_args()
    
    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",")]
    
    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        quant_format=args.quant_format,
        group_size=args.group_size,
        bits=args.bits,
        target_modules=target_modules,
        device=args.device,
    )


if __name__ == "__main__":
    main()
