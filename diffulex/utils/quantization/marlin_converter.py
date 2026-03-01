"""Marlin format conversion utilities.

This module provides unified conversion from standard GPTQ/AWQ formats
to Marlin-optimized formats. It eliminates duplication between:
- Runtime conversion in QuantizedLinearDelegate
- Loading-time conversion in loader_adapter
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from diffulex.utils.quantization.core import GPTQWeight, AWQWeight, GPTQMarlinWeight, AWQMarlinWeight


def convert_gptq_to_marlin(
    container: "GPTQWeight",
    device: torch.device,
) -> "GPTQMarlinWeight":
    """Convert standard GPTQ weights to Marlin format.
    
    Args:
        container: GPTQ weight container with standard format
        device: Target CUDA device for the converted weights
        
    Returns:
        GPTQMarlinWeight container with repacked weights
        
    Raises:
        RuntimeError: If vLLM Marlin utilities are not available
    """
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_empty_g_idx,
            marlin_make_workspace_new,
            marlin_permute_scales,
            marlin_sort_g_idx,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to import vLLM Marlin utils: {e}") from e
    
    from diffulex.utils.quantization.core import GPTQMarlinWeight
    
    if container.g_idx.numel() > 0:
        g_idx_sorted, g_idx_sort = marlin_sort_g_idx(
            container.g_idx.to(device=device, dtype=torch.int32)
        )
    else:
        g_idx_sorted = marlin_make_empty_g_idx(device)
        g_idx_sort = marlin_make_empty_g_idx(device)
    
    workspace = marlin_make_workspace_new(device)
    
    marlin_qweight = ops.gptq_marlin_repack(
        container.qweight.to(device).contiguous(),
        perm=g_idx_sort,
        size_k=container.in_features,
        size_n=container.out_features,
        num_bits=container.bits,
        is_a_8bit=False,
    )
    
    marlin_scales = marlin_permute_scales(
        container.scales.to(device).contiguous(),
        size_k=container.in_features,
        size_n=container.out_features,
        group_size=container.group_size,
        is_a_8bit=False,
    )
    
    return GPTQMarlinWeight(
        qweight=marlin_qweight.contiguous(),
        scales=marlin_scales.contiguous(),
        zp=marlin_make_empty_g_idx(device),
        g_idx=g_idx_sorted.contiguous(),
        g_idx_sort_indices=g_idx_sort.contiguous(),
        workspace=workspace,
        bits=container.bits,
        group_size=container.group_size,
        out_features=container.out_features,
        in_features=container.in_features,
    )


def convert_awq_to_marlin(
    container: "AWQWeight",
    device: torch.device,
) -> "AWQMarlinWeight":
    """Convert standard AWQ weights to Marlin format.
    
    Args:
        container: AWQ weight container with standard format
        device: Target CUDA device for the converted weights
        
    Returns:
        AWQMarlinWeight container with repacked weights
        
    Raises:
        RuntimeError: If vLLM Marlin utilities are not available
    """
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            awq_to_marlin_zero_points,
            marlin_make_workspace_new,
            marlin_permute_scales,
            marlin_make_empty_g_idx,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to import vLLM Marlin utils: {e}") from e
    
    from diffulex.utils.quantization.core import AWQMarlinWeight
    
    num_groups = container.in_features // container.group_size
    
    workspace = marlin_make_workspace_new(device)
    
    marlin_qweight = ops.awq_marlin_repack(
        container.qweight.to(device).contiguous(),
        size_k=container.in_features,
        size_n=container.out_features,
        num_bits=container.bits,
        is_a_8bit=False,
    )
    
    marlin_scales = marlin_permute_scales(
        container.scales.to(device).contiguous(),
        size_k=container.in_features,
        size_n=container.out_features,
        group_size=container.group_size,
        is_a_8bit=False,
    )
    
    marlin_zp = awq_to_marlin_zero_points(
        container.qzeros.to(device).contiguous(),
        size_k=num_groups,
        size_n=container.out_features,
        num_bits=container.bits,
        is_a_8bit=False,
    )
    
    return AWQMarlinWeight(
        qweight=marlin_qweight.contiguous(),
        scales=marlin_scales.contiguous(),
        zp=marlin_zp.contiguous(),
        workspace=workspace,
        group_size=container.group_size,
        out_features=container.out_features,
        in_features=container.in_features,
        bits=container.bits,
    )


def convert_gptq_tensors_to_marlin(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
    device: torch.device,
) -> "GPTQMarlinWeight":
    """Convert raw GPTQ tensors to Marlin format (for loading-time conversion).
    
    This is a convenience wrapper that creates a temporary GPTQWeight container
    and converts it to Marlin format.
    
    Args:
        qweight: Packed quantized weights
        scales: Quantization scales
        g_idx: Group indices tensor
        bits: Quantization bits (2, 4, or 8)
        group_size: Group size for quantization
        out_features: Output features dimension
        in_features: Input features dimension
        device: Target device
        
    Returns:
        GPTQMarlinWeight container with repacked weights
    """
    from diffulex.utils.quantization.core import GPTQWeight
    
    temp_container = GPTQWeight(
        qweight=qweight,
        qzeros=torch.empty(0, dtype=torch.int32, device=qweight.device),
        scales=scales,
        g_idx=g_idx if g_idx is not None else torch.empty(0, dtype=torch.int32),
        bits=bits,
        group_size=group_size,
        out_features=out_features,
        in_features=in_features,
    )
    return convert_gptq_to_marlin(temp_container, device)


def convert_awq_tensors_to_marlin(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
    device: torch.device,
) -> "AWQMarlinWeight":
    """Convert raw AWQ tensors to Marlin format (for loading-time conversion).
    
    Args:
        qweight: Packed quantized weights
        scales: Quantization scales
        qzeros: Packed zero points
        bits: Quantization bits
        group_size: Group size for quantization
        out_features: Output features dimension
        in_features: Input features dimension
        device: Target device
        
    Returns:
        AWQMarlinWeight container with repacked weights
    """
    from diffulex.utils.quantization.core import AWQWeight
    
    temp_container = AWQWeight(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        group_size=group_size,
        out_features=out_features,
        in_features=in_features,
        bits=bits,
    )
    return convert_awq_to_marlin(temp_container, device)
