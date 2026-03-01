"""Loader adapter for the quantization system.

Handles quantization-specific weight loading:
- Shape inference for GPTQ/AWQ formats
- Group size inference
- Dummy qzeros creation for marlin
- Weight container creation
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from diffulex.utils.quantization.core import WeightContainerFactory, GPTQWeight, AWQWeight, GPTQMarlinWeight, AWQMarlinWeight
from diffulex.utils.quantization.delegate import QuantizedLinearDelegate

# Optional vLLM import for Marlin
try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace,
    )
except Exception:
    marlin_make_workspace = None

if TYPE_CHECKING:
    from diffulex.layer.linear import LinearBase


def _ensure_delegate(module: "LinearBase") -> QuantizedLinearDelegate:
    """Ensure module has a QuantizedLinearDelegate."""
    if not module.has_delegate():
        from diffulex.utils.quantization import create_quantized_delegate
        module.set_delegate(create_quantized_delegate(module.quant_kind))
    
    delegate = module.get_delegate()
    if not isinstance(delegate, QuantizedLinearDelegate):
        raise RuntimeError(f"Module {module} does not have a QuantizedLinearDelegate")
    return delegate


def _infer_module_device(module: nn.Module) -> torch.device:
    """Infer the device of a module."""
    w = getattr(module, "weight", None)
    if isinstance(w, torch.Tensor):
        return w.device
    for p in module.parameters(recurse=False):
        return p.device
    for b in module.buffers(recurse=False):
        return b.device
    return torch.device("cpu")


def _infer_shape_info(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    format: str,
    bits: int | None,
    is_marlin: bool = False,
) -> tuple[int, int]:
    """Infer (out_features, in_features) from tensors."""
    if format == "gptq":
        if is_marlin:
            # Marlin format: qweight shape is [in_features/16, out_features*(16//bits)/16]
            # For 4-bit: qweight is [K/16, N*2] -> in_features = K/16 * 16, out_features = N*2 / 2
            # For 2-bit: qweight is [K/16, N]   -> in_features = K/16 * 16, out_features = N
            in_features = int(qweight.shape[0]) * 16
            if bits == 4:
                out_features = int(qweight.shape[1]) // 2
            elif bits == 2:
                out_features = int(qweight.shape[1])
            else:  # 8-bit
                out_features = int(qweight.shape[1]) * 2
        else:
            # Standard GPTQ format
            out_features = int(qweight.shape[1])
            pack_factor = (32 // bits) if bits else 8
            in_features = int(qweight.shape[0]) * pack_factor
    else:  # awq
        out_features = int(scales.shape[1]) if scales.ndim == 2 else int(qweight.shape[1])
        in_features = int(qweight.shape[0])
    
    return out_features, in_features


def _infer_group_size(
    in_features: int,
    qzeros: Optional[torch.Tensor],
    scales: torch.Tensor,
) -> int:
    """Infer group_size from tensor shapes."""
    num_groups = int(qzeros.shape[0]) if (qzeros is not None and qzeros.numel() > 0) else int(scales.shape[0])
    if num_groups > 0 and in_features % num_groups == 0:
        return in_features // num_groups
    return 128


def _make_packed_qzeros_constant(
    num_groups: int,
    out_features: int,
    bits: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Create a GPTQ-style packed qzeros tensor filled with a constant."""
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bits={bits}")
    pack_factor = 32 // bits
    out_packed = out_features // pack_factor
    
    z = (1 << (bits - 1)) - 1
    packed_val = sum((z & ((1 << bits) - 1)) << (bits * i) for i in range(pack_factor))
    
    return torch.full((num_groups, out_packed), packed_val, dtype=torch.int32, device=device)


def load_offline_quantized_weight(
    module: "LinearBase",
    *,
    qweight: torch.Tensor,
    qzeros: Optional[torch.Tensor],
    scales: torch.Tensor,
    g_idx: Optional[torch.Tensor],
    format: str,
    bits: int,
    group_size: int | None,
    is_marlin: bool,
) -> None:
    """Load offline quantized weight (GPTQ or AWQ) into module.
    
    Handles shape inference, group_size inference, dummy qzeros for marlin,
    and weight container creation. Tensors should already be TP-sharded.
    """
    # Infer shapes
    out_features, in_features = _infer_shape_info(qweight, scales, format, bits, is_marlin)
    
    # Infer group_size if not provided
    if group_size is None or group_size <= 0:
        group_size = _infer_group_size(in_features, qzeros, scales)
    
    # Handle empty qzeros for marlin
    if format == "gptq" and (qzeros is None or qzeros.numel() == 0) and is_marlin and bits in (2, 4, 8):
        num_groups = in_features // (group_size if group_size > 0 else in_features)
        qzeros = _make_packed_qzeros_constant(num_groups, out_features, bits, qweight.device)
    
    # Clear empty g_idx
    if g_idx is not None and g_idx.numel() == 0:
        g_idx = None
    
    # Set weight via delegate
    delegate = _ensure_delegate(module)
    
    if format == "gptq":
        if is_marlin:
            # For marlin, create container directly
            device = _infer_module_device(module)
            target_device = device if device.type == "cuda" else torch.device("cuda")
            
            # Create workspace for Marlin kernel (required by vLLM)
            # workspace size depends on output features (N dimension)
            if marlin_make_workspace is not None:
                workspace = marlin_make_workspace(out_features, device=target_device)
            else:
                # Fallback: create empty workspace (will fail at runtime if vLLM is available)
                workspace = torch.empty(0, dtype=torch.int32, device=target_device)
            
            container = WeightContainerFactory.from_gptq_marlin(
                qweight=qweight,
                scales=scales,
                zp=torch.empty(0, dtype=torch.int32, device=target_device),
                g_idx=g_idx if g_idx is not None else torch.empty(0, dtype=torch.int32, device=target_device),
                g_idx_sort_indices=torch.empty(0, dtype=torch.int32, device=target_device),
                workspace=workspace,
                bits=bits,
                group_size=group_size,
                out_features=out_features,
                in_features=in_features,
                device=target_device,
            )
            delegate.set_container(container)
        else:
            delegate.set_offline_quantized_weight(
                format="gptq",
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                out_features=out_features,
                in_features=in_features,
                group_size=group_size,
            )
    else:  # awq
        delegate.set_offline_quantized_weight(
            format="awq",
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            out_features=out_features,
            in_features=in_features,
            group_size=group_size,
        )


# Backward compatible aliases
def set_offline_gptq_weight(
    module: "LinearBase",
    *,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: Optional[torch.Tensor],
    out_features: int,
    in_features: int,
    group_size: int = 128,
    bits: Optional[int] = None,
) -> None:
    """Set GPTQ offline quantized weight."""
    load_offline_quantized_weight(
        module=module,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        g_idx=g_idx,
        format="gptq",
        bits=bits or 4,
        group_size=group_size,
        is_marlin=False,
    )


def set_offline_awq_weight(
    module: "LinearBase",
    *,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int = 128,
    bits: int = 4,
) -> None:
    """Set AWQ offline quantized weight."""
    load_offline_quantized_weight(
        module=module,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        g_idx=None,
        format="awq",
        bits=bits,
        group_size=group_size,
        is_marlin=False,
    )


def set_offline_gptq_marlin_weight(
    module: "LinearBase",
    *,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
) -> None:
    """Set GPTQ Marlin repacked weight."""
    delegate = _ensure_delegate(module)
    device = _infer_module_device(module)
    
    container = WeightContainerFactory.from_gptq_marlin(
        qweight=qweight,
        scales=scales,
        zp=zp,
        g_idx=g_idx,
        g_idx_sort_indices=g_idx_sort_indices,
        workspace=workspace,
        bits=bits,
        group_size=group_size,
        out_features=out_features,
        in_features=in_features,
        device=device,
    )
    delegate.set_container(container)


def set_offline_awq_marlin_weight(
    module: "LinearBase",
    *,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zp: torch.Tensor,
    workspace: torch.Tensor,
    group_size: int,
    out_features: int,
    in_features: int,
) -> None:
    """Set AWQ Marlin repacked weight."""
    delegate = _ensure_delegate(module)
    device = _infer_module_device(module)
    
    container = WeightContainerFactory.from_awq_marlin(
        qweight=qweight,
        scales=scales,
        zp=zp,
        workspace=workspace,
        group_size=group_size,
        out_features=out_features,
        in_features=in_features,
        device=device,
    )
    delegate.set_container(container)


def prepare_gptq_marlin_from_standard(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
    device: torch.device,
) -> GPTQMarlinWeight:
    """Prepare GPTQ Marlin weight container from standard GPTQ tensors.
    
    This function repacks standard GPTQ weights into Marlin format.
    
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
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_empty_g_idx,
            marlin_make_workspace_new,
            marlin_permute_scales,
            marlin_sort_g_idx,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to import vllm Marlin utils: {e}")
    
    if g_idx.numel() > 0:
        g_idx_sorted, g_idx_sort = marlin_sort_g_idx(
            g_idx.to(device=device, dtype=torch.int32)
        )
    else:
        g_idx_sorted = marlin_make_empty_g_idx(device)
        g_idx_sort = marlin_make_empty_g_idx(device)
    
    workspace = marlin_make_workspace_new(device)
    
    marlin_qweight = ops.gptq_marlin_repack(
        qweight.to(device).contiguous(),
        perm=g_idx_sort,
        size_k=in_features,
        size_n=out_features,
        num_bits=bits,
        is_a_8bit=False,
    )
    
    marlin_scales = marlin_permute_scales(
        scales.to(device).contiguous(),
        size_k=in_features,
        size_n=out_features,
        group_size=group_size,
        is_a_8bit=False,
    )
    
    return GPTQMarlinWeight(
        qweight=marlin_qweight.contiguous(),
        scales=marlin_scales.contiguous(),
        zp=marlin_make_empty_g_idx(device),
        g_idx=g_idx_sorted.contiguous(),
        g_idx_sort_indices=g_idx_sort.contiguous(),
        workspace=workspace,
        bits=bits,
        group_size=group_size,
        out_features=out_features,
        in_features=in_features,
    )


def prepare_awq_marlin_from_standard(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
    device: torch.device,
) -> AWQMarlinWeight:
    """Prepare AWQ Marlin weight container from standard AWQ tensors.
    
    This function repacks standard AWQ weights into Marlin format.
    
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
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            awq_to_marlin_zero_points,
            marlin_make_workspace_new,
            marlin_permute_scales,
            marlin_make_empty_g_idx,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to import vllm Marlin utils: {e}")
    
    num_groups = in_features // group_size
    
    workspace = marlin_make_workspace_new(device)
    
    marlin_qweight = ops.awq_marlin_repack(
        qweight.to(device).contiguous(),
        size_k=in_features,
        size_n=out_features,
        num_bits=bits,
        is_a_8bit=False,
    )
    
    marlin_scales = marlin_permute_scales(
        scales.to(device).contiguous(),
        size_k=in_features,
        size_n=out_features,
        group_size=group_size,
        is_a_8bit=False,
    )
    
    marlin_zp = awq_to_marlin_zero_points(
        qzeros.to(device).contiguous(),
        size_k=num_groups,
        size_n=out_features,
        num_bits=bits,
        is_a_8bit=False,
    )
    
    return AWQMarlinWeight(
        qweight=marlin_qweight.contiguous(),
        scales=marlin_scales.contiguous(),
        zp=marlin_zp.contiguous(),
        workspace=workspace,
        group_size=group_size,
        out_features=out_features,
        in_features=in_features,
        bits=bits,
    )
