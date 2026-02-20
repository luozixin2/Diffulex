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

from diffulex.utils.quantization.core import WeightContainerFactory
from diffulex.utils.quantization.delegate import QuantizedLinearDelegate

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
) -> tuple[int, int]:
    """Infer (out_features, in_features) from tensors."""
    if format == "gptq":
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
    out_features, in_features = _infer_shape_info(qweight, scales, format, bits)
    
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
            container = WeightContainerFactory.from_gptq_marlin(
                qweight=qweight,
                scales=scales,
                zp=torch.empty(0, dtype=torch.int32, device=qweight.device),
                g_idx=g_idx if g_idx is not None else torch.empty(0, dtype=torch.int32, device=qweight.device),
                g_idx_sort_indices=torch.empty(0, dtype=torch.int32, device=qweight.device),
                workspace=torch.empty(0, dtype=torch.int32, device=qweight.device),
                bits=bits,
                group_size=group_size,
                out_features=out_features,
                in_features=in_features,
                device=device,
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
