"""Weight container factory."""

from __future__ import annotations

from typing import Any, Optional

import torch

from diffulex.utils.quantization.core.container import (
    BF16Weight,
    W8A16Weight,
    W8A8Weight,
    GPTQWeight,
    AWQWeight,
    GPTQMarlinWeight,
    AWQMarlinWeight,
    QuantizedWeight,
)
from diffulex.utils.quantization.core.protocol import WeightFormat


class WeightContainerFactory:
    """Factory for creating QuantizedWeight containers."""
    
    @staticmethod
    def from_bf16(weight: torch.Tensor) -> BF16Weight:
        """Create BF16 weight container.
        
        Args:
            weight: Weight tensor [out_features, in_features]
            
        Returns:
            BF16Weight container
        """
        return BF16Weight(weight)
    
    @staticmethod
    def from_w8a16(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        original_shape: Optional[tuple[int, int]] = None,
    ) -> W8A16Weight:
        """Create W8A16 weight container.
        
        Args:
            qweight: Quantized weight tensor
            scales: Scale tensor
            original_shape: Original shape if qweight is packed/reordered
            
        Returns:
            W8A16Weight container
        """
        qweight = qweight.contiguous()
        scales = scales.contiguous()
        return W8A16Weight(qweight, scales, original_shape)
    
    @staticmethod
    def from_w8a8(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        original_shape: Optional[tuple[int, int]] = None,
    ) -> W8A8Weight:
        """Create W8A8 weight container.
        
        Args:
            qweight: Quantized weight tensor
            scales: Scale tensor
            original_shape: Original shape if qweight is packed/reordered
            
        Returns:
            W8A8Weight container
        """
        qweight = qweight.contiguous()
        scales = scales.contiguous()
        return W8A8Weight(qweight, scales, original_shape)
    
    @staticmethod
    def from_gptq(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: Optional[torch.Tensor],
        out_features: int,
        in_features: int,
        group_size: int = 128,
        bits: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> GPTQWeight:
        """Create GPTQ weight container with validation.
        
        Args:
            qweight: Packed weight tensor [in_features/pack, out_features]
            qzeros: Packed zeros tensor [num_groups, out_features/pack]
            scales: Scales tensor [num_groups, out_features]
            g_idx: Optional group indices [in_features]
            out_features: Output dimension
            in_features: Input dimension
            group_size: Quantization group size
            bits: Quantization bits (inferred if None)
            device: Target device
            
        Returns:
            GPTQWeight container
            
        Raises:
            ValueError: If tensor shapes are invalid
        """
        if bits is None:
            if qweight.shape[0] <= 0 or in_features % int(qweight.shape[0]) != 0:
                raise ValueError(
                    f"Cannot infer GPTQ bits from qweight shape: "
                    f"in_features={in_features}, qweight.shape={tuple(qweight.shape)}"
                )
            pack_factor = in_features // int(qweight.shape[0])
            if 32 % pack_factor != 0:
                raise ValueError(f"Invalid pack_factor={pack_factor} for GPTQ")
            bits = 32 // pack_factor
        
        if bits not in (2, 4, 8):
            raise ValueError(f"GPTQ bits must be 2, 4, or 8, got {bits}")
        
        pack_factor = 32 // bits
        num_groups = in_features // (in_features if group_size == -1 else group_size)
        
        expected_qweight = (in_features // pack_factor, out_features)
        expected_qzeros = (num_groups, out_features // pack_factor)
        expected_scales = (num_groups, out_features)
        
        if qweight.shape != expected_qweight:
            raise ValueError(
                f"GPTQ qweight shape mismatch: got {tuple(qweight.shape)}, "
                f"expected {expected_qweight}"
            )
        if qzeros.shape != expected_qzeros:
            raise ValueError(
                f"GPTQ qzeros shape mismatch: got {tuple(qzeros.shape)}, "
                f"expected {expected_qzeros}"
            )
        if scales.shape != expected_scales:
            raise ValueError(
                f"GPTQ scales shape mismatch: got {tuple(scales.shape)}, "
                f"expected {expected_scales}"
            )
        
        if qweight.dtype != torch.int32:
            raise TypeError(f"GPTQ qweight must be int32, got {qweight.dtype}")
        if qzeros.dtype != torch.int32:
            raise TypeError(f"GPTQ qzeros must be int32, got {qzeros.dtype}")
        if scales.dtype != torch.float16:
            scales = scales.to(torch.float16)
        
        if g_idx is None or g_idx.numel() == 0:
            g_idx = torch.empty(0, dtype=torch.int32)
        else:
            if g_idx.shape != (in_features,):
                raise ValueError(
                    f"GPTQ g_idx shape mismatch: got {g_idx.shape}, expected ({in_features},)"
                )
            g_idx = g_idx.to(dtype=torch.int32).contiguous()
        
        target_device = device or qweight.device
        if qweight.device != target_device:
            qweight = qweight.to(target_device)
        if qzeros.device != target_device:
            qzeros = qzeros.to(target_device)
        if scales.device != target_device:
            scales = scales.to(target_device)
        if g_idx.device != target_device and g_idx.numel() > 0:
            g_idx = g_idx.to(target_device)
        
        qweight = qweight.contiguous()
        qzeros = qzeros.contiguous()
        scales = scales.contiguous()
        
        return GPTQWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bits=bits,
            group_size=group_size,
            out_features=out_features,
            in_features=in_features,
        )
    
    @staticmethod
    def from_awq(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        out_features: int,
        in_features: int,
        group_size: int = 128,
        bits: int = 4,
        device: Optional[torch.device] = None,
    ) -> AWQWeight:
        """Create AWQ weight container with validation.
        
        Args:
            qweight: Packed weight tensor [in_features, out_features/pack]
            qzeros: Packed zeros tensor [num_groups, out_features/pack]
            scales: Scales tensor [num_groups, out_features]
            out_features: Output dimension
            in_features: Input dimension
            group_size: Quantization group size
            bits: Quantization bits (AWQ is typically 4-bit)
            device: Target device
            
        Returns:
            AWQWeight container
        """
        if bits != 4:
            raise ValueError(f"AWQ currently only supports 4-bit, got {bits}")
        
        pack_factor = 32 // bits
        num_groups = in_features // (in_features if group_size == -1 else group_size)
        
        expected_qweight = (in_features, out_features // pack_factor)
        expected_qzeros = (num_groups, out_features // pack_factor)
        expected_scales = (num_groups, out_features)
        
        if qweight.shape != expected_qweight:
            raise ValueError(
                f"AWQ qweight shape mismatch: got {tuple(qweight.shape)}, "
                f"expected {expected_qweight}"
            )
        if qzeros.shape != expected_qzeros:
            raise ValueError(
                f"AWQ qzeros shape mismatch: got {tuple(qzeros.shape)}, "
                f"expected {expected_qzeros}"
            )
        if scales.shape != expected_scales:
            raise ValueError(
                f"AWQ scales shape mismatch: got {tuple(scales.shape)}, "
                f"expected {expected_scales}"
            )
        
        if qweight.dtype != torch.int32:
            raise TypeError(f"AWQ qweight must be int32, got {qweight.dtype}")
        if qzeros.dtype != torch.int32:
            raise TypeError(f"AWQ qzeros must be int32, got {qzeros.dtype}")
        if scales.dtype != torch.float16:
            scales = scales.to(torch.float16)
        
        target_device = device or qweight.device
        if qweight.device != target_device:
            qweight = qweight.to(target_device)
        if qzeros.device != target_device:
            qzeros = qzeros.to(target_device)
        if scales.device != target_device:
            scales = scales.to(target_device)
        
        qweight = qweight.contiguous()
        qzeros = qzeros.contiguous()
        scales = scales.contiguous()
        
        return AWQWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            group_size=group_size,
            out_features=out_features,
            in_features=in_features,
            bits=bits,
        )
    
    @staticmethod
    def quantize_at_runtime(
        weight: torch.Tensor,
        strategy: Any,  # LinearQuantizationProtocol
        device: Optional[torch.device] = None,
    ) -> QuantizedWeight:
        """Quantize a weight tensor at runtime using the provided strategy.
        
        Args:
            weight: Weight tensor to quantize
            strategy: Quantization strategy to use
            device: Target device
            
        Returns:
            QuantizedWeight container
        """
        target_device = device or weight.device
        weight = weight.to(target_device)
        
        weight_format = strategy.weight_format
        
        if weight_format == WeightFormat.BF16:
            return BF16Weight(weight)
        
        elif weight_format == WeightFormat.INT8:
            qweight, scales = strategy.quantize_weight(weight)
            act_format = getattr(strategy, 'linear_act_format', 'bf16')
            if act_format == 'int8':
                return W8A8Weight(qweight, scales, weight.shape)
            else:
                return W8A16Weight(qweight, scales, weight.shape)
        
        else:
            raise NotImplementedError(
                f"Runtime quantization not implemented for format: {weight_format}"
            )
    
    @staticmethod
    def from_gptq_marlin(
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
        device: Optional[torch.device] = None,
    ) -> GPTQMarlinWeight:
        """Create GPTQ Marlin repacked weight container."""
        target_device = device or qweight.device
        
        if qweight.device != target_device:
            qweight = qweight.to(target_device)
        if scales.device != target_device:
            scales = scales.to(target_device)
        if zp.device != target_device and zp.numel() > 0:
            zp = zp.to(target_device)
        if g_idx.device != target_device and g_idx.numel() > 0:
            g_idx = g_idx.to(target_device)
        if g_idx_sort_indices.device != target_device and g_idx_sort_indices.numel() > 0:
            g_idx_sort_indices = g_idx_sort_indices.to(target_device)
        if workspace.device != target_device:
            workspace = workspace.to(target_device)
        
        return GPTQMarlinWeight(
            qweight=qweight.contiguous(),
            scales=scales.contiguous(),
            zp=zp,
            g_idx=g_idx,
            g_idx_sort_indices=g_idx_sort_indices,
            workspace=workspace,
            bits=bits,
            group_size=group_size,
            out_features=out_features,
            in_features=in_features,
        )
    
    @staticmethod
    def from_awq_marlin(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zp: torch.Tensor,
        workspace: torch.Tensor,
        group_size: int,
        out_features: int,
        in_features: int,
        device: Optional[torch.device] = None,
    ) -> AWQMarlinWeight:
        """Create AWQ Marlin repacked weight container."""
        target_device = device or qweight.device
        
        if qweight.device != target_device:
            qweight = qweight.to(target_device)
        if scales.device != target_device:
            scales = scales.to(target_device)
        if zp.device != target_device and zp.numel() > 0:
            zp = zp.to(target_device)
        if workspace.device != target_device:
            workspace = workspace.to(target_device)
        
        return AWQMarlinWeight(
            qweight=qweight.contiguous(),
            scales=scales.contiguous(),
            zp=zp,
            workspace=workspace,
            group_size=group_size,
            out_features=out_features,
            in_features=in_features,
            bits=4,
        )
