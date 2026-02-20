"""Quantized weight container implementations."""

from __future__ import annotations

from typing import Any, Optional
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.core.protocol import WeightFormat


class QuantizedWeight(ABC):
    """Abstract base class for quantized weight containers."""
    
    @property
    @abstractmethod
    def weight_format(self) -> WeightFormat:
        """Return the format identifier."""
        pass
    
    @property
    @abstractmethod
    def out_features(self) -> int:
        """Output feature dimension."""
        pass
    
    @property
    @abstractmethod
    def in_features(self) -> int:
        """Input feature dimension."""
        pass
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        strategy: Any,  # LinearQuantizationProtocol
    ) -> torch.Tensor:
        """Execute forward pass using the provided strategy."""
        pass
    
    @abstractmethod
    def to(self, device: torch.device) -> QuantizedWeight:
        """Move to device and return a new container (or self if already there)."""
        pass
    
    def prepare(self) -> None:
        """Lazy preparation hook. Override if needed."""
        pass


class BF16Weight(QuantizedWeight):
    """Standard BF16/FP16 weight (no quantization)."""
    
    def __init__(self, weight: torch.Tensor):
        self.weight = weight
    
    @property
    def weight_format(self) -> WeightFormat:
        return WeightFormat.BF16
    
    @property
    def out_features(self) -> int:
        return self.weight.shape[0]
    
    @property
    def in_features(self) -> int:
        return self.weight.shape[1]
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        strategy: Any,
    ) -> torch.Tensor:
        return F.linear(x, self.weight, bias)
    
    def to(self, device: torch.device) -> QuantizedWeight:
        if self.weight.device == device:
            return self
        return BF16Weight(self.weight.to(device))


class W8A16Weight(QuantizedWeight):
    """W8A16 quantized weight (int8 weight + bf16 activation)."""
    
    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        original_shape: Optional[tuple[int, int]] = None,
    ):
        """
        Args:
            qweight: Quantized weight tensor [out_features, in_features] or packed format
            scales: Scale tensor [out_features] or [1, out_features]
            original_shape: Original shape before packing (if different)
        """
        self.qweight = qweight
        self.scales = scales
        self._original_shape = original_shape
    
    @property
    def weight_format(self) -> WeightFormat:
        return WeightFormat.INT8
    
    @property
    def out_features(self) -> int:
        if self._original_shape:
            return self._original_shape[0]
        if self.qweight.dim() == 2:
            return self.qweight.shape[0]
        return self.scales.shape[-1]
    
    @property
    def in_features(self) -> int:
        if self._original_shape:
            return self._original_shape[1]
        if self.qweight.dim() == 2:
            return self.qweight.shape[1]
        return 0
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        strategy: Any,
    ) -> torch.Tensor:
        return strategy.linear_forward(
            x,
            self,
            bias,
            quant_scales=self.scales,
            out_features=self.out_features,
        )
    
    def to(self, device: torch.device) -> QuantizedWeight:
        if self.qweight.device == device:
            return self
        return W8A16Weight(
            self.qweight.to(device),
            self.scales.to(device),
            self._original_shape,
        )


class W8A8Weight(QuantizedWeight):
    """W8A8 quantized weight (int8 weight + int8 activation)."""
    
    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        original_shape: Optional[tuple[int, int]] = None,
    ):
        self.qweight = qweight
        self.scales = scales
        self._original_shape = original_shape
    
    @property
    def weight_format(self) -> WeightFormat:
        return WeightFormat.INT8
    
    @property
    def out_features(self) -> int:
        if self._original_shape:
            return self._original_shape[0]
        return self.qweight.shape[0] if self.qweight.dim() == 2 else self.scales.shape[-1]
    
    @property
    def in_features(self) -> int:
        if self._original_shape:
            return self._original_shape[1]
        return self.qweight.shape[1] if self.qweight.dim() == 2 else 0
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        strategy: Any,
    ) -> torch.Tensor:
        return strategy.linear_forward(
            x,
            self,
            bias,
            quant_scales=self.scales,
            out_features=self.out_features,
        )
    
    def to(self, device: torch.device) -> QuantizedWeight:
        if self.qweight.device == device:
            return self
        return W8A8Weight(
            self.qweight.to(device),
            self.scales.to(device),
            self._original_shape,
        )


class GPTQWeight(QuantizedWeight):
    """GPTQ offline quantized weight (W4A16 or similar)."""
    
    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        bits: int,
        group_size: int,
        out_features: int,
        in_features: int,
    ):
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx
        self.bits = bits
        self.group_size = group_size
        self._out_features = out_features
        self._in_features = in_features
        self._is_prepared = False
    
    @property
    def weight_format(self) -> WeightFormat:
        return WeightFormat.GPTQ
    
    @property
    def out_features(self) -> int:
        return self._out_features
    
    @property
    def in_features(self) -> int:
        return self._in_features
    
    def prepare(self) -> None:
        """Lazy preparation: perform gptq_shuffle on first use."""
        if self._is_prepared:
            return
        
        try:
            from vllm import _custom_ops as ops
        except Exception as e:
            raise RuntimeError(
                "GPTQ requires vLLM CUDA custom ops but they are not available."
            ) from e
        
        if self.g_idx.numel() == 0:
            g_idx = torch.empty((0,), device=self.qweight.device, dtype=torch.int)
        else:
            g_idx = self.g_idx.to(dtype=torch.int)
        
        ops.gptq_shuffle(self.qweight, g_idx, self.bits)
        self._is_prepared = True
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        strategy: Any,
    ) -> torch.Tensor:
        self.prepare()
        return strategy.linear_forward(
            x,
            self,
            bias,
            gptq_qweight=self.qweight,
            gptq_qzeros=self.qzeros,
            gptq_scales=self.scales,
            gptq_g_idx=self.g_idx,
            weight_bits=self.bits,
            out_features=self.out_features,
            in_features=self.in_features,
            group_size=self.group_size,
        )
    
    def to(self, device: torch.device) -> QuantizedWeight:
        if self.qweight.device == device:
            return self
        return GPTQWeight(
            self.qweight.to(device),
            self.qzeros.to(device),
            self.scales.to(device),
            self.g_idx.to(device) if self.g_idx.numel() > 0 else self.g_idx,
            self.bits,
            self.group_size,
            self._out_features,
            self._in_features,
        )


class AWQWeight(QuantizedWeight):
    """AWQ offline quantized weight (W4A16)."""
    
    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        out_features: int,
        in_features: int,
        bits: int = 4,
    ):
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.group_size = group_size
        self.bits = bits
        self._out_features = out_features
        self._in_features = in_features
        self._pack_factor = 32 // bits
    
    @property
    def weight_format(self) -> WeightFormat:
        return WeightFormat.AWQ
    
    @property
    def out_features(self) -> int:
        return self._out_features
    
    @property
    def in_features(self) -> int:
        return self._in_features
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        strategy: Any,
    ) -> torch.Tensor:
        return strategy.linear_forward(
            x,
            self,
            bias,
            awq_qweight=self.qweight,
            awq_qzeros=self.qzeros,
            awq_scales=self.scales,
            pack_factor=self._pack_factor,
            out_features=self.out_features,
            in_features=self.in_features,
            group_size=self.group_size,
        )
    
    def to(self, device: torch.device) -> QuantizedWeight:
        if self.qweight.device == device:
            return self
        return AWQWeight(
            self.qweight.to(device),
            self.qzeros.to(device),
            self.scales.to(device),
            self.group_size,
            self._out_features,
            self._in_features,
            self.bits,
        )


class GPTQMarlinWeight(QuantizedWeight):
    """GPTQ Marlin repacked weight."""
    
    def __init__(
        self,
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
    ):
        self.qweight = qweight
        self.scales = scales
        self.zp = zp
        self.g_idx = g_idx
        self.g_idx_sort_indices = g_idx_sort_indices
        self.workspace = workspace
        self.bits = bits
        self.group_size = group_size
        self._out_features = out_features
        self._in_features = in_features
    
    @property
    def weight_format(self) -> WeightFormat:
        return WeightFormat.GPTQ_MARLIN
    
    @property
    def out_features(self) -> int:
        return self._out_features
    
    @property
    def in_features(self) -> int:
        return self._in_features
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        strategy: Any,
    ) -> torch.Tensor:
        return strategy.linear_forward(
            x,
            self,
            bias,
            qweight=self.qweight,
            scales=self.scales,
            zp=self.zp,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            in_features=self.in_features,
            out_features=self.out_features,
            group_size=self.group_size,
            weight_bits=self.bits,
        )
    
    def to(self, device: torch.device) -> QuantizedWeight:
        if self.qweight.device == device:
            return self
        return GPTQMarlinWeight(
            self.qweight.to(device),
            self.scales.to(device),
            self.zp.to(device) if self.zp.numel() > 0 else self.zp,
            self.g_idx.to(device) if self.g_idx.numel() > 0 else self.g_idx,
            self.g_idx_sort_indices.to(device) if self.g_idx_sort_indices.numel() > 0 else self.g_idx_sort_indices,
            self.workspace.to(device),
            self.bits,
            self.group_size,
            self._out_features,
            self._in_features,
        )


class AWQMarlinWeight(QuantizedWeight):
    """AWQ Marlin repacked weight."""
    
    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zp: torch.Tensor,
        workspace: torch.Tensor,
        group_size: int,
        out_features: int,
        in_features: int,
        bits: int = 4,
    ):
        self.qweight = qweight
        self.scales = scales
        self.zp = zp
        self.workspace = workspace
        self.group_size = group_size
        self.bits = bits
        self._out_features = out_features
        self._in_features = in_features
    
    @property
    def weight_format(self) -> WeightFormat:
        return WeightFormat.AWQ_MARLIN
    
    @property
    def out_features(self) -> int:
        return self._out_features
    
    @property
    def in_features(self) -> int:
        return self._in_features
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        strategy: Any,
    ) -> torch.Tensor:
        return strategy.linear_forward(
            x,
            self,
            bias,
            qweight=self.qweight,
            scales=self.scales,
            zp=self.zp,
            workspace=self.workspace,
            in_features=self.in_features,
            out_features=self.out_features,
            group_size=self.group_size,
        )
    
    def to(self, device: torch.device) -> QuantizedWeight:
        if self.qweight.device == device:
            return self
        return AWQMarlinWeight(
            self.qweight.to(device),
            self.scales.to(device),
            self.zp.to(device),
            self.workspace.to(device),
            self.group_size,
            self._out_features,
            self._in_features,
            self.bits,
        )
