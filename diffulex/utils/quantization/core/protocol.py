"""Protocol definitions for quantization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol
from enum import Enum

import torch


class WeightFormat(str, Enum):
    """Standard weight format identifiers."""
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    GPTQ = "gptq"
    GPTQ_MARLIN = "gptq_marlin"
    AWQ = "awq"
    AWQ_MARLIN = "awq_marlin"


class QuantizedWeight(Protocol):
    """Protocol for quantized weight containers."""
    
    @property
    def weight_format(self) -> WeightFormat:
        """Return the format identifier."""
        ...
    
    @property
    def out_features(self) -> int:
        """Output feature dimension."""
        ...
    
    @property
    def in_features(self) -> int:
        """Input feature dimension."""
        ...
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        strategy: LinearQuantizationProtocol,
    ) -> torch.Tensor:
        """Execute forward pass using the provided strategy."""
        ...
    
    def to(self, device: torch.device) -> QuantizedWeight:
        """Move to device and return self (for chaining)."""
        ...
    
    def prepare(self) -> None:
        """Lazy preparation (e.g., gptq_shuffle, marlin repack)."""
        ...


class LinearQuantizationProtocol(ABC):
    """Protocol for linear quantization strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @property
    @abstractmethod
    def weight_format(self) -> WeightFormat:
        """The weight format this strategy handles."""
        pass
    
    @abstractmethod
    def quantize_weight(
        self,
        weight: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Quantize a weight tensor.
        
        Args:
            weight: Weight tensor to quantize [out_features, in_features]
            **kwargs: Additional quantization parameters
            
        Returns:
            (quantized_weight, metadata) tuple
        """
        pass
    
    @abstractmethod
    def linear_forward(
        self,
        x: torch.Tensor,
        weight_container: QuantizedWeight,
        bias: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute linear forward pass.
        
        Args:
            x: Input tensor
            weight_container: Container with quantized weights
            bias: Optional bias tensor
            **kwargs: Additional arguments (quant_scales, etc.)
            
        Returns:
            Output tensor
        """
        pass
    
    def configure(self, *, diffulex_config: Any | None = None) -> None:
        """Optional configuration hook."""
        pass
