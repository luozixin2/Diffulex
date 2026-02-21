"""INT8 quantization support for DiffuLex Edge models.

Provides INT8 weight-only quantization for 4x model compression.
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseQuantizer, QuantizationConfig, QuantizationResult, QuantizationDtype
from .core_quant import QuantizedLinear, WeightOnlyQuantizer


class INT8Quantizer(BaseQuantizer):
    """INT8 weight-only quantizer.
    
    Converts Linear layer weights to INT8 for 4x compression.
    Weights are stored as int8 with per-channel scales.
    
    Example:
        >>> quantizer = INT8Quantizer()
        >>> result = quantizer.quantize(model)
        >>> print(f"Compression: {result.metrics['compression_ratio']:.1f}x")
    """
    
    def is_available(self) -> bool:
        """INT8 quantization is always available."""
        return True
    
    def quantize(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None
    ) -> QuantizationResult:
        """Apply INT8 weight-only quantization.
        
        Args:
            model: Model to quantize
            config: Quantization configuration (optional)
            
        Returns:
            QuantizationResult with INT8 model
        """
        # Use WeightOnlyQuantizer with INT8 dtype
        config = config or QuantizationConfig(dtype=QuantizationDtype.INT8)
        config.dtype = QuantizationDtype.INT8  # Force INT8
        
        quantizer = WeightOnlyQuantizer()
        return quantizer.quantize(model, config)


def convert_to_int8(model: nn.Module) -> nn.Module:
    """Convenience function to convert model to INT8.
    
    Args:
        model: Model to convert
        
    Returns:
        INT8 quantized model
    """
    quantizer = INT8Quantizer()
    result = quantizer.quantize(model)
    return result.model


def is_int8_available() -> bool:
    """Check if INT8 quantization is available.
    
    Returns:
        True (always available)
    """
    return True


__all__ = [
    "INT8Quantizer",
    "convert_to_int8",
    "is_int8_available",
]
