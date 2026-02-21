"""Quantization module for DiffuLex Edge.

Provides unified quantization workflows compatible with ExecuTorch:
- FP16: Half precision conversion (2x compression)
- INT8: Weight-only quantization (4x compression)
- INT4: Extreme compression via torchao (8x compression, with fallback)

Quick Start:
    >>> from diffulex_edge.quant import quantize_model
    >>> 
    >>> # FP16 quantization (2x compression)
    >>> model_fp16 = quantize_model(model, dtype="fp16")
    >>>
    >>> # INT8 weight-only (4x compression)
    >>> model_int8 = quantize_model(model, dtype="int8")
    >>>
    >>> # INT4 with torchao (8x compression)
    >>> model_int4 = quantize_model(model, dtype="int4")

Advanced Usage:
    >>> from diffulex_edge.quant import FP16Quantizer, QuantizationConfig
    >>> from diffulex_edge.quant.base import QuantizationDtype
    >>> 
    >>> quantizer = FP16Quantizer()
    >>> config = QuantizationConfig(dtype=QuantizationDtype.FP16)
    >>> result = quantizer.quantize(model, config)
    >>> print(f"Compression: {result.metrics['compression_ratio']:.1f}x")
"""

# Base classes and types
from .base import (
    QuantizationDtype,
    QuantizationMode,
    QuantizationConfig,
    QuantizationResult,
    BaseQuantizer,
    get_model_size_info,
)

# Unified API (recommended)
from .core_quant import (
    # Main functions
    quantize_model,
    quantize_to_fp16,
    quantize_to_int8,
    
    # Utility functions
    verify_quantization_accuracy,
    get_model_compression_ratio,
    
    # Core classes
    QuantizedLinear,
    WeightOnlyQuantizer,
)

# Specific quantizers (for advanced use)
from .int8_quantizer import (
    INT8Quantizer,
    convert_to_int8,
    is_int8_available,
)

from .int4_quantizer import (
    INT4Config,
    INT4Quantizer,
    apply_int4_quantization,
    is_int4_available,
)

from .fp16_quantizer import (
    FP16Quantizer,
    convert_to_fp16,
    is_fp16_available,
)

# torchao utilities
from ._torchao_utils import (
    check_torchao_available,
    get_torchao_version,
    requires_torchao,
)

__all__ = [
    # Base classes and types
    "QuantizationDtype",
    "QuantizationMode",
    "QuantizationConfig",
    "QuantizationResult",
    "BaseQuantizer",
    "get_model_size_info",
    
    # Unified API (recommended)
    "quantize_model",
    "quantize_to_fp16",
    "quantize_to_int8",
    
    # Utility functions
    "verify_quantization_accuracy",
    "get_model_compression_ratio",
    
    # Core classes
    "QuantizedLinear",
    "WeightOnlyQuantizer",
    
    # Specific quantizers
    "INT8Quantizer",
    "INT4Config",
    "INT4Quantizer",
    "FP16Quantizer",
    
    # Convenience functions
    "convert_to_int8",
    "convert_to_fp16",
    "apply_int4_quantization",
    
    # Availability checks
    "is_int8_available",
    "is_int4_available",
    "is_fp16_available",
    
    # torchao utilities
    "check_torchao_available",
    "get_torchao_version",
    "requires_torchao",
]
