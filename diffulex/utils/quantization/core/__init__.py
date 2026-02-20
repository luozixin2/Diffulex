"""
Quantization core abstractions.

This module provides the foundational abstractions for the quantization system:
- QuantizedWeight: Abstract container for quantized weights
- WeightContainerFactory: Factory for creating weight containers  
- Protocol definitions for strategies
"""

from diffulex.utils.quantization.core.protocol import (
    WeightFormat,
    LinearQuantizationProtocol,
)
from diffulex.utils.quantization.core.container import (
    QuantizedWeight,
    BF16Weight,
    W8A16Weight,
    W8A8Weight,
    GPTQWeight,
    AWQWeight,
    GPTQMarlinWeight,
    AWQMarlinWeight,
)
from diffulex.utils.quantization.core.factory import WeightContainerFactory

__all__ = [
    # Protocols
    "WeightFormat",
    "LinearQuantizationProtocol",
    # Containers
    "QuantizedWeight",
    "BF16Weight",
    "W8A16Weight", 
    "W8A8Weight",
    "GPTQWeight",
    "AWQWeight",
    "GPTQMarlinWeight",
    "AWQMarlinWeight",
    # Factory
    "WeightContainerFactory",
]
