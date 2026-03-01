"""Strategy resolution for quantized weight containers.

This module centralizes the logic for mapping weight containers to their
appropriate quantization strategies, eliminating the need for large
if-elif chains scattered throughout the codebase.

All strategies are resolved through the registry system, eliminating
lazy imports and ensuring compile-time dependency verification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from diffulex.utils.quantization.registry import (
    get_strategy_by_key,
    create_linear_strategy,
)
from diffulex.utils.quantization.core import (
    GPTQMarlinWeight,
    AWQMarlinWeight,
    GPTQWeight,
    AWQWeight,
    BF16Weight,
)
from diffulex.utils.quantization.context import get_linear_strategy

if TYPE_CHECKING:
    from diffulex.utils.quantization.strategy import LinearQuantizationStrategy


def get_strategy_for_container(
    container,
    quant_kind: str,
) -> Optional["LinearQuantizationStrategy"]:
    """Get appropriate strategy for a weight container.
    
    This function centralizes the mapping from weight container types to
    their corresponding quantization strategies. It handles:
    - Marlin formats (GPTQ and AWQ)
    - Standard GPTQ (4-bit, 8-bit)
    - Standard AWQ
    - Runtime/online quantized weights (delegates to quant_kind)
    
    All strategy lookups go through the registry, eliminating the need
    for lazy imports of concrete strategy classes.
    
    Args:
        container: The weight container to get a strategy for
        quant_kind: Quantization kind ("attn", "mlp", "other") for runtime weights
        
    Returns:
        The appropriate LinearQuantizationStrategy, or None if no strategy
        is configured for the container type.
    """
    # Marlin formats: use key-based lookup through registry
    if isinstance(container, GPTQMarlinWeight):
        return get_strategy_by_key(f"gptq_marlin_w{container.bits}a16")
    
    if isinstance(container, AWQMarlinWeight):
        return get_strategy_by_key("awq_marlin_w4a16")
    
    # Standard GPTQ: use dtype-pair lookup through registry
    if isinstance(container, GPTQWeight):
        # Map bits to strategy
        if container.bits == 4:
            return create_linear_strategy(weight_dtype="gptq", act_dtype="bf16")
        elif container.bits == 8:
            # GPTQ W8A16 - use key-based lookup if available
            strategy = get_strategy_by_key("gptq_w8a16")
            if strategy is not None:
                return strategy
            # Fall back to dtype-pair lookup
            return create_linear_strategy(weight_dtype="gptq", act_dtype="bf16")
        # For other bit widths, fall through to default
    
    # Standard AWQ: use key-based lookup
    if isinstance(container, AWQWeight):
        return get_strategy_by_key("awq_w4a16")
    
    # For runtime quantized weights (BF16, W8A16, W8A8), use quant_kind
    # to look up strategy from context
    if isinstance(container, BF16Weight):
        return get_linear_strategy(quant_kind)
    
    # Fallback: try to create strategy from weight format
    weight_format = getattr(container, 'weight_format', None)
    if weight_format and hasattr(weight_format, 'value'):
        try:
            return create_linear_strategy(
                weight_dtype=weight_format.value,
                act_dtype="bf16"
            )
        except ValueError:
            pass
    
    return None
