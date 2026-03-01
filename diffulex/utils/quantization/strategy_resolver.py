"""Strategy resolution for quantized weight containers.

This module centralizes the logic for mapping weight containers to their
appropriate quantization strategies, eliminating the need for large
if-elif chains scattered throughout the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diffulex.utils.quantization.core import QuantizedWeight
    from diffulex.utils.quantization.strategy import LinearQuantizationStrategy


def _create_marlin_strategy(bits: int) -> "LinearQuantizationStrategy":
    """Dynamically create and cache a GPTQ Marlin strategy."""
    from diffulex.utils.quantization.context import QuantizationContext
    from diffulex.utils.quantization.strategies.linear_gptq_marlin_w4a16 import (
        LinearGPTQMarlinW4A16Strategy,
    )
    
    marlin_kind = f"gptq_marlin_w{bits}a16"
    ctx = QuantizationContext.current()
    strategy = ctx.get_linear_strategy(marlin_kind)
    if strategy is None:
        strategy = LinearGPTQMarlinW4A16Strategy()
        ctx.set_linear_strategy(marlin_kind, strategy)
    return strategy


def _create_awq_marlin_strategy() -> "LinearQuantizationStrategy":
    """Dynamically create and cache an AWQ Marlin strategy."""
    from diffulex.utils.quantization.context import QuantizationContext
    from diffulex.utils.quantization.strategies.linear_awq_marlin_w4a16 import (
        LinearAWQMarlinW4A16Strategy,
    )
    
    ctx = QuantizationContext.current()
    strategy = ctx.get_linear_strategy("awq_marlin_w4a16")
    if strategy is None:
        strategy = LinearAWQMarlinW4A16Strategy()
        ctx.set_linear_strategy("awq_marlin_w4a16", strategy)
    return strategy


def _create_gptq_strategy(bits: int) -> "LinearQuantizationStrategy":
    """Create a GPTQ strategy based on bit width."""
    if bits == 8:
        from diffulex.utils.quantization.strategies.linear_gptq_w8a16 import (
            LinearGPTQW8A16Strategy,
        )
        return LinearGPTQW8A16Strategy()
    elif bits == 4:
        from diffulex.utils.quantization.strategies.linear_gptq_w4a16 import (
            LinearGPTQW4A16Strategy,
        )
        return LinearGPTQW4A16Strategy()
    # For other bits, fall through to default strategy lookup
    return None  # type: ignore


def _create_awq_strategy() -> "LinearQuantizationStrategy":
    """Create an AWQ strategy."""
    from diffulex.utils.quantization.strategies.linear_awq_w4a16 import (
        LinearAWQW4A16Strategy,
    )
    return LinearAWQW4A16Strategy()


def get_strategy_for_container(
    container: "QuantizedWeight",
    quant_kind: str,
) -> Optional["LinearQuantizationStrategy"]:
    """Get appropriate strategy for a weight container.
    
    This function centralizes the mapping from weight container types to
    their corresponding quantization strategies. It handles:
    - Marlin formats (GPTQ and AWQ)
    - Standard GPTQ (4-bit, 8-bit)
    - Standard AWQ
    - Runtime/online quantized weights (delegates to quant_kind)
    
    Args:
        container: The weight container to get a strategy for
        quant_kind: Quantization kind ("attn", "mlp", "other") for runtime weights
        
    Returns:
        The appropriate LinearQuantizationStrategy, or None if no strategy
        is configured for the container type.
    """
    # Import here to avoid circular dependencies
    from diffulex.utils.quantization.core import (
        GPTQMarlinWeight,
        AWQMarlinWeight,
        GPTQWeight,
        AWQWeight,
    )
    from diffulex.utils.quantization.context import get_linear_strategy
    
    # Marlin formats need special handling
    if isinstance(container, GPTQMarlinWeight):
        return _create_marlin_strategy(container.bits)
    
    if isinstance(container, AWQMarlinWeight):
        return _create_awq_marlin_strategy()
    
    # Standard GPTQ with bit-specific strategies
    if isinstance(container, GPTQWeight):
        strategy = _create_gptq_strategy(container.bits)
        if strategy is not None:
            return strategy
        # Fall through to default lookup for other bit widths
    
    # Standard AWQ
    if isinstance(container, AWQWeight):
        return _create_awq_strategy()
    
    # For runtime quantized weights, use quant_kind to select strategy
    return get_linear_strategy(quant_kind)
