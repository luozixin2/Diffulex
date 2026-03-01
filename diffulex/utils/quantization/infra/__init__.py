"""Infrastructure layer for quantization module.

Layer 2: Infrastructure
- context: Context management
- registry: Strategy registry
- factory: Object factory
"""

from diffulex.utils.quantization.infra.context import (
    QuantizationContext,
    get_quantization_context,
    set_kv_cache_strategy,
    get_kv_cache_strategy,
    set_weight_strategy,
    get_weight_strategy,
    set_linear_strategy,
    get_linear_strategy,
    clear_act_quant_cache,
    get_cached_act_quant,
    set_cached_act_quant,
)

from diffulex.utils.quantization.infra.registry import (
    register_kv_cache_strategy,
    create_kv_cache_strategy,
    register_linear_strategy,
    create_linear_strategy,
    register_strategy_key,
    get_strategy_by_key,
    registered_kv_cache_dtypes,
    registered_linear_dtypes,
)

from diffulex.utils.quantization.infra.factory import QuantizationStrategyFactory

__all__ = [
    # Context
    "QuantizationContext",
    "get_quantization_context",
    "set_kv_cache_strategy",
    "get_kv_cache_strategy",
    "set_weight_strategy",
    "get_weight_strategy",
    "set_linear_strategy",
    "get_linear_strategy",
    "clear_act_quant_cache",
    "get_cached_act_quant",
    "set_cached_act_quant",
    # Registry
    "register_kv_cache_strategy",
    "create_kv_cache_strategy",
    "register_linear_strategy",
    "create_linear_strategy",
    "register_strategy_key",
    "get_strategy_by_key",
    "registered_kv_cache_dtypes",
    "registered_linear_dtypes",
    # Factory
    "QuantizationStrategyFactory",
]
