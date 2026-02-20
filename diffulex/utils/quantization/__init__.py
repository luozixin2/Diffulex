"""Quantization utilities for Diffulex."""

from diffulex.utils.quantization.core import (
    WeightFormat,
    LinearQuantizationProtocol,
    QuantizedWeight,
    BF16Weight,
    W8A16Weight,
    W8A8Weight,
    GPTQWeight,
    AWQWeight,
    GPTQMarlinWeight,
    AWQMarlinWeight,
    WeightContainerFactory,
)

from diffulex.utils.quantization.context import (
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

from diffulex.utils.quantization.factory import QuantizationStrategyFactory

from diffulex.utils.quantization.config import (
    KVCacheQuantConfig,
    WeightQuantConfig,
    ActivationQuantConfig,
    QuantizationConfig,
)

from diffulex.utils.quantization.registry import (
    register_kv_cache_strategy,
    create_kv_cache_strategy,
    register_linear_strategy,
    create_linear_strategy,
    registered_kv_cache_dtypes,
    registered_linear_dtypes,
)

from diffulex.utils.quantization.strategy import (
    QuantizationStrategy,
    KVCacheQuantizationStrategy,
    WeightQuantizationStrategy,
    LinearQuantizationStrategy,
)

from diffulex.utils.quantization.delegate import (
    QuantizedLinearDelegate,
    ForwardPlanManager,
    create_quantized_delegate,
)

from diffulex.utils.quantization.loader_adapter import (
    set_offline_gptq_weight,
    set_offline_awq_weight,
    set_offline_gptq_marlin_weight,
    set_offline_awq_marlin_weight,
    prepare_gptq_marlin_from_standard,
    prepare_awq_marlin_from_standard,
)

from diffulex.utils.quantization.kv_cache_dtype import (
    parse_kv_cache_dtype,
    view_fp8_cache,
    _normalize_kv_cache_dtype,
)

__all__ = [
    "WeightFormat",
    "LinearQuantizationProtocol",
    "QuantizedWeight",
    "BF16Weight",
    "W8A16Weight",
    "W8A8Weight",
    "GPTQWeight",
    "AWQWeight",
    "GPTQMarlinWeight",
    "AWQMarlinWeight",
    "WeightContainerFactory",
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
    "QuantizationStrategyFactory",
    "KVCacheQuantConfig",
    "WeightQuantConfig",
    "ActivationQuantConfig",
    "QuantizationConfig",
    "register_kv_cache_strategy",
    "create_kv_cache_strategy",
    "register_linear_strategy",
    "create_linear_strategy",
    "registered_kv_cache_dtypes",
    "registered_linear_dtypes",
    "QuantizationStrategy",
    "KVCacheQuantizationStrategy",
    "WeightQuantizationStrategy",
    "LinearQuantizationStrategy",
    "QuantizedLinearDelegate",
    "ForwardPlanManager",
    "create_quantized_delegate",
    "set_offline_gptq_weight",
    "set_offline_awq_weight",
    "set_offline_gptq_marlin_weight",
    "set_offline_awq_marlin_weight",
    "prepare_gptq_marlin_from_standard",
    "prepare_awq_marlin_from_standard",
    "parse_kv_cache_dtype",
    "view_fp8_cache",
    "_normalize_kv_cache_dtype",
]
