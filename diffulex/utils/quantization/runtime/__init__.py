"""Runtime layer for quantization module.

Layer 3-4: Domain Services and Application
- delegate: Coordinator for weight management and forward execution
- loader_adapter: Adapter for loading offline quantized weights
- strategy_resolver: Strategy resolution for weight containers
- marlin_converter: Marlin format conversion utilities
"""

from diffulex.utils.quantization.runtime.delegate import (
    QuantizedLinearDelegate,
    ForwardPlanManager,
    ForwardPlan,
    create_quantized_delegate,
)

from diffulex.utils.quantization.runtime.loader_adapter import (
    set_offline_gptq_weight,
    set_offline_awq_weight,
    set_offline_gptq_marlin_weight,
    set_offline_awq_marlin_weight,
    prepare_gptq_marlin_from_standard,
    prepare_awq_marlin_from_standard,
    load_offline_quantized_weight,
)

from diffulex.utils.quantization.runtime.strategy_resolver import (
    get_strategy_for_container,
)

from diffulex.utils.quantization.runtime.marlin_converter import (
    convert_gptq_to_marlin,
    convert_awq_to_marlin,
)

__all__ = [
    # Delegate
    "QuantizedLinearDelegate",
    "ForwardPlanManager",
    "ForwardPlan",
    "create_quantized_delegate",
    # Loader Adapter
    "set_offline_gptq_weight",
    "set_offline_awq_weight",
    "set_offline_gptq_marlin_weight",
    "set_offline_awq_marlin_weight",
    "prepare_gptq_marlin_from_standard",
    "prepare_awq_marlin_from_standard",
    "load_offline_quantized_weight",
    # Strategy Resolver
    "get_strategy_for_container",
    # Marlin Converter
    "convert_gptq_to_marlin",
    "convert_awq_to_marlin",
]
