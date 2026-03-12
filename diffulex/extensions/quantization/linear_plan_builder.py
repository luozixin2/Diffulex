"""
Forward Plan Builder

Factory for creating appropriate ForwardPlan based on layer state and strategy.
"""

import torch
from typing import Optional, Any

from .linear_plans import (
    ForwardPlanSig, BF16Plan, QuantInt8W8A16Plan, QuantInt8W8A8Plan,
    QuantFP8W8A8Plan, QuantFP8W8A16Plan,
    OfflineGPTQPlan, OfflineAWQPlan,
    OfflineGPTQMarlinPlan, OfflineAWQMarlinPlan,
    DirectGPTQGemmPlan, DirectAWQGemmPlan, DirectMarlinGemmPlan,
)


def build_forward_plan(layer: torch.nn.Module, example_x: Optional[torch.Tensor] = None,
                       bias: Optional[torch.Tensor] = None) -> Any:
    """
    Build appropriate forward plan for a quantized linear layer.
    
    Args:
        layer: Linear layer with quantization buffers
        example_x: Example input tensor (for signature)
        bias: Optional bias tensor
        
    Returns:
        ForwardPlanBase instance
    """
    # Check for offline quantized weights
    if hasattr(layer, 'has_offline_quantized_weight') and layer.has_offline_quantized_weight():
        return _build_offline_plan(layer, example_x, bias)
    
    # Check for online quantized weights
    if hasattr(layer, 'has_quantized_weight') and layer.has_quantized_weight():
        return _build_online_plan(layer, example_x, bias)
    
    # Default BF16 plan
    return _build_bf16_plan(layer, example_x, bias)


def _build_signature(layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor],
                     mode: str, strategy_name: str) -> ForwardPlanSig:
    """Build plan signature for cache validation."""
    dev = x.device
    return ForwardPlanSig(
        device_type=dev.type,
        device_index=dev.index if dev.index is not None else 0,
        x_dtype=x.dtype,
        x_shape=tuple(x.shape),
        has_bias=bias is not None,
        mode=mode,
        strategy_name=strategy_name
    )


def _build_bf16_plan(layer: torch.nn.Module, example_x: Optional[torch.Tensor],
                     bias: Optional[torch.Tensor]) -> BF16Plan:
    """Build BF16 plan."""
    if example_x is not None:
        sig = _build_signature(layer, example_x, bias, "bf16", "bf16_linear")
    else:
        sig = None
    return BF16Plan(layer, sig)


def _build_online_plan(layer: torch.nn.Module, example_x: torch.Tensor,
                       bias: Optional[torch.Tensor]) -> Any:
    """Build plan for online quantized weights (INT8/FP8)."""
    # Get strategy from layer or context
    strategy = None
    if hasattr(layer, '_quant_strategy'):
        strategy = layer._quant_strategy
    
    if strategy is None:
        # Default to BF16
        return _build_bf16_plan(layer, example_x, bias)
    
    strategy_name = strategy.name
    weight_format = strategy.linear_weight_format
    act_format = strategy.linear_act_format
    
    # Get weight scale
    weight_scale = getattr(layer, 'quant_scales', None)
    zero_point = getattr(layer, 'quant_zero_points', None)
    
    sig = _build_signature(layer, example_x, bias, "quant", strategy_name)
    
    # Build appropriate plan based on formats
    if "int8" in weight_format:
        if "int8" in act_format:
            return QuantInt8W8A8Plan(layer, sig, weight_scale, strategy, zero_point)
        else:
            return QuantInt8W8A16Plan(layer, sig, weight_scale, zero_point)
    elif "fp8" in weight_format:
        if "fp8" in act_format:
            return QuantFP8W8A8Plan(layer, sig, weight_scale, strategy)
        else:
            return QuantFP8W8A16Plan(layer, sig, weight_scale, strategy)
    
    # Fallback to BF16
    return _build_bf16_plan(layer, example_x, bias)


def _build_offline_plan(layer: torch.nn.Module, example_x: torch.Tensor,
                        bias: Optional[torch.Tensor]) -> Any:
    """Build plan for offline quantized weights (GPTQ/AWQ/Marlin)."""
    # Get format
    weight_format = None
    if hasattr(layer, '_offline_quant_format'):
        weight_format = layer._offline_quant_format
    elif hasattr(layer, '_offline_quant_format_py'):
        weight_format = layer._offline_quant_format_py
    
    if weight_format is None:
        return _build_bf16_plan(layer, example_x, bias)
    
    # Get strategy from layer
    strategy = getattr(layer, '_quant_strategy', None)
    strategy_name = strategy.name if strategy else weight_format
    
    sig = _build_signature(layer, example_x, bias, "offline", strategy_name)
    
    # Check for Marlin format
    if "marlin" in weight_format.lower():
        return _build_marlin_plan(layer, sig, strategy)
    
    # Check for GPTQ format
    if "gptq" in weight_format.lower():
        return _build_gptq_plan(layer, sig, strategy)
    
    # Check for AWQ format
    if "awq" in weight_format.lower():
        return _build_awq_plan(layer, sig, strategy)
    
    # Fallback
    return _build_bf16_plan(layer, example_x, bias)


def _build_gptq_plan(layer: torch.nn.Module, sig: ForwardPlanSig,
                     strategy: Any) -> Any:
    """Build GPTQ plan."""
    # Get buffers
    qweight = getattr(layer, 'gptq_qweight', None)
    qzeros = getattr(layer, 'gptq_qzeros', None)
    scales = getattr(layer, 'gptq_scales', None)
    g_idx = getattr(layer, 'gptq_g_idx', None)
    
    if qweight is None or qzeros is None or scales is None:
        return _build_bf16_plan(layer, None, None)
    
    bits = getattr(layer, '_offline_quant_bits', 4)
    is_shuffled = getattr(layer, '_gptq_is_shuffled', False)
    
    # Try direct GEMM if available
    try:
        import vllm._custom_ops as ops
        if hasattr(ops, 'gptq_gemm'):
            return DirectGPTQGemmPlan(
                layer, sig, qweight, qzeros, scales, g_idx,
                bits, is_shuffled, ops.gptq_gemm
            )
    except (ImportError, AttributeError):
        pass
    
    # Use strategy-based plan
    if strategy is not None:
        return OfflineGPTQPlan(
            layer, sig, strategy, qweight, qzeros, scales,
            g_idx, bits, is_shuffled
        )
    
    return _build_bf16_plan(layer, None, None)


def _build_awq_plan(layer: torch.nn.Module, sig: ForwardPlanSig,
                    strategy: Any) -> Any:
    """Build AWQ plan."""
    # Get buffers
    qweight = getattr(layer, 'awq_qweight', None)
    qzeros = getattr(layer, 'awq_qzeros', None)
    scales = getattr(layer, 'awq_scales', None)
    
    if qweight is None or qzeros is None or scales is None:
        return _build_bf16_plan(layer, None, None)
    
    bits = getattr(layer, '_offline_quant_bits', 4)
    
    # Try direct GEMM if available
    try:
        import vllm._custom_ops as ops
        if hasattr(ops, 'awq_gemm'):
            return DirectAWQGemmPlan(
                layer, sig, qweight, qzeros, scales,
                bits, ops.awq_gemm
            )
    except (ImportError, AttributeError):
        pass
    
    # Use strategy-based plan
    if strategy is not None:
        return OfflineAWQPlan(
            layer, sig, strategy, qweight, qzeros, scales, bits
        )
    
    return _build_bf16_plan(layer, None, None)


def _build_marlin_plan(layer: torch.nn.Module, sig: ForwardPlanSig,
                       strategy: Any) -> Any:
    """Build Marlin plan."""
    # Get Marlin buffers
    marlin_qweight = getattr(layer, 'gptq_marlin_qweight', None) or getattr(layer, 'marlin_qweight', None)
    marlin_scales = getattr(layer, 'gptq_marlin_scales', None) or getattr(layer, 'marlin_scales', None)
    marlin_workspace = getattr(layer, 'gptq_marlin_workspace', None) or getattr(layer, 'marlin_workspace', None)
    
    if marlin_qweight is None or marlin_scales is None or marlin_workspace is None:
        return _build_bf16_plan(layer, None, None)
    
    bits = getattr(layer, '_offline_quant_bits', 4)
    
    # Try direct GEMM
    try:
        import vllm._custom_ops as ops
        if hasattr(ops, 'gptq_marlin_gemm'):
            return DirectMarlinGemmPlan(
                layer, sig, marlin_qweight, marlin_scales,
                marlin_workspace, bits, True, ops.gptq_marlin_gemm
            )
    except (ImportError, AttributeError):
        pass
    
    # Use strategy-based plan
    if strategy is not None:
        if "gptq" in strategy.name:
            return OfflineGPTQMarlinPlan(
                layer, sig, marlin_qweight, marlin_scales,
                marlin_workspace, bits, True, strategy
            )
        else:
            return OfflineAWQMarlinPlan(
                layer, sig, marlin_qweight, marlin_scales,
                marlin_workspace, bits, True, strategy
            )
    
    return _build_bf16_plan(layer, None, None)


def rebuild_plan_if_needed(layer: torch.nn.Module, x: torch.Tensor,
                           bias: Optional[torch.Tensor] = None) -> bool:
    """
    Check if plan needs rebuild and rebuild if necessary.
    
    Returns:
        True if plan was rebuilt, False otherwise
    """
    if not hasattr(layer, '_forward_plan') or layer._forward_plan is None:
        # No plan exists, build one
        layer._forward_plan = build_forward_plan(layer, x, bias)
        return True
    
    plan = layer._forward_plan
    sig = plan.get_signature()
    
    if sig is None:
        # Plan doesn't support signature validation, rebuild
        layer._forward_plan = build_forward_plan(layer, x, bias)
        return True
    
    # Check signature match
    dev = x.device
    if (sig.device_type == dev.type and 
        sig.device_index == (dev.index if dev.index is not None else 0) and
        sig.x_dtype == x.dtype and 
        sig.x_shape == tuple(x.shape) and 
        sig.has_bias == (bias is not None)):
        return False
    
    # Signature mismatch, rebuild
    layer._forward_plan = build_forward_plan(layer, x, bias)
    return True
