"""
Forward Plans for Quantized Linear Layers

CUDA-graph friendly quantized linear dispatch with plan caching.
Each plan represents a specific execution strategy for a given configuration.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ForwardPlanSig:
    """
    Validation signature for forward plan caching.
    
    Used to validate if cached plan is still valid for current inputs.
    """
    device_type: str
    device_index: int
    x_dtype: torch.dtype
    x_shape: tuple
    has_bias: bool
    mode: str  # "bf16", "quant", "offline"
    strategy_name: str


class ForwardPlanBase:
    """Base class for forward execution plans."""
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute the plan."""
        raise NotImplementedError
    
    def get_signature(self) -> ForwardPlanSig:
        """Get the signature for cache validation."""
        raise NotImplementedError


class BF16Plan(ForwardPlanBase):
    """Standard BF16 linear forward plan."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig):
        self.layer = layer
        self.signature = signature
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute BF16 linear forward."""
        return F.linear(x, self.layer.weight, bias)
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class QuantInt8W8A16Plan(ForwardPlanBase):
    """INT8 weight + BF16 activation plan."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 weight_scale: torch.Tensor, zero_point: Optional[torch.Tensor] = None):
        self.layer = layer
        self.signature = signature
        self.weight_scale = weight_scale
        self.zero_point = zero_point
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute INT8 W8A16 forward."""
        q_weight = self.layer.quant_weight_int8
        
        # Dequantize weight
        if self.zero_point is not None:
            weight = (q_weight.to(torch.bfloat16) - self.zero_point.to(torch.bfloat16)) * self.weight_scale.to(torch.bfloat16)
        else:
            weight = q_weight.to(torch.bfloat16) * self.weight_scale.to(torch.bfloat16)
        
        return F.linear(x, weight, bias)
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class QuantInt8W8A8Plan(ForwardPlanBase):
    """INT8 weight + INT8 activation plan."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 weight_scale: torch.Tensor, strategy: Any,
                 zero_point: Optional[torch.Tensor] = None):
        self.layer = layer
        self.signature = signature
        self.weight_scale = weight_scale
        self.zero_point = zero_point
        self.strategy = strategy
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute INT8 W8A8 forward."""
        # Use strategy's forward
        return self.strategy.linear_forward(
            x, self.layer.quant_weight_int8, bias,
            quant_kind=getattr(self.layer, 'quant_kind', 'other'),
            weight_scale=self.weight_scale,
            weight_zero_point=self.zero_point
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class QuantFP8W8A8Plan(ForwardPlanBase):
    """FP8 weight + FP8 activation plan."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 weight_scale: torch.Tensor, strategy: Any):
        self.layer = layer
        self.signature = signature
        self.weight_scale = weight_scale
        self.strategy = strategy
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute FP8 W8A8 forward."""
        return self.strategy.linear_forward(
            x, self.layer.quant_weight_int8, bias,
            quant_kind=getattr(self.layer, 'quant_kind', 'other'),
            weight_scale=self.weight_scale
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class QuantFP8W8A16Plan(ForwardPlanBase):
    """FP8 weight + BF16 activation plan."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 weight_scale: torch.Tensor, strategy: Any):
        self.layer = layer
        self.signature = signature
        self.weight_scale = weight_scale
        self.strategy = strategy
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute FP8 W8A16 forward."""
        return self.strategy.linear_forward(
            x, self.layer.quant_weight_int8, bias,
            quant_kind=getattr(self.layer, 'quant_kind', 'other'),
            weight_scale=self.weight_scale
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class OfflineGPTQPlan(ForwardPlanBase):
    """GPTQ offline quantized plan."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 strategy: Any, qweight: torch.Tensor, qzeros: torch.Tensor,
                 scales: torch.Tensor, g_idx: Optional[torch.Tensor],
                 bits: int, is_shuffled: bool):
        self.layer = layer
        self.signature = signature
        self.strategy = strategy
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx
        self.bits = bits
        self.is_shuffled = is_shuffled
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute GPTQ forward."""
        return self.strategy.linear_forward(
            x, None, bias,
            quant_kind=getattr(self.layer, 'quant_kind', 'other'),
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            g_idx=self.g_idx,
            bits=self.bits,
            is_shuffled=self.is_shuffled
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class OfflineAWQPlan(ForwardPlanBase):
    """AWQ offline quantized plan."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 strategy: Any, qweight: torch.Tensor, qzeros: torch.Tensor,
                 scales: torch.Tensor, bits: int):
        self.layer = layer
        self.signature = signature
        self.strategy = strategy
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.bits = bits
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute AWQ forward."""
        return self.strategy.linear_forward(
            x, None, bias,
            quant_kind=getattr(self.layer, 'quant_kind', 'other'),
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            bits=self.bits
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class DirectGPTQGemmPlan(ForwardPlanBase):
    """Direct GPTQ GEMM via vLLM ops (bypassing Python strategy)."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 qweight: torch.Tensor, qzeros: torch.Tensor,
                 scales: torch.Tensor, g_idx: torch.Tensor,
                 bits: int, is_shuffled: bool, gemm_op: Callable):
        self.layer = layer
        self.signature = signature
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx
        self.bits = bits
        self.is_shuffled = is_shuffled
        self.gemm_op = gemm_op
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute direct GPTQ GEMM."""
        x_2d = x.reshape(-1, x.shape[-1])
        
        output = self.gemm_op(
            x_2d,
            self.qweight,
            self.qzeros,
            self.scales,
            self.g_idx if self.g_idx is not None else torch.empty(0, dtype=torch.int32, device=x.device),
            self.is_shuffled,
            self.bits
        )
        
        output_shape = list(x.shape[:-1]) + [self.scales.shape[1]]
        output = output.reshape(output_shape)
        
        if bias is not None:
            output = output + bias
        
        return output
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class DirectAWQGemmPlan(ForwardPlanBase):
    """Direct AWQ GEMM via vLLM ops."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 qweight: torch.Tensor, qzeros: torch.Tensor,
                 scales: torch.Tensor, bits: int, gemm_op: Callable):
        self.layer = layer
        self.signature = signature
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.bits = bits
        self.gemm_op = gemm_op
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute direct AWQ GEMM."""
        x_2d = x.reshape(-1, x.shape[-1])
        
        output = self.gemm_op(
            x_2d,
            self.qweight,
            self.scales,
            self.qzeros,
            self.bits
        )
        
        output_shape = list(x.shape[:-1]) + [self.scales.shape[1]]
        output = output.reshape(output_shape)
        
        if bias is not None:
            output = output + bias
        
        return output
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class DirectMarlinGemmPlan(ForwardPlanBase):
    """Direct Marlin GEMM via vLLM ops."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 marlin_qweight: torch.Tensor, marlin_scales: torch.Tensor,
                 marlin_workspace: torch.Tensor, num_bits: int, is_k_full: bool,
                 gemm_op: Callable):
        self.layer = layer
        self.signature = signature
        self.marlin_qweight = marlin_qweight
        self.marlin_scales = marlin_scales
        self.marlin_workspace = marlin_workspace
        self.num_bits = num_bits
        self.is_k_full = is_k_full
        self.gemm_op = gemm_op
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute direct Marlin GEMM."""
        x_2d = x.reshape(-1, x.shape[-1])
        
        output = self.gemm_op(
            x_2d,
            self.marlin_qweight,
            self.marlin_scales,
            self.marlin_workspace,
            self.num_bits,
            self.is_k_full
        )
        
        output_shape = list(x.shape[:-1]) + [self.marlin_scales.shape[1]]
        output = output.reshape(output_shape)
        
        if bias is not None:
            output = output + bias
        
        return output
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class OfflineGPTQMarlinPlan(ForwardPlanBase):
    """GPTQ + Marlin repacked format plan."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 marlin_qweight: torch.Tensor, marlin_scales: torch.Tensor,
                 marlin_workspace: torch.Tensor, num_bits: int, is_k_full: bool,
                 strategy: Any):
        self.layer = layer
        self.signature = signature
        self.marlin_qweight = marlin_qweight
        self.marlin_scales = marlin_scales
        self.marlin_workspace = marlin_workspace
        self.num_bits = num_bits
        self.is_k_full = is_k_full
        self.strategy = strategy
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute GPTQ Marlin forward."""
        return self.strategy.linear_forward(
            x, None, bias,
            quant_kind=getattr(self.layer, 'quant_kind', 'other'),
            marlin_qweight=self.marlin_qweight,
            marlin_scales=self.marlin_scales,
            marlin_workspace=self.marlin_workspace,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class OfflineAWQMarlinPlan(ForwardPlanBase):
    """AWQ + Marlin repacked format plan."""
    
    def __init__(self, layer: torch.nn.Module, signature: ForwardPlanSig,
                 marlin_qweight: torch.Tensor, marlin_scales: torch.Tensor,
                 marlin_workspace: torch.Tensor, num_bits: int, is_k_full: bool,
                 strategy: Any):
        self.layer = layer
        self.signature = signature
        self.marlin_qweight = marlin_qweight
        self.marlin_scales = marlin_scales
        self.marlin_workspace = marlin_workspace
        self.num_bits = num_bits
        self.is_k_full = is_k_full
        self.strategy = strategy
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute AWQ Marlin forward."""
        return self.strategy.linear_forward(
            x, None, bias,
            quant_kind=getattr(self.layer, 'quant_kind', 'other'),
            marlin_qweight=self.marlin_qweight,
            marlin_scales=self.marlin_scales,
            marlin_workspace=self.marlin_workspace,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature
