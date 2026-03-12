"""
FP8 W8A16 Linear Strategy

FP8 weight + BF16 activation quantization.
Only weights are quantized, activations remain in BF16.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy


# FP8 constants
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0


def get_fp8_max(dtype: torch.dtype) -> float:
    """Get max representable value for FP8 dtype."""
    if dtype == torch.float8_e4m3fn:
        return FP8_E4M3_MAX
    elif dtype == torch.float8_e5m2:
        return FP8_E5M2_MAX
    else:
        return FP8_E4M3_MAX


def fp8_quantize(x: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8."""
    fp8_max = get_fp8_max(dtype)
    x_max = x.abs().max().float()
    scale = x_max / fp8_max
    scale = torch.clamp(scale, min=1e-12)
    x_fp8 = (x / scale).to(dtype)
    return x_fp8, scale


def fp8_dequantize(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor."""
    return x_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)


@register_linear_strategy("fp8_e4m3", "bf16")
class FP8W8A16LinearStrategy(LinearQuantizationStrategy):
    """FP8 W8A16 linear - only weights quantized."""
    
    def __init__(self, weight_dtype: str = "fp8_e4m3"):
        self.weight_dtype_str = weight_dtype
        self.weight_dtype = torch.float8_e4m3fn if "e4m3" in weight_dtype else torch.float8_e5m2
    
    @property
    def name(self) -> str:
        return f"fp8_w8a16_{self.weight_dtype_str}"
    
    @property
    def linear_weight_format(self) -> str:
        return self.weight_dtype_str
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (self.weight_dtype, 1)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to FP8."""
        return fp8_quantize(x, self.weight_dtype)
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize from FP8."""
        return fp8_dequantize(q_x, scale)
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize weight for kernel consumption."""
        q_weight, scale = fp8_quantize(weight, self.weight_dtype)
        return q_weight, {"scale": scale}
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """No activation quantization in W8A16."""
        return x, {}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """FP8 W8A16 linear forward."""
        # Check if weight is already quantized
        if weight.dtype == self.weight_dtype:
            q_weight = weight
            w_scale = kwargs.get("weight_scale")
        else:
            # Quantize weight on-the-fly
            q_weight, w_meta = self.quantize_weight_for_kernel(weight)
            w_scale = w_meta.get("scale")
        
        # Dequantize weight to BF16 for matmul
        if w_scale is not None:
            weight_bf16 = fp8_dequantize(q_weight, w_scale)
        else:
            weight_bf16 = q_weight.to(torch.bfloat16)
        
        # Standard BF16 matmul (activation stays in BF16)
        return F.linear(x, weight_bf16, bias)


@register_linear_strategy("fp8_e5m2", "bf16")
class FP8E5M2W8A16LinearStrategy(FP8W8A16LinearStrategy):
    """FP8 E5M2 W8A16 linear."""
    
    def __init__(self):
        super().__init__("fp8_e5m2")
