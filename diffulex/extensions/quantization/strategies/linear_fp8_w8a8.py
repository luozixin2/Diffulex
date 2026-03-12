"""
FP8 W8A8 Linear Strategy

FP8 weight + FP8 activation quantization.
Uses vLLM's Fp8LinearOp if available, warns when falling back.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..context import get_cached_act_quant, set_cached_act_quant
from ..kernel_availability import warn_kernel_unavailable


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
    """Quantize tensor to FP8 with per-tensor scaling."""
    fp8_max = get_fp8_max(dtype)
    x_max = x.abs().max().float()
    scale = x_max / fp8_max
    scale = torch.clamp(scale, min=1e-12)
    x_fp8 = (x / scale).to(dtype)
    return x_fp8, scale


def fp8_dequantize(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor."""
    return x_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)


class FP8W8A8BaseStrategy(LinearQuantizationStrategy):
    """Base class for FP8 W8A8 linear quantization."""
    
    def __init__(self, weight_dtype: str = "fp8_e4m3", act_dtype: str = "fp8_e4m3"):
        self.weight_dtype_str = weight_dtype
        self.act_dtype_str = act_dtype
        
        # Map to torch dtypes
        self.weight_dtype = torch.float8_e4m3fn if "e4m3" in weight_dtype else torch.float8_e5m2
        self.act_dtype = torch.float8_e4m3fn if "e4m3" in act_dtype else torch.float8_e5m2
        
        # Try to use vLLM's optimized op
        self.fp8_op = None
        self._kernel_warned = False
        self._vllm_fp8_available = False
        
        try:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import Fp8LinearOp
            self.fp8_op = Fp8LinearOp(
                cutlass_fp8_supported=True,
                use_per_token_if_dynamic=True,
            )
            self._vllm_fp8_available = True
        except (ImportError, Exception):
            pass
    
    @property
    def name(self) -> str:
        return f"fp8_w8a8_{self.weight_dtype_str}_{self.act_dtype_str}"
    
    @property
    def linear_weight_format(self) -> str:
        return self.weight_dtype_str
    
    @property
    def linear_act_format(self) -> str:
        return self.act_dtype_str
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (self.weight_dtype, 1)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to FP8."""
        return fp8_quantize(x, self.act_dtype)
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize from FP8."""
        return fp8_dequantize(q_x, scale)
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize weight tensor for FP8 kernel."""
        w_quant, w_scale = fp8_quantize(weight, self.weight_dtype)
        return w_quant, {'scale': w_scale}
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize activation tensor for FP8 kernel with caching."""
        if cache_key is not None:
            from ..context import get_cached_act_quant, set_cached_act_quant
            cached = get_cached_act_quant(cache_key)
            if cached is not None:
                return cached['tensor'], {'scale': cached['scale']}
        
        x_quant, x_scale = fp8_quantize(x, self.act_dtype)
        
        if cache_key is not None:
            from ..context import set_cached_act_quant
            set_cached_act_quant(cache_key, x_quant, x_scale)
        
        return x_quant, {'scale': x_scale}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """FP8 linear forward."""
        w_quant = kwargs.get('w_quant')
        w_scale = kwargs.get('w_scale')
        
        if w_quant is None or w_scale is None:
            w_quant, w_scale = fp8_quantize(weight, self.weight_dtype)
        
        # Try vLLM's optimized FP8 op first
        if self.fp8_op is not None:
            try:
                output = self.fp8_op.apply(
                    x,
                    w_quant,
                    w_scale,
                    bias,
                    self.act_dtype
                )
                return output
            except Exception as e:
                if not self._kernel_warned:
                    warn_kernel_unavailable(
                        "vllm.Fp8LinearOp",
                        self.name,
                        f"manual FP8 matmul (error: {e})"
                    )
                    self._kernel_warned = True
        else:
            if not self._kernel_warned:
                warn_kernel_unavailable(
                    "vllm.Fp8LinearOp",
                    self.name,
                    "manual FP8 matmul (native PyTorch)"
                )
                self._kernel_warned = True
        
        # Fallback: manual FP8 matmul
        x_quant, x_scale = fp8_quantize(x, self.act_dtype)
        
        output = torch.matmul(
            x_quant.to(torch.bfloat16),
            w_quant.t().to(torch.bfloat16)
        )
        
        output = output * (x_scale * w_scale).to(torch.bfloat16)
        
        if bias is not None:
            output = output + bias
        
        return output


@register_linear_strategy("fp8_e4m3", "fp8_e4m3")
class FP8W8A8LinearStrategy(FP8W8A8BaseStrategy):
    """FP8 E4M3 W8A8 linear quantization."""
    
    def __init__(self, weight_dtype: str = "fp8_e4m3", act_dtype: str = "fp8_e4m3"):
        super().__init__(weight_dtype, act_dtype)


@register_linear_strategy("fp8_e5m2", "fp8_e5m2")
class FP8E5M2W8A8LinearStrategy(FP8W8A8BaseStrategy):
    """FP8 E5M2 W8A8 linear quantization."""
    
    def __init__(self):
        super().__init__("fp8_e5m2", "fp8_e5m2")
