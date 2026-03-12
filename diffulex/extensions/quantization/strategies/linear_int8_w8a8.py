"""
INT8 W8A8 Linear Strategy

INT8 weight + INT8 activation quantization.
Uses vLLM's cutlass_scaled_mm for optimized GEMM.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..kernel_availability import warn_kernel_unavailable, check_vllm_op_available
from ..context import get_cached_act_quant, set_cached_act_quant


def int8_quantize_symmetric(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantization."""
    x_max = x.abs().max().float()
    scale = x_max / 127.0
    scale = torch.clamp(scale, min=1e-12)
    q_x = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return q_x, scale


def int8_dequantize_symmetric(q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize symmetric INT8 tensor."""
    return q_x.to(torch.bfloat16) * scale.to(torch.bfloat16)


@register_linear_strategy("int8", "int8")
class INT8W8A8LinearStrategy(LinearQuantizationStrategy):
    """INT8 W8A8 linear quantization using vLLM's cutlass_scaled_mm."""
    
    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
        
        # Try to use vLLM's cutlass_scaled_mm
        self.cutlass_mm = None
        try:
            import vllm._custom_ops as ops
            if hasattr(ops, 'cutlass_scaled_mm'):
                self.cutlass_mm = ops.cutlass_scaled_mm
        except (ImportError, AttributeError):
            pass
    
    @property
    def name(self) -> str:
        mode = "sym" if self.symmetric else "asym"
        return f"int8_w8a8_{mode}"
    
    @property
    def linear_weight_format(self) -> str:
        return "int8"
    
    @property
    def linear_act_format(self) -> str:
        return "int8"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.int8, 1)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to INT8."""
        return int8_quantize_symmetric(x)
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize from INT8."""
        return int8_dequantize_symmetric(q_x, scale)
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize weight for kernel consumption."""
        q_weight, scale = int8_quantize_symmetric(weight)
        return q_weight, {"scale": scale}
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize activation with caching support."""
        # Check cache
        if cache_key is not None:
            cached = get_cached_act_quant(x)
            if cached is not None:
                return cached[0], {"scale": cached[1]}
        
        # Quantize
        q_x, scale = int8_quantize_symmetric(x)
        meta = {"scale": scale}
        
        # Cache
        if cache_key is not None:
            set_cached_act_quant(x, q_x, scale)
        
        return q_x, meta
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """
        INT8 W8A8 linear forward using vLLM's cutlass_scaled_mm.
        Falls back to dequantize + matmul if cutlass is not available.
        """
        # Handle weight
        if weight.dtype == torch.int8:
            q_weight = weight
            w_scale = kwargs.get("weight_scale")
        else:
            q_weight, w_meta = self.quantize_weight_for_kernel(weight)
            w_scale = w_meta.get("scale")
        
        # Handle activation
        q_x, x_meta = self.quantize_act_for_kernel(x, cache_key=f"{quant_kind}_act")
        x_scale = x_meta.get("scale")
        
        # Try cutlass_scaled_mm if available and dimensions are valid
        if self.cutlass_mm is not None and w_scale is not None and x_scale is not None:
            try:
                # cutlass_scaled_mm expects:
                # a: (M, K) INT8 or BF16/FP16
                # b: (K, N) INT8, contiguous with stride(0)==1
                # scale_a: (1,) or (M,) - per-tensor or per-row
                # scale_b: (1,) or (N,) - per-tensor or per-column
                
                x_2d = q_x.reshape(-1, q_x.shape[-1])  # (M, K)
                out_features = q_weight.shape[0]
                in_features = q_weight.shape[1]
                
                # Weight is (out_features, in_features), need (K, N) = (in_features, out_features)
                weight_t = q_weight.t().contiguous()
                
                # Check alignment requirements
                if weight_t.shape[0] % 16 == 0 and weight_t.shape[1] % 16 == 0:
                    # Use per-tensor scales (scalar)
                    output = self.cutlass_mm(
                        x_2d,
                        weight_t,
                        x_scale,  # activation scale
                        w_scale,  # weight scale
                        torch.bfloat16,
                        bias
                    )
                    
                    output = output.reshape(*q_x.shape[:-1], out_features)
                    return output
            except Exception:
                pass
        
        # Fallback: dequantize and compute in BF16
        x_deq = int8_dequantize_symmetric(q_x, x_scale) if x_scale is not None else q_x.to(torch.bfloat16)
        w_deq = int8_dequantize_symmetric(q_weight, w_scale) if w_scale is not None else q_weight.to(torch.bfloat16)
        
        return F.linear(x_deq, w_deq, bias)
