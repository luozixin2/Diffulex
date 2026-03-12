"""
INT8 W8A16 Linear Strategy

INT8 weight + BF16 activation quantization.
Uses vLLM's allspark_w8a16_gemm kernel if available for optimized inference.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..kernel_availability import warn_kernel_unavailable, check_vllm_op_available


def int8_quantize_symmetric(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantization."""
    x_max = x.abs().max().float()
    scale = x_max / 127.0
    scale = torch.clamp(scale, min=1e-12)
    q_x = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return q_x, scale


def int8_dequantize_symmetric(q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize symmetric INT8 tensor to BF16."""
    return q_x.to(torch.bfloat16) * scale.to(torch.bfloat16)


@register_linear_strategy("int8", "bf16")
class INT8W8A16LinearStrategy(LinearQuantizationStrategy):
    """
    INT8 W8A16 linear quantization.
    Weights are quantized to INT8 for storage, dequantized to BF16 for compute.
    Uses allspark_w8a16_gemm kernel if available for optimized GEMM.
    """
    
    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
        
        # Check for allspark kernel (W8A16 optimized)
        self.allspark_gemm = None
        self.allspark_repack = None
        self._kernel_warned = False
        
        if check_vllm_op_available('allspark_w8a16_gemm'):
            import vllm._custom_ops as ops
            self.allspark_gemm = ops.allspark_w8a16_gemm
            if hasattr(ops, 'allspark_repack_weight'):
                self.allspark_repack = ops.allspark_repack_weight
        # Note: Warning is deferred - standard dequantization is acceptable for W8A16
    
    @property
    def name(self) -> str:
        mode = "sym" if self.symmetric else "asym"
        return f"int8_w8a16_{mode}"
    
    @property
    def linear_weight_format(self) -> str:
        return "int8"
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.int8, 1)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to INT8."""
        return int8_quantize_symmetric(x)
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize from INT8 to BF16."""
        return int8_dequantize_symmetric(q_x, scale)
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize weight for storage."""
        q_weight, scale = int8_quantize_symmetric(weight)
        return q_weight, {"scale": scale}
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """No activation quantization in W8A16 - keep BF16."""
        return x, {}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """
        INT8 W8A16 linear forward.
        
        Uses allspark kernel if available, otherwise falls back to 
        dequantize + standard BF16 matmul.
        """
        # Handle weight quantization
        if weight.dtype == torch.int8:
            q_weight = weight
            w_scale = kwargs.get("weight_scale")
        else:
            q_weight, w_meta = self.quantize_weight_for_kernel(weight)
            w_scale = w_meta.get("scale")
        
        # Try allspark kernel if available (only for Ampere+, SM80+)
        if self.allspark_gemm is not None and x.is_cuda:
            try:
                # Get device properties for allspark
                device = x.device
                properties = torch.cuda.get_device_properties(device)
                sm_count = properties.multi_processor_count
                sm_version = properties.major * 10 + properties.minor
                
                # Check if SM version supports allspark (>= 80 for Ampere)
                if sm_version >= 80:
                    # Prepare weights if needed (repack to N32K16 format)
                    # Note: We assume weights are already repacked if using allspark
                    # For dynamic quantization, we'd need to repack here
                    
                    reshaped_x = x.reshape(-1, x.shape[-1])
                    out_features = q_weight.shape[0]  # INT8 weight shape
                    
                    # Call allspark kernel
                    # Note: allspark expects specific weight format
                    output = self.allspark_gemm(
                        a=reshaped_x,
                        b_qweight=q_weight,
                        b_scales=w_scale,
                        b_qzeros=None,
                        n=out_features,
                        group_size=-1,  # Per-tensor quantization
                        sm_count=sm_count,
                        sm_version=sm_version,
                        CUBLAS_M_THRESHOLD=4096,  # Default threshold
                        has_zp=False,
                        n32k16_reorder=False,  # Assume not repacked for dynamic case
                    )
                    
                    output_shape = list(x.shape[:-1]) + [out_features]
                    output = output.reshape(output_shape)
                    
                    if bias is not None:
                        output = output + bias
                    
                    return output
            except Exception as e:
                # Allspark failed, fall through to dequantize path
                if not self._kernel_warned:
                    warn_kernel_unavailable(
                        "vllm.allspark_w8a16_gemm",
                        self.name,
                        f"dequantize + BF16 matmul (error: {e})"
                    )
                    self._kernel_warned = True
        
        # Standard path: dequantize weight to BF16 for computation
        if w_scale is not None:
            weight_bf16 = int8_dequantize_symmetric(q_weight, w_scale)
        else:
            weight_bf16 = q_weight.to(torch.bfloat16)
        
        # Standard BF16 matmul (activation stays in BF16)
        return F.linear(x, weight_bf16, bias)
    
    def prepare_allspark_weight(self, q_weight: torch.Tensor, scale: torch.Tensor,
                                device: torch.device) -> Dict[str, Any]:
        """
        Prepare weight for allspark kernel (repack to N32K16 format).
        
        Should be called during weight loading for optimal performance.
        """
        if self.allspark_repack is None:
            # Return as-is if repack not available
            return {
                'weight': q_weight,
                'scale': scale,
                'repacked': False,
            }
        
        try:
            # Get device properties
            properties = torch.cuda.get_device_properties(device)
            sm_version = properties.major * 10 + properties.minor
            
            # Repack weight to N32K16 format
            # Input: [out_features, in_features] INT8
            # Output: [out_features, in_features] UINT8 in N32K16 layout
            weight_uint8 = q_weight.to(torch.uint8)
            
            repacked_weight, repacked_scale, _ = self.allspark_repack(
                weight_uint8,
                scale,
                None,  # qzeros
                False,  # has_zp
            )
            
            return {
                'weight': repacked_weight,
                'scale': repacked_scale,
                'repacked': True,
                'sm_version': sm_version,
            }
        except Exception as e:
            warn_kernel_unavailable(
                "vllm.allspark_repack_weight",
                self.name,
                f"using non-repacked weights (error: {e})"
            )
            return {
                'weight': q_weight,
                'scale': scale,
                'repacked': False,
            }
