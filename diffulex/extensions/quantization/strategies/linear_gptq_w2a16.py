"""
GPTQ W2A16 Linear Strategy

2-bit weight quantization with BF16 activation.
Uses vLLM's gptq_gemm op for optimized inference.
"""

import torch
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..kernel_availability import warn_kernel_unavailable, check_vllm_op_available


@register_linear_strategy("gptq_w2a16", "bf16")
class GPTQW2A16LinearStrategy(LinearQuantizationStrategy):
    """
    GPTQ W2A16 linear quantization (2-bit).
    
    Uses vLLM's gptq_gemm op with bits=2.
    """
    
    def __init__(self, group_size: int = 128, desc_act: bool = False):
        self.bits = 2
        self.group_size = group_size
        self.desc_act = desc_act
        
        # Check for vLLM GPTQ ops
        self.gptq_gemm = None
        self.shuffle_weights = None
        self._kernel_warned = False
        
        if check_vllm_op_available('gptq_gemm'):
            import vllm._custom_ops as ops
            self.gptq_gemm = ops.gptq_gemm
            if hasattr(ops, 'gptq_shuffle'):
                self.shuffle_weights = ops.gptq_shuffle
        # Note: Warning is deferred to first forward call to avoid spam during import
    
    @property
    def name(self) -> str:
        return f"gptq_w2a16_g{self.group_size}"
    
    @property
    def linear_weight_format(self) -> str:
        return "gptq_int2"
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.int32, 0)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("GPTQ requires offline quantization")
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("GPTQ requires kernel-based dequantization")
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError("GPTQ requires offline weight quantization")
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return x, {}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """
        GPTQ W2A16 linear forward.
        """
        qweight = kwargs.get('qweight')
        qzeros = kwargs.get('qzeros')
        scales = kwargs.get('scales')
        g_idx = kwargs.get('g_idx')
        is_shuffled = kwargs.get('is_shuffled', False)
        
        if qweight is None or qzeros is None or scales is None:
            raise ValueError("GPTQ forward requires qweight, qzeros, and scales buffers")
        
        # Use vLLM GPTQ GEMM if available
        if self.gptq_gemm is not None:
            try:
                x_2d = x.reshape(-1, x.shape[-1])
                
                output = self.gptq_gemm(
                    x_2d,
                    qweight,
                    qzeros,
                    scales,
                    g_idx if g_idx is not None else torch.empty(0, dtype=torch.int32, device=x.device),
                    is_shuffled,
                    self.bits
                )
                
                output_shape = list(x.shape[:-1]) + [scales.shape[1]]
                output = output.reshape(output_shape)
                
                if bias is not None:
                    output = output + bias
                
                return output
            except Exception as e:
                if not self._kernel_warned:
                    warn_kernel_unavailable(
                        "vllm.gptq_gemm",
                        self.name,
                        f"manual dequantization (error: {e})"
                    )
                    self._kernel_warned = True
        else:
            if not self._kernel_warned:
                warn_kernel_unavailable(
                    "vllm.gptq_gemm",
                    self.name,
                    "manual dequantization (CPU-based)"
                )
                self._kernel_warned = True
        
        # Fallback: manual dequantization (very slow for 2-bit)
        raise RuntimeError(
            f"[{self.name}] vllm.gptq_gemm kernel is required for 2-bit GPTQ. "
            f"Manual dequantization is not supported for 2-bit weights. "
            f"Please install vLLM with CUDA support."
        )
    
    def prepare_gptq_weight(self, qweight: torch.Tensor, qzeros: torch.Tensor,
                           scales: torch.Tensor, g_idx: Optional[torch.Tensor],
                           device: torch.device = None) -> Dict[str, Any]:
        """Prepare GPTQ weight buffers from model checkpoint."""
        if device is not None:
            qweight = qweight.to(device)
            qzeros = qzeros.to(device)
            scales = scales.to(device)
            if g_idx is not None:
                g_idx = g_idx.to(device)
        
        return {
            'qweight': qweight,
            'qzeros': qzeros,
            'scales': scales,
            'g_idx': g_idx,
            'bits': self.bits,
            'is_shuffled': False,
        }
