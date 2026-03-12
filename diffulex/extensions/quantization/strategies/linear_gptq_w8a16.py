"""
GPTQ W8A16 Linear Strategy

8-bit weight quantization with BF16 activation.
Uses vLLM's gptq_gemm op for optimized inference.
"""

import torch
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..kernel_availability import warn_kernel_unavailable, check_vllm_op_available


@register_linear_strategy("gptq_w8a16", "bf16")
class GPTQW8A16LinearStrategy(LinearQuantizationStrategy):
    """
    GPTQ W8A16 linear quantization (8-bit).
    
    Uses vLLM's gptq_gemm op with bits=8.
    """
    
    def __init__(self, group_size: int = 128, desc_act: bool = False):
        self.bits = 8
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
        return f"gptq_w8a16_g{self.group_size}"
    
    @property
    def linear_weight_format(self) -> str:
        return "gptq_int8"
    
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
        GPTQ W8A16 linear forward.
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
        
        # Fallback: manual dequantization for 8-bit
        w_deq = self._manual_dequantize(qweight, qzeros, scales, g_idx, self.bits)
        return torch.nn.functional.linear(x, w_deq.t(), bias)
    
    def _manual_dequantize(self, qweight: torch.Tensor, qzeros: torch.Tensor,
                          scales: torch.Tensor, g_idx: Optional[torch.Tensor],
                          bits: int) -> torch.Tensor:
        """Manual dequantization for fallback."""
        in_features = qweight.shape[0] * 32 // bits
        out_features = qweight.shape[1]
        num_groups = scales.shape[0]
        
        # Unpack weights
        wf = torch.arange(0, 32, bits, device=qweight.device)
        
        weight = torch.bitwise_right_shift(
            qweight.unsqueeze(-1).expand(-1, -1, 32 // bits),
            wf
        ).bitwise_and(2 ** bits - 1)
        weight = weight.reshape(in_features, out_features)
        
        # Unpack zeros
        zeros = torch.bitwise_right_shift(
            qzeros.unsqueeze(-1).expand(-1, -1, 32 // bits),
            wf
        ).bitwise_and(2 ** bits - 1)
        zeros = zeros.reshape(num_groups, out_features)
        zeros = zeros + 1
        
        # Dequantize
        if g_idx is not None and g_idx.numel() > 0:
            scale_idx = g_idx.unsqueeze(1).expand(-1, out_features)
            weight = weight.to(torch.bfloat16) - zeros[scale_idx].to(torch.bfloat16)
            weight = weight * scales[scale_idx].to(torch.bfloat16)
        else:
            group_size = in_features // num_groups
            weight = weight.to(torch.bfloat16)
            for i in range(num_groups):
                start = i * group_size
                end = min((i + 1) * group_size, in_features)
                weight[start:end] = (weight[start:end] - zeros[i].to(torch.bfloat16)) * scales[i].to(torch.bfloat16)
        
        return weight
    
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
