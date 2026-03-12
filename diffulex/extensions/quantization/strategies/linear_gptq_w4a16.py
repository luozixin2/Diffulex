"""
GPTQ W4A16 Linear Strategy

4-bit weight quantization with BF16 activation.
Uses GPTQ format with packed weights and group-wise scales.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..kernel_availability import warn_kernel_unavailable, check_vllm_op_available


@register_linear_strategy("gptq_w4a16", "bf16")
class GPTQW4A16LinearStrategy(LinearQuantizationStrategy):
    """
    GPTQ W4A16 linear quantization.
    
    Expects pre-quantized weights in GPTQ format:
    - qweight: int32 packed weights
    - qzeros: int32 packed zeros
    - scales: float16/bfloat16 scales
    - g_idx: int32 group indices (optional)
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
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
        return f"gptq_w4a16_g{self.group_size}"
    
    @property
    def linear_weight_format(self) -> str:
        return "gptq_int4"
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        # 4-bit storage, packed into int32
        return (torch.int32, 0)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPTQ doesn't support runtime quantization."""
        raise NotImplementedError("GPTQ requires offline quantization")
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantization handled by kernel."""
        raise NotImplementedError("GPTQ requires kernel-based dequantization")
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """GPTQ doesn't support runtime weight quantization."""
        raise NotImplementedError("GPTQ requires offline weight quantization")
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """No activation quantization in W4A16."""
        return x, {}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """
        GPTQ linear forward.
        
        Expects GPTQ-specific kwargs:
        - qweight: int32 packed weights [in_features // 8, out_features]
        - qzeros: int32 packed zeros [in_features // group_size, out_features // 8]
        - scales: float scales [in_features // group_size, out_features]
        - g_idx: int32 group indices [in_features] (optional)
        - bits: int (default 4)
        - is_shuffled: bool (default False)
        """
        qweight = kwargs.get('qweight')
        qzeros = kwargs.get('qzeros')
        scales = kwargs.get('scales')
        g_idx = kwargs.get('g_idx')
        bits = kwargs.get('bits', self.bits)
        is_shuffled = kwargs.get('is_shuffled', False)
        
        if qweight is None:
            raise ValueError("GPTQ forward requires 'qweight' buffer")
        if qzeros is None:
            raise ValueError("GPTQ forward requires 'qzeros' buffer")
        if scales is None:
            raise ValueError("GPTQ forward requires 'scales' buffer")
        
        # Use vLLM GPTQ GEMM if available
        if self.gptq_gemm is not None:
            try:
                # Ensure inputs are in correct format
                x_2d = x.reshape(-1, x.shape[-1])
                
                # Call vLLM GPTQ gemm
                output = self.gptq_gemm(
                    x_2d,
                    qweight,
                    qzeros,
                    scales,
                    g_idx if g_idx is not None else torch.empty(0, dtype=torch.int32, device=x.device),
                    is_shuffled,
                    bits
                )
                
                # Reshape output
                output_shape = list(x.shape[:-1]) + [scales.shape[1]]
                output = output.reshape(output_shape)
                
                if bias is not None:
                    output = output + bias
                
                return output
            except Exception as e:
                # Kernel failed, fall through to fallback with warning
                if not self._kernel_warned:
                    warn_kernel_unavailable(
                        "vllm.gptq_gemm", 
                        self.name,
                        f"manual dequantization (error: {e})"
                    )
                    self._kernel_warned = True
        else:
            # Kernel not available
            if not self._kernel_warned:
                warn_kernel_unavailable(
                    "vllm.gptq_gemm", 
                    self.name,
                    "manual dequantization (CPU-based)"
                )
                self._kernel_warned = True
        
        # Fallback: manual dequantization (slow, for compatibility only)
        # This is not recommended for production use
        w_deq = self._manual_dequantize(
            qweight, qzeros, scales, g_idx, bits
        )
        
        return F.linear(x, w_deq.t(), bias)
    
    def _manual_dequantize(self, qweight: torch.Tensor, qzeros: torch.Tensor,
                          scales: torch.Tensor, g_idx: Optional[torch.Tensor],
                          bits: int) -> torch.Tensor:
        """
        Manual dequantization for fallback.
        
        This is slow and should only be used when vLLM ops are unavailable.
        """
        in_features = qweight.shape[0] * 32 // bits
        out_features = qweight.shape[1]
        num_groups = scales.shape[0]
        
        # Unpack weights
        wf = torch.arange(0, 32, bits, device=qweight.device)
        
        # Reshape and unpack
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
            # Use group indices
            scale_idx = g_idx.unsqueeze(1).expand(-1, out_features)
            weight = weight.to(torch.bfloat16) - zeros[scale_idx].to(torch.bfloat16)
            weight = weight * scales[scale_idx].to(torch.bfloat16)
        else:
            # Uniform groups
            group_size = in_features // num_groups
            weight = weight.to(torch.bfloat16)
            for i in range(num_groups):
                start = i * group_size
                end = min((i + 1) * group_size, in_features)
                weight[start:end] = (weight[start:end] - zeros[i].to(torch.bfloat16)) * scales[i].to(torch.bfloat16)
        
        return weight
    
    def prepare_gptq_weight(self, qweight: torch.Tensor, qzeros: torch.Tensor,
                           scales: torch.Tensor, g_idx: Optional[torch.Tensor],
                           bits: int = 4, device: torch.device = None) -> Dict[str, Any]:
        """
        Prepare GPTQ weight buffers from model checkpoint.
        
        Returns dict with prepared buffers ready for linear_forward.
        """
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
            'bits': bits,
            'is_shuffled': False,
        }
