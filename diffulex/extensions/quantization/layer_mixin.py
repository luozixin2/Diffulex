"""
Linear Quantization Mixin

Core quantization logic for Linear layers.
Provides forward dispatch, weight management, and plan caching.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Any, Dict

from .linear_plan_builder import build_forward_plan, rebuild_plan_if_needed
from .context import get_linear_strategy


class LinearQuantizationMixin:
    """
    Mixin for quantized linear layers.
    
    Adds quantization capabilities to any linear layer class.
    """
    
    def init_quantization(self, quant_kind: str = "other"):
        """
        Initialize quantization-related buffers.
        
        Call this in __init__ after weight initialization.
        """
        self.quant_kind = quant_kind
        self._forward_out_features = None
        self._forward_plan_enabled = False
        self._forward_plan = None
        self._quant_strategy = None
        
        # Quantization buffers (registered via register_buffer)
        self._weight_is_quantized = False
        self._weight_is_quantized_py = False
        self._offline_quant_format = 0  # 0=none, 1=GPTQ, 2=AWQ, 3=Marlin
        self._offline_quant_format_py = None
        self._offline_quant_bits = 0
        self._offline_quant_group_size = 0
        self._gptq_is_shuffled = False
        self._gptq_is_shuffled_py = False
        self._gptq_marlin_is_prepared = False
        self._gptq_marlin_is_prepared_py = False
    
    def enable_forward_plan(self, enabled: bool = True):
        """Enable/disable forward plan caching."""
        self._forward_plan_enabled = enabled
        if not enabled:
            self._forward_plan = None
    
    def build_forward_plan_for_static(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Build forward plan for static shapes (decode step)."""
        if not self._forward_plan_enabled:
            return
        self._forward_plan = build_forward_plan(self, x, bias)
    
    def has_quantized_weight(self) -> bool:
        """Check if layer has online quantized weights."""
        return getattr(self, '_weight_is_quantized_py', False) or \
               bool(getattr(self, '_weight_is_quantized', False))
    
    def has_offline_quantized_weight(self) -> bool:
        """Check if layer has offline quantized weights (GPTQ/AWQ)."""
        fmt = getattr(self, '_offline_quant_format_py', None) or \
              int(getattr(self, '_offline_quant_format', 0))
        return fmt != 0
    
    def set_quantized_weight(self, qweight: torch.Tensor, scales: torch.Tensor,
                             zero_points: Optional[torch.Tensor] = None):
        """
        Set online quantized weight (INT8/FP8).
        
        Args:
            qweight: Quantized weight tensor
            scales: Scale tensor
            zero_points: Optional zero points for asymmetric quantization
        """
        self.register_buffer('quant_weight_int8', qweight)
        self.register_buffer('quant_scales', scales)
        if zero_points is not None:
            self.register_buffer('quant_zero_points', zero_points)
        
        self._weight_is_quantized_py = True
        if hasattr(self, '_weight_is_quantized'):
            self._weight_is_quantized = torch.tensor(1, dtype=torch.uint8)
    
    def set_offline_quantized_weight(self, qweight: torch.Tensor, 
                                     qzeros: torch.Tensor,
                                     scales: torch.Tensor,
                                     g_idx: Optional[torch.Tensor] = None,
                                     bits: int = 4,
                                     group_size: int = 128,
                                     format_type: str = "gptq"):
        """
        Set offline quantized weight (GPTQ/AWQ).
        
        Args:
            qweight: Packed quantized weights (int32)
            qzeros: Packed zero points (int32)
            scales: Scale tensor
            g_idx: Optional group indices
            bits: Quantization bits (4 or 8)
            group_size: Group size for quantization
            format_type: "gptq" or "awq"
        """
        if format_type == "gptq":
            self.register_buffer('gptq_qweight', qweight)
            self.register_buffer('gptq_qzeros', qzeros)
            self.register_buffer('gptq_scales', scales)
            if g_idx is not None:
                self.register_buffer('gptq_g_idx', g_idx)
            self._offline_quant_format_py = "gptq"
            if hasattr(self, '_offline_quant_format'):
                self._offline_quant_format = torch.tensor(1, dtype=torch.uint8)
        elif format_type == "awq":
            self.register_buffer('awq_qweight', qweight)
            self.register_buffer('awq_qzeros', qzeros)
            self.register_buffer('awq_scales', scales)
            self._offline_quant_format_py = "awq"
            if hasattr(self, '_offline_quant_format'):
                self._offline_quant_format = torch.tensor(2, dtype=torch.uint8)
        
        self._offline_quant_bits = bits
        self._offline_quant_group_size = group_size
    
    def _forward_base(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Unified forward dispatcher.
        
        Routes to appropriate implementation based on quantization state.
        """
        # Check for cached plan
        if self._forward_plan_enabled and self._forward_plan is not None:
            rebuild_plan_if_needed(self, x, bias)
            return self._forward_plan(x, bias)
        
        # Get strategy from context if not set
        if self._quant_strategy is None:
            self._quant_strategy = get_linear_strategy(self.quant_kind)
        
        # Route based on quantization state
        if self.has_offline_quantized_weight():
            return self._forward_offline_quantized(x, bias)
        elif self.has_quantized_weight():
            return self._forward_online_quantized(x, bias)
        else:
            return self._forward_bf16(x, bias)
    
    def _forward_bf16(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard BF16 forward."""
        return F.linear(x, self.weight, bias)
    
    def _forward_online_quantized(self, x: torch.Tensor, 
                                   bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward with online quantized weights."""
        if self._quant_strategy is None:
            # No strategy, use BF16
            return self._forward_bf16(x, bias)
        
        # Get quantized weight
        qweight = getattr(self, 'quant_weight_int8', None)
        if qweight is None:
            return self._forward_bf16(x, bias)
        
        # Use strategy for forward
        return self._quant_strategy.linear_forward(
            x, qweight, bias,
            quant_kind=self.quant_kind,
            weight_scale=getattr(self, 'quant_scales', None),
            weight_zero_point=getattr(self, 'quant_zero_points', None)
        )
    
    def _forward_offline_quantized(self, x: torch.Tensor,
                                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward with offline quantized weights (GPTQ/AWQ)."""
        if self._quant_strategy is None:
            return self._forward_bf16(x, bias)
        
        weight_format = getattr(self, '_offline_quant_format_py', None) or \
                       str(getattr(self, '_offline_quant_format', ''))
        
        # Build kwargs based on format
        kwargs = {'quant_kind': self.quant_kind}
        
        if 'gptq' in weight_format.lower():
            kwargs.update({
                'qweight': getattr(self, 'gptq_qweight', None),
                'qzeros': getattr(self, 'gptq_qzeros', None),
                'scales': getattr(self, 'gptq_scales', None),
                'g_idx': getattr(self, 'gptq_g_idx', None),
                'bits': getattr(self, '_offline_quant_bits', 4),
                'is_shuffled': getattr(self, '_gptq_is_shuffled_py', False) or \
                              bool(getattr(self, '_gptq_is_shuffled', False)),
            })
        elif 'awq' in weight_format.lower():
            kwargs.update({
                'qweight': getattr(self, 'awq_qweight', None),
                'qzeros': getattr(self, 'awq_qzeros', None),
                'scales': getattr(self, 'awq_scales', None),
                'bits': getattr(self, '_offline_quant_bits', 4),
            })
        
        return self._quant_strategy.linear_forward(x, None, bias, **kwargs)
    
    def _maybe_prepare_offline_gptq(self):
        """
        Prepare GPTQ weights for use.
        
        Shuffles weights if needed using vLLM ops.
        """
        if not self.has_offline_quantized_weight():
            return
        
        fmt = getattr(self, '_offline_quant_format_py', None)
        if fmt != "gptq":
            return
        
        if getattr(self, '_gptq_is_shuffled_py', False):
            return
        
        # Try to shuffle using vLLM
        try:
            import vllm._custom_ops as ops
            if hasattr(ops, 'gptq_shuffle'):
                qweight = getattr(self, 'gptq_qweight', None)
                g_idx = getattr(self, 'gptq_g_idx', None)
                if qweight is not None and g_idx is not None:
                    ops.gptq_shuffle(qweight, g_idx, self._offline_quant_bits)
                    self._gptq_is_shuffled_py = True
                    if hasattr(self, '_gptq_is_shuffled'):
                        self._gptq_is_shuffled = torch.tensor(1, dtype=torch.uint8)
        except (ImportError, AttributeError):
            pass
    
    def _maybe_prepare_marlin(self):
        """
        Prepare Marlin format weights.
        
        Repacks GPTQ/AWQ weights to Marlin format if needed.
        """
        if not self.has_offline_quantized_weight():
            return
        
        if getattr(self, '_gptq_marlin_is_prepared_py', False):
            return
        
        fmt = getattr(self, '_offline_quant_format_py', None)
        
        # Try to repack to Marlin
        try:
            import vllm._custom_ops as ops
            
            if fmt == "gptq" and hasattr(ops, 'gptq_marlin_repack'):
                self._repack_gptq_to_marlin(ops)
            elif fmt == "awq" and hasattr(ops, 'awq_marlin_repack'):
                self._repack_awq_to_marlin(ops)
        except (ImportError, AttributeError):
            pass
    
    def _repack_gptq_to_marlin(self, ops):
        """Repack GPTQ weights to Marlin format."""
        qweight = getattr(self, 'gptq_qweight', None)
        qzeros = getattr(self, 'gptq_qzeros', None)
        scales = getattr(self, 'gptq_scales', None)
        g_idx = getattr(self, 'gptq_g_idx', None)
        
        if qweight is None or qzeros is None or scales is None:
            return
        
        marlin_qweight, marlin_scales = ops.gptq_marlin_repack(
            qweight, qzeros, scales, g_idx, self._offline_quant_bits
        )
        
        self.register_buffer('gptq_marlin_qweight', marlin_qweight)
        self.register_buffer('gptq_marlin_scales', marlin_scales)
        
        # Allocate workspace
        workspace = torch.zeros(
            marlin_qweight.shape[1] * 32 // self._offline_quant_bits,
            dtype=torch.int32,
            device=qweight.device
        )
        self.register_buffer('gptq_marlin_workspace', workspace)
        
        self._gptq_marlin_is_prepared_py = True
        if hasattr(self, '_gptq_marlin_is_prepared'):
            self._gptq_marlin_is_prepared = torch.tensor(1, dtype=torch.uint8)
        
        # Update format
        self._offline_quant_format_py = "gptq_marlin"
        if hasattr(self, '_offline_quant_format'):
            self._offline_quant_format = torch.tensor(3, dtype=torch.uint8)
    
    def _repack_awq_to_marlin(self, ops):
        """Repack AWQ weights to Marlin format."""
        qweight = getattr(self, 'awq_qweight', None)
        qzeros = getattr(self, 'awq_qzeros', None)
        scales = getattr(self, 'awq_scales', None)
        
        if qweight is None or qzeros is None or scales is None:
            return
        
        marlin_qweight, marlin_scales = ops.awq_marlin_repack(
            qweight, qzeros, scales, self._offline_quant_bits
        )
        
        self.register_buffer('awq_marlin_qweight', marlin_qweight)
        self.register_buffer('awq_marlin_scales', marlin_scales)
        
        workspace = torch.zeros(
            marlin_qweight.shape[1] * 32 // self._offline_quant_bits,
            dtype=torch.int32,
            device=qweight.device
        )
        self.register_buffer('awq_marlin_workspace', workspace)
        
        self._gptq_marlin_is_prepared_py = True
        if hasattr(self, '_gptq_marlin_is_prepared'):
            self._gptq_marlin_is_prepared = torch.tensor(1, dtype=torch.uint8)
        
        self._offline_quant_format_py = "awq_marlin"
        if hasattr(self, '_offline_quant_format'):
            self._offline_quant_format = torch.tensor(4, dtype=torch.uint8)
