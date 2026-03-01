"""QuantizedLinearDelegate implementation."""

from __future__ import annotations

from typing import Optional, Callable, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffulex.utils.quantization.core import (
    QuantizedWeight,
    BF16Weight,
    W8A16Weight,
    W8A8Weight,
    WeightContainerFactory,
    WeightFormat,
)
from diffulex.utils.quantization.core.protocol import LinearQuantizationProtocol
from diffulex.utils.quantization.context import get_linear_strategy
from diffulex.utils.quantization.strategy_resolver import get_strategy_for_container
from diffulex.utils.quantization.marlin_converter import (
    convert_gptq_to_marlin,
    convert_awq_to_marlin,
)


@dataclass(frozen=True)
class ForwardPlanSig:
    """Signature for validating cached forward plans."""
    device_type: str
    device_index: int
    x_dtype: torch.dtype
    x_shape: tuple[int, ...]
    has_bias: bool
    weight_format: str
    strategy_name: str
    quant_kind: str


class ForwardPlan:
    """Cached forward execution plan.
    
    Mirrors the old code's pattern: store direct references to strategy, weight, etc.
    to avoid indirection overhead in the hot path.
    """
    
    def __init__(
        self,
        sig: ForwardPlanSig,
        strategy: Any,
        weight: torch.Tensor,
        scales: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        quant_kind: str,
        out_features: int,
    ):
        self.sig = sig
        self._strategy = strategy
        self._weight = weight  # qweight for quantized, regular weight for bf16
        self._scales = scales  # quant scales, None for bf16
        self._bias = bias
        self._quant_kind = quant_kind
        self._out_features = out_features
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Direct call to strategy, matching old code pattern
        b = bias if bias is not None else self._bias
        if self._scales is not None:
            # Quantized path: W8A8, W8A16, etc.
            return self._strategy.linear_forward(
                x,
                self._weight,
                b,
                quant_kind=self._quant_kind,
                quant_scales=self._scales,
                out_features=self._out_features,
            )
        else:
            # BF16 path
            return self._strategy.linear_forward(
                x,
                self._weight,
                b,
                quant_kind=self._quant_kind,
            )


class ForwardPlanManager:
    """Manages forward plan caching for quantized layers."""
    
    def __init__(self):
        self._enabled: bool = False
        self._plan: Optional[ForwardPlan] = None
    
    def enable(self, enabled: bool = True) -> None:
        """Enable/disable plan caching."""
        self._enabled = bool(enabled)
        if not enabled:
            self.invalidate()
    
    def invalidate(self) -> None:
        """Invalidate cached plan."""
        self._plan = None
    
    def get_plan(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        container: QuantizedWeight,
        quant_kind: str = "other",
    ) -> Optional[ForwardPlan]:
        """Get cached plan if signature matches.
        
        Note: strategy is not passed here because plan already captures it in executor.
        This avoids expensive strategy lookup on the hot path.
        """
        if not self._enabled:
            return None
        
        if self._plan is None:
            return None
        
        sig = self._plan.sig
        device = x.device
        dev_idx = self._device_index(device)
        
        # Check signature components (strategy_name is already validated in sig)
        if (
            sig.device_type == device.type
            and sig.device_index == dev_idx
            and sig.x_dtype == x.dtype
            and sig.x_shape == tuple(int(v) for v in x.shape)
            and sig.has_bias == (bias is not None)
            and sig.weight_format == container.weight_format.value
            and sig.quant_kind == quant_kind
        ):
            return self._plan
        
        return None
    
    def build_plan(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        container: QuantizedWeight,
        strategy: LinearQuantizationProtocol,
        quant_kind: str = "other",
    ) -> ForwardPlan:
        """Build a new forward plan."""
        device = x.device
        dev_idx = self._device_index(device)
        
        sig = ForwardPlanSig(
            device_type=device.type,
            device_index=dev_idx,
            x_dtype=x.dtype,
            x_shape=tuple(int(v) for v in x.shape),
            has_bias=bias is not None,
            weight_format=container.weight_format.value,
            strategy_name=getattr(strategy, 'name', 'unknown'),
            quant_kind=quant_kind,
        )
        
        # Extract weight and scales from container for direct access in Plan
        if hasattr(container, 'qweight'):
            # Quantized weights (W8A8, W8A16, GPTQ, etc.)
            weight = container.qweight
            scales = getattr(container, 'scales', None)
        else:
            # BF16 path
            weight = getattr(container, 'weight', None)
            scales = None
        
        self._plan = ForwardPlan(
            sig=sig,
            strategy=strategy,
            weight=weight,
            scales=scales,
            bias=bias,
            quant_kind=quant_kind,
            out_features=container.out_features,
        )
        return self._plan
    
    @staticmethod
    def _device_index(device: torch.device) -> int:
        if device.type == "cuda" and device.index is not None:
            return int(device.index)
        return -1


class QuantizedLinearDelegate:
    """Linear layer implementation with quantization support.
    
    Implements the LinearDelegate protocol. Handles weight storage,
    computation via strategies, forward plan caching, and weight loading.
    """
    
    def __init__(
        self,
        quant_kind: str = "other",
        enable_forward_plan: bool = False,
    ):
        self.quant_kind = quant_kind
        
        self._container: Optional[QuantizedWeight] = None
        self._plan_manager = ForwardPlanManager()
        self._plan_manager.enable(enable_forward_plan)
        
        self._loaded_shards: set[object] = set()
    
    @property
    def weight_format(self) -> str:
        """Return weight format identifier."""
        if self._container is None:
            return "bf16"
        return self._container.weight_format.value
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Execute forward pass with quantization support."""
        if self._container is None:
            raise RuntimeError("QuantizedLinearDelegate: no weight container set")
        
        # Try cached forward plan first (fast path for CUDA Graph)
        # This avoids expensive get_strategy_for_container() call on hot path
        if self._plan_manager._enabled:
            plan = self._plan_manager.get_plan(x, bias, self._container, quant_kind=self.quant_kind)
            if plan is not None:
                return plan(x, bias)
        
        # Slow path: get strategy and optionally build plan
        strategy = get_strategy_for_container(self._container, self.quant_kind)
        if strategy is None:
            if isinstance(self._container, BF16Weight):
                return F.linear(x, self._container.weight, bias)
            raise RuntimeError("QuantizedLinearDelegate: no strategy configured for quantized weight")
        
        if self._plan_manager._enabled:
            plan = self._plan_manager.build_plan(x, bias, self._container, strategy, quant_kind=self.quant_kind)
            return plan(x, bias)
        
        return self._container.forward(x, bias, strategy, quant_kind=self.quant_kind)
    
    def load_weight(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        tp_size: int,
        tp_dim: Optional[int],
    ) -> None:
        """Load weight with optional quantization."""
        param_data = param.data
        
        if tp_size > 1 and tp_dim is not None:
            shard_size = param_data.size(tp_dim)
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(tp_dim, start_idx, shard_size)
        
        param_data.copy_(loaded_weight)
        
        self._maybe_quantize_weight(param)
    
    def load_weight_shard(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: object,
        expected_shards: set[object],
        tp_rank: int,
        tp_size: int,
        tp_dim: Optional[int],
    ) -> bool:
        """Load weight shard. Returns True if all shards loaded."""
        param_data = param.data
        
        self._loaded_shards.add(shard_id)
        
        if tp_size > 1 and tp_dim is not None:
            shard_size = param_data.size(tp_dim)
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(tp_dim, start_idx, shard_size)
        
        param_data.copy_(loaded_weight)
        
        if self._loaded_shards == expected_shards:
            self._maybe_quantize_weight(param)
            return True
        
        return False
    
    def set_container(self, container: QuantizedWeight) -> None:
        """Set the weight container."""
        self._container = container
        self._plan_manager.invalidate()
    
    def get_container(self) -> Optional[QuantizedWeight]:
        """Get the current weight container."""
        return self._container
    
    def _maybe_quantize_weight(self, param: nn.Parameter) -> None:
        """Quantize weight at load time if configured."""
        strategy = get_linear_strategy(self.quant_kind)
        if strategy is None:
            return
        
        if getattr(strategy, "name", "").startswith("linear_stub"):
            return
        
        weight_format = getattr(strategy, 'linear_weight_format', None) or getattr(strategy, 'weight_format', None)
        if weight_format is None or weight_format == WeightFormat.BF16 or weight_format == 'bf16':
            # For BF16, create a BF16Weight container so forward() works properly
            from diffulex.utils.quantization.core.container import BF16Weight
            self.set_container(BF16Weight(param.data))
            return
        
        try:
            container = WeightContainerFactory.quantize_at_runtime(
                param.data,
                strategy,
                device=param.data.device,
            )
            self.set_container(container)
        except Exception:
            pass
    
    def has_quantized_weight(self) -> bool:
        """Check if online quantized weight is present."""
        if self._container is None:
            return False
        return isinstance(self._container, (W8A16Weight, W8A8Weight))
    
    def has_offline_quantized_weight(self) -> bool:
        """Check if offline quantized weight is present."""
        if self._container is None:
            return False
        return isinstance(self._container, (GPTQWeight, AWQWeight, GPTQMarlinWeight, AWQMarlinWeight))
    
    def set_quantized_weight(
        self,
        quant_weight: torch.Tensor,
        quant_scales: torch.Tensor,
    ) -> None:
        """Set online quantized weight (W8A16/W8A8)."""
        from .core.container import W8A8Weight
        
        scale_dtype = torch.bfloat16
        is_w8a8 = False
        strategy = get_linear_strategy(self.quant_kind)
        if strategy is not None:
            weight_format = getattr(strategy, 'linear_weight_format', None)
            act_format = getattr(strategy, 'linear_act_format', None)
            if weight_format == "int8" and act_format == "int8":
                scale_dtype = torch.float32
                is_w8a8 = True
            elif weight_format in ("fp8_e4m3", "fp8_e5m2"):
                scale_dtype = torch.float32
        
        if quant_scales.dtype != scale_dtype:
            quant_scales = quant_scales.to(scale_dtype)
        
        # Create the appropriate container based on quantization type
        if is_w8a8:
            container = W8A8Weight(
                qweight=quant_weight.contiguous(),
                scales=quant_scales.contiguous(),
            )
        else:
            container = W8A16Weight(
                qweight=quant_weight.contiguous(),
                scales=quant_scales.contiguous(),
            )
        self.set_container(container)
    
    def set_offline_quantized_weight(
        self,
        format: str,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        *,
        out_features: int,
        in_features: int,
        group_size: int = 128,
        g_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Set offline quantized weight (GPTQ or AWQ)."""
        format = format.strip().lower()
        device = qweight.device
        
        if format == "gptq":
            pack_factor = in_features // int(qweight.shape[0])
            bits = 32 // pack_factor
            
            container = WeightContainerFactory.from_gptq(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                out_features=out_features,
                in_features=in_features,
                group_size=group_size,
                bits=bits,
                device=device,
            )
        elif format == "awq":
            # AWQ requires GPU tensors for vLLM kernels
            target_device = torch.device("cuda") if device.type == "cpu" else device
            container = WeightContainerFactory.from_awq(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                out_features=out_features,
                in_features=in_features,
                group_size=group_size,
                bits=4,
                device=target_device,
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.set_container(container)
    
    def enable_forward_plan(self, enabled: bool = True) -> None:
        """Enable/disable forward plan caching."""
        self._plan_manager.enable(enabled)
    
    def _maybe_prepare_offline_gptq(self, x: torch.Tensor) -> None:
        """Prepare GPTQ weights for first use."""
        if isinstance(self._container, GPTQWeight):
            self._container.prepare()
    
    def _maybe_prepare_offline_gptq_marlin(self, x: torch.Tensor) -> None:
        """Prepare GPTQ Marlin weights."""
        if not isinstance(self._container, GPTQWeight):
            return
        try:
            marlin_container = convert_gptq_to_marlin(self._container, x.device)
            self.set_container(marlin_container)
        except Exception:
            pass
    
    def _maybe_prepare_offline_awq_marlin(self, x: torch.Tensor) -> None:
        """Prepare AWQ Marlin weights."""
        if not isinstance(self._container, AWQWeight):
            return
        try:
            marlin_container = convert_awq_to_marlin(self._container, x.device)
            self.set_container(marlin_container)
        except Exception:
            pass


def create_quantized_delegate(
    quant_kind: str = "other",
    enable_forward_plan: bool = False,
) -> QuantizedLinearDelegate:
    """Factory function to create a quantized delegate.
    
    Args:
        quant_kind: Quantization kind ("attn", "mlp", "other")
        enable_forward_plan: Whether to enable forward plan caching
        
    Returns:
        QuantizedLinearDelegate instance
    """
    return QuantizedLinearDelegate(
        quant_kind=quant_kind,
        enable_forward_plan=enable_forward_plan,
    )
