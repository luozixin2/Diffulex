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
    GPTQWeight,
    AWQWeight,
    GPTQMarlinWeight,
    AWQMarlinWeight,
    WeightContainerFactory,
    WeightFormat,
)
from diffulex.utils.quantization.core.protocol import LinearQuantizationProtocol
from diffulex.utils.quantization.context import get_linear_strategy


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


class ForwardPlan:
    """Cached forward execution plan."""
    
    def __init__(
        self,
        sig: ForwardPlanSig,
        executor: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    ):
        self.sig = sig
        self._executor = executor
    
    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        return self._executor(x, bias)


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
        strategy: LinearQuantizationProtocol,
    ) -> Optional[ForwardPlan]:
        """Get cached plan if signature matches."""
        if not self._enabled:
            return None
        
        if self._plan is None:
            return None
        
        sig = self._plan.sig
        device = x.device
        dev_idx = self._device_index(device)
        
        if (
            sig.device_type == device.type
            and sig.device_index == dev_idx
            and sig.x_dtype == x.dtype
            and sig.x_shape == tuple(int(v) for v in x.shape)
            and sig.has_bias == (bias is not None)
        ):
            return self._plan
        
        return None
    
    def build_plan(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        container: QuantizedWeight,
        strategy: LinearQuantizationProtocol,
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
        )
        
        def executor(x_in: torch.Tensor, bias_in: Optional[torch.Tensor]) -> torch.Tensor:
            return container.forward(x_in, bias_in, strategy)
        
        self._plan = ForwardPlan(sig, executor)
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
        
        strategy = get_linear_strategy(self.quant_kind)
        if strategy is None:
            if isinstance(self._container, BF16Weight):
                return F.linear(x, self._container.weight, bias)
            raise RuntimeError("QuantizedLinearDelegate: no strategy configured for quantized weight")
        
        plan = self._plan_manager.get_plan(x, bias, self._container, strategy)
        if plan is not None:
            return plan(x, bias)
        
        if self._plan_manager._enabled:
            plan = self._plan_manager.build_plan(x, bias, self._container, strategy)
            return plan(x, bias)
        
        return self._container.forward(x, bias, strategy)
    
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
        
        weight_format = getattr(strategy, 'weight_format', None)
        if weight_format is None or weight_format == WeightFormat.BF16:
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
        scale_dtype = torch.bfloat16
        strategy = get_linear_strategy(self.quant_kind)
        if strategy is not None:
            weight_format = getattr(strategy, 'linear_weight_format', None)
            act_format = getattr(strategy, 'linear_act_format', None)
            if weight_format == "int8" and act_format == "int8":
                scale_dtype = torch.float32
            elif weight_format in ("fp8_e4m3", "fp8_e5m2"):
                scale_dtype = torch.float32
        
        if quant_scales.dtype != scale_dtype:
            quant_scales = quant_scales.to(scale_dtype)
        
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
            container = WeightContainerFactory.from_awq(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                out_features=out_features,
                in_features=in_features,
                group_size=group_size,
                bits=4,
                device=device,
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
            from vllm import _custom_ops as ops
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                marlin_make_empty_g_idx,
                marlin_make_workspace_new,
                marlin_permute_scales,
                marlin_sort_g_idx,
            )
        except Exception:
            return
        
        device = x.device
        container = self._container
        
        if container.g_idx.numel() > 0:
            g_idx_sorted, g_idx_sort = marlin_sort_g_idx(
                container.g_idx.to(device=device, dtype=torch.int32)
            )
        else:
            g_idx_sorted = marlin_make_empty_g_idx(device)
            g_idx_sort = marlin_make_empty_g_idx(device)
        
        workspace = marlin_make_workspace_new(device)
        
        marlin_qweight = ops.gptq_marlin_repack(
            container.qweight.to(device).contiguous(),
            perm=g_idx_sort,
            size_k=container.in_features,
            size_n=container.out_features,
            num_bits=container.bits,
            is_a_8bit=False,
        )
        
        marlin_scales = marlin_permute_scales(
            container.scales.to(device).contiguous(),
            size_k=container.in_features,
            size_n=container.out_features,
            group_size=container.group_size,
            is_a_8bit=False,
        )
        
        marlin_container = GPTQMarlinWeight(
            qweight=marlin_qweight.contiguous(),
            scales=marlin_scales.contiguous(),
            zp=marlin_make_empty_g_idx(device),
            g_idx=g_idx_sorted.contiguous(),
            g_idx_sort_indices=g_idx_sort.contiguous(),
            workspace=workspace,
            bits=container.bits,
            group_size=container.group_size,
            out_features=container.out_features,
            in_features=container.in_features,
        )
        
        self.set_container(marlin_container)
    
    def _maybe_prepare_offline_awq_marlin(self, x: torch.Tensor) -> None:
        """Prepare AWQ Marlin weights."""
        if not isinstance(self._container, AWQWeight):
            return
        
        try:
            from vllm import _custom_ops as ops
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                awq_to_marlin_zero_points,
                marlin_make_workspace_new,
                marlin_permute_scales,
                marlin_make_empty_g_idx,
            )
        except Exception:
            return
        
        device = x.device
        container = self._container
        num_groups = container.in_features // container.group_size
        
        workspace = marlin_make_workspace_new(device)
        
        marlin_qweight = ops.awq_marlin_repack(
            container.qweight.to(device).contiguous(),
            size_k=container.in_features,
            size_n=container.out_features,
            num_bits=container.bits,
            is_a_8bit=False,
        )
        
        marlin_scales = marlin_permute_scales(
            container.scales.to(device).contiguous(),
            size_k=container.in_features,
            size_n=container.out_features,
            group_size=container.group_size,
            is_a_8bit=False,
        )
        
        marlin_zp = awq_to_marlin_zero_points(
            container.qzeros.to(device).contiguous(),
            size_k=num_groups,
            size_n=container.out_features,
            num_bits=container.bits,
            is_a_8bit=False,
        )
        
        marlin_container = AWQMarlinWeight(
            qweight=marlin_qweight.contiguous(),
            scales=marlin_scales.contiguous(),
            zp=marlin_zp.contiguous(),
            workspace=workspace,
            group_size=container.group_size,
            out_features=container.out_features,
            in_features=container.in_features,
            bits=container.bits,
        )
        
        self.set_container(marlin_container)


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
