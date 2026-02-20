"""Linear layer implementations with delegate pattern.

LinearBase uses a delegate for implementation details. The delegate is provided
by external modules (e.g., quantization) and handles forward pass, weight loading,
and format-specific logic.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator: int, denominator: int) -> int:
    """Integer division with assertion."""
    assert numerator % denominator == 0, f"{numerator} % {denominator} != 0"
    return numerator // denominator


@runtime_checkable
class LinearDelegate(Protocol):
    """Protocol for linear layer implementation delegate."""
    
    @property
    def weight_format(self) -> str:
        """Return weight format identifier (e.g., 'bf16', 'int8', 'gptq')."""
        ...
    
    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Execute forward pass.
        
        Args:
            x: Input tensor
            bias: Optional bias tensor
            
        Returns:
            Output tensor
        """
        ...
    
    def load_weight(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        tp_size: int,
        tp_dim: Optional[int],
    ) -> None:
        """Load weight from checkpoint.
        
        Args:
            param: The parameter being loaded
            loaded_weight: The weight tensor from checkpoint
            tp_rank: Tensor parallel rank
            tp_size: Tensor parallel size
            tp_dim: Tensor parallel dimension (0 for column, 1 for row)
        """
        ...
    
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
        """Load a weight shard (for multi-shard parameters like QKV).
        
        Args:
            param: The parameter being loaded
            loaded_weight: The weight tensor from checkpoint
            shard_id: Identifier for this shard
            expected_shards: Set of all expected shard IDs
            tp_rank: Tensor parallel rank
            tp_size: Tensor parallel size
            tp_dim: Tensor parallel dimension
            
        Returns:
            True if all shards are loaded and weight is ready
        """
        ...


class LoRAMixin:
    """Mixin class to add LoRA support to linear layers."""
    
    def __init_lora__(
        self,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        if r > 0:
            self.r = r
            self.lora_alpha = lora_alpha
            self.scaling = lora_alpha / r
            
            out_features = getattr(
                self, 'output_size_per_partition',
                getattr(self, 'output_size', 0)
            )
            in_features = getattr(
                self, 'input_size_per_partition',
                getattr(self, 'input_size', 0)
            )
            
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
            self.merged = False
            
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.r = 0
            self.merged = True
    
    def merge_lora(self) -> None:
        """Merge LoRA weights into base weight."""
        if not (hasattr(self, 'r') and self.r > 0 and not self.merged):
            return
        
        weight = getattr(self, "weight", None)
        if weight is None or not hasattr(weight, "data"):
            return
        
        self.weight.data += self.scaling * torch.mm(self.lora_B, self.lora_A)
        self.merged = True
    
    def lora_forward(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply LoRA forward pass."""
        if not hasattr(self, 'r') or self.r == 0 or self.merged:
            return base_output
        
        lora_out = F.linear(self.lora_dropout(x), self.lora_A.T)
        lora_out = F.linear(lora_out, self.lora_B.T)
        return base_output + lora_out * self.scaling


class LinearBase(nn.Module):
    """Minimal base class for linear layers.
    
    LinearBase holds tensor parallelism info and delegates implementation
    details to a LinearDelegate provided by external modules.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: Optional[int] = None,
        quant_kind: str = "other",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.quant_kind = quant_kind
        
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self._forward_out_features: int = int(output_size)
        self._delegate: Optional[LinearDelegate] = None
    
    def set_delegate(self, delegate: LinearDelegate) -> None:
        """Set the implementation delegate.
        
        Args:
            delegate: Implementation delegate
        """
        self._delegate = delegate
    
    def get_delegate(self) -> Optional[LinearDelegate]:
        """Get the current delegate."""
        return self._delegate
    
    def has_delegate(self) -> bool:
        """Check if delegate is set."""
        return self._delegate is not None
    
    def _forward_impl(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        """Internal forward implementation.
        
        Delegates to the LinearDelegate if set, otherwise uses F.linear
        with self.weight.
        """
        if self._delegate is not None:
            return self._delegate.forward(x, bias)
        
        weight = getattr(self, "weight", None)
        if weight is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: no delegate set and no weight available. "
                "Either call set_delegate() or ensure weight parameter exists."
            )
        return F.linear(x, weight, bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError


class ReplicatedLinear(LinearBase, LoRAMixin):
    """Replicated linear layer (no tensor parallelism)."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        LinearBase.__init__(self, input_size, output_size, None, quant_kind)
        
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
        self.__init_lora__(r, lora_alpha, lora_dropout)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load weight from checkpoint."""
        if self._delegate is not None:
            self._delegate.load_weight(
                param, loaded_weight,
                self.tp_rank, self.tp_size, self.tp_dim
            )
        else:
            param.data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        base_out = self._forward_impl(x, self.bias)
        return self.lora_forward(x, base_out)


class ColumnParallelLinear(LinearBase, LoRAMixin):
    """Column-parallel linear layer (output is partitioned)."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        LinearBase.__init__(self, input_size, output_size, 0, quant_kind)
        
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self._forward_out_features = int(self.output_size_per_partition)
        
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            input_size,
        ))
        self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
        self.__init_lora__(r, lora_alpha, lora_dropout)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load weight shard for this partition."""
        if self._delegate is not None:
            self._delegate.load_weight(
                param, loaded_weight,
                self.tp_rank, self.tp_size, self.tp_dim
            )
        else:
            param_data = param.data
            shard_size = param_data.size(self.tp_dim)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
            param_data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        base_out = self._forward_impl(x, self.bias)
        return self.lora_forward(x, base_out)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel linear with multiple output shards (e.g., QKV)."""
    
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            sum(output_sizes),
            bias=bias,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quant_kind=quant_kind,
        )
        self._shard_tracking: set[object] = set()
    
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        """Load weight for a specific shard."""
        if self._delegate is not None:
            ready = self._delegate.load_weight_shard(
                param, loaded_weight, loaded_shard_id,
                set(range(len(self.output_sizes))),
                self.tp_rank, self.tp_size, self.tp_dim
            )
            if ready:
                return
        
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV parallel linear layer (for attention)."""
    
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "attn",
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * head_size
        
        super().__init__(input_size, output_size, bias, r, lora_alpha, lora_dropout, quant_kind)
    
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ) -> None:
        """Load Q, K, or V weight shard."""
        assert loaded_shard_id in ["q", "k", "v"]
        
        if self._delegate is not None:
            ready = self._delegate.load_weight_shard(
                param, loaded_weight, loaded_shard_id,
                {"q", "k", "v"},
                self.tp_rank, self.tp_size, self.tp_dim
            )
            if ready:
                return
        
        param_data = param.data
        
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # v
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase, LoRAMixin):
    """Row-parallel linear layer (input is partitioned)."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        LinearBase.__init__(self, input_size, output_size, 1, quant_kind)
        
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        
        self.weight = nn.Parameter(torch.empty(
            output_size,
            self.input_size_per_partition,
        ))
        self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
        self.__init_lora__(r, lora_alpha, lora_dropout)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load weight shard for this partition."""
        if self._delegate is not None:
            self._delegate.load_weight(
                param, loaded_weight,
                self.tp_rank, self.tp_size, self.tp_dim
            )
        else:
            param_data = param.data
            shard_size = param_data.size(self.tp_dim)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
            param_data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with all-reduce."""
        bias = self.bias if self.tp_rank == 0 else None
        y = self._forward_impl(x, bias)
        
        if self.tp_size > 1:
            dist.all_reduce(y)
        
        return self.lora_forward(x, y)
