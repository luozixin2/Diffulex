"""MLP layers for DiffuLex Edge models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP layer.
    
    Standardized implementation used across all DiffuLex Edge models.
    Uses SwiGLU activation: SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Intermediate dimension (typically 4x hidden_size)
        bias: Whether to use bias in projections (default: False)
        
    Reference:
        https://arxiv.org/abs/2002.05202
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Three projections for SwiGLU
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU MLP.
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Output tensor [..., hidden_size]
        """
        # SwiGLU: silu(gate) * up @ down
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


class GatedMLP(nn.Module):
    """Gated MLP with configurable activation.
    
    Alternative to SwiGLU for models that require different gating.
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Intermediate dimension
        activation: Activation function name ("gelu", "silu", "relu")
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        # Select activation
        activations = {
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
        }
        self.activation = activations.get(activation, nn.GELU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through gated MLP."""
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
