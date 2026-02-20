"""Normalization layers for DiffuLex Edge models."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Standardized implementation used across all DiffuLex Edge models.
    Compatible with XNNPACK and ExecuTorch export.
    
    Args:
        hidden_size: Size of the hidden dimension
        eps: Small constant for numerical stability
    
    Reference:
        https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor of shape [..., hidden_size]
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute in float32 for numerical stability
        original_dtype = x.dtype
        x_float32 = x.to(torch.float32)
        variance = x_float32.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x_float32 * torch.rsqrt(variance + self.eps)
        return (x_normalized * self.weight).to(original_dtype)
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"
