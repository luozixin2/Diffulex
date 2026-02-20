"""Rotary Position Embedding (RoPE) for DiffuLex Edge models."""

from typing import Tuple
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    Standardized implementation used across all DiffuLex Edge models.
    Precomputes cos/sin tables for efficiency.
    
    Args:
        head_dim: Dimension of each attention head (must be even)
        max_position_embeddings: Maximum sequence length to support
        base: Base for the exponential decay of rotation angles
        
    Reference:
        https://arxiv.org/abs/2104.09864
    """
    
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin tables
        self._precompute_rotary_embeddings()
    
    def _precompute_rotary_embeddings(self):
        """Precompute cos and sin for all positions."""
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate for complex number representation
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to q and k tensors.
        
        Args:
            positions: Position indices [batch_size, seq_len]
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_kv_heads, seq_len, head_dim]
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Get cos and sin for the given positions
        # Shape: [batch_size, seq_len, head_dim] -> [batch_size, 1, seq_len, head_dim]
        cos = self.cos_cached[positions].unsqueeze(1)
        sin = self.sin_cached[positions].unsqueeze(1)
        
        q_rot = self._apply_rotary_pos_emb(q, cos, sin)
        k_rot = self._apply_rotary_pos_emb(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.
        
        Rotates the input by splitting into two halves and applying
        the rotation matrix:
        [x1, x2] @ [[cos, -sin], [sin, cos]] = [x1*cos - x2*sin, x1*sin + x2*cos]
        
        Args:
            x: Input tensor [..., head_dim]
            cos: Cosine values [..., head_dim]
            sin: Sine values [..., head_dim]
            
        Returns:
            Rotated tensor of same shape as x
        """
        # Split into two halves
        x1, x2 = x.chunk(2, dim=-1)
        
        # Rotate: [-x2, x1] is the rotated version
        rotated = torch.cat([-x2, x1], dim=-1)
        
        # Apply: x * cos + rotated * sin
        return x * cos + rotated * sin
    
    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"base={self.base}"
        )
