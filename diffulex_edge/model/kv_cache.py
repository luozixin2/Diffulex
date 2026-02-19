"""
Static KV Cache for Edge Inference
===================================

This module implements static KV cache management for edge deployment.
Unlike server-side PagedAttention, this uses a simple tensor-based cache
that is passed as input and output to enable ExecuTorch export.

Key Design:
- KV Cache is passed as input to forward()
- Updated KV Cache is returned as output
- Supports both prefill and decode modes
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class KVCacheConfig:
    """Configuration for KV Cache.
    
    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length (static allocation)
        dtype: Data type for cache tensors
    """
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype = torch.float32


class KVCache:
    """Static KV Cache container.
    
    Shape: [num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim]
    Where dim 1 is: 0 for key, 1 for value
    
    This format allows easy slicing per layer during forward pass.
    """
    
    def __init__(self, config: KVCacheConfig, batch_size: int = 1):
        self.config = config
        self.batch_size = batch_size
        self.current_seq_len = 0
        
        # Pre-allocate full cache
        # Shape: [layers, 2, batch, kv_heads, max_seq, head_dim]
        self.cache = torch.zeros(
            config.num_layers,
            2,  # k and v
            batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
            dtype=config.dtype,
        )
    
    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K and V cache for a specific layer.
        
        Returns:
            Tuple of (k_cache, v_cache) each of shape
            [batch, num_kv_heads, max_seq_len, head_dim]
        """
        k = self.cache[layer_idx, 0]  # [batch, kv_heads, max_seq, head_dim]
        v = self.cache[layer_idx, 1]
        return k, v
    
    def update_layer_cache(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Update cache for a specific layer at given position.
        
        Args:
            layer_idx: Layer index
            new_k: New key values [batch, kv_heads, seq_len, head_dim]
            new_v: New value values [batch, kv_heads, seq_len, head_dim]
            start_pos: Starting position to update
        """
        seq_len = new_k.shape[2]
        self.cache[layer_idx, 0, :, :, start_pos:start_pos + seq_len, :] = new_k
        self.cache[layer_idx, 1, :, :, start_pos:start_pos + seq_len, :] = new_v
        
        # Update sequence length tracker
        self.current_seq_len = max(self.current_seq_len, start_pos + seq_len)
    
    def get_cache_tensor(self) -> torch.Tensor:
        """Get the full cache tensor for passing to model.
        
        Returns:
            Cache tensor of shape
            [num_layers, 2, batch, kv_heads, max_seq, head_dim]
        """
        return self.cache
    
    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        config: KVCacheConfig,
        current_seq_len: int = 0,
    ) -> "KVCache":
        """Create KVCache from existing tensor.
        
        Used when receiving updated cache from model forward pass.
        """
        batch_size = tensor.shape[2]
        cache = cls.__new__(cls)
        cache.config = config
        cache.batch_size = batch_size
        cache.current_seq_len = current_seq_len
        cache.cache = tensor
        return cache
    
    def clear(self) -> None:
        """Clear the cache (reset to zeros)."""
        self.cache.zero_()
        self.current_seq_len = 0


def create_kv_caches(
    config: KVCacheConfig,
    batch_size: int = 1,
) -> KVCache:
    """Create a new KV cache.
    
    Args:
        config: KV cache configuration
        batch_size: Batch size (usually 1 for edge)
        
    Returns:
        Initialized KVCache
    """
    return KVCache(config, batch_size)


# Type alias for KV cache in model signatures
KVCacheTensor = torch.Tensor  # [num_layers, 2, batch, kv_heads, max_seq, head_dim]
