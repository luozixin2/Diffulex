"""SDAR model for DiffuLex Edge.

Fixed-shape for export:
- seq_len = diffusion_block_size (4 for SDAR)
- KV cache: [num_layers, 2, batch, num_kv_heads, max_seq_len, head_dim]
- positions: indicates current positions in sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SDAREdgeConfig:
    """Configuration for SDAR Edge model."""
    vocab_size: int = 151936
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 6144
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    head_dim: Optional[int] = None
    diffusion_block_size: int = 4


class SDARRMSNorm(nn.Module):
    """RMS Norm - matches HF implementation, inline for XNNPACK compatibility."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inline RMSNorm computation using only standard ops
        # x: [..., hidden_size]
        input_dtype = x.dtype
        # Compute variance in float32 for numerical stability
        x_f32 = x.float()
        # mean(x^2, dim=-1, keepdim=True)
        variance = torch.mean(x_f32 * x_f32, dim=-1, keepdim=True)
        # rsqrt(variance + eps)
        inv_std = torch.rsqrt(variance + self.eps)
        # Normalize and scale
        x_norm = x_f32 * inv_std * self.weight.float()
        return x_norm.to(input_dtype)


class SDARRotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) - inline for XNNPACK compatibility."""
    
    def __init__(self, head_dim: int, max_position: int = 4096, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        # inv_freq: [head_dim // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin tables
        t = torch.arange(max_position, dtype=torch.float32)
        # freqs: [max_position, head_dim // 2]
        freqs = torch.outer(t, self.inv_freq)
        # emb: [max_position, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # positions: [batch, seq_len]
        # q, k: [batch, num_heads, seq_len, head_dim]
        
        # Get cos/sin: [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
        cos = self.cos_cached[positions].unsqueeze(1)
        sin = self.sin_cached[positions].unsqueeze(1)
        
        # RoPE using standard ops only
        # Split into two halves
        head_dim = q.shape[-1]
        half = head_dim // 2
        
        q1 = q[..., :half]  # First half: [batch, num_heads, seq_len, half]
        q2 = q[..., half:]  # Second half
        k1 = k[..., :half]
        k2 = k[..., half:]
        
        # Split cos/sin into two halves: [batch, 1, seq_len, head_dim] -> two [batch, 1, seq_len, half]
        cos1 = cos[..., :half]
        cos2 = cos[..., half:]
        sin1 = sin[..., :half]
        sin2 = sin[..., half:]
        
        # Apply rotation: [x1, x2] rotated = [x1*cos1 - x2*sin1, x1*sin2 + x2*cos2]
        # Standard RoPE: x1' = x1*cos - x2*sin, x2' = x1*sin + x2*cos
        q_rot1 = q1 * cos1 - q2 * sin1
        q_rot2 = q1 * sin2 + q2 * cos2
        k_rot1 = k1 * cos1 - k2 * sin1
        k_rot2 = k1 * sin2 + k2 * cos2
        
        # Concatenate back
        q_rot = torch.cat([q_rot1, q_rot2], dim=-1)
        k_rot = torch.cat([k_rot1, k_rot2], dim=-1)
        
        return q_rot, k_rot


class SDARAttention(nn.Module):
    """SDAR attention with per-head Q/K RMSNorm."""
    
    def __init__(self, config: SDAREdgeConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        self.q_norm = SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self.rotary_emb = SDARRotaryEmbedding(self.head_dim, max_position=config.max_position_embeddings, base=config.rope_theta)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_len: int = 0,  # Number of valid positions in cache
        max_cache_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with dynamic KV cache (for Python inference).
        
        This method uses dynamic control flow (if, slice) and is NOT suitable for export.
        
        Args:
            positions: [batch, seq_len]
            hidden_states: [batch, seq_len, hidden]
            k_cache: [batch, num_kv_heads, cache_len, head_dim] - dynamic shape
            v_cache: [batch, num_kv_heads, cache_len, head_dim] - dynamic shape
            cache_len: Number of valid positions in cache (0 if no cache)
            max_cache_len: Maximum cache length (for truncation)
            
        Returns:
            output: [batch, seq_len, hidden]
            new_k: [batch, num_kv_heads, new_cache_len, head_dim]
            new_v: [batch, num_kv_heads, new_cache_len, head_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        if max_cache_len is None:
            max_cache_len = k_cache.shape[2] if k_cache is not None else seq_len
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # Ensure consistent dtype
        target_dtype = q.dtype
        k = k.to(target_dtype)
        v = v.to(target_dtype)
        
        q, k = self.rotary_emb(positions, q, k)
        
        # Concatenate with cache (dynamic, uses cache_len)
        if k_cache is not None and v_cache is not None and cache_len > 0:
            # Slice valid cache positions
            k_valid = k_cache[:, :, :cache_len, :].to(target_dtype)
            v_valid = v_cache[:, :, :cache_len, :].to(target_dtype)
            k = torch.cat([k_valid, k], dim=2)
            v = torch.cat([v_valid, v], dim=2)
        
        # Keep last max_cache_len for fixed output shape
        if k.shape[2] > max_cache_len:
            k = k[:, :, -max_cache_len:, :]
            v = v[:, :, -max_cache_len:, :]
        
        new_k, new_v = k.contiguous(), v.contiguous()
        
        # Attention
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, scale=self.scaling
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output), new_k, new_v
    
    def forward_export(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        old_k_cache: torch.Tensor,  # [batch, num_kv_heads, max_len, head_dim]
        old_v_cache: torch.Tensor,  # [batch, num_kv_heads, max_len, head_dim]
        attention_mask: torch.Tensor,  # [batch, 1, block_size, max_len + block_size]
        insert_matrix: torch.Tensor,   # [batch, 1, max_len, block_size]
        keep_mask: torch.Tensor,       # [batch, 1, max_len, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Static forward for Block Diffusion / ExecuTorch export.
        
        All operations are static (no if, no slice, no item()).
        Cache update is controlled by CPU-provided masks.
        
        Args:
            positions: [batch, block_size]
            hidden_states: [batch, block_size, hidden]
            old_k_cache: [batch, num_kv_heads, max_len, head_dim]
            old_v_cache: [batch, num_kv_heads, max_len, head_dim]
            attention_mask: [batch, 1, block_size, max_len + block_size]
                            CPU-computed mask: 0 for valid, -10000 for invalid positions
            insert_matrix: [batch, 1, max_len, block_size]
                           Maps block_k to cache positions via matrix multiplication
                           CPU sets to 0 for iteration (no update), specific matrix for commit
            keep_mask: [batch, 1, max_len, 1]
                       1 = keep old cache, 0 = overwrite with new block
                       CPU sets to 1 for iteration, 0 at commit positions
            
        Returns:
            block_out: [batch, block_size, hidden]
            updated_k_cache: [batch, num_kv_heads, max_len, head_dim]
            updated_v_cache: [batch, num_kv_heads, max_len, head_dim]
        """
        batch_size, block_size, _ = hidden_states.shape
        
        # 1. Compute Q, K, V for current block
        q = self.q_proj(hidden_states).view(batch_size, block_size, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, block_size, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, block_size, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.transpose(1, 2).contiguous()  # [batch, num_heads, block_size, head_dim]
        k = k.transpose(1, 2).contiguous()  # [batch, num_kv_heads, block_size, head_dim]
        v = v.transpose(1, 2).contiguous()
        
        # Ensure consistent dtype
        target_dtype = q.dtype
        k = k.to(target_dtype)
        v = v.to(target_dtype)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # ---------------------------------------------------------
        # Step 1: Static Attention (concat cache + block, then matmul)
        # ---------------------------------------------------------
        # full_k shape: [batch, num_kv_heads, max_len + block_size, head_dim]
        full_k = torch.cat([old_k_cache.to(target_dtype), k], dim=2)
        full_v = torch.cat([old_v_cache.to(target_dtype), v], dim=2)
        
        # Handle GQA: repeat KV heads to match Q heads
        if self.num_kv_heads != self.num_heads:
            num_repeat = self.num_heads // self.num_kv_heads
            full_k = full_k.repeat_interleave(num_repeat, dim=1)  # [batch, num_heads, ...]
            full_v = full_v.repeat_interleave(num_repeat, dim=1)  # [batch, num_heads, ...]
        
        # scores shape: [batch, num_heads, block_size, max_len + block_size]
        scores = torch.matmul(q, full_k.transpose(-1, -2)) * self.scaling
        
        # Expand attention_mask for all heads: [batch, num_heads, block_size, max_len + block_size]
        # attention_mask input is [batch, 1, block_size, max_len + block_size]
        scores = scores + attention_mask.to(target_dtype)
        
        probs = torch.softmax(scores, dim=-1)
        
        # block_out shape: [batch, num_heads, block_size, head_dim]
        block_out = torch.matmul(probs, full_v)
        
        # Reshape for output projection
        block_out = block_out.transpose(1, 2).contiguous().view(batch_size, block_size, -1)
        block_out = self.o_proj(block_out)
        
        # ---------------------------------------------------------
        # Step 2: Static Cache Update (matrix multiply + mask, no if/slice)
        # ---------------------------------------------------------
        # insert_matrix: [batch, 1, max_len, block_size]
        # k: [batch, num_kv_heads, block_size, head_dim]
        # We need to apply insert_matrix per kv_head
        
        # Expand insert_matrix for all kv_heads: [batch, num_kv_heads, max_len, block_size]
        insert_expanded = insert_matrix.expand(-1, self.num_kv_heads, -1, -1)
        
        # expanded_k shape: [batch, num_kv_heads, max_len, head_dim]
        # insert_expanded @ k: [max_len, block_size] @ [block_size, head_dim] -> [max_len, head_dim]
        expanded_k = torch.matmul(insert_expanded.to(target_dtype), k)
        expanded_v = torch.matmul(insert_expanded.to(target_dtype), v)
        
        # keep_mask: [batch, 1, max_len, 1] -> expand for kv_heads and head_dim
        keep_expanded = keep_mask.expand(-1, self.num_kv_heads, -1, self.head_dim)
        
        # Static cache update: (Old * Mask) + New
        # Iteration: keep_mask=1, insert_matrix=0 -> output = Old
        # Commit: keep_mask=0 at positions, insert_matrix maps -> output = New at those positions
        updated_k_cache = (old_k_cache.to(target_dtype) * keep_expanded.to(target_dtype)) + expanded_k
        updated_v_cache = (old_v_cache.to(target_dtype) * keep_expanded.to(target_dtype)) + expanded_v
        
        return block_out, updated_k_cache, updated_v_cache


class SDARMLP(nn.Module):
    """MLP with SiLU."""
    
    def __init__(self, config: SDAREdgeConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class SDARDecoderLayer(nn.Module):
    """Decoder layer."""
    
    def __init__(self, config: SDAREdgeConfig):
        super().__init__()
        self.self_attn = SDARAttention(config)
        self.mlp = SDARMLP(config)
        self.input_layernorm = SDARRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SDARRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_len: int = 0,
        max_cache_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with dynamic KV cache (for Python inference)."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_k, new_v = self.self_attn(positions, hidden_states, k_cache, v_cache, cache_len, max_cache_len)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, new_k, new_v
    
    def forward_export(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        old_k_cache: torch.Tensor,
        old_v_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        insert_matrix: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Static forward for ExecuTorch export.
        
        Args:
            positions: [batch, block_size]
            hidden_states: [batch, block_size, hidden]
            old_k_cache: [batch, num_kv_heads, max_len, head_dim]
            old_v_cache: [batch, num_kv_heads, max_len, head_dim]
            attention_mask: [batch, 1, block_size, max_len + block_size]
            insert_matrix: [batch, 1, max_len, block_size]
            keep_mask: [batch, 1, max_len, 1]
            
        Returns:
            hidden_states: [batch, block_size, hidden]
            updated_k_cache: [batch, num_kv_heads, max_len, head_dim]
            updated_v_cache: [batch, num_kv_heads, max_len, head_dim]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, updated_k, updated_v = self.self_attn.forward_export(
            positions, hidden_states, old_k_cache, old_v_cache,
            attention_mask, insert_matrix, keep_mask
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, updated_k, updated_v


class SDAREdge(nn.Module):
    """SDAR model supporting both dynamic and fixed-shape KV cache.
    
    For Python inference (DiffusionEngine): dynamic KV cache list
    For export: fixed-shape KV cache tensor
    """
    
    def __init__(self, config: SDAREdgeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SDARDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = SDARRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # Ignored for SDAR (non-causal)
        kv_cache: Optional[list] = None,  # Dynamic KV cache for Python inference
        max_seq_len: Optional[int] = None,  # Max cache length (for consistency with forward_export)
    ) -> Tuple[torch.Tensor, list]:
        """Forward with dynamic KV cache (for DiffusionEngine).
        
        Args:
            input_ids: [batch, seq_len]
            positions: [batch, seq_len]
            mask: Ignored for SDAR
            kv_cache: List of (k, v) tuples per layer, or None
            max_seq_len: Maximum cache length (defaults to config.max_position_embeddings)
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            new_kv_cache: List of (k, v) tuples per layer
        """
        if max_seq_len is None:
            max_seq_len = self.config.max_position_embeddings
        
        hidden_states = self.embed_tokens(input_ids)
        
        new_kv_cache = []
        
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = None, None
            cache_len = 0
            if kv_cache is not None and i < len(kv_cache):
                k_cache, v_cache = kv_cache[i]
                cache_len = k_cache.shape[2] if k_cache is not None else 0
            
            # Use layer's forward with max_cache_len for consistent cache truncation
            hidden_states, new_k, new_v = layer(positions, hidden_states, k_cache, v_cache, cache_len, max_seq_len)
            
            new_kv_cache.append((new_k, new_v))
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, new_kv_cache
    
    def forward_export(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attention_mask: torch.Tensor,  # [num_layers, batch, 1, block_size, max_len + block_size]
        insert_matrix: torch.Tensor,   # [num_layers, batch, 1, max_len, block_size]
        keep_mask: torch.Tensor,       # [num_layers, batch, 1, max_len, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with fixed-shape KV cache for Block Diffusion / ExecuTorch export.
        
        All operations are static (no if, no slice, no item()).
        Cache update is controlled by CPU-provided masks.
        
        Args:
            input_ids: [batch, block_size]
            positions: [batch, block_size]
            kv_cache: [num_layers, 2, batch, num_kv_heads, max_len, head_dim]
            attention_mask: [num_layers, batch, 1, block_size, max_len + block_size]
                            CPU-computed mask per layer: 0 for valid, -10000 for invalid
            insert_matrix: [num_layers, batch, 1, max_len, block_size]
                           CPU-computed matrix per layer for cache update
            keep_mask: [num_layers, batch, 1, max_len, 1]
                       CPU-computed mask per layer: 1=keep old, 0=overwrite
            
        Returns:
            logits: [batch, block_size, vocab_size]
            updated_kv_cache: same shape as input kv_cache
        """
        hidden_states = self.embed_tokens(input_ids)
        num_layers = len(self.layers)
        updated_kv_cache = torch.zeros_like(kv_cache)
        
        for i in range(num_layers):
            layer = self.layers[i]
            k_cache = kv_cache[i, 0]  # [batch, num_kv_heads, max_len, head_dim]
            v_cache = kv_cache[i, 1]
            
            # Use layer's forward_export with static masks
            hidden_states, new_k, new_v = layer.forward_export(
                positions, hidden_states, k_cache, v_cache,
                attention_mask[i], insert_matrix[i], keep_mask[i]
            )
            
            # Static cache update - always write full cache
            updated_kv_cache[i, 0] = new_k
            updated_kv_cache[i, 1] = new_v
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, updated_kv_cache
    
    def get_export_wrapper(self):
        """Get export wrapper for SDAR model.
        
        Returns:
            BlockDiffusionWrapper for ExecuTorch export
        """
        from .wrapper import BlockDiffusionWrapper
        return BlockDiffusionWrapper(
            self,
            block_size=self.config.diffusion_block_size,
            max_seq_len=self.config.max_position_embeddings,
        )
    
    def get_export_inputs(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        device: str = "cpu",
    ):
        """Create example inputs for SDAR export.
        
        SDAR uses Block Diffusion format with special mask inputs.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length (used as max_seq_len for SDAR)
            device: Device to create tensors on
            
        Returns:
            Tuple of input tensors for forward_export
        """
        block_size = self.config.diffusion_block_size
        max_seq_len = seq_len  # Use seq_len as max_seq_len
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim or (self.config.hidden_size // self.config.num_attention_heads)
        
        # Standard inputs
        input_ids = torch.zeros(batch_size, block_size, dtype=torch.long, device=device)
        positions = torch.arange(
            block_size, dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # KV cache
        kv_cache = torch.zeros(
            num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim,
            dtype=torch.float32, device=device
        )
        
        # Block Diffusion masks (for empty cache)
        attention_mask = torch.zeros(
            num_layers, batch_size, 1, block_size, max_seq_len + block_size,
            dtype=torch.float32, device=device
        )
        attention_mask[:, :, :, :, 0:max_seq_len] = -10000.0
        
        insert_matrix = torch.zeros(
            num_layers, batch_size, 1, max_seq_len, block_size,
            dtype=torch.float32, device=device
        )
        for i in range(block_size):
            insert_matrix[:, :, :, i, i] = 1.0
        
        keep_mask = torch.ones(
            num_layers, batch_size, 1, max_seq_len, 1,
            dtype=torch.float32, device=device
        )
        keep_mask[:, :, :, 0:block_size, :] = 0.0
        
        return (input_ids, positions, kv_cache, attention_mask, insert_matrix, keep_mask)


__all__ = ["SDAREdgeConfig", "SDAREdge"]
