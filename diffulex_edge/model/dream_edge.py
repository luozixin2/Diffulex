"""Dream model for DiffuLex Edge.

Edge-optimized version without tensor parallelism.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DreamEdgeConfig:
    """Configuration for Dream Edge model."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # GQA support
    intermediate_size: int = 11008
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_bias: bool = True  # Dream uses bias in attention
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False


class DreamRMSNorm(nn.Module):
    """RMS Norm for Dream."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class DreamRotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) for Dream."""
    
    def __init__(
        self,
        head_dim: int,
        max_position: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin for common positions
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to q and k."""
        cos = self.cos_cached[positions].unsqueeze(1)  # [B, 1, S, D]
        sin = self.sin_cached[positions].unsqueeze(1)  # [B, 1, S, D]
        
        q_embed = self._rotate_half(q, cos, sin)
        k_embed = self._rotate_half(k, cos, sin)
        
        return q_embed, k_embed
    
    def _rotate_half(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1) * sin + x * cos


class DreamAttention(nn.Module):
    """Dream attention mechanism (Edge version without tensor parallelism)."""
    
    def __init__(self, config: DreamEdgeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.rotary_emb = DreamRotaryEmbedding(
            self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embedding
        q, k = self.rotary_emb(positions, q, k)
        
        # Repeat k/v for GQA
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Attention
        attn_output = self._attention(q, k, v, mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
    
    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Scaled dot-product attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)


class DreamMLP(nn.Module):
    """Dream MLP with SiLU activation."""
    
    def __init__(self, config: DreamEdgeConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class DreamDecoderLayer(nn.Module):
    """Dream transformer decoder layer."""
    
    def __init__(self, config: DreamEdgeConfig):
        super().__init__()
        self.self_attn = DreamAttention(config)
        self.mlp = DreamMLP(config)
        self.input_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states + residual)
            residual = hidden_states
        
        hidden_states = self.self_attn(positions, hidden_states, mask)
        hidden_states = self.post_attention_layernorm(hidden_states + residual)
        residual = hidden_states
        
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DreamEdge(nn.Module):
    """Dream model for diffusion language modeling (Edge version)."""
    
    def __init__(self, config: DreamEdgeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DreamDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            positions: [batch_size, seq_len] or None
            mask: Optional attention mask
            kv_cache: Optional KV cache tensor
            start_pos: Starting position for KV cache
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            kv_cache: Updated KV cache or None
        """
        if positions is None:
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, mask)
        
        hidden_states = self.norm(hidden_states + residual)
        logits = self.lm_head(hidden_states)
        
        return logits, kv_cache


__all__ = [
    "DreamEdgeConfig",
    "DreamEdge",
]
