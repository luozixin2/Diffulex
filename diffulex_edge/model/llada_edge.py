"""LLaDA model for DiffuLex Edge.

Edge-optimized version without tensor parallelism.
Uses shared components from components/ module.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

from ..components import RMSNorm, RotaryEmbedding, SwiGLUMLP


@dataclass
class LLaDAEdgeConfig:
    """Configuration for LLaDA Edge model."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16  # GQA support
    intermediate_size: int = 5632
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_bias: bool = True  # LLaDA uses bias in attention
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    mask_token_id: int = 126336  # LLaDA specific
    confidence_threshold: float = 0.9


class LLaDAAttention(nn.Module):
    """LLaDA attention mechanism (Edge version without tensor parallelism)."""
    
    def __init__(self, config: LLaDAEdgeConfig):
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
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
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
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rotary_emb(positions, q, k)
        
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        attn_output = self._attention(q, k, v, mask)
        
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


class LLaDADecoderLayer(nn.Module):
    """LLaDA transformer decoder layer."""
    
    def __init__(self, config: LLaDAEdgeConfig):
        super().__init__()
        self.self_attn = LLaDAAttention(config)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
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


class LLaDAEdge(nn.Module):
    """LLaDA model for diffusion language modeling (Edge version)."""
    
    def __init__(self, config: LLaDAEdgeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLaDADecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        """Forward pass."""
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
    "LLaDAEdgeConfig",
    "LLaDAEdge",
]
