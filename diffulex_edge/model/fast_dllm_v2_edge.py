"""
FastdLLM V2 Edge - Simplified model for edge deployment
========================================================

This is a simplified version of FastdLLM V2 with:
- No Tensor Parallel (single device)
- PyTorch native SDPA instead of custom flash attention kernels
- Standard nn.Linear instead of parallel linear layers
- Static KV Cache for incremental inference
- Support for ExecuTorch export

Architecture:
    Embedding -> N x DecoderLayer -> RMSNorm -> LM Head
    
    DecoderLayer:
        input_layernorm -> Attention -> post_attention_layernorm -> MLP
        
KV Cache Mode:
    Prefill (first pass):
        input_ids [B, S] -> logits [B, S, V]
        
    Decode (incremental):
        input_ids [B, 1], k_cache, v_cache -> logits [B, 1, V], new_k_cache, new_v_cache
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components import RMSNorm, RotaryEmbedding, SwiGLUMLP


@dataclass
class FastdLLMV2EdgeConfig:
    """Configuration for FastdLLM V2 Edge model.
    
    Simplified config without distributed/parallel settings.
    """
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    intermediate_size: int = 14336
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Tuple] = None
    head_dim: Optional[int] = None
    tie_word_embeddings: bool = False
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class AttentionEdge(nn.Module):
    """Multi-head attention using PyTorch native SDPA with KV Cache.
    
    Replaces custom flash attention kernels with F.scaled_dot_product_attention.
    Supports Grouped Query Attention (GQA) and static KV Cache for incremental
    inference.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Projection layers (standard nn.Linear, no parallel)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.scaling = head_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with optional KV Cache.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            positions: [batch_size, seq_len] position indices
            rotary_emb: RotaryEmbedding module
            attention_mask: Optional attention mask
            k_cache: Optional key cache [batch, kv_heads, max_seq, head_dim]
            v_cache: Optional value cache [batch, kv_heads, max_seq, head_dim]
            start_pos: Starting position for cache update (for incremental)
            
        Returns:
            Tuple of (output, updated_k_cache, updated_v_cache)
            If k_cache/v_cache not provided, returns (output, None, None)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention: [B, S, H*D] -> [B, H, S, D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        q, k = rotary_emb(positions, q, k)
        
        # Handle KV Cache
        # Save original k, v for cache return (before any concatenation)
        k_new, v_new = k, v
        
        if k_cache is not None and v_cache is not None:
            # Incremental mode: concat cached keys/values with new ones
            # k_cache: [batch, kv_heads, max_seq, head_dim]
            # Get cached portion up to start_pos
            k_cached = k_cache[:, :, :start_pos, :]
            v_cached = v_cache[:, :, :start_pos, :]
            
            # Concatenate: cached + new
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)
        
        # GQA: repeat k, v heads if needed (for attention computation only)
        # Save original k, v for cache before expansion
        k_orig, v_orig = k, v
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Use PyTorch native SDPA
        # Handle causal mask manually when using KV cache (q_len != k_len)
        if attention_mask is None and k_cache is not None and v_cache is not None:
            # Incremental mode: need custom causal mask
            # q: [B, H, q_len, D], k: [B, H, k_len, D] where k_len > q_len
            q_len = q.shape[2]
            k_len = k.shape[2]
            
            if q_len != k_len:
                # Create causal mask for this case
                # Query at positions [start_pos, start_pos + q_len)
                # Key at positions [0, k_len)
                # Mask should allow attending to positions <= query position
                # For simplicity, allow attending to all cached positions
                # (since we're doing autoregressive generation)
                attention_mask = torch.zeros(
                    q_len, k_len, dtype=q.dtype, device=q.device
                )
                # Use standard SDPA without is_causal
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scaling,
                )
            else:
                # q_len == k_len (prefill), can use is_causal
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=self.scaling,
                )
        else:
            # Standard mode (no cache or explicit mask)
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=attention_mask is None,
                scale=self.scaling,
            )
        
        # Reshape back: [B, H, S, D] -> [B, S, H*D]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        # Prepare updated cache if input was provided
        # Use k_new, v_new (before GQA expansion and cache concatenation)
        # These have shape [batch, num_kv_heads, seq_len, head_dim]
        if k_cache is not None and v_cache is not None:
            return output, k_new, v_new
        
        return output, None, None


class DecoderLayer(nn.Module):
    """Single transformer decoder layer with KV Cache support."""
    
    def __init__(self, config: FastdLLMV2EdgeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.self_attn = AttentionEdge(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qkv_bias=True,
        )
        
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward with residual connections and KV Cache.
        
        Args:
            hidden_states: [B, S, H]
            positions: [B, S]
            rotary_emb: Rotary embedding module
            attention_mask: Optional mask
            k_cache: Optional key cache [B, kv_heads, max_seq, head_dim]
            v_cache: Optional value cache [B, kv_heads, max_seq, head_dim]
            start_pos: Starting position for cache
            
        Returns:
            Tuple of (hidden_states, new_k, new_v)
        """
        # Self Attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_k, new_v = self.self_attn(
            hidden_states,
            positions,
            rotary_emb,
            attention_mask,
            k_cache,
            v_cache,
            start_pos,
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, new_k, new_v


class FastdLLMV2Edge(nn.Module):
    """Complete FastdLLM V2 model for edge deployment with KV Cache.
    
    Simplified architecture compatible with ExecuTorch export.
    Supports both prefill and incremental decode modes.
    """
    
    def __init__(self, config: FastdLLMV2EdgeConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Rotary embedding (shared across layers)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        
        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small standard deviation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional KV Cache.
        
        Args:
            input_ids: [batch_size, seq_len] token indices
            positions: [batch_size, seq_len] position indices (auto if None)
            attention_mask: Optional attention mask
            kv_cache: Optional KV cache tensor of shape
                     [num_layers, 2, batch, kv_heads, max_seq, head_dim]
            start_pos: Starting position for incremental generation
            
        Returns:
            Tuple of (logits, updated_kv_cache)
            logits: [batch_size, seq_len, vocab_size]
            updated_kv_cache: Updated cache if input cache was provided
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.arange(
                start_pos, start_pos + seq_len,
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare cache updates
        if kv_cache is not None:
            # Extract k and v caches: [layers, 2, batch, kv_heads, max_seq, head_dim]
            k_caches = kv_cache[:, 0]  # [layers, batch, kv_heads, max_seq, head_dim]
            v_caches = kv_cache[:, 1]
            
            # Will collect new k,v for each layer
            new_kv_list = []
        
        # Apply transformer layers
        for layer_idx, layer in enumerate(self.layers):
            if kv_cache is not None:
                # Get cache for this layer
                k_cache = k_caches[layer_idx]
                v_cache = v_caches[layer_idx]
                
                hidden_states, new_k, new_v = layer(
                    hidden_states,
                    positions,
                    self.rotary_emb,
                    attention_mask,
                    k_cache,
                    v_cache,
                    start_pos,
                )
                
                # Store updated cache for this layer
                # new_k, new_v: [batch, kv_heads, seq_len, head_dim]
                new_kv_list.append(torch.stack([new_k, new_v], dim=0))
            else:
                hidden_states, _, _ = layer(
                    hidden_states,
                    positions,
                    self.rotary_emb,
                    attention_mask,
                )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Prepare updated cache
        if kv_cache is not None:
            # Stack all layer caches: [layers, 2, batch, kv_heads, seq_len, head_dim]
            updated_kv = torch.stack(new_kv_list, dim=0)
            return logits, updated_kv
        
        return logits, None
    
    def generate_step(
        self,
        input_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single generation step for autoregressive decoding.
        
        This is a convenience method for incremental generation.
        
        Args:
            input_ids: [batch_size, 1] single token
            kv_cache: [num_layers, 2, batch, kv_heads, max_seq, head_dim]
            start_pos: Current position in sequence
            
        Returns:
            Tuple of (logits, updated_kv_cache)
            logits: [batch_size, 1, vocab_size]
            updated_kv_cache: Full updated cache
        """
        logits, updated_kv = self.forward(
            input_ids=input_ids,
            start_pos=start_pos,
            kv_cache=kv_cache,
        )
        return logits, updated_kv
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings for weight tying."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set input embeddings."""
        self.embed_tokens = value
    
    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return next(self.parameters()).dtype
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device


__all__ = [
    "FastdLLMV2EdgeConfig",
    "FastdLLMV2Edge",
]
