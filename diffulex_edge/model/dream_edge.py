"""Dream model for DiffuLex Edge.

Edge-optimized version that aligns with HF Dream implementation.
Uses shared components from components/ module.

Key differences from standard causal LLM:
- is_causal=False (non-causal attention for diffusion)
- Uses KV cache for efficient generation
- Supports 4D attention mask [B, 1, N, N]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from .base import ModelConfig, DiffusionModel
from ..components import RMSNorm, RotaryEmbedding, SwiGLUMLP


@dataclass
class DreamEdgeConfig(ModelConfig):
    """Configuration for Dream Edge model."""
    # Dream-specific defaults (7B model)
    vocab_size: int = 152064
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4  # GQA: 28 attention heads, 4 kv heads
    intermediate_size: int = 18944
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    attention_bias: bool = True  # Dream uses bias in Q, K, V projections
    attention_dropout: float = 0.0
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    
    # Diffusion-specific tokens
    mask_token_id: int = 151666
    pad_token_id: int = 151643
    bos_token_id: int = 151643
    eos_token_id: int = 151643


class DreamAttention(nn.Module):
    """Dream attention mechanism with KV cache support.
    
    Aligns with HF Dream implementation:
    - Non-causal attention (is_causal=False)
    - Supports 4D attention mask [B, 1, N, N]
    - Uses GQA (Grouped Query Attention)
    - Supports KV cache for efficient generation
    """
    
    def __init__(self, config: DreamEdgeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # Dream uses non-causal attention
        
        # Projections with bias (Dream specific)
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_len: int = 0,
        max_cache_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with dynamic KV cache (for Python inference).
        
        Args:
            positions: Position indices [batch_size, seq_len]
            hidden_states: [batch_size, seq_len, hidden_size]
            k_cache: Optional key cache [batch_size, num_kv_heads, cache_len, head_dim]
            v_cache: Optional value cache [batch_size, num_kv_heads, cache_len, head_dim]
            cache_len: Length of valid cache entries
            max_cache_len: Maximum cache length (unused, for API compatibility)
            
        Returns:
            Tuple of (attn_output, new_k, new_v)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        
        # Transpose for attention: [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Concatenate with cache (dynamic)
        if k_cache is not None and v_cache is not None and cache_len > 0:
            k_valid = k_cache[:, :, :cache_len, :].to(k.dtype)
            v_valid = v_cache[:, :, :cache_len, :].to(v.dtype)
            k = torch.cat([k_valid, k], dim=2)
            v = torch.cat([v_valid, v], dim=2)
        
        new_k, new_v = k.contiguous(), v.contiguous()
        
        # Handle GQA
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_kv_groups)
            v = repeat_kv(v, self.num_kv_groups)
        
        # Attention using SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, scale=self.scaling
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        return self.o_proj(attn_output), new_k, new_v
    
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
        """Static forward for Block Diffusion / ExecuTorch export.
        
        Args:
            positions: Position indices [batch_size, seq_len]
            hidden_states: [batch_size, seq_len, hidden_size]
            old_k_cache: [batch_size, num_kv_heads, max_len, head_dim]
            old_v_cache: [batch_size, num_kv_heads, max_len, head_dim]
            attention_mask: [batch_size, 1, seq_len, max_len + seq_len]
            insert_matrix: [batch_size, 1, max_len, seq_len]
            keep_mask: [batch_size, 1, max_len, 1]
            
        Returns:
            Tuple of (block_out, updated_k, updated_v)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        
        # Transpose for attention
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Step 1: Static Attention
        full_k = torch.cat([old_k_cache.to(k.dtype), k], dim=2)
        full_v = torch.cat([old_v_cache.to(v.dtype), v], dim=2)
        
        # Handle GQA
        if self.num_kv_heads != self.num_heads:
            num_repeat = self.num_heads // self.num_kv_heads
            full_k = full_k.repeat_interleave(num_repeat, dim=1)
            full_v = full_v.repeat_interleave(num_repeat, dim=1)
        
        # Attention computation
        scores = torch.matmul(q, full_k.transpose(-1, -2)) * self.scaling
        scores = scores + attention_mask.to(q.dtype)
        probs = torch.softmax(scores, dim=-1)
        block_out = torch.matmul(probs, full_v)
        
        block_out = block_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        block_out = self.o_proj(block_out)
        
        # Step 2: Static Cache Update
        insert_expanded = insert_matrix.expand(-1, self.num_kv_heads, -1, -1)
        expanded_k = torch.matmul(insert_expanded.to(k.dtype), k)
        expanded_v = torch.matmul(insert_expanded.to(v.dtype), v)
        
        keep_expanded = keep_mask.expand(-1, self.num_kv_heads, -1, self.head_dim)
        updated_k = (old_k_cache.to(k.dtype) * keep_expanded.to(k.dtype)) + expanded_k
        updated_v = (old_v_cache.to(v.dtype) * keep_expanded.to(v.dtype)) + expanded_v
        
        return block_out, updated_k, updated_v


class DreamDecoderLayer(nn.Module):
    """Dream transformer decoder layer."""
    
    def __init__(self, config: DreamEdgeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DreamAttention(config)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_len: int = 0,
        max_cache_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with dynamic KV cache."""
        # Self Attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_k, new_v = self.self_attn(
            positions, hidden_states, k_cache, v_cache, cache_len, max_cache_len
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
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
        """Static forward for ExecuTorch export."""
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


class DreamEdge(DiffusionModel):
    """Dream model for diffusion language modeling (Edge version).
    
    Aligns with HF DreamModel implementation:
    - Non-causal attention for diffusion
    - Supports KV cache
    - 4D attention mask support
    """
    
    def __init__(self, config: DreamEdgeConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        # Ensure padding_idx is within vocab range
        self.padding_idx = min(config.pad_token_id, config.vocab_size - 1)
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            DreamDecoderLayer(config) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Token IDs for diffusion
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with std from config."""
        std = self.config.hidden_size ** -0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Forward with dynamic KV cache (for DiffusionEngine).
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            positions: Position indices [batch_size, seq_len]
            mask: Optional attention mask (unused in standard Dream)
            kv_cache: Optional list of (k, v) tuples from previous forward
            max_seq_len: Maximum sequence length for cache
            
        Returns:
            Tuple of (logits, new_kv_cache)
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
            
            hidden_states, new_k, new_v = layer(
                positions, hidden_states, k_cache, v_cache, cache_len, max_seq_len
            )
            new_kv_cache.append((new_k, new_v))
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, new_kv_cache
    
    def forward_export(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        insert_matrix: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with fixed-shape KV cache for Block Diffusion / ExecuTorch export.
        
        Args:
            input_ids: [batch_size, seq_len]
            positions: [batch_size, seq_len]
            kv_cache: [num_layers, 2, batch_size, num_kv_heads, max_len, head_dim]
            attention_mask: [num_layers, batch_size, 1, seq_len, max_len + seq_len]
            insert_matrix: [num_layers, batch_size, 1, max_len, seq_len]
            keep_mask: [num_layers, batch_size, 1, max_len, 1]
            
        Returns:
            Tuple of (logits, updated_kv_cache)
        """
        hidden_states = self.embed_tokens(input_ids)
        num_layers = len(self.layers)
        updated_kv_cache = torch.zeros_like(kv_cache)
        
        for i in range(num_layers):
            layer = self.layers[i]
            k_cache = kv_cache[i, 0]
            v_cache = kv_cache[i, 1]
            
            hidden_states, new_k, new_v = layer.forward_export(
                positions, hidden_states, k_cache, v_cache,
                attention_mask[i], insert_matrix[i], keep_mask[i]
            )
            
            updated_kv_cache[i, 0] = new_k
            updated_kv_cache[i, 1] = new_v
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, updated_kv_cache
    
    def get_export_wrapper(self) -> Optional[nn.Module]:
        """Get the export wrapper for Dream.
        
        Dream uses BlockDiffusionWrapper for its special mask-based export format.
        """
        from .wrapper import BlockDiffusionWrapper
        return BlockDiffusionWrapper(
            self,
            block_size=1,  # Dream doesn't use block diffusion by default
            max_seq_len=self.config.max_position_embeddings,
        )
    
    def get_export_inputs(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, ...]:
        """Create example inputs for Dream export.
        
        Uses Block Diffusion format with special mask inputs.
        """
        max_seq_len = self.config.max_position_embeddings
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        
        # Standard inputs
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        positions = torch.arange(
            seq_len, dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # KV cache
        kv_cache = torch.zeros(
            num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim,
            dtype=torch.float32, device=device
        )
        
        # Block Diffusion masks
        attention_mask = torch.zeros(
            num_layers, batch_size, 1, seq_len, max_seq_len + seq_len,
            dtype=torch.float32, device=device
        )
        attention_mask[:, :, :, :, 0:max_seq_len] = -10000.0
        
        insert_matrix = torch.zeros(
            num_layers, batch_size, 1, max_seq_len, seq_len,
            dtype=torch.float32, device=device
        )
        for i in range(seq_len):
            insert_matrix[:, :, :, i, i] = 1.0
        
        keep_mask = torch.ones(
            num_layers, batch_size, 1, max_seq_len, 1,
            dtype=torch.float32, device=device
        )
        keep_mask[:, :, :, 0:seq_len, :] = 0.0
        
        return (input_ids, positions, kv_cache, attention_mask, insert_matrix, keep_mask)
    
__all__ = [
    "DreamEdgeConfig",
    "DreamEdge",
    "DreamAttention",
    "DreamDecoderLayer",
]
