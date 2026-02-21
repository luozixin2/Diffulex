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
from dataclasses import dataclass
from typing import Optional, Tuple

from ..components import RMSNorm, RotaryEmbedding, SwiGLUMLP


@dataclass
class DreamEdgeConfig:
    """Configuration for Dream Edge model."""
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
    
    def __post_init__(self):
        """Validate and set derived values."""
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for GQA.
    
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DreamAttention(nn.Module):
    """Dream attention mechanism with KV cache support.
    
    Aligns with HF Dream implementation:
    - Non-causal attention (is_causal=False)
    - Supports 4D attention mask [B, 1, N, N]
    - Uses GQA (Grouped Query Attention)
    - Supports KV cache for efficient generation
    """
    
    def __init__(self, config: DreamEdgeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV cache.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional 4D mask [batch_size, 1, seq_len, kv_len]
            position_ids: [batch_size, seq_len] or [seq_len]
            past_key_value: Optional tuple of (k_cache, v_cache)
                k_cache/v_cache: [batch_size, num_kv_heads, cache_len, head_dim]
            use_cache: Whether to return updated KV cache
            
        Returns:
            Tuple of (attn_output, present_key_value)
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if position_ids is None:
            # Default to sequential positions starting from cache length
            if past_key_value is not None:
                past_len = past_key_value[0].shape[2]
                position_ids = torch.arange(
                    past_len, past_len + q_len, dtype=torch.long, device=hidden_states.device
                ).unsqueeze(0)
            else:
                position_ids = torch.arange(
                    0, q_len, dtype=torch.long, device=hidden_states.device
                ).unsqueeze(0)
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        
        query_states, key_states = self.rotary_emb(position_ids, query_states, key_states)
        
        # Update KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        present_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat k/v heads for GQA
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        
        # Apply attention mask
        if attention_mask is not None:
            # Slice mask to match kv_len
            kv_len = key_states.shape[-2]
            causal_mask = attention_mask[..., :kv_len]
            attn_weights = attn_weights + causal_mask
        
        # Softmax (upcast to fp32 for stability)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value


class DreamDecoderLayer(nn.Module):
    """Dream transformer decoder layer."""
    
    def __init__(self, config: DreamEdgeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DreamAttention(config, layer_idx)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional 4D mask
            position_ids: Position indices
            past_key_value: Optional KV cache
            use_cache: Whether to return updated cache
            
        Returns:
            Tuple of (hidden_states, present_key_value)
        """
        residual = hidden_states
        
        # Self Attention with pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MLP with post-attention norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class DreamEdge(nn.Module):
    """Dream model for diffusion language modeling (Edge version).
    
    Aligns with HF DreamModel implementation:
    - Non-causal attention for diffusion
    - Supports KV cache
    - 4D attention mask support
    """
    
    def __init__(self, config: DreamEdgeConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            DreamDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        """Forward pass aligned with HF DreamModel.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional 4D mask [batch_size, 1, seq_len, kv_len]
                or 2D mask [batch_size, seq_len] that will be converted to 4D
            position_ids: [batch_size, seq_len] or [seq_len]
            past_key_values: Optional tuple of layer caches
                Each layer cache is (k_cache, v_cache) with shape
                [batch_size, num_kv_heads, cache_len, head_dim]
            use_cache: Whether to return updated KV cache
            
        Returns:
            Tuple of (logits, present_key_values)
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert 2D attention mask to 4D if needed
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = self._prepare_4d_attention_mask(
                attention_mask, input_ids.dtype, seq_len
            )
        
        # Embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        # Prepare position ids
        if position_ids is None:
            if past_key_values is not None:
                past_len = past_key_values[0][0].shape[2]  # First layer, key cache, seq dim
                position_ids = torch.arange(
                    past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device
                )
            else:
                position_ids = torch.arange(
                    0, seq_len, dtype=torch.long, device=input_ids.device
                )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Decode layers
        present_key_values = [] if use_cache else None
        
        for layer_idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            
            hidden_states, present_kv = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                present_key_values.append(present_kv)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if use_cache:
            present_key_values = tuple(present_key_values)
        
        return logits, present_key_values
    
    def _prepare_4d_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
        tgt_len: int,
    ) -> torch.Tensor:
        """Convert 2D attention mask to 4D causal mask.
        
        Args:
            attention_mask: 2D mask [batch_size, seq_len]
            dtype: Target dtype
            tgt_len: Target sequence length
            
        Returns:
            4D mask [batch_size, 1, seq_len, seq_len]
        """
        bsz, src_len = attention_mask.shape
        
        # Ensure dtype is float for finfo
        if not dtype.is_floating_point:
            dtype = torch.float32
        
        # Convert to float and expand
        mask = attention_mask.to(dtype)
        
        # Create 4D mask: [bsz, 1, tgt_len, src_len]
        # For non-causal attention, we just mask out padded positions
        expanded_mask = mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, tgt_len, src_len)
        
        # Convert 0s to large negative values
        inverted_mask = 1.0 - expanded_mask
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )
        
        return inverted_mask
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
    ) -> dict:
        """Prepare inputs for generation with KV cache.
        
        This method helps align the interface with HF transformers.
        """
        batch_size, seq_len = input_ids.shape
        
        if past_key_values is not None:
            # Only use the last token if we have past KV
            past_len = past_key_values[0][0].shape[2]
            input_ids = input_ids[:, -1:]
            position_ids = torch.tensor([[past_len]], device=input_ids.device)
        else:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }


__all__ = [
    "DreamEdgeConfig",
    "DreamEdge",
    "DreamAttention",
    "DreamDecoderLayer",
]
