"""Fast DLLM V2 Edge - API aligned with SDAREdge.

This version aligns with SDAREdge API:
- Inherits from DiffusionModel base class
- Uses same forward signature: (input_ids, positions, mask=None, kv_cache=None, max_seq_len=None)
- Supports ExecuTorch export via forward_export()

Architecture (same as HF Fast_dLLM_Qwen):
- GQA: num_heads=12, num_kv_heads=2
- Q/K/V projections with bias
- O projection without bias
- tie_word_embeddings=True
- Block diffusion attention mask for generation
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelConfig, DiffusionModel
from ..components import RMSNorm, RotaryEmbedding, SwiGLUMLP


@dataclass
class FastDLLMv2EdgeConfig(ModelConfig):
    """Configuration for Fast DLLM V2 Edge model."""
    # Defaults from Fast_dLLM_v2_1.5B
    vocab_size: int = 151936
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2  # GQA
    intermediate_size: int = 8960
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    
    # Fast DLLM v2 specific
    attention_bias: bool = True  # Q/K/V have bias
    tie_word_embeddings: bool = True
    bd_size: int = 32  # Block diffusion size
    mask_token_id: int = 151665  # Fast dLLM v2 mask token
    pad_token_id: int = 151643
    
    def __post_init__(self):
        """Set derived values."""
        super().__post_init__()
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class FastDLLMv2Attention(nn.Module):
    """Fast DLLM V2 attention with KV cache support.
    
    Key differences from SDAR:
    - Q/K/V projections have bias (SDAR doesn't)
    - O projection doesn't have bias (same as SDAR)
    - Uses standard RoPE without per-head Q/K norm
    - Supports block diffusion attention mask
    """
    
    def __init__(self, config: FastDLLMv2EdgeConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # Q/K/V projections with bias (Fast DLLM v2 specific)
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        # O projection without bias
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        
        # Shared RoPE
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with dynamic KV cache.
        
        Args:
            positions: [batch_size, seq_len] position indices
            hidden_states: [batch_size, seq_len, hidden_size]
            k_cache: Optional [batch, kv_heads, max_cache_len, head_dim]
            v_cache: Optional [batch, kv_heads, max_cache_len, head_dim]
            cache_len: Current valid cache length
            max_cache_len: Maximum cache length
            attention_mask: Optional attention mask [batch, num_heads, q_len, kv_len]
            
        Returns:
            Tuple of (output, new_k, new_v)
        """
        batch_size, seq_len, _ = hidden_states.shape
        if max_cache_len is None:
            max_cache_len = k_cache.shape[2] if k_cache is not None else seq_len
        
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
        
        # Ensure consistent dtype
        target_dtype = q.dtype
        k = k.to(target_dtype)
        v = v.to(target_dtype)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Concatenate with cache (dynamic)
        if k_cache is not None and v_cache is not None and cache_len > 0:
            k_valid = k_cache[:, :, :cache_len, :].to(target_dtype)
            v_valid = v_cache[:, :, :cache_len, :].to(target_dtype)
            k = torch.cat([k_valid, k], dim=2)
            v = torch.cat([v_valid, v], dim=2)
        
        # Truncate to max_cache_len
        if k.shape[2] > max_cache_len:
            k = k[:, :, -max_cache_len:, :]
            v = v[:, :, -max_cache_len:, :]
        
        new_k, new_v = k.contiguous(), v.contiguous()
        
        # Handle GQA: repeat k, v heads if needed
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask should be in additive format (0 for keep, -inf for mask)
            scores = scores + attention_mask.to(target_dtype)
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(target_dtype)
        attn_output = torch.matmul(attn_weights, v)
        
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
        """Static forward for Block Diffusion / ExecuTorch export."""
        batch_size, block_size, _ = hidden_states.shape
        
        # Compute Q, K, V
        q = self.q_proj(hidden_states).view(
            batch_size, block_size, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch_size, block_size, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch_size, block_size, self.num_kv_heads, self.head_dim
        )
        
        # Transpose for attention: [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # Ensure consistent dtype
        target_dtype = q.dtype
        k = k.to(target_dtype)
        v = v.to(target_dtype)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Step 1: Static Attention
        full_k = torch.cat([old_k_cache.to(target_dtype), k], dim=2)
        full_v = torch.cat([old_v_cache.to(target_dtype), v], dim=2)
        
        # Handle GQA
        if self.num_kv_heads != self.num_heads:
            num_repeat = self.num_heads // self.num_kv_heads
            full_k = full_k.repeat_interleave(num_repeat, dim=1)
            full_v = full_v.repeat_interleave(num_repeat, dim=1)
        
        # Attention computation
        scores = torch.matmul(q, full_k.transpose(-1, -2)) * self.scaling
        scores = scores + attention_mask.to(target_dtype)
        probs = torch.softmax(scores, dim=-1)
        block_out = torch.matmul(probs, full_v)
        
        block_out = block_out.transpose(1, 2).contiguous().view(
            batch_size, block_size, -1
        )
        block_out = self.o_proj(block_out)
        
        # Step 2: Static Cache Update
        insert_expanded = insert_matrix.expand(-1, self.num_kv_heads, -1, -1)
        expanded_k = torch.matmul(insert_expanded.to(target_dtype), k)
        expanded_v = torch.matmul(insert_expanded.to(target_dtype), v)
        
        keep_expanded = keep_mask.expand(-1, self.num_kv_heads, -1, self.head_dim)
        updated_k = (old_k_cache.to(target_dtype) * keep_expanded.to(target_dtype)) + expanded_k
        updated_v = (old_v_cache.to(target_dtype) * keep_expanded.to(target_dtype)) + expanded_v
        
        return block_out, updated_k, updated_v


class FastDLLMv2DecoderLayer(nn.Module):
    """Fast DLLM V2 decoder layer."""
    
    def __init__(self, config: FastDLLMv2EdgeConfig):
        super().__init__()
        self.self_attn = FastDLLMv2Attention(config)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size, bias=False)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_len: int = 0,
        max_cache_len: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with dynamic KV cache."""
        # Self Attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_k, new_v = self.self_attn(
            positions, hidden_states, k_cache, v_cache, cache_len, max_cache_len,
            attention_mask
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


class FastDLLMv2Edge(DiffusionModel):
    """Fast DLLM V2 model supporting both dynamic and fixed-shape KV cache.
    
    This version aligns with SDAREdge API:
    - Inherits from DiffusionModel base class
    - Same forward signature
    - Supports ExecuTorch export
    
    Key features:
    - tie_word_embeddings: LM head shares weights with embedding
    - GQA with bias in Q/K/V projections
    - Block diffusion attention mask support
    """
    
    def __init__(self, config: FastDLLMv2EdgeConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            FastDLLMv2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head (bias=False like HF)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        
        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
    
    def _create_block_diffusion_mask(
        self,
        seq_len: int,
        cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create block diffusion attention mask.
        
        For Fast DLLM v2 inference, this creates a mask where each query token
        can attend to all key tokens in the same block or earlier blocks.
        
        Args:
            seq_len: Length of query sequence
            cache_len: Length of cached key/value sequence
            device: Device for the mask
            dtype: Dtype for the mask
            
        Returns:
            Attention mask of shape [seq_len, cache_len + seq_len]
            with 0.0 for positions to attend to and -inf for positions to mask.
        """
        bd_size = self.config.bd_size
        
        # Create position indices
        q_indices = torch.arange(seq_len, device=device) + cache_len
        k_indices = torch.arange(cache_len + seq_len, device=device)
        
        # Compute block indices
        q_blocks = q_indices // bd_size
        k_blocks = k_indices // bd_size
        
        # Mask: query can attend to key if query's block >= key's block
        mask = q_blocks.unsqueeze(1) >= k_blocks.unsqueeze(0)
        
        # Convert to additive mask (0 for keep, -inf for mask)
        additive_mask = torch.where(mask, 0.0, float('-inf'))
        
        return additive_mask.to(dtype)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward with dynamic KV cache (for DiffusionEngine).
        
        Args:
            input_ids: [batch_size, seq_len] token indices
            positions: [batch_size, seq_len] position indices
            mask: Optional attention mask [batch, 1, seq_len, total_len]
            kv_cache: Optional list of (k, v) tuples from previous forward pass
            max_seq_len: Maximum sequence length for cache
            
        Returns:
            Tuple of (logits, new_kv_cache)
            - logits: [batch_size, seq_len, vocab_size]
            - new_kv_cache: List of (k, v) tuples
        """
        if max_seq_len is None:
            max_seq_len = self.config.max_position_embeddings
        
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        new_kv_cache = []
        
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = None, None
            cache_len = 0
            if kv_cache is not None and i < len(kv_cache):
                k_cache, v_cache = kv_cache[i]
                cache_len = k_cache.shape[2] if k_cache is not None else 0
            
            hidden_states, new_k, new_v = layer(
                positions, hidden_states, k_cache, v_cache, cache_len, max_seq_len,
                mask
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
        """Forward with fixed-shape KV cache for Block Diffusion / ExecuTorch export."""
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
        """Get the export wrapper for Fast DLLM V2.
        
        Uses BlockDiffusionWrapper for its special mask-based export format.
        """
        from .wrapper import BlockDiffusionWrapper
        return BlockDiffusionWrapper(
            self,
            block_size=self.config.bd_size,
            max_seq_len=self.config.max_position_embeddings,
        )
    
    def get_export_inputs(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, ...]:
        """Create example inputs for Fast DLLM V2 export.
        
        Uses Block Diffusion format with special mask inputs.
        """
        block_size = self.config.bd_size
        max_seq_len = self.config.max_position_embeddings
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        
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
        
        # Block Diffusion masks
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


__all__ = ["FastDLLMv2EdgeConfig", "FastDLLMv2Edge"]
