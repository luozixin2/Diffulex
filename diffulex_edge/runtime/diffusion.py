"""Diffusion sampling for dLLM models.

Provides diffusion-based generation capabilities for DiffuLex Edge models,
including DiffusionBlock management, logits shifting, and confidence-based
token acceptance.

Note: This module now serves as a compatibility layer. Core classes are imported
from block.py and engine.py for a cleaner architecture.
"""

import logging
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Iterator

# Import core classes from their canonical locations
from diffulex_edge.runtime.block import (
    DiffusionBlock as _DiffusionBlock,
    DiffusionBlockManager as _DiffusionBlockManager,
    BlockStatus,
)
from diffulex_edge.runtime.engine import (
    DiffusionEngine as _DiffusionEngine,
    DiffusionGenerationConfig as _DiffusionGenerationConfig,
)

# Re-export for backward compatibility
DiffusionBlock = _DiffusionBlock
DiffusionBlockManager = _DiffusionBlockManager
DiffusionEngine = _DiffusionEngine
DiffusionGenerationConfig = _DiffusionGenerationConfig

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class SampleOutput:
    """Output from diffusion sampling.
    
    Attributes:
        sampled_tokens: Map from position to sampled token ID
        confidence_scores: Map from position to confidence score
        accepted_tokens: Map from position to accepted token ID (high confidence)
        block_states: Final state of each block
    """
    sampled_tokens: Dict[int, int] = field(default_factory=dict)
    confidence_scores: Dict[int, float] = field(default_factory=dict)
    accepted_tokens: Dict[int, int] = field(default_factory=dict)
    block_states: List[Dict[str, Any]] = field(default_factory=list)


class DiffusionSampler:
    """Diffusion sampler for dLLM models.
    
    Implements the core diffusion sampling algorithm with:
    - Logits shifting for dependency modeling
    - Confidence-based token acceptance
    - Temperature, top-k, and top-p sampling
    - Margin confidence and negative entropy options (aligned with Diffulex)
    """
    
    def __init__(
        self,
        mask_token_id: int = 151665,  # Fast dLLM v2 default
        confidence_threshold: float = 0.9,  # Aligned with diffulex
        temperature: float = 1.0,
        top_k: int = 0,  # 0 means disabled
        top_p: float = 1.0,  # 1.0 means disabled
        margin_confidence: bool = False,
        neg_entropy: bool = False,
    ):
        """Initialize diffusion sampler.
        
        Args:
            mask_token_id: Token ID for mask tokens (Fast dLLM v2: 151665, Dream: 151666)
            confidence_threshold: Threshold for accepting tokens (0-1), aligned with diffulex: 0.9
            temperature: Sampling temperature
            top_k: Top-k sampling parameter (0 = disabled)
            top_p: Top-p (nucleus) sampling parameter (1.0 = disabled)
            margin_confidence: Use margin confidence (top1 - top2 probs)
            neg_entropy: Use negative entropy as confidence
        """
        self.mask_token_id = mask_token_id
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.margin_confidence = margin_confidence
        self.neg_entropy = neg_entropy
        
        # Cache for last logits (for multi-step generation)
        self._last_logits: Optional[torch.Tensor] = None
    
    def shift_logits(
        self, 
        logits: torch.Tensor, 
        last_logits: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Shift logits right by one position.
        
        This is the core operation for diffusion models that shifts
        the prediction target to model token dependencies.
        
        Aligned with Diffulex SamplerShiftLogits._shift_logits:
        - Position 0 gets last_logits (or 1.0 if not provided)
        - Position i gets logits from position i-1
        
        Args:
            logits: Logits tensor [seq_len, vocab_size]
            last_logits: Logits for the last position (for circular shift)
                        If None and use_cache=True, use cached value or last logits
            use_cache: Whether to use/update cached last_logits
                        
        Returns:
            Shifted logits of same shape
            
        Raises:
            ValueError: If logits has invalid shape or contains NaN/Inf
        """
        if logits.dim() != 2:
            raise ValueError(f"Expected 2D logits [seq_len, vocab_size], got {logits.shape}")
        
        if torch.isnan(logits).any():
            raise ValueError("Logits contain NaN values")
        
        if torch.isinf(logits).any():
            raise ValueError("Logits contain Inf values")
        
        seq_len, vocab_size = logits.shape
        
        if seq_len == 0:
            # Empty logits - return as is
            return logits
        
        # Determine last_logits to use
        if last_logits is None:
            if use_cache and self._last_logits is not None:
                last_logits = self._last_logits
            else:
                # Use last logit from current batch
                last_logits = logits[-1]
        
        if use_cache:
            # Cache for next iteration
            self._last_logits = last_logits.clone()
        
        # Shift right: position i gets logits from position i-1
        # Position 0 gets last_logits (or 1.0 if no last_logits available)
        shifted = torch.zeros_like(logits)
        if last_logits is not None:
            shifted[0] = last_logits
        else:
            # Fallback: use 1.0 (aligned with Diffulex)
            shifted[0] = 1.0
        
        if seq_len > 1:
            shifted[1:] = logits[:-1]
        
        return shifted
    
    def reset_cache(self):
        """Reset cached last logits."""
        self._last_logits = None
    
    def compute_confidence(
        self, 
        logits: torch.Tensor,
        sampled_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute confidence scores for sampled tokens.
        
        Confidence is defined as the probability mass of the sampled token
        in the softmax distribution.
        
        Args:
            logits: Logits tensor [num_positions, vocab_size]
            sampled_tokens: Sampled token IDs [num_positions]
            
        Returns:
            Confidence scores [num_positions] in range [0, 1]
        """
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Gather probabilities of sampled tokens
        confidence = probs.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        
        return confidence
    
    def sample_tokens(
        self, 
        logits: torch.Tensor,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample tokens from logits with top-k/top-p filtering.
        
        Aligned with Diffulex SamplerBase.sample_tokens:
        Returns (confidence, sampled_tokens, initial_confidence)
        
        Args:
            logits: Logits tensor [num_positions, vocab_size]
            temperature: Override for temperature (uses self.temperature if None)
            top_p: Override for top_p (uses self.top_p if None)
            top_k: Override for top_k (uses self.top_k if None)
            
        Returns:
            Tuple of (confidence, sampled_tokens, initial_confidence)
            - confidence: [num_positions] (may be modified by margin_confidence/neg_entropy)
            - sampled_tokens: [num_positions] token IDs
            - initial_confidence: [num_positions] original probabilities
        """
        batch_size, vocab_size = logits.shape
        device = logits.device
        
        # Use instance defaults if not provided
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-p filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative prob > top_p
            sorted_indices_to_remove = cumsum_probs > top_p
            # Shift to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            k = min(top_k, vocab_size)
            indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample or take argmax
        if temperature > 0:
            try:
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                initial_confidence = probs.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                # Fallback to argmax if multinomial fails
                initial_confidence, sampled_tokens = probs.max(dim=-1)
        else:
            # Greedy (temperature = 0)
            initial_confidence, sampled_tokens = probs.max(dim=-1)
        
        confidence = initial_confidence.clone()
        
        # Apply margin confidence if enabled
        if self.margin_confidence:
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            top1_probs = sorted_probs[:, 0]
            top2_probs = sorted_probs[:, 1]
            confidence = top1_probs - top2_probs
        
        # Apply negative entropy if enabled
        if self.neg_entropy:
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            confidence = -torch.sum(probs * log_probs, dim=-1)
        
        return confidence, sampled_tokens, initial_confidence
    
    def sample_blocks(
        self,
        block_manager: DiffusionBlockManager,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        is_prefill: bool = True,
    ) -> SampleOutput:
        """Sample tokens for all active blocks.
        
        Aligned with Diffulex FastdLLMV2SamplerForDiffusionLM.forward and
        DreamSamplerForDiffusionLM.forward:
        - Samples from mask positions in each block
        - Accepts tokens based on confidence threshold
        - Always accepts at least the highest confidence token
        - Supports Dream's pre_block_complete logic
        
        Args:
            block_manager: Manager containing active blocks
            logits: Logits tensor [seq_len, vocab_size]
            temperature: Optional override for temperature
            top_p: Optional override for top_p
            top_k: Optional override for top_k
            threshold: Optional override for confidence_threshold
            is_prefill: Whether in prefill phase (affects mask position selection)
            
        Returns:
            SampleOutput with sampled tokens and acceptance info
        """
        output = SampleOutput()
        temp = temperature if temperature is not None else self.temperature
        global_threshold = threshold if threshold is not None else self.confidence_threshold
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        
        # Update pre_block_complete status for Dream
        block_manager.update_pre_block_complete()
        
        # Get active blocks
        active_blocks = block_manager.get_active_blocks()
        
        for block_id, block in active_blocks:
            # Skip inactive or empty blocks
            if not block.is_active:
                continue
            
            # Check if block has mask tokens (aligned with Diffulex)
            if sum(block.local_mask_token_ids) == 0:
                continue
            
            # Get mask positions for this block
            mask_positions = block.global_mask_token_ids
            
            if not mask_positions:
                continue
            
            # Extract logits for mask positions
            # Aligned with Diffulex: use local_mask_token_ids in decode, global in prefill
            if is_prefill:
                mask_token_logits = logits[mask_positions]  # [num_mask, vocab_size]
            else:
                local_mask_ids = block.local_mask_token_ids
                mask_token_logits = logits[local_mask_ids]  # [num_mask, vocab_size]
            
            # Sample tokens (aligned with Diffulex)
            confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                mask_token_logits,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
            )
            
            # Use per-block threshold if available (Dream), otherwise use global
            block_threshold = block.accept_threshold if hasattr(block, 'accept_threshold') else global_threshold
            
            # Determine which tokens to accept based on confidence
            # Aligned with DreamSamplerForDiffusionLM.forward
            high_conf_indices = torch.where(initial_confidence > block_threshold)[0]
            
            if block.pre_block_complete:
                # Dream logic: Previous blocks are complete
                # Can accept all high confidence tokens
                if len(high_conf_indices) == 0:
                    # No high confidence tokens, accept only top-1
                    number_transfer_tokens = 1
                    _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    accept_indices = transfer_index
                else:
                    # Accept all high confidence tokens
                    accept_indices = high_conf_indices
            else:
                # Dream logic: Previous blocks not complete
                # Only accept high confidence tokens
                accept_indices = high_conf_indices
            
            # Update output
            for i, pos in enumerate(mask_positions):
                output.sampled_tokens[pos] = sampled_tokens[i].item()
                output.confidence_scores[pos] = initial_confidence[i].item()
            
            # Record accepted tokens
            accept_list = accept_indices.cpu().tolist()
            for idx in accept_list:
                pos = mask_positions[idx]
                output.accepted_tokens[pos] = sampled_tokens[idx].item()
            
            # Update block with accepted tokens
            accepted_local = [
                pos - block.start_pos 
                for pos in output.accepted_tokens.keys()
                if block.start_pos <= pos < block.start_pos + block.length
                and pos in mask_positions
            ]
            accepted_token_ids = [
                output.accepted_tokens[block.start_pos + local_pos]
                for local_pos in accepted_local
            ]
            
            if accepted_local:
                block_manager.update_block(block_id, accepted_local, accepted_token_ids)
        
        # Record block states
        for block in block_manager.blocks:
            output.block_states.append({
                "is_active": block.is_active,
                "is_complete": block.is_complete,
                "num_accepted": len(block.accepted_token_ids),
                "num_masked": len(block.global_mask_token_ids),
                "pre_block_complete": block.pre_block_complete,
            })
        
        return output


# Backward compatibility aliases
InferenceEngine = DiffusionEngine
GenerationConfig = DiffusionGenerationConfig


__all__ = [
    # From block.py (re-exported for compatibility)
    "DiffusionBlock",
    "DiffusionBlockManager",
    "BlockStatus",
    # From engine.py (re-exported for compatibility)
    "DiffusionEngine",
    "DiffusionGenerationConfig",
    "InferenceEngine",  # Backward compatibility alias
    "GenerationConfig",  # Backward compatibility alias
    # Native to this module
    "SampleOutput",
    "DiffusionSampler",
]
