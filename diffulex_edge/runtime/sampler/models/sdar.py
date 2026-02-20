"""SDAR sampler with block confirmation mechanism.

Aligned with diffulex behavior:
1. Only sample tokens in the ACTIVE block
2. Use shifted logits (like FastdLLM V2)
3. Global threshold with "accept at least one" policy
"""

import torch
from typing import Dict, List
from dataclasses import dataclass, field

from diffulex_edge.runtime.sampler.base import sample_tokens
from diffulex_edge.runtime.sampler.shift import ShiftLogitsSampler
from diffulex_edge.runtime.block import DiffusionBlockManager, BlockStatus


@dataclass
class SDARSampleOutput:
    """Output from SDAR sampling."""
    accepted_local_positions: List[int] = field(default_factory=list)  # Positions within active block
    accepted_tokens: List[int] = field(default_factory=list)  # Token IDs
    block_confirmed: bool = False  # Whether this block is now fully confirmed


class SDARSampler:
    """SDAR sampler with block confirmation.
    
    Key changes from original:
    1. Only samples ACTIVE block (not all blocks)
    2. Returns which positions were accepted
    3. Tracks block confirmation status
    """
    
    def __init__(
        self,
        mask_token_id: int = 126336,
        threshold: float = 0.95,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        margin_confidence: bool = False,
        neg_entropy: bool = False,
    ):
        self.mask_token_id = mask_token_id
        self.threshold = threshold
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.margin_confidence = margin_confidence
        self.neg_entropy = neg_entropy
        
        # SDAR uses shifted logits
        self.shift_sampler = ShiftLogitsSampler()
    
    def sample(
        self,
        block_manager: DiffusionBlockManager,
        logits: torch.Tensor,
    ) -> SDARSampleOutput:
        """Sample tokens for the ACTIVE block only.
        
        Args:
            block_manager: Block manager with active block
            logits: Logits for active block [block_len, vocab_size]
            
        Returns:
            SDARSampleOutput with accepted positions and tokens
        """
        active_block = block_manager.get_active_block()
        if active_block is None:
            return SDARSampleOutput(block_confirmed=True)
        
        # Get mask positions within the active block
        mask_positions = active_block.get_local_mask_positions()
        if not mask_positions:
            # Block already complete
            return SDARSampleOutput(block_confirmed=True)
        
        # Logits correspond to active block positions directly
        # Get logits for mask positions only
        block_logits = logits[mask_positions]  # [num_masks, vocab_size]
        
        # Sample tokens
        confidence, sampled_tokens, initial_confidence = sample_tokens(
            block_logits,
            temperature=self.temperature,
            top_p=self.top_p if self.top_p < 1.0 else None,
            top_k=self.top_k if self.top_k > 0 else None,
            margin_confidence=self.margin_confidence,
            neg_entropy=self.neg_entropy,
        )
        
        # Accept tokens above threshold (SDAR logic: always accept at least one)
        high_conf_indices = torch.where(initial_confidence > self.threshold)[0]
        
        if len(high_conf_indices) == 0:
            # Always accept at least one (highest confidence)
            max_prob_idx = initial_confidence.argmax().item()
            accepted_relative_indices = [max_prob_idx]
        else:
            # Accept all high confidence + at least one
            max_prob_idx = initial_confidence.argmax().item()
            accepted_relative_indices = torch.unique(
                torch.cat([high_conf_indices, torch.tensor([max_prob_idx], device=high_conf_indices.device)])
            ).cpu().tolist()
        
        # Convert to local positions and tokens
        accepted_local_positions = []
        accepted_tokens = []
        
        for rel_idx in accepted_relative_indices:
            local_pos = mask_positions[rel_idx]
            token_id = sampled_tokens[rel_idx].item()
            
            # Confirm token in block
            is_last = active_block.confirm_token(local_pos, token_id)
            
            accepted_local_positions.append(local_pos)
            accepted_tokens.append(token_id)
        
        # Check if block is now confirmed
        block_confirmed = active_block.is_confirmed
        
        return SDARSampleOutput(
            accepted_local_positions=accepted_local_positions,
            accepted_tokens=accepted_tokens,
            block_confirmed=block_confirmed,
        )
    
    def reset(self) -> None:
        """Reset sampler state."""
        self.shift_sampler.reset_cache()
