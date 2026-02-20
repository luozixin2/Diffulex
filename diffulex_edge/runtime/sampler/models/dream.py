"""Dream sampler with block confirmation mechanism.

Key differences from SDAR:
- Uses shifted logits (like SDAR)
- Per-block threshold (not global)
- pre_block_complete logic
- Mask is "fully visible" (can attend to all confirmed blocks)

Key differences from LLaDA:
- Uses shifted logits (LLaDA doesn't)
"""

import torch
from typing import List
from dataclasses import dataclass, field

from diffulex_edge.runtime.sampler.base import sample_tokens
from diffulex_edge.runtime.sampler.shift import ShiftLogitsSampler
from diffulex_edge.runtime.block import DiffusionBlockManager, BlockStatus


@dataclass
class DreamSampleOutput:
    """Output from Dream sampling."""
    accepted_local_positions: List[int] = field(default_factory=list)
    accepted_tokens: List[int] = field(default_factory=list)
    block_confirmed: bool = False


class DreamSampler:
    """Dream sampler with block confirmation.
    
    Key characteristics:
    - Uses shifted logits (like SDAR, unlike LLaDA)
    - Per-block threshold
    - pre_block_complete: if previous block confirmed, must accept at least 1
    """
    
    def __init__(
        self,
        mask_token_id: int = 126336,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        margin_confidence: bool = False,
        neg_entropy: bool = False,
    ):
        self.mask_token_id = mask_token_id
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.margin_confidence = margin_confidence
        self.neg_entropy = neg_entropy
        self.shift_sampler = ShiftLogitsSampler()
    
    def sample(self, block_manager: DiffusionBlockManager, logits: torch.Tensor) -> DreamSampleOutput:
        """Sample tokens for the ACTIVE block.
        
        Dream uses shifted logits + per-block threshold + pre_block_complete logic.
        """
        active_block = block_manager.get_active_block()
        if active_block is None:
            return DreamSampleOutput(block_confirmed=True)
        
        mask_positions = active_block.get_local_mask_positions()
        if not mask_positions:
            return DreamSampleOutput(block_confirmed=True)
        
        # Shift logits (key difference from LLaDA)
        last_logits = self.shift_sampler.fetch_last_logits(logits)
        shifted_logits = self.shift_sampler.shift(logits, last_logits)
        
        block_logits = shifted_logits[mask_positions]
        
        confidence, sampled_tokens, initial_confidence = sample_tokens(
            block_logits,
            temperature=self.temperature,
            top_p=self.top_p if self.top_p < 1.0 else None,
            top_k=self.top_k if self.top_k > 0 else None,
            margin_confidence=self.margin_confidence,
            neg_entropy=self.neg_entropy,
        )
        
        # Per-block threshold
        threshold = active_block.accept_threshold
        high_conf_indices = torch.where(initial_confidence > threshold)[0]
        
        # Check if previous block is confirmed (pre_block_complete logic)
        prev_block_confirmed = True  # First block always "complete"
        if active_block.block_id > 0:
            for block in block_manager.blocks:
                if block.block_id == active_block.block_id - 1:
                    prev_block_confirmed = (block.status == BlockStatus.CONFIRMED)
                    break
        
        if prev_block_confirmed:
            # Must accept at least one token
            if len(high_conf_indices) == 0:
                _, top_indices = torch.topk(confidence, 1)
                accepted_relative_indices = top_indices.cpu().tolist()
            else:
                accepted_relative_indices = high_conf_indices.cpu().tolist()
        else:
            # Only accept high confidence tokens
            if len(high_conf_indices) == 0:
                accepted_relative_indices = []
            else:
                accepted_relative_indices = high_conf_indices.cpu().tolist()
        
        accepted_local_positions = []
        accepted_tokens = []
        
        for rel_idx in accepted_relative_indices:
            local_pos = mask_positions[rel_idx]
            token_id = sampled_tokens[rel_idx].item()
            active_block.confirm_token(local_pos, token_id)
            accepted_local_positions.append(local_pos)
            accepted_tokens.append(token_id)
        
        return DreamSampleOutput(
            accepted_local_positions=accepted_local_positions,
            accepted_tokens=accepted_tokens,
            block_confirmed=active_block.is_confirmed,
        )
    
    def reset(self) -> None:
        self.shift_sampler.reset_cache()
