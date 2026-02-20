"""FastdLLM V2 sampler with block confirmation mechanism."""

import torch
from typing import List
from dataclasses import dataclass, field

from diffulex_edge.runtime.sampler.base import sample_tokens
from diffulex_edge.runtime.sampler.shift import ShiftLogitsSampler
from diffulex_edge.runtime.block import DiffusionBlockManager


@dataclass
class FastdLLMV2SampleOutput:
    """Output from FastdLLM V2 sampling."""
    accepted_local_positions: List[int] = field(default_factory=list)
    accepted_tokens: List[int] = field(default_factory=list)
    block_confirmed: bool = False


class FastdLLMV2Sampler:
    """FastdLLM V2 sampler with block confirmation.
    
    Key characteristics:
    - Uses shifted logits
    - Global threshold
    - Always accepts at least one token
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
        self.shift_sampler = ShiftLogitsSampler()
    
    def sample(self, block_manager: DiffusionBlockManager, logits: torch.Tensor) -> FastdLLMV2SampleOutput:
        """Sample tokens for the ACTIVE block only."""
        active_block = block_manager.get_active_block()
        if active_block is None:
            return FastdLLMV2SampleOutput(block_confirmed=True)
        
        mask_positions = active_block.get_local_mask_positions()
        if not mask_positions:
            return FastdLLMV2SampleOutput(block_confirmed=True)
        
        # Shift logits for active block
        last_logits = self.shift_sampler.fetch_last_logits(logits)
        shifted_logits = self.shift_sampler.shift(logits, last_logits)
        
        # Sample
        block_logits = shifted_logits[mask_positions]
        confidence, sampled_tokens, initial_confidence = sample_tokens(
            block_logits,
            temperature=self.temperature,
            top_p=self.top_p if self.top_p < 1.0 else None,
            top_k=self.top_k if self.top_k > 0 else None,
            margin_confidence=self.margin_confidence,
            neg_entropy=self.neg_entropy,
        )
        
        # Accept logic: always at least one
        high_conf_indices = torch.where(initial_confidence > self.threshold)[0]
        
        if len(high_conf_indices) == 0:
            max_prob_idx = initial_confidence.argmax().item()
            accepted_relative_indices = [max_prob_idx]
        else:
            max_prob_idx = initial_confidence.argmax().item()
            accepted_relative_indices = torch.unique(
                torch.cat([high_conf_indices, torch.tensor([max_prob_idx], device=high_conf_indices.device)])
            ).cpu().tolist()
        
        accepted_local_positions = []
        accepted_tokens = []
        
        for rel_idx in accepted_relative_indices:
            local_pos = mask_positions[rel_idx]
            token_id = sampled_tokens[rel_idx].item()
            active_block.confirm_token(local_pos, token_id)
            accepted_local_positions.append(local_pos)
            accepted_tokens.append(token_id)
        
        return FastdLLMV2SampleOutput(
            accepted_local_positions=accepted_local_positions,
            accepted_tokens=accepted_tokens,
            block_confirmed=active_block.is_confirmed,
        )
    
    def reset(self) -> None:
        self.shift_sampler.reset_cache()
