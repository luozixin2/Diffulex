"""Dream sampler aligned with diffulex implementation."""

import torch
from typing import Dict, List
from dataclasses import dataclass, field

from diffulex_edge.runtime.sampler.base import sample_tokens
from diffulex_edge.runtime.sampler.shift import ShiftLogitsSampler
from diffulex_edge.runtime.block import DiffusionBlockManager


@dataclass
class DreamSampleOutput:
    """
    Align with: diffulex.sampler.dream.DreamSampleOutputForDiffusionLM
    """
    true_local_ids_map: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    accepted_ids_map: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    sampled_tokens_map: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)


class DreamSampler:
    """
    Align with: diffulex.sampler.dream.DreamSamplerForDiffusionLM
    
    Key characteristics:
    1. Uses ShiftLogitsSampler (like FastdLLM V2)
    2. Per-block accept_threshold (like LLaDA)
    3. pre_block_complete logic (like LLaDA)
    
    Dream combines aspects of both FastdLLM V2 and LLaDA:
    - Uses shifted logits for autoregressive dependency
    - Has per-block thresholds
    - Has pre_block_complete conditional acceptance
    
    The pre_block_complete logic is identical to LLaDA:
    - If previous block is complete and no high-confidence tokens: transfer 1
    - Otherwise: accept only high-confidence tokens
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
        
        # Dream uses shifted logits
        self.shift_sampler = ShiftLogitsSampler()
    
    def sample(
        self,
        block_manager: DiffusionBlockManager,
        logits: torch.Tensor,
    ) -> DreamSampleOutput:
        """
        Sample tokens for all active blocks.
        
        Align with: DreamSamplerForDiffusionLM.forward
        """
        # Fetch and shift logits (like FastdLLM V2)
        last_logits = self.shift_sampler.fetch_last_logits(logits)
        shifted_logits = self.shift_sampler.shift(logits, last_logits)
        
        # Initialize output maps
        true_local_ids_map: Dict[str, Dict[str, List[int]]] = {}
        accepted_ids_map: Dict[str, Dict[str, List[int]]] = {}
        sampled_tokens_map: Dict[str, Dict[str, List[int]]] = {}
        
        seq_idx = "0"
        true_local_ids_sub_map: Dict[str, List[int]] = {}
        accepted_ids_sub_map: Dict[str, List[int]] = {}
        sampled_tokens_sub_map: Dict[str, List[int]] = {}
        
        # Process each active block
        for block_id, block in block_manager.get_active_blocks():
            if not block.is_active or block.is_complete:
                continue
            
            mask_positions = block.global_mask_token_ids
            if not mask_positions:
                continue
            
            # Use shifted logits (Dream uses shift like FastdLLM V2)
            mask_token_logits = shifted_logits[mask_positions]
            
            # Sample tokens
            confidence, sampled_tokens, initial_confidence = sample_tokens(
                mask_token_logits,
                temperature=self.temperature,
                top_p=self.top_p if self.top_p < 1.0 else None,
                top_k=self.top_k if self.top_k > 0 else None,
                margin_confidence=self.margin_confidence,
                neg_entropy=self.neg_entropy,
            )
            
            # Get per-block parameters (like LLaDA)
            block_threshold = getattr(block, 'accept_threshold', 0.95)
            pre_block_complete = getattr(block, 'pre_block_complete', False)
            
            # pre_block_complete logic (like LLaDA)
            if pre_block_complete:
                high_conf_indices = torch.where(
                    initial_confidence > block_threshold
                )[0]
                
                if len(high_conf_indices) == 0:
                    number_transfer_tokens = 1
                    _, transfer_index = torch.topk(
                        confidence, number_transfer_tokens
                    )
                else:
                    transfer_index = torch.tensor(
                        [], device=sampled_tokens.device, dtype=torch.long
                    )
                
                accepted_ids = torch.unique(
                    torch.cat([transfer_index, high_conf_indices])
                )
            else:
                high_conf_indices = torch.where(
                    initial_confidence > block_threshold
                )[0]
                accepted_ids = high_conf_indices
            
            # Record results
            accepted_ids_list = accepted_ids.cpu().tolist()
            true_local_ids_sub_map[str(block_id)] = [
                mask_positions[i] for i in accepted_ids_list
            ]
            accepted_ids_sub_map[str(block_id)] = accepted_ids_list
            sampled_tokens_sub_map[str(block_id)] = sampled_tokens.cpu().tolist()
            
            # Update block
            for idx in accepted_ids_list:
                global_pos = mask_positions[idx]
                local_pos = global_pos - block.start_pos
                token_id = sampled_tokens[idx].item()
                block.accept_token(local_pos, token_id)
        
        true_local_ids_map[seq_idx] = true_local_ids_sub_map
        accepted_ids_map[seq_idx] = accepted_ids_sub_map
        sampled_tokens_map[seq_idx] = sampled_tokens_sub_map
        
        return DreamSampleOutput(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map,
        )
    
    def reset(self) -> None:
        """Reset sampler state."""
        self.shift_sampler.reset_cache()