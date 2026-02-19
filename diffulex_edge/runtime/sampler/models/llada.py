"""LLaDA sampler aligned with diffulex implementation."""

import torch
from typing import Dict, List
from dataclasses import dataclass, field

from diffulex_edge.runtime.sampler.base import sample_tokens
from diffulex_edge.runtime.sampler.shift import NoShiftLogitsSampler
from diffulex_edge.runtime.block import DiffusionBlockManager


@dataclass
class LLaDASampleOutput:
    """
    Align with: diffulex.sampler.llada.LLaDASampleOutputForDiffusionLM
    
    Attributes:
        true_local_ids_map: Map of accepted local indices per sequence and block
        accepted_ids_map: Map of accepted indices per sequence and block
        sampled_tokens_map: Map of all sampled tokens per sequence and block
    """
    true_local_ids_map: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    accepted_ids_map: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    sampled_tokens_map: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)


class LLaDASampler:
    """
    Align with: diffulex.sampler.llada.LLaDASamplerForDiffusionLM
    
    Key characteristics:
    1. Uses NoShiftLogitsSampler (NO logits shifting!)
    2. Per-block accept_threshold (not global)
    3. pre_block_complete logic for conditional acceptance
    
    This is the critical difference from FastdLLM V2:
    - LLaDA does NOT shift logits (uses SamplerNoShiftLogits)
    - LLaDA uses per-block thresholds
    - LLaDA has special handling when pre_block_complete=True
    
    The pre_block_complete logic:
    - If previous block is complete and current block has no high-confidence tokens:
      Transfer 1 token from highest confidence to ensure progress
    - Otherwise: Accept only high-confidence tokens
    
    Example:
        >>> sampler = LLaDASampler(temperature=0.8)
        >>> block_manager = DiffusionBlockManager()
        >>> # Create block with per-block threshold
        >>> block_manager.create_block(start_pos=10, length=5, accept_threshold=0.9)
        >>> logits = torch.randn(20, 1000)  # Model output
        >>> output = sampler.sample(block_manager, logits)
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
        """
        Initialize LLaDA sampler.
        
        Args:
            mask_token_id: Token ID used for masking
            temperature: Sampling temperature (0=greedy, >0=stochastic)
            top_k: Top-k sampling parameter (0=disabled)
            top_p: Top-p (nucleus) sampling parameter (1.0=disabled)
            margin_confidence: Use (top1_prob - top2_prob) as confidence
            neg_entropy: Use negative entropy as confidence metric
        """
        self.mask_token_id = mask_token_id
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.margin_confidence = margin_confidence
        self.neg_entropy = neg_entropy
        
        # CRITICAL: LLaDA does NOT use shifted logits
        self.shift_sampler = NoShiftLogitsSampler()
    
    def sample(
        self,
        block_manager: DiffusionBlockManager,
        logits: torch.Tensor,
    ) -> LLaDASampleOutput:
        """
        Sample tokens for all active blocks.
        
        Align with: LLaDASamplerForDiffusionLM.forward
        
        Args:
            block_manager: Manager containing active diffusion blocks
            logits: Model output logits [seq_len, vocab_size]
            
        Returns:
            LLaDASampleOutput with acceptance information
        """
        # CRITICAL: No logits shifting for LLaDA
        seq_logits = logits
        
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
            
            # Get mask positions for this block
            mask_positions = block.global_mask_token_ids
            if not mask_positions:
                continue
            
            # CRITICAL: Use UNshifted logits for LLaDA
            mask_token_logits = seq_logits[mask_positions]
            
            # Sample tokens
            confidence, sampled_tokens, initial_confidence = sample_tokens(
                mask_token_logits,
                temperature=self.temperature,
                top_p=self.top_p if self.top_p < 1.0 else None,
                top_k=self.top_k if self.top_k > 0 else None,
                margin_confidence=self.margin_confidence,
                neg_entropy=self.neg_entropy,
            )
            
            # CRITICAL: LLaDA-specific acceptance logic with pre_block_complete
            # Get per-block threshold (not global)
            block_threshold = getattr(block, 'accept_threshold', 0.95)
            pre_block_complete = getattr(block, 'pre_block_complete', False)
            
            if pre_block_complete:
                # Previous block is complete - special handling
                high_conf_indices = torch.where(
                    initial_confidence > block_threshold
                )[0]
                
                if len(high_conf_indices) == 0:
                    # No high-confidence tokens - transfer 1 token
                    number_transfer_tokens = 1
                    _, transfer_index = torch.topk(
                        confidence, number_transfer_tokens
                    )
                else:
                    # Has high-confidence tokens - no transfer needed
                    transfer_index = torch.tensor(
                        [], device=sampled_tokens.device, dtype=torch.long
                    )
                
                # Combine transfer and high-confidence indices
                accepted_ids = torch.unique(
                    torch.cat([transfer_index, high_conf_indices])
                )
            else:
                # Normal acceptance - only high-confidence tokens
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
            
            # Update block with accepted tokens
            for idx in accepted_ids_list:
                global_pos = mask_positions[idx]
                local_pos = global_pos - block.start_pos
                token_id = sampled_tokens[idx].item()
                block.accept_token(local_pos, token_id)
        
        # Store results for this sequence
        true_local_ids_map[seq_idx] = true_local_ids_sub_map
        accepted_ids_map[seq_idx] = accepted_ids_sub_map
        sampled_tokens_map[seq_idx] = sampled_tokens_sub_map
        
        return LLaDASampleOutput(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map,
        )
    
    def reset(self) -> None:
        """Reset sampler state (no-op for LLaDA)."""
        pass