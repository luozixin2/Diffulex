"""FastdLLM V2 sampler aligned with diffulex implementation."""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from diffulex_edge.runtime.sampler.base import sample_tokens
from diffulex_edge.runtime.sampler.shift import ShiftLogitsSampler
from diffulex_edge.runtime.block import DiffusionBlockManager


@dataclass
class FastdLLMV2SampleOutput:
    """
    Align with: diffulex.sampler.fast_dllm_v2.FastdLLMV2SampleOutputForDiffusionLM
    
    Attributes:
        true_local_ids_map: Map of accepted local indices per sequence and block
        accepted_ids_map: Map of accepted indices per sequence and block
        sampled_tokens_map: Map of all sampled tokens per sequence and block
    """
    true_local_ids_map: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    accepted_ids_map: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    sampled_tokens_map: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)


class FastdLLMV2Sampler:
    """
    Align with: diffulex.sampler.fast_dllm_v2.FastdLLMV2SamplerForDiffusionLM
    
    Key characteristics:
    1. Uses ShiftLogitsSampler for logits shifting (autoregressive dependency)
    2. Uses global threshold parameter for token acceptance
    3. Always accepts at least the highest confidence token
    
    This sampler implements the FastdLLM V2 diffusion sampling strategy:
    - Shifts logits right by one position to model left-to-right dependencies
    - Samples tokens from mask positions in each block
    - Accepts tokens with confidence > threshold
    - If no tokens meet threshold, accepts the highest confidence one
    
    Example:
        >>> sampler = FastdLLMV2Sampler(threshold=0.95, temperature=0.8)
        >>> block_manager = DiffusionBlockManager()
        >>> block_manager.create_block(start_pos=10, length=5)
        >>> logits = torch.randn(20, 1000)  # Model output
        >>> output = sampler.sample(block_manager, logits)
        >>> # output.accepted_ids_map contains accepted token indices
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
        """
        Initialize FastdLLM V2 sampler.
        
        Args:
            mask_token_id: Token ID used for masking
            threshold: Confidence threshold for accepting tokens (0-1)
            temperature: Sampling temperature (0=greedy, >0=stochastic)
            top_k: Top-k sampling parameter (0=disabled)
            top_p: Top-p (nucleus) sampling parameter (1.0=disabled)
            margin_confidence: Use (top1_prob - top2_prob) as confidence
            neg_entropy: Use negative entropy as confidence metric
        """
        self.mask_token_id = mask_token_id
        self.threshold = threshold
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.margin_confidence = margin_confidence
        self.neg_entropy = neg_entropy
        
        # FastdLLM V2 uses shifted logits
        self.shift_sampler = ShiftLogitsSampler()
    
    def sample(
        self,
        block_manager: DiffusionBlockManager,
        logits: torch.Tensor,
    ) -> FastdLLMV2SampleOutput:
        """
        Sample tokens for all active blocks.
        
        Align with: FastdLLMV2SamplerForDiffusionLM.forward
        
        Args:
            block_manager: Manager containing active diffusion blocks
            logits: Model output logits [seq_len, vocab_size]
            
        Returns:
            FastdLLMV2SampleOutput with acceptance information
        """
        # Fetch and shift logits (aligned with original)
        last_logits = self.shift_sampler.fetch_last_logits(logits)
        shifted_logits = self.shift_sampler.shift(logits, last_logits)
        
        # Initialize output maps
        true_local_ids_map: Dict[str, Dict[str, List[int]]] = {}
        accepted_ids_map: Dict[str, Dict[str, List[int]]] = {}
        sampled_tokens_map: Dict[str, Dict[str, List[int]]] = {}
        
        # Process each sequence (simplified: single sequence)
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
            
            # Extract logits for mask positions (using shifted logits)
            mask_token_logits = shifted_logits[mask_positions]
            
            # Sample tokens (aligned with original)
            confidence, sampled_tokens, initial_confidence = sample_tokens(
                mask_token_logits,
                temperature=self.temperature,
                top_p=self.top_p if self.top_p < 1.0 else None,
                top_k=self.top_k if self.top_k > 0 else None,
                margin_confidence=self.margin_confidence,
                neg_entropy=self.neg_entropy,
            )
            
            # Determine which tokens to accept (aligned with original)
            # FastdLLM V2: Accept tokens with confidence > threshold
            high_conf_indices = torch.where(initial_confidence > self.threshold)[0]
            
            # Always accept at least the highest confidence token
            if len(high_conf_indices) == 0:
                max_prob_idx = initial_confidence.argmax().view(1)
                accepted_ids = max_prob_idx
            else:
                max_prob_idx = initial_confidence.argmax().view(1)
                accepted_ids = torch.unique(torch.cat([high_conf_indices, max_prob_idx]))
            
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
        
        return FastdLLMV2SampleOutput(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map,
        )
    
    def reset(self) -> None:
        """Reset sampler state."""
        self.shift_sampler.reset_cache()