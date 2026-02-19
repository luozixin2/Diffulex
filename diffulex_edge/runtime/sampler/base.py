"""Base sampling functions aligned with diffulex sampler.base.

This module provides the core sampling primitives that are numerically
equivalent to diffulex.sampler.base.SamplerBase methods.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def top_p_logits(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    """
    Align with: diffulex.sampler.base.SamplerBase.top_p_logits
    
    Apply top-p (nucleus) filtering to logits.
    
    Args:
        logits: Logits tensor [batch, vocab_size] or [vocab_size]
        top_p: Nucleus sampling threshold (None or >=1.0 to disable)
        
    Returns:
        Filtered logits with same shape as input
        
    Example:
        >>> logits = torch.tensor([[5.0, 4.0, 1.0, 0.5]])
        >>> filtered = top_p_logits(logits, top_p=0.9)
        >>> # Top 2 tokens remain, others are masked to -inf
    """
    if top_p is None or top_p >= 1.0:
        return logits
    
    # Clone to avoid modifying input (aligned with original behavior)
    logits = logits.clone()
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability > top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift to keep at least the first token (aligned with original)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Scatter back to original indices (aligned with original implementation)
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    """
    Align with: diffulex.sampler.base.SamplerBase.top_k_logits
    
    Apply top-k filtering to logits.
    
    Args:
        logits: Logits tensor [batch, vocab_size] or [vocab_size]
        top_k: Number of top tokens to keep (None or <=0 to disable)
        
    Returns:
        Filtered logits with same shape as input
        
    Example:
        >>> logits = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
        >>> filtered = top_k_logits(logits, top_k=2)
        >>> # Top 2 tokens remain, others are masked
    """
    if top_k is None or top_k <= 0:
        return logits
    
    # Clone to avoid modifying input
    logits = logits.clone()
    
    # Ensure k doesn't exceed vocab size (aligned with original)
    k = min(top_k, logits.size(-1))
    
    # Find k-th largest logit and mask everything below it
    indices_to_remove = logits < torch.topk(logits, k, dim=-1)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align with: diffulex.sampler.base.SamplerBase.sample_tokens
    
    Sample tokens from logits with optional temperature, top-p, and top-k filtering.
    Also supports margin confidence and negative entropy confidence metrics.
    
    Args:
        logits: Logits tensor [batch, vocab_size]
        temperature: Sampling temperature (0 for greedy, >0 for stochastic)
        top_p: Nucleus sampling threshold (None to disable)
        top_k: Top-k sampling threshold (None to disable)
        margin_confidence: If True, use (top1_prob - top2_prob) as confidence
        neg_entropy: If True, use sum(p * log(p)) as confidence (negative for peaked dists)
        
    Returns:
        Tuple of (confidence, sampled_tokens, initial_confidence):
        - confidence: [batch] confidence scores (may be modified by margin/entropy)
        - sampled_tokens: [batch] sampled token IDs
        - initial_confidence: [batch] original probability of sampled tokens
        
    Note:
        When neg_entropy=True, confidence is sum(p * log(p)) which is NEGATIVE
        for peaked distributions (since log(p) < 0). This aligns with the original
        diffulex implementation.
        
    Example:
        >>> logits = torch.randn(2, 1000)
        >>> conf, tokens, init_conf = sample_tokens(logits, temperature=0.7, top_p=0.9)
        >>> # tokens: [2] sampled token IDs
        >>> # conf: [2] confidence scores
        >>> # init_conf: [2] original probabilities
    """
    batch_size, vocab_size = logits.shape
    device = logits.device
    
    # Apply temperature scaling
    if temperature > 0:
        logits = logits / temperature
    
    # Apply top-p filtering
    if top_p is not None and top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    
    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        logits = top_k_logits(logits, top_k)
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample or take argmax (aligned with original)
    if temperature > 0:
        try:
            # Use multinomial sampling (equivalent to Categorical.sample())
            sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            initial_confidence = probs.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            # Fallback to argmax if multinomial fails (aligned with original try/except)
            initial_confidence, sampled_tokens = probs.max(dim=-1)
    else:
        # Greedy decoding when temperature = 0
        initial_confidence, sampled_tokens = probs.max(dim=-1)
    
    # Start with initial confidence
    confidence = initial_confidence.clone()
    
    # Apply margin confidence: confidence = top1_prob - top2_prob
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs
    
    # Apply negative entropy: confidence = sum(p * log(p))
    # Note: This returns NEGATIVE values for peaked distributions (log(p) < 0)
    # which is the same behavior as the original diffulex implementation
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        # sum(p * log(p)) is the expected log probability
        # For a peaked distribution, this is close to log(1) = 0 (less negative)
        # For a uniform distribution, this is more negative
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, sampled_tokens, initial_confidence