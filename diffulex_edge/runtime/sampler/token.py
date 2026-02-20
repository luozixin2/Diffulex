"""Sampling strategies for text generation.

Provides various sampling methods from greedy decoding to
nucleus sampling.
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, List
import random


class Sampler(ABC):
    """Base class for sampling strategies."""
    
    @abstractmethod
    def sample(self, logits: torch.Tensor) -> int:
        """Sample a token from logits.
        
        Args:
            logits: Logits tensor of shape [vocab_size] or [batch, vocab_size]
            
        Returns:
            Sampled token index
        """
        pass
    
    def reset(self):
        """Reset sampler state (if any)."""
        pass


class GreedySampler(Sampler):
    """Greedy decoding - always pick highest probability token."""
    
    def sample(self, logits: torch.Tensor) -> int:
        """Sample using greedy strategy."""
        # Handle batched input
        if logits.dim() == 2:
            logits = logits[0]
        
        return int(torch.argmax(logits).item())


class TopKSampler(Sampler):
    """Top-k sampling - sample from top k most likely tokens.
    
    Reduces noise from unlikely tokens while maintaining diversity.
    """
    
    def __init__(self, k: int = 50, temperature: float = 1.0, seed: Optional[int] = None):
        """Initialize Top-K sampler.
        
        Args:
            k: Number of top tokens to consider
            temperature: Softmax temperature (lower = more deterministic)
            seed: Random seed for reproducibility
        """
        self.k = k
        self.temperature = temperature
        self._rng = random.Random(seed)
    
    def sample(self, logits: torch.Tensor) -> int:
        """Sample using top-k strategy."""
        # Handle batched input
        if logits.dim() == 2:
            logits = logits[0]
        
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Get top-k
        top_k_logits, top_k_indices = torch.topk(logits, min(self.k, logits.size(-1)))
        
        # Sample from top-k
        probs = F.softmax(top_k_logits, dim=-1)
        
        # Use torch multinomial for proper sampling
        sample_idx = torch.multinomial(probs, num_samples=1).item()
        
        return int(top_k_indices[sample_idx].item())
    
    def reset(self):
        """Reset random state."""
        self._rng = random.Random(self._rng.seed)


class TopPSampler(Sampler):
    """Nucleus (top-p) sampling - sample from smallest set of tokens
    whose cumulative probability exceeds p.
    
    Dynamically adjusts the candidate pool based on probability distribution.
    """
    
    def __init__(
        self,
        p: float = 0.9,
        temperature: float = 1.0,
        min_tokens_to_keep: int = 1,
        seed: Optional[int] = None,
    ):
        """Initialize Top-P sampler.
        
        Args:
            p: Cumulative probability threshold
            temperature: Softmax temperature
            min_tokens_to_keep: Minimum tokens to consider (safety)
            seed: Random seed
        """
        self.p = p
        self.temperature = temperature
        self.min_tokens_to_keep = min_tokens_to_keep
        self._rng = random.Random(seed)
    
    def sample(self, logits: torch.Tensor) -> int:
        """Sample using top-p strategy."""
        # Handle batched input
        if logits.dim() == 2:
            logits = logits[0]
        
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff index
        sorted_indices_to_remove = cumsum_probs > self.p
        
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :self.min_tokens_to_keep] = False
        
        # Create filtered distribution
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(filtered_logits, dim=-1)
        sample_idx = torch.multinomial(probs, num_samples=1).item()
        
        return int(sample_idx)
    
    def reset(self):
        """Reset random state."""
        self._rng = random.Random(self._rng.seed)


class TemperatureSampler(Sampler):
    """Simple temperature sampling without top-k or top-p filtering."""
    
    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None):
        """Initialize temperature sampler.
        
        Args:
            temperature: Softmax temperature
            seed: Random seed
        """
        self.temperature = temperature
        self._rng = random.Random(seed)
    
    def sample(self, logits: torch.Tensor) -> int:
        """Sample using temperature strategy."""
        # Handle batched input
        if logits.dim() == 2:
            logits = logits[0]
        
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        sample_idx = torch.multinomial(probs, num_samples=1).item()
        
        return int(sample_idx)


def get_sampler(
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: Optional[int] = None,
) -> Sampler:
    """Factory function to create samplers.
    
    Args:
        strategy: Sampling strategy ("greedy", "top_k", "top_p", "temperature")
        temperature: Softmax temperature
        top_k: K for top-k sampling
        top_p: P for top-p sampling
        seed: Random seed
        
    Returns:
        Configured sampler
    """
    strategy = strategy.lower()
    
    if strategy == "greedy":
        return GreedySampler()
    elif strategy == "top_k":
        return TopKSampler(k=top_k, temperature=temperature, seed=seed)
    elif strategy == "top_p":
        return TopPSampler(p=top_p, temperature=temperature, seed=seed)
    elif strategy == "temperature":
        return TemperatureSampler(temperature=temperature, seed=seed)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
