"""Logits shifting strategies aligned with diffulex sampler.

Provides ShiftLogitsSampler (for FastdLLM V2, Dream, SDAR) and
NoShiftLogitsSampler (for LLaDA).
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional


class ShiftLogitsBase(ABC):
    """Abstract base class for logits shifting strategies."""
    
    @abstractmethod
    def shift(
        self,
        logits: torch.Tensor,
        last_logits: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Shift logits according to model-specific strategy.
        
        Args:
            logits: Input logits [seq_len, vocab_size]
            last_logits: Logits from last position [vocab_size] or None
            use_cache: Whether to cache last_logits for next iteration
            
        Returns:
            Shifted logits [seq_len, vocab_size]
        """
        pass
    
    @abstractmethod
    def fetch_last_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fetch logits for the last position.
        
        Args:
            logits: Input logits [seq_len, vocab_size]
            
        Returns:
            Logits for last position [vocab_size]
        """
        pass
    
    @abstractmethod
    def reset_cache(self) -> None:
        """Reset any cached state."""
        pass


class ShiftLogitsSampler(ShiftLogitsBase):
    """
    Align with: diffulex.sampler.base.SamplerShiftLogits
    
    Shifts logits right by one position for autoregressive dependency modeling.
    This is used by FastdLLM V2, Dream, and SDAR models.
    
    The shifting operation:
    - Position 0 gets last_logits (or 1.0 if not provided)
    - Position i gets logits from position i-1
    
    This creates a dependency where each position predicts the NEXT token,
    which is essential for diffusion models with left-to-right dependency.
    
    Example:
        >>> sampler = ShiftLogitsSampler()
        >>> logits = torch.randn(10, 1000)
        >>> shifted = sampler.shift(logits, use_cache=True)
        >>> # shifted[0] = last_logits (or logits[-1] from cache)
        >>> # shifted[1:] = logits[:-1]
    """
    
    def __init__(self):
        self._cached_last_logits: Optional[torch.Tensor] = None
    
    def shift(
        self,
        logits: torch.Tensor,
        last_logits: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Shift logits right by one position.
        
        Align with: SamplerShiftLogits._shift_logits
        
        Args:
            logits: [seq_len, vocab_size]
            last_logits: [vocab_size] or None. If None and use_cache=True, uses cached value
            use_cache: Whether to cache the last logit for next call
            
        Returns:
            Shifted logits [seq_len, vocab_size]
            
        Raises:
            ValueError: If logits has invalid shape
        """
        if logits.dim() != 2:
            raise ValueError(
                f"Expected 2D logits [seq_len, vocab_size], got {logits.shape}"
            )
        
        seq_len, vocab_size = logits.shape
        
        if seq_len == 0:
            raise ValueError("Logits sequence length is 0")
        
        # Determine last_logits to use (aligned with original _fetch_last_logits logic)
        if last_logits is None:
            if use_cache and self._cached_last_logits is not None:
                last_logits = self._cached_last_logits
            else:
                # Use last logit from current batch (fallback behavior)
                last_logits = logits[-1]
        
        # Cache for next iteration if requested
        if use_cache:
            self._cached_last_logits = last_logits.clone()
        
        # Perform shifting (aligned with original _shift_logits)
        shifted = torch.zeros_like(logits)
        
        if last_logits is not None:
            shifted[0] = last_logits
        else:
            # Fallback: use 1.0 (aligned with original)
            shifted[0] = 1.0
        
        if seq_len > 1:
            shifted[1:] = logits[:-1]
        
        return shifted
    
    def fetch_last_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fetch logits for the last position.
        
        Align with: SamplerShiftLogits._fetch_last_logits
        
        Args:
            logits: [seq_len, vocab_size]
            
        Returns:
            Logits for last position [vocab_size]
        """
        if logits.shape[0] == 0:
            raise ValueError("Cannot fetch last logits from empty tensor")
        return logits[-1]
    
    def reset_cache(self) -> None:
        """Reset cached last logits."""
        self._cached_last_logits = None


class NoShiftLogitsSampler(ShiftLogitsBase):
    """
    Align with: diffulex.sampler.base.SamplerNoShiftLogits
    
    No logits shifting - returns logits unchanged.
    This is used by LLaDA model.
    
    LLaDA uses a different diffusion approach that doesn't require
    left-to-right dependency modeling, so no shifting is needed.
    
    Example:
        >>> sampler = NoShiftLogitsSampler()
        >>> logits = torch.randn(10, 1000)
        >>> output = sampler.shift(logits)
        >>> # output is identical to logits
    """
    
    def shift(
        self,
        logits: torch.Tensor,
        last_logits: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Return logits unchanged.
        
        Args:
            logits: Input logits [seq_len, vocab_size]
            last_logits: Ignored (for API compatibility)
            use_cache: Ignored (no caching needed)
            
        Returns:
            Unchanged logits [seq_len, vocab_size]
        """
        return logits
    
    def fetch_last_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return last logits (for API compatibility).
        
        Args:
            logits: [seq_len, vocab_size]
            
        Returns:
            Logits for last position [vocab_size] or None
        """
        if logits.shape[0] == 0:
            return None
        return logits[-1]
    
    def reset_cache(self) -> None:
        """No-op (no caching)."""
        pass