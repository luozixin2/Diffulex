"""Base classes for DiffuLex Edge models.

Provides abstract interfaces that all models must implement,
enabling backend-agnostic export and inference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Base configuration for all DiffuLex Edge models.
    
    This serves as the common configuration interface across all models.
    Individual models can extend this with model-specific parameters.
    """
    # Core model dimensions
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # For GQA
    intermediate_size: int = 5504
    
    # Sequence and position
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Attention
    attention_bias: bool = False
    head_dim: Optional[int] = None
    
    # Embeddings
    tie_word_embeddings: bool = False
    
    def __post_init__(self):
        """Validate and set derived values."""
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Ensure num_key_value_heads is valid for GQA
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class DiffusionModel(nn.Module, ABC):
    """Abstract base class for all DiffuLex Edge models.
    
    This interface ensures all models can be:
    1. Loaded and used for inference
    2. Exported to ExecuTorch format
    3. Integrated with the runtime engine
    
    All concrete model implementations must inherit from this class
    and implement the abstract methods.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass for inference.
        
        This is the standard inference interface used by DiffusionEngine.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            positions: Position indices [batch_size, seq_len]
            **kwargs: Model-specific arguments (e.g., kv_cache, mask)
            
        Returns:
            Tuple containing:
            - logits: [batch_size, seq_len, vocab_size]
            - Additional model-specific outputs (e.g., kv_cache)
        """
        pass
    
    @property
    def supports_export(self) -> bool:
        """Whether this model supports static-graph export.
        
        Models that support export must implement forward_export().
        
        Returns:
            True if the model can be exported to ExecuTorch
        """
        return hasattr(self, 'forward_export')
    
    def get_export_wrapper(self) -> Optional[nn.Module]:
        """Get a wrapper module for ExecuTorch export.
        
        This method returns a wrapper that:
        1. Exposes forward_export as forward
        2. Handles parameter name sanitization (for ExecuTorch compatibility)
        3. Provides any model-specific export preprocessing
        
        Returns:
            Wrapper module if supports_export is True, None otherwise
        """
        if not self.supports_export:
            return None
        
        # Import here to avoid circular imports
        from .wrapper import ExportWrapper
        return ExportWrapper(self)
    
    def get_export_inputs(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        device: str = "cpu",
    ) -> Tuple[Any, ...]:
        """Create example inputs for export.
        
        Each model must provide appropriate example inputs for torch.export.
        This enables the export script to work without knowing model details.
        
        Args:
            batch_size: Batch size for example inputs
            seq_len: Sequence length for example inputs
            device: Device to create tensors on
            
        Returns:
            Tuple of tensors suitable for torch.export
        """
        # Default implementation for simple models
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size, seq_len), device=device
        )
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        return (input_ids, positions)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information metadata.
        
        Returns:
            Dictionary with model metadata
        """
        num_params = sum(p.numel() for p in self.parameters())
        return {
            "model_type": self.__class__.__name__,
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_hidden_layers,
            "num_heads": self.config.num_attention_heads,
            "num_kv_heads": self.config.num_key_value_heads,
            "head_dim": self.config.head_dim,
            "max_position_embeddings": self.config.max_position_embeddings,
            "num_parameters": num_params,
            "num_parameters_human": f"{num_params / 1e6:.1f}M",
            "supports_export": self.supports_export,
        }


class KVCacheModel(DiffusionModel, ABC):
    """Base class for models that support KV cache.
    
    This extends DiffusionModel with KV cache-specific functionality
    for efficient autoregressive inference.
    """
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional KV cache.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            positions: Position indices [batch_size, seq_len]
            kv_cache: Optional KV cache from previous forward pass
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (logits, updated_kv_cache)
        """
        pass
