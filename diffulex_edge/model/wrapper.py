"""Export wrappers for DiffuLex Edge models.

This module provides wrapper classes that prepare models for ExecuTorch export.
Wrappers handle:
1. Exposing forward_export as the standard forward
2. Parameter name sanitization (ExecuTorch doesn't allow '.' in names)
3. Input/output format conversion
"""

from typing import Any, Optional
import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """Generic export wrapper for DiffuLex Edge models.
    
    This wrapper prepares any model with forward_export method for ExecuTorch
    export by:
    1. Redirecting forward() to the model's forward_export()
    2. Sanitizing parameter names (replacing '.' with '_')
    
    Args:
        model: The inner model to wrap (must have forward_export method)
        
    Example:
        >>> model = SDAREdge(config)
        >>> wrapper = ExportWrapper(model)
        >>> # Now wrapper.forward calls model.forward_export
        >>> result = wrapper(input_ids, positions, kv_cache, ...)
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.inner = model
        
        # Sanitize parameter names for ExecuTorch compatibility
        # ExecuTorch doesn't allow '.' in parameter names
        self._param_map: dict = {}  # Maps sanitized name -> original name
        self._sanitize_parameters()
    
    def _sanitize_parameters(self):
        """Register all parameters with sanitized names.
        
        This ensures ExecuTorch can handle the parameter names correctly.
        We keep the original parameters in self.inner but also register
        them directly on the wrapper with safe names.
        """
        for name, param in self.inner.named_parameters():
            safe_name = name.replace('.', '_')
            self._param_map[safe_name] = name
            # Register as a parameter so it's included in state_dict
            self.register_parameter(safe_name, param)
    
    def forward(self, *args, **kwargs) -> Any:
        """Call the inner model's forward_export method.
        
        This method is called during ExecuTorch export tracing.
        All arguments are passed through to forward_export.
        
        Args:
            *args: Positional arguments for forward_export
            **kwargs: Keyword arguments for forward_export
            
        Returns:
            Output from forward_export
        """
        if not hasattr(self.inner, 'forward_export'):
            raise RuntimeError(
                f"Model {type(self.inner).__name__} does not have forward_export method. "
                "Only models with forward_export can be wrapped for export."
            )
        return self.inner.forward_export(*args, **kwargs)
    
    @property
    def config(self) -> Any:
        """Access the inner model's config."""
        return self.inner.config
    
    def get_model_info(self) -> dict:
        """Get model information from inner model."""
        if hasattr(self.inner, 'get_model_info'):
            return self.inner.get_model_info()
        return {
            "model_type": type(self.inner).__name__,
            "wrapped": True,
        }
    
    def extra_repr(self) -> str:
        return f"inner_model={type(self.inner).__name__}, " \
               f"num_params={len(self._param_map)}"


class BlockDiffusionWrapper(ExportWrapper):
    """Specialized wrapper for Block Diffusion models (e.g., SDAR).
    
    Block Diffusion models use a specific export format with mask-based
    cache control:
    - attention_mask: Controls which positions can be attended to
    - insert_matrix: Controls cache updates via matrix multiplication
    - keep_mask: Controls which cache entries to keep
    
    This wrapper ensures proper handling of these special inputs.
    
    Args:
        model: The inner Block Diffusion model
        block_size: Size of each diffusion block
        max_seq_len: Maximum sequence length for KV cache
    """
    
    def __init__(
        self,
        model: nn.Module,
        block_size: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__(model)
        self.block_size = block_size
        self.max_seq_len = max_seq_len
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        insert_matrix: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> tuple:
        """Forward for Block Diffusion models.
        
        Args:
            input_ids: [batch, block_size]
            positions: [batch, block_size]
            kv_cache: [num_layers, 2, batch, num_kv_heads, max_len, head_dim]
            attention_mask: [num_layers, batch, 1, block_size, max_len + block_size]
            insert_matrix: [num_layers, batch, 1, max_len, block_size]
            keep_mask: [num_layers, batch, 1, max_len, 1]
            
        Returns:
            Tuple of (logits, updated_kv_cache)
        """
        # Delegate to inner model's forward_export
        return self.inner.forward_export(
            input_ids, positions, kv_cache,
            attention_mask, insert_matrix, keep_mask
        )
    
    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, block_size={self.block_size}, " \
               f"max_seq_len={self.max_seq_len}"


def get_export_wrapper(model: nn.Module) -> Optional[nn.Module]:
    """Get the appropriate export wrapper for a model.
    
    This factory function automatically selects the right wrapper
    based on the model's capabilities.
    
    Args:
        model: The model to wrap
        
    Returns:
        Appropriate wrapper if model supports export, None otherwise
    """
    # Check if model already has a get_export_wrapper method
    if hasattr(model, 'get_export_wrapper'):
        return model.get_export_wrapper()
    
    # Check if model has forward_export
    if hasattr(model, 'forward_export'):
        # Check if it's a Block Diffusion model
        if hasattr(model.config, 'diffusion_block_size'):
            return BlockDiffusionWrapper(
                model,
                block_size=getattr(model.config, 'diffusion_block_size', 4),
                max_seq_len=getattr(model.config, 'max_position_embeddings', 2048),
            )
        return ExportWrapper(model)
    
    # Model doesn't support export
    return None
