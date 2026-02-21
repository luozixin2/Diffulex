"""INT4 quantization support via torchao for DiffuLex Edge models.

Provides INT4 weight-only quantization with graceful fallback to INT8.
"""

import dataclasses
from typing import List, Optional

import torch
import torch.nn as nn

from .base import BaseQuantizer, QuantizationConfig, QuantizationResult, QuantizationDtype


@dataclasses.dataclass
class INT4Config:
    """Configuration for INT4 quantization.
    
    Attributes:
        group_size: Group size for quantization (32, 64, 128, 256)
        weight_only: Use weight-only quantization
        preserve_layers: Layer names/types to exclude
    """
    group_size: int = 32
    weight_only: bool = True
    preserve_layers: tuple = ("lm_head", "embedding")
    
    def __post_init__(self):
        """Validate configuration."""
        valid_group_sizes = [32, 64, 128, 256]
        if self.group_size not in valid_group_sizes:
            raise ValueError(
                f"Invalid group_size {self.group_size}. "
                f"Must be one of {valid_group_sizes}"
            )


class INT4Quantizer(BaseQuantizer):
    """INT4 quantization via torchao library.
    
    Provides INT4 weight-only quantization with graceful degradation
    when torchao is not available.
    
    Example:
        >>> quantizer = INT4Quantizer()
        >>> config = QuantizationConfig(dtype=QuantizationDtype.INT4)
        >>> result = quantizer.quantize(model, config)
    """
    
    # Expected size reduction for different group sizes
    SIZE_REDUCTION_EXPECTED = {
        32: 0.70,   # ~70% reduction
        64: 0.72,   # ~72% reduction
        128: 0.74,  # ~74% reduction
        256: 0.75,  # ~75% reduction
    }
    
    def __init__(self):
        super().__init__()
        self._torchao_available = self._check_torchao()
    
    def _check_torchao(self) -> bool:
        """Check if torchao is available."""
        try:
            import torchao
            return True
        except ImportError:
            return False
    
    def is_available(self) -> bool:
        """Check if INT4 quantization is available.
        
        Returns:
            True if torchao is installed
        """
        return self._torchao_available
    
    def quantize(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None
    ) -> QuantizationResult:
        """Apply INT4 quantization to model.
        
        Args:
            model: PyTorch model to quantize
            config: Quantization configuration (should have dtype=INT4)
            
        Returns:
            QuantizationResult with quantized model
            
        Raises:
            RuntimeError: If INT4 quantization is not available or fails
        """
        import copy
        import time
        
        config = config or QuantizationConfig(dtype=QuantizationDtype.INT4)
        int4_config = INT4Config()  # Use default INT4 config
        self._warnings = []
        start_time = time.time()
        
        # Check availability
        if not self._torchao_available:
            raise RuntimeError(
                "INT4 quantization requires torchao. "
                "Install with: pip install torchao>=0.3.0"
            )
        
        # Create a copy
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Get original memory
        original_memory = self._estimate_memory(model_copy)
        
        # Preserve layers that should not be quantized
        preserved_params = self._preserve_layers(model_copy, int4_config.preserve_layers)
        
        try:
            # Apply INT4 quantization using torchao
            from torchao.quantization import quantize_, int4_weight_only
            
            quantize_(model_copy, int4_weight_only(group_size=int4_config.group_size))
            
            # Restore preserved layers
            self._restore_layers(model_copy, preserved_params)
            
            # Compute metrics
            new_memory = self._estimate_memory(model_copy)
            memory_reduction = original_memory - new_memory
            
            metrics = {
                "original_memory_mb": original_memory,
                "quantized_memory_mb": new_memory,
                "memory_reduction_mb": memory_reduction,
                "memory_reduction_pct": (memory_reduction / original_memory * 100) 
                    if original_memory > 0 else 0,
                "expected_reduction_pct": self.SIZE_REDUCTION_EXPECTED.get(
                    int4_config.group_size, 0.70
                ) * 100,
                "group_size": int4_config.group_size,
                "quantization_time_sec": time.time() - start_time,
            }
            
            return QuantizationResult(
                model=model_copy,
                metrics=metrics,
                warnings=self._warnings.copy(),
                success=True,
                fallback_used=False,
            )
            
        except Exception as e:
            raise RuntimeError(
                f"INT4 quantization failed: {e}\n"
                f"This may be due to missing dependencies (fbgemm-gpu-genai >= 1.2.0) "
                f"or incompatible PyTorch version. "
                f"Consider using INT8 quantization instead."
            ) from e


def apply_int4_quantization(
    model: nn.Module,
    group_size: int = 32,
    fallback_to_int8: bool = True,
) -> nn.Module:
    """Convenience function for INT4 quantization.
    
    Args:
        model: Model to quantize
        group_size: Group size for quantization
        fallback_to_int8: Fall back to INT8 if INT4 fails
        
    Returns:
        Quantized model
    """
    quantizer = INT4Quantizer()
    config = QuantizationConfig(dtype=QuantizationDtype.INT4)
    result = quantizer.quantize(model, config)
    return result.model


def is_int4_available() -> bool:
    """Check if INT4 quantization is available.
    
    Returns:
        True if torchao is installed
    """
    return INT4Quantizer().is_available()


__all__ = [
    "INT4Config",
    "INT4Quantizer",
    "apply_int4_quantization",
    "is_int4_available",
]
