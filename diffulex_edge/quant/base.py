"""Base classes and utilities for quantization.

Provides shared infrastructure for all quantization types.
"""

import dataclasses
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class QuantizationDtype(Enum):
    """Supported quantization data types."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class QuantizationMode(Enum):
    """Quantization mode."""
    DYNAMIC = auto()     # Dynamic quantization (weights only, runtime activation quant)
    STATIC = auto()      # Static quantization (weights + activations with calibration)
    WEIGHT_ONLY = auto() # Weight-only quantization
    QAT = auto()         # Quantization Aware Training


@dataclasses.dataclass
class QuantizationConfig:
    """Unified configuration for quantization.
    
    Attributes:
        dtype: Target quantization data type
        mode: Quantization mode
        exclude_layers: Layer names/types to exclude from quantization
        preserve_precision_layers: Layer types that should remain in FP32
        is_per_channel: Use per-channel quantization (backward compatibility)
        is_qat: Quantization Aware Training mode (backward compatibility)
        scheme: Quantization scheme (backward compatibility with quantizer.py)
    """
    dtype: QuantizationDtype = QuantizationDtype.INT8
    mode: QuantizationMode = QuantizationMode.DYNAMIC  # Default to DYNAMIC for backward compatibility
    exclude_layers: Tuple[str, ...] = ("lm_head", "norm", "embedding")
    preserve_precision_layers: Tuple[str, ...] = ("layernorm", "rmsnorm", "softmax")
    is_per_channel: bool = True  # For backward compatibility
    is_qat: bool = False  # For backward compatibility
    scheme: Optional[str] = None  # For backward compatibility with quantizer.py
    calibration_samples: int = 100  # For backward compatibility
    preserve_activation_precision_layers: Tuple[str, ...] = ("lm_head",)  # For backward compatibility
    
    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.dtype, str):
            self.dtype = QuantizationDtype(self.dtype.lower())
        if isinstance(self.mode, str):
            self.mode = QuantizationMode[self.mode.upper()]
        
        # Auto-set is_qat when mode is QAT
        if self.mode == QuantizationMode.QAT and not self.is_qat:
            self.is_qat = True


@dataclasses.dataclass
class QuantizationResult:
    """Result of quantization operation."""
    model: nn.Module
    metrics: Dict[str, Any]
    warnings: List[str] = dataclasses.field(default_factory=list)
    success: bool = True
    fallback_used: bool = False


class BaseQuantizer(ABC):
    """Base class for all quantizers.
    
    Provides shared utilities and defines the quantization interface.
    """
    
    def __init__(self):
        self._warnings: List[str] = []
    
    @abstractmethod
    def quantize(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None
    ) -> QuantizationResult:
        """Apply quantization to model.
        
        Args:
            model: PyTorch model to quantize
            config: Quantization configuration
            
        Returns:
            QuantizationResult with quantized model and metrics
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this quantizer is available.
        
        Returns:
            True if the quantizer can be used
        """
        pass
    
    # =========================================================================
    # Shared utility methods
    # =========================================================================
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters())
    
    def _estimate_memory(self, model: nn.Module) -> float:
        """Estimate model memory usage in MB."""
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_bytes + buffer_bytes) / (1024 * 1024)
    
    def _should_convert_layer(
        self,
        name: str,
        module: nn.Module,
        config: QuantizationConfig
    ) -> bool:
        """Determine if a layer should be converted.
        
        Args:
            name: Layer name
            module: Layer module
            config: Quantization config with exclusion patterns
            
        Returns:
            True if layer should be converted
        """
        # Check excluded layer names
        for exclude_pattern in config.exclude_layers:
            if exclude_pattern.lower() in name.lower():
                return False
        
        # Check layer types to preserve
        module_type = type(module).__name__.lower()
        for preserve_type in config.preserve_precision_layers:
            if preserve_type.lower() in module_type:
                return False
        
        # Only convert layers with float parameters
        if not any(p.dtype == torch.float32 for p in module.parameters(recurse=False)):
            return False
        
        return True
    
    def _preserve_layers(
        self,
        model: nn.Module,
        preserve_patterns: Tuple[str, ...]
    ) -> Dict[str, torch.Tensor]:
        """Save parameters of layers that should not be quantized.
        
        Args:
            model: Model to preserve layers from
            preserve_patterns: Patterns to match layer names
            
        Returns:
            Dictionary of preserved parameters
        """
        preserved = {}
        
        for name, module in model.named_modules():
            should_preserve = any(
                pattern.lower() in name.lower()
                for pattern in preserve_patterns
            )
            
            if should_preserve:
                for param_name, param in module.named_parameters(recurse=False):
                    preserved[f"{name}.{param_name}"] = param.data.clone()
        
        return preserved
    
    def _restore_layers(
        self,
        model: nn.Module,
        preserved_params: Dict[str, torch.Tensor]
    ) -> None:
        """Restore preserved layer parameters.
        
        Args:
            model: Model to restore layers to
            preserved_params: Dictionary of preserved parameters
        """
        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}"
                if full_name in preserved_params:
                    param.data = preserved_params[full_name]
    
    def validate_numerical_accuracy(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        tolerance: float = 1e-3
    ) -> Dict[str, float]:
        """Validate numerical accuracy of quantized model.
        
        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            example_inputs: Example inputs for comparison
            tolerance: Maximum acceptable difference
            
        Returns:
            Dictionary with accuracy metrics
        """
        original_model.eval()
        quantized_model.eval()
        
        with torch.no_grad():
            orig_output = original_model(*example_inputs)
            quant_output = quantized_model(*example_inputs)
            
            # Handle tuple outputs
            if isinstance(orig_output, tuple):
                orig_output = orig_output[0]
            if isinstance(quant_output, tuple):
                quant_output = quant_output[0]
            
            # Convert to same dtype for comparison
            if orig_output.dtype != quant_output.dtype:
                quant_output = quant_output.to(orig_output.dtype)
            
            abs_diff = (orig_output - quant_output).abs()
            rel_diff = abs_diff / (orig_output.abs() + 1e-8)
            
            metrics = {
                "max_absolute_error": abs_diff.max().item(),
                "mean_absolute_error": abs_diff.mean().item(),
                "max_relative_error": rel_diff.max().item(),
                "mean_relative_error": rel_diff.mean().item(),
                "within_tolerance": (abs_diff <= tolerance).all().item(),
            }
        
        return metrics
    
    def _compute_metrics(
        self,
        model: nn.Module,
        original_memory: float,
        start_time: float
    ) -> Dict[str, Any]:
        """Compute standard quantization metrics.
        
        Args:
            model: Quantized model
            original_memory: Original memory usage in MB
            start_time: Start time for timing
            
        Returns:
            Dictionary with metrics
        """
        new_memory = self._estimate_memory(model)
        memory_reduction = original_memory - new_memory
        
        return {
            "original_memory_mb": original_memory,
            "quantized_memory_mb": new_memory,
            "memory_reduction_mb": memory_reduction,
            "memory_reduction_pct": (memory_reduction / original_memory * 100) 
                if original_memory > 0 else 0,
            "quantization_time_sec": time.time() - start_time,
            "num_parameters": self._count_parameters(model),
        }


def get_model_size_info(model: nn.Module) -> Tuple[float, float, float]:
    """Get model size information.
    
    Returns:
        (param_size_mb, buffer_size_mb, total_size_mb)
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes
    
    return (
        param_bytes / 1024 / 1024,
        buffer_bytes / 1024 / 1024,
        total_bytes / 1024 / 1024,
    )


__all__ = [
    "QuantizationDtype",
    "QuantizationMode", 
    "QuantizationConfig",
    "QuantizationResult",
    "BaseQuantizer",
    "get_model_size_info",
]
