"""FP16 quantization support for DiffuLex Edge models.

Provides FP16 precision conversion using the shared quantizer infrastructure.
"""

import time
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseQuantizer, QuantizationConfig, QuantizationResult, QuantizationDtype


class FP16Quantizer(BaseQuantizer):
    """FP16 precision converter.
    
    Converts FP32 tensors to FP16 for reduced memory usage
    and improved inference speed on supported devices.
    
    Example:
        >>> quantizer = FP16Quantizer()
        >>> config = QuantizationConfig(dtype=QuantizationDtype.FP16)
        >>> result = quantizer.quantize(model, config)
        >>> print(f"Memory reduction: {result.metrics['memory_reduction_pct']:.1f}%")
    """
    
    def is_available(self) -> bool:
        """Check if FP16 is available.
        
        FP16 is always available on CPU (though may be slow).
        CUDA requires compute capability >= 5.0.
        
        Returns:
            True if FP16 is supported
        """
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            return capability[0] >= 5  # Maxwell+
        return True  # CPU always supports FP16
    
    def quantize(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None
    ) -> QuantizationResult:
        """Convert model to FP16 precision.
        
        Args:
            model: PyTorch model in FP32
            config: Quantization configuration (dtype should be FP16)
            
        Returns:
            QuantizationResult with FP16 model
        """
        import copy
        
        config = config or QuantizationConfig(dtype=QuantizationDtype.FP16)
        self._warnings = []
        start_time = time.time()
        
        # Create a copy to avoid modifying original
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Get original memory
        original_memory = self._estimate_memory(model_copy)
        
        # Convert to FP16
        fp16_model = self._convert_model(model_copy, config)
        
        # Compute metrics
        metrics = self._compute_metrics(fp16_model, original_memory, start_time)
        metrics["layers_converted"] = len([
            m for m in fp16_model.modules() 
            if hasattr(m, 'weight') and m.weight.dtype == torch.float16
        ])
        
        return QuantizationResult(
            model=fp16_model,
            metrics=metrics,
            warnings=self._warnings.copy(),
        )
    
    def _convert_model(
        self,
        model: nn.Module,
        config: QuantizationConfig
    ) -> nn.Module:
        """Convert model to FP16.
        
        Args:
            model: Model to convert
            config: Quantization config
            
        Returns:
            FP16 model
        """
        # Store excluded layer dtypes
        excluded_modules = {}
        
        for name, module in model.named_modules():
            if not self._should_convert_layer(name, module, config):
                for param_name, param in module.named_parameters(recurse=False):
                    excluded_modules[f"{name}.{param_name}"] = param.dtype
        
        # Convert model to half precision
        model = model.half()
        
        # Restore excluded layers to original dtype
        for name, module in model.named_modules():
            full_name_prefix = f"{name}."
            for param_name, original_dtype in excluded_modules.items():
                if param_name.startswith(full_name_prefix) or param_name == name:
                    param_local_name = param_name[len(full_name_prefix):] if param_name.startswith(full_name_prefix) else ""
                    if hasattr(module, param_local_name):
                        param = getattr(module, param_local_name)
                        if param.dtype != original_dtype:
                            param.data = param.data.to(original_dtype)
        
        return model


def convert_to_fp16(
    model: nn.Module,
    exclude_layers: Optional[tuple] = None,
) -> nn.Module:
    """Convenience function to convert model to FP16.
    
    Args:
        model: Model to convert
        exclude_layers: Additional layer names to exclude
        
    Returns:
        FP16 converted model
    """
    config = QuantizationConfig(dtype=QuantizationDtype.FP16)
    if exclude_layers:
        config.exclude_layers = config.exclude_layers + exclude_layers
    
    quantizer = FP16Quantizer()
    result = quantizer.quantize(model, config)
    return result.model


def is_fp16_available() -> bool:
    """Check if FP16 is available on current device.
    
    Returns:
        True if FP16 computation is supported
    """
    return FP16Quantizer().is_available()


__all__ = [
    "FP16Quantizer",
    "convert_to_fp16",
    "is_fp16_available",
]
