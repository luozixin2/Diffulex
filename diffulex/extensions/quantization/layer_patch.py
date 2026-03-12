"""
Layer Monkey Patch

Zero-intrusive layer replacement using Monkey Patching.
Replaces original Linear classes with quantization-aware versions.
"""

import torch
import torch.nn as nn
from typing import Optional, Type

from .layer_mixin import LinearQuantizationMixin


# Store original classes
_original_classes: dict = {}


def create_quantized_linear_class(base_class: Type[nn.Module]) -> Type[nn.Module]:
    """
    Create a quantized version of a linear class.
    
    Args:
        base_class: Original linear class (e.g., ColumnParallelLinear)
        
    Returns:
        New class that inherits from both base_class and LinearQuantizationMixin
    """
    
    class QuantizedLinear(LinearQuantizationMixin, base_class):
        """
        Quantized version of linear layer.
        
        Inherits from both the original class and LinearQuantizationMixin.
        """
        
        def __init__(self, *args, quant_kind: str = "other", **kwargs):
            # Initialize base class
            super().__init__(*args, **kwargs)
            
            # Initialize quantization
            self.init_quantization(quant_kind)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward with quantization support."""
            # Get bias if available
            bias = getattr(self, 'bias', None)
            
            # Use unified forward dispatcher
            return self._forward_base(x, bias)
    
    # Set class name
    QuantizedLinear.__name__ = f"Quant{base_class.__name__}"
    QuantizedLinear.__qualname__ = f"Quant{base_class.__name__}"
    
    return QuantizedLinear


def patch_linear_layers():
    """
    Monkey patch linear layer classes with quantization support.
    
    This function replaces the original Linear classes in diffulex.layer.linear
    with quantized versions that support all quantization formats.
    """
    try:
        import diffulex.layer.linear as linear_module
    except ImportError:
        # diffulex not loaded yet, will patch when available
        return False
    
    global _original_classes
    
    # Classes to patch
    classes_to_patch = [
        'ReplicatedLinear',
        'ColumnParallelLinear', 
        'RowParallelLinear',
        'MergedColumnParallelLinear',
        'QKVParallelLinear',
    ]
    
    for class_name in classes_to_patch:
        if hasattr(linear_module, class_name):
            base_class = getattr(linear_module, class_name)
            
            # Skip if already patched
            if class_name in _original_classes:
                continue
            
            # Store original
            _original_classes[class_name] = base_class
            
            # Create quantized version
            quant_class = create_quantized_linear_class(base_class)
            
            # Replace in module
            setattr(linear_module, class_name, quant_class)
    
    return True


def unpatch_linear_layers():
    """
    Restore original linear layer classes.
    
    Reverses the monkey patching done by patch_linear_layers().
    """
    try:
        import diffulex.layer.linear as linear_module
    except ImportError:
        return False
    
    global _original_classes
    
    for class_name, original_class in _original_classes.items():
        setattr(linear_module, class_name, original_class)
    
    _original_classes.clear()
    return True


def is_patched() -> bool:
    """Check if layers are currently patched."""
    return len(_original_classes) > 0


def get_original_class(class_name: str) -> Optional[Type[nn.Module]]:
    """Get original (unpatched) class by name."""
    return _original_classes.get(class_name)


# Helper functions for creating layer instances with specific quantization kinds
def create_quantized_layer(base_class_name: str, *args, quant_kind: str = "other", **kwargs):
    """
    Create a quantized layer instance.
    
    Args:
        base_class_name: Name of base class (e.g., "ColumnParallelLinear")
        quant_kind: Layer kind ("attn", "mlp", "other")
        *args, **kwargs: Arguments for base class constructor
    """
    try:
        import diffulex.layer.linear as linear_module
    except ImportError:
        raise RuntimeError("diffulex not loaded")
    
    base_class = getattr(linear_module, base_class_name, None)
    if base_class is None:
        raise ValueError(f"Unknown linear class: {base_class_name}")
    
    # Create with quant_kind
    instance = base_class(*args, **kwargs)
    
    # Set quant kind if mixin is applied
    if hasattr(instance, 'init_quantization'):
        instance.init_quantization(quant_kind)
    
    return instance


# Dynamic layer replacement for TP (Tensor Parallel) scenarios
class DynamicQuantizedLinear:
    """
    Dynamic layer that chooses implementation based on context.
    
    This allows runtime selection between quantized and non-quantized paths.
    """
    
    def __new__(cls, base_class: Type[nn.Module], *args, 
                quant_kind: str = "other", enable_quant: bool = True, **kwargs):
        """
        Create appropriate layer instance.
        
        Args:
            base_class: Base linear class
            quant_kind: Layer kind for strategy selection
            enable_quant: Whether to enable quantization for this layer
            *args, **kwargs: Constructor arguments
        """
        # Get context to check if quantization is enabled
        if enable_quant:
            from .context import get_context
            ctx = get_context()
            
            # Check if we have a strategy for this kind
            if ctx.has_strategy(f"linear_{quant_kind}"):
                # Create quantized instance
                instance = base_class(*args, **kwargs)
                
                # Apply mixin dynamically if not already patched
                if not hasattr(instance, '_forward_base'):
                    LinearQuantizationMixin.init_quantization(instance, quant_kind)
                    # Bind methods
                    instance._forward_base = LinearQuantizationMixin._forward_base.__get__(instance, type(instance))
                    instance.has_quantized_weight = LinearQuantizationMixin.has_quantized_weight.__get__(instance, type(instance))
                    instance.has_offline_quantized_weight = LinearQuantizationMixin.has_offline_quantized_weight.__get__(instance, type(instance))
                    instance.set_quantized_weight = LinearQuantizationMixin.set_quantized_weight.__get__(instance, type(instance))
                    instance.set_offline_quantized_weight = LinearQuantizationMixin.set_offline_quantized_weight.__get__(instance, type(instance))
                    instance.enable_forward_plan = LinearQuantizationMixin.enable_forward_plan.__get__(instance, type(instance))
                
                return instance
        
        # Return standard instance
        return base_class(*args, **kwargs)
