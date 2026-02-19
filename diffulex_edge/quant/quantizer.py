"""Main quantizer for DiffuLex Edge models.

Provides quantization workflows compatible with ExecuTorch.
"""

import dataclasses
from enum import Enum, auto
from typing import Optional, Callable, List, Tuple, Any
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer import Quantizer
# Try different import paths for compatibility
try:
    from torch._export import capture_pre_autograd_graph
except ImportError:
    try:
        from torch.export import capture_pre_autograd_graph
    except ImportError:
        capture_pre_autograd_graph = None

from .observers import get_default_observers, get_dynamic_quant_observers


class QuantizationMode(Enum):
    """Quantization mode."""
    STATIC = auto()      # Static quantization (weights + activations)
    DYNAMIC = auto()     # Dynamic quantization (weights only, runtime activation quant)
    QAT = auto()         # Quantization Aware Training
    WEIGHT_ONLY = auto() # Weight-only quantization


class QuantizationScheme(Enum):
    """Quantization scheme."""
    INT8 = "int8"
    INT4 = "int4"  # May require specialized kernels
    FP16 = "fp16"  # Actually just casting, not quantization


@dataclasses.dataclass
class QuantizationConfig:
    """Configuration for quantization.
    
    Attributes:
        mode: Quantization mode (static/dynamic/QAT/weight-only)
        scheme: Quantization scheme (INT8/INT4/FP16)
        is_per_channel: Use per-channel quantization for weights
        is_qat: Use quantization aware training
        calibration_samples: Number of samples for calibration
        preserve_activation_precision_layers: Layer types to keep in FP32
    """
    mode: QuantizationMode = QuantizationMode.DYNAMIC
    scheme: QuantizationScheme = QuantizationScheme.INT8
    is_per_channel: bool = True
    is_qat: bool = False
    calibration_samples: int = 100
    preserve_activation_precision_layers: Tuple[str, ...] = ("lm_head",)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.scheme == QuantizationScheme.INT4:
            raise NotImplementedError("INT4 quantization not yet implemented")
        if self.mode == QuantizationMode.QAT and not self.is_qat:
            self.is_qat = True


class DiffuLexQuantizer:
    """Quantizer for DiffuLex Edge models.
    
    Usage:
        # 1. Create quantizer
        quantizer = DiffuLexQuantizer(config)
        
        # 2. Prepare model (after export)
        prepared = quantizer.prepare_for_quantization(model, example_inputs)
        
        # 3. Calibrate (static mode) or train (QAT mode)
        for batch in calibration_data:
            prepared(*batch)
        
        # 4. Convert to quantized model
        quantized = quantizer.convert(prepared)
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """Initialize quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()
        self._quantizer: Optional[Quantizer] = None
        self._prepared_model: Optional[torch.nn.Module] = None
        
    def _get_xnnpack_quantizer(self) -> Optional[Any]:
        """Get XNNPACK quantizer if available."""
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )
            
            xnn_config = get_symmetric_quantization_config(
                is_per_channel=self.config.is_per_channel,
                is_qat=self.config.is_qat,
            )
            
            quantizer = XNNPACKQuantizer()
            quantizer.set_global(xnn_config)
            return quantizer
        except ImportError:
            return None
    
    def _get_qnn_quantizer(self) -> Optional[Any]:
        """Get QNN quantizer if available (for mobile/edge devices)."""
        try:
            from executorch.backends.qualcomm.quantizer import QnnQuantizer
            return QnnQuantizer()
        except ImportError:
            return None
    
    def _get_default_quantizer(self) -> Quantizer:
        """Get default PyTorch quantizer."""
        # Fallback to PyTorch native quantizer
        return None  # Will use PyTorch's default behavior
    
    def prepare_for_quantization(
        self,
        model: nn.Module,
        example_inputs: Tuple[Any, ...],
    ) -> nn.Module:
        """Prepare model for quantization.
        
        This captures the pre-autograd graph and prepares for PT2E quantization.
        
        Args:
            model: The model to quantize
            example_inputs: Example inputs for tracing
            
        Returns:
            Prepared model ready for calibration or QAT
        """
        # Set model to eval mode for calibration
        model.eval()
        
        # Capture pre-autograd graph (required for PT2E)
        # This is the new way in PyTorch 2.x
        if capture_pre_autograd_graph is not None:
            try:
                captured = capture_pre_autograd_graph(model, example_inputs)
            except Exception as e:
                print(f"Warning: capture_pre_autograd_graph failed, using model directly: {e}")
                captured = model
        else:
            captured = model
        
        # Check if captured model has required attributes for PT2E
        # PT2E requires an ExportedProgram-like object with .meta attribute
        if not hasattr(captured, 'meta') or captured is model:
            # Fallback to eager mode quantization for simple models
            print("Warning: Model not suitable for PT2E, using fallback quantization")
            return self._prepare_fallback(captured, example_inputs)
        
        # Get quantizer
        quantizer = self._get_xnnpack_quantizer()
        
        if quantizer is not None:
            # Use XNNPACK quantizer
            if self.config.mode == QuantizationMode.QAT:
                prepared = prepare_qat_pt2e(captured, quantizer)
            else:
                prepared = prepare_pt2e(captured, quantizer)
        else:
            # Fallback: use PyTorch native quantization
            print("Warning: XNNPACK quantizer not available, using fallback")
            prepared = self._prepare_fallback(captured, example_inputs)
        
        self._prepared_model = prepared
        return prepared
    
    def _prepare_fallback(
        self,
        model: nn.Module,
        example_inputs: Tuple[Any, ...],
    ) -> nn.Module:
        """Fallback preparation using PyTorch native quantization."""
        # For dynamic quantization, we don't need observers
        if self.config.mode == QuantizationMode.DYNAMIC:
            # Dynamic quantization doesn't require calibration
            # Just return the model, quantization happens at convert time
            return model
        
        # For static quantization, use default qconfig
        qconfig = get_default_qconfig("x86")
        model.qconfig = qconfig
        
        # Note: Full implementation would use torch.ao.quantization.prepare
        # but this requires more setup for the specific model
        return model
    
    def calibrate(
        self,
        prepared_model: nn.Module,
        calibration_data: List[Tuple[Any, ...]],
    ) -> None:
        """Run calibration on prepared model.
        
        This collects statistics for static quantization.
        
        Args:
            prepared_model: Model prepared for quantization
            calibration_data: List of input tuples for calibration
        """
        if self.config.mode == QuantizationMode.DYNAMIC:
            print("Warning: Calibration not needed for dynamic quantization")
            return
        
        prepared_model.eval()
        with torch.no_grad():
            for inputs in calibration_data:
                # Handle both tuple and dict inputs
                if isinstance(inputs, dict):
                    prepared_model(**inputs)
                else:
                    prepared_model(*inputs)
    
    def convert(self, prepared_model: nn.Module) -> nn.Module:
        """Convert prepared model to quantized model.
        
        Args:
            prepared_model: Model after calibration/QAT
            
        Returns:
            Quantized model ready for ExecuTorch export
        """
        if hasattr(prepared_model, '_exported_program'):
            # PT2E path
            quantized = convert_pt2e(prepared_model)
            return quantized
        else:
            # Fallback path
            return self._convert_fallback(prepared_model)
    
    def _convert_fallback(self, model: nn.Module) -> nn.Module:
        """Fallback conversion for dynamic quantization."""
        if self.config.mode == QuantizationMode.DYNAMIC:
            # Apply dynamic quantization to linear layers
            quantized = torch.ao.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8,
            )
            return quantized
        
        return model
    
    def quantize_weights_only(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """Apply weight-only quantization.
        
        This quantizes weights offline without affecting computation.
        Good for reducing model size while maintaining FP32 computation.
        
        Args:
            model: The model to quantize
            
        Returns:
            Model with quantized weights
        """
        # Weight-only quantization using PyTorch quantization
        # This is a simplified version - full implementation would use
        # torch.ao.quantization.quantize_dynamic with appropriate settings
        
        model.eval()
        
        # Quantize all Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize weights to int8
                weight = module.weight.data
                
                # Compute scale and zero point
                w_min = weight.min()
                w_max = weight.max()
                
                # Symmetric quantization
                max_abs = max(abs(w_min), abs(w_max))
                scale = max_abs / 127.0
                
                if scale > 0:
                    # Quantize
                    weight_q = torch.round(weight / scale).clamp(-128, 127)
                    # Dequantize (simulating int8 storage with fp32 compute)
                    module.weight.data = weight_q * scale
        
        return model


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to a model.
    
    Convenience function for simple dynamic quantization use case.
    
    Args:
        model: Model to quantize
        
    Returns:
        Dynamically quantized model
    """
    quantizer = DiffuLexQuantizer(QuantizationConfig(mode=QuantizationMode.DYNAMIC))
    
    # For dynamic quantization, we just need to convert
    # No preparation/calibration needed
    return quantizer._convert_fallback(model)


def apply_weight_only_quantization(model: nn.Module) -> nn.Module:
    """Apply weight-only quantization to a model.
    
    Args:
        model: Model to quantize
        
    Returns:
        Weight-only quantized model
    """
    quantizer = DiffuLexQuantizer(QuantizationConfig(mode=QuantizationMode.WEIGHT_ONLY))
    return quantizer.quantize_weights_only(model)
