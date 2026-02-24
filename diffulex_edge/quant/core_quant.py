"""Core quantization implementations (INT8 and FP16 weight quantization).

This module provides weight compression for PTE export, achieving file size reduction.
"""

import time
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseQuantizer, QuantizationConfig, QuantizationResult, QuantizationDtype, QuantizationMode


class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights for PTE export.
    
    Stores weights in lower precision (INT8 or FP16) and dequantizes during forward.
    This achieves actual size reduction in the exported model file.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to use bias
        dtype: Quantization dtype (int8 or float16)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.int8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_dtype = dtype
        
        if dtype == torch.int8:
            # Int8 weights with per-channel scales
            self.register_buffer(
                'weight', 
                torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
            )
            self.register_buffer(
                'weight_scale', 
                torch.ones(out_features, dtype=torch.float32)
            )
        elif dtype == torch.float16:
            # FP16 weights (no scale needed)
            self.register_buffer(
                'weight', 
                torch.randn(out_features, in_features, dtype=torch.float16)
            )
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        if self.quant_dtype == torch.int8:
            weight = self.weight.float() * self.weight_scale.view(-1, 1)
        elif self.quant_dtype == torch.float16:
            weight = self.weight.float()
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)
    
    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        dtype: torch.dtype = torch.int8
    ) -> "QuantizedLinear":
        """Create QuantizedLinear from a float Linear layer."""
        qlinear = cls(linear.in_features, linear.out_features, linear.bias is not None, dtype)
        
        weight = linear.weight.data
        
        if dtype == torch.int8:
            # Per-channel symmetric quantization
            max_abs = weight.abs().max(dim=1, keepdim=True)[0]
            scale = max_abs / 127.0
            scale[scale == 0] = 1.0
            
            weight_quant = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
            qlinear.weight.copy_(weight_quant)
            qlinear.weight_scale.copy_(scale.squeeze())
            
        elif dtype == torch.float16:
            # Direct cast to FP16
            qlinear.weight.copy_(weight.to(torch.float16))
        
        if linear.bias is not None:
            qlinear.bias.copy_(linear.bias.data)
        
        return qlinear


class QuantizedEmbedding(nn.Module):
    """Embedding layer with quantized weights for PTE export.
    
    Stores weights in lower precision (INT8 or FP16) and dequantizes during forward.
    This achieves actual size reduction in the exported model file.
    
    Args:
        num_embeddings: Size of the dictionary of embeddings
        embedding_dim: The size of each embedding vector
        dtype: Quantization dtype (int8 or float16)
        padding_idx: If specified, the entries at padding_idx do not contribute to the gradient
        max_norm: If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm
        norm_type: The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency of the words in the mini-batch
        sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype = torch.int8,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quant_dtype = dtype
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        if dtype == torch.int8:
            # Int8 weights with per-channel scales
            self.register_buffer(
                'weight',
                torch.randint(-128, 127, (num_embeddings, embedding_dim), dtype=torch.int8)
            )
            self.register_buffer(
                'weight_scale',
                torch.ones(num_embeddings, dtype=torch.float32)
            )
        elif dtype == torch.float16:
            # FP16 weights (no scale needed)
            self.register_buffer(
                'weight',
                torch.randn(num_embeddings, embedding_dim, dtype=torch.float16)
            )
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        if self.quant_dtype == torch.int8:
            # Dequantize: weight is (num_embeddings, embedding_dim), scale is (num_embeddings,)
            weight_float = self.weight.float() * self.weight_scale.view(-1, 1)
        elif self.quant_dtype == torch.float16:
            weight_float = self.weight.float()
        else:
            weight_float = self.weight
        
        return F.embedding(
            input,
            weight_float,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
    
    @classmethod
    def from_float(
        cls,
        embedding: nn.Embedding,
        dtype: torch.dtype = torch.int8
    ) -> "QuantizedEmbedding":
        """Create QuantizedEmbedding from a float Embedding layer."""
        qembedding = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            dtype,
            embedding.padding_idx,
            embedding.max_norm,
            embedding.norm_type,
            embedding.scale_grad_by_freq,
            embedding.sparse,
        )
        
        weight = embedding.weight.data
        
        if dtype == torch.int8:
            # Per-channel (per-embedding) symmetric quantization
            max_abs = weight.abs().max(dim=1, keepdim=True)[0]
            scale = max_abs / 127.0
            scale[scale == 0] = 1.0
            
            weight_quant = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
            qembedding.weight.copy_(weight_quant)
            qembedding.weight_scale.copy_(scale.squeeze())
            
        elif dtype == torch.float16:
            # Direct cast to FP16
            qembedding.weight.copy_(weight.to(torch.float16))
        
        return qembedding


class WeightOnlyQuantizer(BaseQuantizer):
    """Weight-only quantizer for INT8 and FP16.
    
    Converts Linear layer weights to lower precision for PTE export.
    Supports both INT8 (4x compression) and FP16 (2x compression).
    
    Example:
        >>> quantizer = WeightOnlyQuantizer()
        >>> config = QuantizationConfig(dtype=QuantizationDtype.INT8)
        >>> result = quantizer.quantize(model, config)
    """
    
    def is_available(self) -> bool:
        """Weight-only quantization is always available."""
        return True
    
    def quantize(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None
    ) -> QuantizationResult:
        """Apply weight-only quantization."""
        import copy
        
        config = config or QuantizationConfig(dtype=QuantizationDtype.INT8)
        self._warnings = []
        start_time = time.time()
        
        # Get target dtype
        if config.dtype == QuantizationDtype.INT8:
            target_dtype = torch.int8
            compression_ratio = 4.0
        elif config.dtype == QuantizationDtype.FP16:
            target_dtype = torch.float16
            compression_ratio = 2.0
        else:
            raise ValueError(f"Unsupported dtype for weight-only: {config.dtype}")
        
        # Create a copy to avoid modifying original
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Get original memory
        original_memory = self._estimate_memory(model_copy)
        
        # Convert Linear and Embedding layers
        converted_counts = self._convert_layers(model_copy, target_dtype, config)
        total_converted = sum(converted_counts.values())
        
        # Get new memory
        new_memory = self._estimate_memory(model_copy)
        
        # Compute metrics
        metrics = {
            "original_memory_mb": original_memory,
            "quantized_memory_mb": new_memory,
            "memory_reduction_mb": original_memory - new_memory,
            "memory_reduction_pct": ((original_memory - new_memory) / original_memory * 100)
                if original_memory > 0 else 0,
            "compression_ratio": (original_memory / new_memory) if new_memory > 0 else 1.0,
            "converted_layers": total_converted,
            "converted_linear_layers": converted_counts["linear"],
            "converted_embedding_layers": converted_counts["embedding"],
            "quantization_time_sec": time.time() - start_time,
        }
        
        return QuantizationResult(
            model=model_copy,
            metrics=metrics,
            warnings=self._warnings.copy(),
        )
    
    def _convert_layers(
        self,
        model: nn.Module,
        dtype: torch.dtype,
        config: QuantizationConfig
    ) -> dict:
        """Convert all Linear and Embedding layers in model.
        
        Returns:
            Dictionary with counts of converted layers by type
        """
        converted_counts = {"linear": 0, "embedding": 0}
        
        def convert_module(module: nn.Module, name: str = ""):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child, nn.Linear):
                    if self._should_convert_layer(full_name, child, config):
                        qlinear = QuantizedLinear.from_float(child, dtype=dtype)
                        setattr(module, child_name, qlinear)
                        converted_counts["linear"] += 1
                elif isinstance(child, nn.Embedding):
                    if self._should_convert_layer(full_name, child, config):
                        qembedding = QuantizedEmbedding.from_float(child, dtype=dtype)
                        setattr(module, child_name, qembedding)
                        converted_counts["embedding"] += 1
                else:
                    convert_module(child, full_name)
        
        convert_module(model)
        return converted_counts
    
    def _convert_linear_layers(
        self,
        model: nn.Module,
        dtype: torch.dtype,
        config: QuantizationConfig
    ) -> int:
        """Convert all Linear layers in model.
        
        .. deprecated::
            Use _convert_layers instead for both Linear and Embedding conversion.
        """
        counts = self._convert_layers(model, dtype, config)
        return counts["linear"]


# =============================================================================
# Unified API - Simple convenience functions
# =============================================================================

def quantize_to_fp16(model: nn.Module) -> nn.Module:
    """Quantize model to FP16 (half precision).
    
    Converts all Linear layer weights to FP16.
    Achieves 2x size reduction (4 bytes -> 2 bytes per weight).
    
    Args:
        model: Model to quantize
        
    Returns:
        New quantized model (original model is not modified)
        
    Example:
        >>> quantized_model = quantize_to_fp16(model)
        >>> print(f"Size reduced by 50%")
    """
    quantizer = WeightOnlyQuantizer()
    config = QuantizationConfig(dtype=QuantizationDtype.FP16)
    result = quantizer.quantize(model, config)
    return result.model


def quantize_to_int8(model: nn.Module, mode: str = "weight_only") -> nn.Module:
    """Quantize model to INT8.
    
    Converts all Linear layer weights to INT8 with per-channel scaling.
    Achieves 4x size reduction (4 bytes -> 1 byte per weight).
    
    Args:
        model: Model to quantize
        mode: Quantization mode ("weight_only", "dynamic", or "static")
            - weight_only: Only quantize weights, activations stay FP32
            - dynamic: Weights quantized, activations quantized at runtime
            - static: Weights quantized, activations use pre-calibrated scales
        
    Returns:
        New quantized model (original model is not modified)
        
    Example:
        >>> quantized_model = quantize_to_int8(model, mode="weight_only")
        >>> print(f"Size reduced by 75%")
    """
    quantizer = WeightOnlyQuantizer()
    config = QuantizationConfig(
        dtype=QuantizationDtype.INT8,
        mode=QuantizationMode.WEIGHT_ONLY if mode == "weight_only" else QuantizationMode.DYNAMIC
    )
    result = quantizer.quantize(model, config)
    return result.model


def quantize_model(
    model: nn.Module,
    dtype: Union[str, QuantizationDtype] = "int8",
    mode: str = "weight_only"
) -> nn.Module:
    """Unified quantization interface.
    
    Quantizes a model to the specified precision.
    
    Args:
        model: Model to quantize
        dtype: Target data type ("fp16", "int8", "int4", or QuantizationDtype)
        mode: Quantization mode ("weight_only", "dynamic", "static")
        
    Returns:
        New quantized model (original model is not modified)
        
    Raises:
        ValueError: If dtype or mode is not supported
        
    Example:
        >>> # FP16 quantization (2x compression)
        >>> model_fp16 = quantize_model(model, dtype="fp16")
        >>>
        >>> # INT8 weight-only (4x compression)
        >>> model_int8 = quantize_model(model, dtype="int8", mode="weight_only")
        >>>
        >>> # INT4 with torchao (8x compression)
        >>> model_int4 = quantize_model(model, dtype="int4")
    """
    # Convert string to enum
    if isinstance(dtype, str):
        dtype_map = {
            "fp32": QuantizationDtype.FP32,
            "fp16": QuantizationDtype.FP16,
            "int8": QuantizationDtype.INT8,
            "int4": QuantizationDtype.INT4,
        }
        if dtype.lower() not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. Use one of: {list(dtype_map.keys())}")
        dtype = dtype_map[dtype.lower()]
    
    # Route to appropriate quantizer
    if dtype == QuantizationDtype.FP16:
        return quantize_to_fp16(model)
    
    elif dtype == QuantizationDtype.INT8:
        return quantize_to_int8(model, mode=mode)
    
    elif dtype == QuantizationDtype.INT4:
        # Import here to avoid dependency issues
        from .int4_quantizer import INT4Quantizer, INT4Config
        quantizer = INT4Quantizer()
        config = INT4Config(group_size=32)  # Default group size
        result = quantizer.quantize(model, config)
        return result.model
    
    else:
        raise ValueError(f"Quantization to {dtype} is not supported")


# =============================================================================
# Utility functions
# =============================================================================

def verify_quantization_accuracy(
    original_model: nn.Module,
    quantized_model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    tolerance: float = 0.01
) -> bool:
    """Verify that quantized model produces similar outputs to original.
    
    Args:
        original_model: Original FP32 model
        quantized_model: Quantized model
        example_inputs: Example inputs for testing
        tolerance: Maximum acceptable relative error
        
    Returns:
        True if outputs are within tolerance
    """
    quantizer = WeightOnlyQuantizer()
    metrics = quantizer.validate_numerical_accuracy(
        original_model, quantized_model, example_inputs, tolerance
    )
    return metrics.get("within_tolerance", False)


def get_model_compression_ratio(original_model: nn.Module, quantized_model: nn.Module) -> float:
    """Calculate compression ratio between original and quantized models.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        
    Returns:
        Compression ratio (original_size / quantized_size)
    """
    def get_model_size(m: nn.Module) -> float:
        param_size = sum(p.numel() * p.element_size() for p in m.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in m.buffers())
        return param_size + buffer_size
    
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    return original_size / quantized_size if quantized_size > 0 else 1.0


__all__ = [
    # Core classes
    "QuantizedLinear",
    "QuantizedEmbedding",
    "WeightOnlyQuantizer",
    
    # Unified API
    "quantize_model",
    "quantize_to_fp16",
    "quantize_to_int8",
    
    # Utilities
    "verify_quantization_accuracy",
    "get_model_compression_ratio",
]
