"""
Quantization strategy interfaces.

This module defines abstract base classes for different types of quantization strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch


class QuantizationStrategy(ABC):
    """Quantization strategy abstract base class."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @abstractmethod
    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        """
        Returns storage dtype and itemsize.
        
        Returns:
            (storage_dtype, itemsize): Tuple of storage dtype and item size in bytes.
        """
        pass
    
    @abstractmethod
    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """
        Quantize a tensor.
        
        Args:
            tensor: Input tensor to quantize.
            **kwargs: Additional arguments for quantization.
        
        Returns:
            (quantized_tensor, scale_or_metadata): Tuple of quantized tensor and scale/metadata.
        """
        pass
    
    @abstractmethod
    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """
        Dequantize a tensor.
        
        Args:
            quantized: Quantized tensor to dequantize.
            scale_or_metadata: Scale or metadata needed for dequantization.
            **kwargs: Additional arguments for dequantization.
        
        Returns:
            Dequantized tensor.
        """
        pass
    
    @abstractmethod
    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        """
        Returns the shape of scale tensor.
        
        Args:
            original_shape: Original tensor shape.
            **kwargs: Additional arguments (e.g., num_kv_heads for KV cache).
        
        Returns:
            Scale tensor shape.
        """
        pass


class KVCacheQuantizationStrategy(QuantizationStrategy):
    """KV Cache quantization strategy interface (extended interface)."""
    
    @abstractmethod
    def compute_scales(self, k: torch.Tensor, v: torch.Tensor, 
                      num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantization scales for K and V.
        
        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            num_kv_heads: Number of KV heads
            device: Target device
        
        Returns:
            (k_scale, v_scale): Tuple of K and V scales, shape [num_kv_heads]
        """
        pass
    
    @abstractmethod
    def update_scales(self, k: torch.Tensor, v: torch.Tensor,
                     k_scale: Optional[torch.Tensor], v_scale: Optional[torch.Tensor],
                     num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update quantization scales (e.g., using running max strategy).
        
        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            k_scale: Current K scale (None if first time)
            v_scale: Current V scale (None if first time)
            num_kv_heads: Number of KV heads
            device: Target device
        
        Returns:
            (updated_k_scale, updated_v_scale): Updated scales, shape [num_kv_heads]
        """
        pass
    
    def init_scales(self, num_kv_heads: int, device: torch.device) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Initialize quantization scales for K and V.
        
        This method should be called once per layer to initialize scale tensors.
        Strategies that don't require scales (e.g., BF16) should return (None, None).
        
        Args:
            num_kv_heads: Number of KV heads
            device: Target device
        
        Returns:
            (k_scale, v_scale): Initial scales, shape [num_kv_heads], or (None, None) if not needed
        """
        # Default implementation: return None (no scales needed)
        return None, None


class WeightQuantizationStrategy(QuantizationStrategy):
    """Weight quantization strategy interface (for future extension)."""
    
    @abstractmethod
    def quantize_weight(self, weight: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """
        Quantize model weights.
        
        Args:
            weight: Weight tensor to quantize.
            **kwargs: Additional arguments for quantization.
        
        Returns:
            (quantized_weight, scale_or_metadata): Tuple of quantized weight and scale/metadata.
        """
        pass
    
    @abstractmethod
    def dequantize_weight(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """
        Dequantize model weights.
        
        Args:
            quantized: Quantized weight tensor.
            scale_or_metadata: Scale or metadata needed for dequantization.
            **kwargs: Additional arguments for dequantization.
        
        Returns:
            Dequantized weight tensor.
        """
        pass

