"""
Quantization strategy factory.

This module provides factory functions to create quantization strategies from configuration.
"""

from typing import Optional
from diffulex.utils.quantization.context import QuantizationContext
from diffulex.utils.quantization.strategy import KVCacheQuantizationStrategy
from diffulex.utils.quantization.strategies import (
    NoQuantizationStrategy,
    KVCacheBF16Strategy,
    KVCacheFP8RunningMaxStrategy,
)


class QuantizationStrategyFactory:
    """Quantization strategy factory."""
    
    @staticmethod
    def create_kv_cache_strategy(dtype: Optional[str] = None) -> KVCacheQuantizationStrategy:
        """
        Create KV Cache quantization strategy.
        
        Args:
            dtype: KV cache dtype string:
                - None or "bf16": BF16 (no quantization)
                - "fp16": FP16 (no quantization, future support)
                - "fp32": FP32 (no quantization, future support)
                - "fp8" or "fp8_e4m3": FP8 E4M3 with running max
                - "fp8_e5m2": FP8 E5M2 with running max
        
        Returns:
            KV Cache quantization strategy instance
        
        Raises:
            ValueError: If dtype is not supported
        """
        if dtype is None or dtype.lower() == "bf16":
            return KVCacheBF16Strategy()
        
        dtype_lower = dtype.lower()
        
        if dtype_lower in ("fp16", "float16"):
            # TODO: Implement FP16 strategy if needed
            # For now, use BF16 strategy (no quantization)
            return KVCacheBF16Strategy()
        
        if dtype_lower in ("fp32", "float32"):
            # TODO: Implement FP32 strategy if needed
            # For now, use BF16 strategy (no quantization)
            return KVCacheBF16Strategy()
        
        if dtype_lower in ("fp8", "fp8_e4m3", "e4m3"):
            return KVCacheFP8RunningMaxStrategy("fp8_e4m3")
        
        if dtype_lower in ("fp8_e5m2", "e5m2"):
            return KVCacheFP8RunningMaxStrategy("fp8_e5m2")
        
        raise ValueError(f"Unsupported kv_cache_dtype: {dtype}")
    
    @staticmethod
    def create_from_config(config) -> QuantizationContext:
        """
        Create and configure quantization context from config object.
        
        Args:
            config: Configuration object that may contain quantization-related fields:
                - kv_cache_dtype: KV cache dtype string
                - weight_dtype: Weight dtype string (future)
        
        Returns:
            Configured quantization context
        """
        ctx = QuantizationContext.current()
        
        # KV Cache strategy
        kv_cache_dtype = getattr(config, 'kv_cache_dtype', None)
        if kv_cache_dtype:
            strategy = QuantizationStrategyFactory.create_kv_cache_strategy(kv_cache_dtype)
            ctx.set_strategy('kv_cache', strategy)
        
        # Future: Weight strategy
        # weight_dtype = getattr(config, 'weight_dtype', None)
        # if weight_dtype:
        #     strategy = QuantizationStrategyFactory.create_weight_strategy(weight_dtype)
        #     ctx.set_strategy('weight', strategy)
        
        return ctx

