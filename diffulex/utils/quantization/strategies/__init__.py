"""
Quantization strategy implementations.
"""

from diffulex.utils.quantization.strategies.no_quantization import NoQuantizationStrategy
from diffulex.utils.quantization.strategies.kv_cache_bf16 import KVCacheBF16Strategy
from diffulex.utils.quantization.strategies.kv_cache_fp8_running_max import KVCacheFP8RunningMaxStrategy

__all__ = [
    'NoQuantizationStrategy',
    'KVCacheBF16Strategy',
    'KVCacheFP8RunningMaxStrategy',
]

