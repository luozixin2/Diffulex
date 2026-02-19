"""
Diffulex Edge: Edge-optimized diffusion LLM inference
======================================================

A simplified version of Diffulex for edge devices (iOS/Android).

Key differences from server version:
- No Tensor Parallel (single device)
- No custom CUDA kernels (use PyTorch SDPA)
- Static KV Cache (input/output mode)
- XNNPACK/CoreML backend support via ExecuTorch
"""

__version__ = "0.2.0"

from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.model.kv_cache import KVCache, KVCacheConfig, create_kv_caches

__all__ = [
    "FastdLLMV2Edge",
    "FastdLLMV2EdgeConfig",
    "KVCache",
    "KVCacheConfig",
    "create_kv_caches",
]
