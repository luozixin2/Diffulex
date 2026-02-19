"""Diffulex Edge models.

Supported diffusion language models for edge deployment:
- FastdLLMV2Edge: FastdLLM V2 architecture
- DreamEdge: Dream model architecture  
- LLaDAEdge: LLaDA model architecture
- SDAREdge: SDAR model architecture (with per-head Q/K norm)
"""

# Import all edge models
from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.model.dream_edge import DreamEdge, DreamEdgeConfig
from diffulex_edge.model.llada_edge import LLaDAEdge, LLaDAEdgeConfig
from diffulex_edge.model.sdar_edge import SDAREdge, SDAREdgeConfig
from diffulex_edge.model.kv_cache import KVCache, KVCacheConfig, create_kv_caches

# Import model loader
from diffulex_edge.model.model_loader import (
    load_hf_model,
    load_hf_weights_to_edge,
    create_edge_config,
    detect_model_type,
    MODEL_REGISTRY,
)

__all__ = [
    # Models
    "FastdLLMV2Edge",
    "DreamEdge",
    "LLaDAEdge",
    "SDAREdge",
    # Configs
    "FastdLLMV2EdgeConfig",
    "DreamEdgeConfig",
    "LLaDAEdgeConfig",
    "SDAREdgeConfig",
    # Model Loader
    "load_hf_model",
    "load_hf_weights_to_edge",
    "create_edge_config",
    "detect_model_type",
    "MODEL_REGISTRY",
    # KV Cache
    "KVCache",
    "KVCacheConfig",
    "create_kv_caches",
]
