"""Sampler modules for DiffuLex Edge.

Provides two types of sampling:
1. Token-level sampling: Greedy, TopK, TopP, Temperature for autoregressive generation
2. Block-level sampling: For diffusion-based generation with block confirmation

Usage:
    # Token-level sampling (autoregressive)
    from diffulex_edge.runtime.sampler import GreedySampler, TopKSampler
    
    # Block-level sampling (diffusion)
    from diffulex_edge.runtime.sampler import FastdLLMV2Sampler, SDARSampler
    
    # Base utilities
    from diffulex_edge.runtime.sampler import sample_tokens, top_p_logits, top_k_logits
"""

# Token-level samplers (autoregressive)
from diffulex_edge.runtime.sampler.token import (
    Sampler,
    GreedySampler,
    TopKSampler,
    TopPSampler,
    TemperatureSampler,
    get_sampler,
)

# Base sampling utilities
from diffulex_edge.runtime.sampler.base import (
    sample_tokens,
    top_p_logits,
    top_k_logits,
)

# Logits shifting for diffusion
from diffulex_edge.runtime.sampler.shift import (
    ShiftLogitsSampler,
    NoShiftLogitsSampler,
)

# Block-level samplers (diffusion models)
from diffulex_edge.runtime.sampler.models import (
    FastdLLMV2Sampler,
    FastdLLMV2SampleOutput,
    LLaDASampler,
    LLaDASampleOutput,
    DreamSampler,
    DreamSampleOutput,
    SDARSampler,
    SDARSampleOutput,
    SAMPLER_REGISTRY,
    get_sampler_class,
)

__all__ = [
    # Token-level samplers
    "Sampler",
    "GreedySampler",
    "TopKSampler",
    "TopPSampler",
    "TemperatureSampler",
    "get_sampler",
    # Base utilities
    "sample_tokens",
    "top_p_logits",
    "top_k_logits",
    # Logits shifting
    "ShiftLogitsSampler",
    "NoShiftLogitsSampler",
    # Block-level samplers
    "FastdLLMV2Sampler",
    "FastdLLMV2SampleOutput",
    "LLaDASampler",
    "LLaDASampleOutput",
    "DreamSampler",
    "DreamSampleOutput",
    "SDARSampler",
    "SDARSampleOutput",
    # Registry
    "SAMPLER_REGISTRY",
    "get_sampler_class",
]
