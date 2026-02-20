"""Model-specific samplers aligned with diffulex implementations.

Each sampler is implemented independently to match the original behavior:
- FastdLLMV2Sampler: Uses ShiftLogits, always accepts at least 1 token
- LLaDASampler: Uses NoShiftLogits, pre_block_complete logic
- DreamSampler: Uses ShiftLogits, pre_block_complete logic
- SDARSampler: Uses ShiftLogits, similar to FastdLLM V2
"""

from typing import Dict, Type

from diffulex_edge.runtime.sampler.models.fast_dllm_v2 import (
    FastdLLMV2Sampler,
    FastdLLMV2SampleOutput,
)
from diffulex_edge.runtime.sampler.models.llada import (
    LLaDASampler,
    LLaDASampleOutput,
)
from diffulex_edge.runtime.sampler.models.dream import (
    DreamSampler,
    DreamSampleOutput,
)
from diffulex_edge.runtime.sampler.models.sdar import (
    SDARSampler,
    SDARSampleOutput,
)

__all__ = [
    "FastdLLMV2Sampler",
    "FastdLLMV2SampleOutput",
    "LLaDASampler",
    "LLaDASampleOutput",
    "DreamSampler",
    "DreamSampleOutput",
    "SDARSampler",
    "SDARSampleOutput",
    "SAMPLER_REGISTRY",
    "get_sampler_class",
]

# Type variable for sampler classes
SamplerClass = Type[
    FastdLLMV2Sampler | LLaDASampler | DreamSampler | SDARSampler
]

# Registry for model type lookup
SAMPLER_REGISTRY: Dict[str, SamplerClass] = {
    "fast_dllm_v2": FastdLLMV2Sampler,
    "llada": LLaDASampler,
    "dream": DreamSampler,
    "sdar": SDARSampler,
}


def get_sampler_class(model_type: str) -> SamplerClass:
    """Get sampler class by model type.
    
    Args:
        model_type: One of "fast_dllm_v2", "llada", "dream", "sdar"
        
    Returns:
        Sampler class
        
    Raises:
        ValueError: If model_type is unknown
    """
    if model_type not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(SAMPLER_REGISTRY.keys())}"
        )
    return SAMPLER_REGISTRY[model_type]