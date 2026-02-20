"""Runtime module for DiffuLex Edge.

Provides inference engines and samplers for running exported models.

Two main engine types are available:
- DiffusionEngine: For diffusion-based generation (supports both PyTorch and PTE)
- InferenceEngine: Alias for DiffusionEngine (backward compatibility)

Usage:
    # PyTorch model with KV cache
    from diffulex_edge.runtime import DiffusionEngine
    engine = DiffusionEngine.from_model(model, model_type="sdar")
    tokens = engine.generate(prompt_tokens)
    
    # ExecuTorch PTE model
    engine = DiffusionEngine.from_pte("model.pte", model_type="sdar", max_seq_len=2048)
    tokens = engine.generate(prompt_tokens)
    
    # Token-level sampling (autoregressive)
    from diffulex_edge.runtime.sampler import GreedySampler, TopKSampler
    
    # Block-level sampling (diffusion)
    from diffulex_edge.runtime.sampler import SDARSampler, FastdLLMV2Sampler
"""

# Engine (unified - from engine.py)
from diffulex_edge.runtime.engine import (
    DiffusionEngine,
    DiffusionGenerationConfig,
    InferenceEngine,  # Backward compatibility alias
    GenerationConfig,  # Backward compatibility alias
)

# Block management (from block.py)
from diffulex_edge.runtime.block import (
    DiffusionBlock,
    DiffusionBlockManager,
    BlockStatus,
)

# Additional diffusion components (from diffusion.py)
from diffulex_edge.runtime.diffusion import (
    SampleOutput,
    DiffusionSampler,
)

__all__ = [
    # Engines
    "DiffusionEngine",
    "InferenceEngine",  # Alias for backward compatibility
    "DiffusionGenerationConfig",
    "GenerationConfig",  # Alias for backward compatibility
    # Block management
    "DiffusionBlock",
    "DiffusionBlockManager",
    "BlockStatus",
    # Diffusion components
    "SampleOutput",
    "DiffusionSampler",
]
