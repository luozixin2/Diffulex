"""Runtime module for DiffuLex Edge.

Provides inference engines for running exported models.

Two main engines are available:
- InferenceEngine: Autoregressive generation engine (PyTorch + PTE)
- DiffusionEngine: Diffusion-based generation engine (PyTorch + PTE)

Usage:
    # PyTorch model
    engine = InferenceEngine.from_model(model)
    # or
    engine = DiffusionEngine.from_model(model)
    
    # ExecuTorch PTE model
    engine = InferenceEngine.from_pte("model.pte")
    # or
    engine = DiffusionEngine.from_pte("model.pte")
"""

from .engine import InferenceEngine, GenerationConfig
from .sampler import Sampler, GreedySampler, TopKSampler, TopPSampler
from .diffusion import (
    DiffusionBlock,
    DiffusionBlockManager,
    DiffusionSampler,
    DiffusionEngine,
    DiffusionGenerationConfig,
    SampleOutput,
)

__all__ = [
    # Inference engines
    "InferenceEngine",
    "DiffusionEngine",
    # Configurations
    "GenerationConfig",
    "DiffusionGenerationConfig",
    # Samplers
    "Sampler",
    "GreedySampler",
    "TopKSampler",
    "TopPSampler",
    # Diffusion components
    "DiffusionBlock",
    "DiffusionBlockManager",
    "DiffusionSampler",
    "SampleOutput",
]
