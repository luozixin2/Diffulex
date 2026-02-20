# DiffuLex Edge Runtime

Unified inference runtime for diffusion language models.

## Structure

```
runtime/
├── __init__.py          # Main exports
├── engine.py            # Unified DiffusionEngine (PyTorch + PTE)
├── block.py             # Block management for diffusion
├── diffusion.py         # Core diffusion components
└── sampler/             # Sampling strategies
    ├── __init__.py      # Sampler exports
    ├── token.py         # Token-level samplers (Greedy, TopK, TopP)
    ├── base.py          # Base sampling utilities
    ├── shift.py         # Logits shifting for diffusion
    └── models/          # Model-specific samplers
        ├── __init__.py
        ├── fast_dllm_v2.py
        ├── llada.py
        ├── dream.py
        └── sdar.py
```

## Quick Start

### Engine Usage

```python
from diffulex_edge.runtime import DiffusionEngine, DiffusionGenerationConfig

# PyTorch model with KV cache
engine = DiffusionEngine.from_model(model, model_type="sdar", use_kv_cache=True)

# PyTorch model without KV cache
engine = DiffusionEngine.from_model(model, model_type="sdar", use_kv_cache=False)

# ExecuTorch PTE model
engine = DiffusionEngine.from_pte("model.pte", model_type="sdar", max_seq_len=2048)

# Generate
tokens = engine.generate(prompt_tokens, DiffusionGenerationConfig(max_new_tokens=50))
```

### Token-Level Samplers (Autoregressive)

```python
from diffulex_edge.runtime.sampler import GreedySampler, TopKSampler, TopPSampler

# Greedy sampling
sampler = GreedySampler()
token = sampler.sample(logits)

# Top-K sampling
sampler = TopKSampler(k=50, temperature=0.8)
token = sampler.sample(logits)

# Top-P (nucleus) sampling
sampler = TopPSampler(p=0.9, temperature=1.0)
token = sampler.sample(logits)
```

### Block-Level Samplers (Diffusion)

```python
from diffulex_edge.runtime.sampler import SDARSampler, FastdLLMV2Sampler

# SDAR sampler
sampler = SDARSampler(threshold=0.9, temperature=1.0)

# FastdLLM V2 sampler
sampler = FastdLLMV2Sampler(threshold=0.95, temperature=1.0)
```

## Backward Compatibility

The following aliases are maintained for backward compatibility:

- `InferenceEngine` → `DiffusionEngine`
- `GenerationConfig` → `DiffusionGenerationConfig`

## Migration Notes

### From old structure:

```python
# Old (before refactoring)
from diffulex_edge.runtime.sampler import GreedySampler  # From sampler.py
from diffulex_edge.runtime.engine_simple import DiffusionEngine
```

### To new structure:

```python
# New (after refactoring)
from diffulex_edge.runtime.sampler import GreedySampler  # From sampler/token.py
from diffulex_edge.runtime import DiffusionEngine  # Unified engine
```

All imports remain backward compatible - no code changes required.
