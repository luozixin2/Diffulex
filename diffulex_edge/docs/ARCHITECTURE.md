# DiffuLex Edge - Architecture

Technical architecture, implementation details, and refactoring history.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Module Structure](#module-structure)
3. [New Architecture (Refactored)](#new-architecture-refactored)
4. [Architecture Evolution](#architecture-evolution)
5. [Best Practices](#best-practices)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
├─────────────────────────────────────────────────────────────┤
│                   DiffuLex Edge Runtime                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Tokenizer  │  │   Sampler   │  │   KV Cache Manager  │  │
│  │  (Hugging   │  │  (Diffusion)│  │    (Static)         │  │
│  │   Face)     │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              ExecuTorch Runtime (.pte model)                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Transformer Model                      │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │    │
│  │  │Embedding│→ │  Layer  │→ │  ...    │→ │ LM Head│ │    │
│  │  └─────────┘  │(Attn+MLP)│  └─────────┘  └────────┘ │    │
│  │               └─────────┘                           │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Backend Execution                         │
│     XNNPACK (CPU)    │    QNN (Qualcomm)   │  CoreML (Apple)│
└─────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
diffulex_edge/
├── components/          # Shared components (NEW)
│   ├── normalization.py # RMSNorm
│   ├── rope.py         # RotaryEmbedding
│   └── mlp.py          # SwiGLUMLP
├── model/              # Model implementations
│   ├── base.py         # Base classes (NEW)
│   ├── wrapper.py      # Export wrappers (NEW)
│   ├── sdar_edge.py    # SDAR model
│   ├── fast_dllm_v2_edge.py
│   ├── dream_edge.py
│   └── llada_edge.py
├── backends/           # Export backends
│   ├── base.py
│   ├── xnnpack_backend.py
│   ├── coreml_backend.py
│   └── qnn_backend.py
├── runtime/            # Inference engine
│   ├── engine.py
│   ├── sampler/
│   └── block.py
├── export/             # Export configuration
│   ├── config.py
│   └── exporter.py
└── quant/              # Quantization support
```

---

## New Architecture (Refactored)

### Shared Components (`components/`)

Reusable transformer components shared across all models:

```python
from diffulex_edge.components import RMSNorm, RotaryEmbedding, SwiGLUMLP

# RMSNorm - Layer normalization
norm = RMSNorm(hidden_size=512)

# RotaryEmbedding - Positional encoding
rope = RotaryEmbedding(head_dim=64, max_position_embeddings=2048)

# SwiGLUMLP - Feed-forward network
mlp = SwiGLUMLP(hidden_size=512, intermediate_size=2048)
```

**Benefits:**
- Eliminates ~420 lines of duplicate code
- Bug fixes only need to be applied once
- Consistent behavior across all models

### Base Classes (`model/base.py`)

```python
class ModelConfig:
    """Base configuration for all models."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    # ... common parameters

class DiffusionModel(nn.Module, ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def forward(self, input_ids, positions, **kwargs): ...
    
    def get_export_wrapper(self) -> Optional[nn.Module]:
        """Return export wrapper if supports_export."""
        
    def get_export_inputs(self, **kwargs) -> Tuple[Any, ...]:
        """Create example inputs for export."""
```

**Benefits:**
- Enforces consistent interface
- Clear extension points
- Polymorphic usage

### Export Wrappers (`model/wrapper.py`)

```python
class ExportWrapper(nn.Module):
    """Generic wrapper for ExecuTorch export."""
    def __init__(self, model):
        self.inner = model
        # Sanitize parameter names for ExecuTorch
        
    def forward(self, *args, **kwargs):
        return self.inner.forward_export(*args, **kwargs)

class BlockDiffusionWrapper(ExportWrapper):
    """Specialized wrapper for Block Diffusion models."""
    def __init__(self, model, block_size, max_seq_len): ...
```

**Benefits:**
- Wrapper logic moved from backends to models
- Backends simplified by ~30%
- All backends automatically support all models

### Backend Decoupling

All backends now use a generic model preparation method:

```python
def _prepare_model_for_export(self, model: nn.Module) -> nn.Module:
    """Prepare model for export (backend-agnostic)."""
    
    # Check if model provides an export wrapper
    if hasattr(model, 'get_export_wrapper'):
        wrapper = model.get_export_wrapper()
        if wrapper is not None:
            return wrapper
    
    # Backward compatibility: legacy forward_export
    if hasattr(model, 'forward_export'):
        return ExportWrapper(model)
    
    # Use model as-is
    return model
```

---

## Architecture Evolution

### Problems Identified (Pre-Refactor)

| Problem | Severity | Description |
|---------|----------|-------------|
| Wrapper in backend | 🔴 High | `SDAREdgeExportWrapper` embedded in XNNPACK backend |
| Code duplication | 🔴 High | RMSNorm, RoPE, MLP repeated 4 times across models |
| Inconsistent interfaces | 🟡 Medium | Different `forward()` signatures |
| Backend-model coupling | 🔴 High | Adding models required backend modifications |

### Architecture Principles Score

| Principle | Before | After | Status |
|-----------|--------|-------|--------|
| Single Responsibility | ⚠️ | ✅ | Backend only handles export |
| Open/Closed | ❌ | ✅ | Extend without modifying backends |
| Liskov Substitution | ⚠️ | ✅ | Unified interfaces |
| Interface Segregation | ⚠️ | ✅ | Clear abstractions |
| Dependency Inversion | ❌ | ✅ | Depend on abstractions |
| DRY | ❌ | ✅ | No duplication |
| High Cohesion | ⚠️ | ✅ | Components centralized |
| Low Coupling | ❌ | ✅ | Backend-model decoupled |

### Refactoring Commit

```
b4236a6 refactor(architecture): implement new modular architecture with shared components
```

**Key Changes:**
- Created `components/` module with shared implementations
- Created `model/base.py` with abstract base classes
- Created `model/wrapper.py` with export wrappers
- Simplified all backends with generic `_prepare_model_for_export()`
- Added `get_export_wrapper()` and `get_export_inputs()` to SDAREdge
- Maintained 100% backward compatibility

---

## Best Practices

### For Adding New Models

1. **Use shared components:**
   ```python
   from diffulex_edge.components import RMSNorm, RotaryEmbedding, SwiGLUMLP
   ```

2. **Inherit from base class:**
   ```python
   from diffulex_edge.model.base import DiffusionModel, ModelConfig
   
   class MyModel(DiffusionModel):
       def forward(self, input_ids, positions, **kwargs): ...
   ```

3. **Implement export interface:**
   ```python
   def get_export_wrapper(self):
       return ExportWrapper(self)
   
   def get_export_inputs(self, batch_size, seq_len, device):
       return (input_ids, positions)
   ```

### For Adding New Backends

1. **Inherit from base:**
   ```python
   from diffulex_edge.backends.base import EdgeBackend
   
   class MyBackend(EdgeBackend):
       def export(self, model, example_inputs, config): ...
   ```

2. **Use generic model preparation:**
   ```python
   export_model = self._prepare_model_for_export(model)
   ```

3. **No model-specific logic** - the preparation method handles all models automatically.

---

## References

- [Export Guide](EXPORT.md) - Model export documentation
- [Implementation Notes](IMPLEMENTATION.md) - Development details
- [CLI Usage](CLI_USAGE.md) - Command-line interface
- [Block Diffusion Export](BLOCK_DIFFUSION_EXPORT.md) - SDAR-specific export
