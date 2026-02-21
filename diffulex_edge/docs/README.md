# DiffuLex Edge Documentation

## Quick Navigation

### Getting Started
- [Architecture](ARCHITECTURE.md) - System design and module structure
- [User Guide](USER_GUIDE.md) - Model export and CLI usage
- [Quantization](QUANTIZATION.md) - Quantization support (FP16/INT8/INT4)

### Advanced Topics
- [Technical Details](TECHNICAL.md) - Block diffusion and KV cache internals

### Development
- [Test Plan](TEST_PLAN.md) - Testing strategy and coverage

---

## Documentation Structure

```
docs/
├── ARCHITECTURE.md      # System architecture & refactoring history
├── USER_GUIDE.md        # Export guide and CLI usage (merged)
├── QUANTIZATION.md      # Quantization implementation (replaces planning doc)
├── TECHNICAL.md         # Block diffusion and KV cache details (merged)
├── TEST_PLAN.md         # Testing strategy
└── README.md            # This file
```

---

## Architecture Overview

The architecture follows a modular design with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│  Components (Shared)                                         │
│  - RMSNorm, RotaryEmbedding, SwiGLUMLP                      │
├─────────────────────────────────────────────────────────────┤
│  Models                                                      │
│  - SDAREdge, FastdLLMV2Edge, DreamEdge, LLaDAEdge         │
│  - Base classes: DiffusionModel, ModelConfig                │
│  - Export wrappers: ExportWrapper, BlockDiffusionWrapper    │
├─────────────────────────────────────────────────────────────┤
│  Backends                                                    │
│  - XNNPACK, CoreML, QNN (backend-agnostic)                  │
├─────────────────────────────────────────────────────────────┤
│  Runtime                                                     │
│  - DiffusionEngine, Samplers, KV Cache                      │
├─────────────────────────────────────────────────────────────┤
│  Quantization                                                │
│  - FP16Quantizer, INT8Quantizer, INT4Quantizer             │
│  - BaseQuantizer with unified API                           │
└─────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed information.
