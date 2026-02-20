# DiffuLex Edge Documentation

## Quick Navigation

### Getting Started
- [Architecture](ARCHITECTURE.md) - System design and module structure
- [Export Guide](EXPORT.md) - How to export models to ExecuTorch
- [CLI Usage](CLI_USAGE.md) - Command-line interface reference

### Advanced Topics
- [Block Diffusion Export](BLOCK_DIFFUSION_EXPORT.md) - SDAR model export details
- [Implementation Notes](IMPLEMENTATION.md) - Development internals

### Planning
- [Quantization Plan](planning/QUANTIZATION_PLAN.md) - Quantization roadmap
- [Test Plan](planning/TEST_PLAN.md) - Testing strategy

---

## Documentation Structure

```
docs/
├── ARCHITECTURE.md          # System architecture & refactoring history
├── EXPORT.md                # Model export guide
├── CLI_USAGE.md             # CLI reference
├── BLOCK_DIFFUSION_EXPORT.md # SDAR-specific export
├── IMPLEMENTATION.md        # Implementation details
├── planning/
│   ├── QUANTIZATION_PLAN.md
│   └── TEST_PLAN.md
└── README.md                # This file
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
└─────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed information.
