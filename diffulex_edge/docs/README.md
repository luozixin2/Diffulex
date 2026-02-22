# DiffuLex Edge Documentation

Edge deployment toolkit for diffusion language models.

---

## Quick Start

```bash
# Install dependencies
pip install torch transformers

# Run CLI demo
python -m diffulex_edge

# Load and run a model
python -m diffulex_edge --model-path /path/to/model
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | System design, module structure, and architecture principles |
| [User Guide](USER_GUIDE.md) | Model export, CLI usage, and model-specific notes |
| [Quantization](QUANTIZATION.md) | FP16/INT8/INT4 quantization support |
| [Technical Details](TECHNICAL.md) | Block diffusion, KV cache, and static graph internals |
| [Test Plan](TEST_PLAN.md) | Testing strategy and coverage |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DiffuLex Edge                             │
├─────────────────────────────────────────────────────────────┤
│  Components (Shared)                                         │
│  - RMSNorm, RotaryEmbedding, SwiGLUMLP                      │
├─────────────────────────────────────────────────────────────┤
│  Models                                                      │
│  - SDAR, FastdLLMV2, Dream, LLaDA                          │
│  - Base: DiffusionModel, ModelConfig                        │
├─────────────────────────────────────────────────────────────┤
│  Runtime                                                     │
│  - DiffusionEngine, Samplers, KV Cache                      │
├─────────────────────────────────────────────────────────────┤
│  Export                                                      │
│  - XNNPACK, CoreML, QNN backends                            │
├─────────────────────────────────────────────────────────────┤
│  Quantization                                                │
│  - FP16, INT8, INT4 weight-only quantization                │
└─────────────────────────────────────────────────────────────┘
```

---

## Supported Models

| Model | Size | Mask Token ID | Attention |
|-------|------|---------------|-----------|
| SDAR | 1.7B | 126336 | Block Causal |
| Fast dLLM v2 | 1.5B | 151665 | Block Causal |
| Dream | 7B | 151666 | Bidirectional |
| LLaDA | 8B | 126336 | Bidirectional |

---

## Project Structure

```
diffulex_edge/
├── components/          # Shared transformer components
├── model/              # Model implementations
├── runtime/            # Inference engine and samplers
├── export/             # Export configuration
├── backends/           # Platform backends
├── quant/              # Quantization
└── docs/               # Documentation
```
