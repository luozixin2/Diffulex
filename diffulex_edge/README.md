# DiffuLex Edge

Edge-optimized diffusion LLM inference framework based on ExecuTorch.

[![Tests](https://img.shields.io/badge/tests-200%2B%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()

## Overview

DiffuLex Edge is a simplified version of the DiffuLex dLLM (diffusion language model) inference framework, designed for deployment on edge devices including iOS, Android, and embedded systems.

### Key Differences from Server Version

| Feature | Server Version | Edge Version |
|---------|---------------|--------------|
| Tensor Parallel | ✅ Multi-GPU | ❌ Single device |
| Attention | Flash Attention (CUDA) | PyTorch SDPA |
| KV Cache | PagedAttention | Static KV Cache |
| Quantization | GPTQ/AWQ/Marlin | XNNPACK/QNN 8-bit/4-bit |
| Runtime | vLLM | ExecuTorch |

### Key Features

- **Multi-Model Support**: FastdLLM V2, Dream, LLaDA, SDAR
- **Static KV Cache**: Optimized for incremental generation
- **Quantization**: FP16/INT8/INT4 with dynamic/static/weight-only modes
- **ExecuTorch Export**: End-to-end .pte export
- **Diffusion Sampling**: Parallel token generation via diffusion
- **Multi-Backend**: XNNPACK (CPU), CoreML (ANE), QNN (NPU)

## Project Status

| Phase | Status | Tests |
|-------|--------|-------|
| Phase 1: Model Architecture | ✅ Complete | 22/22 |
| Phase 2: Quantization | ✅ Complete | 17/17 |
| Phase 3: Export | ✅ Complete | 11/11 |
| **Total** | **✅ Complete** | **50/50** |

### Supported Quantization

| Type | Status | Compression |
|------|--------|-------------|
| FP32 (Baseline) | ✅ Ready | 1.0x |
| FP16 | ✅ Ready | ~1.7x |
| INT8 Weight-only | ✅ Ready | ~2.5x |
| INT4 Weight-only | ✅ Ready | ~4.0x |

## Quick Start

### Installation

```bash
pip install -e .
# Optional: for .pte export
pip install executorch
# Optional: for INT4 quantization
pip install torchao
```

### Basic Usage

```python
from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.runtime import DiffusionEngine, DiffusionGenerationConfig

# Create model
config = FastdLLMV2EdgeConfig(
    vocab_size=32000,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=4,  # GQA
)
model = FastdLLMV2Edge(config)

# Run inference
engine = DiffusionEngine.from_model(model, model_type="fast_dllm_v2")
tokens = engine.generate(
    prompt_tokens=[1, 2, 3],
    config=DiffusionGenerationConfig(max_new_tokens=20)
)
```

### Model Export

```bash
# Export HuggingFace model to ExecuTorch
python -m diffulex_edge.scripts.export_model \
  --hf-model-path /path/to/hf/model \
  --output-path model.pte \
  --backend xnnpack \
  --dtype int8
```

### CLI Usage

```bash
# Interactive chat with PTE model
python -m diffulex_edge --pte-path model.pte --tokenizer gpt2
```

## Project Structure

```
diffulex_edge/
├── components/         # Shared components (RMSNorm, RoPE, MLP)
├── model/              # Model implementations (4 models)
│   ├── base.py         # Base classes
│   ├── wrapper.py      # Export wrappers
│   ├── sdar_edge.py    # SDAR model
│   ├── fast_dllm_v2_edge.py
│   ├── dream_edge.py
│   └── llada_edge.py
├── runtime/            # Inference runtime
│   ├── engine.py       # DiffusionEngine
│   ├── diffusion.py    # Diffusion components
│   └── sampler/        # Sampling strategies
├── export/             # Model export
├── backends/           # Backend implementations
│   ├── base.py
│   ├── xnnpack_backend.py
│   ├── coreml_backend.py
│   └── qnn_backend.py
├── quant/              # Quantization support
│   ├── base.py         # BaseQuantizer, configs
│   ├── core_quant.py   # QuantizedLinear
│   ├── fp16_quantizer.py
│   ├── int8_quantizer.py
│   └── int4_quantizer.py
└── tests/              # Test suite (200+ tests)
```

## Model Support

| Model | Status | Shift Logits | Per-block Threshold | from_pte |
|-------|--------|--------------|---------------------|----------|
| FastdLLM V2 | ✅ Full | ✅ | ❌ | ✅ |
| Dream | ✅ Full | ✅ | ✅ | ✅ |
| LLaDA | ✅ Full | ❌ | ✅ | ✅ |
| SDAR | ✅ Full | ✅ | ❌ | ✅ |

## Backend Support

| Backend | Linux | macOS | Android | iOS |
|---------|-------|-------|---------|-----|
| XNNPACK | ✅ | ✅ | ✅ | ✅ |
| CoreML | ❌ | ✅ | ✅** | ✅** |
| QNN | ✅ | ❌ | ✅ | ❌ |

**Via ExecuTorch runtime

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
├─────────────────────────────────────────────────────────────┤
│                   DiffuLex Edge Runtime                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Tokenizer  │  │   Sampler   │  │   KV Cache Manager  │  │
│  │  (Hugging   │  │ (Diffusion) │  │    (Static)         │  │
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
│                    Backend Execution Engine                  │
│     XNNPACK (CPU)    │    QNN (Qualcomm)   │  CoreML (ANE)  │
└─────────────────────────────────────────────────────────────┘
```

## Testing

```bash
# Run all tests
pytest diffulex_edge/tests/ -v

# Run with coverage
pytest --cov=diffulex_edge --cov-report=html
```

**Test Count**: 200+ tests passing

## Documentation

- [User Guide](docs/USER_GUIDE.md) - Export and CLI usage
- [Architecture](docs/ARCHITECTURE.md) - System architecture and design
- [Quantization](docs/QUANTIZATION.md) - Quantization support details
- [Test Plan](docs/TEST_PLAN.md) - Testing strategy and coverage
- [Runtime README](runtime/README.md) - Runtime usage guide

## Known Limitations

1. **INT4**: Requires both `torchao` and `fbgemm-gpu-genai >= 1.2.0`
2. **CoreML**: Apple platforms only
3. **QNN**: Linux/Android only

## Roadmap

- [x] Model simplification and basic architecture
- [x] Static KV Cache implementation
- [x] PyTorch SDPA attention
- [x] Inference engine with generation
- [x] XNNPACK backend support
- [x] CoreML backend support
- [x] QNN backend support
- [x] Quantization (FP16, INT8, INT4)
- [x] Multi-model support (4 models)
- [ ] End-to-end benchmark suite
- [ ] Mobile deployment examples (iOS/Android)
- [ ] Model compression techniques

## License

Same as main DiffuLex project.
