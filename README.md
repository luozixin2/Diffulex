# DiffuLex Edge

Edge-optimized diffusion LLM inference framework based on ExecuTorch.

[![Tests](https://img.shields.io/badge/tests-87%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Overview

DiffuLex Edge is a simplified version of the DiffuLex dLLM (diffusion language model) inference framework, designed for deployment on edge devices including iOS, Android, and embedded systems.

### Key Differences from Server Version

| Feature | Server Version | Edge Version |
|---------|---------------|--------------|
| Tensor Parallel | ✅ Multi-GPU | ❌ Single device |
| Attention | Flash Attention (CUDA) | PyTorch SDPA |
| KV Cache | PagedAttention | Static KV Cache |
| Quantization | GPTQ/AWQ/Marlin | XNNPACK/QNN 8-bit |
| Runtime | vLLM | ExecuTorch |

## Project Status

### Implementation Progress

| Phase | Status | Tests |
|-------|--------|-------|
| Phase 1: Model Simplification | ✅ Complete | 34/34 |
| Phase 2: Runtime Implementation | ✅ Complete | 17/17 |
| Phase 3: Quantization | ✅ Complete | 17/17 |
| Phase 4: Multi-Backend Support | ✅ Complete | 10/10 |
| Phase 5: Integration & Testing | 🟡 In Progress | Ongoing |

**Total: 87 tests passing, 0 failures**

### Supported Backends

| Backend | Platform | Status |
|---------|----------|--------|
| XNNPACK | ARM64/x86 CPU | ✅ Ready |
| CoreML | Apple Neural Engine | ✅ Ready (macOS/iOS) |
| QNN | Qualcomm NPU | ✅ Ready (Android) |

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch executorch

# Optional: Backend-specific dependencies
pip install coremltools  # For CoreML backend
```

### Basic Usage

```python
from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig

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
engine = InferenceEngine.from_model(model)
tokens = engine.generate(
    prompt_tokens=[1, 2, 3],
    config=GenerationConfig(max_new_tokens=20)
)
```

### Model Export

```python
from diffulex_edge.backends import XNNPACKBackend, BackendConfig

# Export with XNNPACK backend
backend = XNNPACKBackend(BackendConfig(
    quantize=True,
    quantization_mode="weight_only"
))

result = backend.export(model, example_inputs)
if result.success:
    with open("model.pte", "wb") as f:
        f.write(result.buffer)
```

## Project Structure

```
diffulex_edge/
├── model/              # Simplified model implementation
│   ├── fast_dllm_v2_edge.py   # Edge model with KV cache
│   └── kv_cache.py            # Static KV cache
├── runtime/            # Inference runtime
│   ├── engine.py              # Inference engine
│   └── sampler.py             # Sampling strategies
├── export/             # Model export
│   ├── exporter.py            # ExecuTorch exporter
│   └── config.py              # Export configuration
├── quant/              # Quantization
│   ├── quantizer.py           # PT2E quantizer
│   └── observers.py           # Quantization observers
├── backends/           # Backend implementations
│   ├── base.py                # Backend abstraction
│   ├── xnnpack_backend.py     # XNNPACK CPU backend
│   ├── qnn_backend.py         # Qualcomm QNN backend
│   └── coreml_backend.py      # Apple CoreML backend
└── tests/              # Test suite
    ├── test_model_simplified.py
    ├── test_kv_cache.py
    ├── test_engine.py
    ├── test_export.py
    ├── test_quantization.py
    ├── test_backends.py
    └── integration/
        └── test_full_pipeline.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
├─────────────────────────────────────────────────────────────┤
│                   DiffuLex Edge Runtime                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Tokenizer  │  │   Sampler   │  │   KV Cache Manager  │  │
│  │  (Hugging   │  │ (Greedy/    │  │    (Static)         │  │
│  │   Face)     │  │  Top-k/p)   │  │                     │  │
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

## Features

### Model Features
- ✅ Simplified transformer architecture
- ✅ Grouped Query Attention (GQA)
- ✅ Rotary Position Embedding (RoPE)
- ✅ RMSNorm layer normalization
- ✅ SwiGLU activation in FFN
- ✅ Static KV Cache for incremental inference

### Runtime Features
- ✅ PyTorch eager mode inference
- ✅ ExecuTorch runtime support
- ✅ Multiple sampling strategies (Greedy, Top-K, Top-P)
- ✅ Temperature scaling
- ✅ Repetition penalty
- ✅ Stop sequences

### Export Features
- ✅ Multi-backend export (XNNPACK, CoreML, QNN)
- ✅ Dynamic INT8 quantization
- ✅ Static INT8 quantization
- ✅ Weight-only quantization
- ✅ FP16 casting

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest diffulex_edge/tests/ -v

# Run specific test module
python -m pytest diffulex_edge/tests/test_model_simplified.py -v
python -m pytest diffulex_edge/tests/test_backends.py -v
```

## Examples

See the `examples/` directory for complete usage examples:

- `edge_inference_example.py` - End-to-end inference demo
- `export_model.py` - Command-line export tool

## Roadmap

- [x] Model simplification and basic architecture
- [x] Static KV Cache implementation
- [x] PyTorch SDPA attention
- [x] Inference engine with generation
- [x] XNNPACK backend support
- [x] CoreML backend support
- [x] QNN backend support
- [x] Quantization (dynamic, static, weight-only)
- [ ] End-to-end benchmark suite
- [ ] Mobile deployment examples (iOS/Android)
- [ ] Model compression techniques

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This project is based on the DiffuLex dLLM framework, adapted for edge deployment using Meta's ExecuTorch runtime.
