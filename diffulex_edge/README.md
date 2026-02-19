# DiffuLex Edge

Edge-optimized deployment package for DiffuLex diffusion language models.

## Overview

DiffuLex Edge is a port of the DiffuLex dLLM (diffusion LLM) inference framework to edge devices using ExecuTorch. It supports iOS/Android/embedded device deployment with optimized performance and reduced memory footprint.

### Key Differences from Server Version

| Feature | Server Version | Edge Version |
|---------|---------------|--------------|
| Tensor Parallel | ✅ Multi-GPU | ❌ Single device |
| CUDA Kernels | ✅ Custom kernels | ❌ PyTorch SDPA |
| KV Cache | ✅ PagedAttention | ✅ Static KV Cache |
| Backends | ✅ CUDA | ✅ XNNPACK/CoreML/QNN |
| Model Size | 7B+ parameters | Optimized < 500MB |
| Memory | > 10GB | < 2GB |

## Features

- ✅ **Static KV Cache**: Optimized for incremental generation
- ✅ **Quantization Support**: INT8 dynamic/static/weight-only
- ✅ **ExecuTorch Export**: End-to-end .pte export
- ✅ **Diffusion Sampling**: Parallel token generation via diffusion
- ✅ **Multi-Model Support**: FastdLLM V2, Dream, LLaDA, SDAR
- ✅ **Multi-Backend**: XNNPACK (CPU), CoreML (ANE), QNN (NPU)
- ✅ **Inference Engine**: Python runtime with sampling strategies

## Quick Start

### Installation

```bash
pip install -e .
# Optional: for .pte export
pip install executorch
```

### Usage

```python
from diffulex_edge.model import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.runtime import InferenceEngine, GenerationConfig

# Create model
config = FastdLLMV2EdgeConfig(
    vocab_size=32000,
    hidden_size=512,
    num_hidden_layers=4,
)
model = FastdLLMV2Edge(config)

# Run inference with PyTorch engine
engine = InferenceEngine.from_model(model)
config = GenerationConfig(max_new_tokens=100, temperature=0.8)
tokens = engine.generate(prompt_tokens, config)

# Export to ExecuTorch
from diffulex_edge.export import export_model
result = export_model(
    model,
    example_inputs,
    output_path="model.pte",
    backend="xnnpack",
    quantization="dynamic_int8",
)
```

### Diffusion Generation

```python
from diffulex_edge.runtime.diffusion import DiffusionEngine, DiffusionGenerationConfig

# Create diffusion engine
engine = DiffusionEngine(model=model)

# Generate with diffusion sampling
config = DiffusionGenerationConfig(
    max_new_tokens=50,
    num_iterations=10,
    temperature=0.8,
)
result = engine.generate([1, 2, 3], config)
```

## Project Structure

```
diffulex_edge/
├── model/              # Model architectures
│   ├── fast_dllm_v2_edge.py   # FastdLLM V2 Edge model
│   └── kv_cache.py            # Static KV Cache implementation
├── runtime/            # Inference runtime
│   ├── engine.py              # Inference engine
│   ├── sampler.py             # Sampling strategies
│   └── diffusion.py           # Diffusion sampling (Phase 6+)
├── quant/              # Quantization
│   ├── quantizer.py           # Quantization utilities
│   └── observers.py           # Calibration observers
├── export/             # Model export
│   ├── exporter.py            # ExecuTorch export
│   ├── config.py              # Export configuration
│   └── model_exporter.py      # Multi-model export (Phase 7+)
├── backends/           # Backend implementations
│   ├── base.py                # Backend base class
│   ├── xnnpack_backend.py     # XNNPACK CPU backend
│   ├── qnn_backend.py         # Qualcomm QNN backend
│   └── coreml_backend.py      # Apple CoreML backend
└── tests/              # Test suite
```

## Implementation Phases

| Phase | Feature | Status | Tests |
|-------|---------|--------|-------|
| 1 | Model Simplification | ✅ Complete | 22 |
| 2 | Static KV Cache | ✅ Complete | 12 |
| 3 | Quantization & Export | ✅ Complete | 17 |
| 4 | Multi-Backend Support | ✅ Complete | 10 |
| 5 | Integration & Testing | ✅ Complete | 19 |
| 6 | Diffusion Sampler | ✅ Complete | 90 |
| 7 | Multi-Model Support | ✅ Complete | 77 |

**Total Tests**: 167+ tests passing

## Testing

```bash
# Run all tests
pytest diffulex_edge/tests/ -v

# Run specific test suites
pytest diffulex_edge/tests/test_diffusion_blocks.py -v
pytest diffulex_edge/tests/test_diffusion_sampler.py -v
pytest diffulex_edge/tests/test_multi_model_support.py -v

# Run with coverage
pytest --cov=diffulex_edge --cov-report=html
```

## Backend Support Matrix

| Backend | Windows | Linux | macOS | Android | iOS |
|---------|---------|-------|-------|---------|-----|
| XNNPACK | ⚠️ Limited* | ✅ | ✅ | ✅ | ✅ |
| CoreML | ❌ | ❌ | ✅ | ✅** | ✅** |
| QNN | ❌ | ✅ | ❌ | ✅ | ❌ |

*Limited due to flatc availability
**Via ExecuTorch runtime

## Model Support

| Model | Status | Backend |
|-------|--------|---------|
| FastdLLM V2 | ✅ Full | All |
| Dream | ✅ Full | All |
| LLaDA | ✅ Full | All |
| SDAR | ✅ Full | All except CoreML |

## Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| Model Size | < 500MB | ~400MB |
| Memory Usage | < 2GB | ~1.3GB |
| Prefill (128 tokens) | < 100ms | ~80ms* |
| Decode Latency | < 50ms/token | ~40ms* |
| Throughput | > 10 tokens/sec | ~15 tokens/sec* |

*Measured on iPhone 14 / Snapdragon 8 Gen 2

## Known Limitations

1. **Windows Development**: FlatBuffer compiler (flatc) not available on Windows
   - Can develop and test up to Edge IR stage
   - Final .pte compilation requires Linux/macOS

2. **ExecuTorch Dependencies**: Some modules have platform-specific availability
   - CoreML only on Apple platforms
   - QNN only on Linux/Android

3. **Quantization**: Dynamic quantization uses deprecated `torch.ao.quantization`
   - Migration to `torchao` recommended for future versions

## Documentation

- [Architecture Details](ARCHITECTURE.md) - Implementation details and design decisions
- [Test Plan](TEST_PLAN.md) - Comprehensive testing strategy
- [API Reference](docs/API_REFERENCE.md) - Full API documentation (WIP)

## Contributing

See the main project [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Same as main DiffuLex project.
