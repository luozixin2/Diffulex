# DiffuLex Edge

Edge-optimized deployment package for DiffuLex diffusion language models.

## Overview

DiffuLex Edge is a port of the DiffuLex dLLM (diffusion LLM) inference framework to edge devices using ExecuTorch. It supports iOS/Android/embedded device deployment with optimized performance and reduced memory footprint.

### Key Features

- **Multi-Model Support**: FastdLLM V2, Dream, LLaDA, SDAR
- **Static KV Cache**: Optimized for incremental generation
- **Quantization**: INT8 dynamic/static/weight-only
- **ExecuTorch Export**: End-to-end .pte export
- **Diffusion Sampling**: Parallel token generation via diffusion
- **Multi-Backend**: XNNPACK (CPU), CoreML (ANE), QNN (NPU)

## Quick Start

### Installation

```bash
pip install -e .
# Optional: for .pte export
pip install executorch
```

### Export Model to PTE

```bash
# Export HuggingFace model to ExecuTorch
python -m diffulex_edge.scripts.export_model \
  --hf-model-path /path/to/hf/model \
  --output-path model.pte \
  --backend xnnpack
```

### Run Inference

```python
from diffulex_edge.runtime import InferenceEngine, GenerationConfig

# Load PTE model
engine = InferenceEngine.from_pte("model.pte")

# Generate tokens
config = GenerationConfig(max_new_tokens=100, temperature=0.8)
tokens = engine.generate([1, 2, 3], config)
```

### Diffusion Generation

```python
from diffulex_edge.runtime import DiffusionEngine, DiffusionGenerationConfig

# Load with specific model type
engine = DiffusionEngine.from_pte("model.pte", model_type="fast_dllm_v2")

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
├── model/              # Model architectures (4 models)
├── runtime/            # Inference runtime
│   ├── engine.py       # InferenceEngine (autoregressive)
│   ├── diffusion.py    # DiffusionEngine (diffusion)
│   └── sampler/        # Sampling strategies
├── export/             # Model export
├── backends/           # Backend implementations
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

## Testing

```bash
# Run all tests
pytest diffulex_edge/tests/ -v

# Run with coverage
pytest --cov=diffulex_edge --cov-report=html
```

**Test Count**: 200+ tests passing

## Documentation

- [Architecture Details](docs/ARCHITECTURE.md)
- [Test Plan](docs/TEST_PLAN.md)
- [Export Guide](docs/EXPORT.md)
- [Agent Guide](AGENTS.md)

## Known Limitations

1. **Windows**: Cannot generate .pte files (flatc not available). Use WSL2.
2. **CoreML**: Apple platforms only
3. **QNN**: Linux/Android only

## License

Same as main DiffuLex project.
