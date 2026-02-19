# DiffuLex Edge - Agent Guide

This guide provides essential information for AI agents working on the DiffuLex Edge project.

## Quick Navigation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview, quick start, features |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical implementation details |
| [TEST_PLAN.md](TEST_PLAN.md) | Testing strategy and running tests |

## Project Structure

```
diffulex_edge/
├── README.md              # Main documentation
├── ARCHITECTURE.md        # Technical architecture
├── TEST_PLAN.md           # Testing documentation
├── AGENTS.md             # This file
├── __init__.py           # Package init
├── model/                # Model implementations
│   ├── fast_dllm_v2_edge.py
│   └── kv_cache.py
├── runtime/              # Inference runtime
│   ├── engine.py
│   ├── sampler.py
│   └── diffusion.py      # Diffusion sampling
├── quant/                # Quantization
│   ├── quantizer.py
│   └── observers.py
├── export/               # Model export
│   ├── exporter.py
│   ├── config.py
│   └── model_exporter.py
├── backends/             # Backend implementations
│   ├── base.py
│   ├── xnnpack_backend.py
│   ├── qnn_backend.py
│   └── coreml_backend.py
├── scripts/              # CLI tools
│   └── export_model.py
└── tests/                # Test suite
    ├── test_*.py
    └── integration/
```

## Key Implementation Notes

### 1. Model Architecture

- **FastdLLMV2Edge**: Simplified model without tensor parallel
- Uses `F.scaled_dot_product_attention` instead of custom CUDA kernels
- Static KV cache with input/output mode for ExecuTorch compatibility

### 2. Diffusion Sampling (Phase 6+)

Core files:
- `runtime/diffusion.py` - DiffusionSampler, DiffusionBlockManager

Key concepts:
- Diffusion blocks divide sequence into chunks
- Shift logits operation for dependency handling
- Confidence-based token acceptance (threshold 0.95)
- Iterative denoising process

### 3. Multi-Model Support (Phase 7+)

Core files:
- `export/model_exporter.py` - ModelExporterFactory, per-model exporters

Supported models:
- FastdLLM V2
- Dream
- LLaDA
- SDAR

### 4. Backend System

All backends inherit from `EdgeBackend` base class:
- XNNPACKBackend - Generic CPU
- QNNBackend - Qualcomm NPU
- CoreMLBackend - Apple Neural Engine

### 5. Testing

**Always run tests before submitting changes:**

```bash
# Quick sanity check
pytest diffulex_edge/tests/test_diffusion_blocks.py -v

# Full test suite
pytest diffulex_edge/tests/ -v --tb=short

# With coverage
pytest --cov=diffulex_edge --cov-report=html
```

**Current test count**: 167+ tests passing

## Common Tasks

### Adding a New Model

1. Create model class in `model/{model_name}_edge.py`
2. Create sampler in `runtime/diffusion.py` or separate file
3. Add exporter in `export/model_exporter.py`
4. Register in `ModelExporterFactory`
5. Add tests in `tests/test_multi_model_support.py`

### Adding a New Backend

1. Create backend class in `backends/{name}_backend.py`
2. Inherit from `EdgeBackend`
3. Implement `export()` and `load()` methods
4. Register in `BackendRegistry`
5. Add tests in `tests/test_backends.py`

### Modifying Diffusion Sampler

1. Edit `runtime/diffusion.py`
2. Ensure `shift_logits()` maintains numerical precision
3. Update `compute_confidence()` for new confidence methods
4. Run diffusion tests:
   ```bash
   pytest diffulex_edge/tests/test_diffusion_*.py -v
   ```

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings for public methods
- Keep functions focused and small

## Platform Considerations

**Windows Development**:
- Can develop and test up to Edge IR stage
- Cannot generate .pte files (flatc not available)
- Use WSL2 or remote Linux for full export testing

**Linux/macOS**:
- Full development environment
- Can generate .pte files
- Required for backend testing (QNN on Linux, CoreML on macOS)

## Dependencies

Core:
- torch >= 2.3.0
- numpy >= 1.24.0

Optional:
- executorch >= 0.3.0 (for .pte export)
- transformers >= 4.40.0 (for tokenizer)

Backend-specific:
- executorch[xnnpack]
- executorch[qnn]
- executorch[coreml]

## Troubleshooting

### Import Errors

```bash
# Ensure package is installed in editable mode
pip install -e .
```

### ExecuTorch Not Available

```bash
# Install ExecuTorch
pip install executorch

# Or run without export tests
pytest diffulex_edge/tests/ -v --ignore=diffulex_edge/tests/test_export.py
```

### Test Failures

1. Check Python version (3.10+ recommended)
2. Update dependencies: `pip install -e . -U`
3. Run with verbose output: `pytest -vvv`

## Resources

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [ExecuTorch LLM Example](https://github.com/pytorch/executorch/tree/main/examples/models/llama)
- Main project: See root README.md

## Contact

For questions about the codebase, refer to:
- Architecture questions → ARCHITECTURE.md
- Testing questions → TEST_PLAN.md
- General usage → README.md
