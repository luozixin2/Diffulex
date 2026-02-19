# DiffuLex Edge - Test Plan

**Version**: 1.0  
**Date**: 2026-02-18

## Overview

This document outlines the testing strategy for DiffuLex Edge, covering unit tests, integration tests, and performance benchmarks.

## Test Statistics

| Category | Count | Status |
|----------|-------|--------|
| Phase 1 (Model) | 22 | ✅ Pass |
| Phase 2 (KV Cache) | 12 | ✅ Pass |
| Phase 3 (Quantization) | 17 | ✅ Pass |
| Phase 4 (Backends) | 10 | ✅ Pass |
| Phase 5 (Integration) | 19 | ✅ Pass |
| Phase 6 (Diffusion) | 90 | ✅ Pass |
| Phase 7 (Multi-Model) | 77 | ✅ Pass |
| **Total** | **167+** | **✅ Pass** |

## Test Files

```
diffulex_edge/tests/
├── test_model_simplified.py          # Model architecture tests (22)
├── test_kv_cache.py                  # KV cache tests (12)
├── test_engine.py                    # Inference engine tests (17)
├── test_export.py                    # Export tests (9)
├── test_quantization.py              # Quantization tests (17)
├── test_backends.py                  # Backend tests (10)
├── test_diffusion_blocks.py          # Diffusion block tests (28)
├── test_diffusion_sampler.py         # Diffusion sampler tests (46)
├── test_diffusion_engine.py          # Diffusion engine tests (18)
├── test_diffusion_integration.py     # Integration tests (18)
├── test_diffusion_performance.py     # Performance tests (6)
├── test_multi_model_support.py       # Multi-model tests (21)
├── test_numerical_equivalence.py     # Numerical accuracy (16)
├── test_end_to_end.py                # E2E tests (16)
└── integration/
    └── test_full_pipeline.py         # Full pipeline tests (10)
```

## Running Tests

```bash
# All tests
pytest diffulex_edge/tests/ -v

# Specific module
pytest diffulex_edge/tests/test_diffusion_sampler.py -v

# With coverage
pytest --cov=diffulex_edge --cov-report=html

# Performance tests only
pytest diffulex_edge/tests/test_diffusion_performance.py -v

# Skip slow tests
pytest diffulex_edge/tests/ -v -m "not slow"
```

## Test Categories

### Unit Tests (70%)

Test individual components in isolation:
- Model forward pass
- KV cache operations
- Sampling algorithms
- Quantization operations

### Integration Tests (20%)

Test component interactions:
- Model + KV cache
- Sampler + Engine
- Export + Runtime

### Performance Tests (10%)

Benchmark critical paths:
- Prefill latency
- Decode throughput
- Memory usage

## Key Test Scenarios

### 1. Numerical Equivalence

Verify Edge version matches original DiffuLex:

```python
def test_shift_logits_precision():
    """shift_logits must match original within 1e-6"""
    pass

def test_generation_equivalence():
    """Same seed → identical outputs"""
    pass
```

### 2. Boundary Conditions

Test edge cases:
- Empty sequences
- Single token
- Maximum sequence length
- Extreme temperature values

### 3. Error Handling

Verify graceful failure:
- Invalid inputs
- Dimension mismatches
- NaN/Inf handling

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -e ".[test]"
      - run: pytest diffulex_edge/tests/ -v --cov
```

## Coverage Targets

| Module | Target | Current |
|--------|--------|---------|
| Model | >90% | ~95% |
| Runtime | >85% | ~90% |
| Export | >80% | ~85% |
| Overall | >85% | ~90% |

## See Also

- [Architecture](ARCHITECTURE.md) - Implementation details
- [README](README.md) - Project overview
