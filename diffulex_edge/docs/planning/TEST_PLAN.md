# DiffuLex Edge - Test Plan

Testing strategy and coverage for DiffuLex Edge.

## Test Statistics

| Category | Count | Status |
|----------|-------|--------|
| Model Tests | 22 | ✅ Pass |
| KV Cache Tests | 12 | ✅ Pass |
| Quantization Tests | 17 | ✅ Pass |
| Sampler Alignment | 38 | ✅ Pass |
| Diffusion Engine | 37 | ✅ Pass |
| PTE Tests | 37 | ✅ Pass |
| Multi-Model Tests | 21 | ✅ Pass |
| Integration Tests | 31 | ✅ Pass |
| **Total** | **200+** | **✅ Pass** |

## Test Structure

```
diffulex_edge/tests/
├── test_model_simplified.py          # Model architecture (22)
├── test_kv_cache.py                  # KV cache operations (12)
├── test_engine.py                    # Inference engine (17)
├── test_quantization.py              # Quantization (17)
├── test_sampler_alignment.py         # Sampler alignment (38)
├── test_diffusion_blocks.py          # Diffusion blocks (28)
├── test_diffusion_sampler.py         # Diffusion sampling (46)
├── test_diffusion_engine.py          # Diffusion engine (18)
├── test_diffusion_pte.py             # PTE diffusion (37)
├── test_diffusion_integration.py     # Integration (18)
├── test_multi_model_support.py       # Multi-model (21)
├── test_numerical_equivalence.py     # Numerical accuracy (16)
├── test_pte_*.py                     # PTE tests (60+)
├── test_end_to_end.py                # E2E tests (16)
└── integration/                      # Integration tests
```

## Running Tests

```bash
# All tests
pytest diffulex_edge/tests/ -v

# Specific module
pytest diffulex_edge/tests/test_sampler_alignment.py -v

# With coverage
pytest --cov=diffulex_edge --cov-report=html

# Skip slow tests
pytest diffulex_edge/tests/ -v -m "not slow"

# Parallel execution
pytest diffulex_edge/tests/ -n auto
```

## Test Categories

### 1. Sampler Alignment Tests (`test_sampler_alignment.py`)

Verify Edge sampler matches original diffulex behavior.

```python
class TestTopPLogits:
    """Align with SamplerBase.top_p_logits"""
    def test_top_p_filtering_basic()
    def test_top_p_keeps_at_least_one()
    def test_top_p_batch_processing()

class TestShiftLogitsSampler:
    """Align with SamplerShiftLogits"""
    def test_shift_with_last_logit()
    def test_shift_with_caching()
    
class TestFastdLLMV2Sampler:
    """Align with FastdLLMV2SamplerForDiffusionLM"""
    def test_always_accepts_one()
    def test_respects_threshold()

class TestLLaDASampler:
    """Align with LLaDASamplerForDiffusionLM"""
    def test_no_shift()
    def test_pre_block_complete_logic()
    def test_per_block_threshold()
```

### 2. PTE Tests (`test_diffusion_pte.py`)

Test DiffusionEngine PTE support.

```python
class TestDiffusionEnginePTE:
    """DiffusionEngine with PTE models"""
    def test_pte_loading()
    def test_pte_forward()
    def test_pte_vs_pytorch_consistency()  # Numerical match
    def test_pte_generate()
    def test_pte_error_handling()
```

### 3. Numerical Equivalence Tests (`test_numerical_equivalence.py`)

Compare HF and Edge model outputs.

```python
def test_forward_outputs_match()      # Same input -> same output
def test_kv_cache_consistency()       # Cache update correctness
def test_rotary_embedding_match()     # RoPE alignment
```

### 4. Multi-Model Tests (`test_multi_model_support.py`)

Test all 4 model types.

```python
def test_fast_dllm_v2_forward()
def test_dream_forward()
def test_llada_forward()
def test_sdar_forward()
def test_model_registry()
```

## Coverage Targets

| Module | Target | Current |
|--------|--------|---------|
| Model | >90% | ~95% |
| Runtime | >85% | ~90% |
| Export | >80% | ~85% |
| Sampler | >90% | ~95% |
| **Overall** | **>85%** | **~90%** |

## Key Test Scenarios

### Numerical Equivalence

```python
def test_shift_logits_precision():
    """shift_logits must match original within 1e-6"""
    
def test_generation_equivalence():
    """Same seed → identical outputs"""
```

### Boundary Conditions

- Empty sequences
- Single token
- Maximum sequence length
- Extreme temperature values
- All-mask sequences

### Error Handling

- Invalid inputs
- Dimension mismatches
- NaN/Inf handling
- Missing files
- Invalid PTE files

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[test]"
      - run: pytest diffulex_edge/tests/ -v --cov
```

## Adding New Tests

### Pattern

```python
# tests/test_feature.py
import torch
import pytest
from diffulex_edge import feature

class TestFeatureName:
    """Test feature description"""
    
    def test_basic_functionality(self):
        """Test basic case"""
        input = torch.tensor([1, 2, 3])
        result = feature.process(input)
        assert result.shape == expected_shape
    
    def test_edge_case(self):
        """Test edge case"""
        with pytest.raises(ValueError):
            feature.process(invalid_input)
    
    @pytest.mark.skipif(not HAS_DEPS, reason="Optional deps not installed")
    def test_optional_feature(self):
        """Test requiring optional dependencies"""
        pass
```

## See Also

- [Architecture](ARCHITECTURE.md) - Implementation details
- [README](../README.md) - Project overview
