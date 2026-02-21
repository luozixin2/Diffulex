# DiffuLex Edge - Test Plan

Testing strategy and coverage for DiffuLex Edge.

---

## Test Structure

```
diffulex_edge/tests/
├── test_model_numerical_equivalence.py   # HF vs Edge numerical alignment
├── test_sampler_alignment.py             # Sampler algorithm alignment
├── test_model_simplified.py              # Model architecture tests
├── test_quantization.py                  # Quantization tests
├── test_export.py                        # Export functionality
└── conftest.py                           # Shared fixtures
```

---

## Running Tests

```bash
# All tests
pytest diffulex_edge/tests/ -v

# Specific test file
pytest diffulex_edge/tests/test_sampler_alignment.py -v

# With coverage
pytest --cov=diffulex_edge --cov-report=html

# Skip slow tests
pytest diffulex_edge/tests/ -v -m "not slow"
```

---

## Key Test Categories

### 1. Numerical Equivalence Tests

Verify Edge models produce identical outputs to HuggingFace implementations.

```bash
python diffulex_edge/tests/test_model_numerical_equivalence.py
```

**Coverage:**
- Hidden layer outputs (layers 0-27)
- LM head outputs
- End-to-end generation

### 2. Sampler Alignment Tests

Verify sampling algorithms match between diffulex (main) and diffulex_edge.

```bash
python diffulex_edge/tests/test_sampler_alignment.py
```

**Coverage:**
- Dream/SDAR/FastDLLM samplers
- Token acceptance logic
- Block completion handling

### 3. Model Tests

Test model architecture and forward pass.

```python
class TestFastdLLMV2Edge:
    def test_model_creation()
    def test_forward_pass()
    def test_generation()
```

### 4. Quantization Tests

```python
class TestFP16Quantization:
    def test_fp16_compression_ratio()      # ~1.7x

class TestINT8Quantization:
    def test_int8_compression_ratio()      # ~2.5x
```

---

## Coverage Targets

| Module | Target | Status |
|--------|--------|--------|
| Model | >90% | ✅ ~95% |
| Runtime | >85% | ✅ ~90% |
| Sampler | >90% | ✅ ~95% |
| Export | >80% | ✅ ~85% |

---

## Adding New Tests

```python
# tests/test_feature.py
import torch
import pytest
from diffulex_edge import feature

class TestFeatureName:
    """Test feature description"""
    
    def test_basic_functionality(self):
        input = torch.tensor([1, 2, 3])
        result = feature.process(input)
        assert result.shape == expected_shape
    
    def test_edge_case(self):
        with pytest.raises(ValueError):
            feature.process(invalid_input)
    
    @pytest.mark.skipif(not HAS_DEPS, reason="Optional deps not installed")
    def test_optional_feature(self):
        pass
```

---

## See Also

- [Architecture](ARCHITECTURE.md) - Implementation details
- [User Guide](USER_GUIDE.md) - Usage guide
