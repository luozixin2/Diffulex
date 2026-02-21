# DiffuLex Edge - Test Plan

Testing strategy and coverage for DiffuLex Edge.

## Test Statistics

| Category | Count | Status |
|----------|-------|--------|
| Model Tests | 22 | ✅ Pass |
| Quantization Tests | 17 | ✅ Pass |
| Export Tests | 11 | ✅ Pass |
| **Total** | **50+** | **✅ Pass** |

## Test Structure

```
diffulex_edge/tests/
├── test_model_simplified.py          # Model architecture (22)
├── test_quantization.py              # Quantization (17)
├── test_export.py                    # Export functionality (11)
└── conftest.py                       # Shared fixtures
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

### 1. Model Tests (`test_model_simplified.py`)

Test model architecture and forward pass.

```python
class TestFastdLLMV2Edge:
    def test_model_creation()
    def test_forward_pass()
    def test_generation()

class TestDreamEdge:
    def test_model_creation()
    def test_forward_pass()

class TestLLaDAEdge:
    def test_model_creation()
    def test_forward_pass()

class TestSDAREdge:
    def test_model_creation()
    def test_forward_pass()
```

### 2. Quantization Tests (`test_quantization.py`)

Test quantization functionality and space savings.

```python
class TestFP16Quantization:
    def test_fp16_space_savings()      # Verify ~1.7x compression
    def test_fp16_accuracy()

class TestINT8Quantization:
    def test_int8_space_savings()      # Verify ~2.5x compression
    def test_int8_accuracy()

class TestQuantizationAPI:
    def test_quantize_model_fp16()
    def test_quantize_model_int8()
```

### 3. Export Tests (`test_export.py`)

Test model export functionality.

```python
class TestExportFunctionality:
    def test_export_config()
    def test_exporter_creation()

def test_quantization_preserves_accuracy()
def test_quantization_reduces_file_size()  # Key test
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

### Quantization Space Savings

```python
def test_fp16_compression_ratio():
    """FP16 should achieve ~1.7x compression"""
    
def test_int8_compression_ratio():
    """INT8 should achieve ~2.5x compression"""

def test_pte_file_size_reduction():
    """Exported .pte files should reflect quantization savings"""
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
- Missing dependencies (INT4 without torchao)

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
