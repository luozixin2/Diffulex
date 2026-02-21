# DiffuLex Edge - Quantization Support

Quantization implementation for edge deployment with FP16, INT8, and INT4 support.

## Overview

DiffuLex Edge provides flexible precision options to balance model accuracy, size, and inference speed across different edge devices.

## Supported Quantization Types

| Type | Status | Backend | Compression | Notes |
|------|--------|---------|-------------|-------|
| FP32 (Baseline) | ✅ Ready | All | 1.0x | Reference implementation |
| FP16 | ✅ Ready | XNNPACK, CoreML | ~1.7x | Simple weight cast to FP16 |
| INT8 Weight-only | ✅ Ready | All | ~2.5x | Per-channel symmetric quantization |
| INT4 Weight-only | ✅ Ready | XNNPACK | ~4.0x | Requires torchao + fbgemm-gpu-genai |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Quantization Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Quantizer Facade                                │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │   │
│  │  │   INT8      │  │    FP16      │  │      INT4        │   │   │
│  │  │  Quantizer  │  │  Quantizer   │  │   Quantizer      │   │   │
│  │  │  (native)   │  │   (native)   │  │  (torchao)       │   │   │
│  │  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘   │   │
│  └─────────┼────────────────┼───────────────────┼─────────────┘   │
│            │                │                   │                  │
│            ▼                ▼                   ▼                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Backend Abstraction Layer                       │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │   │
│  │  │ XNNPACK  │ │  CoreML  │ │   QNN    │                     │   │
│  │  └──────────┘ └──────────┘ └──────────┘                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Usage

### FP16 Quantization

```python
from diffulex_edge.quant import FP16Quantizer, QuantizationConfig, QuantizationDtype

# Create quantizer
quantizer = FP16Quantizer()

# Quantize model
config = QuantizationConfig(
    dtype=QuantizationDtype.FP16,
    exclude_layers=["lm_head", "norm"]  # Optional
)
result = quantizer.quantize(model, config)

print(f"Memory reduction: {result.memory_reduction_pct:.1f}%")
print(f"Compression ratio: {result.compression_ratio:.2f}x")
```

### INT8 Weight-Only Quantization

```python
from diffulex_edge.quant import INT8Quantizer, QuantizationConfig, QuantizationDtype

# Create quantizer
quantizer = INT8Quantizer()

# Quantize model
config = QuantizationConfig(
    dtype=QuantizationDtype.INT8,
    mode="weight_only",
    exclude_layers=["lm_head"]
)
result = quantizer.quantize(model, config)
```

### INT4 Weight-Only Quantization (torchao)

```python
from diffulex_edge.quant import INT4Quantizer, QuantizationConfig, QuantizationDtype

# Check availability
from diffulex_edge.quant import is_int4_available
if not is_int4_available():
    print("torchao not installed, INT4 unavailable")
    return

# Create quantizer
quantizer = INT4Quantizer()

# Quantize model with group size
config = QuantizationConfig(
    dtype=QuantizationDtype.INT4,
    mode="weight_only",
    group_size=32  # 32, 64, 128, or 256
)
result = quantizer.quantize(model, config)
```

### Backend Integration

```python
from diffulex_edge.backends import XNNPACKBackend, BackendConfig
from diffulex_edge.quant import QuantizationConfig, QuantizationDtype

# Export with quantization
backend = XNNPACKBackend(BackendConfig(
    quant_config=QuantizationConfig(
        dtype=QuantizationDtype.INT8,
        mode="weight_only"
    )
))

result = backend.export(model, example_inputs)
```

## Backend Compatibility Matrix

| Quantization | XNNPACK | CoreML | QNN |
|--------------|:-------:|:------:|:---:|
| FP32 | ✅ | ✅ | ✅ |
| FP16 | ✅ | ✅ | ⚠️ |
| INT8 Dynamic | ✅ | ✅ | ✅ |
| INT8 Static | ✅ | ✅ | ✅ |
| INT8 Weight-only | ✅ | ✅ | ✅ |
| INT4 Weight-only | ✅ | ❌ | ⚠️ |

Legend: ✅ Full Support | ⚠️ Partial/Limited | ❌ Not Supported

## Implementation Details

### FP16 Quantizer

Simple weight casting to FP16 with optional layer exclusion:

```python
# File: diffulex_edge/quant/fp16_quantizer.py
class FP16Quantizer(BaseQuantizer):
    def quantize(self, model, config):
        # Convert Linear layer weights to FP16
        for module in model.modules():
            if isinstance(module, nn.Linear):
                qlinear = QuantizedLinear.from_float(module, dtype=torch.float16)
                # Replace module...
```

### INT8 Quantizer

Per-channel symmetric quantization:

```python
# Scale = max_abs / 127.0
# Weight_quant = round(weight / scale).clamp(-128, 127)
```

Storage format:
- `weight`: INT8 tensor `[out_features, in_features]`
- `weight_scale`: FP32 tensor `[out_features]` (per-channel)

### INT4 Quantizer (torchao)

Uses torchao's `int4_weight_only` quantization:

```python
from torchao.quantization import quantize_, int4_weight_only

quantize_(model, int4_weight_only(group_size=group_size))
```

Graceful fallback when torchao unavailable.

## Performance Benchmarks

### Memory Reduction (Measured)

| Model | FP32 | FP16 | INT8 | INT4 |
|-------|------|------|------|------|
| Tiny (4 layers, 256 hidden) | 11.1 MB | 6.6 MB (1.7x) | 4.4 MB (2.5x) | ~2.8 MB (4.0x) |

### File Size Reduction (.pte export)

| Format | Size | Compression |
|--------|------|-------------|
| FP32 | 11.2 MB | 1.0x |
| FP16 | 6.7 MB | 1.7x |
| INT8 | 4.5 MB | 2.5x |
| INT4 | ~2.8 MB | 4.0x |

## Dependencies

### Required
- `torch >= 2.0.0`

### Optional
- `torchao >= 0.3.0` - For INT4 quantization
- `executorch` - For .pte export

## Limitations

1. **INT4 Dependencies**: Requires both `torchao` and `fbgemm-gpu-genai >= 1.2.0`
2. **CoreML**: INT4 not supported (minimum INT8)
3. **QNN**: INT4 support is partial via HTP backend
4. **FP16**: Requires hardware support (Apple A12+, ARM64 with FP16, CUDA 5.3+)

## See Also

- [Architecture](ARCHITECTURE.md) - System architecture
- [User Guide](USER_GUIDE.md) - Export and usage guide
