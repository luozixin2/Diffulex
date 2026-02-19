# Model Export Guide

Guide for exporting DiffuLex models to ExecuTorch (.pte) format.

## Overview

DiffuLex Edge provides unified export functionality through `DiffuLexExporter`. Supported formats:

- **ExecuTorch (.pte)**: Primary format for edge deployment
- **TorchScript**: For PyTorch Mobile
- **ONNX**: For cross-platform deployment

## Quick Export

### Command Line

```bash
python -m diffulex_edge.scripts.export_model \
  --hf-model-path /path/to/hf/model \
  --output-path model.pte \
  --backend xnnpack \
  --dtype bfloat16
```

### Python API

```python
from diffulex_edge.model import load_hf_model
from diffulex_edge.export import DiffuLexExporter, ExportConfig
import torch

# Load from HuggingFace
model, model_type, hf_config = load_hf_model(
    "/path/to/hf/model",
    dtype=torch.bfloat16
)

# Configure export
export_config = ExportConfig(
    output_path="model.pte",
    backend="xnnpack",  # or "coreml", "qnn"
    dtype=torch.bfloat16,
)

# Export
exporter = DiffuLexExporter(export_config)
example_inputs = (torch.zeros(1, 128, dtype=torch.long),)  # (input_ids,)
result = exporter.export(model, example_inputs)

print(f"Exported to: {result.output_path}")
```

## Export Configurations

### Backends

| Backend | Target Platform | Quantization |
|---------|----------------|--------------|
| `xnnpack` | Generic CPU (ARM64/x86) | INT8, FP16 |
| `coreml` | Apple (ANE/GPU) | FP16 |
| `qnn` | Qualcomm (NPU) | INT8 |

### Data Types

```python
# FP32 (default)
ExportConfig(dtype=torch.float32)

# FP16 (recommended for mobile)
ExportConfig(dtype=torch.float16)

# BF16 (recommended for 1B+ models)
ExportConfig(dtype=torch.bfloat16)
```

### Quantization

```python
from diffulex_edge.export import QuantizationType

# Dynamic INT8 (balanced)
ExportConfig(
    quantization=QuantizationType.DYNAMIC_INT8
)

# Static INT8 (requires calibration)
ExportConfig(
    quantization=QuantizationType.STATIC_INT8,
    calibration_data=calibration_loader,
)

# Weight-only INT8 (minimal accuracy loss)
ExportConfig(
    quantization=QuantizationType.WEIGHT_ONLY_INT8
)
```

## Model-Specific Notes

### FastdLLM V2

Standard export, no special handling needed.

```python
model, model_type, hf_config = load_hf_model("path/to/fastdllm-v2")
```

### LLaDA

Uses NoShiftLogits sampler. Ensure `model_type="llada"` when loading.

```python
engine = DiffusionEngine.from_pte("llada.pte", model_type="llada")
```

### Dream

Uses ShiftLogits + per-block threshold.

```python
engine = DiffusionEngine.from_pte("dream.pte", model_type="dream")
```

### SDAR

May require lower precision for large models due to memory constraints.

```bash
# For 1.7B model, use bf16 to fit in 4GB
python -m diffulex_edge.scripts.export_model \
  --hf-model-path SDAR-1.7B-Chat \
  --dtype bfloat16 \
  --output-path sdar-1.7b.pte
```

## Troubleshooting

### Memory Issues

```python
# Use smaller batch size for export
example_inputs = (torch.zeros(1, 64, dtype=torch.long),)  # Reduce seq_len

# Use quantization to reduce memory
ExportConfig(quantization=QuantizationType.DYNAMIC_INT8)
```

### Export Failures

```bash
# Check model compatibility
python -m diffulex_edge.scripts.export_model \
  --hf-model-path /path/to/model \
  --validate-only

# Enable debug logging
export DIFFULEX_LOG_LEVEL=DEBUG
```

### Numerical Verification

```python
from diffulex_edge.tests.test_model_numerical_equivalence import verify_model

# Compare HF vs Edge outputs
verify_model("path/to/hf/model", tolerance=1e-5)
```

## Performance Tuning

| Model Size | Recommended Config | Memory | Speed |
|------------|-------------------|--------|-------|
| < 500M | FP16, XNNPACK | ~200MB | Baseline |
| 500M - 1B | BF16, XNNPACK | ~800MB | 0.9x |
| 1B - 3B | BF16/INT8, XNNPACK | ~1.5GB | 0.8x |
| > 3B | INT8, QNN/CoreML | ~2GB | 0.7x |

## See Also

- [Architecture](ARCHITECTURE.md) - Export pipeline details
- [API Reference](API_REFERENCE.md) - Full API documentation
