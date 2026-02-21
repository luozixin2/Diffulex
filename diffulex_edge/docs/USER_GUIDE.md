# DiffuLex Edge - User Guide

Complete guide for model export and CLI usage.

---

## Table of Contents

1. [Model Export](#model-export)
2. [CLI Usage](#cli-usage)
3. [Model-Specific Notes](#model-specific-notes)
4. [Troubleshooting](#troubleshooting)

---

## Model Export

### Quick Export

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
example_inputs = (torch.zeros(1, 128, dtype=torch.long),)
result = exporter.export(model, example_inputs)
```

### Export Configurations

#### Backends

| Backend | Target Platform | Quantization |
|---------|----------------|--------------|
| `xnnpack` | Generic CPU (ARM64/x86) | INT8, FP16, INT4 |
| `coreml` | Apple (ANE/GPU) | FP16, INT8 |
| `qnn` | Qualcomm (NPU) | INT8 |

#### Quantization

```python
from diffulex_edge.export import ExportConfig, BackendType, QuantizationType

# FP16 weight-only
ExportConfig(
    backend=BackendType.XNNPACK,
    quantization=QuantizationType.FP16,
)

# INT8 weight-only
ExportConfig(
    backend=BackendType.XNNPACK,
    quantization=QuantizationType.WEIGHT_ONLY_INT8,
)
```

Or use the unified quantization API:

```python
from diffulex_edge.quant import quantize_model

model_fp16 = quantize_model(model, dtype="fp16")
model_int8 = quantize_model(model, dtype="int8")
```

### Performance Tuning

| Model Size | Recommended Config | Memory |
|------------|-------------------|--------|
| < 500M | FP16, XNNPACK | ~200MB |
| 500M - 1B | BF16, XNNPACK | ~800MB |
| 1B - 3B | BF16/INT8, XNNPACK | ~1.5GB |
| > 3B | INT8, QNN/CoreML | ~2GB |

---

## CLI Usage

### Running the CLI

```bash
# Demo model
python -m diffulex_edge

# PyTorch model
python -m diffulex_edge --model-path path/to/model.pt

# PTE model
python -m diffulex_edge --pte-path path/to/model.pte

# With HuggingFace Tokenizer
python -m diffulex_edge --model-path model.pt --tokenizer gpt2
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/quit`, `/exit` | Exit program |
| `/clear` | Clear conversation history |
| `/temp <0.0-2.0>` | Set temperature |
| `/topk <n>` | Set top_k (0=disable) |
| `/topp <0.0-1.0>` | Set top_p |
| `/max <n>` | Set max new tokens |
| `/iter <n>` | Set diffusion iterations |
| `/conf <0.0-1.0>` | Set confidence threshold |

### Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | - | PyTorch model file path |
| `--pte-path` | - | ExecuTorch PTE model path |
| `--tokenizer` | - | HuggingFace tokenizer name or path |
| `--max-tokens` | 100 | Max generation tokens |
| `--temperature` | 0.8 | Sampling temperature |
| `--top-k` | 50 | Top-K sampling |
| `--top-p` | 0.9 | Top-P (nucleus) sampling |
| `--device` | cpu | Device (cpu/cuda) |

---

## Model-Specific Notes

### SDAR

Standard export, no special handling needed.

```python
model, model_type, hf_config = load_hf_model("path/to/sdar-model")
```

**Config:**
- mask_token_id: 126336
- Attention: Causal

### Fast dLLM v2

Standard export, no special handling needed.

```python
model, model_type, hf_config = load_hf_model("path/to/fastdllm-v2")
```

**Config:**
- mask_token_id: 151665
- Attention: Causal
- tie_word_embeddings: True

### Dream

Uses bidirectional (non-causal) attention for diffusion-based generation.

```python
model, model_type, hf_config = load_hf_model(
    "path/to/Dream-v0-Instruct-7B",
    dtype=torch.float16,
    device="cpu",
)

# For inference with PTE
engine = DiffusionEngine.from_pte("dream.pte", model_type="dream")
```

**Config:**
- mask_token_id: 151666
- Attention: **Bidirectional** (is_causal=False)
- Hidden Size: 3584
- Num Layers: 28
- Vocab Size: 152064

**Architecture Details:**
- Q, K, V projections have bias
- Output projection (o_proj) has NO bias
- Uses ShiftLogits sampler with per-block threshold

**Memory Usage:**
- FP32: ~28 GB
- FP16: ~14 GB
- INT8: ~7 GB

### LLaDA

Uses NoShiftLogits sampler.

```python
engine = DiffusionEngine.from_pte("llada.pte", model_type="llada")
```

**Config:**
- mask_token_id: 126336
- Attention: Bidirectional

---

## Troubleshooting

### Memory Issues

```python
# Use smaller batch size
example_inputs = (torch.zeros(1, 64, dtype=torch.long),)

# Use quantization
ExportConfig(
    backend=BackendType.XNNPACK,
    quantization=QuantizationType.WEIGHT_ONLY_INT8,
)
```

### Numerical Verification

```python
from diffulex_edge.tests.test_model_numerical_equivalence import verify_model

# Compare HF vs Edge outputs
verify_model("path/to/hf/model", tolerance=1e-5)
```

### CLI Issues

| Issue | Solution |
|-------|----------|
| Tokenizer loading fails | Check path or network connection |
| Out of Memory | Use `--max-tokens 50` |
| ExecuTorch not found | Install ExecuTorch or use PyTorch model |

### Known Limitations

1. **Windows**: Cannot generate .pte files (flatc not available). Use WSL2.
2. **CoreML**: Apple platforms only
3. **QNN**: Linux/Android only
4. **INT4**: Requires Linux/macOS for end-to-end testing

---

## See Also

- [Architecture](ARCHITECTURE.md) - System architecture
- [Quantization](QUANTIZATION.md) - Quantization support details
- [Technical Details](TECHNICAL.md) - Block diffusion and KV cache internals
