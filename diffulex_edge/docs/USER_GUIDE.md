# DiffuLex Edge - User Guide

Complete guide for model export and CLI usage.

---

## Table of Contents

1. [Model Export](#model-export)
2. [CLI Usage](#cli-usage)
3. [Troubleshooting](#troubleshooting)

---

## Model Export

### Overview

DiffuLex Edge provides unified export functionality through `DiffuLexExporter`. Supported formats:

- **ExecuTorch (.pte)**: Primary format for edge deployment
- **TorchScript**: For PyTorch Mobile
- **ONNX**: For cross-platform deployment

### Quick Export

#### Command Line

```bash
python -m diffulex_edge.scripts.export_model \
  --hf-model-path /path/to/hf/model \
  --output-path model.pte \
  --backend xnnpack \
  --dtype bfloat16
```

#### Python API

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

### Export Configurations

#### Backends

| Backend | Target Platform | Quantization |
|---------|----------------|--------------|
| `xnnpack` | Generic CPU (ARM64/x86) | INT8, FP16, INT4 |
| `coreml` | Apple (ANE/GPU) | FP16, INT8 |
| `qnn` | Qualcomm (NPU) | INT8 |

#### Data Types

```python
# FP32 (default)
ExportConfig(dtype=torch.float32)

# FP16 (recommended for mobile)
ExportConfig(dtype=torch.float16)

# BF16 (recommended for 1B+ models)
ExportConfig(dtype=torch.bfloat16)
```

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

# INT4 weight-only (requires torchao + fbgemm-gpu-genai)
ExportConfig(
    backend=BackendType.XNNPACK,
    quantization=QuantizationType.INT4,
)
```

Or use the unified quantization API before export:

```python
from diffulex_edge.quant import quantize_model

# Quantize model before export
model_fp16 = quantize_model(model, dtype="fp16")
model_int8 = quantize_model(model, dtype="int8")
```

### Model-Specific Notes

#### FastdLLM V2

Standard export, no special handling needed.

```python
model, model_type, hf_config = load_hf_model("path/to/fastdllm-v2")
```

#### LLaDA

Uses NoShiftLogits sampler. Ensure `model_type="llada"` when loading.

```python
engine = DiffusionEngine.from_pte("llada.pte", model_type="llada")
```

#### Dream

Uses ShiftLogits + per-block threshold.

```python
engine = DiffusionEngine.from_pte("dream.pte", model_type="dream")
```

#### SDAR

May require lower precision for large models due to memory constraints.

```bash
# For 1.7B model, use bf16 to fit in 4GB
python -m diffulex_edge.scripts.export_model \
  --hf-model-path SDAR-1.7B-Chat \
  --dtype bfloat16 \
  --output-path sdar-1.7b.pte
```

### Performance Tuning

| Model Size | Recommended Config | Memory | Speed |
|------------|-------------------|--------|-------|
| < 500M | FP16, XNNPACK | ~200MB | Baseline |
| 500M - 1B | BF16, XNNPACK | ~800MB | 0.9x |
| 1B - 3B | BF16/INT8, XNNPACK | ~1.5GB | 0.8x |
| > 3B | INT8, QNN/CoreML | ~2GB | 0.7x |

---

## CLI Usage

### Overview

DiffuLex Edge provides an interactive command-line interface (CLI) for chat generation with PyTorch and PTE models. Full support for **HuggingFace Tokenizer**.

### Installation Requirements

```bash
# Base functionality
pip install torch

# Full functionality (HuggingFace Tokenizer)
pip install transformers
```

### Running the CLI

#### Method 1: Demo Model
```bash
python -m diffulex_edge
```

#### Method 2: PyTorch Model
```bash
python -m diffulex_edge --model-path path/to/model.pt
```

#### Method 3: PTE Model
```bash
python -m diffulex_edge --pte-path path/to/model.pte
```

#### Method 4: With HuggingFace Tokenizer

```bash
# Use pretrained tokenizer (auto-download)
python -m diffulex_edge --model-path model.pt --tokenizer gpt2

# Use local tokenizer directory
python -m diffulex_edge --model-path model.pt --tokenizer ./models/tokenizer/

# Disable chat template
python -m diffulex_edge --tokenizer gpt2 --no-chat-template
```

#### Method 5: Full Parameters
```bash
python -m diffulex_edge \
    --pte-path model.pte \
    --tokenizer gpt2 \
    --max-tokens 200 \
    --temperature 0.7 \
    --top-k 40 \
    --top-p 0.95
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help information |
| `/quit` or `/exit` | Exit program |
| `/clear` | Clear conversation history |
| `/config` | Show current configuration |
| `/temp <0.0-2.0>` | Set temperature |
| `/topk <n>` | Set top_k (0=disable) |
| `/topp <0.0-1.0>` | Set top_p |
| `/max <n>` | Set max new tokens |
| `/iter <n>` | Set diffusion iterations |
| `/conf <0.0-1.0>` | Set confidence threshold |

### HuggingFace Tokenizer Support

#### Auto-Detection

If `--model-path` is specified, CLI auto-detects tokenizer files in the same directory:
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`

#### Chat Template

Supports HuggingFace's `apply_chat_template` for automatic conversation formatting:

```bash
# Use model with chat template support (e.g., Llama-2)
python -m diffulex_edge \
    --model-path model.pt \
    --tokenizer meta-llama/Llama-2-7b-chat-hf
```

### Parameter Reference

#### Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | - | PyTorch model file path |
| `--pte-path` | - | ExecuTorch PTE model path |
| `--tokenizer` | - | HuggingFace tokenizer name or path |
| `--max-tokens` | 100 | Max generation tokens |
| `--temperature` | 0.8 | Sampling temperature (0.0-2.0) |
| `--top-k` | 50 | Top-K sampling (0=disable) |
| `--top-p` | 0.9 | Top-P (nucleus) sampling |
| `--device` | cpu | Device (cpu/cuda) |
| `--no-chat-template` | False | Disable chat template |

#### Generation Parameters

**Temperature**:
- 0.0: Fully deterministic
- 0.5-0.8: Balanced (recommended)
- 1.0+: More random, creative

**Top-K**: Sample only from top K tokens by probability
- 0: Use full vocabulary
- 40-50: Common values

**Top-P**: Sample from smallest set with cumulative probability >= P
- 0.9: Common value
- 1.0: Disable

**Diffusion-specific**:
- `num_iterations`: Diffusion iterations (default: 10)
- `confidence_threshold`: Threshold for token acceptance (default: 0.9)

---

## Troubleshooting

### Memory Issues

```python
# Use smaller batch size for export
example_inputs = (torch.zeros(1, 64, dtype=torch.long),)  # Reduce seq_len

# Use quantization to reduce memory
from diffulex_edge.export import ExportConfig, BackendType, QuantizationType

ExportConfig(
    backend=BackendType.XNNPACK,
    quantization=QuantizationType.WEIGHT_ONLY_INT8,
)
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

### CLI Issues

#### Tokenizer Loading Fails

```
Warning: Failed to load tokenizer from xxx: ...
Falling back to simple tokenizer
```
Solution: Check tokenizer path or network connection

#### Transformers Not Installed

```
Warning: transformers not installed, using simple tokenizer demo
```
Solution: `pip install transformers`

#### Model Loading Fails

```
Error: Cannot specify both --model-path and --pte-path
```
Solution: Use only one parameter

#### Out of Memory

```bash
# Use smaller max_tokens
python -m diffulex_edge --tokenizer gpt2 --max-tokens 50
```

#### ExecuTorch Not Installed

```
Loading PTE failed: No module named 'executorch'
```
Solution: Install ExecuTorch or use PyTorch model

### Known Limitations

1. **Windows**: Cannot generate .pte files (flatc not available). Use WSL2.
2. **CoreML**: Apple platforms only
3. **QNN**: Linux/Android only
4. **INT4**: Requires Linux/macOS for end-to-end testing

---

## See Also

- [Architecture](ARCHITECTURE.md) - System architecture
- [Quantization](QUANTIZATION.md) - Quantization support details
- [Test Plan](TEST_PLAN.md) - Testing strategy
