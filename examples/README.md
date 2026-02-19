# DiffuLex Edge Examples

This directory contains examples demonstrating how to use DiffuLex Edge for edge deployment.

## Examples

### 1. Edge Inference Example (`edge_inference_example.py`)

A comprehensive end-to-end example showing:
- Model creation and configuration
- PyTorch inference with KV cache
- Model export to ExecuTorch format
- Quantization options

Run:
```bash
python examples/edge_inference_example.py
```

### 2. Export Script (`export_model.py`)

Command-line tool for exporting models:

```bash
python -m diffulex_edge.scripts.export_model \
    --output model.pte \
    --hidden-size 512 \
    --num-layers 4 \
    --backend xnnpack \
    --quantization dynamic_int8
```

## Quick Start

```python
from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig

# Create model
config = FastdLLMV2EdgeConfig(
    vocab_size=32000,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
)
model = FastdLLMV2Edge(config)

# Run inference
engine = InferenceEngine.from_model(model)
tokens = engine.generate([1, 2, 3], GenerationConfig(max_new_tokens=20))
```

## Export for Different Platforms

### XNNPACK (Generic CPU)
```python
from diffulex_edge.backends import XNNPACKBackend, BackendConfig

backend = XNNPACKBackend(BackendConfig(quantize=True))
result = backend.export(model, example_inputs)
if result.success:
    with open("model.pte", "wb") as f:
        f.write(result.buffer)
```

### CoreML (Apple Neural Engine)
```python
from diffulex_edge.backends import CoreMLBackend, BackendConfig

backend = CoreMLBackend(BackendConfig(
    quantize=True,
    backend_options={"compute_unit": "cpu_and_ane"}
))
result = backend.export(model, example_inputs)
```

### QNN (Qualcomm NPU)
```python
from diffulex_edge.backends import QNNBackend, BackendConfig

backend = QNNBackend(BackendConfig(
    quantize=True,
    backend_options={"soc_model": "SM8550"}
))
result = backend.export(model, example_inputs)
```
