# DiffuLex Edge - Architecture

This document describes the architecture and implementation details of DiffuLex Edge.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Model Architecture](#model-architecture)
3. [KV Cache Design](#kv-cache-design)
4. [Diffusion Sampling](#diffusion-sampling)
5. [Quantization](#quantization)
6. [Backend System](#backend-system)
7. [Export Pipeline](#export-pipeline)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
├─────────────────────────────────────────────────────────────┤
│                   DiffuLex Edge Runtime                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Tokenizer  │  │   Sampler   │  │   KV Cache Manager  │  │
│  │  (Hugging   │  │  (Diffusion)│  │    (Static)         │  │
│  │   Face)     │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              ExecuTorch Runtime (.pte model)                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Transformer Model                      │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │    │
│  │  │Embedding│→ │  Layer  │→ │  ...    │→ │ LM Head│ │    │
│  │  └─────────┘  │(Attn+MLP)│  └─────────┘  └────────┘ │    │
│  │               └─────────┘                           │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Backend Execution                         │
│     XNNPACK (CPU)    │    QNN (Qualcomm)   │  CoreML (Apple)│
└─────────────────────────────────────────────────────────────┘
```

## Model Architecture

### FastdLLMV2Edge

Simplified version of FastdLLM V2 with the following modifications:

| Component | Original | Edge Version |
|-----------|----------|--------------|
| Linear | Column/RowParallelLinear | nn.Linear |
| Attention | Custom flash kernels | F.scaled_dot_product_attention |
| KV Cache | PagedAttention | Static input/output |
| RMSNorm | torch.compile | Standard implementation |

### Key Classes

```python
class FastdLLMV2EdgeConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8        # GQA
    intermediate_size: int = 14336
    max_position_embeddings: int = 32768

class FastdLLMV2Edge(nn.Module):
    def forward(self, input_ids, kv_cache=None, start_pos=0)
    def generate_step(self, input_ids, kv_cache, start_pos)
```

## KV Cache Design

### Static KV Cache

Pre-allocated fixed-size cache for edge device compatibility:

```
Cache Shape: [num_layers, 2, batch, kv_heads, max_seq, head_dim]
             │           │  │     │        │         └─ 64
             │           │  │     │        └─ 2048
             │           │  │     └─ 8 (GQA)
             │           │  └─ 1
             │           └─ K/V (2)
             └─ 32 layers
```

### Memory Calculation

```
Size = layers × 2 × batch × kv_heads × max_seq × head_dim × dtype_bytes

Example (FP32, 2 layers, 512 seq):
= 2 × 2 × 1 × 2 × 512 × 64 × 4 = 1.0 MB

With INT8 quantization:
= 0.25 MB (75% reduction)
```

### Usage Pattern

```python
# Prefill phase
logits, new_kv = model(input_ids, kv_cache=cache.get_cache_tensor(), start_pos=0)
cache.update_from_tensor(new_kv, start_pos=0)

# Decode phase (incremental)
for i in range(max_new_tokens):
    logits, new_kv = model(next_token, kv_cache=cache.get_cache_tensor(), start_pos=current_len)
    cache.update_from_tensor(new_kv, start_pos=current_len)
    current_len += 1
```

## Diffusion Sampling

### Core Concepts

Diffusion LLMs generate tokens in parallel through an iterative denoising process:

1. **Diffusion Blocks**: Sequence divided into blocks of mask positions
2. **Shift Logits**: Special operation for dependency handling
3. **Confidence-based Acceptance**: High-confidence tokens accepted early
4. **Iterative Refinement**: Remaining masks refined in next iteration

### DiffusionBlock

```python
@dataclass
class DiffusionBlock:
    start_pos: int           # Block start in sequence
    end_pos: int             # Block end in sequence
    mask_positions: List[int]  # Relative positions still masked
    is_active: bool          # Whether block needs more iterations
    accept_threshold: float  # Confidence threshold (default 0.95)
```

### DiffusionSampler

```python
class DiffusionSampler:
    def sample_blocks(logits, blocks, positions) -> SampleOutput
    def shift_logits(logits, last_logits) -> shifted_logits
    def compute_confidence(probs, method) -> confidence_scores
```

### Generation Flow

```
Input:  [The, mask, mask, is, mask]
        └───────┬───────┘  └─┬─┘
            Block 0      Block 1

Iteration 1:
  Block 0: [The, weather, mask]  → accept "weather" (conf 0.97)
  Block 1: [is, mask]            → no high conf tokens

Iteration 2:
  Block 0: [The, weather, today] → accept "today" (conf 0.96)
  Block 1: [is, sunny]           → accept "sunny" (conf 0.98)

Output: [The, weather, today, is, sunny]
```

## Quantization

### Supported Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Dynamic INT8 | Weight-only, dynamic activation | Balanced speed/quality |
| Static INT8 | Both weights and activations | Maximum speed |
| Weight-only | Offline weight quantization | Model size reduction |

### Implementation

```python
# Dynamic quantization
from diffulex_edge.quant import apply_dynamic_quantization
q_model = apply_dynamic_quantization(model)

# Static quantization with calibration
from diffulex_edge.quant import DiffuLexQuantizer
quantizer = DiffuLexQuantizer()
q_model = quantizer.quantize_static(model, calibration_data)
```

## Backend System

### Backend Hierarchy

```
EdgeBackend (ABC)
├── XNNPACKBackend      # Generic CPU (ARM64/x86)
├── QNNBackend          # Qualcomm NPU
└── CoreMLBackend       # Apple Neural Engine
```

### Backend Selection

```python
from diffulex_edge.backends import BackendRegistry

# Auto-select based on platform
backend = BackendRegistry.get_default()

# Manual selection
backend = BackendRegistry.create("xnnpack")

# Export
result = backend.export(model, example_inputs)
```

### Backend Compatibility

| Model | XNNPACK | QNN | CoreML |
|-------|---------|-----|--------|
| FastdLLM V2 | ✅ | ✅ | ✅ |
| Dream | ✅ | ⚠️ | ⚠️ |
| LLaDA | ✅ | ⚠️ | ⚠️ |
| SDAR | ✅ | ⚠️ | ❌ |

## Export Pipeline

### Standard Export Flow

```python
from diffulex_edge.export import export_model

result = export_model(
    model,
    example_inputs,
    output_path="model.pte",
    backend="xnnpack",           # or "qnn", "coreml"
    quantization="dynamic_int8", # or "static_int8", "none"
)
```

### Internal Pipeline

```
PyTorch Model
     ↓
torch.export.export()  → ExportedProgram
     ↓
to_edge()              → Edge IR
     ↓
to_backend()           → Backend-specific IR
     ↓
to_executorch()        → .pte file
```

### Multi-Model Export

```python
from diffulex_edge.export.model_exporter import export_model

# Export with automatic format detection
paths = export_model(
    model=model,
    model_type="fast_dllm_v2",
    output_dir="./exported",
)
# Returns: {'torchscript': ..., 'onnx': ..., 'metadata': ...}
```

## Performance Considerations

### Memory Optimization

1. **Static KV Cache**: Pre-allocated, no dynamic allocation during inference
2. **Quantization**: INT8 reduces memory by 50-75%
3. **Block-wise Generation**: Only compute necessary positions

### Speed Optimization

1. **XNNPACK**: Optimized CPU kernels for ARM64
2. **Diffusion Sampling**: Parallel token acceptance vs autoregressive
3. **Kernel Fusion**: ExecuTorch fuses compatible operations

### Trade-offs

| Optimization | Benefit | Cost |
|--------------|---------|------|
| INT8 Quantization | 2x speed, 50% memory | <2% quality loss |
| Diffusion Blocks | Parallel generation | Multiple iterations |
| Static KV Cache | Predictable memory | Fixed max sequence |

## Testing Architecture

### Test Hierarchy

```
Unit Tests (70%)
├── Model: 22 tests
├── KV Cache: 12 tests
├── Sampler: 46 tests
├── Engine: 18 tests
└── Backend: 10 tests

Integration Tests (20%)
├── Cross-module: 15 tests
└── End-to-end: 16 tests

Performance Tests (10%)
├── Latency: 5 tests
└── Memory: 3 tests
```

### Test Categories

| Category | Count | Purpose |
|----------|-------|---------|
| Functional | 33 | Basic functionality |
| Boundary | 30 | Edge cases |
| Error Handling | 18 | Exception handling |
| Numerical | 13 | Precision/accuracy |
| Integration | 15 | Module interaction |
| Performance | 5 | Speed benchmarks |
