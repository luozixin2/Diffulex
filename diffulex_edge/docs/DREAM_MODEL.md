# DreamEdge Model

This document describes the DreamEdge model implementation in DiffuLex Edge.

## Overview

DreamEdge is an edge-optimized implementation of the Dream diffusion language model. It aligns with the HuggingFace Dream implementation while being optimized for edge deployment.

## Key Features

### Architecture

- **Hidden Size**: 3584
- **Num Layers**: 28
- **Attention Heads**: 28
- **KV Heads**: 4 (GQA - Grouped Query Attention)
- **Intermediate Size**: 18944
- **Vocab Size**: 152064
- **Max Position Embeddings**: 131072
- **RoPE Theta**: 1000000.0

### Attention Mechanism

Dream uses **bidirectional (non-causal) attention**, which is different from standard autoregressive language models:

```python
self.is_causal = False  # Not causal attention
```

This means:
- Each position can attend to all positions (not just previous ones)
- Required for diffusion-based generation
- KV cache works differently compared to causal models

### Bias Configuration

Following the HF Dream implementation:

```python
# Q, K, V projections have bias
self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)

# Output projection has NO bias
self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
```

### Special Tokens

```python
mask_token_id = 151666  # Used for diffusion masking
pad_token_id = 151643
bos_token_id = 151643
eos_token_id = 151643
```

## Usage

### Loading the Model

```python
from diffulex_edge.model.model_loader import load_hf_model

model_path = "/path/to/Dream-v0-Instruct-7B"
model, model_type, config = load_hf_model(
    model_path,
    dtype=torch.float16,
    device="cpu",
    optimize_cpu=True,
)
```

### Inference

```python
import torch

# Prepare input
input_ids = torch.randint(0, 152064, (1, 10))

# Forward pass
with torch.no_grad():
    logits, _ = model(input_ids)

# Get predictions
predictions = logits.argmax(dim=-1)
```

### With Attention Mask

```python
# Create attention mask (2D)
attention_mask = torch.ones(1, 10, dtype=torch.long)
attention_mask[0, 5:] = 0  # Mask out positions 5+

# Forward with mask
with torch.no_grad():
    logits, _ = model(input_ids, attention_mask=attention_mask)
```

### With KV Cache (Limited Use)

Note: For Dream's bidirectional attention, KV cache has limited use because each position depends on all positions. However, it can still be used for specific scenarios:

```python
# Prefill
logits, past_kv = model(input_ids[:, :5], use_cache=True)

# Decode (note: results may differ from full forward due to bidirectional attention)
logits_next, past_kv = model(
    input_ids[:, 5:6],
    position_ids=torch.tensor([[5]]),
    past_key_values=past_kv,
    use_cache=True
)
```

## Implementation Details

### Model Structure

```
DreamEdge
├── embed_tokens: nn.Embedding(152064, 3584)
├── layers: ModuleList(28 x DreamDecoderLayer)
│   ├── self_attn: DreamAttention
│   │   ├── q_proj, k_proj, v_proj, o_proj
│   │   └── rotary_emb: RotaryEmbedding
│   ├── mlp: SwiGLUMLP
│   ├── input_layernorm: RMSNorm
│   └── post_attention_layernorm: RMSNorm
├── norm: RMSNorm
└── lm_head: nn.Linear(3584, 152064)
```

### Attention Mask Format

The model supports:
- **2D mask**: `[batch_size, seq_len]` - Automatically converted to 4D
- **4D mask**: `[batch_size, 1, seq_len, kv_len]` - Directly used

For 2D masks, padded positions (0s) are converted to large negative values to prevent attention.

### Rotary Position Embedding (RoPE)

Uses standard RoPE with:
- Base frequency: 1000000.0
- Max positions: 131072

## Testing

Run the alignment tests:

```bash
cd /path/to/Diffulex

# Lightweight tests (fast)
python diffulex_edge/tests/test_dream_alignment_light.py

# Full alignment tests (requires model)
python diffulex_edge/tests/test_dream_alignment.py
```

## Differences from HF Dream

The DreamEdge implementation is designed to be numerically equivalent to the HF Dream model, with these edge-specific adaptations:

1. **No Flash Attention**: Uses PyTorch native attention operations only
2. **Simplified Cache**: Static KV cache instead of DynamicCache
3. **CPU Optimized**: Includes CPU-specific optimizations for edge deployment
4. **ExecuTorch Ready**: Designed for export to ExecuTorch format

## Performance Considerations

### CPU Inference

- Set thread count appropriately for your hardware:
  ```python
  torch.set_num_threads(8)  # Adjust based on CPU cores
  ```

- Use FP16 or BF16 for faster inference on supported CPUs

### Memory Usage

- 7B model in FP32: ~28 GB
- 7B model in FP16: ~14 GB
- 7B model in INT8: ~7 GB (with quantization)

## Export to ExecuTorch

The model can be exported to ExecuTorch format for edge deployment:

```python
from diffulex_edge.export import export_model

# Export
export_model(
    model,
    example_inputs=(input_ids,),
    output_path="dream.pte",
    quantization="int8",  # Optional: "fp16", "int8", "int4"
)
```

## References

- HuggingFace Dream Implementation: `/root/autodl-tmp/Dream-v0-Instruct-7B/`
- Dream Paper: (Add paper reference when available)
