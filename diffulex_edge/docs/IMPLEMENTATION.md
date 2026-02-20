# DiffuLex Edge Implementation Details

Technical implementation notes for block diffusion and KV cache.

## Block Diffusion Mechanism

### Overview

Block diffusion processes generation in fixed-size blocks (typically 4 tokens):

1. **Prefill Phase**: Process full prompt, initialize first active block
2. **Decode Phase**: Iterate on active block until confidence threshold
3. **Block Advance**: Freeze completed block, create new active block

### Block States

```python
class BlockStatus(Enum):
    ACTIVE = "active"      # Currently generating
    FROZEN = "frozen"      # Completed (simulates KV cached)
```

### Sequence Structure

```
[prompt tokens] [frozen block 0] [frozen block 1] [active block] [future masks]
     ↓                ↓                ↓                ↓             ↓
   可见所有         可见prompt      可见prompt      可见所有已      不可见
   已确认          + 自己           + 前面已确认     确认+自己      任何东西
```

## KV Cache Design

### Static KV Cache Format

For edge compatibility, we use pre-allocated fixed-size cache:

```
Shape: [num_layers, 2, batch, kv_heads, max_seq, head_dim]
       │           │  │     │        │         └─ 64/128
       │           │  │     │        └─ 2048 (max_seq_len)
       │           │  │     └─ 4/8 (GQA)
       │           │  └─ 1 (batch)
       │           └─ K/V (2)
       └─ 22-32 layers
```

### Memory Calculation

```
Size = layers × 2 × batch × kv_heads × max_seq × head_dim × dtype_bytes

Example (BF16, 28 layers, 2048 seq, 8 kv_heads, 128 head_dim):
= 28 × 2 × 1 × 8 × 2048 × 128 × 2 = 117 MB
```

### PTE Model Constraints

Current PTE models have fixed input shapes. The implementation works around this by:
1. Passing full sequence each forward
2. Using position indices to indicate active region
3. Updating KV cache after each block completes

## Attention Mask Pattern

For block diffusion with causal attention:

```
          prompt  block0  block1(active)  future
prompt      ✓       ✓         ✓            ✗
block0      ✓       ✓         ✓            ✗
block1      ✓       ✓         ✓            ✗
future      ✗       ✗         ✗            ✗
```

Implementation uses `scaled_dot_product_attention` with `is_causal=False` and custom masks.

## See Also

- [Architecture](ARCHITECTURE.md) - System architecture
- [Export Guide](EXPORT.md) - Model export
