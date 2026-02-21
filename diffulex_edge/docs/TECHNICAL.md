# DiffuLex Edge - Technical Details

Implementation details for block diffusion, KV cache, and static graph export.

---

## Table of Contents

1. [Block Diffusion Mechanism](#block-diffusion-mechanism)
2. [KV Cache Design](#kv-cache-design)
3. [Static Graph Export](#static-graph-export)
4. [SDAR Block Diffusion Export](#sdar-block-diffusion-export)

---

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
   Visible         Visible          Visible          Visible       Masked
   to all        prompt only      prompt +         All confirmed
                                               previous blocks   + itself
```

---

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

---

## Static Graph Export

### Overview

The SDAREdge model supports two inference modes:

1. **`forward()`** - Dynamic KV cache for Python inference with full flexibility
2. **`forward_export()`** - Static computation graph for ExecuTorch export with Block Diffusion

### Problem Solved

ExecuTorch requires static computation graphs without:
- Data-dependent control flow (`if tensor > 0:`)
- Dynamic tensor slicing (`tensor[:dynamic_val]`)
- Python `item()` calls on tensors

### Solution: Block Diffusion Masks

Instead of dynamic cache operations, we use pre-computed masks from CPU:

| Mask | Shape | Purpose |
|------|-------|---------|
| `attention_mask` | `[num_layers, batch, 1, block_size, max_len+block_size]` | Controls attention visibility (0=valid, -10000=invalid) |
| `insert_matrix` | `[num_layers, batch, 1, max_len, block_size]` | Maps block KV to cache positions via matrix multiplication |
| `keep_mask` | `[num_layers, batch, 1, max_len, 1]` | Selects cache positions to keep (1) or overwrite (0) |

### Static Attention Computation

```python
# Concatenate cache and new block
full_k = torch.cat([old_k_cache, block_k], dim=-2)

# Compute scores with scaling
scores = torch.matmul(block_q, full_k.transpose(-1, -2)) * self.scaling

# Apply attention mask (adds -10000 to invalid positions)
scores = scores + attention_mask
probs = torch.softmax(scores, dim=-1)

# Compute output
block_out = torch.matmul(probs, full_v)
```

### Static Cache Update

```python
# Expand block KV to max_len via matrix multiplication
expanded_k = torch.matmul(insert_matrix, block_k)
expanded_v = torch.matmul(insert_matrix, block_v)

# Selectively update cache
updated_k_cache = (old_k_cache * keep_mask) + expanded_k
updated_v_cache = (old_v_cache * keep_mask) + expanded_v
```

### Usage Modes

#### Iteration Mode (Refinement)
```python
# Cache unchanged
keep_mask = torch.ones(..., max_len, 1)           # Keep all
insert_matrix = torch.zeros(..., max_len, block_size)  # No new data
```

#### Commit Mode (Write to Cache)
```python
# Write block to cache positions [start:end]
keep_mask = torch.ones(..., max_len, 1)
keep_mask[..., start:end, :] = 0.0  # Mark positions to overwrite

insert_matrix = torch.zeros(..., max_len, block_size)
insert_matrix[..., start+i, i] = 1.0  # Map block[i] to cache[start+i]
```

---

## SDAR Block Diffusion Export

### Interface

#### Model Input (forward_export)

```python
def forward_export(
    self,
    input_ids: torch.Tensor,          # [batch, block_size]
    positions: torch.Tensor,          # [batch, block_size]
    kv_cache: torch.Tensor,           # [num_layers, 2, batch, num_kv_heads, max_len, head_dim]
    attention_mask: torch.Tensor,     # [num_layers, batch, 1, block_size, max_len+block_size]
    insert_matrix: torch.Tensor,      # [num_layers, batch, 1, max_len, block_size]
    keep_mask: torch.Tensor,          # [num_layers, batch, 1, max_len, 1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns: (logits, updated_kv_cache)
```

### Export Example

```bash
python -m diffulex_edge.scripts.export_model \
    /path/to/sdar-model \
    --backend xnnpack \
    -o model.pte
```

### Implementation Files

1. **`diffulex_edge/model/sdar_edge.py`**
   - `SDARAttention.forward_export()` - static attention with masks
   - `SDARDecoderLayer.forward_export()` - wrapper for attention
   - `SDAREdge.forward_export()` - Block Diffusion interface

2. **`diffulex_edge/model/wrapper.py`**
   - `ExportWrapper` - generic wrapper for ExecuTorch export
   - Handles parameter name sanitization
   - Maps `forward_export` to `forward` for ExecuTorch

3. **`diffulex_edge/backends/xnnpack_backend.py`**
   - Uses generic `_prepare_model_for_export()` for backend-agnostic export

### Numerical Verification

All tests pass with < 1e-5 tolerance:
- Single-layer output consistency
- Multi-layer output consistency  
- End-to-end token generation (5 tokens)
- Block generation scenarios

### Export Verification

Tested configurations:
- Small model (2 layers, 128 hidden): 2.2 MB .pte file ✓
- 4-layer model (4 layers, 2048 hidden): 3.2 GB .pte file ✓
- Full model (28 layers, 2048 hidden): Supported, requires sufficient disk space

---

## Technical Notes

### GQA (Grouped Query Attention)

KV heads are repeated to match Q heads count:
```python
if self.num_kv_heads != self.num_heads:
    num_repeat = self.num_heads // self.num_kv_heads
    full_k = full_k.repeat_interleave(num_repeat, dim=1)
    full_v = full_v.repeat_interleave(num_repeat, dim=1)
```

### RoPE (Rotary Position Embedding)

Pre-computed cos/sin tables indexed by positions:
```python
cos = self.cos_cached[positions].unsqueeze(1)
sin = self.sin_cached[positions].unsqueeze(1)
```

### Attention Mask Pattern

For block diffusion with causal attention:

```
          prompt  block0  block1(active)  future
prompt      ✓       ✓         ✓            ✗
block0      ✓       ✓         ✓            ✗
block1      ✓       ✓         ✓            ✗
future      ✗       ✗         ✗            ✗
```

Implementation uses `scaled_dot_product_attention` with `is_causal=False` and custom masks.

### Attention Scaling

Apply scaling factor to scores:
```python
scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
# where self.scaling = head_dim ** -0.5
```

---

## See Also

- [Architecture](ARCHITECTURE.md) - System architecture
- [User Guide](USER_GUIDE.md) - Export and CLI usage
- [Quantization](QUANTIZATION.md) - Quantization support
