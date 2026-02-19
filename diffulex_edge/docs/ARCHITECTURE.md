# DiffuLex Edge - Architecture

Technical architecture and implementation details.

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

### Unified Forward Interface

All 4 models implement a consistent forward interface:

```python
def forward(
    self,
    input_ids: torch.Tensor,           # [batch, seq_len]
    positions: Optional[torch.Tensor] = None,
    kv_cache: Optional[torch.Tensor] = None,
    start_pos: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
        logits: [batch, seq_len, vocab_size]
        kv_cache: [num_layers, 2, batch, kv_heads, max_seq, head_dim] or None
    """
```

### Model Configurations

| Model | Vocab | Hidden | Layers | Heads | KV Heads | Intermediate |
|-------|-------|--------|--------|-------|----------|--------------|
| FastdLLM V2 | 126k | 2048 | 22 | 16 | 4 | 5504 |
| Dream | 100k | 2048 | 24 | 32 | 8 | 5504 |
| LLaDA | 100k | 2048 | 26 | 32 | 8 | 5504 |
| SDAR-1.7B | 152k | 2048 | 28 | 16 | 8 | 6144 |

### Rotary Embedding

All models use unified RotaryEmbedding implementation:

```python
class RotaryEmbedding(nn.Module):
    """
    Cache shape: [max_position, dim]
    Indexing: cos_cached[positions].unsqueeze(1) -> [batch, 1, seq, dim]
    """
    def forward(
        self,
        positions: torch.Tensor,      # [batch, seq_len]
        q: torch.Tensor,              # [batch, heads, seq, head_dim]
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[positions].unsqueeze(1)  # [B, 1, S, D]
        sin = self.sin_cached[positions].unsqueeze(1)
        # Apply rotation
        return q_rot, k_rot
```

## KV Cache Design

### Static KV Cache

Pre-allocated fixed-size cache for edge compatibility:

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

Example (BF16, 28 layers, 2048 seq):
= 28 × 2 × 1 × 8 × 2048 × 64 × 2 = 117 MB
```

## Diffusion Sampling

### Sampler Architecture

```
sampler/
├── base.py                      # Core sampling functions
│   ├── sample_tokens()          # Temperature, top-p, top-k
│   ├── top_p_logits()           # Nucleus filtering
│   └── top_k_logits()           # Top-k filtering
├── shift.py                     # Logits shifting
│   ├── ShiftLogitsSampler       # FastdLLM V2, Dream, SDAR
│   └── NoShiftLogitsSampler     # LLaDA
└── models/                      # Model-specific samplers
    ├── fast_dllm_v2.py          # Global threshold, always accept 1
    ├── llada.py                 # No shift, per-block threshold
    ├── dream.py                 # Shift + per-block threshold
    └── sdar.py                  # Like FastdLLM V2
```

### DiffusionBlock

```python
@dataclass
class DiffusionBlock:
    start_pos: int
    length: int
    accept_threshold: float = 0.95      # Per-block (LLaDA/Dream)
    pre_block_complete: bool = False    # Previous block status
    
    @property
    def global_mask_token_ids(self) -> List[int]:
        """Positions still masked in this block"""
```

## Export Pipeline

```
PyTorch Model (nn.Module)
         ↓
torch.export.export()           # Capture computation graph
         ↓
ExportedProgram
         ↓
to_edge()                       # Convert to Edge IR
         ↓
Edge IR (EdgeProgramManager)
         ↓
to_backend()                    # Lower to backend
         ↓
Backend IR (XNNPACK/CoreML/QNN)
         ↓
to_executorch()                 # Generate .pte
         ↓
model.pte
```

### Backend Integration

```python
class EdgeBackend(ABC):
    @abstractmethod
    def export(self, model, example_inputs) -> ExportResult:
        """Export model to backend format"""
        
    @abstractmethod
    def load(self, path) -> RuntimeModule:
        """Load exported model for inference"""
```

## Testing Architecture

### Test Hierarchy

```
Unit Tests (70%)
├── Model: 22 tests
├── KV Cache: 12 tests
├── Sampler: 38 tests
├── Engine: 37 tests (PTE)
└── Backend: 10 tests

Integration Tests (20%)
├── Cross-module: 15 tests
└── End-to-end: 16 tests

Performance Tests (10%)
├── Latency: 5 tests
└── Memory: 3 tests
```

### Numerical Equivalence Testing

Framework for verifying HF vs Edge model outputs:

```python
def verify_numerical_equivalence(
    hf_model_path: str,
    edge_model,
    test_inputs: List[torch.Tensor],
    tolerance: float = 1e-5,
) -> bool:
    """Verify Edge model matches HF outputs within tolerance"""
```

## Performance Considerations

### Memory Optimization

1. **Static KV Cache**: Pre-allocated, no dynamic allocation
2. **Quantization**: INT8 reduces memory by 50%
3. **Block-wise Generation**: Only compute necessary positions

### Speed Optimization

1. **XNNPACK**: Optimized CPU kernels for ARM64
2. **Diffusion Sampling**: Parallel token acceptance
3. **Kernel Fusion**: ExecuTorch fuses compatible operations

### Trade-offs

| Optimization | Benefit | Cost |
|--------------|---------|------|
| INT8 Quantization | 2x speed, 50% memory | <2% quality loss |
| Diffusion Blocks | Parallel generation | Multiple iterations |
| Static KV Cache | Predictable memory | Fixed max sequence |

## See Also

- [Test Plan](TEST_PLAN.md) - Testing strategy
- [Export Guide](EXPORT.md) - Export documentation
