# Diffulex Quantization Module Refactoring Plan

## Architecture Vision

```
Current (B+)                        Target (A)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implicit deps (lazy import)  →      Explicit dependency injection
Global Context state         →      Explicit passing / DI
Delegate multi-responsibility →     Split responsibilities + Coordinator
Cross-layer dependencies     →      Strict layering + DIP
```

## Phase 1: Eliminate Lazy Imports, Establish Strategy Registry Center

### Goal
- Eliminate all lazy imports in `strategy_resolver.py`
- Establish unified strategy registration and discovery mechanism
- Resolve circular dependency issues

### Changes

#### 1.1 Extend registry to support dynamic keys

Add to `registry.py`:
```python
# New: Key-based strategy registration for runtime resolution
_STRATEGY_BY_KEY: Dict[str, Callable] = {}

def register_strategy_key(key: str) -> Callable:
    """Register a strategy for a specific key (e.g., 'gptq_marlin_w4a16')"""
    def decorator(builder):
        _STRATEGY_BY_KEY[key] = builder
        return builder
    return decorator

def get_strategy_by_key(key: str) -> Optional[QuantizationStrategy]:
    """Get strategy by key, returns None if not found"""
    builder = _STRATEGY_BY_KEY.get(key)
    return builder() if builder else None
```

#### 1.2 Modify strategy registration

In each strategy file (e.g., `strategies/linear_gptq_marlin_w4a16.py`):
```python
@register_linear_strategy(weight_dtype="gptq_marlin_w4a16", act_dtype="bf16")
@register_strategy_key("gptq_marlin_w4a16")  # NEW
@register_strategy_key("gptq_marlin_w4a16")  # For different bit widths
def _build_linear_gptq_marlin_w4a16() -> LinearGPTQW4A16Strategy:
    return LinearGPTQW4A16Strategy()
```

#### 1.3 Rewrite strategy_resolver

```python
# strategy_resolver.py - after refactoring
from diffulex.utils.quantization.registry import get_strategy_by_key, create_linear_strategy

def get_strategy_for_container(container, quant_kind):
    # NO lazy imports - use registry directly
    if isinstance(container, GPTQMarlinWeight):
        return get_strategy_by_key(f"gptq_marlin_w{container.bits}a16")
    if isinstance(container, AWQMarlinWeight):
        return get_strategy_by_key("awq_marlin_w4a16")
    if isinstance(container, GPTQWeight):
        return create_linear_strategy(weight_dtype="gptq", act_dtype="bf16")
    # ... etc
```

### Verification
```python
# Test: no lazy imports in strategy_resolver
import ast
with open('strategy_resolver.py') as f:
    tree = ast.parse(f.read())
    # Verify no imports inside functions
```

---

## Phase 2: Split Delegate, Introduce Dependency Injection

### Goal
- Reduce `delegate.py` from 476 lines to ~250 lines
- Extract `WeightManager`, `ForwardCache`
- Use dependency injection instead of direct instantiation

### Changes

#### 2.1 Create weight_manager.py (new file)

```python
@dataclass
class WeightManager:
    """Manages weight container lifecycle"""
    container: Optional[QuantizedWeight] = None
    loaded_shards: set = field(default_factory=set)
    
    def set_container(self, container: QuantizedWeight) -> None:
        self.container = container
        self.loaded_shards.clear()
        
    def get_weight_format(self) -> str:
        return self.container.weight_format.value if self.container else "bf16"
        
    def maybe_quantize(self, param: nn.Parameter, strategy) -> None:
        """Runtime quantization at weight loading time"""
        
    def load_shard(self, param, loaded_weight, shard_id, ...) -> bool:
        """Load weight shard with tracking"""
```

#### 2.2 Create forward_cache.py (new file)

```python
@dataclass 
class ForwardCache:
    """Manages ForwardPlan caching for CUDA Graph optimization"""
    plan: Optional[ForwardPlan] = None
    enabled: bool = False
    
    def enable(self, enabled: bool = True) -> None:
        self.enabled = enabled
        if not enabled:
            self.invalidate()
    
    def get(self, x, bias, container) -> Optional[ForwardPlan]:
        """Get cached plan if signature matches"""
        
    def build(self, x, bias, container, strategy) -> ForwardPlan:
        """Build new forward plan"""
        
    def invalidate(self) -> None:
        self.plan = None
```

#### 2.3 Rewrite delegate.py

```python
class QuantizedLinearDelegate:
    """Coordinate weight management and forward execution"""
    
    def __init__(
        self,
        quant_kind: str = "other",
        enable_forward_plan: bool = False,
        weight_manager: Optional[WeightManager] = None,
        forward_cache: Optional[ForwardCache] = None,
        strategy_resolver: Optional[Callable] = None,
    ):
        self.quant_kind = quant_kind
        self.weight_manager = weight_manager or WeightManager()
        self.forward_cache = forward_cache or ForwardCache()
        self.forward_cache.enable(enable_forward_plan)
        self.strategy_resolver = strategy_resolver or get_strategy_for_container
```

### Verification
```python
# Test: can mock all dependencies for unit testing
def test_delegate_forward():
    mock_weight_mgr = Mock()
    mock_cache = Mock()
    mock_resolver = Mock(return_value=Mock(linear_forward=Mock(return_value=torch.tensor([1.0]))))
    
    delegate = QuantizedLinearDelegate(
        weight_manager=mock_weight_mgr,
        forward_cache=mock_cache,
        strategy_resolver=mock_resolver,
    )
```

---

## Phase 3: Make Context Explicit, Eliminate Global State

### Goal
- Eliminate `QuantizationContext` global singleton
- Change to explicit passing or dependency injection
- Maintain thread safety

### Changes

#### 3.1 Modify context.py

Keep current implementation but add explicit factory methods:

```python
class QuantizationContext:
    def __init__(self):
        self._strategies: Dict[str, QuantizationStrategy] = {}
        self._act_quant_cache: Dict[tuple, tuple] = {}
    
    # Keep for backward compatibility
    _thread_local = local()
    
    @classmethod
    def current(cls) -> 'QuantizationContext':
        """Get current thread's context (legacy, for backward compatibility)"""
        if not hasattr(cls._thread_local, 'context'):
            cls._thread_local.context = QuantizationContext()
        return cls._thread_local.context
    
    # NEW: Explicit factory
    @classmethod
    def create_isolated(cls) -> "QuantizationContext":
        """Create a new isolated context for testing or explicit management"""
        return cls()
```

#### 3.2 Add optional explicit context to Delegate

```python
class QuantizedLinearDelegate:
    def __init__(
        self,
        quant_kind: str = "other",
        context: Optional[QuantizationContext] = None,  # NEW
        ...
    ):
        self.context = context or QuantizationContext.current()
```

### Verification
```python
# Test: multiple isolated contexts don't interfere
ctx1 = QuantizationContext.create_isolated()
ctx2 = QuantizationContext.create_isolated()

ctx1.set_strategy("linear_attn", strategy_a)
ctx2.set_strategy("linear_attn", strategy_b)

assert ctx1.get_strategy("linear_attn") != ctx2.get_strategy("linear_attn")
```

---

## Phase 4: Establish Strict Layer Boundaries

### Layer Architecture Definition

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4: Application Layer                              │
│  - delegate.py (coordinator)                             │
│  - loader_adapter.py (adapter)                           │
│  Dependencies: Layer 3, Layer 2                          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Domain Services Layer                          │
│  - strategy_resolver.py (strategy resolution)            │
│  - marlin_converter.py (format conversion)               │
│  - weight_manager.py (weight management)                 │
│  - forward_cache.py (cache management)                   │
│  Dependencies: Layer 2, Layer 1                          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Infrastructure Layer                           │
│  - context.py (context management)                       │
│  - registry.py (strategy registry)                       │
│  - factory.py (object factory)                           │
│  Dependencies: Layer 1, Layer 0                          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Data Model Layer                               │
│  - core/container.py (Weight containers)                 │
│  Dependencies: Layer 0                                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 0: Interface Definition Layer                     │
│  - core/protocol.py (protocols/constants)                │
│  - strategy.py (strategy interfaces)                     │
│  Dependencies: None                                      │
└─────────────────────────────────────────────────────────┘
```

### Layer Boundary Check Script

Create `tests/architecture/test_layer_boundary.py`:

```python
def check_layer_boundary():
    """Verify layering rules"""
    rules = {
        "core/protocol.py": [],  # Layer 0: no dependencies
        "core/container.py": ["core/protocol"],  # Layer 1
        "context.py": ["strategy"],  # Layer 2
        "registry.py": ["strategy"],  # Layer 2
        "delegate.py": ["core", "strategy_resolver", "marlin_converter"],  # Layer 4
    }
    # Check each file only depends on allowed modules
```

---

## Phase 5: Comprehensive Testing

### Test Strategy

```
Unit Tests
├── test_weight_manager.py       # WeightManager isolated test
├── test_forward_cache.py        # ForwardCache isolated test
├── test_strategy_resolver.py    # Test with mock strategies
└── test_marlin_converter.py     # Test with mock vLLM

Integration Tests
├── test_delegate_integration.py # Real component integration
└── test_end_to_end.py           # Full inference pipeline

Architecture Tests
├── test_layer_boundary.py       # Verify layering rules
└── test_no_lazy_import.py       # Verify no lazy imports
```

---

## Implementation Schedule

| Phase | Duration | Risk | Schedule |
|-------|----------|------|----------|
| Phase 1 | 2-3 days | Low | Week 1 |
| Phase 2 | 3-4 days | Medium | Week 1-2 |
| Phase 3 | 2-3 days | High | Week 3 |
| Phase 4 | 1-2 days | Low | Week 4 |
| Phase 5 | 3-4 days | Low | Ongoing |

**Principle**: Submit in phases, run `diffulex_bench` after each phase before proceeding to next.

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Core code lines | 2815 lines | < 2500 lines |
| Average module size | 280 lines | < 200 lines |
| Lazy import count | 8+ locations | 0 |
| Unit test coverage | - | > 80% |
| Circular dependencies | 2-3 | 0 |

**Final Target Rating: A (Excellent Architecture)**
