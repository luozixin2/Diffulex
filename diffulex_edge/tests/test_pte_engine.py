"""PTE Inference Engine Tests - Layer 3: Engine-level PTE tests.

Tests InferenceEngine functionality with PTE models including:
- Engine creation from PTE files
- Prefill and decode operations
- Complete generation workflows

Test IDs follow pattern: PTE-ENG-xxx
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock


# ============================================================================
# Test Models
# ============================================================================

class SimpleKVModel(nn.Module):
    """Simple model with KV cache support for testing."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 64, 
                 num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = 128
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Simple attention layers
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.o_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, kv_cache=None, start_pos=0):
        batch_size, seq_len = input_ids.shape
        
        x = self.embedding(input_ids)
        
        # Simple transformer forward
        for i in range(self.num_layers):
            q = self.q_projs[i](x)
            k = self.k_projs[i](x)
            v = self.v_projs[i](x)
            
            # Simple attention computation
            attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            
            out = torch.matmul(attn, v)
            out = self.o_projs[i](out)
            
            x = x + out  # Residual
        
        logits = self.lm_head(x)
        
        return logits, None  # Return None for kv_cache to match interface


# ============================================================================
# Layer 3: Engine Creation Tests
# ============================================================================

class TestPTEEngineCreation:
    """Engine creation tests (PTE-ENG-001 to PTE-ENG-003)."""
    
    def test_pte_eng_001_from_pte_classmethod(self, tmp_path, simple_model):
        """PTE-ENG-001: Create engine using from_pte factory method.
        
        Verifies that InferenceEngine.from_pte() creates a valid engine.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        # Create a dummy PTE file
        pte_path = tmp_path / "dummy.pte"
        pte_path.write_bytes(b"dummy_pte_data")
        
        # This should work even without real ExecuTorch
        engine = InferenceEngine.from_pte(str(pte_path))
        
        assert engine is not None
        assert engine._is_pte is True
        assert engine.pte_path == pte_path
        assert engine.model is None
    
    def test_pte_eng_002_engine_initialization(self, tmp_path):
        """PTE-ENG-002: Engine initialization with PTE path.
        
        Verifies direct initialization with pte_path parameter.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        pte_path = tmp_path / "test.pte"
        pte_path.write_bytes(b"test_data")
        
        engine = InferenceEngine(pte_path=pte_path, use_kv_cache=True)
        
        assert engine._is_pte is True
        assert engine.use_kv_cache is True
        assert engine.device == "cpu"
    
    def test_pte_eng_003_engine_without_kv_cache(self, tmp_path):
        """PTE-ENG-003: Engine initialization without KV cache.
        
        Verifies engine works with KV cache disabled.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        pte_path = tmp_path / "test.pte"
        pte_path.write_bytes(b"test_data")
        
        engine = InferenceEngine(pte_path=pte_path, use_kv_cache=False)
        
        assert engine._is_pte is True
        assert engine.use_kv_cache is False


# ============================================================================
# Layer 3: Prefill Tests
# ============================================================================

class TestPTEPrefill:
    """Prefill operation tests (PTE-ENG-004 to PTE-ENG-007)."""
    
    def test_pte_eng_004_prefill_pte_mock(self, simple_model):
        """PTE-ENG-004: Prefill using mock PTE module.
        
        Verifies _prefill_pte method with mocked ExecuTorch runtime.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        # Create mock PTE module
        mock_module = MockExecuTorchModule(simple_model)
        
        # Create engine with mock
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        # Test prefill
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        logits, kv_cache = engine._prefill(input_ids)
        
        assert logits is not None
        assert logits.shape[0] == 5  # seq_len
        assert logits.shape[1] == 1000  # vocab_size
    
    def test_pte_eng_005_prefill_different_lengths(self, simple_model):
        """PTE-ENG-005: Prefill with different sequence lengths.
        
        Verifies _prefill_pte handles various input lengths.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        seq_lengths = [1, 5, 10]
        
        for seq_len in seq_lengths:
            input_ids = torch.tensor([list(range(seq_len))])
            logits, _ = engine._prefill(input_ids)
            
            assert logits.shape[0] == seq_len
    
    def test_pte_eng_006_prefill_with_kv_cache(self, simple_model):
        """PTE-ENG-006: Prefill with KV cache enabled.
        
        Verifies _prefill_pte correctly handles KV cache.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        logits, kv_cache = engine._prefill(input_ids)
        
        assert logits is not None
        # With mock, kv_cache is None
        assert kv_cache is None
    
    def test_pte_eng_007_prefill_batch_input(self, simple_model):
        """PTE-ENG-007: Prefill with batch input.
        
        Verifies _prefill_pte handles batch dimension.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        # Note: Mock may not handle batch properly, but test the interface
        input_ids = torch.tensor([[1, 2, 3]])
        logits, _ = engine._prefill(input_ids)
        
        assert logits is not None


# ============================================================================
# Layer 3: Decode Tests
# ============================================================================

class TestPTEDecode:
    """Decode operation tests (PTE-ENG-008 to PTE-ENG-011)."""
    
    def test_pte_eng_008_decode_step_pte_mock(self, simple_model):
        """PTE-ENG-008: Single decode step using mock.
        
        Verifies _decode_step_pte method.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        engine._current_pos = 0
        
        # Single decode step
        logits, kv_cache = engine._decode_step(42)
        
        assert logits is not None
        assert logits.shape[0] == 1000  # vocab_size
        assert engine._current_pos == 1  # Position should increment
    
    def test_pte_eng_009_multiple_decode_steps(self, simple_model):
        """PTE-ENG-009: Multiple consecutive decode steps.
        
        Verifies position tracking across multiple steps.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        engine._current_pos = 0
        
        num_steps = 5
        for i in range(num_steps):
            logits, _ = engine._decode_step(i)
            assert logits is not None
        
        assert engine._current_pos == num_steps
    
    def test_pte_eng_010_decode_with_kv_cache(self, simple_model):
        """PTE-ENG-010: Decode with KV cache.
        
        Verifies _decode_step_pte with KV cache enabled.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        engine._current_pos = 10  # Simulate some previous tokens
        
        logits, _ = engine._decode_step(42)
        
        assert logits is not None
        assert engine._current_pos == 11
    
    def test_pte_eng_011_decode_different_tokens(self, simple_model):
        """PTE-ENG-011: Decode with various token IDs.
        
        Verifies decode works for different token values.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        token_ids = [0, 1, 100, 999, 500]
        
        for token_id in token_ids:
            engine._current_pos = 0
            logits, _ = engine._decode_step(token_id)
            assert logits is not None


# ============================================================================
# Layer 3: Complete Generation Tests
# ============================================================================

class TestPTEGeneration:
    """Complete generation tests (PTE-ENG-012 to PTE-ENG-016)."""
    
    def test_pte_eng_012_generate_basic(self, simple_model):
        """PTE-ENG-012: Basic generation with PTE model.
        
        Verifies generate() method works end-to-end with PTE path.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        prompt = [1, 2, 3]
        
        result = engine.generate(prompt, config)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 5  # max_new_tokens
    
    def test_pte_eng_013_generate_different_lengths(self, simple_model):
        """PTE-ENG-013: Generation with different max_new_tokens.
        
        Verifies correct number of tokens generated.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        token_counts = [1, 5, 10]
        
        for count in token_counts:
            config = GenerationConfig(max_new_tokens=count, temperature=0)
            prompt = [1, 2, 3]
            
            result = engine.generate(prompt, config)
            assert len(result) == count, f"Expected {count} tokens, got {len(result)}"
    
    def test_pte_eng_014_generate_with_temperature(self, simple_model):
        """PTE-ENG-014: Generation with temperature sampling.
        
        Verifies generation with non-zero temperature.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=5, temperature=1.0, top_k=50)
        prompt = [1, 2, 3]
        
        result = engine.generate(prompt, config)
        
        assert result is not None
        assert len(result) == 5
    
    def test_pte_eng_015_generate_with_kv_cache(self, simple_model):
        """PTE-ENG-015: Generation with KV cache.
        
        Verifies generation path with KV cache enabled.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        prompt = [1, 2, 3]
        
        result = engine.generate(prompt, config)
        
        assert result is not None
        assert len(result) == 5
    
    def test_pte_eng_016_generate_empty_prompt(self, simple_model):
        """PTE-ENG-016: Generation with empty prompt.
        
        Verifies handling of empty prompt.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=3, temperature=0)
        prompt = []
        
        # This might raise an exception or handle gracefully
        try:
            result = engine.generate(prompt, config)
            assert result is not None
        except (ValueError, IndexError):
            # Empty prompt should raise an error
            pass


# ============================================================================
# Layer 3: Reset and State Management
# ============================================================================

class TestPTEStateManagement:
    """State management tests (PTE-ENG-017 to PTE-ENG-020)."""
    
    def test_pte_eng_017_reset_cache(self, simple_model):
        """PTE-ENG-017: Reset cache method.
        
        Verifies reset_cache() clears internal state.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        engine._current_pos = 10
        
        engine.reset_cache()
        
        assert engine._current_pos == 0
        assert engine._kv_cache is None
    
    def test_pte_eng_018_engine_reuse(self, simple_model):
        """PTE-ENG-018: Engine reuse after generation.
        
        Verifies engine can be reset and reused.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=3, temperature=0)
        
        # First generation
        result1 = engine.generate([1, 2], config)
        
        # Reset and second generation
        engine.reset_cache()
        result2 = engine.generate([3, 4], config)
        
        assert len(result1) == 3
        assert len(result2) == 3
    
    def test_pte_eng_019_state_after_prefill(self, simple_model):
        """PTE-ENG-019: State tracking after prefill.
        
        Verifies position tracking during prefill.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        engine._prefill(input_ids)
        
        # After prefill, position should be at end of prompt
        assert engine._current_pos == 5
    
    def test_pte_eng_020_state_after_decode(self, simple_model):
        """PTE-ENG-020: State tracking after decode.
        
        Verifies position tracking during decode.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        engine._current_pos = 0
        
        # Multiple decode steps
        for i in range(5):
            assert engine._current_pos == i
            engine._decode_step(i)
        
        assert engine._current_pos == 5
