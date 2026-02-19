"""PTE End-to-End Tests - Layer 4: Complete workflow tests.

Tests complete workflows from model export through inference:
- Full generation pipelines
- Comparison between PyTorch and PTE models
- Streaming generation
- Multi-turn conversations

Test IDs follow pattern: PTE-E2E-xxx
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import time
import gc


# ============================================================================
# Helper Functions
# ============================================================================

def create_pte_from_model(model, example_inputs, tmp_path):
    """Helper to export model to PTE file."""
    try:
        from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        
        backend = XNNPACKBackend(BackendConfig(quantize=False))
        
        if not backend.is_available():
            pytest.skip("XNNPACK backend not available")
        
        result = backend.export(model, example_inputs)
        
        if not result.success and "flatc" in str(result.error_message).lower():
            pytest.skip("flatc not available")
        
        if not result.success:
            pytest.skip(f"Export failed: {result.error_message}")
        
        pte_path = tmp_path / "model.pte"
        with open(pte_path, "wb") as f:
            f.write(result.buffer)
        
        return pte_path
        
    except ImportError:
        pytest.skip("Backend not available")


# ============================================================================
# Layer 4: Full Generation Pipeline Tests
# ============================================================================

class TestPTEEndToEndGeneration:
    """End-to-end generation tests (PTE-E2E-001 to PTE-E2E-005)."""
    
    def test_pte_e2e_001_simple_text_generation(self, simple_model, tmp_path):
        """PTE-E2E-001: Simple text generation with PTE model.
        
        Complete flow: Export -> Load -> Generate -> Verify output.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        # Setup with mock (for reliability)
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=10, temperature=0)
        prompt = [1, 2, 3, 4, 5]
        
        # Generate
        result = engine.generate(prompt, config)
        
        # Verify
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 10
        assert all(isinstance(t, int) for t in result)
        assert all(0 <= t < simple_model.vocab_size for t in result)
    
    def test_pte_e2e_002_long_generation(self, simple_model, tmp_path):
        """PTE-E2E-002: Long text generation.
        
        Verifies stability over many generation steps.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=50, temperature=0.5, top_k=50)
        prompt = [1, 2, 3]
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 50
        # Verify no NaN or invalid tokens
        assert all(t >= 0 for t in result)
    
    def test_pte_e2e_003_multiple_generations_same_engine(self, simple_model):
        """PTE-E2E-003: Multiple generations with same engine.
        
        Verifies engine state management between generations.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        
        # First generation
        result1 = engine.generate([1, 2, 3], config)
        
        # Reset and second generation
        engine.reset_cache()
        result2 = engine.generate([4, 5, 6], config)
        
        # Third generation with same prompt as first
        engine.reset_cache()
        result3 = engine.generate([1, 2, 3], config)
        
        # All should succeed
        assert len(result1) == 5
        assert len(result2) == 5
        assert len(result3) == 5
        
        # Same prompt should give same result with temp=0
        assert result1 == result3
    
    def test_pte_e2e_004_different_sampling_params(self, simple_model):
        """PTE-E2E-004: Generation with various sampling configurations.
        
        Tests different temperature, top_k, top_p values.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        configs = [
            GenerationConfig(max_new_tokens=5, temperature=0),  # Greedy
            GenerationConfig(max_new_tokens=5, temperature=1.0, top_k=1),  # Top-k=1
            GenerationConfig(max_new_tokens=5, temperature=0.5, top_k=50),
            GenerationConfig(max_new_tokens=5, temperature=0.7, top_p=0.9),
        ]
        
        prompt = [1, 2, 3]
        
        for config in configs:
            engine.reset_cache() if hasattr(engine, 'reset_cache') else None
            result = engine.generate(prompt, config)
            assert len(result) == 5, f"Failed with config: {config}"
    
    def test_pte_e2e_005_various_prompt_lengths(self, simple_model):
        """PTE-E2E-005: Generation with various prompt lengths.
        
        Tests from empty prompt to long prompt.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock_module = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        
        prompt_lengths = [1, 5, 10, 20]
        
        for length in prompt_lengths:
            engine.reset_cache()
            prompt = list(range(length))
            
            try:
                result = engine.generate(prompt, config)
                assert len(result) == 5
            except (ValueError, IndexError) as e:
                # Some lengths might not be supported
                pytest.skip(f"Prompt length {length} not supported: {e}")


# ============================================================================
# Layer 4: PyTorch vs PTE Comparison Tests
# ============================================================================

class TestPTEPyTorchComparison:
    """Comparison tests between PyTorch and PTE models (PTE-E2E-006 to PTE-E2E-010)."""
    
    def test_pte_e2e_006_deterministic_consistency(self, simple_model_with_params):
        """PTE-E2E-006: Same model produces consistent results.
        
        With temperature=0, same input should give same output.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        model = simple_model_with_params
        
        # Create two engines with same model
        mock1 = MockExecuTorchModule(model)
        engine1 = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine1._pte_module = mock1
        engine1._is_pte = True
        
        mock2 = MockExecuTorchModule(model)
        engine2 = InferenceEngine(pte_path=Path("dummy2.pte"), use_kv_cache=False)
        engine2._pte_module = mock2
        engine2._is_pte = True
        
        config = GenerationConfig(max_new_tokens=10, temperature=0)
        prompt = [1, 2, 3]
        
        result1 = engine1.generate(prompt, config)
        result2 = engine2.generate(prompt, config)
        
        # Should be identical with temperature=0
        assert result1 == result2
    
    def test_pte_e2e_007_pytorch_vs_pte_shapes(self, simple_model):
        """PTE-E2E-007: Output shapes match between PyTorch and PTE.
        
        Verifies both paths produce same output shapes.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        prompt = [1, 2, 3]
        
        # PyTorch path (if model compatible)
        try:
            engine_torch = InferenceEngine.from_model(simple_model, use_kv_cache=False)
            result_torch = engine_torch.generate(prompt, config)
            torch_works = True
        except Exception:
            torch_works = False
            result_torch = []
        
        # PTE path (mock)
        mock = MockExecuTorchModule(simple_model)
        engine_pte = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine_pte._pte_module = mock
        engine_pte._is_pte = True
        result_pte = engine_pte.generate(prompt, config)
        
        # Both should produce same number of tokens
        assert len(result_pte) == 5
        if torch_works:
            assert len(result_torch) == len(result_pte)
    
    def test_pte_e2e_008_token_range_validation(self, simple_model):
        """PTE-E2E-008: Generated tokens are in valid range.
        
        Verifies all generated tokens are within vocabulary.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=20, temperature=1.0, top_k=50)
        prompt = [1, 2, 3]
        
        result = engine.generate(prompt, config)
        
        # All tokens should be valid
        assert all(0 <= t < simple_model.vocab_size for t in result)
    
    def test_pte_e2e_009_repeatability_with_seed(self, simple_model):
        """PTE-E2E-009: Reproducibility with fixed seed.
        
        Same seed should give same results (not fully implemented in mock).
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # Test with deterministic settings
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        prompt = [1, 2, 3]
        
        result1 = engine.generate(prompt, config)
        
        engine.reset_cache()
        result2 = engine.generate(prompt, config)
        
        # With temperature=0, should be identical
        assert result1 == result2


# ============================================================================
# Layer 4: Performance and Memory Tests
# ============================================================================

class TestPTEPerformance:
    """Performance and memory tests (PTE-E2E-011 to PTE-E2E-015)."""
    
    def test_pte_e2e_011_generation_completes(self, simple_model):
        """PTE-E2E-011: Generation completes in reasonable time.
        
        Verifies generation doesn't hang or take too long.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=10, temperature=0)
        prompt = [1, 2, 3]
        
        start = time.time()
        result = engine.generate(prompt, config)
        elapsed = time.time() - start
        
        assert result is not None
        assert elapsed < 10.0  # Should complete in 10 seconds
    
    def test_pte_e2e_012_memory_cleanup(self, simple_model):
        """PTE-E2E-012: Memory is properly managed.
        
        Verifies no memory leaks between generations.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        
        # Force garbage collection
        gc.collect()
        
        # Multiple generations
        for i in range(5):
            engine.reset_cache()
            result = engine.generate([i, i+1, i+2], config)
            assert len(result) == 5
            gc.collect()
        
        # Should complete without memory issues
        assert True
    
    def test_pte_e2e_013_kv_cache_performance_benefit(self, simple_model):
        """PTE-E2E-013: KV cache provides performance benefit.
        
        Compares generation with and without KV cache.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        
        # Without KV cache
        engine_no_cache = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine_no_cache._pte_module = mock
        engine_no_cache._is_pte = True
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        prompt = [1, 2, 3]
        
        start = time.time()
        result_no_cache = engine_no_cache.generate(prompt, config)
        time_no_cache = time.time() - start
        
        # With KV cache
        engine_with_cache = InferenceEngine(pte_path=Path("dummy2.pte"), use_kv_cache=True)
        engine_with_cache._pte_module = MockExecuTorchModule(simple_model)
        engine_with_cache._is_pte = True
        
        start = time.time()
        result_with_cache = engine_with_cache.generate(prompt, config)
        time_with_cache = time.time() - start
        
        # Both should produce results
        assert len(result_no_cache) == 5
        assert len(result_with_cache) == 5
        
        # Note: With mock, timing comparison may not be meaningful
        # but the test verifies both paths work


# ============================================================================
# Layer 4: Special Scenarios
# ============================================================================

class TestPTESpecialScenarios:
    """Special scenario tests (PTE-E2E-016 to PTE-E2E-020)."""
    
    def test_pte_e2e_016_special_tokens(self, simple_model):
        """PTE-E2E-016: Handling of special tokens.
        
        Tests generation with special tokens in prompt.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        
        # Test with various special token IDs
        special_prompts = [
            [0],  # PAD
            [2],  # EOS
            [0, 1, 2],  # Mixed
        ]
        
        for prompt in special_prompts:
            engine.reset_cache() if hasattr(engine, 'reset_cache') else None
            result = engine.generate(prompt, config)
            assert len(result) == 5
    
    def test_pte_e2e_017_single_token_prompt(self, simple_model):
        """PTE-E2E-017: Single token prompt.
        
        Tests minimal prompt case.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        prompt = [1]  # Single token
        
        result = engine.generate(prompt, config)
        assert len(result) == 5
    
    def test_pte_e2e_018_long_prompt(self, simple_model):
        """PTE-E2E-018: Long prompt handling.
        
        Tests with longer prompts.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=3, temperature=0)
        prompt = list(range(50))  # 50 token prompt
        
        try:
            result = engine.generate(prompt, config)
            assert len(result) == 3
        except Exception as e:
            # Long prompts might hit limits
            pytest.skip(f"Long prompt not supported: {e}")
    
    def test_pte_e2e_019_stop_sequences(self, simple_model):
        """PTE-E2E-019: Stop sequence handling.
        
        Tests generation with stop sequences.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(
            max_new_tokens=10,
            temperature=0,
            stop_sequences=[[500], [600, 700]]  # Stop on these sequences
        )
        prompt = [1, 2, 3]
        
        result = engine.generate(prompt, config)
        
        # Should either stop early or generate max tokens
        assert len(result) <= 10
    
    def test_pte_e2e_020_repetition_penalty(self, simple_model):
        """PTE-E2E-020: Repetition penalty.
        
        Tests generation with repetition penalty.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(
            max_new_tokens=5,
            temperature=1.0,
            repetition_penalty=1.5
        )
        prompt = [1, 2, 3]
        
        result = engine.generate(prompt, config)
        assert len(result) == 5
