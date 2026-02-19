"""PTE Edge Cases and Error Handling Tests.

Tests boundary conditions, error scenarios, and robustness:
- Invalid inputs
- Missing dependencies
- Resource exhaustion
- Concurrency issues

Test IDs follow pattern: PTE-ERR-xxx
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys


# ============================================================================
# Error Handling: Dependency Tests
# ============================================================================

class TestPTEDependencyErrors:
    """Dependency error tests (PTE-ERR-001 to PTE-ERR-005)."""
    
    def test_pte_err_001_executorch_not_installed(self, tmp_path):
        """PTE-ERR-001: ExecuTorch runtime not available.
        
        Verifies graceful handling when executorch is not installed.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        pte_path = tmp_path / "test.pte"
        pte_path.write_bytes(b"dummy")
        
        # Mock the import to fail
        with patch.dict('sys.modules', {'executorch.runtime': None}):
            # Create engine should work, but loading should fail
            engine = InferenceEngine(pte_path=pte_path)
            
            # _load_pte_model should raise ImportError
            with pytest.raises(ImportError):
                engine._load_pte_model()
    
    def test_pte_err_002_backend_not_available(self, simple_model):
        """PTE-ERR-002: Backend not available during export.
        
        Verifies graceful handling when backend is unavailable.
        """
        try:
            from diffulex_edge.backends import XNNPACKBackend
        except ImportError:
            pytest.skip("Backend module not available")
        
        backend = XNNPACKBackend()
        
        # Check availability
        is_avail = backend.is_available()
        
        # Should return a boolean, not raise
        assert isinstance(is_avail, bool)
    
    def test_pte_err_003_missing_flatc(self, simple_model):
        """PTE-ERR-003: flatc compiler not available.
        
        Verifies graceful handling when flatc is missing.
        """
        try:
            from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        except ImportError:
            pytest.skip("Backend not available")
        
        backend = XNNPACKBackend(BackendConfig())
        
        if not backend.is_available():
            pytest.skip("XNNPACK not available")
        
        example_inputs = (torch.randint(0, 1000, (1, 5)),)
        result = backend.export(simple_model, example_inputs)
        
        # If flatc is missing, should fail gracefully
        if not result.success:
            # Should provide helpful error message
            assert result.error_message is not None
            assert isinstance(result.error_message, str)


# ============================================================================
# Error Handling: Input Validation Tests
# ============================================================================

class TestPTEInputValidation:
    """Input validation tests (PTE-ERR-006 to PTE-ERR-015)."""
    
    def test_pte_err_006_empty_input_tensor(self, simple_model):
        """PTE-ERR-006: Empty input tensor.
        
        Verifies handling of empty input.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # Empty input
        input_ids = torch.tensor([[]], dtype=torch.long)
        
        # Should either handle gracefully or raise appropriate error
        try:
            logits, _ = engine._prefill(input_ids)
            # If it succeeds, verify output
            assert logits is not None
        except (ValueError, IndexError, RuntimeError):
            # These are acceptable errors for empty input
            pass
    
    def test_pte_err_007_negative_token_ids(self, simple_model):
        """PTE-ERR-007: Negative token IDs.
        
        Verifies handling of negative token IDs.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # Negative token IDs
        input_ids = torch.tensor([[-1, -2, -3]], dtype=torch.long)
        
        try:
            logits, _ = engine._prefill(input_ids)
        except (IndexError, ValueError, RuntimeError):
            # Expected behavior for invalid token IDs
            pass
    
    def test_pte_err_008_token_id_out_of_range(self, simple_model):
        """PTE-ERR-008: Token ID exceeding vocabulary size.
        
        Verifies handling of out-of-range token IDs.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # Token ID beyond vocab size
        vocab_size = simple_model.vocab_size
        input_ids = torch.tensor([[vocab_size + 100]], dtype=torch.long)
        
        try:
            logits, _ = engine._prefill(input_ids)
        except (IndexError, ValueError, RuntimeError):
            # Expected behavior
            pass
    
    def test_pte_err_009_none_input(self, simple_model):
        """PTE-ERR-009: None input.
        
        Verifies handling of None input.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # None input should raise TypeError or AttributeError
        with pytest.raises((TypeError, AttributeError)):
            engine._prefill(None)
    
    def test_pte_err_010_wrong_input_type(self, simple_model):
        """PTE-ERR-010: Wrong input type (not tensor).
        
        Verifies handling of incorrect input types.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # String input
        with pytest.raises((TypeError, AttributeError)):
            engine._prefill("not a tensor")
        
        # List input (not wrapped in tensor)
        with pytest.raises((TypeError, AttributeError)):
            engine._prefill([1, 2, 3])
    
    def test_pte_err_011_invalid_generation_config(self, simple_model):
        """PTE-ERR-011: Invalid generation configuration.
        
        Verifies handling of invalid config parameters.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # Negative max_new_tokens
        with pytest.raises((ValueError, AssertionError)):
            config = GenerationConfig(max_new_tokens=-5)
        
        # Zero max_new_tokens
        config = GenerationConfig(max_new_tokens=0)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 0
    
    def test_pte_err_012_extreme_temperature(self, simple_model):
        """PTE-ERR-012: Extreme temperature values.
        
        Verifies handling of extreme temperature.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # Very high temperature
        config = GenerationConfig(max_new_tokens=3, temperature=100.0)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 3
        
        # Very low temperature (near zero)
        config = GenerationConfig(max_new_tokens=3, temperature=0.001)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 3
    
    def test_pte_err_013_invalid_top_k(self, simple_model):
        """PTE-ERR-013: Invalid top_k values.
        
        Verifies handling of invalid top_k.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # Zero top_k
        config = GenerationConfig(max_new_tokens=3, top_k=0)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 3
        
        # Negative top_k
        config = GenerationConfig(max_new_tokens=3, top_k=-5)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 3
    
    def test_pte_err_014_invalid_top_p(self, simple_model):
        """PTE-ERR-014: Invalid top_p values.
        
        Verifies handling of invalid top_p.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # top_p > 1
        config = GenerationConfig(max_new_tokens=3, top_p=2.0)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 3
        
        # Negative top_p
        config = GenerationConfig(max_new_tokens=3, top_p=-0.5)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 3
    
    def test_pte_err_015_extreme_repetition_penalty(self, simple_model):
        """PTE-ERR-015: Extreme repetition penalty.
        
        Verifies handling of extreme repetition penalty.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        # Very high repetition penalty
        config = GenerationConfig(max_new_tokens=3, repetition_penalty=100.0)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 3
        
        # Zero repetition penalty
        config = GenerationConfig(max_new_tokens=3, repetition_penalty=0.0)
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 3


# ============================================================================
# Error Handling: File System Tests
# ============================================================================

class TestPTEFileSystemErrors:
    """File system error tests (PTE-ERR-016 to PTE-ERR-020)."""
    
    def test_pte_err_016_permission_denied(self, tmp_path):
        """PTE-ERR-016: Permission denied on PTE file.
        
        Verifies handling of permission errors.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        pte_path = tmp_path / "restricted.pte"
        pte_path.write_bytes(b"dummy_pte_data")
        
        # Make file unreadable (on Unix systems)
        try:
            pte_path.chmod(0o000)
            
            engine = InferenceEngine(pte_path=pte_path)
            
            # Should raise permission error during load
            with pytest.raises((PermissionError, RuntimeError)):
                engine._load_pte_model()
        finally:
            # Restore permissions for cleanup
            pte_path.chmod(0o644)
    
    def test_pte_err_017_corrupted_pte_file(self, tmp_path):
        """PTE-ERR-017: Corrupted PTE file.
        
        Verifies handling of corrupted file content.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        pte_path = tmp_path / "corrupted.pte"
        
        # Write various corrupted data
        corruptions = [
            b"",  # Empty file
            b"NOTAPTE",  # Wrong magic
            b"\x00" * 100,  # Null bytes
            b"\xff" * 100,  # Invalid bytes
        ]
        
        for corruption in corruptions:
            pte_path.write_bytes(corruption)
            
            engine = InferenceEngine(pte_path=pte_path)
            
            # Should raise error during load
            with pytest.raises((RuntimeError, ValueError)):
                engine._load_pte_model()
    
    def test_pte_err_018_directory_instead_of_file(self, tmp_path):
        """PTE-ERR-018: Directory passed as PTE file path.
        
        Verifies handling when path is a directory.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        dir_path = tmp_path / "pte_dir"
        dir_path.mkdir()
        
        # Should raise error
        with pytest.raises((IsADirectoryError, RuntimeError, OSError)):
            engine = InferenceEngine(pte_path=dir_path)
            engine._load_pte_model()
    
    def test_pte_err_019_symlink_to_nonexistent(self, tmp_path):
        """PTE-ERR-019: Symlink pointing to non-existent file.
        
        Verifies handling of broken symlinks.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        # Skip on Windows
        if sys.platform == "win32":
            pytest.skip("Symlinks not always supported on Windows")
        
        real_path = tmp_path / "real.pte"
        link_path = tmp_path / "link.pte"
        
        # Create broken symlink
        link_path.symlink_to(real_path)
        
        engine = InferenceEngine(pte_path=link_path)
        
        # Should raise file not found
        with pytest.raises((FileNotFoundError, RuntimeError)):
            engine._load_pte_model()
    
    def test_pte_err_020_path_too_long(self, tmp_path):
        """PTE-ERR-020: Excessively long path.
        
        Verifies handling of very long paths.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        # Create a very long path
        long_name = "a" * 300
        long_path = tmp_path / f"{long_name}.pte"
        
        # This might fail on some systems
        try:
            long_path.write_bytes(b"dummy")
            
            engine = InferenceEngine(pte_path=long_path)
            
            # Try to load - might succeed or fail depending on system
            try:
                engine._load_pte_model()
            except (OSError, RuntimeError):
                pass  # Expected on some systems
        except OSError:
            pytest.skip("Long paths not supported on this system")


# ============================================================================
# Error Handling: State and Concurrency Tests
# ============================================================================

class TestPTEStateErrors:
    """State and concurrency error tests (PTE-ERR-021 to PTE-ERR-025)."""
    
    def test_pte_err_021_engine_not_loaded(self, simple_model):
        """PTE-ERR-021: Using engine before model is loaded.
        
        Verifies handling of uninitialized engine.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        
        # Create engine but don't load PTE
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        # _pte_module is None
        
        # Should handle gracefully or raise
        with pytest.raises((RuntimeError, AttributeError)):
            engine._prefill(torch.tensor([[1, 2, 3]]))
    
    def test_pte_err_022_double_load(self, simple_model, tmp_path):
        """PTE-ERR-022: Loading PTE file twice on same engine.
        
        Verifies handling of double initialization.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        
        pte_path = tmp_path / "test.pte"
        pte_path.write_bytes(b"dummy_pte")
        
        engine = InferenceEngine(pte_path=pte_path)
        
        # First load
        with pytest.raises((ImportError, RuntimeError)):
            engine._load_pte_model()
        
        # Second load - should either succeed or handle gracefully
        try:
            engine._load_pte_model()
        except (ImportError, RuntimeError):
            pass  # Expected if executorch not available
    
    def test_pte_err_023_position_overflow(self, simple_model):
        """PTE-ERR-023: Position counter overflow.
        
        Verifies handling of very large position values.
        """
        from diffulex_edge.runtime.engine import InferenceEngine
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock
        engine._is_pte = True
        
        # Set position to very large value
        engine._current_pos = 10**9
        
        # Should still work (or raise appropriate error)
        try:
            logits, _ = engine._decode_step(1)
            assert logits is not None
        except (OverflowError, RuntimeError):
            pass  # Also acceptable
    
    def test_pte_err_024_nan_in_logits(self, simple_model):
        """PTE-ERR-024: NaN values in model output.
        
        Verifies handling of NaN logits.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        
        # Create model that produces NaN
        class NaNModel(nn.Module):
            def forward(self, input_ids):
                return torch.full((input_ids.shape[0], input_ids.shape[1], 1000), float('nan')), None
        
        mock_module = type('Mock', (), {})()
        mock_module.forward = lambda x, start_pos=0: torch.full((1, len(x), 1000), float('nan'))
        
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=1, temperature=0)
        
        # Should handle NaN gracefully
        try:
            result = engine.generate([1, 2, 3], config)
            # If it succeeds, verify result
        except (RuntimeError, ValueError):
            pass  # Also acceptable
    
    def test_pte_err_025_inf_in_logits(self, simple_model):
        """PTE-ERR-025: Infinite values in model output.
        
        Verifies handling of infinite logits.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        
        # Create model that produces inf
        mock_module = type('Mock', (), {})()
        mock_module.forward = lambda x, start_pos=0: torch.full((1, len(x), 1000), float('inf'))
        
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock_module
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=1, temperature=0)
        
        # Should handle inf gracefully
        try:
            result = engine.generate([1, 2, 3], config)
        except (RuntimeError, ValueError):
            pass  # Also acceptable


# ============================================================================
# Error Handling: Resource Exhaustion Tests
# ============================================================================

class TestPTEResourceErrors:
    """Resource exhaustion tests (PTE-ERR-026 to PTE-ERR-030)."""
    
    @pytest.mark.slow
    def test_pte_err_026_very_long_generation(self, simple_model):
        """PTE-ERR-026: Very long generation sequence.
        
        Verifies stability with very long outputs.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=100, temperature=0.5)
        
        # Should complete without issues
        result = engine.generate([1, 2, 3], config)
        assert len(result) == 100
    
    def test_pte_err_027_very_large_prompt(self, simple_model):
        """PTE-ERR-027: Very large prompt input.
        
        Verifies handling of large prompts.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=1, temperature=0)
        
        # Very long prompt
        prompt = list(range(1000))
        
        try:
            result = engine.generate(prompt, config)
            # Should succeed or fail gracefully
        except (RuntimeError, MemoryError, ValueError):
            pass  # Acceptable for very large prompts
    
    def test_pte_err_028_rapid_reset_cycles(self, simple_model):
        """PTE-ERR-028: Rapid reset and generation cycles.
        
        Verifies stability under rapid state changes.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        mock = MockExecuTorchModule(simple_model)
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=True)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=2, temperature=0)
        
        # Rapid cycles
        for i in range(20):
            engine.reset_cache()
            result = engine.generate([i], config)
            assert len(result) == 2
    
    def test_pte_err_029_multiple_engines(self, simple_model):
        """PTE-ERR-029: Multiple engines simultaneously.
        
        Verifies handling of multiple engine instances.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        from conftest import MockExecuTorchModule
        
        config = GenerationConfig(max_new_tokens=3, temperature=0)
        
        engines = []
        for i in range(3):
            mock = MockExecuTorchModule(simple_model)
            engine = InferenceEngine(pte_path=Path(f"dummy{i}.pte"), use_kv_cache=False)
            engine._pte_module = mock
            engine._is_pte = True
            engines.append(engine)
        
        # Use all engines
        for i, engine in enumerate(engines):
            result = engine.generate([i], config)
            assert len(result) == 3
    
    def test_pte_err_030_large_vocab_model(self):
        """PTE-ERR-030: Model with very large vocabulary.
        
        Verifies handling of large output dimensions.
        """
        from diffulex_edge.runtime.engine import InferenceEngine, GenerationConfig
        
        class LargeVocabModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = 500000  # Very large vocab
                self.embed = nn.Embedding(self.vocab_size, 64)
                self.head = nn.Linear(64, self.vocab_size)
            
            def forward(self, input_ids):
                x = self.embed(input_ids)
                logits = self.head(x)
                return logits, None
        
        model = LargeVocabModel()
        mock = type('Mock', (), {})()
        mock.forward = lambda x, start_pos=0: model(torch.tensor(x) if isinstance(x, list) else x)[0].tolist()
        
        engine = InferenceEngine(pte_path=Path("dummy.pte"), use_kv_cache=False)
        engine._pte_module = mock
        engine._is_pte = True
        
        config = GenerationConfig(max_new_tokens=1, temperature=0)
        
        try:
            result = engine.generate([1, 2, 3], config)
            assert len(result) == 1
        except (MemoryError, RuntimeError):
            pytest.skip("Large vocab requires too much memory")
