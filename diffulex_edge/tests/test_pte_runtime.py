"""PTE Runtime Tests - Layer 1-2: Base export and runtime loading tests.

Tests the basic functionality of exporting models to PTE format and
loading them with ExecuTorch runtime.

Test IDs follow pattern: PTE-{LAYER}-{SEQUENCE}
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile


# ============================================================================
# Layer 1: Base Export Tests
# ============================================================================

class TestPTEBaseExport:
    """Base export tests (PTE-BASE-xxx)."""
    
    def test_pte_base_001_simple_model_export(self, simple_model, tmp_path):
        """PTE-BASE-001: Export simplest Linear model to PTE.
        
        Verifies that a simple model can be exported to valid .pte format.
        """
        try:
            from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        except ImportError:
            pytest.skip("Backends not available")
        
        backend = XNNPACKBackend(BackendConfig(quantize=False))
        
        if not backend.is_available():
            pytest.skip("XNNPACK not available")
        
        example_inputs = (torch.randint(0, 1000, (1, 5)),)
        result = backend.export(simple_model, example_inputs)
        
        if not result.success and "flatc" in str(result.error_message).lower():
            pytest.skip("flatc not available")
        
        assert result.success, f"Export failed: {result.error_message}"
        assert result.buffer is not None
        assert len(result.buffer) > 0
        
        # Save and verify file
        pte_path = tmp_path / "simple_model.pte"
        with open(pte_path, "wb") as f:
            f.write(result.buffer)
        
        assert pte_path.exists()
        assert pte_path.stat().st_size > 0
    
    def test_pte_base_002_different_batch_sizes(self, simple_model):
        """PTE-BASE-002: Export with different batch sizes.
        
        Verifies that model can handle various batch sizes.
        """
        try:
            from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        except ImportError:
            pytest.skip("Backends not available")
        
        backend = XNNPACKBackend(BackendConfig(quantize=False))
        
        if not backend.is_available():
            pytest.skip("XNNPACK not available")
        
        batch_sizes = [1, 2]
        
        for batch_size in batch_sizes:
            example_inputs = (torch.randint(0, 1000, (batch_size, 5)),)
            result = backend.export(simple_model, example_inputs)
            
            if not result.success and "flatc" in str(result.error_message).lower():
                pytest.skip("flatc not available")
            
            assert result.success, f"Export failed for batch_size={batch_size}"
            assert result.buffer is not None
    
    def test_pte_base_003_different_sequence_lengths(self, simple_model):
        """PTE-BASE-003: Export with different sequence lengths.
        
        Verifies that model can handle various input sequence lengths.
        """
        try:
            from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        except ImportError:
            pytest.skip("Backends not available")
        
        backend = XNNPACKBackend(BackendConfig(quantize=False))
        
        if not backend.is_available():
            pytest.skip("XNNPACK not available")
        
        seq_lengths = [1, 5, 10, 20]
        
        for seq_len in seq_lengths:
            example_inputs = (torch.randint(0, 1000, (1, seq_len)),)
            result = backend.export(simple_model, example_inputs)
            
            if not result.success and "flatc" in str(result.error_message).lower():
                pytest.skip("flatc not available")
            
            assert result.success, f"Export failed for seq_len={seq_len}"
            assert result.buffer is not None
    
    def test_pte_base_004_export_metadata(self, simple_model):
        """PTE-BASE-004: Verify export metadata is correct.
        
        Verifies that export result contains expected metadata.
        """
        try:
            from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        except ImportError:
            pytest.skip("Backends not available")
        
        backend = XNNPACKBackend(BackendConfig(quantize=False))
        
        if not backend.is_available():
            pytest.skip("XNNPACK not available")
        
        example_inputs = (torch.randint(0, 1000, (1, 10)),)
        result = backend.export(simple_model, example_inputs)
        
        if not result.success and "flatc" in str(result.error_message).lower():
            pytest.skip("flatc not available")
        
        assert result.success
        assert result.metadata is not None
        assert "backend" in result.metadata
        assert result.metadata["backend"] == "xnnpack"
        assert "buffer_size_bytes" in result.metadata
        assert result.metadata["buffer_size_bytes"] == len(result.buffer)


# ============================================================================
# Layer 2: Runtime Loading Tests
# ============================================================================

class TestPTERuntime:
    """Runtime loading tests (PTE-RT-xxx)."""
    
    def test_pte_rt_001_load_valid_pte_file(self, exported_pte_file):
        """PTE-RT-001: Load a valid PTE file.
        
        Verifies that ExecuTorch runtime can load a valid .pte file.
        """
        pte_path = exported_pte_file
        
        try:
            from executorch.runtime import Runtime, Verification
            
            runtime = Runtime.get()
            program = runtime.load_program(str(pte_path), verification=Verification.Minimal)
            
            assert program is not None
            assert hasattr(program, 'load_method')
            
            # Try to load forward method
            method = program.load_method("forward")
            assert method is not None
            
        except ImportError:
            pytest.skip("ExecuTorch runtime not available")
    
    def test_pte_rt_002_load_nonexistent_file(self, tmp_path):
        """PTE-RT-002: Attempt to load non-existent PTE file.
        
        Verifies that loading a non-existent file raises appropriate error.
        """
        try:
            from executorch.runtime import Runtime, Verification
            
            runtime = Runtime.get()
            nonexistent_path = tmp_path / "nonexistent.pte"
            
            with pytest.raises((FileNotFoundError, RuntimeError)):
                runtime.load_program(str(nonexistent_path), verification=Verification.Minimal)
                
        except ImportError:
            pytest.skip("ExecuTorch runtime not available")
    
    def test_pte_rt_003_load_invalid_file(self, tmp_path):
        """PTE-RT-003: Attempt to load invalid/corrupted PTE file.
        
        Verifies that loading corrupted data raises appropriate error.
        """
        try:
            from executorch.runtime import Runtime, Verification
            
            runtime = Runtime.get()
            
            # Create an invalid file
            invalid_path = tmp_path / "invalid.pte"
            with open(invalid_path, "wb") as f:
                f.write(b"This is not a valid PTE file\x00\x01\x02")
            
            with pytest.raises((RuntimeError, ValueError)):
                runtime.load_program(str(invalid_path), verification=Verification.Minimal)
                
        except ImportError:
            pytest.skip("ExecuTorch runtime not available")
    
    def test_pte_rt_004_forward_call_basic(self, exported_pte_file):
        """PTE-RT-004: Basic forward call on loaded PTE model.
        
        Verifies that forward method works on loaded model.
        """
        pte_path = exported_pte_file
        
        try:
            from executorch.runtime import Runtime, Verification
            
            runtime = Runtime.get()
            program = runtime.load_program(str(pte_path), verification=Verification.Minimal)
            
            # Load forward method
            method = program.load_method("forward")
            
            # Test forward with simple input
            input_ids = [[1, 2, 3, 4, 5]]
            result = method.execute(input_ids)
            
            assert result is not None
            
        except ImportError:
            pytest.skip("ExecuTorch runtime not available")
    
    def test_pte_rt_005_multiple_loads_same_file(self, exported_pte_file):
        """PTE-RT-005: Multiple loads of same PTE file.
        
        Verifies that file can be loaded multiple times.
        """
        pte_path = exported_pte_file
        
        try:
            from executorch.runtime import Runtime, Verification
            
            runtime = Runtime.get()
            
            # Load twice
            program1 = runtime.load_program(str(pte_path), verification=Verification.Minimal)
            program2 = runtime.load_program(str(pte_path), verification=Verification.Minimal)
            
            assert program1 is not None
            assert program2 is not None
            
            # Both should work
            method1 = program1.load_method("forward")
            method2 = program2.load_method("forward")
            
            result1 = method1.execute([[1, 2, 3]])
            result2 = method2.execute([[1, 2, 3]])
            
            assert result1 is not None
            assert result2 is not None
            
        except ImportError:
            pytest.skip("ExecuTorch runtime not available")


# ============================================================================
# Mock Runtime Tests (for CI without ExecuTorch runtime)
# ============================================================================

class TestPTEMockRuntime:
    """Mock runtime tests using PyTorch backend."""
    
    def test_pte_mock_001_forward_call(self, mock_pte_module):
        """Test forward call on mock ExecuTorch module."""
        input_ids = [[1, 2, 3, 4, 5]]
        
        result = mock_pte_module.forward(input_ids)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 5  # One output per input token
        assert len(result[0]) == 1000  # vocab_size
    
    def test_pte_mock_002_forward_with_start_pos(self, mock_pte_module):
        """Test forward with start_pos parameter."""
        input_ids = [[1, 2, 3]]
        start_pos = 0
        
        result = mock_pte_module.forward(input_ids, start_pos)
        
        assert result is not None
        assert isinstance(result, list)
    
    def test_pte_mock_003_single_token_input(self, mock_pte_module):
        """Test forward with single token input."""
        input_ids = [[42]]
        
        result = mock_pte_module.forward(input_ids)
        
        assert result is not None
        assert isinstance(result, list)
    
    def test_pte_mock_004_deterministic_output(self, simple_model_with_params):
        """Test that same input produces same output."""
        mock_module = type('Mock', (), {})()
        
        # Wrap with mock
        from conftest import MockExecuTorchModule
        mock = MockExecuTorchModule(simple_model_with_params)
        
        input_ids = [[1, 2, 3]]
        
        result1 = mock.forward(input_ids)
        result2 = mock.forward(input_ids)
        
        assert result1 == result2
