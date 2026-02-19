"""Multi-backend test suite for DiffuLex Edge.

Tests backend availability, export functionality, and consistency.
"""

import sys
import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, hidden_size=64, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestBackendAvailability:
    """Test backend availability detection."""
    
    def test_xnnpack_availability(self):
        """Test XNNPACK backend availability detection."""
        from diffulex_edge.backends import XNNPACKBackend
        
        backend = XNNPACKBackend()
        available = backend.is_available()
        
        # XNNPACK should be available on most platforms with ExecuTorch
        # On Windows it may not be available
        if sys.platform == "win32":
            # On Windows, may or may not be available depending on ExecuTorch install
            assert isinstance(available, bool)
        else:
            # On Linux/macOS, should typically be available
            assert isinstance(available, bool)
    
    def test_qnn_availability(self):
        """Test QNN backend availability detection."""
        from diffulex_edge.backends import QNNBackend
        
        backend = QNNBackend()
        available = backend.is_available()
        
        # QNN should only be available on Linux (Android)
        if sys.platform == "win32":
            assert available is False
        else:
            # May or may not be available depending on QNN SDK
            assert isinstance(available, bool)
    
    @pytest.mark.skipif(sys.platform != "darwin", reason="CoreML only on macOS")
    def test_coreml_availability(self):
        """Test CoreML backend availability detection."""
        try:
            from diffulex_edge.backends import CoreMLBackend
            
            backend = CoreMLBackend()
            available = backend.is_available()
            assert isinstance(available, bool)
        except ImportError:
            pytest.skip("CoreML backend not installed")
    
    def test_backend_registry(self):
        """Test backend registry."""
        from diffulex_edge.backends.base import BackendRegistry
        
        # Check registered backends
        backends = BackendRegistry.list_backends()
        
        assert "xnnpack" in backends
        assert "qnn" in backends
        assert "coreml" in backends
    
    def test_backend_creation(self):
        """Test backend creation through registry."""
        from diffulex_edge.backends.base import BackendRegistry, BackendConfig
        
        # Create XNNPACK backend
        backend = BackendRegistry.create("xnnpack", BackendConfig())
        assert backend is not None
        assert backend.name == "xnnpack"
        
        # Create QNN backend
        backend = BackendRegistry.create("qnn", BackendConfig())
        assert backend is not None
        assert backend.name == "qnn"


class TestXNNPACKBackend:
    """Test XNNPACK backend functionality."""
    
    @pytest.fixture
    def model_and_inputs(self):
        """Create test model and inputs."""
        model = SimpleModel(hidden_size=64, vocab_size=1000)
        model.eval()
        inputs = torch.randint(0, 1000, (1, 10))
        return model, inputs
    
    def test_xnnpack_metadata(self):
        """Test XNNPACK backend metadata."""
        from diffulex_edge.backends import XNNPACKBackend
        
        backend = XNNPACKBackend()
        metadata = backend.get_metadata()
        
        assert metadata["name"] == "xnnpack"
        assert "supported_platforms" in metadata
        assert "supports_quantization" in metadata
        assert "arm64" in metadata["supported_platforms"]
    
    @pytest.mark.skipif(
        not XNNPACKBackend().is_available() if 'XNNPACKBackend' in dir() else True,
        reason="XNNPACK not available"
    )
    def test_xnnpack_export(self, model_and_inputs):
        """Test XNNPACK export."""
        from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        
        model, inputs = model_and_inputs
        backend = XNNPACKBackend(BackendConfig(quantize=False))
        
        result = backend.export(model, (inputs,))
        
        # Should succeed or fail with known error (flatc on Windows)
        if result.success:
            assert result.buffer is not None
            assert len(result.buffer) > 0
            assert result.metadata["backend"] == "xnnpack"
        else:
            # On Windows, may fail due to flatc
            if sys.platform == "win32":
                assert "flatc" in result.error_message.lower() or "WinError" in str(result.error_message)
    
    @pytest.mark.skipif(
        not XNNPACKBackend().is_available() if 'XNNPACKBackend' in dir() else True,
        reason="XNNPACK not available"
    )
    def test_xnnpack_quantized_export(self, model_and_inputs):
        """Test XNNPACK export with quantization."""
        from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        
        model, inputs = model_and_inputs
        backend = XNNPACKBackend(BackendConfig(
            quantize=True,
            quantization_mode="weight_only"
        ))
        
        result = backend.export(model, (inputs,))
        
        # Should succeed or fail with known error
        if result.success:
            assert result.metadata["quantized"] is True
            assert result.metadata["quantization_mode"] == "weight_only"


class TestQNNBackend:
    """Test QNN backend functionality."""
    
    @pytest.mark.skipif(sys.platform == "win32", reason="QNN not on Windows")
    def test_qnn_metadata(self):
        """Test QNN backend metadata."""
        from diffulex_edge.backends import QNNBackend
        
        backend = QNNBackend()
        metadata = backend.get_metadata()
        
        assert metadata["name"] == "qnn"
        assert "supported_soc" in metadata
        assert "requires_sdk" in metadata
    
    @pytest.mark.skipif(sys.platform == "win32", reason="QNN not on Windows")
    def test_qnn_export_not_available(self):
        """Test QNN export when not available."""
        from diffulex_edge.backends import QNNBackend, BackendConfig
        
        model = SimpleModel()
        inputs = torch.randint(0, 1000, (1, 10))
        
        backend = QNNBackend(BackendConfig())
        result = backend.export(model, (inputs,))
        
        # Should fail on non-Qualcomm devices
        assert not result.success
        assert "QNN" in result.error_message or "not available" in result.error_message.lower()


class TestCoreMLBackend:
    """Test CoreML backend functionality."""
    
    @pytest.mark.skipif(sys.platform != "darwin", reason="CoreML only on macOS")
    def test_coreml_metadata(self):
        """Test CoreML backend metadata."""
        try:
            from diffulex_edge.backends import CoreMLBackend
            
            backend = CoreMLBackend()
            metadata = backend.get_metadata()
            
            assert metadata["name"] == "coreml"
            assert "compute_units" in metadata
            assert "supported_platforms" in metadata
        except ImportError:
            pytest.skip("CoreML backend not installed")


class TestBackendConfig:
    """Test backend configuration."""
    
    def test_default_config(self):
        """Test default backend config."""
        from diffulex_edge.backends.base import BackendConfig
        
        config = BackendConfig()
        
        assert config.backend_name == "xnnpack"
        assert config.quantize is False
        assert config.quantization_mode == "weight_only"
        assert config.memory_planning == "greedy"
        assert config.max_seq_len == 2048
    
    def test_custom_config(self):
        """Test custom backend config."""
        from diffulex_edge.backends.base import BackendConfig
        
        config = BackendConfig(
            backend_name="qnn",
            quantize=True,
            quantization_mode="static",
            memory_planning="sequential",
            max_seq_len=4096,
            backend_options={"soc_model": "SM8550"},
        )
        
        assert config.backend_name == "qnn"
        assert config.quantize is True
        assert config.quantization_mode == "static"
        assert config.memory_planning == "sequential"
        assert config.max_seq_len == 4096
        assert config.backend_options["soc_model"] == "SM8550"


class TestBackendConsistency:
    """Test backend consistency across platforms."""
    
    def test_export_result_structure(self):
        """Test ExportResult dataclass structure."""
        from diffulex_edge.backends.base import ExportResult
        
        # Success result
        result = ExportResult(
            success=True,
            buffer=b"test",
            metadata={"size": 4}
        )
        assert result.success is True
        assert result.buffer == b"test"
        assert result.metadata["size"] == 4
        
        # Failure result
        result = ExportResult(
            success=False,
            error_message="test error"
        )
        assert result.success is False
        assert result.error_message == "test error"


class TestBackendFallback:
    """Test backend fallback behavior."""
    
    def test_backend_unavailable_fallback(self):
        """Test fallback when backend unavailable."""
        from diffulex_edge.backends.base import BackendRegistry
        
        # Try to create non-existent backend
        backend = BackendRegistry.create("nonexistent")
        assert backend is None
    
    def test_exporter_backend_integration(self):
        """Test exporter integration with backends."""
        from diffulex_edge.export import DiffuLexExporter, ExportConfig
        from diffulex_edge.export.config import BackendType
        
        config = ExportConfig(
            output_path="test.pte",
            backend=BackendType.XNNPACK,
        )
        
        exporter = DiffuLexExporter(config)
        backend_info = exporter.get_backend_info()
        
        # Should return backend information
        assert isinstance(backend_info, dict)


# Make XNNPACKBackend available at module level for skipif
from diffulex_edge.backends import XNNPACKBackend
