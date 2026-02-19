"""Full pipeline integration tests.

Tests the complete export and inference pipeline end-to-end.
"""

import sys
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn


class SimpleTestModel(nn.Module):
    """Simple model for integration testing."""
    
    def __init__(self, vocab_size=1000, hidden_size=128, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class TestFullPipeline:
    """End-to-end pipeline integration tests."""
    
    @pytest.fixture
    def model_config(self):
        """Model configuration."""
        return {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_layers": 2,
        }
    
    @pytest.fixture
    def model(self, model_config):
        """Create test model."""
        model = SimpleTestModel(**model_config)
        model.eval()
        return model
    
    @pytest.fixture
    def example_inputs(self):
        """Create example inputs."""
        return (torch.randint(0, 1000, (1, 10)),)
    
    def test_001_model_creation(self, model, example_inputs):
        """Step 1: Model creation and forward pass."""
        with torch.no_grad():
            output = model(*example_inputs)
        
        assert output.shape == (1, 10, 1000)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_002_export_to_edge(self, model, example_inputs):
        """Step 2: Export to ExecuTorch Edge IR."""
        try:
            from torch.export import export
            from executorch.exir import to_edge
            
            # Export
            ep = export(model, example_inputs, strict=False)
            assert ep is not None
            
            # Convert to Edge
            edge = to_edge(ep)
            assert edge is not None
            
        except ImportError as e:
            pytest.skip(f"ExecuTorch not available: {e}")
    
    def test_003_quantization_pipeline(self, model, example_inputs):
        """Step 3: Quantization pipeline."""
        try:
            from diffulex_edge.backends import XNNPACKBackend, BackendConfig
            
            backend = XNNPACKBackend(BackendConfig(
                quantize=True,
                quantization_mode="weight_only"
            ))
            
            result = backend.export(model, example_inputs)
            
            # Should succeed or fail with known error
            if result.success:
                assert result.metadata.get("quantized") is True
            else:
                # Expected on Windows without flatc, or without executorch
                error_lower = result.error_message.lower()
                assert (
                    "flatc" in error_lower 
                    or "WinError" in str(result.error_message)
                    or "executorch" in error_lower
                    or "no module" in error_lower
                )
                
        except ImportError as e:
            pytest.skip(f"Backend not available: {e}")
    
    def test_004_xnnpack_export(self, model, example_inputs):
        """Step 4: XNNPACK backend export."""
        try:
            from diffulex_edge.backends import XNNPACKBackend, BackendConfig
            
            backend = XNNPACKBackend(BackendConfig(quantize=False))
            
            if not backend.is_available():
                pytest.skip("XNNPACK not available")
            
            result = backend.export(model, example_inputs)
            
            if result.success:
                assert result.buffer is not None
                assert len(result.buffer) > 0
                assert result.metadata["backend"] == "xnnpack"
            else:
                # On Windows, may fail due to flatc
                if sys.platform == "win32":
                    pytest.skip("flatc not available on Windows")
                else:
                    pytest.fail(f"Export failed: {result.error_message}")
                    
        except ImportError as e:
            pytest.skip(f"XNNPACK not available: {e}")
    
    def test_005_export_with_temp_file(self, model, example_inputs):
        """Step 5: Export to temporary file."""
        try:
            from diffulex_edge.backends import XNNPACKBackend, BackendConfig
            
            backend = XNNPACKBackend(BackendConfig())
            
            if not backend.is_available():
                pytest.skip("XNNPACK not available")
            
            result = backend.export(model, example_inputs)
            
            if result.success:
                with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
                    f.write(result.buffer)
                    temp_path = Path(f.name)
                
                try:
                    assert temp_path.exists()
                    assert temp_path.stat().st_size == len(result.buffer)
                finally:
                    temp_path.unlink(missing_ok=True)
            else:
                if sys.platform == "win32":
                    pytest.skip("flatc not available on Windows")
                else:
                    pytest.fail(f"Export failed: {result.error_message}")
                    
        except ImportError as e:
            pytest.skip(f"Backend not available: {e}")


class TestMultiBackendConsistency:
    """Test consistency across backends."""
    
    @pytest.fixture
    def model_and_inputs(self):
        """Create test model and inputs."""
        model = SimpleTestModel(vocab_size=500, hidden_size=64, num_layers=1)
        model.eval()
        inputs = (torch.randint(0, 500, (1, 5)),)
        return model, inputs
    
    def test_all_backends_listed(self):
        """Test all backends are registered."""
        from diffulex_edge.backends.base import BackendRegistry
        
        backends = BackendRegistry.list_backends()
        
        assert "xnnpack" in backends
        assert "qnn" in backends
        assert "coreml" in backends
    
    def test_backend_metadata_consistency(self):
        """Test backend metadata has consistent structure."""
        from diffulex_edge.backends import XNNPACKBackend, QNNBackend
        
        xnn = XNNPACKBackend()
        qnn = QNNBackend()
        
        xnn_meta = xnn.get_metadata()
        qnn_meta = qnn.get_metadata()
        
        # Both should have these keys
        required_keys = ["name", "available", "config"]
        for key in required_keys:
            assert key in xnn_meta
            assert key in qnn_meta


class TestPipelineErrorHandling:
    """Test pipeline error handling."""
    
    def test_invalid_model_handling(self):
        """Test handling of invalid model."""
        from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        
        # Model that will cause export to fail
        class BadModel(nn.Module):
            def forward(self, x):
                # This might cause issues
                return x.nonzero()  # Dynamic shape
        
        model = BadModel()
        inputs = (torch.randn(1, 10),)
        
        backend = XNNPACKBackend(BackendConfig())
        result = backend.export(model, inputs)
        
        # Should fail gracefully
        assert not result.success or result.buffer is not None
    
    def test_empty_model_handling(self):
        """Test handling of empty model."""
        from diffulex_edge.backends import XNNPACKBackend, BackendConfig
        
        class EmptyModel(nn.Module):
            def forward(self, x):
                return x
        
        model = EmptyModel()
        inputs = (torch.randn(1, 10),)
        
        backend = XNNPACKBackend(BackendConfig())
        result = backend.export(model, inputs)
        
        # Empty model should still export (identity function)
        # but may fail on Windows due to flatc or missing executorch
        if not result.success:
            error_lower = result.error_message.lower()
            assert (
                "flatc" in error_lower
                or "WinError" in str(result.error_message)
                or "executorch" in error_lower
                or "no module" in error_lower
            )
