"""Tests for export module."""

import pytest
import torch
import sys
from pathlib import Path
import tempfile

from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.export import (
    ExportConfig,
    BackendType,
    QuantizationType,
    DiffuLexExporter,
    export_model,
)


# Check if running on Windows (where flatc may not be available)
def is_windows():
    return sys.platform == "win32"


# Custom marker for tests requiring flatc
requires_flatc = pytest.mark.skipif(
    is_windows(),
    reason="flatc compiler not available on Windows, .pte compilation requires WSL2/Linux"
)


class TestExportConfig:
    """Test export configuration."""
    
    def test_default_config(self):
        """Test default export configuration."""
        config = ExportConfig()
        
        assert config.output_path == Path("model.pte")
        assert config.backend == BackendType.REFERENCE
        assert config.quantization == QuantizationType.NONE
        assert config.use_kv_cache is True
    
    def test_config_post_init(self):
        """Test configuration derived values."""
        config = ExportConfig(
            hidden_size=512,
            num_heads=8,
        )
        
        assert config.head_dim == 64  # 512 // 8
        assert config.intermediate_size == 2048  # 4 * 512
    
    def test_config_with_kv_heads(self):
        """Test configuration with separate KV heads."""
        config = ExportConfig(
            num_heads=8,
            num_kv_heads=2,
        )
        
        assert config.num_kv_heads == 2
    
    def test_config_json_serialization(self):
        """Test JSON serialization."""
        config = ExportConfig(
            output_path=Path("test.pte"),
            backend=BackendType.XNNPACK,
            quantization=QuantizationType.DYNAMIC_INT8,
        )
        
        json_str = config.to_json()
        restored = ExportConfig.from_json(json_str)
        
        assert restored.output_path == config.output_path
        assert restored.backend == config.backend
        assert restored.quantization == config.quantization


class TestDiffuLexExporter:
    """Test DiffuLex exporter."""
    
    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = FastdLLMV2EdgeConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=256,
            max_position_embeddings=128,
        )
        return FastdLLMV2Edge(config)
    
    def test_exporter_initialization(self):
        """Test exporter initialization."""
        config = ExportConfig()
        exporter = DiffuLexExporter(config)
        
        assert exporter.config == config
    
    def test_export_basic(self, small_model):
        """Test basic export functionality.
        
        Note: On Windows, .pte compilation may fail due to missing flatc.
        We verify that torch.export and to_edge work correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pte"
            config = ExportConfig(
                output_path=output_path,
                backend=BackendType.REFERENCE,
                quantization=QuantizationType.NONE,
                use_kv_cache=True,
                vocab_size=100,
                hidden_size=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=16,
                intermediate_size=256,
                max_seq_len=128,
            )
            
            exporter = DiffuLexExporter(config)
            
            # Create example inputs
            input_ids = torch.randint(0, 100, (1, 8))
            positions = torch.arange(8).unsqueeze(0)
            kv_cache = torch.zeros(2, 2, 1, 2, 128, 16)  # layers, 2, batch, kv_heads, max_seq, head_dim
            example_inputs = (input_ids, positions, None, kv_cache, 0)
            
            result = exporter.export(small_model, example_inputs)
            
            # On Windows, we may not have flatc, so .pte compilation can fail
            # But torch.export and to_edge should succeed
            if not result.success and is_windows() and "flatc" in str(result.error_message).lower():
                pytest.skip(f".pte compilation not available on Windows: {result.error_message}")
            
            assert result.success is True
            assert result.output_path == output_path
            assert output_path.exists()
            assert result.file_size_mb > 0
    
    def test_export_with_dynamic_quantization(self, small_model):
        """Test export with dynamic quantization.
        
        Note: With real weight quantization, DYNAMIC_INT8 now works with torch.export
        by converting weights to INT8 storage format, achieving actual file size reduction.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_int8.pte"
            config = ExportConfig(
                output_path=output_path,
                backend=BackendType.REFERENCE,
                quantization=QuantizationType.DYNAMIC_INT8,
                use_kv_cache=True,
                vocab_size=100,
                hidden_size=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=16,
                intermediate_size=256,
                max_seq_len=128,
            )
            
            exporter = DiffuLexExporter(config)
            
            input_ids = torch.randint(0, 100, (1, 8))
            positions = torch.arange(8).unsqueeze(0)
            kv_cache = torch.zeros(2, 2, 1, 2, 128, 16)
            example_inputs = (input_ids, positions, None, kv_cache, 0)
            
            result = exporter.export(small_model, example_inputs)
            
            # With real weight quantization, DYNAMIC_INT8 should succeed
            assert result.success
            assert result.file_size_mb > 0
            assert output_path.exists()
    
    def test_export_weight_only_quantization(self, small_model):
        """Test export with weight-only quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_weight.pte"
            config = ExportConfig(
                output_path=output_path,
                backend=BackendType.REFERENCE,
                quantization=QuantizationType.WEIGHT_ONLY_INT8,
                vocab_size=100,
                hidden_size=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=16,
                intermediate_size=256,
                max_seq_len=128,
            )
            
            exporter = DiffuLexExporter(config)
            
            input_ids = torch.randint(0, 100, (1, 8))
            positions = torch.arange(8).unsqueeze(0)
            example_inputs = (input_ids, positions)
            
            result = exporter.export(small_model, example_inputs)
            
            # On Windows, we may not have flatc, so .pte compilation can fail
            if not result.success and is_windows() and "flatc" in str(result.error_message).lower():
                pytest.skip(f".pte compilation not available on Windows: {result.error_message}")
            
            assert result.success is True
    
    def test_export_without_kv_cache(self, small_model):
        """Test export without KV cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_no_cache.pte"
            config = ExportConfig(
                output_path=output_path,
                use_kv_cache=False,
                vocab_size=100,
                hidden_size=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=16,
                intermediate_size=256,
                max_seq_len=128,
            )
            
            exporter = DiffuLexExporter(config)
            
            input_ids = torch.randint(0, 100, (1, 8))
            positions = torch.arange(8).unsqueeze(0)
            example_inputs = (input_ids, positions)
            
            result = exporter.export(small_model, example_inputs)
            
            # On Windows, we may not have flatc, so .pte compilation can fail
            if not result.success and is_windows() and "flatc" in str(result.error_message).lower():
                pytest.skip(f".pte compilation not available on Windows: {result.error_message}")
            
            assert result.success is True


class TestExportConvenienceFunction:
    """Test the convenience export function."""
    
    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = FastdLLMV2EdgeConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        return FastdLLMV2Edge(config)
    
    def test_export_model_function(self, small_model):
        """Test the export_model convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model.pte")
            
            input_ids = torch.randint(0, 100, (1, 8))
            positions = torch.arange(8).unsqueeze(0)
            example_inputs = (input_ids, positions)
            
            result = export_model(
                small_model,
                example_inputs,
                output_path=output_path,
                backend="reference",
                quantization="none",
                hidden_size=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=16,
                max_seq_len=128,
            )
            
            # On Windows, we may not have flatc, so .pte compilation can fail
            if not result.success and is_windows() and "flatc" in str(result.error_message).lower():
                pytest.skip(f".pte compilation not available on Windows: {result.error_message}")
            
            assert result.success is True
            assert Path(output_path).exists()


class TestExportValidation:
    """Test export validation."""
    
    def test_export_result_repr_success(self):
        """Test ExportResult representation for success."""
        from diffulex_edge.export.config import ExportResult
        
        result = ExportResult(
            success=True,
            output_path=Path("model.pte"),
            file_size_mb=10.5,
        )
        
        repr_str = repr(result)
        assert "success=True" in repr_str
        assert "10.5MB" in repr_str
    
    def test_export_result_repr_failure(self):
        """Test ExportResult representation for failure."""
        from diffulex_edge.export.config import ExportResult
        
        result = ExportResult(
            success=False,
            error_message="Test error",
        )
        
        repr_str = repr(result)
        assert "success=False" in repr_str
        assert "Test error" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
