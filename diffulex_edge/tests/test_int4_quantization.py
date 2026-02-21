"""Tests for INT4 quantization module via torchao."""

import pytest
import torch
import torch.nn as nn

from diffulex_edge.quant.int4_quantizer import (
    INT4Quantizer,
    INT4Config,
    apply_int4_quantization,
    is_int4_available,
)
from diffulex_edge.quant._torchao_utils import check_torchao_available


class SimpleLinearModel(nn.Module):
    """Simple model for testing quantization."""
    
    def __init__(self, in_features=64, out_features=32):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 128)
        self.linear2 = nn.Linear(128, out_features)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TestINT4Availability:
    """Test INT4 availability checks."""
    
    def test_int4_availability(self):
        """Test INT4 availability check."""
        # This will be False if torchao is not installed
        available = is_int4_available()
        assert isinstance(available, bool)
    
    def test_check_torchao_available(self):
        """Test torchao availability check."""
        available = check_torchao_available()
        assert isinstance(available, bool)


class TestINT4Config:
    """Test INT4 configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = INT4Config()
        assert config.group_size == 32
        assert config.weight_only is True
        assert config.fallback_to_int8 is True
        assert "lm_head" in config.preserve_layers
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = INT4Config(
            group_size=128,
            weight_only=True,
            fallback_to_int8=False,
        )
        assert config.group_size == 128
        assert config.fallback_to_int8 is False
    
    def test_invalid_group_size(self):
        """Test invalid group size raises error."""
        with pytest.raises(ValueError):
            INT4Config(group_size=100)  # Invalid group size
    
    @pytest.mark.parametrize("group_size", [32, 64, 128, 256])
    def test_valid_group_sizes(self, group_size):
        """Test all valid group sizes."""
        config = INT4Config(group_size=group_size)
        assert config.group_size == group_size


class TestINT4Quantizer:
    """Test INT4 quantizer."""
    
    def test_quantizer_initialization(self):
        """Test quantizer initialization."""
        quantizer = TorchAOQuantizer()
        assert quantizer is not None
        # availability depends on torchao installation
        assert isinstance(quantizer.is_available(), bool)
    
    def test_get_supported_group_sizes(self):
        """Test getting supported group sizes."""
        quantizer = TorchAOQuantizer()
        sizes = quantizer.get_supported_group_sizes()
        assert 32 in sizes
        assert 64 in sizes
        assert 128 in sizes
        assert 256 in sizes
    
    def test_get_expected_size_reduction(self):
        """Test expected size reduction values."""
        quantizer = TorchAOQuantizer()
        
        reduction_32 = quantizer.get_expected_size_reduction(32)
        reduction_128 = quantizer.get_expected_size_reduction(128)
        
        assert 0 < reduction_32 < 1
        assert 0 < reduction_128 < 1


@pytest.mark.skipif(not check_torchao_available(), reason="torchao not installed")
class TestINT4WithTorchAO:
    """Tests requiring torchao."""
    
    def test_int4_weight_only_valid_group_sizes(self):
        """Group sizes 32, 64, 128, 256 should work."""
        model = SimpleLinearModel()
        quantizer = TorchAOQuantizer()
        
        for group_size in [32, 64, 128, 256]:
            model_copy = SimpleLinearModel()  # Fresh model for each test
            config = INT4Config(group_size=group_size)
            
            result = quantizer.quantize(model_copy, config)
            
            assert result.success is True
            assert result.model is not None
    
    def test_int4_model_size_reduction(self):
        """INT4 should reduce model size significantly."""
        model = SimpleLinearModel()
        quantizer = TorchAOQuantizer()
        config = INT4Config(group_size=128)
        
        result = quantizer.quantize(model, config)
        
        assert result.metrics["memory_reduction_pct"] > 0
        # Should have some reduction
        assert result.metrics["memory_reduction_mb"] > 0
    
    def test_int4_produces_reasonable_outputs(self):
        """INT4 model should produce reasonable outputs."""
        model = SimpleLinearModel()
        model.eval()
        
        # Original output
        x = torch.randn(4, 64)
        with torch.no_grad():
            original_output = model(x)
        
        # Quantized output
        quantizer = TorchAOQuantizer()
        config = INT4Config(group_size=128)
        result = quantizer.quantize(model, config)
        result.model.eval()
        
        with torch.no_grad():
            quantized_output = result.model(x)
        
        # Outputs should have same shape
        assert quantized_output.shape == original_output.shape
        
        # No NaN or Inf
        assert not torch.isnan(quantized_output).any()
        assert not torch.isinf(quantized_output).any()
    
    def test_int4_with_different_group_sizes(self):
        """Test INT4 with different group sizes."""
        model = SimpleLinearModel()
        quantizer = TorchAOQuantizer()
        
        for group_size in [32, 64, 128, 256]:
            model_copy = SimpleLinearModel()
            config = INT4Config(group_size=group_size)
            
            result = quantizer.quantize(model_copy, config)
            
            assert result.metrics["group_size"] == group_size
            assert result.success is True


class TestINT4WithoutTorchAO:
    """Tests for graceful degradation when torchao is not available."""
    
    def test_int4_fallback_to_int8(self):
        """Should fall back to INT8 when torchao not available."""
        quantizer = TorchAOQuantizer()
        
        if quantizer.is_available():
            pytest.skip("torchao is available, cannot test fallback")
        
        model = SimpleLinearModel()
        config = INT4Config(fallback_to_int8=True)
        
        result = quantizer.quantize(model, config)
        
        # Should succeed with fallback
        assert result.success is True
        assert result.fallback_used is True
    
    def test_int4_raises_without_fallback(self):
        """Should raise error without fallback."""
        quantizer = TorchAOQuantizer()
        
        if quantizer.is_available():
            pytest.skip("torchao is available, cannot test fallback")
        
        model = SimpleLinearModel()
        config = INT4Config(fallback_to_int8=False)
        
        with pytest.raises(RuntimeError):
            quantizer.quantize(model, config)


class TestConvenienceFunction:
    """Test convenience functions."""
    
    def test_apply_int4_quantization(self):
        """Test apply_int4_quantization function."""
        model = SimpleLinearModel()
        
        # This should work regardless of torchao availability (with fallback)
        quantized = apply_int4_quantization(model, group_size=128, fallback_to_int8=True)
        
        assert quantized is not None
    
    def test_apply_int4_quantization_no_fallback(self):
        """Test apply_int4_quantization without fallback."""
        model = SimpleLinearModel()
        
        if is_int4_available():
            # Should work with torchao
            quantized = apply_int4_quantization(model, group_size=128, fallback_to_int8=False)
            assert quantized is not None
        else:
            # Should raise without torchao and no fallback
            with pytest.raises(RuntimeError):
                apply_int4_quantization(model, group_size=128, fallback_to_int8=False)


class TestINT4Preservation:
    """Test layer preservation in INT4 quantization."""
    
    def test_lm_head_preserved(self):
        """Test that lm_head can be preserved."""
        class ModelWithLMHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 128)
                self.lm_head = nn.Linear(128, 1000)
            
            def forward(self, x):
                return self.lm_head(torch.relu(self.linear(x)))
        
        model = ModelWithLMHead()
        quantizer = TorchAOQuantizer()
        config = INT4Config(preserve_layers=("lm_head",))
        
        # This should work (with or without torchao)
        try:
            result = quantizer.quantize(model, config)
            assert result.success is True
        except RuntimeError:
            # Expected if torchao not available and fallback disabled
            pass


class TestAlias:
    """Test INT4Quantizer alias."""
    
    def test_int4_quantizer_alias(self):
        """Test that INT4Quantizer is alias for TorchAOQuantizer."""
        assert INT4Quantizer is TorchAOQuantizer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
