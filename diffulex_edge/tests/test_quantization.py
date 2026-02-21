"""Tests for quantization module."""

import pytest
import torch
import torch.nn as nn

from diffulex_edge.quant import (
    quantize_model,
    quantize_to_fp16,
    quantize_to_int8,
    QuantizationConfig,
    QuantizationDtype,
    QuantizationMode,
    WeightOnlyQuantizer,
)


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


class TestQuantizationConfig:
    """Test quantization configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = QuantizationConfig()
        assert config.mode == QuantizationMode.DYNAMIC
        assert config.is_per_channel is True
        assert config.is_qat is False
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = QuantizationConfig(mode=QuantizationMode.QAT)
        assert config.is_qat is True  # Auto-set when QAT mode
    
    def test_config_modes(self):
        """Test all quantization modes."""
        for mode in QuantizationMode:
            config = QuantizationConfig(mode=mode)
            assert config.mode == mode


class TestUnifiedAPI:
    """Test the unified quantize_model API."""
    
    def test_quantize_model_fp16(self):
        """Test quantize_model with FP16."""
        model = SimpleLinearModel()
        model.eval()
        
        # Quantize using unified API
        quantized = quantize_model(model, dtype="fp16")
        
        # Check model works
        x = torch.randn(2, 64)
        with torch.no_grad():
            output = quantized(x)
        
        assert output.shape == (2, 32)
        assert torch.isfinite(output).all()
    
    def test_quantize_model_int8(self):
        """Test quantize_model with INT8."""
        model = SimpleLinearModel()
        model.eval()
        
        # Quantize using unified API
        quantized = quantize_model(model, dtype="int8", mode="weight_only")
        
        # Check model works
        x = torch.randn(2, 64)
        with torch.no_grad():
            output = quantized(x)
        
        assert output.shape == (2, 32)
        assert torch.isfinite(output).all()
    
    def test_quantize_model_invalid_dtype(self):
        """Test quantize_model with invalid dtype."""
        model = SimpleLinearModel()
        
        with pytest.raises(ValueError):
            quantize_model(model, dtype="invalid")


class TestQuantizeToFP16:
    """Test FP16 quantization."""
    
    def test_fp16_basic(self):
        """Test basic FP16 quantization."""
        model = SimpleLinearModel()
        model.eval()
        
        quantized = quantize_to_fp16(model)
        
        # Check that model works
        x = torch.randn(2, 64)
        with torch.no_grad():
            output = quantized(x)
        
        assert output.shape == (2, 32)
    
    def test_fp16_outputs(self):
        """Test that FP16 model produces reasonable outputs."""
        model = SimpleLinearModel()
        model.eval()
        
        x = torch.randn(4, 64)
        
        with torch.no_grad():
            original_output = model(x)
        
        quantized = quantize_to_fp16(model)
        
        with torch.no_grad():
            quantized_output = quantized(x)
        
        # Outputs should be very similar for FP16
        assert quantized_output.shape == original_output.shape
        max_error = (original_output - quantized_output).abs().max().item()
        assert max_error < 0.01  # Very small error for FP16


class TestQuantizeToInt8:
    """Test INT8 quantization."""
    
    def test_int8_weight_only_basic(self):
        """Test basic INT8 weight-only quantization."""
        model = SimpleLinearModel()
        
        # Store original weights
        original_weight1 = model.linear1.weight.data.clone()
        original_weight2 = model.linear2.weight.data.clone()
        
        # Apply quantization
        quantized = quantize_to_int8(model, mode="weight_only")
        
        # Check that weights are now stored as INT8 (char type)
        assert quantized.linear1.weight.dtype == torch.int8
        assert quantized.linear2.weight.dtype == torch.int8
        
        # Check that weights have scale factors for dequantization
        assert hasattr(quantized.linear1, 'weight_scale')
        assert hasattr(quantized.linear2, 'weight_scale')
    
    def test_int8_outputs(self):
        """Test that INT8 quantization produces reasonable outputs."""
        model = SimpleLinearModel()
        model.eval()
        
        x = torch.randn(4, 64)
        
        with torch.no_grad():
            original_output = model(x)
        
        quantized = quantize_to_int8(model, mode="weight_only")
        
        with torch.no_grad():
            quantized_output = quantized(x)
        
        # Outputs should be similar
        assert quantized_output.shape == original_output.shape
        
        # Error should be small for INT8 weight quantization
        max_error = (original_output - quantized_output).abs().max().item()
        assert max_error < 1.0  # Reasonable error bound for this simple model


class TestWeightOnlyQuantizer:
    """Test the WeightOnlyQuantizer class."""
    
    def test_quantizer_initialization(self):
        """Test quantizer initialization."""
        quantizer = WeightOnlyQuantizer()
        assert quantizer.is_available() is True
    
    def test_quantizer_fp16(self):
        """Test quantizer with FP16."""
        model = SimpleLinearModel()
        config = QuantizationConfig(dtype=QuantizationDtype.FP16)
        quantizer = WeightOnlyQuantizer()
        
        result = quantizer.quantize(model, config)
        
        assert result.success is True
        assert result.model is not None
        assert "compression_ratio" in result.metrics
    
    def test_quantizer_int8(self):
        """Test quantizer with INT8."""
        model = SimpleLinearModel()
        config = QuantizationConfig(dtype=QuantizationDtype.INT8)
        quantizer = WeightOnlyQuantizer()
        
        result = quantizer.quantize(model, config)
        
        assert result.success is True
        assert result.model is not None
        assert result.metrics["compression_ratio"] > 3.0  # INT8 should give ~4x


class TestQuantizationWithKVCache:
    """Test quantization with KV cache models."""
    
    def test_model_quantization_compatible(self):
        """Test that our edge model can be quantized."""
        from diffulex_edge.model.fast_dllm_v2_edge import (
            FastdLLMV2Edge, FastdLLMV2EdgeConfig
        )
        
        config = FastdLLMV2EdgeConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
        
        model = FastdLLMV2Edge(config)
        model.eval()
        
        # Apply INT8 quantization
        quantized = quantize_to_int8(model, mode="weight_only")
        
        # Test inference with KV cache
        input_ids = torch.randint(0, 1000, (1, 8))
        positions = torch.arange(8).unsqueeze(0)
        kv_cache = torch.zeros(
            config.num_hidden_layers, 2, 1, config.num_key_value_heads,
            config.max_position_embeddings, config.head_dim
        )
        
        with torch.no_grad():
            output, new_kv = quantized(input_ids, positions, None, kv_cache, 0)
        
        assert output.shape == (1, 8, 1000)
        assert new_kv.shape == (config.num_hidden_layers, 2, 1, config.num_key_value_heads, 8, config.head_dim)


class TestQuantizationNumericalStability:
    """Test numerical stability of quantization."""
    
    def test_no_nan_outputs(self):
        """Test that quantization doesn't produce NaN."""
        model = SimpleLinearModel()
        model.eval()
        
        # Test with different inputs
        for _ in range(10):
            x = torch.randn(8, 64) * 10  # Larger inputs
            
            quantized = quantize_to_int8(model, mode="weight_only")
            with torch.no_grad():
                output = quantized(x)
            
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_consistent_outputs(self):
        """Test that quantization produces consistent outputs."""
        model = SimpleLinearModel()
        model.eval()
        
        quantized = quantize_to_int8(model, mode="weight_only")
        
        x = torch.randn(4, 64)
        
        with torch.no_grad():
            output1 = quantized(x)
            output2 = quantized(x)
        
        # Same input should produce same output
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
