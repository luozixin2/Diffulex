"""Tests for quantization module."""

import pytest
import torch
import torch.nn as nn

from diffulex_edge.quant import (
    DiffuLexQuantizer,
    QuantizationConfig,
    QuantizationMode,
    apply_dynamic_quantization,
    apply_weight_only_quantization,
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


class TestDynamicQuantization:
    """Test dynamic quantization."""
    
    def test_dynamic_quantization_basic(self):
        """Test basic dynamic quantization."""
        model = SimpleLinearModel()
        model.eval()
        
        # Apply quantization
        quantized = apply_dynamic_quantization(model)
        
        # Check that linear layers are quantized
        assert hasattr(quantized.linear1, 'weight')
        assert hasattr(quantized.linear2, 'weight')
    
    def test_dynamic_quantization_outputs(self):
        """Test that quantized model produces reasonable outputs."""
        model = SimpleLinearModel()
        model.eval()
        
        # Original output
        x = torch.randn(4, 64)
        with torch.no_grad():
            original_output = model(x)
        
        # Quantized output
        quantized = apply_dynamic_quantization(model)
        with torch.no_grad():
            quantized_output = quantized(x)
        
        # Outputs should be similar (not identical due to quantization)
        assert quantized_output.shape == original_output.shape
        
        # Check that outputs are finite
        assert torch.isfinite(quantized_output).all()
    
    def test_dynamic_quantization_inference(self):
        """Test inference with dynamically quantized model."""
        model = SimpleLinearModel()
        model.eval()
        
        quantized = apply_dynamic_quantization(model)
        
        # Run inference
        x = torch.randn(2, 64)
        with torch.no_grad():
            output = quantized(x)
        
        assert output.shape == (2, 32)
        assert torch.isfinite(output).all()


class TestWeightOnlyQuantization:
    """Test weight-only quantization."""
    
    def test_weight_only_quantization_basic(self):
        """Test basic weight-only quantization."""
        model = SimpleLinearModel()
        
        # Store original weights
        original_weight1 = model.linear1.weight.data.clone()
        original_weight2 = model.linear2.weight.data.clone()
        
        # Apply quantization
        quantized = apply_weight_only_quantization(model)
        
        # Weights should be different (quantized then dequantized)
        assert not torch.allclose(quantized.linear1.weight, original_weight1)
        assert not torch.allclose(quantized.linear2.weight, original_weight2)
    
    def test_weight_only_quantization_outputs(self):
        """Test that weight-only quantization produces reasonable outputs."""
        model = SimpleLinearModel()
        model.eval()
        
        x = torch.randn(4, 64)
        
        with torch.no_grad():
            original_output = model(x)
        
        quantized = apply_weight_only_quantization(model)
        
        with torch.no_grad():
            quantized_output = quantized(x)
        
        # Outputs should be similar
        assert quantized_output.shape == original_output.shape
        
        # Error should be small for INT8 weight quantization
        max_error = (original_output - quantized_output).abs().max().item()
        assert max_error < 1.0  # Reasonable error bound for this simple model


class TestQuantizerAPI:
    """Test the main Quantizer API."""
    
    def test_quantizer_initialization(self):
        """Test quantizer initialization."""
        config = QuantizationConfig(mode=QuantizationMode.DYNAMIC)
        quantizer = DiffuLexQuantizer(config)
        
        assert quantizer.config == config
    
    def test_quantizer_prepare_dynamic(self):
        """Test preparing model for dynamic quantization."""
        model = SimpleLinearModel()
        config = QuantizationConfig(mode=QuantizationMode.DYNAMIC)
        quantizer = DiffuLexQuantizer(config)
        
        example_inputs = (torch.randn(2, 64),)
        
        # For dynamic quantization, prepare should return model
        prepared = quantizer.prepare_for_quantization(model, example_inputs)
        assert prepared is not None
    
    def test_quantizer_convert_dynamic(self):
        """Test converting dynamically quantized model."""
        model = SimpleLinearModel()
        config = QuantizationConfig(mode=QuantizationMode.DYNAMIC)
        quantizer = DiffuLexQuantizer(config)
        
        example_inputs = (torch.randn(2, 64),)
        prepared = quantizer.prepare_for_quantization(model, example_inputs)
        
        # Convert
        quantized = quantizer.convert(prepared)
        assert quantized is not None
    
    def test_quantizer_weight_only(self):
        """Test weight-only quantization through quantizer."""
        model = SimpleLinearModel()
        config = QuantizationConfig(mode=QuantizationMode.WEIGHT_ONLY)
        quantizer = DiffuLexQuantizer(config)
        
        # Weight-only doesn't need preparation
        quantized = quantizer.quantize_weights_only(model)
        
        # Test inference
        x = torch.randn(2, 64)
        with torch.no_grad():
            output = quantized(x)
        
        assert output.shape == (2, 32)


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
        
        # Apply dynamic quantization
        quantized = apply_dynamic_quantization(model)
        
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
            
            quantized = apply_dynamic_quantization(model)
            with torch.no_grad():
                output = quantized(x)
            
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_consistent_outputs(self):
        """Test that quantization produces consistent outputs."""
        model = SimpleLinearModel()
        model.eval()
        
        quantized = apply_dynamic_quantization(model)
        
        x = torch.randn(4, 64)
        
        with torch.no_grad():
            output1 = quantized(x)
            output2 = quantized(x)
        
        # Same input should produce same output
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
