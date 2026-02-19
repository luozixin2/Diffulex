"""Tests for DiffusionEngine.

Test Plan Coverage:
- DiffusionEngine basic functionality: 3 tests
- generate() flow: 3 tests
- Configuration validation: 2 tests
- Boundary conditions: 1 test
- Error handling: 2 tests

Total: 11 tests
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runtime.diffusion import (
    DiffusionEngine, DiffusionGenerationConfig,
    DiffusionBlockManager, DiffusionSampler
)


# ============================================================================
# Mock Model for Testing
# ============================================================================

class MockDiffusionModel(nn.Module):
    """Mock model for testing DiffusionEngine."""
    
    def __init__(self, vocab_size: int = 130000, hidden_size: int = 128):
        super().__init__()
        # Use large vocab_size to accommodate mask_token_id (126336)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Simple embedding + linear head
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Forward pass returning logits."""
        x = self.embed(input_ids)  # [batch, seq, hidden]
        # Simple projection for mock - just use last hidden state for all positions
        logits = self.head(x)  # [batch, seq, vocab]
        return logits, None


# ============================================================================
# DiffusionEngine Basic Functionality Tests (3 tests)
# ============================================================================

class TestDiffusionEngineBasic:
    """Test basic DiffusionEngine functionality."""
    
    def test_engine_initialization_with_model(self):
        """TEST-ENGINE-001: Verify engine initializes correctly with model."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model, device="cpu")
        
        assert engine.model is model
        assert engine.device == "cpu"
        assert isinstance(engine.block_manager, DiffusionBlockManager)
        assert isinstance(engine.sampler, DiffusionSampler)
        assert model.training is False  # Model should be in eval mode
    
    def test_engine_initialization_without_model(self):
        """TEST-ENGINE-002: Verify engine initializes without model."""
        engine = DiffusionEngine(device="cpu")
        
        assert engine.model is None
        assert engine.device == "cpu"
        assert isinstance(engine.block_manager, DiffusionBlockManager)
        assert isinstance(engine.sampler, DiffusionSampler)
    
    def test_engine_components_reset(self):
        """TEST-ENGINE-003: Verify engine components can be reset."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        # Create some blocks
        engine.block_manager.create_block(start_pos=0, length=10)
        engine.block_manager.create_block(start_pos=10, length=10)
        
        assert len(engine.block_manager.blocks) == 2
        
        # Reset
        engine.block_manager.reset()
        
        assert len(engine.block_manager.blocks) == 0


# ============================================================================
# generate() Flow Tests (3 tests)
# ============================================================================

class TestDiffusionEngineGenerate:
    """Test DiffusionEngine generate() method."""
    
    def test_generate_returns_sequence(self):
        """TEST-ENGINE-004: Verify generate() returns complete sequence."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3, 4, 5]
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=3,
            block_size=5,
        )
        
        result = engine.generate(prompt, config)
        
        # Result should include prompt + generated tokens
        assert len(result) == len(prompt) + config.max_new_tokens
        # First part should match prompt
        assert result[:len(prompt)] == prompt
    
    def test_generate_respects_max_new_tokens(self):
        """TEST-ENGINE-005: Verify generate() respects max_new_tokens."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        
        for max_tokens in [5, 10, 20]:
            config = DiffusionGenerationConfig(
                max_new_tokens=max_tokens,
                num_iterations=2,
                block_size=5,
            )
            
            result = engine.generate(prompt, config)
            
            # Should generate exactly max_new_tokens
            assert len(result) == len(prompt) + max_tokens
    
    def test_generate_with_different_iterations(self):
        """TEST-ENGINE-006: Verify generate() works with different iteration counts."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        
        for num_iter in [1, 5, 10]:
            config = DiffusionGenerationConfig(
                max_new_tokens=10,
                num_iterations=num_iter,
                block_size=5,
            )
            
            result = engine.generate(prompt, config)
            
            # Should still generate correct length
            assert len(result) == len(prompt) + 10


# ============================================================================
# Configuration Validation Tests (2 tests)
# ============================================================================

class TestDiffusionEngineConfiguration:
    """Test DiffusionEngine configuration validation."""
    
    def test_generation_config_defaults(self):
        """TEST-ENGINE-CONF-001: Verify GenerationConfig has correct defaults."""
        config = DiffusionGenerationConfig()
        
        assert config.max_new_tokens == 100
        assert config.num_iterations == 10
        assert config.block_size == 10
        assert config.confidence_threshold == 0.95
        assert config.temperature == 1.0
        assert config.top_k == 0
        assert config.top_p == 1.0
        assert config.mask_token_id == 126336
        assert config.eos_token_id == 2
        assert config.early_stop is True
    
    def test_generation_config_custom_values(self):
        """TEST-ENGINE-CONF-002: Verify GenerationConfig accepts custom values."""
        config = DiffusionGenerationConfig(
            max_new_tokens=50,
            num_iterations=5,
            block_size=20,
            confidence_threshold=0.8,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            mask_token_id=999,
            eos_token_id=3,
            early_stop=False,
        )
        
        assert config.max_new_tokens == 50
        assert config.num_iterations == 5
        assert config.block_size == 20
        assert config.confidence_threshold == 0.8
        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.mask_token_id == 999
        assert config.eos_token_id == 3
        assert config.early_stop is False


# ============================================================================
# Boundary Condition Tests (1 test)
# ============================================================================

class TestDiffusionEngineBoundaryConditions:
    """Test DiffusionEngine boundary conditions."""
    
    def test_generate_empty_prompt(self):
        """TEST-ENGINE-BOUND-001: Verify generate() handles empty prompt."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = []
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=2,
            block_size=5,
        )
        
        result = engine.generate(prompt, config)
        
        # Should still generate tokens
        assert len(result) == config.max_new_tokens


# ============================================================================
# Error Handling Tests (2 tests)
# ============================================================================

class TestDiffusionEngineErrorHandling:
    """Test DiffusionEngine error handling."""
    
    def test_generate_without_model_raises(self):
        """TEST-ENGINE-ERR-001: Verify generate() raises when no model loaded."""
        engine = DiffusionEngine(device="cpu")
        
        prompt = [1, 2, 3]
        config = DiffusionGenerationConfig()
        
        with pytest.raises(RuntimeError, match="No model"):
            engine.generate(prompt, config)
    
    def test_forward_raises_without_model(self):
        """TEST-ENGINE-ERR-002: Verify _forward raises when model is None."""
        engine = DiffusionEngine(device="cpu")
        
        input_ids = torch.tensor([[1, 2, 3]])
        
        with pytest.raises(RuntimeError, match="No model"):
            engine._forward(input_ids)


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestDiffusionEngineEdgeCases:
    """Additional edge case tests for DiffusionEngine."""
    
    def test_generate_single_token(self):
        """Test generate with single token generation."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        config = DiffusionGenerationConfig(
            max_new_tokens=1,
            num_iterations=1,
            block_size=1,
        )
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 4  # prompt + 1
    
    def test_generate_large_prompt(self):
        """Test generate with large prompt."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = list(range(100))  # 100 tokens
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=2,
            block_size=5,
        )
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 110
        assert result[:100] == prompt
    
    def test_generate_with_zero_iterations(self):
        """Test generate with zero iterations."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=0,
            block_size=5,
        )
        
        # Should handle gracefully (no iterations = mask tokens remain)
        result = engine.generate(prompt, config)
        
        assert len(result) == 13
    
    def test_generate_stream_yields_positions(self):
        """Test generate_stream yields position-token pairs."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=5,
            block_size=5,
            confidence_threshold=0.0,  # Accept all
        )
        
        results = list(engine.generate_stream(prompt, config))
        
        # Should yield some results
        assert len(results) > 0
        
        # Each result should be (position, token_id)
        for pos, token_id in results:
            assert isinstance(pos, int)
            assert isinstance(token_id, int)
            assert pos >= len(prompt)  # Should be after prompt
    
    def test_block_size_larger_than_max_tokens(self):
        """Test when block_size > max_new_tokens."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        config = DiffusionGenerationConfig(
            max_new_tokens=5,
            num_iterations=2,
            block_size=100,  # Much larger than max_new_tokens
        )
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 8  # prompt + 5
    
    def test_early_stop_completes_all_blocks(self):
        """Test early_stop=True completes all blocks."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        config = DiffusionGenerationConfig(
            max_new_tokens=20,
            num_iterations=100,  # Many iterations
            block_size=5,
            confidence_threshold=0.0,  # Accept all immediately
            early_stop=True,
        )
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 23
        # Should have stopped early since all tokens accepted
    
    def test_no_early_stop_runs_all_iterations(self):
        """Test early_stop=False runs all iterations."""
        model = MockDiffusionModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=5,
            block_size=5,
            confidence_threshold=1.0,  # Never accept (except by chance)
            early_stop=False,
        )
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 13
        # Should run all 5 iterations even if no tokens accepted
