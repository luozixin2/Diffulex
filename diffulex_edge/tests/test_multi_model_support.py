"""Tests for multi-model support (Phase 7).

Test Plan Coverage:
- Dream model support: 4 tests
- LLaDA model support: 4 tests
- SDAR model support: 4 tests
- Auto model selection: 2 tests
- Cross-model consistency: 1 test

Total: 15 tests
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runtime.diffusion import (
    DiffusionEngine, DiffusionGenerationConfig,
    DiffusionSampler, DiffusionBlockManager
)


# ============================================================================
# Model Configurations for Different dLLM Architectures
# ============================================================================

@dataclass
class DreamModelConfig:
    """Configuration for Dream model."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    max_length: int = 2048
    noise_schedule: str = "cosine"
    num_diffusion_steps: int = 50


@dataclass
class LLaDAModelConfig:
    """Configuration for LLaDA model."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 24
    max_length: int = 4096
    mask_token_id: int = 126336
    confidence_threshold: float = 0.9


@dataclass
class SDARModelConfig:
    """Configuration for SDAR model."""
    vocab_size: int = 50000
    hidden_size: int = 2048
    num_layers: int = 20
    max_length: int = 1024
    diffusion_steps: int = 32
    schedule: str = "linear"


# ============================================================================
# Mock Models for Different Architectures
# ============================================================================

class MockDreamModel(nn.Module):
    """Mock Dream model with cosine noise schedule."""
    
    def __init__(self, config: DreamModelConfig = None):
        super().__init__()
        self.config = config or DreamModelConfig()
        self.model_type = "dream"
        # Ensure vocab_size is large enough for mask_token_id
        if self.config.vocab_size < 130000:
            self.config.vocab_size = 130000
        
        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=16,
                dim_feedforward=4096,
                batch_first=True
            ),
            num_layers=4  # Reduced for testing
        )
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Forward pass."""
        x = self.embed(input_ids)  # [batch, seq, hidden]
        x = self.transformer(x)
        logits = self.head(x)  # [batch, seq, vocab]
        return logits, None
    
    def get_noise_schedule(self, timesteps: int) -> torch.Tensor:
        """Get cosine noise schedule."""
        steps = torch.arange(timesteps + 1)
        alphas = torch.cos(((steps / timesteps) + 0.008) / 1.008 * torch.pi / 2) ** 2
        return alphas


class MockLLaDAModel(nn.Module):
    """Mock LLaDA model with masking strategy."""
    
    def __init__(self, config: LLaDAModelConfig = None):
        super().__init__()
        self.config = config or LLaDAModelConfig()
        self.model_type = "llada"
        # Ensure vocab_size is large enough for mask_token_id
        if self.config.vocab_size < 130000:
            self.config.vocab_size = 130000
        
        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=16,
                dim_feedforward=4096,
                batch_first=True
            )
            for _ in range(4)  # Reduced for testing
        ])
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Forward pass."""
        x = self.embed(input_ids)  # [batch, seq, hidden]
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)  # [batch, seq, vocab]
        return logits, None


class MockSDARModel(nn.Module):
    """Mock SDAR model with step-wise diffusion."""
    
    def __init__(self, config: SDARModelConfig = None):
        super().__init__()
        self.config = config or SDARModelConfig()
        self.model_type = "sdar"
        # Ensure vocab_size is large enough for mask_token_id
        if self.config.vocab_size < 130000:
            self.config.vocab_size = 130000
        
        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.encoder = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Forward pass."""
        x = self.embed(input_ids)  # [batch, seq, hidden]
        x, _ = self.encoder(x)
        logits = self.head(x)  # [batch, seq, vocab]
        return logits, None
    
    def get_step_schedule(self) -> torch.Tensor:
        """Get linear step schedule."""
        return torch.linspace(1.0, 0.0, self.config.diffusion_steps + 1)


# ============================================================================
# Dream Model Support Tests (4 tests)
# ============================================================================

class TestDreamModelSupport:
    """Test Dream model support."""
    
    def test_dream_model_initialization(self):
        """TEST-MODEL-DREAM-001: Verify Dream model can be initialized."""
        config = DreamModelConfig()
        model = MockDreamModel(config)
        
        assert model.model_type == "dream"
        assert model.config.vocab_size == 32000
        assert model.config.noise_schedule == "cosine"
    
    def test_dream_model_forward_pass(self):
        """TEST-MODEL-DREAM-002: Verify Dream model forward pass works."""
        model = MockDreamModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3, 4, 5]
        config = DiffusionGenerationConfig(max_new_tokens=10, num_iterations=2)
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 15
        assert all(0 <= t < model.config.vocab_size for t in result)
    
    def test_dream_model_noise_schedule(self):
        """TEST-MODEL-DREAM-003: Verify Dream noise schedule is accessible."""
        model = MockDreamModel()
        schedule = model.get_noise_schedule(50)
        
        assert len(schedule) == 51  # timesteps + 1
        assert schedule[0] > schedule[-1]  # Decreasing schedule
        assert torch.all(schedule >= 0) and torch.all(schedule <= 1)
    
    def test_dream_model_generation_with_custom_steps(self):
        """TEST-MODEL-DREAM-004: Verify Dream model with custom diffusion steps."""
        model = MockDreamModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        
        for num_steps in [10, 20, 50]:
            config = DiffusionGenerationConfig(
                max_new_tokens=10,
                num_iterations=num_steps,
                block_size=5,
            )
            
            result = engine.generate(prompt, config)
            
            assert len(result) == 13


# ============================================================================
# LLaDA Model Support Tests (4 tests)
# ============================================================================

class TestLLaDAModelSupport:
    """Test LLaDA model support."""
    
    def test_llada_model_initialization(self):
        """TEST-MODEL-LLADA-001: Verify LLaDA model can be initialized."""
        config = LLaDAModelConfig()
        model = MockLLaDAModel(config)
        
        assert model.model_type == "llada"
        assert model.config.mask_token_id == 126336
        assert model.config.confidence_threshold == 0.9
    
    def test_llada_model_forward_pass(self):
        """TEST-MODEL-LLADA-002: Verify LLaDA model forward pass works."""
        model = MockLLaDAModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3, 4, 5]
        config = DiffusionGenerationConfig(max_new_tokens=10, num_iterations=2)
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 15
    
    def test_llada_model_with_mask_token(self):
        """TEST-MODEL-LLADA-003: Verify LLaDA model handles mask token correctly."""
        model = MockLLaDAModel()
        
        # Verify mask token ID is accessible
        mask_id = model.config.mask_token_id
        assert mask_id == 126336
        
        # Verify engine can use this mask token
        engine = DiffusionEngine(model=model)
        engine.sampler = DiffusionSampler(mask_token_id=mask_id)
        
        assert engine.sampler.mask_token_id == mask_id
    
    def test_llada_model_high_confidence_threshold(self):
        """TEST-MODEL-LLADA-004: Verify LLaDA model with high confidence threshold."""
        model = MockLLaDAModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=10,
            confidence_threshold=0.99,  # Very high threshold
            block_size=5,
        )
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 13


# ============================================================================
# SDAR Model Support Tests (4 tests)
# ============================================================================

class TestSDARModelSupport:
    """Test SDAR model support."""
    
    def test_sdar_model_initialization(self):
        """TEST-MODEL-SDAR-001: Verify SDAR model can be initialized."""
        config = SDARModelConfig()
        model = MockSDARModel(config)
        
        assert model.model_type == "sdar"
        assert model.config.diffusion_steps == 32
        assert model.config.schedule == "linear"
    
    def test_sdar_model_forward_pass(self):
        """TEST-MODEL-SDAR-002: Verify SDAR model forward pass works."""
        model = MockSDARModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3, 4, 5]
        config = DiffusionGenerationConfig(max_new_tokens=10, num_iterations=2)
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 15
    
    def test_sdar_model_step_schedule(self):
        """TEST-MODEL-SDAR-003: Verify SDAR step schedule is accessible."""
        model = MockSDARModel()
        schedule = model.get_step_schedule()
        
        assert len(schedule) == 33  # diffusion_steps + 1
        assert schedule[0] == 1.0
        assert schedule[-1] == 0.0
        # Linear decrease
        for i in range(len(schedule) - 1):
            assert schedule[i] >= schedule[i + 1]
    
    def test_sdar_model_with_variable_steps(self):
        """TEST-MODEL-SDAR-004: Verify SDAR model with variable diffusion steps."""
        model = MockSDARModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        
        for steps in [16, 32, 64]:
            config = DiffusionGenerationConfig(
                max_new_tokens=10,
                num_iterations=steps,
                block_size=5,
            )
            
            result = engine.generate(prompt, config)
            
            assert len(result) == 13


# ============================================================================
# Auto Model Selection Tests (2 tests)
# ============================================================================

def get_model_type_from_config(config: Dict[str, Any]) -> str:
    """Determine model type from configuration."""
    if "dream" in config.get("architecture", "").lower():
        return "dream"
    elif "llada" in config.get("architecture", "").lower():
        return "llada"
    elif "sdar" in config.get("architecture", "").lower():
        return "sdar"
    else:
        return "unknown"


class TestAutoModelSelection:
    """Test automatic model selection."""
    
    def test_auto_model_selection_dream(self):
        """TEST-MODEL-AUTO-001: Verify auto-selection for Dream model."""
        config = {"architecture": "dream", "vocab_size": 32000}
        model_type = get_model_type_from_config(config)
        
        assert model_type == "dream"
    
    def test_auto_model_selection_variants(self):
        """TEST-MODEL-AUTO-002: Verify auto-selection for different model variants."""
        test_cases = [
            ({"architecture": "llada"}, "llada"),
            ({"architecture": "LLaDA"}, "llada"),
            ({"architecture": "sdar"}, "sdar"),
            ({"architecture": "SDAR"}, "sdar"),
            ({"architecture": "unknown"}, "unknown"),
            ({}, "unknown"),
        ]
        
        for config, expected in test_cases:
            result = get_model_type_from_config(config)
            assert result == expected, f"Failed for config {config}"


# ============================================================================
# Cross-Model Consistency Test (1 test)
# ============================================================================

class TestCrossModelConsistency:
    """Test consistency across different model types."""
    
    def test_all_models_produce_valid_output(self):
        """TEST-MODEL-CROSS-001: Verify all model types produce valid output."""
        models = [
            ("dream", MockDreamModel()),
            ("llada", MockLLaDAModel()),
            ("sdar", MockSDARModel()),
        ]
        
        prompt = [1, 2, 3, 4, 5]
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=3,
            block_size=5,
        )
        
        results = {}
        
        for name, model in models:
            engine = DiffusionEngine(model=model)
            result = engine.generate(prompt, config)
            results[name] = result
            
            # Verify result
            assert len(result) == 15, f"{name} produced wrong length"
            assert all(isinstance(t, int) for t in result), f"{name} produced non-int"
            assert all(t >= 0 for t in result), f"{name} produced negative token"
        
        # All models should produce results of same length
        assert len(set(len(r) for r in results.values())) == 1


# ============================================================================
# Model Registry and Factory Tests
# ============================================================================

class ModelRegistry:
    """Simple model registry for testing."""
    
    _models = {}
    
    @classmethod
    def register(cls, name: str, model_class):
        """Register a model class."""
        cls._models[name] = model_class
    
    @classmethod
    def get(cls, name: str):
        """Get a model class by name."""
        return cls._models.get(name)
    
    @classmethod
    def create(cls, name: str, config=None):
        """Create a model instance."""
        model_class = cls.get(name)
        if model_class is None:
            raise ValueError(f"Unknown model: {name}")
        return model_class(config)


# Register models
ModelRegistry.register("dream", MockDreamModel)
ModelRegistry.register("llada", MockLLaDAModel)
ModelRegistry.register("sdar", MockSDARModel)


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_model_registry_creation(self):
        """Test creating models through registry."""
        for name in ["dream", "llada", "sdar"]:
            model = ModelRegistry.create(name)
            assert model is not None
            assert hasattr(model, "forward")
    
    def test_model_registry_unknown_model(self):
        """Test registry raises for unknown model."""
        with pytest.raises(ValueError):
            ModelRegistry.create("unknown_model")


# ============================================================================
# Configuration Compatibility Tests
# ============================================================================

class TestConfigurationCompatibility:
    """Test configuration compatibility across models."""
    
    def test_dream_config_compatibility(self):
        """Test Dream config works with engine."""
        config = DreamModelConfig(
            vocab_size=50000,
            num_diffusion_steps=100
        )
        model = MockDreamModel(config)
        engine = DiffusionEngine(model=model)
        
        result = engine.generate(
            [1, 2, 3],
            DiffusionGenerationConfig(max_new_tokens=5, num_iterations=10)
        )
        
        assert len(result) == 8
    
    def test_llada_config_compatibility(self):
        """Test LLaDA config works with engine."""
        config = LLaDAModelConfig(
            vocab_size=100000,
            confidence_threshold=0.95
        )
        model = MockLLaDAModel(config)
        engine = DiffusionEngine(model=model)
        
        result = engine.generate(
            [1, 2, 3],
            DiffusionGenerationConfig(max_new_tokens=5, num_iterations=10)
        )
        
        assert len(result) == 8
    
    def test_sdar_config_compatibility(self):
        """Test SDAR config works with engine."""
        config = SDARModelConfig(
            diffusion_steps=64,
            schedule="cosine"
        )
        model = MockSDARModel(config)
        engine = DiffusionEngine(model=model)
        
        result = engine.generate(
            [1, 2, 3],
            DiffusionGenerationConfig(max_new_tokens=5, num_iterations=10)
        )
        
        assert len(result) == 8
