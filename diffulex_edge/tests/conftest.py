"""Pytest configuration and fixtures for DiffuLex Edge tests.

Provides shared fixtures for:
- Diffusion sampling tests
- Model testing
- Performance benchmarking
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runtime.diffusion import (
    DiffusionBlock,
    DiffusionBlockManager,
    DiffusionSampler,
    DiffusionEngine,
    DiffusionGenerationConfig,
    SampleOutput,
)


# ============================================================================
# Diffusion Block Fixtures
# ============================================================================

@pytest.fixture
def empty_block_manager():
    """Return an empty DiffusionBlockManager."""
    return DiffusionBlockManager()


@pytest.fixture
def populated_block_manager():
    """Return a DiffusionBlockManager with multiple blocks."""
    manager = DiffusionBlockManager(mask_token_id=126336)
    
    # Create 3 blocks of different sizes
    manager.create_block(start_pos=0, length=10)    # Block 0: positions 0-9
    manager.create_block(start_pos=15, length=5)    # Block 1: positions 15-19
    manager.create_block(start_pos=30, length=8)    # Block 2: positions 30-37
    
    return manager


@pytest.fixture
def completed_block():
    """Return a DiffusionBlock that is completely filled."""
    block = DiffusionBlock(start_pos=0, length=3, mask_token_id=100)
    block.accept_token(0, 1000)
    block.accept_token(1, 1001)
    block.accept_token(2, 1002)
    return block


@pytest.fixture
def partially_completed_block():
    """Return a DiffusionBlock that is partially filled."""
    block = DiffusionBlock(start_pos=10, length=5, mask_token_id=100)
    block.accept_token(1, 101)  # Global position 11
    block.accept_token(3, 103)  # Global position 13
    return block


# ============================================================================
# Diffusion Sampler Fixtures
# ============================================================================

@pytest.fixture
def default_sampler():
    """Return a DiffusionSampler with default parameters."""
    return DiffusionSampler()


@pytest.fixture
def high_confidence_sampler():
    """Return a DiffusionSampler with high confidence threshold."""
    return DiffusionSampler(
        confidence_threshold=0.99,
        temperature=1.0,
    )


@pytest.fixture
def low_temperature_sampler():
    """Return a DiffusionSampler with low temperature (more deterministic)."""
    return DiffusionSampler(
        temperature=0.01,
        top_k=1,
    )


@pytest.fixture
def top_k_sampler():
    """Return a DiffusionSampler with top-k filtering."""
    return DiffusionSampler(
        top_k=50,
        temperature=1.0,
    )


@pytest.fixture
def top_p_sampler():
    """Return a DiffusionSampler with top-p (nucleus) filtering."""
    return DiffusionSampler(
        top_p=0.9,
        temperature=1.0,
    )


@pytest.fixture
def sample_logits():
    """Return sample logits tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(10, 1000)  # 10 positions, 1000 vocab


@pytest.fixture
def extreme_logits():
    """Return logits with extreme values for testing numerical stability."""
    return torch.tensor([
        [100.0, 1.0, 1.0, 1.0],  # Very dominant first token
        [1.0, 100.0, 1.0, 1.0],  # Very dominant second token
        [1.0, 1.0, 100.0, 1.0],  # Very dominant third token
    ])


# ============================================================================
# Diffusion Engine Fixtures
# ============================================================================

class MockModel(nn.Module):
    """Simple mock model for testing."""
    
    def __init__(self, vocab_size: int = 130000, hidden_size: int = 128):
        super().__init__()
        # Large vocab_size to accommodate mask_token_id (126336)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x = self.embed(input_ids)  # [batch, seq, hidden]
        logits = self.head(x)  # [batch, seq, vocab]
        return logits, None


@pytest.fixture
def mock_model():
    """Return a simple mock model."""
    return MockModel()


@pytest.fixture
def diffusion_engine(mock_model):
    """Return a DiffusionEngine with mock model."""
    return DiffusionEngine(model=mock_model, device="cpu")


@pytest.fixture
def default_generation_config():
    """Return a default GenerationConfig."""
    return DiffusionGenerationConfig()


@pytest.fixture
def fast_generation_config():
    """Return a GenerationConfig optimized for fast testing."""
    return DiffusionGenerationConfig(
        max_new_tokens=10,
        num_iterations=2,
        block_size=5,
        confidence_threshold=0.0,  # Accept all
    )


@pytest.fixture
def deterministic_generation_config():
    """Return a GenerationConfig for deterministic testing."""
    return DiffusionGenerationConfig(
        max_new_tokens=10,
        num_iterations=3,
        block_size=5,
        temperature=0.01,  # Nearly deterministic
        top_k=1,
    )


# ============================================================================
# Prompt Fixtures
# ============================================================================

@pytest.fixture
def short_prompt():
    """Return a short prompt."""
    return [1, 2, 3]


@pytest.fixture
def medium_prompt():
    """Return a medium-length prompt."""
    return list(range(50))


@pytest.fixture
def long_prompt():
    """Return a long prompt."""
    return list(range(200))


@pytest.fixture
def empty_prompt():
    """Return an empty prompt."""
    return []


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance tests"
    )
    config.addinivalue_line(
        "markers", "numerical: marks numerical precision tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test file
    for item in items:
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "numerical" in item.nodeid:
            item.add_marker(pytest.mark.numerical)
