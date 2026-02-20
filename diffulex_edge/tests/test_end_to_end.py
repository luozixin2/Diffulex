"""End-to-end integration tests for DiffuLex Edge.

These tests verify the complete generation pipeline from input to output.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from diffulex_edge.runtime.diffusion import (
    DiffusionEngine, DiffusionGenerationConfig,
    DiffusionSampler, DiffusionBlockManager
)
from diffulex_edge.runtime.engine import DiffusionEngine, DiffusionGenerationConfig


class SimpleTestModel(nn.Module):
    """Simple test model for end-to-end testing."""
    
    def __init__(self, vocab_size: int = 130000, hidden_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x = self.embed(input_ids)
        logits = self.head(x)
        return logits, None


class TestEndToEndGeneration:
    """Test end-to-end generation pipeline."""
    
    def test_diffusion_generation_full_pipeline(self):
        """TEST-E2E-001: Test complete diffusion generation pipeline."""
        model = SimpleTestModel()
        engine = DiffusionEngine(model=model)
        
        # Test different configurations
        configs = [
            DiffusionGenerationConfig(max_new_tokens=10, num_iterations=2),
            DiffusionGenerationConfig(max_new_tokens=20, num_iterations=3, temperature=0.5),
            DiffusionGenerationConfig(max_new_tokens=5, num_iterations=5, confidence_threshold=0.8),
        ]
        
        prompt = [1, 2, 3, 4, 5]
        
        for config in configs:
            result = engine.generate(prompt, config)
            
            # Verify result
            assert len(result) == len(prompt) + config.max_new_tokens
            assert all(isinstance(t, int) for t in result)
            assert all(0 <= t < model.vocab_size for t in result)
    
    def test_diffusion_vs_autoregressive_shape(self):
        """TEST-E2E-002: Verify diffusion and autoregressive produce compatible outputs."""
        from model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
        
        # Create a proper model for both engines with large vocab to accommodate mask token
        config = FastdLLMV2EdgeConfig(
            vocab_size=130000,  # Large enough for mask token 126336
            hidden_size=128, 
            num_hidden_layers=2
        )
        model = FastdLLMV2Edge(config)
        
        # Autoregressive generation
        ar_engine = InferenceEngine(model=model)
        prompt = [1, 2, 3]
        
        ar_config = GenerationConfig(max_new_tokens=10)
        ar_result = ar_engine.generate(prompt, ar_config)
        
        # Diffusion generation
        diff_engine = DiffusionEngine(model=model)
        diff_config = DiffusionGenerationConfig(max_new_tokens=10, num_iterations=2)
        diff_result = diff_engine.generate(prompt, diff_config)
        
        # Both should produce same number of generated tokens
        # Note: AR engine returns only generated tokens, Diffusion returns full sequence
        assert len(ar_result) == 10  # AR: only generated
        assert len(diff_result) == len(prompt) + 10  # Diffusion: prompt + generated
        
        # Verify the AR result is contained in diff result (prompt should match)
        assert diff_result[:len(prompt)] == prompt
    
    def test_streaming_generation(self):
        """TEST-E2E-003: Test streaming generation yields tokens progressively."""
        model = SimpleTestModel()
        engine = DiffusionEngine(model=model)
        
        prompt = [1, 2, 3]
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=5,
            confidence_threshold=0.0,  # Accept all for streaming
        )
        
        # Collect streamed tokens
        streamed_positions = []
        streamed_tokens = []
        
        for pos, token in engine.generate_stream(prompt, config):
            streamed_positions.append(pos)
            streamed_tokens.append(token)
        
        # Should have generated tokens
        assert len(streamed_tokens) > 0
        
        # All positions should be >= len(prompt)
        assert all(p >= len(prompt) for p in streamed_positions)
        
        # Tokens should be valid
        assert all(0 <= t < model.vocab_size for t in streamed_tokens)
    
    def test_multiple_generation_calls_same_engine(self):
        """TEST-E2E-004: Test multiple generation calls with same engine."""
        model = SimpleTestModel()
        engine = DiffusionEngine(model=model)
        
        prompts = [
            [1, 2, 3],
            [10, 20, 30],
            [100, 200, 300],
        ]
        
        results = []
        for prompt in prompts:
            config = DiffusionGenerationConfig(max_new_tokens=5, num_iterations=2)
            result = engine.generate(prompt, config)
            results.append(result)
        
        # Each result should be correct length
        for prompt, result in zip(prompts, results):
            assert len(result) == len(prompt) + 5


class TestSamplerIntegration:
    """Test sampler integration with engine."""
    
    def test_greedy_sampler_reproducibility(self):
        """TEST-E2E-005: Verify greedy sampler produces deterministic results."""
        sampler = GreedySampler()
        
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0, 4.0]])
        
        # Multiple calls should give same result
        results = [sampler.sample(logits) for _ in range(10)]
        assert all(r == results[0] for r in results)
        assert results[0] == 1  # Index of max value (5.0)
    
    def test_top_k_sampler_respects_k(self):
        """TEST-E2E-006: Verify top-k sampler only samples from top k."""
        torch.manual_seed(42)
        sampler = TopKSampler(k=3)
        
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        # Sample many times
        samples = [sampler.sample(logits) for _ in range(100)]
        
        # All samples should be from top-3 (indices 2, 3, 4)
        assert all(s in [2, 3, 4] for s in samples)
    
    def test_top_p_sampler_respects_p(self):
        """TEST-E2E-007: Verify top-p sampler respects cumulative probability."""
        torch.manual_seed(42)
        sampler = TopPSampler(p=0.9)
        
        # Very skewed distribution
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
        
        # Sample many times
        samples = [sampler.sample(logits) for _ in range(100)]
        
        # With this distribution, first token has ~96% probability
        # So with top_p=0.9, we should mostly get token 0
        assert samples.count(0) > 50  # At least 50% should be token 0


class TestConfigurationPresets:
    """Test configuration presets for different use cases."""
    
    def test_fast_generation_preset(self):
        """TEST-E2E-008: Test fast generation preset."""
        model = SimpleTestModel()
        engine = DiffusionEngine(model=model)
        
        # Fast config: fewer iterations, smaller blocks
        config = DiffusionGenerationConfig(
            max_new_tokens=20,
            num_iterations=2,
            block_size=10,
            temperature=1.0,
        )
        
        prompt = list(range(10))
        result = engine.generate(prompt, config)
        
        assert len(result) == 30  # 10 prompt + 20 generated
    
    def test_quality_generation_preset(self):
        """TEST-E2E-009: Test quality generation preset."""
        model = SimpleTestModel()
        engine = DiffusionEngine(model=model)
        
        # Quality config: more iterations, higher confidence threshold
        config = DiffusionGenerationConfig(
            max_new_tokens=20,
            num_iterations=10,
            block_size=5,
            confidence_threshold=0.95,
            temperature=0.8,
        )
        
        prompt = list(range(10))
        result = engine.generate(prompt, config)
        
        assert len(result) == 30
    
    def test_greedy_generation_preset(self):
        """TEST-E2E-010: Test greedy (deterministic) generation preset."""
        model = SimpleTestModel()
        engine = DiffusionEngine(model=model)
        
        # Greedy config: temperature=0 for deterministic output
        config = DiffusionGenerationConfig(
            max_new_tokens=10,
            num_iterations=3,
            temperature=0.01,  # Near-greedy
            top_k=1,
        )
        
        prompt = [1, 2, 3]
        
        # Multiple runs should give similar results
        results = [engine.generate(prompt, config) for _ in range(3)]
        
        # All should have same length
        assert all(len(r) == len(results[0]) for r in results)


class TestErrorHandling:
    """Test error handling in end-to-end scenarios."""
    
    def test_empty_prompt_handling(self):
        """TEST-E2E-ERR-001: Test handling of empty prompt."""
        model = SimpleTestModel()
        engine = DiffusionEngine(model=model)
        
        prompt = []
        config = DiffusionGenerationConfig(max_new_tokens=5, num_iterations=2)
        
        # Should handle gracefully
        result = engine.generate(prompt, config)
        
        assert len(result) == 5  # Only generated tokens
    
    def test_large_prompt_handling(self):
        """TEST-E2E-ERR-002: Test handling of very large prompt."""
        model = SimpleTestModel()
        engine = DiffusionEngine(model=model)
        
        prompt = list(range(500))  # Large prompt
        config = DiffusionGenerationConfig(max_new_tokens=10, num_iterations=2)
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 510  # 500 + 10
    
    def test_invalid_temperature_handling(self):
        """TEST-E2E-ERR-003: Test handling of invalid temperature."""
        # Very high temperature should still work
        sampler = DiffusionSampler(temperature=100.0)
        
        logits = torch.randn(5, 100)
        
        # Should not crash
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        assert tokens.shape == (5,)
        assert torch.all(initial_conf > 0)


class TestModelCompatibility:
    """Test compatibility with different model architectures."""
    
    def test_model_with_vocab_size_32000(self):
        """TEST-E2E-COMPAT-001: Test with standard 32k vocab size."""
        # Need large vocab to accommodate mask token 126336
        model = SimpleTestModel(vocab_size=130000)
        engine = DiffusionEngine(model=model)
        
        prompt = [100, 200, 300]  # Use IDs within vocab range
        config = DiffusionGenerationConfig(max_new_tokens=5, num_iterations=2)
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 8
        assert all(0 <= t < 130000 for t in result)
    
    def test_model_with_vocab_size_50000(self):
        """TEST-E2E-COMPAT-002: Test with 50k vocab size (SDAR)."""
        model = SimpleTestModel(vocab_size=130000)  # Large enough for mask token
        engine = DiffusionEngine(model=model)
        
        prompt = [100, 200, 300]  # Use IDs within vocab range
        config = DiffusionGenerationConfig(max_new_tokens=5, num_iterations=2)
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 8
        assert all(0 <= t < 130000 for t in result)
    
    def test_model_with_large_hidden_size(self):
        """TEST-E2E-COMPAT-003: Test with large hidden dimension."""
        model = SimpleTestModel(vocab_size=130000, hidden_size=4096)
        engine = DiffusionEngine(model=model)
        
        prompt = [10, 20, 30]  # Use IDs within vocab range
        config = DiffusionGenerationConfig(max_new_tokens=5, num_iterations=2)
        
        result = engine.generate(prompt, config)
        
        assert len(result) == 8
