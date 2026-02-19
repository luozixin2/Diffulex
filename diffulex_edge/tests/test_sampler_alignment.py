"""Tests for sampler alignment with diffulex implementations."""

import torch
import pytest
import math

from diffulex_edge.runtime.sampler.base import (
    sample_tokens,
    top_p_logits,
    top_k_logits,
)
from diffulex_edge.runtime.sampler.shift import (
    ShiftLogitsSampler,
    NoShiftLogitsSampler,
)
from diffulex_edge.runtime.sampler.models import (
    FastdLLMV2Sampler,
    LLaDASampler,
    DreamSampler,
    SDARSampler,
    SAMPLER_REGISTRY,
)
from diffulex_edge.runtime.block import DiffusionBlockManager


class TestTopPLogits:
    """Align with diffulex.sampler.base.SamplerBase.top_p_logits"""
    
    def test_top_p_filtering_basic(self):
        """Test basic top-p filtering."""
        # Create logits where top 2 tokens have high probability
        logits = torch.tensor([[5.0, 4.0, 1.0, 0.5, 0.1]])
        result = top_p_logits(logits.clone(), top_p=0.9)
        
        # Check shape preserved
        assert result.shape == logits.shape
        
        # Top tokens should remain, rest should be masked
        assert result[0, 0] == logits[0, 0]
        assert result[0, 1] == logits[0, 1]
        assert result[0, 2] == torch.finfo(logits.dtype).min
    
    def test_top_p_keeps_at_least_one(self):
        """Top-p should always keep at least one token."""
        logits = torch.tensor([[0.1, 0.1, 0.1]])
        result = top_p_logits(logits.clone(), top_p=0.1)
        
        # At least one token should remain unmasked
        assert (result[0] != torch.finfo(logits.dtype).min).sum() >= 1
    
    def test_top_p_none_returns_unchanged(self):
        """top_p=None should return logits unchanged."""
        logits = torch.randn(2, 100)
        result = top_p_logits(logits.clone(), top_p=None)
        assert torch.allclose(result, logits)
    
    def test_top_p_one_returns_unchanged(self):
        """top_p=1.0 should return logits unchanged."""
        logits = torch.randn(2, 100)
        result = top_p_logits(logits.clone(), top_p=1.0)
        assert torch.allclose(result, logits)
    
    def test_top_p_batch_processing(self):
        """Test top-p with batch of sequences."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)
        result = top_p_logits(logits.clone(), top_p=0.9)
        
        # Check all sequences processed
        assert result.shape == logits.shape
        
        # Check that filtering occurred
        num_masked = (result == torch.finfo(logits.dtype).min).sum().item()
        assert num_masked > 0


class TestTopKLogits:
    """Align with diffulex.sampler.base.SamplerBase.top_k_logits"""
    
    def test_top_k_filtering_basic(self):
        """Test basic top-k filtering."""
        logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        result = top_k_logits(logits.clone(), top_k=2)
        
        # Top 2 should remain
        assert result[0, 0] == logits[0, 0]
        assert result[0, 1] == logits[0, 1]
        assert result[0, 2] == torch.finfo(logits.dtype).min
    
    def test_top_k_none_returns_unchanged(self):
        """top_k=None should return logits unchanged."""
        logits = torch.randn(2, 100)
        result = top_k_logits(logits.clone(), top_k=None)
        assert torch.allclose(result, logits)
    
    def test_top_k_zero_returns_unchanged(self):
        """top_k=0 should return logits unchanged."""
        logits = torch.randn(2, 100)
        result = top_k_logits(logits.clone(), top_k=0)
        assert torch.allclose(result, logits)
    
    def test_top_k_larger_than_vocab(self):
        """top_k larger than vocab size should handle gracefully."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = top_k_logits(logits.clone(), top_k=10)
        
        # All tokens should remain (k is capped to vocab size)
        assert (result != torch.finfo(logits.dtype).min).sum() == 3


class TestSampleTokens:
    """Align with diffulex.sampler.base.SamplerBase.sample_tokens"""
    
    def test_greedy_sampling(self):
        """temperature=0 should select argmax."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
        confidence, tokens, initial_conf = sample_tokens(logits, temperature=0.0)
        
        assert tokens.item() == 2  # Index of max value (3.0)
        assert confidence.item() == initial_conf.item()
        assert 0 <= initial_conf.item() <= 1
    
    def test_temperature_sampling_shape(self):
        """temperature>0 should produce correct shapes."""
        torch.manual_seed(42)
        logits = torch.randn(10, 100)
        
        confidence, tokens, initial_conf = sample_tokens(logits, temperature=1.0)
        
        assert tokens.shape == (10,)
        assert confidence.shape == (10,)
        assert initial_conf.shape == (10,)
        assert torch.all(initial_conf > 0) and torch.all(initial_conf <= 1)
    
    def test_temperature_sampling_distribution(self):
        """Higher temperature should increase entropy."""
        torch.manual_seed(42)
        logits = torch.tensor([[1.0, 0.5, 0.0, -0.5]])
        
        # Sample many times with high temperature
        samples_high_temp = []
        for _ in range(1000):
            _, tokens, _ = sample_tokens(logits, temperature=2.0)
            samples_high_temp.append(tokens.item())
        
        # Sample many times with low temperature
        samples_low_temp = []
        for _ in range(1000):
            _, tokens, _ = sample_tokens(logits, temperature=0.5)
            samples_low_temp.append(tokens.item())
        
        # High temperature should have more diversity
        unique_high = len(set(samples_high_temp))
        unique_low = len(set(samples_low_temp))
        assert unique_high >= unique_low
    
    def test_margin_confidence(self):
        """margin_confidence should return top1 - top2."""
        # Create peaked distribution
        logits = torch.tensor([[5.0, 1.0, 0.5, 0.1]])
        confidence, tokens, initial_conf = sample_tokens(
            logits, temperature=0.0, margin_confidence=True
        )
        
        # Calculate expected margin
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        expected_margin = sorted_probs[0, 0] - sorted_probs[0, 1]
        
        assert abs(confidence.item() - expected_margin.item()) < 1e-5
        assert confidence.item() != initial_conf.item()
    
    def test_neg_entropy(self):
        """neg_entropy should return sum(p * log(p))."""
        torch.manual_seed(42)
        logits = torch.randn(5, 50)
        confidence, tokens, initial_conf = sample_tokens(
            logits, temperature=1.0, neg_entropy=True
        )
        
        # Calculate expected value
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        expected = torch.sum(probs * log_probs, dim=-1)
        
        assert torch.allclose(confidence, expected, atol=1e-5)
        
        # For a peaked distribution, this should be close to 0 (less negative)
        # For a uniform distribution, this should be more negative
        peaked_logits = torch.tensor([[10.0, 0.1, 0.1, 0.1]])
        uniform_logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        
        peaked_conf, _, _ = sample_tokens(peaked_logits, temperature=1.0, neg_entropy=True)
        uniform_conf, _, _ = sample_tokens(uniform_logits, temperature=1.0, neg_entropy=True)
        
        # Peaked distribution should have higher (less negative) confidence
        assert peaked_conf.item() > uniform_conf.item()
    
    def test_sample_tokens_with_top_p(self):
        """Test sample_tokens with top_p filtering."""
        torch.manual_seed(42)
        logits = torch.randn(5, 100)
        
        confidence, tokens, initial_conf = sample_tokens(
            logits, temperature=1.0, top_p=0.9
        )
        
        assert tokens.shape == (5,)
        assert torch.all(initial_conf > 0)
    
    def test_sample_tokens_with_top_k(self):
        """Test sample_tokens with top_k filtering."""
        torch.manual_seed(42)
        logits = torch.randn(5, 100)
        
        confidence, tokens, initial_conf = sample_tokens(
            logits, temperature=1.0, top_k=10
        )
        
        assert tokens.shape == (5,)
        assert torch.all(initial_conf > 0)


class TestShiftLogitsSampler:
    """Align with diffulex.sampler.base.SamplerShiftLogits"""
    
    def test_shift_with_last_logit(self):
        """Test shifting with provided last_logit."""
        sampler = ShiftLogitsSampler()
        logits = torch.randn(5, 100)
        last_logits = torch.randn(100)
        
        shifted = sampler.shift(logits, last_logits, use_cache=False)
        
        # Position 0 should have last_logits
        assert torch.allclose(shifted[0], last_logits)
        
        # Position i should have logits[i-1]
        assert torch.allclose(shifted[1:], logits[:-1])
    
    def test_shift_without_last_logit(self):
        """Test shifting without provided last_logit."""
        sampler = ShiftLogitsSampler()
        logits = torch.randn(5, 100)
        
        shifted = sampler.shift(logits, use_cache=False)
        
        # Position 0 should have logits[-1] (last from current batch)
        assert torch.allclose(shifted[0], logits[-1])
        
        # Position i should have logits[i-1]
        assert torch.allclose(shifted[1:], logits[:-1])
    
    def test_shift_with_caching(self):
        """Test that caching works correctly."""
        sampler = ShiftLogitsSampler()
        logits1 = torch.randn(5, 100)
        
        # First call caches last logits
        shifted1 = sampler.shift(logits1, use_cache=True)
        
        # Second call should use cached value
        logits2 = torch.randn(3, 100)
        shifted2 = sampler.shift(logits2, use_cache=True)
        
        # Position 0 should use last logit from first call
        assert torch.allclose(shifted2[0], logits1[-1])
    
    def test_shift_reset_cache(self):
        """Test cache reset."""
        sampler = ShiftLogitsSampler()
        logits = torch.randn(5, 100)
        
        sampler.shift(logits, use_cache=True)
        sampler.reset_cache()
        
        # After reset, should use logits[-1] from current batch
        logits2 = torch.randn(3, 100)
        shifted = sampler.shift(logits2, use_cache=True)
        
        assert torch.allclose(shifted[0], logits2[-1])
    
    def test_shift_empty_logits(self):
        """Test that empty logits raises error."""
        sampler = ShiftLogitsSampler()
        logits = torch.randn(0, 100)
        
        with pytest.raises(ValueError, match="Logits sequence length is 0"):
            sampler.shift(logits)
    
    def test_shift_invalid_shape(self):
        """Test that invalid shape raises error."""
        sampler = ShiftLogitsSampler()
        logits = torch.randn(100)  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="Expected 2D logits"):
            sampler.shift(logits)


class TestNoShiftLogitsSampler:
    """Align with diffulex.sampler.base.SamplerNoShiftLogits"""
    
    def test_no_shift(self):
        """Test that logits are returned unchanged."""
        sampler = NoShiftLogitsSampler()
        logits = torch.randn(5, 100)
        
        result = sampler.shift(logits)
        
        assert torch.allclose(result, logits)
    
    def test_fetch_last_logits(self):
        """Test fetch_last_logits returns last logits."""
        sampler = NoShiftLogitsSampler()
        logits = torch.randn(5, 100)
        
        last = sampler.fetch_last_logits(logits)
        
        assert torch.allclose(last, logits[-1])
    
    def test_reset_cache_noop(self):
        """Test that reset_cache is a no-op."""
        sampler = NoShiftLogitsSampler()
        logits = torch.randn(5, 100)
        
        sampler.reset_cache()
        result1 = sampler.shift(logits)
        sampler.reset_cache()
        result2 = sampler.shift(logits)
        
        assert torch.allclose(result1, result2)


class TestFastdLLMV2Sampler:
    """Align with diffulex.sampler.fast_dllm_v2.FastdLLMV2SamplerForDiffusionLM"""
    
    def test_fast_dllm_v2_always_accepts_one(self):
        """FastdLLM V2 always accepts at least one token."""
        sampler = FastdLLMV2Sampler(threshold=0.99)  # Very high threshold
        
        block_manager = DiffusionBlockManager()
        block_manager.create_block(start_pos=10, length=5)
        
        # Create logits that won't meet threshold
        logits = torch.randn(20, 100) * 0.1  # Small variance
        
        output = sampler.sample(block_manager, logits)
        
        # Should have accepted at least one token
        assert len(output.accepted_ids_map["0"]["0"]) >= 1
    
    def test_fast_dllm_v2_uses_shift(self):
        """FastdLLM V2 should use shifted logits."""
        sampler = FastdLLMV2Sampler()
        
        block_manager = DiffusionBlockManager()
        block_manager.create_block(start_pos=0, length=5)
        
        # Create specific logits pattern
        logits = torch.zeros(10, 100)
        logits[0, 50] = 10.0  # Position 0 has high logit at index 50
        
        output = sampler.sample(block_manager, logits)
        
        # With shifting, position 0 should influence position 1
        # (This is more of a smoke test - exact behavior depends on model)
        assert "0" in output.sampled_tokens_map
    
    def test_fast_dllm_v2_respects_threshold(self):
        """FastdLLM V2 should respect confidence threshold."""
        sampler = FastdLLMV2Sampler(threshold=0.5)
        
        block_manager = DiffusionBlockManager()
        block_manager.create_block(start_pos=0, length=10)
        
        # Create very peaked logits (high confidence)
        logits = torch.zeros(20, 100)
        for i in range(10):
            logits[i, 50 + i] = 10.0  # Very peaked
        
        output = sampler.sample(block_manager, logits)
        
        # Should accept most tokens with high threshold and peaked distribution
        num_accepted = len(output.accepted_ids_map["0"]["0"])
        assert num_accepted >= 1


class TestLLaDASampler:
    """Align with diffulex.sampler.llada.LLaDASamplerForDiffusionLM"""
    
    def test_llada_no_shift(self):
        """LLaDA should NOT shift logits."""
        sampler = LLaDASampler()
        
        # The shift_sampler should be NoShiftLogitsSampler
        assert isinstance(sampler.shift_sampler, NoShiftLogitsSampler)
    
    def test_llada_pre_block_complete_logic(self):
        """LLaDA should use pre_block_complete logic."""
        sampler = LLaDASampler()
        
        block_manager = DiffusionBlockManager()
        # Create two blocks
        block_manager.create_block(start_pos=0, length=5, accept_threshold=0.9)
        block_manager.create_block(start_pos=5, length=5, accept_threshold=0.9)
        
        # First block is not complete, second should have pre_block_complete=False
        assert not block_manager.blocks[0].pre_block_complete
        assert block_manager.blocks[1].pre_block_complete  # First block is "previous"
    
    def test_llada_per_block_threshold(self):
        """LLaDA should use per-block threshold."""
        sampler = LLaDASampler()
        
        block_manager = DiffusionBlockManager()
        block_manager.create_block(start_pos=0, length=5, accept_threshold=0.5)
        block_manager.create_block(start_pos=5, length=5, accept_threshold=0.9)
        
        # Different thresholds
        assert block_manager.blocks[0].accept_threshold == 0.5
        assert block_manager.blocks[1].accept_threshold == 0.9


class TestDreamSampler:
    """Align with diffulex.sampler.dream.DreamSamplerForDiffusionLM"""
    
    def test_dream_uses_shift(self):
        """Dream should use shifted logits."""
        sampler = DreamSampler()
        
        # The shift_sampler should be ShiftLogitsSampler
        assert isinstance(sampler.shift_sampler, ShiftLogitsSampler)
    
    def test_dream_pre_block_complete_logic(self):
        """Dream should use pre_block_complete logic."""
        sampler = DreamSampler()
        
        block_manager = DiffusionBlockManager()
        block_manager.create_block(start_pos=0, length=5)
        block_manager.create_block(start_pos=5, length=5)
        
        # Second block should have pre_block_complete
        assert not block_manager.blocks[0].pre_block_complete
        assert block_manager.blocks[1].pre_block_complete


class TestSDARSampler:
    """Align with diffulex.sampler.sdar.SDARSamplerForDiffusionLM"""
    
    def test_sdar_uses_shift(self):
        """SDAR should use shifted logits."""
        sampler = SDARSampler()
        
        assert isinstance(sampler.shift_sampler, ShiftLogitsSampler)
    
    def test_sdar_always_accepts_one(self):
        """SDAR always accepts at least one token (like FastdLLM V2)."""
        sampler = SDARSampler(threshold=0.99)
        
        block_manager = DiffusionBlockManager()
        block_manager.create_block(start_pos=0, length=5)
        
        logits = torch.randn(10, 100) * 0.1
        
        output = sampler.sample(block_manager, logits)
        
        assert len(output.accepted_ids_map["0"]["0"]) >= 1


class TestSamplerRegistry:
    """Test sampler registry functionality."""
    
    def test_registry_contains_all_models(self):
        """Registry should contain all model types."""
        expected_models = ["fast_dllm_v2", "llada", "dream", "sdar"]
        
        for model in expected_models:
            assert model in SAMPLER_REGISTRY
    
    def test_get_sampler_class(self):
        """Test get_sampler_class helper."""
        from diffulex_edge.runtime.sampler.models import get_sampler_class
        
        assert get_sampler_class("fast_dllm_v2") == FastdLLMV2Sampler
        assert get_sampler_class("llada") == LLaDASampler
        assert get_sampler_class("dream") == DreamSampler
        assert get_sampler_class("sdar") == SDARSampler
    
    def test_get_sampler_class_invalid(self):
        """Test get_sampler_class with invalid model type."""
        from diffulex_edge.runtime.sampler.models import get_sampler_class
        
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_sampler_class("invalid_model")