"""Tests for DiffusionSampler.

Test Plan Coverage:
- DiffusionSampler basic functionality: 10 tests
- shift_logits boundary conditions: 8 tests
- Numerical precision: 8 tests
- Error handling: 7 tests
- top_k/top_p boundary conditions: 5 tests
- temperature boundary conditions: 4 tests
- End-to-end diffusion flow: 4 tests

Total: 46 tests
"""

import pytest
import torch
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runtime.diffusion import (
    DiffusionSampler, DiffusionBlockManager, SampleOutput
)


# ============================================================================
# DiffusionSampler Basic Functionality Tests (10 tests)
# ============================================================================

class TestDiffusionSamplerBasic:
    """Test basic DiffusionSampler functionality."""
    
    def test_sampler_initialization(self):
        """TEST-SAMPLER-001: Verify sampler initializes with correct parameters."""
        sampler = DiffusionSampler(
            mask_token_id=100,
            confidence_threshold=0.9,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )
        
        assert sampler.mask_token_id == 100
        assert sampler.confidence_threshold == 0.9
        assert sampler.temperature == 0.8
        assert sampler.top_k == 50
        assert sampler.top_p == 0.95
    
    def test_sampler_default_parameters(self):
        """TEST-SAMPLER-002: Verify sampler has correct default parameters."""
        sampler = DiffusionSampler()
        
        assert sampler.mask_token_id == 126336  # FastdLLM default
        assert sampler.confidence_threshold == 0.95
        assert sampler.temperature == 1.0
        assert sampler.top_k == 0  # Disabled by default
        assert sampler.top_p == 1.0  # Disabled by default
    
    def test_shift_logits_basic(self):
        """TEST-SAMPLER-003: Verify shift_logits shifts right by one position."""
        sampler = DiffusionSampler()
        
        # Create simple logits
        logits = torch.tensor([
            [1.0, 2.0, 3.0],  # Position 0
            [4.0, 5.0, 6.0],  # Position 1
            [7.0, 8.0, 9.0],  # Position 2
        ])
        
        shifted = sampler.shift_logits(logits)
        
        # Position 0 should get original position 2 (last)
        assert torch.allclose(shifted[0], logits[2])
        # Position 1 should get original position 0
        assert torch.allclose(shifted[1], logits[0])
        # Position 2 should get original position 1
        assert torch.allclose(shifted[2], logits[1])
    
    def test_shift_logits_with_last_logits(self):
        """TEST-SAMPLER-004: Verify shift_logits uses provided last_logits."""
        sampler = DiffusionSampler()
        
        logits = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        
        last_logits = torch.tensor([9.0, 9.0, 9.0])
        shifted = sampler.shift_logits(logits, last_logits)
        
        # Position 0 should get custom last_logits
        assert torch.allclose(shifted[0], last_logits)
        # Position 1 should get original position 0
        assert torch.allclose(shifted[1], logits[0])
    
    def test_compute_confidence_basic(self):
        """TEST-SAMPLER-005: Verify compute_confidence returns correct values."""
        sampler = DiffusionSampler(temperature=1.0)
        
        # Create logits where we know the probabilities
        # For token 0: exp(2.0) / sum(exp([2.0, 1.0, 0.0]))
        logits = torch.tensor([
            [2.0, 1.0, 0.0],  # Position 0
            [0.0, 2.0, 1.0],  # Position 1
        ])
        
        # Sample token 0 from position 0, token 1 from position 1
        sampled_tokens = torch.tensor([0, 1])
        
        confidence = sampler.compute_confidence(logits, sampled_tokens)
        
        # Compute expected probabilities
        probs0 = F.softmax(logits[0], dim=-1)
        probs1 = F.softmax(logits[1], dim=-1)
        
        assert abs(confidence[0].item() - probs0[0].item()) < 1e-5
        assert abs(confidence[1].item() - probs1[1].item()) < 1e-5
    
    def test_compute_confidence_with_temperature(self):
        """TEST-SAMPLER-006: Verify temperature affects confidence."""
        # High temperature = more uniform distribution = lower confidence
        sampler_high_temp = DiffusionSampler(temperature=2.0)
        sampler_low_temp = DiffusionSampler(temperature=0.5)
        
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        sampled_tokens = torch.tensor([0])
        
        conf_high = sampler_high_temp.compute_confidence(logits.clone(), sampled_tokens)
        conf_low = sampler_low_temp.compute_confidence(logits.clone(), sampled_tokens)
        
        # Lower temperature should give higher confidence
        assert conf_low[0].item() > conf_high[0].item()
    
    def test_sample_tokens_deterministic_with_zero_temperature(self):
        """TEST-SAMPLER-007: Verify sampling is deterministic when temperature=0 (greedy)."""
        sampler = DiffusionSampler(temperature=0.0)
        
        logits = torch.tensor([
            [1.0, 5.0, 2.0],  # Token 1 has highest logit
            [3.0, 1.0, 4.0],  # Token 2 has highest logit
        ])
        
        # With temperature=0, should be deterministic (greedy)
        confidence, tokens, initial_conf = sampler.sample_tokens(logits, temperature=0.0)
        
        # Most likely tokens
        assert tokens[0].item() == 1  # Highest logit at position 0
        assert tokens[1].item() == 2  # Highest logit at position 1
    
    def test_sample_tokens_shape(self):
        """TEST-SAMPLER-008: Verify sample_tokens returns correct shapes."""
        sampler = DiffusionSampler()
        
        batch_size = 5
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size)
        
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        assert tokens.shape == (batch_size,)
        assert confidence.shape == (batch_size,)
        assert initial_conf.shape == (batch_size,)
    
    def test_sample_blocks_returns_output(self):
        """TEST-SAMPLER-009: Verify sample_blocks returns SampleOutput."""
        sampler = DiffusionSampler()
        manager = DiffusionBlockManager()
        
        # Create a block
        block_id = manager.create_block(start_pos=5, length=3)
        
        # Create logits for positions 0-9
        logits = torch.randn(10, 100)
        
        output = sampler.sample_blocks(manager, logits)
        
        assert isinstance(output, SampleOutput)
        # At least one token should be sampled (at minimum the highest confidence)
        assert len(output.sampled_tokens) >= 1
    
    def test_sample_blocks_accepts_high_confidence(self):
        """TEST-SAMPLER-010: Verify sample_blocks accepts tokens above threshold."""
        # Use low threshold to ensure acceptance
        sampler = DiffusionSampler(confidence_threshold=0.0)
        manager = DiffusionBlockManager()
        
        manager.create_block(start_pos=0, length=5)
        
        # Use extreme logits to get high confidence
        logits = torch.zeros(5, 100)
        for i in range(5):
            logits[i, i] = 10.0  # Strong preference for token i
        
        output = sampler.sample_blocks(manager, logits)
        
        # With threshold=0, most tokens should be accepted (at least the highest confidence one)
        assert len(output.accepted_tokens) >= 1


# ============================================================================
# shift_logits Boundary Condition Tests (8 tests)
# ============================================================================

class TestShiftLogitsBoundaryConditions:
    """Test shift_logits boundary conditions."""
    
    def test_shift_logits_empty(self):
        """TEST-SAMPLER-BOUND-001: Verify shift_logits handles empty logits."""
        sampler = DiffusionSampler()
        
        empty_logits = torch.zeros(0, 100)
        shifted = sampler.shift_logits(empty_logits)
        
        assert shifted.shape == (0, 100)
    
    def test_shift_logits_single_position(self):
        """TEST-SAMPLER-BOUND-002: Verify shift_logits with single position."""
        sampler = DiffusionSampler()
        
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        shifted = sampler.shift_logits(logits)
        
        # Single position should remain unchanged (shifted by itself)
        assert shifted.shape == (1, 3)
        assert torch.allclose(shifted[0], logits[0])
    
    def test_shift_logits_two_positions(self):
        """TEST-SAMPLER-BOUND-003: Verify shift_logits with two positions."""
        sampler = DiffusionSampler()
        
        logits = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        shifted = sampler.shift_logits(logits)
        
        # Position 0 gets position 1
        assert torch.allclose(shifted[0], logits[1])
        # Position 1 gets position 0
        assert torch.allclose(shifted[1], logits[0])
    
    def test_shift_logits_large_vocab(self):
        """TEST-SAMPLER-BOUND-004: Verify shift_logits with large vocabulary."""
        sampler = DiffusionSampler()
        
        seq_len = 10
        vocab_size = 50000
        logits = torch.randn(seq_len, vocab_size)
        
        shifted = sampler.shift_logits(logits)
        
        assert shifted.shape == (seq_len, vocab_size)
        
        # Verify shift pattern
        assert torch.allclose(shifted[0], logits[-1])
        for i in range(1, seq_len):
            assert torch.allclose(shifted[i], logits[i-1])
    
    def test_shift_logits_large_sequence(self):
        """TEST-SAMPLER-BOUND-005: Verify shift_logits with large sequence."""
        sampler = DiffusionSampler()
        
        seq_len = 10000
        vocab_size = 100
        logits = torch.randn(seq_len, vocab_size)
        
        shifted = sampler.shift_logits(logits)
        
        assert shifted.shape == (seq_len, vocab_size)
        
        # Check boundary conditions
        assert torch.allclose(shifted[0], logits[-1])
        assert torch.allclose(shifted[-1], logits[-2])
    
    def test_shift_logits_uniform_values(self):
        """TEST-SAMPLER-BOUND-006: Verify shift_logits with uniform values."""
        sampler = DiffusionSampler()
        
        # All same values
        logits = torch.ones(10, 100)
        shifted = sampler.shift_logits(logits)
        
        # Should remain all ones
        assert torch.allclose(shifted, logits)
    
    def test_shift_logits_very_small_values(self):
        """TEST-SAMPLER-BOUND-007: Verify shift_logits with very small values."""
        sampler = DiffusionSampler()
        
        logits = torch.full((5, 10), 1e-10)
        shifted = sampler.shift_logits(logits)
        
        assert shifted.shape == (5, 10)
        assert torch.allclose(shifted, logits)
    
    def test_shift_logits_very_large_values(self):
        """TEST-SAMPLER-BOUND-008: Verify shift_logits with very large values."""
        sampler = DiffusionSampler()
        
        logits = torch.full((5, 10), 1e6)
        shifted = sampler.shift_logits(logits)
        
        assert shifted.shape == (5, 10)
        assert torch.allclose(shifted, logits)


# ============================================================================
# Numerical Precision Tests (8 tests)
# ============================================================================

class TestNumericalPrecision:
    """Test numerical precision requirements."""
    
    def test_shift_logits_precision(self):
        """TEST-SAMPLER-NUM-001: Verify shift_logits numerical precision < 1e-6."""
        sampler = DiffusionSampler()
        
        # Create logits with known values
        torch.manual_seed(42)
        logits = torch.randn(100, 1000, dtype=torch.float64)
        
        shifted = sampler.shift_logits(logits)
        
        # Verify shift is exact
        assert torch.allclose(shifted[0], logits[-1], atol=1e-7)
        for i in range(1, 100):
            assert torch.allclose(shifted[i], logits[i-1], atol=1e-7)
    
    def test_compute_confidence_precision(self):
        """TEST-SAMPLER-NUM-002: Verify confidence computation precision < 1e-6."""
        sampler = DiffusionSampler(temperature=1.0)
        
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        sampled_tokens = torch.tensor([2])
        
        confidence = sampler.compute_confidence(logits, sampled_tokens)
        
        # Manual computation
        probs = F.softmax(logits, dim=-1)
        expected = probs[0, 2].item()
        
        assert abs(confidence[0].item() - expected) < 1e-6
    
    def test_softmax_numerical_stability(self):
        """TEST-SAMPLER-NUM-003: Verify softmax handles large logits stably."""
        sampler = DiffusionSampler()
        
        # Very large logits
        logits = torch.tensor([[1000.0, 1001.0, 1002.0]])
        
        # Should not produce NaN
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        assert not torch.isnan(confidence).any()
        assert initial_conf[0].item() > 0
        assert initial_conf[0].item() <= 1.0
    
    def test_softmax_with_extreme_negative_logits(self):
        """TEST-SAMPLER-NUM-004: Verify softmax handles extreme negative logits."""
        sampler = DiffusionSampler()
        
        # Very negative logits
        logits = torch.tensor([[-1000.0, -1001.0, -999.0]])
        
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        assert not torch.isnan(confidence).any()
        assert initial_conf[0].item() > 0
        assert initial_conf[0].item() <= 1.0
    
    def test_temperature_extreme_values_precision(self):
        """TEST-SAMPLER-NUM-005: Verify temperature scaling precision."""
        sampler_low = DiffusionSampler(temperature=0.01)
        sampler_high = DiffusionSampler(temperature=100.0)
        
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        sampled = torch.tensor([2])
        
        conf_low = sampler_low.compute_confidence(logits.clone(), sampled)
        conf_high = sampler_high.compute_confidence(logits.clone(), sampled)
        
        # Low temp should approach 1.0, high temp should approach ~0.33
        assert conf_low[0].item() > 0.99
        assert abs(conf_high[0].item() - 0.333) < 0.01
    
    def test_confidence_sum_constraint(self):
        """TEST-SAMPLER-NUM-006: Verify confidence scores follow probability axioms."""
        sampler = DiffusionSampler(temperature=1.0)
        
        # For a single distribution, confidence of each token equals its probability
        logits = torch.randn(1, 100)
        
        # Sample multiple times to get different tokens
        probs = F.softmax(logits, dim=-1)
        
        for token_id in [0, 50, 99]:
            sampled = torch.tensor([token_id])
            conf = sampler.compute_confidence(logits, sampled)
            expected_prob = probs[0, token_id].item()
            assert abs(conf[0].item() - expected_prob) < 1e-5
    
    def test_cumulative_confidence_bounds(self):
        """TEST-SAMPLER-NUM-007: Verify confidence values are within [0, 1]."""
        sampler = DiffusionSampler()
        
        for _ in range(100):
            logits = torch.randn(10, 100)
            confidence, tokens, initial_conf = sampler.sample_tokens(logits)
            
            assert torch.all(initial_conf >= 0.0)
            assert torch.all(initial_conf <= 1.0)
    
    def test_confidence_monotonicity_with_temperature(self):
        """TEST-SAMPLER-NUM-008: Verify confidence monotonicity with temperature."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        sampled = torch.tensor([2])
        
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        confidences = []
        
        for temp in temperatures:
            sampler = DiffusionSampler(temperature=temp)
            conf = sampler.compute_confidence(logits.clone(), sampled)
            confidences.append(conf[0].item())
        
        # Higher temperature should give lower confidence for the max token
        for i in range(len(confidences) - 1):
            assert confidences[i] > confidences[i + 1]


# ============================================================================
# Error Handling Tests (7 tests)
# ============================================================================

class TestSamplerErrorHandling:
    """Test error handling in DiffusionSampler."""
    
    def test_shift_logits_nan_raises(self):
        """TEST-SAMPLER-ERR-001: Verify shift_logits raises on NaN values."""
        sampler = DiffusionSampler()
        
        logits_with_nan = torch.tensor([
            [1.0, float('nan'), 3.0],
            [4.0, 5.0, 6.0],
        ])
        
        with pytest.raises(ValueError, match="NaN"):
            sampler.shift_logits(logits_with_nan)
    
    def test_shift_logits_inf_raises(self):
        """TEST-SAMPLER-ERR-002: Verify shift_logits raises on Inf values."""
        sampler = DiffusionSampler()
        
        logits_with_inf = torch.tensor([
            [1.0, float('inf'), 3.0],
            [4.0, 5.0, 6.0],
        ])
        
        with pytest.raises(ValueError, match="Inf"):
            sampler.shift_logits(logits_with_inf)
    
    def test_shift_logits_negative_inf_raises(self):
        """TEST-SAMPLER-ERR-003: Verify shift_logits raises on -Inf values."""
        sampler = DiffusionSampler()
        
        logits_with_neg_inf = torch.tensor([
            [1.0, float('-inf'), 3.0],
            [4.0, 5.0, 6.0],
        ])
        
        with pytest.raises(ValueError, match="Inf"):
            sampler.shift_logits(logits_with_neg_inf)
    
    def test_shift_logits_invalid_dimensions_raises(self):
        """TEST-SAMPLER-ERR-004: Verify shift_logits raises on invalid dimensions."""
        sampler = DiffusionSampler()
        
        # 1D logits
        logits_1d = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D"):
            sampler.shift_logits(logits_1d)
        
        # 3D logits
        logits_3d = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="2D"):
            sampler.shift_logits(logits_3d)
    
    def test_compute_confidence_dimension_mismatch(self):
        """TEST-SAMPLER-ERR-005: Verify compute_confidence handles dimension mismatch."""
        sampler = DiffusionSampler()
        
        # More tokens than logits
        logits = torch.randn(3, 100)
        sampled = torch.tensor([0, 1, 2, 3])  # 4 tokens, 3 logits
        
        # Should raise an error
        with pytest.raises(RuntimeError):
            sampler.compute_confidence(logits, sampled)
    
    def test_sample_tokens_empty_logits(self):
        """TEST-SAMPLER-ERR-006: Verify sample_tokens handles empty logits gracefully."""
        sampler = DiffusionSampler()
        
        empty_logits = torch.zeros(0, 100)
        
        # Should handle empty input gracefully
        confidence, tokens, initial_conf = sampler.sample_tokens(empty_logits)
        
        assert tokens.shape == (0,)
        assert confidence.shape == (0,)
        assert initial_conf.shape == (0,)
    
    def test_sample_blocks_no_active_blocks(self):
        """TEST-SAMPLER-ERR-007: Verify sample_blocks handles no active blocks."""
        sampler = DiffusionSampler()
        manager = DiffusionBlockManager()
        
        # Create a block and complete it
        block_id = manager.create_block(start_pos=0, length=3)
        manager.update_block(block_id, [0, 1, 2], [100, 101, 102])
        
        logits = torch.randn(10, 100)
        
        # Should handle gracefully
        output = sampler.sample_blocks(manager, logits)
        
        assert isinstance(output, SampleOutput)
        assert len(output.sampled_tokens) == 0


# ============================================================================
# Top-K/Top-P Boundary Tests (5 tests)
# ============================================================================

class TestTopKTopPBoundaryConditions:
    """Test top-k and top-p boundary conditions."""
    
    def test_top_k_zero_disabled(self):
        """TEST-SAMPLER-TOP-001: Verify top_k=0 disables filtering."""
        sampler = DiffusionSampler(top_k=0, top_p=1.0)
        
        logits = torch.randn(5, 100)
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        # All tokens should be valid
        assert torch.all(tokens >= 0)
        assert torch.all(tokens < 100)
    
    def test_top_k_larger_than_vocab(self):
        """TEST-SAMPLER-TOP-002: Verify top_k > vocab_size handled gracefully."""
        sampler = DiffusionSampler(top_k=1000)  # Larger than vocab
        
        logits = torch.randn(5, 100)  # vocab_size = 100
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        assert tokens.shape == (5,)
        assert torch.all(initial_conf > 0)
    
    def test_top_p_one_disabled(self):
        """TEST-SAMPLER-TOP-003: Verify top_p=1.0 disables filtering."""
        sampler = DiffusionSampler(top_k=0, top_p=1.0)
        
        logits = torch.randn(5, 100)
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        assert tokens.shape == (5,)
    
    def test_top_p_very_small(self):
        """TEST-SAMPLER-TOP-004: Verify very small top_p keeps at least one token."""
        sampler = DiffusionSampler(top_k=0, top_p=0.01)
        
        logits = torch.randn(5, 100)
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        # Should still produce valid tokens
        assert torch.all(tokens >= 0)
        assert torch.all(tokens < 100)
        assert torch.all(initial_conf > 0)
    
    def test_top_k_top_p_combined(self):
        """TEST-SAMPLER-TOP-005: Verify top_k and top_p work together."""
        sampler = DiffusionSampler(top_k=10, top_p=0.5)
        
        logits = torch.randn(5, 100)
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        assert tokens.shape == (5,)
        assert torch.all(initial_conf > 0)


# ============================================================================
# Temperature Boundary Tests (4 tests)
# ============================================================================

class TestTemperatureBoundaryConditions:
    """Test temperature boundary conditions."""
    
    def test_temperature_one_no_effect(self):
        """TEST-SAMPLER-TEMP-001: Verify temperature=1.0 has no effect."""
        sampler = DiffusionSampler(temperature=1.0)
        
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        
        confidence, tokens, initial_conf = sampler.sample_tokens(logits, temperature=1.0)
        
        # Manual softmax at T=1
        expected_probs = F.softmax(logits, dim=-1)
        
        # Verify initial_conf matches softmax probabilities
        for i, token in enumerate(tokens):
            expected = expected_probs[i, token].item()
            assert abs(initial_conf[i].item() - expected) < 1e-5
    
    def test_temperature_very_small(self):
        """TEST-SAMPLER-TEMP-002: Verify very small temperature approaches argmax."""
        sampler = DiffusionSampler(temperature=0.001)
        
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        
        # Sample multiple times to check determinism
        tokens_list = []
        for _ in range(10):
            confidence, tokens, initial_conf = sampler.sample_tokens(logits)
            tokens_list.append(tokens[0].item())
        
        # All samples should be the same (argmax = 1)
        assert all(t == 1 for t in tokens_list)
    
    def test_temperature_very_large(self):
        """TEST-SAMPLER-TEMP-003: Verify very large temperature approaches uniform."""
        sampler = DiffusionSampler(temperature=100.0)
        
        logits = torch.tensor([[1.0, 5.0, 2.0, 0.0, 10.0]])
        
        confidence, tokens, initial_conf = sampler.sample_tokens(logits)
        
        # With high temperature, confidence should approach 1/vocab_size = 0.2
        assert abs(initial_conf[0].item() - 0.2) < 0.05
    
    def test_temperature_affects_distribution(self):
        """TEST-SAMPLER-TEMP-004: Verify temperature changes probability distribution."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        # Sample many times with different temperatures
        temps = [0.5, 1.0, 2.0]
        distributions = []
        
        for temp in temps:
            sampler = DiffusionSampler(temperature=temp)
            counts = [0] * 5
            for _ in range(1000):
                confidence, tokens, initial_conf = sampler.sample_tokens(logits)
                counts[tokens[0].item()] += 1
            distributions.append(counts)
        
        # Lower temperature should have more variance in counts (more peaky)
        # Higher temperature should have more uniform counts
        variance_low = sum((x - 200) ** 2 for x in distributions[0]) / 5
        variance_high = sum((x - 200) ** 2 for x in distributions[2]) / 5
        
        assert variance_low > variance_high


# ============================================================================
# End-to-End Diffusion Flow Tests (4 tests)
# ============================================================================

class TestEndToEndDiffusionFlow:
    """Test end-to-end diffusion sampling flow."""
    
    def test_sample_blocks_multiple_blocks(self):
        """TEST-SAMPLER-E2E-001: Verify sample_blocks handles multiple blocks."""
        sampler = DiffusionSampler(confidence_threshold=0.0)  # Accept all
        manager = DiffusionBlockManager()
        
        # Create multiple blocks
        manager.create_block(start_pos=0, length=3)
        manager.create_block(start_pos=5, length=3)
        manager.create_block(start_pos=10, length=3)
        
        logits = torch.randn(15, 100)
        
        output = sampler.sample_blocks(manager, logits)
        
        # Should sample from all active blocks
        # Block 1: positions 0, 1, 2
        # Block 2: positions 5, 6, 7
        # Block 3: positions 10, 11, 12
        expected_positions = {0, 1, 2, 5, 6, 7, 10, 11, 12}
        assert set(output.sampled_tokens.keys()) == expected_positions
    
    def test_sample_blocks_partially_completed(self):
        """TEST-SAMPLER-E2E-002: Verify sample_blocks with partially completed blocks."""
        sampler = DiffusionSampler(confidence_threshold=0.0)
        manager = DiffusionBlockManager()
        
        # Create block and complete part of it
        block_id = manager.create_block(start_pos=0, length=5)
        manager.update_block(block_id, [0, 2], [100, 102])
        
        logits = torch.randn(5, 100)
        
        output = sampler.sample_blocks(manager, logits)
        
        # Should only sample from positions 1, 3, 4
        expected_positions = {1, 3, 4}
        assert set(output.sampled_tokens.keys()) == expected_positions
    
    def test_sample_blocks_partial_acceptance(self):
        """TEST-SAMPLER-E2E-003: Verify partial acceptance based on confidence."""
        # Use high threshold to accept only very confident predictions
        sampler = DiffusionSampler(confidence_threshold=0.99)
        manager = DiffusionBlockManager()
        
        manager.create_block(start_pos=0, length=5)
        
        # Create logits with one very strong prediction
        logits = torch.zeros(5, 100)
        for i in range(5):
            logits[i, i] = 1.0  # Moderate confidence
        # Make position 2 have very high confidence
        logits[2, 2] = 20.0
        
        output = sampler.sample_blocks(manager, logits)
        
        # All should be sampled
        assert len(output.sampled_tokens) == 5
        # But only position 2 might be accepted due to high threshold
        assert 2 in output.accepted_tokens
    
    def test_full_diffusion_iteration(self):
        """TEST-SAMPLER-E2E-004: Verify full diffusion iteration cycle."""
        sampler = DiffusionSampler(confidence_threshold=0.5)
        manager = DiffusionBlockManager()
        
        # Create a block
        block_id = manager.create_block(start_pos=0, length=10)
        
        # Simulate multiple iterations
        for iteration in range(5):
            # Generate fresh logits
            logits = torch.randn(10, 100)
            
            # Sample blocks
            output = sampler.sample_blocks(manager, logits)
            
            # Verify output structure
            assert isinstance(output, SampleOutput)
            
            # Check if block is complete
            if not manager.has_active_blocks():
                break
        
        # Block should be complete or have accepted tokens
        final_block = manager.get_block(block_id)
        assert final_block is not None
        assert len(final_block.accepted_token_ids) > 0 or final_block.is_active
