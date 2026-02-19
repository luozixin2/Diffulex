"""Tests for inference engine."""

import pytest
import torch

from diffulex_edge.model.fast_dllm_v2_edge import FastdLLMV2Edge, FastdLLMV2EdgeConfig
from diffulex_edge.runtime import InferenceEngine, GenerationConfig
from diffulex_edge.runtime.sampler import GreedySampler, TopKSampler, TopPSampler


class TestSamplers:
    """Test sampling strategies."""
    
    def test_greedy_sampler(self):
        """Test greedy sampling."""
        sampler = GreedySampler()
        
        # Create logits where token 5 is highest
        logits = torch.randn(100)
        logits[5] = 10.0
        
        token = sampler.sample(logits)
        assert token == 5
    
    def test_greedy_sampler_batched(self):
        """Test greedy sampling with batched logits."""
        sampler = GreedySampler()
        
        logits = torch.randn(2, 100)
        logits[0, 3] = 10.0
        
        token = sampler.sample(logits)
        assert token == 3
    
    def test_top_k_sampler(self):
        """Test top-k sampling."""
        sampler = TopKSampler(k=5, temperature=1.0, seed=42)
        
        logits = torch.randn(100)
        
        # Sample multiple times
        tokens = [sampler.sample(logits) for _ in range(20)]
        
        # All tokens should be in top 5
        top_k_indices = torch.topk(logits, 5).indices.tolist()
        assert all(t in top_k_indices for t in tokens)
    
    def test_top_k_sampler_temperature(self):
        """Test temperature affects distribution."""
        torch.manual_seed(42)
        logits = torch.randn(100)
        
        # High temperature = more random
        sampler_high = TopKSampler(k=10, temperature=2.0, seed=42)
        sampler_low = TopKSampler(k=10, temperature=0.5, seed=42)
        
        tokens_high = [sampler_high.sample(logits) for _ in range(50)]
        tokens_low = [sampler_low.sample(logits) for _ in range(50)]
        
        # Low temperature should be more deterministic
        unique_high = len(set(tokens_high))
        unique_low = len(set(tokens_low))
        
        assert unique_high >= unique_low
    
    def test_top_p_sampler(self):
        """Test top-p (nucleus) sampling."""
        sampler = TopPSampler(p=0.9, temperature=1.0, seed=42)
        
        # Create skewed distribution
        logits = torch.randn(100)
        logits[0] = 5.0
        logits[1] = 4.0
        logits[2] = 3.0
        
        tokens = [sampler.sample(logits) for _ in range(20)]
        
        # Most tokens should be from the top few
        # (with p=0.9, we include tokens until cumsum > 0.9)
        assert all(isinstance(t, int) for t in tokens)
    
    def test_sampler_factory(self):
        """Test sampler factory function."""
        from diffulex_edge.runtime.sampler import get_sampler
        
        greedy = get_sampler("greedy")
        assert isinstance(greedy, GreedySampler)
        
        top_k = get_sampler("top_k", temperature=0.8, top_k=40)
        assert isinstance(top_k, TopKSampler)
        
        top_p = get_sampler("top_p", temperature=0.9, top_p=0.95)
        assert isinstance(top_p, TopPSampler)


class TestGenerationConfig:
    """Test generation configuration."""
    
    def test_default_config(self):
        """Test default generation config."""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 100
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
    
    def test_get_sampler(self):
        """Test sampler creation from config."""
        config = GenerationConfig(temperature=0)
        sampler = config.get_sampler()
        assert isinstance(sampler, GreedySampler)
        
        config = GenerationConfig(temperature=0.8)
        sampler = config.get_sampler()
        assert isinstance(sampler, TopPSampler)
    
    def test_stop_sequences(self):
        """Test stop sequences configuration."""
        config = GenerationConfig(
            stop_sequences=[[1, 2, 3], [4, 5]]
        )
        assert config.stop_sequences == [[1, 2, 3], [4, 5]]


class TestInferenceEngine:
    """Test inference engine."""
    
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
    
    def test_engine_creation_from_model(self, small_model):
        """Test creating engine from model."""
        engine = InferenceEngine.from_model(small_model)
        
        assert engine.model is small_model
        assert engine.use_kv_cache is True
        assert engine._is_pte is False
    
    def test_engine_prefill(self, small_model):
        """Test prefill operation."""
        engine = InferenceEngine.from_model(small_model)
        
        prompt_tokens = list(range(10))
        logits, kv_cache = engine._prefill(torch.tensor([prompt_tokens]))
        
        assert logits.shape == (1, 10, 100)  # batch=1, seq=10, vocab=100
        assert kv_cache is not None
        assert engine._current_pos == 10
    
    def test_engine_decode_step(self, small_model):
        """Test single decode step."""
        engine = InferenceEngine.from_model(small_model)
        
        # First prefill
        prompt_tokens = list(range(5))
        engine._prefill(torch.tensor([prompt_tokens]))
        
        # Then decode
        logits, _ = engine._decode_step(0)
        
        assert logits.shape == (100,)  # vocab size
        assert engine._current_pos == 6  # 5 prefill + 1 decode
    
    def test_engine_generate(self, small_model):
        """Test text generation."""
        engine = InferenceEngine.from_model(small_model)
        
        config = GenerationConfig(
            max_new_tokens=5,
            temperature=0,  # Greedy
        )
        
        prompt_tokens = [1, 2, 3]
        generated = engine.generate(prompt_tokens, config)
        
        assert len(generated) <= 5
        assert all(isinstance(t, int) for t in generated)
    
    def test_engine_generate_with_eos(self, small_model):
        """Test generation with EOS token."""
        engine = InferenceEngine.from_model(small_model)
        
        config = GenerationConfig(
            max_new_tokens=10,
            eos_token_id=5,
            temperature=0,
        )
        
        # This may or may not hit EOS depending on the model
        # Just verify it runs without error
        prompt_tokens = [1, 2, 3]
        generated = engine.generate(prompt_tokens, config)
        
        assert isinstance(generated, list)
    
    def test_engine_stream(self, small_model):
        """Test streaming generation."""
        engine = InferenceEngine.from_model(small_model)
        
        config = GenerationConfig(max_new_tokens=5, temperature=0)
        
        prompt_tokens = [1, 2, 3]
        tokens = list(engine.generate_stream(prompt_tokens, config))
        
        assert len(tokens) <= 5
        assert all(isinstance(t, int) for t in tokens)
    
    def test_cache_reset(self, small_model):
        """Test cache reset functionality."""
        engine = InferenceEngine.from_model(small_model)
        
        # Generate first sequence
        engine.generate([1, 2, 3], GenerationConfig(max_new_tokens=3, temperature=0))
        assert engine._current_pos > 0
        
        # Reset cache
        engine.reset_cache()
        assert engine._current_pos == 0
        assert engine._kv_cache is None
    
    def test_engine_without_kv_cache(self, small_model):
        """Test engine without KV cache."""
        engine = InferenceEngine.from_model(small_model, use_kv_cache=False)
        
        config = GenerationConfig(max_new_tokens=3, temperature=0)
        generated = engine.generate([1, 2, 3], config)
        
        assert isinstance(generated, list)
    
    def test_repetition_penalty(self, small_model):
        """Test repetition penalty."""
        engine = InferenceEngine.from_model(small_model)
        
        config = GenerationConfig(
            max_new_tokens=5,
            temperature=0,
            repetition_penalty=1.5,
        )
        
        prompt_tokens = [1, 2, 3, 1, 2, 3]  # Has repeats
        generated = engine.generate(prompt_tokens, config)
        
        assert isinstance(generated, list)
    
    def test_stop_sequences(self, small_model):
        """Test stop sequence detection."""
        engine = InferenceEngine.from_model(small_model)
        
        # Test the check function directly
        assert engine._check_stop_sequences([1, 2, 3, 4, 5], [[4, 5]]) is True
        assert engine._check_stop_sequences([1, 2, 3, 4, 5], [[3, 4]]) is False  # [3,4] is not at the end
        assert engine._check_stop_sequences([1, 2, 3, 4, 5], [[5, 6]]) is False
        assert engine._check_stop_sequences([1, 2], [[1, 2, 3]]) is False
        assert engine._check_stop_sequences([1, 2, 3, 4, 5], [[4, 5], [1, 2]]) is True  # Multiple stop sequences


class TestInferenceEngineBenchmark:
    """Test benchmarking functionality."""
    
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
    
    def test_benchmark(self, small_model):
        """Test benchmark functionality."""
        engine = InferenceEngine.from_model(small_model)
        
        result = engine.benchmark(
            prompt_length=10,
            generate_length=5,
            warmup=1,
            runs=2,
        )
        
        assert "prefill_ms" in result
        assert "decode_ms" in result
        assert "tokens_per_sec" in result
        assert result["prefill_tokens"] == 10
        assert result["decode_tokens"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
