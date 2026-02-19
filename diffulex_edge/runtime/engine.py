"""Inference engine for DiffuLex Edge models.

Provides efficient inference for exported ExecuTorch models with
KV cache support.
"""

import dataclasses
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Iterator, Union
import torch
import torch.nn as nn

from .sampler import Sampler, GreedySampler, get_sampler


@dataclasses.dataclass
class GenerationConfig:
    """Configuration for text generation.
    
    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        stop_sequences: List of token sequences that stop generation
        pad_token_id: Token ID for padding
        eos_token_id: End-of-sequence token ID
        stream: Whether to stream results
    """
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_sequences: Optional[List[List[int]]] = None
    pad_token_id: int = 0
    eos_token_id: int = 2
    stream: bool = False
    
    def get_sampler(self, seed: Optional[int] = None) -> Sampler:
        """Create sampler based on configuration."""
        if self.temperature == 0:
            return GreedySampler()
        return get_sampler(
            strategy="top_p",
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            seed=seed,
        )


class InferenceEngine:
    """Inference engine for DiffuLex Edge models.
    
    Supports both PyTorch models (for development) and ExecuTorch
    models (for deployment).
    
    Usage:
        # Load PyTorch model
        engine = InferenceEngine.from_model(model)
        
        # Or load ExecuTorch model
        engine = InferenceEngine.from_pte("model.pte")
        
        # Generate text
        tokens = engine.generate(prompt_tokens, generation_config)
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        pte_path: Optional[Path] = None,
        device: str = "cpu",
        use_kv_cache: bool = True,
    ):
        """Initialize inference engine.
        
        Args:
            model: PyTorch model (if loading from model object)
            pte_path: Path to .pte file (if loading ExecuTorch model)
            device: Device to run on
            use_kv_cache: Whether to use KV cache
        """
        self.model = model
        self.pte_path = pte_path
        self.device = device
        self.use_kv_cache = use_kv_cache
        
        # State
        self._kv_cache: Optional[torch.Tensor] = None
        self._current_pos: int = 0
        self._is_pte: bool = pte_path is not None
        self._pte_module = None
        
        # Load ExecuTorch model if specified
        if self._is_pte:
            self._load_pte_model()
    
    def _load_pte_model(self):
        """Load ExecuTorch model."""
        try:
            from executorch.runtime import Runtime
            
            runtime = Runtime.get()
            self._pte_module = runtime.load_from_file(str(self.pte_path))
            print(f"Loaded ExecuTorch model from {self.pte_path}")
        except ImportError:
            raise ImportError(
                "ExecuTorch runtime not available. "
                "Install with: pip install executorch"
            )
    
    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        device: str = "cpu",
        use_kv_cache: bool = True,
    ) -> "InferenceEngine":
        """Create engine from PyTorch model.
        
        Args:
            model: PyTorch model
            device: Device to run on
            use_kv_cache: Whether to use KV cache
            
        Returns:
            Configured InferenceEngine
        """
        model.eval()
        model.to(device)
        return cls(model=model, device=device, use_kv_cache=use_kv_cache)
    
    @classmethod
    def from_pte(
        cls,
        pte_path: Union[str, Path],
        device: str = "cpu",
        use_kv_cache: bool = True,
    ) -> "InferenceEngine":
        """Create engine from .pte file.
        
        Args:
            pte_path: Path to .pte file
            device: Device to run on
            use_kv_cache: Whether to use KV cache
            
        Returns:
            Configured InferenceEngine
        """
        return cls(pte_path=Path(pte_path), device=device, use_kv_cache=use_kv_cache)
    
    def reset_cache(self):
        """Reset KV cache for new sequence."""
        self._kv_cache = None
        self._current_pos = 0
    
    def _prefill(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run prefill on input tokens.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            kv_cache: New KV cache (if using cache)
        """
        if self._is_pte:
            return self._prefill_pte(input_ids)
        else:
            return self._prefill_torch(input_ids)
    
    def _prefill_torch(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prefill using PyTorch model."""
        with torch.no_grad():
            if self.use_kv_cache:
                # Reset cache for new sequence
                self.reset_cache()
                
                # Create initial cache
                batch_size = input_ids.shape[0]
                num_layers = len(self.model.layers)
                num_kv_heads = self.model.config.num_key_value_heads
                head_dim = self.model.config.head_dim
                max_seq_len = self.model.config.max_position_embeddings
                
                self._kv_cache = torch.zeros(
                    num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim,
                    dtype=torch.float32, device=self.device
                )
                
                # Forward pass with cache
                logits, new_kv = self.model(
                    input_ids,
                    kv_cache=self._kv_cache,
                    start_pos=0,
                )
                
                # Update cache with new KV
                seq_len = input_ids.shape[1]
                self._kv_cache[:, :, :, :, :seq_len, :] = new_kv
                self._current_pos = seq_len
                
                return logits, self._kv_cache
            else:
                # No cache
                logits, _ = self.model(input_ids)
                return logits, None
    
    def _prefill_pte(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prefill using ExecuTorch model."""
        # ExecuTorch runtime interface
        # The exact interface depends on the ExecuTorch version
        # This is a placeholder implementation
        
        # Convert to list for ExecuTorch
        input_list = input_ids.tolist()
        
        # Run inference
        if self.use_kv_cache:
            # KV cache handling in ExecuTorch
            # This would use the model's exported interface
            result = self._pte_module.forward(input_list, 0)  # start_pos=0
        else:
            result = self._pte_module.forward(input_list)
        
        # Convert result back to tensor
        logits = torch.tensor(result[0] if isinstance(result, (list, tuple)) else result)
        
        return logits, None
    
    def _decode_step(
        self,
        input_id: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run single decode step.
        
        Args:
            input_id: Single token ID
            
        Returns:
            logits: Output logits [vocab_size]
            kv_cache: Updated KV cache
        """
        if self._is_pte:
            return self._decode_step_pte(input_id)
        else:
            return self._decode_step_torch(input_id)
    
    def _decode_step_torch(
        self,
        input_id: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode step using PyTorch model."""
        # Prepare single token input
        input_ids = torch.tensor([[input_id]], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            if self.use_kv_cache and self._kv_cache is not None:
                # Forward with existing cache
                logits, new_kv = self.model(
                    input_ids,
                    kv_cache=self._kv_cache,
                    start_pos=self._current_pos,
                )
                
                # Update cache
                self._kv_cache[:, :, :, :, self._current_pos:self._current_pos+1, :] = new_kv
                self._current_pos += 1
                
                return logits[0, -1], self._kv_cache
            else:
                # No cache - full forward pass (slower)
                logits, _ = self.model(input_ids)
                return logits[0, -1], None
    
    def _decode_step_pte(self, input_id: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode step using ExecuTorch model."""
        # ExecuTorch runtime for single step
        result = self._pte_module.forward([[input_id]], self._current_pos)
        
        logits = torch.tensor(result[0] if isinstance(result, (list, tuple)) else result)
        self._current_pos += 1
        
        return logits, None
    
    def generate(
        self,
        prompt_tokens: List[int],
        config: Optional[GenerationConfig] = None,
    ) -> List[int]:
        """Generate tokens from prompt.
        
        Args:
            prompt_tokens: List of prompt token IDs
            config: Generation configuration
            
        Returns:
            Generated token IDs (excluding prompt)
        """
        config = config or GenerationConfig()
        sampler = config.get_sampler()
        
        # Prefill
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        logits, _ = self._prefill(input_ids)
        
        # Get last token logits
        next_logits = logits[0, -1]
        
        # Track generated tokens
        generated_tokens = []
        all_tokens = list(prompt_tokens)
        
        # Generation loop
        for _ in range(config.max_new_tokens):
            # Apply repetition penalty if needed
            if config.repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, all_tokens, config.repetition_penalty
                )
            
            # Sample next token
            next_token = sampler.sample(next_logits)
            generated_tokens.append(next_token)
            all_tokens.append(next_token)
            
            # Check for stop conditions
            if next_token == config.eos_token_id:
                break
            
            if self._check_stop_sequences(all_tokens, config.stop_sequences):
                break
            
            # Decode next step
            next_logits, _ = self._decode_step(next_token)
        
        return generated_tokens
    
    def generate_stream(
        self,
        prompt_tokens: List[int],
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[int]:
        """Generate tokens with streaming.
        
        Args:
            prompt_tokens: List of prompt token IDs
            config: Generation configuration
            
        Yields:
            Generated token IDs one at a time
        """
        config = config or GenerationConfig()
        sampler = config.get_sampler()
        
        # Prefill
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        logits, _ = self._prefill(input_ids)
        next_logits = logits[0, -1]
        
        all_tokens = list(prompt_tokens)
        
        for _ in range(config.max_new_tokens):
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, all_tokens, config.repetition_penalty
                )
            
            # Sample
            next_token = sampler.sample(next_logits)
            yield next_token
            
            all_tokens.append(next_token)
            
            # Check stop conditions
            if next_token == config.eos_token_id:
                break
            
            if self._check_stop_sequences(all_tokens, config.stop_sequences):
                break
            
            # Next step
            next_logits, _ = self._decode_step(next_token)
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        previous_tokens: List[int],
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        logits = logits.clone()
        for token_id in set(previous_tokens):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        return logits
    
    def _check_stop_sequences(
        self,
        tokens: List[int],
        stop_sequences: Optional[List[List[int]]],
    ) -> bool:
        """Check if any stop sequence is matched."""
        if not stop_sequences:
            return False
        
        for stop_seq in stop_sequences:
            if len(tokens) >= len(stop_seq):
                if tokens[-len(stop_seq):] == stop_seq:
                    return True
        
        return False
    
    def benchmark(
        self,
        prompt_length: int = 128,
        generate_length: int = 100,
        warmup: int = 3,
        runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            prompt_length: Length of prompt tokens
            generate_length: Number of tokens to generate
            warmup: Number of warmup runs
            runs: Number of benchmark runs
            
        Returns:
            Dict with timing statistics
        """
        # Create dummy prompt
        prompt_tokens = list(range(prompt_length))
        
        # Warmup
        for _ in range(warmup):
            self.reset_cache()
            _ = self.generate(
                prompt_tokens,
                GenerationConfig(max_new_tokens=10),
            )
        
        # Benchmark
        prefill_times = []
        decode_times = []
        
        for _ in range(runs):
            self.reset_cache()
            
            # Prefill timing
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            start = time.perf_counter()
            logits, _ = self._prefill(input_ids)
            prefill_times.append(time.perf_counter() - start)
            
            # Decode timing
            start = time.perf_counter()
            for i in range(generate_length):
                if i == 0:
                    next_token = logits[0, -1].argmax().item()
                else:
                    logits_step, _ = self._decode_step(next_token)
                    next_token = logits_step.argmax().item()
            decode_times.append(time.perf_counter() - start)
        
        return {
            "prefill_ms": sum(prefill_times) / len(prefill_times) * 1000,
            "decode_ms": sum(decode_times) / len(decode_times) * 1000,
            "tokens_per_sec": generate_length / (sum(decode_times) / len(decode_times)),
            "prefill_tokens": prompt_length,
            "decode_tokens": generate_length,
        }
