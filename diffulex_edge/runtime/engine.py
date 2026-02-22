"""Unified DiffusionEngine with optional KV cache support.

Supports both PyTorch models and ExecuTorch PTE models:
- PyTorch models: Can use KV cache for efficient inference
- PTE models: Full-sequence inference (KV cache managed by model or disabled)

Usage:
    # PyTorch model with KV cache
    engine = DiffusionEngine.from_model(model, model_type="sdar")
    
    # PyTorch model without KV cache
    engine = DiffusionEngine.from_model(model, model_type="sdar", use_kv_cache=False)
    
    # ExecuTorch PTE model
    engine = DiffusionEngine.from_pte("model.pte", model_type="sdar", max_seq_len=2048)
"""

import logging
from pathlib import Path
from typing import List, Optional, Union, Any, Tuple

import torch

from diffulex_edge.runtime.block import DiffusionBlockManager, BlockStatus
from diffulex_edge.runtime.sampler.models import SAMPLER_REGISTRY

logger = logging.getLogger(__name__)


class DiffusionGenerationConfig:
    """Configuration for diffusion-based generation.
    
    Attributes:
        max_new_tokens: Maximum number of new tokens to generate
        num_iterations: Maximum iterations per block
        block_size: Size of each generation block
        confidence_threshold: Confidence threshold for accepting tokens
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (0 to disable)
        top_p: Top-p (nucleus) sampling parameter (1.0 to disable)
        mask_token_id: Token ID for masked positions
        eos_token_id: End-of-sequence token ID
        model_type: Model type for sampler selection
        margin_confidence: Use margin confidence (top1 - top2)
        neg_entropy: Use negative entropy as confidence
    """
    
    # Model-specific default mask token IDs
    _DEFAULT_MASK_TOKENS = {
        "fast_dllm_v2": 151665,
        "dream": 151666,
        "llada": 126336,
        "sdar": 151669,  # SDAR uses <|MASK|> token at 151669
    }
    
    def __init__(
        self,
        max_new_tokens: int = 100,
        num_iterations: int = 10,
        block_size: int = 4,
        confidence_threshold: float = 0.95,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        mask_token_id: Optional[int] = None,  # None means use model-specific default
        eos_token_id: int = 2,
        model_type: str = "fast_dllm_v2",
        margin_confidence: bool = False,
        neg_entropy: bool = False,
    ):
        self.max_new_tokens = max_new_tokens
        self.num_iterations = num_iterations
        self.block_size = block_size
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # Use model-specific default if mask_token_id not provided
        if mask_token_id is None:
            mask_token_id = self._DEFAULT_MASK_TOKENS.get(model_type, 126336)
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.model_type = model_type
        self.margin_confidence = margin_confidence
        self.neg_entropy = neg_entropy


class DiffusionEngine:
    """Unified inference engine for diffusion models.
    
    Supports both KV cache mode (efficient for PyTorch) and full-sequence mode
    (for PTE compatibility).
    
    Args:
        model: PyTorch model (optional)
        pte_path: Path to ExecuTorch .pte file (optional)
        device: Device to run on ("cpu", "cuda", etc.)
        model_type: Model type for sampler selection
        max_seq_len: Maximum sequence length (required for PTE models)
        use_kv_cache: Whether to use KV cache (PyTorch models only)
    
    Raises:
        ValueError: If both model and pte_path are provided, or if model_type is unknown
        FileNotFoundError: If pte_path does not exist
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        pte_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        model_type: str = "fast_dllm_v2",
        max_seq_len: Optional[int] = None,
        use_kv_cache: bool = True,
    ):
        if model_type not in SAMPLER_REGISTRY:
            raise ValueError(f"Unknown model_type: {model_type}. "
                           f"Available: {list(SAMPLER_REGISTRY.keys())}")
        
        if model is not None and pte_path is not None:
            raise ValueError("Cannot provide both 'model' and 'pte_path'")
        
        if model is None and pte_path is None:
            raise ValueError("Must provide either 'model' or 'pte_path'")
        
        self.model = model
        self.pte_path = Path(pte_path) if pte_path is not None else None
        self.device = device
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.use_kv_cache = use_kv_cache and (model is not None)  # KV cache only for PyTorch
        
        # Use model-specific default mask token
        default_mask_tokens = {
            "fast_dllm_v2": 151665,
            "dream": 151666,
            "llada": 126336,
            "sdar": 151669,  # SDAR uses <|MASK|> token at 151669
        }
        mask_token = default_mask_tokens.get(model_type, 126336)
        self.block_manager = DiffusionBlockManager(mask_token_id=mask_token)
        self.sampler: Optional[Any] = None
        
        # PTE state
        self._is_pte: bool = self.pte_path is not None
        self._pte_module: Optional[Any] = None
        
        if model is not None:
            self.model.eval()
            self.model.to(device)
            logger.info(f"Loaded PyTorch model on {device}, KV cache: {self.use_kv_cache}")
        elif self.pte_path is not None:
            self._load_pte_model()
            self.use_kv_cache = False  # PTE models don't use Python-level KV cache
    
    def _load_pte_model(self) -> None:
        """Load ExecuTorch model."""
        if not self.pte_path.exists():
            raise FileNotFoundError(f"PTE file not found: {self.pte_path}")
        
        try:
            from executorch.runtime import Runtime, Verification
            
            runtime = Runtime.get()
            self._pte_program = runtime.load_program(
                str(self.pte_path),
                verification=Verification.Minimal
            )
            self._pte_module = self._pte_program.load_method("forward")
            logger.info(f"Loaded ExecuTorch model from {self.pte_path}")
        except ImportError as e:
            raise ImportError("ExecuTorch runtime not available. "
                            "Install with: pip install executorch") from e
    
    def _create_sampler(self, config: DiffusionGenerationConfig) -> Any:
        """Create model-specific sampler."""
        sampler_cls = SAMPLER_REGISTRY.get(self.model_type)
        if sampler_cls is None:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        common_kwargs = {
            "mask_token_id": config.mask_token_id,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "margin_confidence": config.margin_confidence,
            "neg_entropy": config.neg_entropy,
        }
        
        if self.model_type in ["fast_dllm_v2", "sdar"]:
            common_kwargs["threshold"] = config.confidence_threshold
        
        return sampler_cls(**common_kwargs)
    
    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        device: str = "cpu",
        model_type: str = "fast_dllm_v2",
        use_kv_cache: bool = True,
    ) -> "DiffusionEngine":
        """Create engine from PyTorch model.
        
        Args:
            model: PyTorch model
            device: Device to run on
            model_type: Model type for sampler selection
            use_kv_cache: Whether to use KV cache
            
        Returns:
            Configured DiffusionEngine
        """
        model.eval()
        model.to(device)
        return cls(model=model, device=device, model_type=model_type, use_kv_cache=use_kv_cache)
    
    @classmethod
    def from_pte(
        cls,
        pte_path: Union[str, Path],
        device: str = "cpu",
        model_type: str = "fast_dllm_v2",
        max_seq_len: Optional[int] = None,
    ) -> "DiffusionEngine":
        """Create engine from ExecuTorch .pte file.
        
        Args:
            pte_path: Path to .pte file
            device: Device to run on (usually "cpu" for ExecuTorch)
            model_type: Model type for sampler selection
            max_seq_len: Maximum sequence length (required for padding)
            
        Returns:
            Configured DiffusionEngine
        """
        return cls(
            pte_path=Path(pte_path),
            device=device,
            model_type=model_type,
            max_seq_len=max_seq_len,
            use_kv_cache=False,
        )
    
    def generate(
        self,
        prompt_tokens: List[int],
        config: Optional[DiffusionGenerationConfig] = None,
    ) -> List[int]:
        """Generate tokens using diffusion-based block confirmation.
        
        Uses iterative block refinement where each block is processed
        multiple times until confidence threshold is met or max iterations.
        
        Args:
            prompt_tokens: Initial prompt token IDs
            config: Generation configuration (uses defaults if None)
            
        Returns:
            List of token IDs (prompt + generated)
        """
        if config is None:
            config = DiffusionGenerationConfig(model_type=self.model_type)
        else:
            config.model_type = self.model_type
        
        self.sampler = self._create_sampler(config)
        self.block_manager.reset()
        if hasattr(self.sampler, 'reset'):
            self.sampler.reset()
        
        prompt_len = len(prompt_tokens)
        
        # Initialize blocks
        self.block_manager.create_blocks(
            prompt_len=prompt_len,
            max_new_tokens=config.max_new_tokens,
            block_size=config.block_size,
            accept_threshold=config.confidence_threshold,
        )
        
        logger.info(f"Generation: model={self.model_type}, prompt={prompt_len}, "
                   f"blocks={len(self.block_manager)}, kv_cache={self.use_kv_cache}")
        
        # Choose generation strategy
        if self.use_kv_cache:
            return self._generate_with_kv_cache(prompt_tokens, config)
        else:
            return self._generate_full_sequence(prompt_tokens, config)
    
    def _generate_with_kv_cache(
        self,
        prompt_tokens: List[int],
        config: DiffusionGenerationConfig,
    ) -> List[int]:
        """Generate using KV cache for efficient incremental inference."""
        prompt_len = len(prompt_tokens)
        kv_cache = None
        cached_len = 0
        
        # Step 1: Prefill - process prompt
        if prompt_len > 0:
            prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            prompt_positions = torch.arange(prompt_len, device=self.device).unsqueeze(0)
            
            # For diffusion models, use block causal mask
            prompt_mask = self._create_diffusion_mask(prompt_len, prompt_len, config.block_size)
            
            _, kv_cache = self._forward(
                input_ids=prompt_ids,
                positions=prompt_positions,
                mask=prompt_mask,
                kv_cache=kv_cache
            )
            cached_len = prompt_len
            logger.debug(f"Prefill complete: cached {cached_len} tokens")
        
        # Step 2: Generate blocks incrementally
        total_iterations = 0
        confirmed_blocks = 0
        
        while self.block_manager.has_active_block():
            active_block = self.block_manager.get_active_block()
            block_len = active_block.length
            
            block_confirmed = False
            final_kv_cache = None
            
            for iteration in range(config.num_iterations):
                total_iterations += 1
                
                # Build input for this block
                block_tokens = active_block.get_sequence_tokens()
                block_ids = torch.tensor([block_tokens], dtype=torch.long, device=self.device)
                block_positions = torch.arange(
                    cached_len, cached_len + block_len,
                    device=self.device
                ).unsqueeze(0)
                
                # Diffusion mask: block can see all cached + itself (block causal)
                block_mask = self._create_diffusion_mask(block_len, cached_len + block_len, config.block_size)
                
                # Truncate KV cache to cached_len before each iteration
                truncated_kv_cache = self._truncate_kv_cache(kv_cache, cached_len)
                
                # Forward
                logits, new_kv_cache = self._forward(
                    input_ids=block_ids,
                    positions=block_positions,
                    mask=block_mask,
                    kv_cache=truncated_kv_cache
                )
                
                final_kv_cache = new_kv_cache
                
                # Sample
                output = self.sampler.sample(self.block_manager, logits[0])
                
                if output.block_confirmed:
                    block_confirmed = True
                    break
            
            confirmed_blocks += 1 if block_confirmed else 0
            
            # Update KV cache with final state
            if final_kv_cache is not None:
                kv_cache = final_kv_cache
            
            cached_len += block_len
            
            # Advance block
            has_more = self.block_manager.confirm_current_block()
            if not has_more:
                break
        
        # Build final sequence
        final_sequence = self.block_manager.build_sequence(prompt_tokens)
        
        logger.info(f"Done: {confirmed_blocks}/{len(self.block_manager)} blocks confirmed, "
                   f"{total_iterations} total iterations")
        return final_sequence
    
    def _generate_full_sequence(
        self,
        prompt_tokens: List[int],
        config: DiffusionGenerationConfig,
    ) -> List[int]:
        """Generate using full-sequence inference (no KV cache).
        
        Used for PTE models or when KV cache is disabled.
        """
        confirmed_tokens = list(prompt_tokens)
        total_iterations = 0
        confirmed_blocks = 0
        
        while self.block_manager.has_active_block():
            active_block = self.block_manager.get_active_block()
            
            # Build full sequence: confirmed + active masks
            sequence = self._build_sequence(confirmed_tokens, active_block, config.mask_token_id)
            
            block_confirmed = False
            for iteration in range(config.num_iterations):
                total_iterations += 1
                
                # Rebuild sequence with confirmed tokens from active block
                sequence = self._build_sequence(
                    confirmed_tokens, 
                    active_block, 
                    config.mask_token_id
                )
                
                # Forward pass
                logits = self._forward_full_sequence(sequence)
                
                # Get logits for active block only
                active_start = len(confirmed_tokens)
                active_end = active_start + active_block.length
                active_logits = logits[active_start:active_end]
                
                # Sample
                output = self.sampler.sample(self.block_manager, active_logits)
                
                if output.block_confirmed:
                    block_confirmed = True
                    break
            
            confirmed_blocks += 1 if block_confirmed else 0
            
            # Get confirmed tokens for this block
            block_tokens = active_block.get_sequence_tokens()
            confirmed_tokens.extend(block_tokens)
            
            # Advance
            has_more = self.block_manager.confirm_current_block()
            if not has_more:
                break
        
        logger.info(f"Done: {confirmed_blocks}/{len(self.block_manager)} blocks confirmed, "
                   f"{total_iterations} total iterations")
        return confirmed_tokens
    
    def _build_sequence(
        self,
        confirmed_tokens: List[int],
        active_block: Any,
        mask_token_id: int
    ) -> List[int]:
        """Build sequence: confirmed + active block tokens (confirmed or mask)."""
        sequence = list(confirmed_tokens)
        # Use active_block's confirmed tokens or mask tokens
        sequence.extend(active_block.get_sequence_tokens())
        
        # Pad to max_seq_len for PTE
        if self._is_pte and self.max_seq_len is not None:
            while len(sequence) < self.max_seq_len:
                sequence.append(mask_token_id)
        
        return sequence
    
    def _truncate_kv_cache(
        self,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        length: int
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Truncate KV cache to specified length.
        
        Creates independent copies (not views) to avoid modifying original.
        """
        if kv_cache is None:
            return None
        
        truncated = []
        for k_cache, v_cache in kv_cache:
            truncated_k = k_cache[:, :, :length, :].clone()
            truncated_v = v_cache[:, :, :length, :].clone()
            truncated.append((truncated_k, truncated_v))
        return truncated
    
    def _create_diffusion_mask(
        self,
        query_len: int,
        kv_len: int,
        block_size: int = 32,
    ) -> torch.Tensor:
        """Create diffusion mask based on model type.
        
        Supports two mask types:
        - Block causal (fast_dllm_v2, sdar): query can attend to KV if query's block >= KV's block
        - Full attention (dream, llada): all positions can attend to all positions
        
        Args:
            query_len: Length of query (current positions)
            kv_len: Length of key/value (cached + current)
            block_size: Size of each block for block causal attention
            
        Returns:
            Mask tensor [1, 1, query_len, kv_len] with 0 for visible, -inf for masked
        """
        # Models that use block causal attention
        block_causal_models = ["fast_dllm_v2", "sdar"]
        
        if self.model_type in block_causal_models:
            # Block causal mask: query at position i can see KV at position j if block(i) >= block(j)
            # Use bd_size from model config if available, otherwise use block_size parameter
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'bd_size'):
                bd_size = self.model.config.bd_size
            else:
                bd_size = block_size
            
            cache_len = kv_len - query_len
            q_indices = torch.arange(query_len, device=self.device) + cache_len
            k_indices = torch.arange(kv_len, device=self.device)
            
            q_blocks = q_indices // bd_size
            k_blocks = k_indices // bd_size
            
            # Block causal: q can attend to k if q_block >= k_block
            mask = q_blocks.unsqueeze(1) >= k_blocks.unsqueeze(0)
            
            # Convert to additive mask (0 for keep, -inf for mask)
            additive_mask = torch.where(mask, 0.0, float('-inf'))
            return additive_mask.unsqueeze(0).unsqueeze(0)
        else:
            # Full attention for other models (dream, llada)
            # All positions can attend to all positions
            mask = torch.zeros(query_len, kv_len, device=self.device)
            return mask.unsqueeze(0).unsqueeze(0)
    
    def _forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass (PyTorch with KV cache)."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        with torch.no_grad():
            return self.model(input_ids, positions, mask, kv_cache)
    
    def _forward_full_sequence(
        self,
        sequence: List[int]
    ) -> torch.Tensor:
        """Forward pass for full sequence (no KV cache).
        
        Used for PTE models or when KV cache is disabled.
        """
        input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
        
        if self._is_pte:
            return self._forward_pte(input_ids)
        else:
            return self._forward_torch_full(input_ids)
    
    def _forward_torch_full(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward using PyTorch model (full sequence, no KV cache)."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Create block causal mask for diffusion models
        # query at position i can see key at position j if block(i) >= block(j)
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'bd_size'):
            bd_size = self.model.config.bd_size
        else:
            bd_size = 32
        
        q_indices = torch.arange(seq_len, device=input_ids.device)
        k_indices = torch.arange(seq_len, device=input_ids.device)
        q_blocks = q_indices // bd_size
        k_blocks = k_indices // bd_size
        
        # Block causal: q can attend to k if q_block >= k_block
        mask_bool = q_blocks.unsqueeze(1) >= k_blocks.unsqueeze(0)
        mask = torch.where(mask_bool, 0.0, float('-inf')).to(input_ids.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        with torch.no_grad():
            logits, _ = self.model(input_ids, positions, mask, None)
        
        return logits[0] if logits.dim() == 3 else logits
    
    def _forward_pte(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward using ExecuTorch model."""
        if self._pte_module is None:
            raise RuntimeError("No PTE model loaded")
        
        result = self._pte_module.execute([input_ids])
        
        # Unwrap result
        if hasattr(result, '__iter__') and not isinstance(result, torch.Tensor):
            result = list(result)[0]
            if hasattr(result, 'to_torch'):
                result = result.to_torch()
        
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result)
        
        if result.dim() == 2:
            result = result.unsqueeze(0)
        
        # Truncate to actual length
        actual_len = input_ids.shape[1]
        if self.max_seq_len is not None and actual_len < self.max_seq_len:
            result = result[0, :actual_len, :]
        else:
            result = result[0]
        
        return result


# Backward compatibility aliases
InferenceEngine = DiffusionEngine
GenerationConfig = DiffusionGenerationConfig
