"""Enhanced DiffusionEngine with model-specific samplers.

This module provides an upgraded DiffusionEngine that supports
all four model types: FastdLLM V2, LLaDA, Dream, and SDAR.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union, Any

import torch

from diffulex_edge.runtime.block import DiffusionBlockManager
from diffulex_edge.runtime.sampler.models import (
    SAMPLER_REGISTRY,
    FastdLLMV2SampleOutput,
    LLaDASampleOutput,
    DreamSampleOutput,
    SDARSampleOutput,
)

logger = logging.getLogger(__name__)


class DiffusionGenerationConfig:
    """Configuration for diffusion-based generation.
    
    Attributes:
        max_new_tokens: Maximum tokens to generate
        num_iterations: Number of diffusion iterations (denoising steps)
        block_size: Size of each diffusion block
        confidence_threshold: Threshold for accepting tokens
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        mask_token_id: Token ID for masking
        eos_token_id: End-of-sequence token ID
        early_stop: Whether to stop when all blocks complete
        model_type: Model type ("fast_dllm_v2", "llada", "dream", "sdar")
    """
    
    def __init__(
        self,
        max_new_tokens: int = 100,
        num_iterations: int = 10,
        block_size: int = 10,
        confidence_threshold: float = 0.95,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        mask_token_id: int = 126336,
        eos_token_id: int = 2,
        early_stop: bool = True,
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
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.early_stop = early_stop
        self.model_type = model_type
        self.margin_confidence = margin_confidence
        self.neg_entropy = neg_entropy


class DiffusionEngine:
    """Enhanced inference engine supporting all model types.
    
    Supports:
    - FastdLLM V2: Shift logits, global threshold, always accept 1
    - LLaDA: No shift, per-block threshold, pre_block_complete logic
    - Dream: Shift logits, per-block threshold, pre_block_complete logic
    - SDAR: Shift logits, global threshold, always accept 1
    
    Usage:
        # Load with specific model type
        engine = DiffusionEngine.from_pte("model.pte", model_type="llada")
        
        # Generate with model-specific sampling
        config = DiffusionGenerationConfig(model_type="llada")
        tokens = engine.generate(prompt_tokens, config)
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        pte_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        model_type: str = "fast_dllm_v2",
    ):
        """Initialize diffusion engine.
        
        Args:
            model: PyTorch model for inference
            pte_path: Path to .pte file (if loading ExecuTorch model)
            device: Device to run on
            model_type: Model type for sampler selection
            
        Raises:
            ValueError: If model_type is invalid
        """
        if model_type not in SAMPLER_REGISTRY:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Available: {list(SAMPLER_REGISTRY.keys())}"
            )
        
        if model is not None and pte_path is not None:
            raise ValueError(
                "Cannot provide both 'model' and 'pte_path'. "
                "Use from_model() or from_pte() for clarity."
            )
        
        self.model = model
        self.pte_path = Path(pte_path) if pte_path is not None else None
        self.device = device
        self.model_type = model_type
        self.block_manager = DiffusionBlockManager()
        self.sampler = None  # Will be created in generate()
        
        # PTE state
        self._is_pte: bool = self.pte_path is not None
        self._pte_module: Any = None
        self._pte_program: Any = None
        
        # Load model if provided
        if model is not None:
            self.model.eval()
            self.model.to(device)
            logger.debug(f"Loaded PyTorch model on {device}")
        elif self.pte_path is not None:
            self._load_pte_model()
    
    def _load_pte_model(self) -> None:
        """Load ExecuTorch model from pte_path."""
        if self.pte_path is None:
            raise RuntimeError("pte_path is None - cannot load PTE model")
        
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
            raise ImportError(
                "ExecuTorch runtime not available. "
                "Install with: pip install executorch"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PTE model from {self.pte_path}: {e}"
            ) from e
    
    def _create_sampler(self, config: DiffusionGenerationConfig):
        """Create model-specific sampler."""
        sampler_cls = SAMPLER_REGISTRY.get(self.model_type)
        if sampler_cls is None:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        # Different samplers have different parameters
        if self.model_type in ["fast_dllm_v2", "sdar"]:
            # Global threshold
            return sampler_cls(
                mask_token_id=config.mask_token_id,
                threshold=config.confidence_threshold,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                margin_confidence=config.margin_confidence,
                neg_entropy=config.neg_entropy,
            )
        else:  # llada, dream
            # Per-block threshold
            return sampler_cls(
                mask_token_id=config.mask_token_id,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                margin_confidence=config.margin_confidence,
                neg_entropy=config.neg_entropy,
            )
    
    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        device: str = "cpu",
        model_type: str = "fast_dllm_v2",
    ) -> "DiffusionEngine":
        """Create engine from PyTorch model.
        
        Args:
            model: PyTorch model
            device: Device to run on
            model_type: Model type for sampler selection
            
        Returns:
            Configured DiffusionEngine
        """
        model.eval()
        model.to(device)
        logger.debug(f"Creating DiffusionEngine from PyTorch model on {device}")
        return cls(model=model, device=device, model_type=model_type)
    
    @classmethod
    def from_pte(
        cls,
        pte_path: Union[str, Path],
        device: str = "cpu",
        model_type: str = "fast_dllm_v2",
    ) -> "DiffusionEngine":
        """Create engine from .pte file.
        
        Args:
            pte_path: Path to .pte file
            device: Device to run on
            model_type: Model type for sampler selection
            
        Returns:
            Configured DiffusionEngine
        """
        logger.debug(f"Creating DiffusionEngine from PTE: {pte_path}")
        return cls(pte_path=Path(pte_path), device=device, model_type=model_type)
    
    def generate(
        self,
        prompt_tokens: List[int],
        config: Optional[DiffusionGenerationConfig] = None,
    ) -> List[int]:
        """Generate tokens using diffusion sampling.
        
        Uses model-specific sampler based on model_type.
        
        Args:
            prompt_tokens: List of prompt token IDs
            config: Generation configuration
            
        Returns:
            Generated token IDs (including prompt)
        """
        if config is None:
            config = DiffusionGenerationConfig(model_type=self.model_type)
        else:
            # Ensure model_type matches
            config.model_type = self.model_type
        
        # Create model-specific sampler
        self.sampler = self._create_sampler(config)
        
        # Reset state
        self.block_manager.reset()
        if hasattr(self.sampler, 'reset'):
            self.sampler.reset()
        
        # Initialize sequence with prompt
        sequence = list(prompt_tokens)
        prompt_len = len(prompt_tokens)
        
        # Create initial mask tokens
        num_blocks = (config.max_new_tokens + config.block_size - 1) // config.block_size
        
        for i in range(num_blocks):
            start_pos = prompt_len + i * config.block_size
            remaining = config.max_new_tokens - i * config.block_size
            block_len = min(config.block_size, remaining)
            
            # Create block with appropriate threshold
            self.block_manager.create_block(
                start_pos=start_pos,
                length=block_len,
                accept_threshold=config.confidence_threshold,
            )
            
            # Add mask tokens to sequence
            sequence.extend([config.mask_token_id] * block_len)
        
        logger.debug(
            f"Starting generation: model_type={self.model_type}, "
            f"prompt_len={prompt_len}, num_blocks={num_blocks}"
        )
        
        # Iterative denoising
        for iteration in range(config.num_iterations):
            if config.early_stop and not self.block_manager.has_active_blocks():
                logger.debug(f"Early stop at iteration {iteration}")
                break
            
            # Prepare input
            input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
            
            # Forward pass
            logits = self._forward(input_ids)  # [1, seq_len, vocab_size]
            logits = logits[0]  # [seq_len, vocab_size]
            
            # Sample using model-specific sampler
            output = self.sampler.sample(self.block_manager, logits)
            
            # Extract accepted tokens from output
            accepted_tokens = {}
            for seq_id, block_map in output.accepted_ids_map.items():
                for block_id, indices in block_map.items():
                    # Get the actual tokens
                    block_tokens = output.sampled_tokens_map[seq_id][block_id]
                    for i, idx in enumerate(indices):
                        # Find global position from local index
                        block = self.block_manager.blocks[int(block_id)]
                        if idx < len(block_tokens):
                            global_pos = block.start_pos + idx
                            accepted_tokens[global_pos] = block_tokens[idx]
            
            # Update sequence with accepted tokens
            for pos, token_id in accepted_tokens.items():
                sequence[pos] = token_id
            
            num_accepted = len(accepted_tokens)
            logger.debug(f"Iteration {iteration}: accepted {num_accepted} tokens")
        
        logger.debug(f"Generation complete: sequence length={len(sequence)}")
        return sequence
    
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass through model."""
        if self._is_pte:
            return self._forward_pte(input_ids)
        else:
            return self._forward_torch(input_ids)
    
    def _forward_torch(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass using PyTorch model."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        with torch.no_grad():
            return self.model(input_ids)[0]
    
    def _forward_pte(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass using ExecuTorch model."""
        if self._pte_module is None:
            raise RuntimeError("No PTE model loaded")
        
        input_list = input_ids.tolist()
        result = self._pte_module.forward(input_list)
        
        if isinstance(result, (list, tuple)):
            logits = torch.tensor(result[0])
        else:
            logits = torch.tensor(result)
        
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0).unsqueeze(0)
        
        return logits