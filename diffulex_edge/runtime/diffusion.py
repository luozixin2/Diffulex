"""Diffusion sampling for dLLM models.

Provides diffusion-based generation capabilities for DiffuLex Edge models,
including DiffusionBlock management, logits shifting, and confidence-based
token acceptance.
"""

import logging
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterator, Union
import math

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class DiffusionBlock:
    """Represents a block of tokens in diffusion generation.
    
    A diffusion block tracks which positions are masked and which have been
    accepted during the iterative denoising process.
    
    Attributes:
        start_pos: Start position of this block in the sequence
        length: Length of the block
        mask_token_id: Token ID used for masking
        is_active: Whether this block is still being processed
        accepted_token_ids: Map from local position to accepted token ID
        global_mask_token_ids: Global positions of tokens still masked
    """
    start_pos: int
    length: int
    mask_token_id: int = 126336  # Default mask token for FastdLLM
    is_active: bool = True
    accepted_token_ids: Dict[int, int] = field(default_factory=dict)
    
    @property
    def global_mask_token_ids(self) -> List[int]:
        """Get global positions of tokens that are still masked."""
        return [
            self.start_pos + i 
            for i in range(self.length) 
            if i not in self.accepted_token_ids
        ]
    
    @property
    def is_complete(self) -> bool:
        """Check if all tokens in this block have been accepted."""
        return len(self.accepted_token_ids) == self.length
    
    def accept_token(self, local_pos: int, token_id: int) -> None:
        """Accept a token at the given local position.
        
        Args:
            local_pos: Position within the block (0 to length-1)
            token_id: Token ID to accept
            
        Raises:
            IndexError: If local_pos is out of range
            ValueError: If token already accepted at this position
        """
        if local_pos < 0 or local_pos >= self.length:
            raise IndexError(
                f"Local position {local_pos} out of range [0, {self.length})"
            )
        if local_pos in self.accepted_token_ids:
            raise ValueError(f"Token already accepted at position {local_pos}")
        
        self.accepted_token_ids[local_pos] = token_id
        
        # Mark block as inactive if complete
        if self.is_complete:
            self.is_active = False
    
    def get_global_pos(self, local_pos: int) -> int:
        """Convert local position to global position."""
        if local_pos < 0 or local_pos >= self.length:
            raise IndexError(
                f"Local position {local_pos} out of range [0, {self.length})"
            )
        return self.start_pos + local_pos


class DiffusionBlockManager:
    """Manages multiple diffusion blocks during generation.
    
    Handles creation, tracking, and lifecycle of diffusion blocks.
    """
    
    def __init__(self, mask_token_id: int = 126336):
        """Initialize the block manager.
        
        Args:
            mask_token_id: Token ID used for masking
        """
        self.mask_token_id = mask_token_id
        self.blocks: List[DiffusionBlock] = []
        self._block_id_counter = 0
    
    def create_block(self, start_pos: int, length: int) -> int:
        """Create a new diffusion block.
        
        Args:
            start_pos: Start position in the sequence
            length: Length of the block
            
        Returns:
            Block ID for the created block
            
        Raises:
            ValueError: If length <= 0 or start_pos < 0
        """
        if length <= 0:
            raise ValueError(f"Block length must be positive, got {length}")
        if start_pos < 0:
            raise ValueError(f"Start position must be non-negative, got {start_pos}")
        
        block = DiffusionBlock(
            start_pos=start_pos,
            length=length,
            mask_token_id=self.mask_token_id,
        )
        
        block_id = self._block_id_counter
        self.blocks.append(block)
        self._block_id_counter += 1
        
        return block_id
    
    def get_block(self, block_id: int) -> Optional[DiffusionBlock]:
        """Get a block by ID.
        
        Args:
            block_id: Block identifier
            
        Returns:
            The DiffusionBlock or None if not found
        """
        if block_id < 0 or block_id >= len(self.blocks):
            return None
        return self.blocks[block_id]
    
    def get_active_blocks(self) -> List[Tuple[int, DiffusionBlock]]:
        """Get all active blocks with their IDs.
        
        Returns:
            List of (block_id, block) tuples for active blocks
        """
        return [
            (i, block) 
            for i, block in enumerate(self.blocks) 
            if block.is_active
        ]
    
    def update_block(
        self, 
        block_id: int, 
        accepted_positions: List[int], 
        accepted_tokens: List[int]
    ) -> None:
        """Update a block with newly accepted tokens.
        
        Args:
            block_id: Block to update
            accepted_positions: Local positions being accepted
            accepted_tokens: Token IDs to accept
            
        Raises:
            IndexError: If block_id is invalid
            ValueError: If positions and tokens length mismatch
        """
        if block_id < 0 or block_id >= len(self.blocks):
            raise IndexError(f"Invalid block_id: {block_id}")
        
        if len(accepted_positions) != len(accepted_tokens):
            raise ValueError(
                f"Mismatched lengths: positions={len(accepted_positions)}, "
                f"tokens={len(accepted_tokens)}"
            )
        
        block = self.blocks[block_id]
        for pos, token in zip(accepted_positions, accepted_tokens):
            block.accept_token(pos, token)
    
    def has_active_blocks(self) -> bool:
        """Check if any blocks are still active."""
        return any(block.is_active for block in self.blocks)
    
    def get_all_mask_positions(self) -> List[int]:
        """Get all global positions that are still masked across all blocks."""
        positions = []
        for block in self.blocks:
            positions.extend(block.global_mask_token_ids)
        return positions
    
    def reset(self) -> None:
        """Clear all blocks and reset state."""
        self.blocks.clear()
        self._block_id_counter = 0
    
    def __len__(self) -> int:
        """Return number of blocks."""
        return len(self.blocks)


@dataclass
class SampleOutput:
    """Output from diffusion sampling.
    
    Attributes:
        sampled_tokens: Map from position to sampled token ID
        confidence_scores: Map from position to confidence score
        accepted_tokens: Map from position to accepted token ID (high confidence)
        block_states: Final state of each block
    """
    sampled_tokens: Dict[int, int] = field(default_factory=dict)
    confidence_scores: Dict[int, float] = field(default_factory=dict)
    accepted_tokens: Dict[int, int] = field(default_factory=dict)
    block_states: List[Dict[str, Any]] = field(default_factory=list)


class DiffusionSampler:
    """Diffusion sampler for dLLM models.
    
    Implements the core diffusion sampling algorithm with:
    - Logits shifting for dependency modeling
    - Confidence-based token acceptance
    - Temperature, top-k, and top-p sampling
    - Margin confidence and negative entropy options (aligned with Diffulex)
    """
    
    def __init__(
        self,
        mask_token_id: int = 126336,
        confidence_threshold: float = 0.95,
        temperature: float = 1.0,
        top_k: int = 0,  # 0 means disabled
        top_p: float = 1.0,  # 1.0 means disabled
        margin_confidence: bool = False,
        neg_entropy: bool = False,
    ):
        """Initialize diffusion sampler.
        
        Args:
            mask_token_id: Token ID for mask tokens
            confidence_threshold: Threshold for accepting tokens (0-1)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter (0 = disabled)
            top_p: Top-p (nucleus) sampling parameter (1.0 = disabled)
            margin_confidence: Use margin confidence (top1 - top2 probs)
            neg_entropy: Use negative entropy as confidence
        """
        self.mask_token_id = mask_token_id
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.margin_confidence = margin_confidence
        self.neg_entropy = neg_entropy
        
        # Cache for last logits (for multi-step generation)
        self._last_logits: Optional[torch.Tensor] = None
    
    def shift_logits(
        self, 
        logits: torch.Tensor, 
        last_logits: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Shift logits right by one position.
        
        This is the core operation for diffusion models that shifts
        the prediction target to model token dependencies.
        
        Aligned with Diffulex SamplerShiftLogits._shift_logits:
        - Position 0 gets last_logits (or 1.0 if not provided)
        - Position i gets logits from position i-1
        
        Args:
            logits: Logits tensor [seq_len, vocab_size]
            last_logits: Logits for the last position (for circular shift)
                        If None and use_cache=True, use cached value or last logits
            use_cache: Whether to use/update cached last_logits
                        
        Returns:
            Shifted logits of same shape
            
        Raises:
            ValueError: If logits has invalid shape or contains NaN/Inf
        """
        if logits.dim() != 2:
            raise ValueError(f"Expected 2D logits [seq_len, vocab_size], got {logits.shape}")
        
        if torch.isnan(logits).any():
            raise ValueError("Logits contain NaN values")
        
        if torch.isinf(logits).any():
            raise ValueError("Logits contain Inf values")
        
        seq_len, vocab_size = logits.shape
        
        if seq_len == 0:
            # Empty logits - return as is
            return logits
        
        # Determine last_logits to use
        if last_logits is None:
            if use_cache and self._last_logits is not None:
                last_logits = self._last_logits
            else:
                # Use last logit from current batch
                last_logits = logits[-1]
        
        if use_cache:
            # Cache for next iteration
            self._last_logits = last_logits.clone()
        
        # Shift right: position i gets logits from position i-1
        # Position 0 gets last_logits (or 1.0 if no last_logits available)
        shifted = torch.zeros_like(logits)
        if last_logits is not None:
            shifted[0] = last_logits
        else:
            # Fallback: use 1.0 (aligned with Diffulex)
            shifted[0] = 1.0
        
        if seq_len > 1:
            shifted[1:] = logits[:-1]
        
        return shifted
    
    def reset_cache(self):
        """Reset cached last logits."""
        self._last_logits = None
    
    def compute_confidence(
        self, 
        logits: torch.Tensor,
        sampled_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute confidence scores for sampled tokens.
        
        Confidence is defined as the probability mass of the sampled token
        in the softmax distribution.
        
        Args:
            logits: Logits tensor [num_positions, vocab_size]
            sampled_tokens: Sampled token IDs [num_positions]
            
        Returns:
            Confidence scores [num_positions] in range [0, 1]
        """
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Gather probabilities of sampled tokens
        confidence = probs.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        
        return confidence
    
    def sample_tokens(
        self, 
        logits: torch.Tensor,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample tokens from logits with top-k/top-p filtering.
        
        Aligned with Diffulex SamplerBase.sample_tokens:
        Returns (confidence, sampled_tokens, initial_confidence)
        
        Args:
            logits: Logits tensor [num_positions, vocab_size]
            temperature: Override for temperature (uses self.temperature if None)
            top_p: Override for top_p (uses self.top_p if None)
            top_k: Override for top_k (uses self.top_k if None)
            
        Returns:
            Tuple of (confidence, sampled_tokens, initial_confidence)
            - confidence: [num_positions] (may be modified by margin_confidence/neg_entropy)
            - sampled_tokens: [num_positions] token IDs
            - initial_confidence: [num_positions] original probabilities
        """
        batch_size, vocab_size = logits.shape
        device = logits.device
        
        # Use instance defaults if not provided
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-p filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative prob > top_p
            sorted_indices_to_remove = cumsum_probs > top_p
            # Shift to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            k = min(top_k, vocab_size)
            indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample or take argmax
        if temperature > 0:
            try:
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                initial_confidence = probs.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                # Fallback to argmax if multinomial fails
                initial_confidence, sampled_tokens = probs.max(dim=-1)
        else:
            # Greedy (temperature = 0)
            initial_confidence, sampled_tokens = probs.max(dim=-1)
        
        confidence = initial_confidence.clone()
        
        # Apply margin confidence if enabled
        if self.margin_confidence:
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            top1_probs = sorted_probs[:, 0]
            top2_probs = sorted_probs[:, 1]
            confidence = top1_probs - top2_probs
        
        # Apply negative entropy if enabled
        if self.neg_entropy:
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            confidence = -torch.sum(probs * log_probs, dim=-1)
        
        return confidence, sampled_tokens, initial_confidence
    
    def sample_blocks(
        self,
        block_manager: DiffusionBlockManager,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> SampleOutput:
        """Sample tokens for all active blocks.
        
        Aligned with Diffulex FastdLLMV2SamplerForDiffusionLM.forward:
        - Samples from mask positions in each block
        - Accepts tokens based on confidence threshold
        - Always accepts at least the highest confidence token
        
        Args:
            block_manager: Manager containing active blocks
            logits: Logits tensor [seq_len, vocab_size]
            temperature: Optional override for temperature
            top_p: Optional override for top_p
            top_k: Optional override for top_k
            threshold: Optional override for confidence_threshold
            
        Returns:
            SampleOutput with sampled tokens and acceptance info
        """
        output = SampleOutput()
        temp = temperature if temperature is not None else self.temperature
        threshold = threshold if threshold is not None else self.confidence_threshold
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        
        # Get active blocks
        active_blocks = block_manager.get_active_blocks()
        
        for block_id, block in active_blocks:
            # Skip inactive or empty blocks
            if not block.is_active:
                continue
            
            # Get mask positions for this block
            mask_positions = block.global_mask_token_ids
            
            if not mask_positions:
                continue
            
            # Extract logits for mask positions
            block_logits = logits[mask_positions]  # [num_mask, vocab_size]
            
            # Sample tokens (aligned with Diffulex)
            confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                block_logits,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
            )
            
            # Determine which tokens to accept based on confidence
            high_conf_indices = torch.where(initial_confidence > threshold)[0]
            
            # Always accept at least the highest confidence token (Diffulex behavior)
            if len(high_conf_indices) == 0:
                max_prob_idx = initial_confidence.argmax().view(1)
                accept_indices = max_prob_idx
            else:
                max_prob_idx = initial_confidence.argmax().view(1)
                accept_indices = torch.unique(torch.cat([high_conf_indices, max_prob_idx]))
            
            # Update output
            for i, pos in enumerate(mask_positions):
                output.sampled_tokens[pos] = sampled_tokens[i].item()
                output.confidence_scores[pos] = initial_confidence[i].item()
            
            # Record accepted tokens
            accept_list = accept_indices.cpu().tolist()
            for idx in accept_list:
                pos = mask_positions[idx]
                output.accepted_tokens[pos] = sampled_tokens[idx].item()
            
            # Update block with accepted tokens
            accepted_local = [
                pos - block.start_pos 
                for pos in output.accepted_tokens.keys()
                if block.start_pos <= pos < block.start_pos + block.length
                and pos in mask_positions
            ]
            accepted_token_ids = [
                output.accepted_tokens[block.start_pos + local_pos]
                for local_pos in accepted_local
            ]
            
            if accepted_local:
                block_manager.update_block(block_id, accepted_local, accepted_token_ids)
        
        # Record block states
        for block in block_manager.blocks:
            output.block_states.append({
                "is_active": block.is_active,
                "is_complete": block.is_complete,
                "num_accepted": len(block.accepted_token_ids),
                "num_masked": len(block.global_mask_token_ids),
            })
        
        return output


@dataclass
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
    """
    max_new_tokens: int = 100
    num_iterations: int = 10
    block_size: int = 10
    confidence_threshold: float = 0.95
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    mask_token_id: int = 126336
    eos_token_id: int = 2
    early_stop: bool = True


class DiffusionEngine:
    """Inference engine for diffusion-based generation.
    
    Supports iterative denoising with confidence-based token acceptance.
    Can run with either PyTorch models (for development) or ExecuTorch
    PTE models (for deployment).
    
    Usage:
        # Load PyTorch model
        engine = DiffusionEngine.from_model(model)
        
        # Or load ExecuTorch model
        engine = DiffusionEngine.from_pte("model.pte")
        
        # Generate text
        tokens = engine.generate(prompt_tokens, config)
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        pte_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ):
        """Initialize diffusion engine.
        
        Args:
            model: PyTorch model for inference
            pte_path: Path to .pte file (if loading ExecuTorch model)
            device: Device to run on
            
        Raises:
            ValueError: If both model and pte_path are provided, or neither
            FileNotFoundError: If pte_path is provided but file doesn't exist
            ImportError: If pte_path is provided but ExecuTorch not installed
        """
        # Validate exclusive parameters
        if model is not None and pte_path is not None:
            raise ValueError(
                "Cannot provide both 'model' and 'pte_path'. "
                "Use from_model() or from_pte() for clarity."
            )
        
        if model is None and pte_path is None:
            logger.warning(
                "Initializing DiffusionEngine without model. "
                "Call from_model() or from_pte() to load a model."
            )
        
        self.model = model
        self.pte_path = Path(pte_path) if pte_path is not None else None
        self.device = device
        self.block_manager = DiffusionBlockManager()
        self.sampler = DiffusionSampler()
        
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
        """Load ExecuTorch model from pte_path.
        
        Raises:
            FileNotFoundError: If PTE file doesn't exist
            ImportError: If ExecuTorch runtime not available
            RuntimeError: If loading fails for other reasons
        """
        if self.pte_path is None:
            raise RuntimeError("pte_path is None - cannot load PTE model")
        
        # Validate file exists
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
    
    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        device: str = "cpu",
    ) -> "DiffusionEngine":
        """Create engine from PyTorch model.
        
        Args:
            model: PyTorch model
            device: Device to run on
            
        Returns:
            Configured DiffusionEngine
        """
        model.eval()
        model.to(device)
        logger.debug(f"Creating DiffusionEngine from PyTorch model on {device}")
        return cls(model=model, device=device)
    
    @classmethod
    def from_pte(
        cls,
        pte_path: Union[str, Path],
        device: str = "cpu",
    ) -> "DiffusionEngine":
        """Create engine from .pte file.
        
        Args:
            pte_path: Path to .pte file
            device: Device to run on
            
        Returns:
            Configured DiffusionEngine
        """
        logger.debug(f"Creating DiffusionEngine from PTE: {pte_path}")
        return cls(pte_path=Path(pte_path), device=device)
    
    def generate(
        self,
        prompt_tokens: List[int],
        config: Optional[DiffusionGenerationConfig] = None,
    ) -> List[int]:
        """Generate tokens using diffusion sampling.
        
        Args:
            prompt_tokens: List of prompt token IDs
            config: Generation configuration
            
        Returns:
            Generated token IDs (including prompt)
        """
        config = config or DiffusionGenerationConfig()
        
        # Initialize sampler
        self.sampler = DiffusionSampler(
            mask_token_id=config.mask_token_id,
            confidence_threshold=config.confidence_threshold,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
        
        # Reset state
        self.block_manager.reset()
        self.sampler.reset_cache()
        
        # Initialize sequence with prompt
        sequence = list(prompt_tokens)
        prompt_len = len(prompt_tokens)
        
        # Create initial mask tokens
        num_blocks = (config.max_new_tokens + config.block_size - 1) // config.block_size
        
        for i in range(num_blocks):
            start_pos = prompt_len + i * config.block_size
            remaining = config.max_new_tokens - i * config.block_size
            block_len = min(config.block_size, remaining)
            
            self.block_manager.create_block(start_pos, block_len)
            
            # Add mask tokens to sequence
            sequence.extend([config.mask_token_id] * block_len)
        
        logger.debug(
            f"Starting generation: prompt_len={prompt_len}, "
            f"num_blocks={num_blocks}, max_new_tokens={config.max_new_tokens}"
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
            
            # Apply shift (with caching for multi-step)
            shifted_logits = self.sampler.shift_logits(logits, use_cache=True)
            
            # Sample blocks
            output = self.sampler.sample_blocks(self.block_manager, shifted_logits)
            
            # Update sequence with accepted tokens
            for pos, token_id in output.accepted_tokens.items():
                sequence[pos] = token_id
            
            num_accepted = len(output.accepted_tokens)
            logger.debug(f"Iteration {iteration}: accepted {num_accepted} tokens")
        
        logger.debug(f"Generation complete: sequence length={len(sequence)}")
        return sequence
    
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass through model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        if self._is_pte:
            return self._forward_pte(input_ids)
        else:
            return self._forward_torch(input_ids)
    
    def _forward_torch(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass using PyTorch model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        with torch.no_grad():
            return self.model(input_ids)[0]
    
    def _forward_pte(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass using ExecuTorch model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        if self._pte_module is None:
            raise RuntimeError("No PTE model loaded")
        
        # Convert to list for ExecuTorch
        input_list = input_ids.tolist()
        
        # Run inference
        result = self._pte_module.forward(input_list)
        
        # Convert result back to tensor
        # Handle different return formats
        if isinstance(result, (list, tuple)):
            logits = torch.tensor(result[0])
        else:
            logits = torch.tensor(result)
        
        # Ensure correct shape [batch, seq_len, vocab_size]
        if logits.dim() == 2:
            # [seq_len, vocab_size] -> [1, seq_len, vocab_size]
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1:
            # [vocab_size] -> [1, 1, vocab_size]
            logits = logits.unsqueeze(0).unsqueeze(0)
        
        return logits
    
    def generate_stream(
        self,
        prompt_tokens: List[int],
        config: Optional[DiffusionGenerationConfig] = None,
    ) -> Iterator[Tuple[int, int]]:
        """Generate tokens with streaming, yielding (position, token_id).
        
        Args:
            prompt_tokens: List of prompt token IDs
            config: Generation configuration
            
        Yields:
            Tuples of (position, token_id) for accepted tokens
        """
        config = config or DiffusionGenerationConfig()
        
        # Initialize
        self.sampler = DiffusionSampler(
            mask_token_id=config.mask_token_id,
            confidence_threshold=config.confidence_threshold,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
        self.block_manager.reset()
        
        sequence = list(prompt_tokens)
        prompt_len = len(prompt_tokens)
        
        # Create blocks
        num_blocks = (config.max_new_tokens + config.block_size - 1) // config.block_size
        for i in range(num_blocks):
            start_pos = prompt_len + i * config.block_size
            remaining = config.max_new_tokens - i * config.block_size
            block_len = min(config.block_size, remaining)
            self.block_manager.create_block(start_pos, block_len)
            sequence.extend([config.mask_token_id] * block_len)
        
        # Track already yielded positions
        yielded_positions = set()
        
        # Iterative denoising
        for iteration in range(config.num_iterations):
            if config.early_stop and not self.block_manager.has_active_blocks():
                break
            
            input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
            
            logits = self._forward(input_ids)[0]
            
            shifted_logits = self.sampler.shift_logits(logits)
            output = self.sampler.sample_blocks(self.block_manager, shifted_logits)
            
            # Update sequence and yield new accepts
            for pos, token_id in output.accepted_tokens.items():
                sequence[pos] = token_id
                if pos not in yielded_positions:
                    yielded_positions.add(pos)
                    yield (pos, token_id)
