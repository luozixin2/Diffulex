"""Simplified diffusion block management - block confirmation mechanism.

Align with autoregressive pattern:
- Autoregressive: generate 1 token → confirm → next
- Block Diffusion: generate N tokens → confirm all → next N

This is simpler than original diffulex's KV-cache approach but achieves
the same "block-by-block" generation semantics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class BlockStatus(Enum):
    """Block status in confirmation mechanism."""
    PENDING = "pending"      # Future block, all masks
    ACTIVE = "active"        # Current block being generated
    CONFIRMED = "confirmed"  # Block confirmed, tokens fixed


@dataclass
class DiffusionBlock:
    """Simplified block - just tracks tokens and confirmation status.
    
    Unlike original diffulex's complex state machine (ACTIVE/TO_CACHE/IN_CACHE),
    this uses a simple confirmation pattern similar to autoregressive generation.
    """
    block_id: int
    start_pos: int
    length: int
    status: BlockStatus = BlockStatus.PENDING
    tokens: Dict[int, int] = field(default_factory=dict)  # local_pos -> token_id
    mask_token_id: int = 126336
    accept_threshold: float = 0.95  # Per-block threshold for LLaDA/Dream
    
    @property
    def is_confirmed(self) -> bool:
        """Check if all positions have confirmed tokens."""
        return len(self.tokens) == self.length
    
    @property
    def is_active(self) -> bool:
        return self.status == BlockStatus.ACTIVE
    
    @property
    def confirmed_count(self) -> int:
        return len(self.tokens)
    
    def confirm_token(self, local_pos: int, token_id: int) -> bool:
        """Confirm a token at local position.
        
        Returns:
            True if this was the last token needed to confirm the block
        """
        if local_pos < 0 or local_pos >= self.length:
            raise IndexError(f"Local position {local_pos} out of range [0, {self.length})")
        
        if local_pos not in self.tokens:
            self.tokens[local_pos] = token_id
        
        return self.is_confirmed
    
    def get_global_mask_positions(self) -> List[int]:
        """Get global positions that are still masked (not confirmed)."""
        return [self.start_pos + i for i in range(self.length) if i not in self.tokens]
    
    def get_local_mask_positions(self) -> List[int]:
        """Get local positions that are still masked."""
        return [i for i in range(self.length) if i not in self.tokens]
    
    def get_sequence_tokens(self) -> List[int]:
        """Get full sequence for this block (confirmed tokens or mask)."""
        return [self.tokens.get(i, self.mask_token_id) for i in range(self.length)]
    
    def activate(self) -> None:
        """Activate this block for generation."""
        self.status = BlockStatus.ACTIVE
    
    def confirm(self) -> None:
        """Mark block as confirmed (all tokens fixed)."""
        if not self.is_confirmed:
            raise ValueError("Cannot confirm block with unconfirmed tokens")
        self.status = BlockStatus.CONFIRMED


class DiffusionBlockManager:
    """Manages blocks with confirmation mechanism.
    
    Similar to autoregressive's "generated tokens" list, but for blocks.
    """
    
    def __init__(self, mask_token_id: int = 126336):
        self.mask_token_id = mask_token_id
        self.blocks: List[DiffusionBlock] = []
        self._current_block_idx: int = 0
    
    def create_blocks(self, prompt_len: int, max_new_tokens: int, block_size: int, 
                      accept_threshold: float = 0.95) -> None:
        """Create all blocks for generation.
        
        Args:
            prompt_len: Length of prompt (for calculating start positions)
            max_new_tokens: Maximum tokens to generate
            block_size: Size of each block
            accept_threshold: Per-block threshold (for LLaDA/Dream)
        """
        self.blocks.clear()
        self._current_block_idx = 0
        
        num_blocks = (max_new_tokens + block_size - 1) // block_size
        
        for i in range(num_blocks):
            start_pos = prompt_len + i * block_size
            remaining = max_new_tokens - i * block_size
            length = min(block_size, remaining)
            
            block = DiffusionBlock(
                block_id=i,
                start_pos=start_pos,
                length=length,
                status=BlockStatus.PENDING,
                mask_token_id=self.mask_token_id,
                accept_threshold=accept_threshold,
            )
            self.blocks.append(block)
        
        # Activate first block
        if self.blocks:
            self.blocks[0].activate()
    
    def get_active_block(self) -> Optional[DiffusionBlock]:
        """Get the currently active block."""
        if 0 <= self._current_block_idx < len(self.blocks):
            block = self.blocks[self._current_block_idx]
            if block.is_active:
                return block
        return None
    
    def confirm_current_block(self) -> bool:
        """Confirm current block and advance to next.
        
        Returns:
            True if there are more blocks to generate
        """
        if not (0 <= self._current_block_idx < len(self.blocks)):
            return False
        
        current = self.blocks[self._current_block_idx]
        if not current.is_confirmed:
            return True  # Not ready to advance
        
        current.confirm()
        self._current_block_idx += 1
        
        # Activate next block
        if self._current_block_idx < len(self.blocks):
            self.blocks[self._current_block_idx].activate()
            return True
        return False  # No more blocks
    
    def has_active_block(self) -> bool:
        """Check if there's an active block."""
        return self.get_active_block() is not None
    
    def get_confirmed_token_count(self) -> int:
        """Get total number of confirmed tokens across all blocks."""
        return sum(len(block.tokens) for block in self.blocks)
    
    def build_sequence(self, prompt_tokens: List[int]) -> List[int]:
        """Build full sequence: prompt + confirmed tokens + active masks + future masks.
        
        This is the key method - it constructs what the model sees.
        """
        sequence = list(prompt_tokens)
        
        for block in self.blocks:
            if block.status == BlockStatus.CONFIRMED:
                # Use confirmed tokens
                sequence.extend(block.get_sequence_tokens())
            elif block.status == BlockStatus.ACTIVE:
                # Active block: use masks (will be sampled)
                sequence.extend([self.mask_token_id] * block.length)
            else:
                # Pending blocks: masks
                sequence.extend([self.mask_token_id] * block.length)
        
        return sequence
    
    def get_confirmed_tokens_list(self) -> List[int]:
        """Get all confirmed tokens as a flat list."""
        tokens = []
        for block in self.blocks:
            if block.status == BlockStatus.CONFIRMED:
                tokens.extend(block.get_sequence_tokens())
        return tokens
    
    def reset(self) -> None:
        """Reset all blocks."""
        self.blocks.clear()
        self._current_block_idx = 0
    
    def __len__(self) -> int:
        return len(self.blocks)
