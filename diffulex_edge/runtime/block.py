"""Diffusion block management for dLLM models.

Aligned with diffulex block concepts including:
- Per-block accept_threshold (for LLaDA/Dream)
- pre_block_complete tracking (for LLaDA/Dream)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DiffusionBlock:
    """
    Represents a block of tokens in diffusion generation.
    
    Aligned with diffulex block concepts:
    - Tracks mask positions and accepted tokens
    - Supports per-block accept_threshold
    - Tracks pre_block_complete status
    
    Attributes:
        start_pos: Start position of this block in the sequence
        length: Length of the block
        mask_token_id: Token ID used for masking
        is_active: Whether this block is still being processed
        accepted_token_ids: Map from local position to accepted token ID
        accept_threshold: Per-block confidence threshold (LLaDA/Dream)
        pre_block_complete: Whether previous block is complete (LLaDA/Dream)
        local_mask_tokens: Local mask status for each position
    """
    start_pos: int
    length: int
    mask_token_id: int = 126336
    is_active: bool = True
    accepted_token_ids: Dict[int, int] = field(default_factory=dict)
    accept_threshold: float = 0.95
    pre_block_complete: bool = False
    
    @property
    def global_mask_token_ids(self) -> List[int]:
        """Get global positions of tokens that are still masked.
        
        Align with: block.global_mask_token_ids in diffulex
        
        Returns:
            List of global positions that haven't been accepted yet
        """
        return [
            self.start_pos + i 
            for i in range(self.length) 
            if i not in self.accepted_token_ids
        ]
    
    @property
    def local_mask_tokens(self) -> List[bool]:
        """Get local mask status for each position.
        
        Returns:
            List of bool indicating whether each local position is masked
        """
        return [i not in self.accepted_token_ids for i in range(self.length)]
    
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
        """Convert local position to global position.
        
        Args:
            local_pos: Position within block (0 to length-1)
            
        Returns:
            Global position in sequence
        """
        if local_pos < 0 or local_pos >= self.length:
            raise IndexError(
                f"Local position {local_pos} out of range [0, {self.length})"
            )
        return self.start_pos + local_pos


class DiffusionBlockManager:
    """Manages multiple diffusion blocks during generation.
    
    Handles creation, tracking, and lifecycle of diffusion blocks.
    Also manages pre_block_complete status across blocks.
    """
    
    def __init__(self, mask_token_id: int = 126336):
        """Initialize the block manager.
        
        Args:
            mask_token_id: Token ID used for masking
        """
        self.mask_token_id = mask_token_id
        self.blocks: List[DiffusionBlock] = []
        self._block_id_counter = 0
    
    def create_block(
        self,
        start_pos: int,
        length: int,
        accept_threshold: float = 0.95,
    ) -> int:
        """Create a new diffusion block.
        
        Args:
            start_pos: Start position in the sequence
            length: Length of the block
            accept_threshold: Per-block confidence threshold
            
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
            accept_threshold=accept_threshold,
        )
        
        block_id = self._block_id_counter
        self.blocks.append(block)
        self._block_id_counter += 1
        
        # Update pre_block_complete for this block
        self._update_pre_block_complete()
        
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
        
        # Update pre_block_complete after block update
        self._update_pre_block_complete()
    
    def _update_pre_block_complete(self) -> None:
        """Update pre_block_complete status for all blocks.
        
        Sets pre_block_complete=True for a block if all previous blocks
        are complete. This is used by LLaDA and Dream samplers.
        """
        prev_complete = True
        for block in self.blocks:
            block.pre_block_complete = prev_complete
            prev_complete = block.is_complete
    
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