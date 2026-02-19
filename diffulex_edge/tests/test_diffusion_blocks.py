"""Tests for DiffusionBlock and DiffusionBlockManager.

Test Plan Coverage:
- DiffusionBlock basic functionality: 6 tests
- DiffusionBlockManager functionality: 10 tests  
- Boundary conditions: 4 tests
- Error handling: 2 tests

Total: 22 tests
"""

import pytest
import torch
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runtime.diffusion import DiffusionBlock, DiffusionBlockManager


# ============================================================================
# DiffusionBlock Basic Functionality Tests (6 tests)
# ============================================================================

class TestDiffusionBlockBasic:
    """Test basic DiffusionBlock functionality."""
    
    def test_block_initialization(self):
        """TEST-BLOCK-001: Verify block initializes with correct attributes."""
        block = DiffusionBlock(start_pos=10, length=5, mask_token_id=100)
        
        assert block.start_pos == 10
        assert block.length == 5
        assert block.mask_token_id == 100
        assert block.is_active is True
        assert len(block.accepted_token_ids) == 0
        assert block.is_complete is False
    
    def test_default_mask_token(self):
        """TEST-BLOCK-002: Verify default mask token ID (126336)."""
        block = DiffusionBlock(start_pos=0, length=10)
        
        # Default mask token ID should be 126336 (FastdLLM default)
        assert block.mask_token_id == 126336
    
    def test_accept_token_updates_state(self):
        """TEST-BLOCK-003: Verify accept_token updates block state."""
        block = DiffusionBlock(start_pos=10, length=5)
        
        # Accept a token
        block.accept_token(0, 100)
        
        assert block.accepted_token_ids[0] == 100
        assert len(block.accepted_token_ids) == 1
        assert 0 not in block.global_mask_token_ids
    
    def test_global_mask_positions_initial(self):
        """TEST-BLOCK-004: Verify global_mask_token_ids returns all positions initially."""
        block = DiffusionBlock(start_pos=10, length=5)
        
        expected = [10, 11, 12, 13, 14]
        assert block.global_mask_token_ids == expected
    
    def test_global_mask_positions_after_accept(self):
        """TEST-BLOCK-005: Verify global_mask_token_ids excludes accepted positions."""
        block = DiffusionBlock(start_pos=10, length=5)
        
        # Accept position 1 and 3
        block.accept_token(1, 101)
        block.accept_token(3, 103)
        
        expected = [10, 12, 14]  # 11 and 13 are accepted
        assert block.global_mask_token_ids == expected
    
    def test_block_complete_when_all_accepted(self):
        """TEST-BLOCK-006: Verify block becomes inactive when all tokens accepted."""
        block = DiffusionBlock(start_pos=0, length=3)
        
        # Accept all tokens
        block.accept_token(0, 100)
        block.accept_token(1, 101)
        block.accept_token(2, 102)
        
        assert block.is_complete is True
        assert block.is_active is False


# ============================================================================
# DiffusionBlockManager Functionality Tests (10 tests)
# ============================================================================

class TestDiffusionBlockManager:
    """Test DiffusionBlockManager functionality."""
    
    def test_manager_initialization(self):
        """TEST-BLOCK-007: Verify manager initializes correctly."""
        manager = DiffusionBlockManager(mask_token_id=100)
        
        assert manager.mask_token_id == 100
        assert len(manager.blocks) == 0
        assert manager._block_id_counter == 0
    
    def test_create_block_returns_incrementing_ids(self):
        """TEST-BLOCK-008: Verify create_block returns incrementing block IDs."""
        manager = DiffusionBlockManager()
        
        id1 = manager.create_block(start_pos=0, length=10)
        id2 = manager.create_block(start_pos=10, length=10)
        id3 = manager.create_block(start_pos=20, length=10)
        
        assert id1 == 0
        assert id2 == 1
        assert id3 == 2
        assert len(manager.blocks) == 3
    
    def test_get_block_returns_correct_block(self):
        """TEST-BLOCK-009: Verify get_block returns correct block by ID."""
        manager = DiffusionBlockManager()
        
        block_id = manager.create_block(start_pos=10, length=5)
        block = manager.get_block(block_id)
        
        assert block is not None
        assert block.start_pos == 10
        assert block.length == 5
    
    def test_get_block_returns_none_for_invalid_id(self):
        """TEST-BLOCK-010: Verify get_block returns None for invalid block_id."""
        manager = DiffusionBlockManager()
        
        # No blocks created yet
        assert manager.get_block(0) is None
        assert manager.get_block(-1) is None
        
        # Create one block
        manager.create_block(start_pos=0, length=10)
        assert manager.get_block(1) is None  # Out of range
        assert manager.get_block(100) is None
    
    def test_get_active_blocks_returns_only_active(self):
        """TEST-BLOCK-011: Verify get_active_blocks returns only active blocks."""
        manager = DiffusionBlockManager()
        
        # Create blocks
        id1 = manager.create_block(start_pos=0, length=3)
        id2 = manager.create_block(start_pos=3, length=3)
        id3 = manager.create_block(start_pos=6, length=3)
        
        # Complete block 1
        for i in range(3):
            manager.blocks[id2].accept_token(i, 100 + i)
        
        active = manager.get_active_blocks()
        active_ids = [bid for bid, _ in active]
        
        assert id1 in active_ids
        assert id2 not in active_ids  # Completed
        assert id3 in active_ids
        assert len(active) == 2
    
    def test_update_block_accepts_multiple_tokens(self):
        """TEST-BLOCK-012: Verify update_block accepts multiple tokens."""
        manager = DiffusionBlockManager()
        
        block_id = manager.create_block(start_pos=0, length=5)
        
        manager.update_block(
            block_id,
            accepted_positions=[0, 2, 4],
            accepted_tokens=[100, 102, 104]
        )
        
        block = manager.blocks[block_id]
        assert block.accepted_token_ids[0] == 100
        assert block.accepted_token_ids[2] == 102
        assert block.accepted_token_ids[4] == 104
        assert len(block.accepted_token_ids) == 3
    
    def test_has_active_blocks_true_when_active(self):
        """TEST-BLOCK-013: Verify has_active_blocks returns True when blocks active."""
        manager = DiffusionBlockManager()
        
        manager.create_block(start_pos=0, length=10)
        assert manager.has_active_blocks() is True
    
    def test_has_active_blocks_false_when_all_complete(self):
        """TEST-BLOCK-014: Verify has_active_blocks returns False when all complete."""
        manager = DiffusionBlockManager()
        
        block_id = manager.create_block(start_pos=0, length=2)
        
        # Complete the block
        manager.update_block(block_id, [0, 1], [100, 101])
        
        assert manager.has_active_blocks() is False
    
    def test_get_all_mask_positions_aggregates_across_blocks(self):
        """TEST-BLOCK-015: Verify get_all_mask_positions aggregates across blocks."""
        manager = DiffusionBlockManager()
        
        # Block 1: positions 0-2 (accept position 1)
        id1 = manager.create_block(start_pos=0, length=3)
        manager.blocks[id1].accept_token(1, 101)
        
        # Block 2: positions 3-5 (all masked)
        manager.create_block(start_pos=3, length=3)
        
        mask_positions = manager.get_all_mask_positions()
        
        # Should be [0, 2] from block 1, [3, 4, 5] from block 2
        expected = [0, 2, 3, 4, 5]
        assert mask_positions == expected
    
    def test_reset_clears_all_blocks(self):
        """TEST-BLOCK-016: Verify reset clears all blocks."""
        manager = DiffusionBlockManager()
        
        manager.create_block(start_pos=0, length=10)
        manager.create_block(start_pos=10, length=10)
        
        assert len(manager.blocks) == 2
        
        manager.reset()
        
        assert len(manager.blocks) == 0
        assert manager._block_id_counter == 0


# ============================================================================
# Boundary Condition Tests (4 tests)
# ============================================================================

class TestDiffusionBlockBoundaryConditions:
    """Test boundary conditions for DiffusionBlock and Manager."""
    
    def test_block_single_position(self):
        """TEST-BLOCK-BOUND-001: Verify block with single position (length=1)."""
        block = DiffusionBlock(start_pos=0, length=1)
        
        assert block.length == 1
        assert block.global_mask_token_ids == [0]
        
        # Accept the only position
        block.accept_token(0, 999)
        assert block.is_complete is True
        assert block.global_mask_token_ids == []
    
    def test_manager_empty_blocks_list(self):
        """TEST-BLOCK-BOUND-002: Verify manager with no blocks."""
        manager = DiffusionBlockManager()
        
        assert len(manager) == 0
        assert manager.has_active_blocks() is False
        assert manager.get_active_blocks() == []
        assert manager.get_all_mask_positions() == []
    
    def test_block_zero_start_position(self):
        """TEST-BLOCK-BOUND-003: Verify block with start_pos=0."""
        block = DiffusionBlock(start_pos=0, length=5)
        
        assert block.start_pos == 0
        assert block.global_mask_token_ids == [0, 1, 2, 3, 4]
        assert block.get_global_pos(0) == 0
        assert block.get_global_pos(4) == 4
    
    def test_large_block_length(self):
        """TEST-BLOCK-BOUND-004: Verify block handles large length."""
        large_length = 10000
        block = DiffusionBlock(start_pos=0, length=large_length)
        
        assert block.length == large_length
        assert len(block.global_mask_token_ids) == large_length
        
        # Accept a token near the end
        block.accept_token(9999, 100)
        assert 9999 not in block.global_mask_token_ids


# ============================================================================
# Error Handling Tests (2 tests)
# ============================================================================

class TestDiffusionBlockErrorHandling:
    """Test error handling in DiffusionBlock and Manager."""
    
    def test_accept_token_invalid_position_raises(self):
        """TEST-BLOCK-ERR-001: Verify accept_token raises for invalid position."""
        block = DiffusionBlock(start_pos=0, length=5)
        
        # Negative position
        with pytest.raises(IndexError):
            block.accept_token(-1, 100)
        
        # Position >= length
        with pytest.raises(IndexError):
            block.accept_token(5, 100)
        
        with pytest.raises(IndexError):
            block.accept_token(100, 100)
    
    def test_accept_duplicate_token_raises(self):
        """TEST-BLOCK-ERR-002: Verify accept_token raises for duplicate acceptance."""
        block = DiffusionBlock(start_pos=0, length=5)
        
        block.accept_token(2, 100)
        
        # Try to accept same position again
        with pytest.raises(ValueError):
            block.accept_token(2, 200)


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestDiffusionBlockEdgeCases:
    """Additional edge case tests."""
    
    def test_block_get_global_pos_boundary(self):
        """Test get_global_pos at boundary positions."""
        block = DiffusionBlock(start_pos=10, length=5)
        
        assert block.get_global_pos(0) == 10
        assert block.get_global_pos(4) == 14
    
    def test_block_get_global_pos_invalid(self):
        """Test get_global_pos with invalid positions."""
        block = DiffusionBlock(start_pos=10, length=5)
        
        with pytest.raises(IndexError):
            block.get_global_pos(-1)
        
        with pytest.raises(IndexError):
            block.get_global_pos(5)
    
    def test_manager_update_invalid_block_id(self):
        """Test update_block with invalid block_id raises IndexError."""
        manager = DiffusionBlockManager()
        
        with pytest.raises(IndexError):
            manager.update_block(0, [0], [100])
        
        # Create one block
        manager.create_block(start_pos=0, length=10)
        
        with pytest.raises(IndexError):
            manager.update_block(1, [0], [100])
        
        with pytest.raises(IndexError):
            manager.update_block(-1, [0], [100])
    
    def test_manager_update_mismatched_lengths(self):
        """Test update_block with mismatched positions and tokens."""
        manager = DiffusionBlockManager()
        block_id = manager.create_block(start_pos=0, length=10)
        
        with pytest.raises(ValueError):
            manager.update_block(block_id, [0, 1, 2], [100, 101])  # 3 positions, 2 tokens
        
        with pytest.raises(ValueError):
            manager.update_block(block_id, [0, 1], [100, 101, 102])  # 2 positions, 3 tokens
    
    def test_manager_create_block_zero_length(self):
        """Test create_block with zero length raises ValueError."""
        manager = DiffusionBlockManager()
        
        with pytest.raises(ValueError):
            manager.create_block(start_pos=0, length=0)
        
        with pytest.raises(ValueError):
            manager.create_block(start_pos=0, length=-1)
    
    def test_manager_create_block_negative_start(self):
        """Test create_block with negative start position."""
        manager = DiffusionBlockManager()
        
        with pytest.raises(ValueError):
            manager.create_block(start_pos=-1, length=10)
