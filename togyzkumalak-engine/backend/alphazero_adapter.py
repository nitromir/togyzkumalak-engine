"""
AlphaZero Adapter for Togyzkumalak

This module provides the game interface adapter for AlphaZero training.
Uses the self-contained alphazero_trainer module for CPU-optimized training.
"""

import numpy as np
from .game_manager import TogyzkumalakBoard
from .alphazero_trainer import TogyzkumalakGame as AlphaZeroGameBase

class TogyzkumalakAlphaZeroGame(AlphaZeroGameBase):
    """
    Adapter class for Togyzkumalak to fit the alpha-zero-general interface.
    
    Extends the base TogyzkumalakGame from alphazero_trainer with additional
    helper methods for integration with the game_manager.
    """
    
    def __init__(self):
        super().__init__()
        self.board_logic = TogyzkumalakBoard()

    def get_observation(self, board):
        """Helper to convert raw fields to NN input observation (128-dim)."""
        if isinstance(board, np.ndarray):
            temp_board = TogyzkumalakBoard(board.tolist())
        else:
            temp_board = TogyzkumalakBoard(list(board))
        return temp_board.to_observation()
    
    def from_game_manager_board(self, gm_board: TogyzkumalakBoard) -> np.ndarray:
        """Convert a game_manager.TogyzkumalakBoard to AlphaZero board format."""
        return np.array(gm_board.fields, dtype=np.float32)
    
    def to_game_manager_board(self, az_board: np.ndarray) -> TogyzkumalakBoard:
        """Convert AlphaZero board format to game_manager.TogyzkumalakBoard."""
        return TogyzkumalakBoard(az_board.tolist())
