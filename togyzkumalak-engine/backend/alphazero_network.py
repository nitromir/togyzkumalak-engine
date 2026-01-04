"""
AlphaZero Network for Togyzkumalak

This module provides the neural network components for AlphaZero.
Uses the self-contained alphazero_trainer module for CPU-optimized training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .game_manager import TogyzkumalakBoard
from .alphazero_trainer import (
    AlphaZeroNetwork,
    NNetWrapper,
    TogyzkumalakGame,
    AlphaZeroConfig,
    device
)


# Re-export AlphaZeroNet for backward compatibility
class AlphaZeroNet(AlphaZeroNetwork):
    """
    Dual-head Neural Network for AlphaZero.
    Policy head: move probabilities.
    Value head: position evaluation [-1, 1].
    
    This is an alias for AlphaZeroNetwork from alphazero_trainer.
    """
    pass


class TogyzkumalakAlphaZeroNet(NNetWrapper):
    """
    Wrapper for AlphaZeroNet to fit the NeuralNet interface.
    
    Uses CPU-only PyTorch for training on machines without GPU.
    """
    
    def __init__(self, game=None, config: AlphaZeroConfig = None):
        # Create default game if not provided
        if game is None:
            game = TogyzkumalakGame()
        
        # Create default config if not provided
        if config is None:
            config = AlphaZeroConfig()
        
        super().__init__(game, config)
                
        # Force CPU device
        self.device = device
        print(f"[AlphaZero Network] Using device: {self.device}")
    
    def get_observation(self, board):
        """Helper to convert raw fields to NN input observation (128-dim)."""
        return self.game.boardToObservation(board)
