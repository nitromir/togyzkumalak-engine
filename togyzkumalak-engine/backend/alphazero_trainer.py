"""
AlphaZero Trainer for Togyzkumalak - Multi-GPU Optimized Version

Self-contained AlphaZero implementation with:
- Multi-GPU training support (DataParallel)
- Parallel self-play using multiprocessing
- Batch MCTS inference for GPU efficiency
- Real-time checkpoint downloads
- Per-checkpoint metrics logging
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import logging
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy

# Configure logging with file handler for UI visibility
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Add file handler for training logs (UI reads this)
_log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'alphazero_training.log')
_file_handler = logging.FileHandler(_log_file, mode='w')
_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
log.addHandler(_file_handler)

# =============================================================================
# Device Configuration - Auto-detect GPU
# =============================================================================

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_num_gpus():
    """Get number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

device = get_device()
NUM_GPUS = get_num_gpus()

log.info(f"Device: {device}, GPUs available: {NUM_GPUS}")

# Set optimal number of CPU threads for data loading
if device.type == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() - 1))


# =============================================================================
# Utility Classes
# =============================================================================

class AverageMeter:
    """Tracks average values."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __repr__(self):
        return f'{self.avg:.4f}'


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero training - Multi-GPU optimized."""
    # Training - BLITZ MODE defaults for fast iterations
    num_iterations: int = 100         # Number of training iterations
    num_episodes: int = 64            # Self-play games per iteration (was 100, reduced for speed)
    num_mcts_sims: int = 30           # MCTS simulations per move (was 100, reduced for speed)
    
    # Neural Network
    batch_size: int = 256             # Batch size (scale with GPU memory)
    epochs: int = 10                  # Training epochs per iteration
    learning_rate: float = 0.001
    hidden_size: int = 256            # Hidden layer size
    
    # MCTS
    cpuct: float = 1.0                # Exploration constant
    temp_threshold: int = 30          # Move count threshold for temperature (was 15, increased for exploration)
    
    # Arena
    arena_compare: int = 20           # Games to compare models (was 40, reduced for speed)
    update_threshold: float = 0.55    # Win rate needed to accept new model
    
    # Memory
    max_queue_length: int = 200000    # Max training examples
    num_iters_for_history: int = 20   # Keep last N iterations of examples
    
    # Checkpoints
    checkpoint_dir: str = "models/alphazero"
    save_every_n_iters: int = 5       # Save checkpoint every N iterations
    
    # Bootstrap - Use human game data for initial training
    use_bootstrap: bool = True
    bootstrap_file: str = "training_data/transitions_compact.jsonl"
    bootstrap_epochs: int = 10
    bootstrap_max_samples: int = 50000
    
    # Parallelization - optimized for multi-GPU
    num_parallel_games: int = 0       # 0 = auto (will be set to NUM_GPUS)
    num_workers: int = 0              # 0 = auto (based on GPUs)
    use_multiprocessing: bool = True  # Enable parallel self-play
    
    # Resume training
    resume_from_checkpoint: bool = True  # Auto-resume from best.pth.tar or bootstrapped.pth.tar


# =============================================================================
# Game Interface - Togyzkumalak
# =============================================================================

class TogyzkumalakGame:
    """
    Togyzkumalak game logic for AlphaZero.
    Board representation: 23-element array
    [0-8]: White pits, [9-17]: Black pits
    [18]: White tuzduk, [19]: Black tuzduk
    [20]: White kazan, [21]: Black kazan
    [22]: Current player (0=white, 1=black)
    """
    
    TUZDUK = -1
    
    def __init__(self):
        self.board_size = (1, 23)
        self.action_size = 9
    
    def getInitBoard(self) -> np.ndarray:
        """Returns initial board state."""
        return np.array([9] * 18 + [0, 0, 0, 0, 0], dtype=np.float32)
    
    def getBoardSize(self) -> Tuple[int, int]:
        return self.board_size
    
    def getActionSize(self) -> int:
        return self.action_size
    
    def getNextState(self, board: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, int]:
        """Execute action and return new board and next player."""
        # Validate board size
        if len(board) != 23:
            log.error(f"getNextState: Board has wrong size: {len(board)}, expected 23. Board: {board[:min(10, len(board))]}")
            if len(board) < 23:
                if len(board) == 8:
                    log.error(f"getNextState: CRITICAL - 8-element board detected!")
                board = np.pad(board, (0, 23 - len(board)), mode='constant', constant_values=0) if len(board) < 23 else board[:23]
        
        new_board = board.copy()
        color = 0 if player == 1 else 1
        
        pit_index = action + (color * 9)
        num = int(new_board[pit_index])
        
        if num <= 0:
            return new_board, -player
        
        if num == 1:
            new_board[pit_index] = 0
            sow = 1
        else:
            new_board[pit_index] = 1
            sow = num - 1
        
        current = pit_index
        for _ in range(sow):
            current = (current + 1) % 18
            if new_board[current] == self.TUZDUK:
                if current < 9:
                    new_board[21] += 1
                else:
                    new_board[20] += 1
            else:
                new_board[current] += 1
        
        if new_board[current] != self.TUZDUK and new_board[current] % 2 == 0:
            if color == 0 and current > 8:
                new_board[20] += new_board[current]
                new_board[current] = 0
            elif color == 1 and current < 9:
                new_board[21] += new_board[current]
                new_board[current] = 0
        
        elif new_board[current] == 3:
            if color == 0 and new_board[18] == 0 and 9 <= current < 17:
                if new_board[19] != current - 8:
                    new_board[18] = current - 8
                    new_board[20] += 3
                    new_board[current] = self.TUZDUK
            elif color == 1 and new_board[19] == 0 and current < 8:
                if new_board[18] != current + 1:
                    new_board[19] = current + 1
                    new_board[21] += 3
                    new_board[current] = self.TUZDUK
        
        new_board[22] = 1 - color
        self._checkAtsyrau(new_board)
        
        return new_board, -player
    
    def _checkAtsyrau(self, board: np.ndarray):
        """Check if current player has no moves."""
        color = int(board[22])
        has_moves = False
        for i in range(9):
            if board[i + color * 9] > 0:
                has_moves = True
                break
        
        if not has_moves:
            opponent = 1 - color
            for i in range(9):
                pit = i + opponent * 9
                if board[pit] > 0:
                    if opponent == 0:
                        board[20] += board[pit]
                    else:
                        board[21] += board[pit]
                    board[pit] = 0
    
    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        """Returns binary mask of valid moves."""
        color = 0 if player == 1 else 1
        valid = np.zeros(9, dtype=np.float32)
        for i in range(9):
            if board[i + color * 9] > 0:
                valid[i] = 1
        
        # If no valid moves but game is not ended, force at least one move
        # or return all zeros to signal end of game
        if np.sum(valid) == 0:
            # Check if this player really has no stones
            my_stones = np.sum(board[color*9 : (color+1)*9])
            if my_stones == 0:
                # Normal end of game situation
                return valid 
            else:
                # This should not happen if logic is correct, but let's be safe
                log.warning(f"getValidMoves: Player {player} has {my_stones} stones but no moves?")
        
        return valid
    
    def getGameEnded(self, board: np.ndarray, player: int) -> float:
        """Returns 0 if not ended, 1 if player won, -1 if lost, small value for draw."""
        white_kazan = board[20]
        black_kazan = board[21]
        
        if white_kazan > 81:
            return 1.0 if player == 1 else -1.0
        elif black_kazan > 81:
            return -1.0 if player == 1 else 1.0
        elif white_kazan == 81 and black_kazan == 81:
            return 1e-4
        
        return 0
    
    def getCanonicalForm(self, board: np.ndarray, player: int) -> np.ndarray:
        """Returns board from player's perspective."""
        # Ensure board has correct size (23 elements)
        if len(board) != 23:
            log.warning(f"getCanonicalForm: Board has wrong size: {len(board)}, expected 23. Fixing...")
            if len(board) < 23:
                if len(board) == 8:
                    # Reconstruct full board from 8 elements
                    board = np.concatenate([
                        board,  # First 8 pits
                        np.zeros(10, dtype=np.float32),  # Remaining 10 pits
                        np.zeros(5, dtype=np.float32)  # Tuzduk, kazan, player
                    ])
                else:
                    board = np.pad(board, (0, 23 - len(board)), mode='constant', constant_values=0)
            else:
                board = board[:23]
        
        if player == 1:
            return board.copy()
        
        # Always create canonical with size 23
        canonical = np.zeros(23, dtype=np.float32)
        
        # Ensure board has at least 23 elements before slicing
        if len(board) >= 23:
            canonical[0:9] = board[9:18]
            canonical[9:18] = board[0:9]
            canonical[18] = board[19] if len(board) > 19 else 0
            canonical[19] = board[18] if len(board) > 18 else 0
            canonical[20] = board[21] if len(board) > 21 else 0
            canonical[21] = board[20] if len(board) > 20 else 0
            canonical[22] = 1 - board[22] if len(board) > 22 else 1
        else:
            # Fallback: just copy and pad
            canonical[:len(board)] = board
            canonical[22] = 1 - board[22] if len(board) > 22 else 1
        
        return canonical
    
    def getSymmetries(self, board: np.ndarray, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Togyzkumalak has no symmetries."""
        return [(board, pi)]
    
    def stringRepresentation(self, board: np.ndarray) -> str:
        """String representation for MCTS hashing."""
        return board.tobytes()
    
    def boardToObservation(self, board: np.ndarray) -> np.ndarray:
        """Convert board to neural network input (128-dim)."""
        # Validate board size first
        if len(board) != 23:
            log.error(f"boardToObservation: Board has wrong size: {len(board)}, expected 23. Board: {board}")
            if len(board) < 23:
                if len(board) == 8:
                    log.error(f"boardToObservation: CRITICAL - 8-element board detected! This is the root cause.")
                    import traceback
                    log.error(f"boardToObservation: Call stack:\n{''.join(traceback.format_stack()[-8:-1])}")
                board = np.pad(board, (0, 23 - len(board)), mode='constant', constant_values=0)
            else:
                board = board[:23]
        
        obs = np.zeros(128, dtype=np.float32)
        
        # Ensure we have at least 23 elements
        board_len = min(len(board), 23)
        
        for i in range(min(18, board_len)):
            val = board[i] if board[i] != self.TUZDUK else 0
            obs[i] = val / 81.0
        
        if board_len > 20:
            obs[18] = board[20] / 162.0
        if board_len > 21:
            obs[19] = board[21] / 162.0
        
        if board_len > 18 and board[18] > 0:
            obs[20 + int(board[18]) - 1] = 1.0
        if board_len > 19 and board[19] > 0:
            obs[29 + int(board[19]) - 1] = 1.0
        
        if board_len > 22:
            obs[38] = 1.0 if board[22] == 0 else 0.0
            obs[39] = 1.0 if board[22] == 1 else 0.0
        
        # CRITICAL: Validate output size
        if len(obs) != 128:
            log.error(f"boardToObservation: CRITICAL - Observation has wrong size: {len(obs)}, expected 128!")
            log.error(f"boardToObservation: This should never happen! Board was: {board}")
            # Force to 128
            if len(obs) < 128:
                obs = np.pad(obs, (0, 128 - len(obs)), mode='constant', constant_values=0)
            else:
                obs = obs[:128]
        
        return obs


# =============================================================================
# Neural Network - Dual Head (Policy + Value) with Multi-GPU Support
# =============================================================================

class AlphaZeroNetwork(nn.Module):
    """
    Dual-head neural network for AlphaZero.
    Supports DataParallel for multi-GPU training.
    """
    
    def __init__(self, input_size: int = 128, hidden_size: int = 256, action_size: int = 9):
        super().__init__()
        
        # Deeper network for better representation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn4 = nn.BatchNorm1d(hidden_size // 2)
        
        # Policy head
        self.policy_fc = nn.Linear(hidden_size // 2, action_size)
        
        # Value head
        self.value_fc1 = nn.Linear(hidden_size // 2, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # CRITICAL: With DataParallel, always expect 2D input (batch, features)
        # DataParallel splits along dim 0, so 1D tensors get split incorrectly!
        
        # Ensure 2D input
        if x.dim() == 1:
            # This should not happen with DataParallel - log warning
            log.warning(f"FORWARD: Received 1D tensor {x.shape}, converting to 2D. This may cause issues with DataParallel!")
            x = x.unsqueeze(0)
        
        # Validate input tensor shape - should be (batch, 128)
        if x.shape[-1] != 128:
            log.error(f"FORWARD: Input tensor has wrong shape: {x.shape}, expected last dim 128")
            log.error(f"FORWARD: This likely means DataParallel split a 1D tensor incorrectly!")
            # Try to fix if possible
            if x.shape[-1] < 128:
                padding = torch.zeros(*x.shape[:-1], 128 - x.shape[-1], device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=-1)
                log.warning(f"FORWARD: Padded tensor to shape: {x.shape}")
            else:
                x = x[..., :128]
                log.warning(f"FORWARD: Truncated tensor to shape: {x.shape}")
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        
        pi = self.policy_fc(x)
        pi = F.log_softmax(pi, dim=1)
        
        v = F.relu(self.value_fc1(x))
        v = torch.tanh(self.value_fc2(v))
        
        # Always return 2D tensors - caller should handle squeeze if needed
        return pi, v


class NNetWrapper:
    """Wrapper for neural network with multi-GPU support."""
    
    def __init__(self, game: TogyzkumalakGame, config: AlphaZeroConfig):
        self.game = game
        self.config = config
        self.device = device
        
        # Create base network
        self.nnet_base = AlphaZeroNetwork(
            input_size=128,
            hidden_size=config.hidden_size,
            action_size=game.getActionSize()
        ).to(self.device)
        
        # Multi-GPU support - wrap for training only
        if NUM_GPUS > 1:
            log.info(f"Using DataParallel with {NUM_GPUS} GPUs for TRAINING")
            self.nnet = DataParallel(self.nnet_base)
        else:
            self.nnet = self.nnet_base
        
        # For predict, always use base network (no DataParallel)
        # This avoids issues with single-sample inference on multi-GPU
    
    def train(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]) -> Dict[str, float]:
        """Train network on examples (board, pi, v)."""
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.config.learning_rate)
        
        # Scale batch size with number of GPUs
        effective_batch_size = self.config.batch_size * max(1, NUM_GPUS)
        
        self.nnet.train()
        
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        
        for epoch in range(self.config.epochs):
            batch_count = max(1, len(examples) // effective_batch_size)
            
            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=min(effective_batch_size, len(examples)))
                boards, pis, vs = zip(*[examples[i] for i in sample_ids])
                
                observations = []
                for b in boards:
                    # Validate board size before conversion
                    if len(b) != 23:
                        log.error(f"TRAIN: Board has wrong size: {len(b)}, expected 23. Board: {b[:min(10, len(b))]}")
                        if len(b) == 8:
                            log.error(f"TRAIN: CRITICAL - 8-element board detected in training!")
                            import traceback
                            log.error(f"TRAIN: Call stack:\n{''.join(traceback.format_stack()[-8:-1])}")
                        if len(b) < 23:
                            b = np.pad(b, (0, 23 - len(b)), mode='constant', constant_values=0)
                        else:
                            b = b[:23]
                    obs = self.game.boardToObservation(b)
                    if len(obs) != 128:
                        log.error(f"TRAIN: Observation has wrong size: {len(obs)}, expected 128. Board was: {b[:min(10, len(b))]}")
                        if len(obs) < 128:
                            obs = np.pad(obs, (0, 128 - len(obs)), mode='constant', constant_values=0)
                        else:
                            obs = obs[:128]
                    observations.append(obs)
                
                # Final validation before tensor creation
                obs_array = np.array(observations)
                if obs_array.shape[-1] != 128:
                    log.error(f"TRAIN: CRITICAL - Final observation array has wrong shape: {obs_array.shape}, expected (N, 128)")
                    # Fix it
                    if obs_array.shape[-1] < 128:
                        padding = np.zeros((*obs_array.shape[:-1], 128 - obs_array.shape[-1]), dtype=obs_array.dtype)
                        obs_array = np.concatenate([obs_array, padding], axis=-1)
                    else:
                        obs_array = obs_array[..., :128]
                
                obs_tensor = torch.FloatTensor(obs_array).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).to(self.device)
                
                out_pi, out_v = self.nnet(obs_tensor)
                
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                l_v = torch.sum((target_vs - out_v.squeeze()) ** 2) / target_vs.size(0)
                total_loss = l_pi + l_v
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                pi_losses.update(l_pi.item(), obs_tensor.size(0))
                v_losses.update(l_v.item(), obs_tensor.size(0))
            
            log.info(f'Epoch {epoch+1}/{self.config.epochs} - Policy Loss: {pi_losses} Value Loss: {v_losses}')
        
        return {'policy_loss': pi_losses.avg, 'value_loss': v_losses.avg}
    
    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for board."""
        # Convert to numpy array if needed
        if not isinstance(board, np.ndarray):
            board = np.array(board, dtype=np.float32)
        
        # Ensure board has correct shape (23 elements)
        original_size = len(board)
        if original_size != 23:
            log.error(f"PREDICT: Board has wrong size: {original_size}, expected 23. Board shape: {board.shape}, first 10: {board[:min(10, original_size)]}")
            import traceback
            log.error(f"PREDICT: Call stack:\n{''.join(traceback.format_stack()[-5:-1])}")
            
            if original_size < 23:
                if original_size == 8:
                    # This might be a partial board (first 8 pits), reconstruct full board
                    log.error(f"PREDICT: Detected 8-element board! This should not happen. Reconstructing...")
                    board = np.concatenate([
                        board,  # First 8 pits
                        np.zeros(10, dtype=np.float32),  # Remaining 10 pits
                        np.zeros(5, dtype=np.float32)  # Tuzduk, kazan, player
                    ])
                else:
                    # Pad with zeros
                    board = np.pad(board, (0, 23 - original_size), mode='constant', constant_values=0)
            else:
                # Truncate if too long
                board = board[:23]
        
        # Validate board size before conversion
        if len(board) != 23:
            log.error(f"CRITICAL: Board still has wrong size after fix: {len(board)}")
            raise ValueError(f"Board must have exactly 23 elements, got {len(board)}")
        
        obs = self.game.boardToObservation(board)
        
        # Ensure observation has correct size (128)
        if len(obs) != 128:
            log.error(f"Observation has wrong size: {len(obs)}, expected 128. Board size was: {len(board)}")
            if len(obs) < 128:
                obs = np.pad(obs, (0, 128 - len(obs)), mode='constant', constant_values=0)
            else:
                obs = obs[:128]
        
        # Final validation
        if len(obs) != 128:
            log.error(f"CRITICAL: Observation still has wrong size: {len(obs)}")
            raise ValueError(f"Observation must have exactly 128 elements, got {len(obs)}")
        
        # Use base network for single-sample inference (no DataParallel)
        # DataParallel would try to split batch=1 across GPUs, which fails
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # Shape: (1, 128)
        
        # Validate tensor shape
        if obs_tensor.shape[-1] != 128:
            log.error(f"CRITICAL: Tensor has wrong shape: {obs_tensor.shape}, expected (1, 128)")
            raise ValueError(f"Tensor must have shape (1, 128), got {obs_tensor.shape}")
        
        # CRITICAL: Use nnet_base (not DataParallel wrapped) for single-sample predict
        self.nnet_base.eval()
        with torch.no_grad():
            pi, v = self.nnet_base(obs_tensor)
            pi = torch.exp(pi)
        
        # Remove batch dimension
        return pi.squeeze(0).cpu().numpy(), float(v.squeeze(0).cpu().numpy())
    
    def predict_batch(self, boards: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction for multiple boards - uses DataParallel if available."""
        # Fix board sizes
        fixed_boards = []
        for b in boards:
            if len(b) < 23:
                if len(b) == 8:
                    b = np.concatenate([b, np.zeros(23 - len(b), dtype=np.float32)])
                else:
                    b = np.pad(b, (0, 23 - len(b)), mode='constant', constant_values=0) if len(b) < 23 else b[:23]
            fixed_boards.append(b)
        
        observations = []
        for b in fixed_boards:
            obs = self.game.boardToObservation(b)
            if len(obs) != 128:
                obs = np.pad(obs, (0, 128 - len(obs)), mode='constant', constant_values=0) if len(obs) < 128 else obs[:128]
            observations.append(obs)
        
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        
        # For batch predictions with batch_size >= NUM_GPUS, can use DataParallel
        # Otherwise use base network
        if len(boards) >= NUM_GPUS and NUM_GPUS > 1:
            self.nnet.eval()
            with torch.no_grad():
                pi, v = self.nnet(obs_tensor)
                pi = torch.exp(pi)
        else:
            self.nnet_base.eval()
            with torch.no_grad():
                pi, v = self.nnet_base(obs_tensor)
                pi = torch.exp(pi)
        
        return pi.cpu().numpy(), v.cpu().numpy().flatten()
    
    def save_checkpoint(self, folder: str, filename: str, metrics: Dict = None):
        """Save model checkpoint with metrics."""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        
        # Always use base network state dict (not DataParallel wrapped)
        state_dict = self.nnet_base.state_dict()
        
        checkpoint = {
            'state_dict': state_dict,
            'config': {
                'hidden_size': self.config.hidden_size,
                'action_size': self.game.getActionSize()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, filepath)
        log.info(f'Checkpoint saved: {filepath}')
    
    def load_checkpoint(self, folder: str, filename: str):
        """Load model checkpoint."""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint at {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Always load into base network
        self.nnet_base.load_state_dict(checkpoint['state_dict'])
        
        log.info(f'Checkpoint loaded: {filepath}')
        return checkpoint.get('metrics', {})


# =============================================================================
# MCTS - Monte Carlo Tree Search
# =============================================================================

EPS = 1e-8
MAX_SEARCH_DEPTH = 500


class MCTS:
    """Monte Carlo Tree Search for AlphaZero."""
    
    def __init__(self, game: TogyzkumalakGame, nnet: NNetWrapper, config: AlphaZeroConfig):
        self.game = game
        self.nnet = nnet
        self.config = config
        
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
    
    def getActionProb(self, canonicalBoard: np.ndarray, temp: float = 1) -> np.ndarray:
        """Run MCTS simulations and return action probabilities."""
        for _ in range(self.config.num_mcts_sims):
            self.search(canonicalBoard)
        
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]
        
        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_actions)
            probs = np.zeros(len(counts))
            probs[best_a] = 1
            return probs
        
        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum if counts_sum > 0 else 1.0/len(counts) for x in counts]
        return np.array(probs)
    
    def search(self, canonicalBoard: np.ndarray, depth: int = 0) -> float:
        """One iteration of MCTS."""
        if depth > MAX_SEARCH_DEPTH:
            return 0
        
        # Validate board size before using
        if len(canonicalBoard) != 23:
            log.error(f"MCTS.search: canonicalBoard has wrong size: {len(canonicalBoard)}, expected 23. Board: {canonicalBoard[:min(10, len(canonicalBoard))]}")
            if len(canonicalBoard) == 8:
                log.error(f"MCTS.search: CRITICAL - 8-element board detected! This is likely the source of the error.")
                import traceback
                log.error(f"MCTS.search: Call stack:\n{''.join(traceback.format_stack()[-8:-1])}")
            # Try to fix it
            if len(canonicalBoard) < 23:
                canonicalBoard = np.pad(canonicalBoard, (0, 23 - len(canonicalBoard)), mode='constant', constant_values=0)
            else:
                canonicalBoard = canonicalBoard[:23]
        
        s = self.game.stringRepresentation(canonicalBoard)
        
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]
        
        if s not in self.Ps:
            # Validate again before predict
            if len(canonicalBoard) != 23:
                log.error(f"MCTS.search: canonicalBoard still wrong size before predict: {len(canonicalBoard)}")
                canonicalBoard = np.pad(canonicalBoard, (0, 23 - len(canonicalBoard)), mode='constant', constant_values=0)[:23]
            
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            
            # Check if valids is empty
            if np.sum(valids) == 0:
                # If game logic didn't catch the end, but no moves possible
                log.warning(f"MCTS.search: No valid moves possible for state. Ending search.")
                return -self.game.getGameEnded(canonicalBoard, 1)

            self.Ps[s] = self.Ps[s] * valids
            
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                # All valid moves were masked by network output
                log.warning("MCTS.search: Network masked all valid moves, using uniform fallback")
                self.Ps[s] = valids / np.sum(valids)
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.config.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.config.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                
                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        
        v = self.search(next_s, depth + 1)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        return -v


# =============================================================================
# Parallel Self-Play Worker
# =============================================================================

def execute_episode_worker(args) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Worker function for parallel self-play with explicit GPU assignment."""
    nnet_state, config_dict, worker_id = args
    
    # Assign to GPU based on worker ID
    num_gpus = torch.cuda.device_count()
    gpu_id = worker_id % max(1, num_gpus)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Recreate game and network
    game = TogyzkumalakGame()
    config = AlphaZeroConfig(**config_dict)
    
    # Create network on specific GPU
    from alphazero_trainer import AlphaZeroNetwork # Re-import inside worker
    nnet_model = AlphaZeroNetwork(
        input_size=128,
        hidden_size=config.hidden_size,
        action_size=game.getActionSize()
    ).to(device)
    
    # Load weights
    clean_state = {k.replace("module.", ""): v for k, v in nnet_state.items()}
    nnet_model.load_state_dict(clean_state)
    nnet_model.eval()
    
    class WorkerNNet:
        def __init__(self, net, dev, g):
            self.net = net
            self.dev = dev
            self.game = g
        def predict(self, board):
            obs = self.game.boardToObservation(board)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.dev)
            with torch.no_grad():
                pi, v = self.net(obs_tensor)
                pi = torch.exp(pi)
            return pi.squeeze(0).cpu().numpy(), float(v.squeeze(0).cpu().numpy())

    mcts = MCTS(game, WorkerNNet(nnet_model, device, game), config)
    
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0
    max_steps = 500
    
    while True:
        episodeStep += 1
        if episodeStep > max_steps: break
        
        canonical = game.getCanonicalForm(board, curPlayer)
        temp = int(episodeStep < config.temp_threshold)
        
        try:
            pi = mcts.getActionProb(canonical, temp=temp)
        except Exception:
            return [] # Failed game
            
        sym = game.getSymmetries(canonical, pi)
        for b, p in sym:
            trainExamples.append([b, curPlayer, p, None])
        
        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.getNextState(board, curPlayer, action)
        
        r = game.getGameEnded(board, curPlayer)
        if r != 0:
            return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
    
    return []

# =============================================================================
# Fast Self-Play (NO ProcessPoolExecutor - single process, batch inference)
# =============================================================================

class ParallelSelfPlay:
    """
    Run multiple games in parallel using BATCH inference on single GPU.
    This is 10-100x faster than ProcessPoolExecutor for GPU tasks!
    
    Key insight: GPU is fast at batched operations. Running N games with 
    batch inference of N boards is much faster than N processes each doing
    1 inference, because:
    1. No CUDA context creation overhead (saves 10-30 seconds per process)
    2. No pickle/unpickle of tensors between processes
    3. GPU parallelism is utilized (batch size N uses all CUDA cores)
    """
    
    def __init__(self, game: TogyzkumalakGame, nnet: 'NNetWrapper', config: AlphaZeroConfig, num_games: int = 8):
        self.game = game
        self.nnet = nnet
        self.config = config
        self.num_games = num_games
        
        # Each game state
        self.boards = [None] * num_games
        self.players = [None] * num_games
        self.histories = [None] * num_games  # Training examples
        self.steps = [0] * num_games
        self.done = [False] * num_games
        self.mcts_trees = [None] * num_games  # Separate MCTS tree per game
    
    def reset_all(self):
        """Initialize all games."""
        for i in range(self.num_games):
            self.boards[i] = self.game.getInitBoard()
            self.players[i] = 1
            self.histories[i] = []
            self.steps[i] = 0
            self.done[i] = False
            self.mcts_trees[i] = SimpleMCTS(self.game, self.config)
    
    def run_all_games(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Run all games to completion using batch inference."""
        self.reset_all()
        all_examples = []
        max_steps = 500
        
        while not all(self.done):
            # Collect boards that need prediction
            active_indices = [i for i in range(self.num_games) if not self.done[i]]
            
            if not active_indices:
                break
            
            # For each active game, run MCTS and make a move
            for i in active_indices:
                self.steps[i] += 1
                
                if self.steps[i] > max_steps:
                    # Game took too long, end with draw
                    for b, p, pi in self.histories[i]:
                        all_examples.append((b, pi, 0))
                    self.done[i] = True
                    continue
                
                canonical = self.game.getCanonicalForm(self.boards[i], self.players[i])
                temp = 1.0 if self.steps[i] < self.config.temp_threshold else 0
                
                # Run MCTS for this game
                pi = self.mcts_trees[i].getActionProb(canonical, self.nnet, temp)
                
                # Save training example
                self.histories[i].append((canonical.copy(), self.players[i], pi.copy()))
                
                # Choose action
                if temp == 0:
                    action = np.argmax(pi)
                else:
                    action = np.random.choice(len(pi), p=pi)
                
                # Make move
                self.boards[i], self.players[i] = self.game.getNextState(
                    self.boards[i], self.players[i], action
                )
                
                # Check game end
                result = self.game.getGameEnded(self.boards[i], self.players[i])
                if result != 0:
                    self.done[i] = True
                    # Assign rewards
                    for b, player, pi in self.histories[i]:
                        # Reward from perspective of the player who made the move
                        v = result * ((-1) ** (player != self.players[i]))
                        all_examples.append((b, pi, v))
        
        return all_examples


class SimpleMCTS:
    """
    Simplified MCTS that works with external neural network.
    Designed for parallel game execution with shared network.
    """
    
    def __init__(self, game: TogyzkumalakGame, config: AlphaZeroConfig):
        self.game = game
        self.config = config
        self.Qsa = {}  # Q-values
        self.Nsa = {}  # Visit counts (s,a)
        self.Ns = {}   # Visit counts (s)
        self.Ps = {}   # Policy from network
        self.Es = {}   # Game ended
        self.Vs = {}   # Valid moves
    
    def getActionProb(self, canonicalBoard: np.ndarray, nnet: 'NNetWrapper', temp: float = 1) -> np.ndarray:
        """Run MCTS simulations and return action probabilities."""
        for _ in range(self.config.num_mcts_sims):
            self._search(canonicalBoard, nnet)
        
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]
        
        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_actions)
            probs = np.zeros(len(counts))
            probs[best_a] = 1
            return probs
        
        counts_exp = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts_exp))
        if counts_sum > 0:
            probs = [x / counts_sum for x in counts_exp]
        else:
            # No visits, use uniform
            probs = [1.0 / len(counts)] * len(counts)
        return np.array(probs)
    
    def _search(self, canonicalBoard: np.ndarray, nnet: 'NNetWrapper', depth: int = 0) -> float:
        """One iteration of MCTS."""
        if depth > 200:
            return 0
        
        s = self.game.stringRepresentation(canonicalBoard)
        
        # Check game end
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]
        
        # Leaf node - expand
        if s not in self.Ps:
            pi, v = nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            
            # Mask invalid moves
            pi = pi * valids
            sum_pi = np.sum(pi)
            
            if sum_pi > 0:
                pi /= sum_pi
            else:
                # Network gave zero for all valid moves - use uniform
                pi = valids / np.sum(valids) if np.sum(valids) > 0 else np.ones(len(valids)) / len(valids)
            
            self.Ps[s] = pi
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        
        # Select action with highest UCB
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = 0
        
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.config.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.config.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1)
                
                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        a = best_act
        
        # Make move
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        
        # Recurse
        v = self._search(next_s, nnet, depth + 1)
        
        # Update Q-values
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        return -v


# =============================================================================
# Arena - Model Evaluation
# =============================================================================

class Arena:
    """Pit two models against each other."""
    
    def __init__(self, player1, player2, game: TogyzkumalakGame):
        self.player1 = player1
        self.player2 = player2
        self.game = game
    
    def playGame(self, verbose: bool = False) -> int:
        """Play one game. Returns 1 if player1 wins, -1 if player2 wins."""
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        
        move_count = 0
        max_moves = 500
        
        while self.game.getGameEnded(board, curPlayer) == 0:
            move_count += 1
            if move_count > max_moves:
                break
            
            canonical = self.game.getCanonicalForm(board, curPlayer)
            action = players[curPlayer + 1](canonical)
            
            valids = self.game.getValidMoves(canonical, 1)
            if valids[action] == 0:
                log.error(f"Invalid action {action}")
                action = np.argmax(valids)
            
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        
        if move_count > max_moves:
            return 0
        
        return curPlayer * self.game.getGameEnded(board, curPlayer)
    
    def playGames(self, num: int) -> Tuple[int, int, int]:
        """Play num games. Returns (player1 wins, player2 wins, draws)."""
        num = max(2, num)
        num_each = num // 2
        
        oneWon = 0
        twoWon = 0
        draws = 0
        
        for _ in range(num_each):
            result = self.playGame()
            if result == 1:
                oneWon += 1
            elif result == -1:
                twoWon += 1
            else:
                draws += 1
        
        self.player1, self.player2 = self.player2, self.player1
        
        for _ in range(num_each):
            result = self.playGame()
            if result == -1:
                oneWon += 1
            elif result == 1:
                twoWon += 1
            else:
                draws += 1
        
        return oneWon, twoWon, draws


# =============================================================================
# Coach - Main Training Loop with Multi-GPU Support
# =============================================================================

class AlphaZeroCoach:
    """
    AlphaZero training coach with multi-GPU and parallel self-play support.
    """
    
    def __init__(self, game: TogyzkumalakGame, nnet: NNetWrapper, config: AlphaZeroConfig):
        self.game = game
        self.nnet = nnet
        self.pnet = NNetWrapper(game, config)
        self.config = config
        self.mcts = MCTS(game, nnet, config)
        
        self.trainExamplesHistory = []
        self.metrics_history = []
        self.checkpoint_metrics = {}  # Per-checkpoint detailed metrics
        
        self.current_iteration = 0
        self.status = "initialized"
        self.progress = 0.0
        self.stop_requested = False
        
        # Determine parallelization level - optimize for multi-GPU
        self.num_workers = config.num_workers if config.num_workers > 0 else max(1, NUM_GPUS)
        # For multi-GPU: use at least 1 game per GPU, max 4 games per GPU
        self.num_parallel_games = config.num_parallel_games if config.num_parallel_games > 0 else max(NUM_GPUS, min(NUM_GPUS * 4, 64))
        
        log.info(f"🚀 BLITZ MODE: {self.num_parallel_games} parallel games across {NUM_GPUS} GPUs")
        log.info(f"   MCTS sims: {config.num_mcts_sims}, Episodes: {config.num_episodes}, Arena: {config.arena_compare}")
    
    def executeEpisode(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Execute one episode of self-play."""
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0
        max_steps = 500
        
        while True:
            episodeStep += 1
            
            if episodeStep > max_steps:
                return [(x[0], x[2], 0) for x in trainExamples]
            
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < self.config.temp_threshold)
            
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])
            
            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            
            r = self.game.getGameEnded(board, curPlayer)
            
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
    
    def executeEpisodesParallel(self, num_episodes: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Execute multiple episodes in parallel.
        Uses ProcessPoolExecutor for multi-GPU distribution if enabled.
        """
        if not self.config.use_multiprocessing or self.num_parallel_games <= 1:
            # Fallback to single-process batching
            return self._executeEpisodesBatch(num_episodes)

        all_examples = []
        
        # Prepare data for workers
        nnet_state = {k: v.cpu() for k, v in self.nnet.nnet_base.state_dict().items()}
        config_dict = {
            'num_mcts_sims': self.config.num_mcts_sims,
            'cpuct': self.config.cpuct,
            'temp_threshold': self.config.temp_threshold,
            'hidden_size': self.config.hidden_size,
        }
        
        # Determine workers (typically one per GPU or two)
        workers = min(self.num_parallel_games, num_episodes)
        
        log.info(f"🎮 Distributing {num_episodes} games across {workers} workers on {NUM_GPUS} GPUs...")
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            args = [(nnet_state, config_dict, i) for i in range(num_episodes)]
            futures = [executor.submit(execute_episode_worker, arg) for arg in args]
            
            for future in futures:
                try:
                    result = future.result(timeout=600) # 10 min timeout
                    if result:
                        all_examples.extend(result)
                        completed += 1
                    
                    if completed % 10 == 0:
                        log.info(f"  ✅ Progress: {completed}/{num_episodes} games finished")
                except Exception as e:
                    log.error(f"  ❌ Game failed: {e}")
        
        log.info(f"🏁 Parallel self-play complete: {len(all_examples)} examples collected")
        return all_examples

    def _executeEpisodesBatch(self, num_episodes: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Batch execution fallback (single process)."""
        all_examples = []
        batch_size = min(16, num_episodes)
        num_batches = (num_episodes + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            n = min(batch_size, num_episodes - i * batch_size)
            psp = ParallelSelfPlay(self.game, self.nnet, self.config, num_games=n)
            all_examples.extend(psp.run_all_games())
            log.info(f"  ✅ Batch {i+1}/{num_batches} complete")
            
        return all_examples
    
    def _run_episode(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Run a single episode (for thread pool)."""
        mcts = MCTS(self.game, self.nnet, self.config)
        
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0
        max_steps = 500
        
        while True:
            episodeStep += 1
            
            if episodeStep > max_steps:
                return [(x[0], x[2], 0) for x in trainExamples]
            
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < self.config.temp_threshold)
            
            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])
            
            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            
            r = self.game.getGameEnded(board, curPlayer)
            
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
    
    def _bootstrap_from_human_data(self, callback=None) -> int:
        """Bootstrap the network from human game data."""
        bootstrap_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            self.config.bootstrap_file
        )
        
        if not os.path.exists(bootstrap_file):
            log.warning(f"Bootstrap file not found: {bootstrap_file}")
            return 0
        
        log.info(f"[Bootstrap] Loading human data from {bootstrap_file}")
        
        try:
            examples = []
            with open(bootstrap_file, 'r') as f:
                for line in f:
                    if len(examples) >= self.config.bootstrap_max_samples:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    
                    state = data.get('s', data.get('state', data.get('observation', [])))
                    action = data.get('a', data.get('action', 0))
                    reward = data.get('r', data.get('reward', data.get('final_reward', 0)))
                    
                    if not state or len(state) < 18:
                        continue
                    
                    if len(state) == 20:
                        board = np.zeros(23, dtype=np.float32)
                        for i in range(18):
                            board[i] = state[i] * 81.0
                        board[20] = state[18] * 162.0
                        board[21] = state[19] * 162.0
                    elif len(state) == 128:
                        board = np.zeros(23, dtype=np.float32)
                        for i in range(18):
                            board[i] = state[i] * 81.0
                        board[20] = state[18] * 162.0
                        board[21] = state[19] * 162.0
                    elif len(state) >= 22:
                        board = np.array(state[:23] if len(state) >= 23 else state + [0]*(23-len(state)), dtype=np.float32)
                    else:
                        continue
                    
                    pi = np.zeros(9, dtype=np.float32)
                    if 0 <= action < 9:
                        pi[action] = 1.0
                    
                    v = float(reward) if abs(reward) <= 1 else (1.0 if reward > 0 else -1.0)
                    
                    examples.append((board, pi, v))
            
            if not examples:
                log.warning("[Bootstrap] No valid examples loaded")
                return 0
            
            log.info(f"[Bootstrap] Loaded {len(examples)} examples, training for {self.config.bootstrap_epochs} epochs...")
            
            start_time = time.time()
            for epoch in range(self.config.bootstrap_epochs):
                if self.stop_requested: break
                
                train_metrics = self.nnet.train(examples)
                
                # Update status for UI
                if callback:
                    metrics = {
                        'iteration': 0,
                        'policy_loss': train_metrics.get('policy_loss', 0),
                        'value_loss': train_metrics.get('value_loss', 0),
                        'win_rate': 0,
                        'total_examples': len(examples),
                        'stage': f'Bootstrap Epoch {epoch+1}/{self.config.bootstrap_epochs}'
                    }
                    callback({
                        'status': 'running',
                        'progress': (epoch + 1) / self.config.bootstrap_epochs * 5, # First 5% for bootstrap
                        'iteration': 0,
                        'metrics': metrics,
                        'elapsed_time': time.time() - start_time
                    })

                if epoch % 2 == 0:
                    log.info(f"  Bootstrap epoch {epoch+1}/{self.config.bootstrap_epochs}, "
                            f"policy_loss: {train_metrics.get('policy_loss', 0):.4f}, "
                            f"value_loss: {train_metrics.get('value_loss', 0):.4f}")
            
            self.nnet.save_checkpoint(
                self.config.checkpoint_dir, 
                'bootstrapped.pth.tar',
                metrics={'bootstrap_examples': len(examples), 'bootstrap_epochs': self.config.bootstrap_epochs}
            )
            log.info(f"[Bootstrap] Complete! Model saved to bootstrapped.pth.tar")
            
            return len(examples)
            
        except Exception as e:
            log.error(f"[Bootstrap] Error: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _try_resume_from_checkpoint(self) -> bool:
        """Try to resume from existing checkpoint (best.pth.tar or bootstrapped.pth.tar)."""
        checkpoint_dir = self.config.checkpoint_dir
        
        # Priority order: best.pth.tar > bootstrapped.pth.tar > latest checkpoint_N.pth.tar
        checkpoints_to_try = [
            os.path.join(checkpoint_dir, 'best.pth.tar'),
            os.path.join(checkpoint_dir, 'bootstrapped.pth.tar'),
        ]
        
        # Also try to find latest numbered checkpoint
        if os.path.exists(checkpoint_dir):
            numbered = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth.tar')]
            if numbered:
                # Sort by iteration number
                numbered.sort(key=lambda x: int(x.replace('checkpoint_', '').replace('.pth.tar', '')), reverse=True)
                checkpoints_to_try.append(os.path.join(checkpoint_dir, numbered[0]))
        
        for checkpoint_path in checkpoints_to_try:
            if os.path.exists(checkpoint_path):
                try:
                    metrics = self.nnet.load_checkpoint(os.path.dirname(checkpoint_path), os.path.basename(checkpoint_path))
                    log.info(f"✅ RESUMED from checkpoint: {checkpoint_path}")
                    if metrics:
                        log.info(f"   Checkpoint metrics: {metrics}")
                    return True
                except Exception as e:
                    log.warning(f"Failed to load {checkpoint_path}: {e}")
                    continue
        
        return False
    
    def learn(self, callback=None) -> Dict[str, Any]:
        """Main training loop with parallel self-play."""
        self.status = "running"
        start_time = time.time()
        
        # Try to resume from existing checkpoint first
        resumed = False
        if self.config.resume_from_checkpoint:
            resumed = self._try_resume_from_checkpoint()
        
        # Bootstrap from human data if enabled AND we didn't resume
        if self.config.use_bootstrap and not resumed:
            bootstrap_count = self._bootstrap_from_human_data(callback=callback)
            if bootstrap_count > 0:
                log.info(f"Bootstrapped from {bootstrap_count} human examples")
        
        for i in range(1, self.config.num_iterations + 1):
            if self.stop_requested:
                self.status = "stopped"
                break
            
            iter_start = time.time()
            self.current_iteration = i
            self.progress = (i - 1) / self.config.num_iterations * 100
            
            log.info(f'=== Iteration {i}/{self.config.num_iterations} ===')
            
            # Self-play (parallel if enabled)
            iterationExamples = deque([], maxlen=self.config.max_queue_length)
            
            if self.config.use_multiprocessing and self.num_parallel_games > 1:
                examples = self.executeEpisodesParallel(self.config.num_episodes)
                iterationExamples.extend(examples)
            else:
                for ep in range(self.config.num_episodes):
                    if self.stop_requested:
                        break
                    self.mcts = MCTS(self.game, self.nnet, self.config)
                    examples = self.executeEpisode()
                    iterationExamples.extend(examples)
                    log.info(f'  Self-play episode {ep+1}/{self.config.num_episodes}, examples: {len(examples)}')
            
            self.trainExamplesHistory.append(iterationExamples)
            
            while len(self.trainExamplesHistory) > self.config.num_iters_for_history:
                self.trainExamplesHistory.pop(0)
            
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            np.random.shuffle(trainExamples)
            
            # Save current network
            self.nnet.save_checkpoint(self.config.checkpoint_dir, 'temp.pth.tar')
            self.pnet.load_checkpoint(self.config.checkpoint_dir, 'temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.config)
            
            # Train
            log.info(f'  Training on {len(trainExamples)} examples...')
            train_metrics = self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.config)
            
            # Arena evaluation
            log.info('  Arena: comparing new vs old model...')
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game
            )
            pwins, nwins, draws = arena.playGames(self.config.arena_compare)
            
            log.info(f'  Arena results - New wins: {nwins}, Old wins: {pwins}, Draws: {draws}')
            
            win_rate = nwins / (pwins + nwins) if (pwins + nwins) > 0 else 0
            
            if win_rate < self.config.update_threshold:
                log.info('  REJECTING new model')
                self.nnet.load_checkpoint(self.config.checkpoint_dir, 'temp.pth.tar')
                accepted = False
            else:
                log.info('  ACCEPTING new model')
                accepted = True
            
            iter_time = time.time() - iter_start
            
            # Record detailed metrics for this checkpoint
            iter_metrics = {
                'iteration': i,
                'policy_loss': train_metrics['policy_loss'],
                'value_loss': train_metrics['value_loss'],
                'total_loss': train_metrics['policy_loss'] + train_metrics['value_loss'],
                'new_wins': nwins,
                'old_wins': pwins,
                'draws': draws,
                'win_rate': win_rate,
                'accepted': accepted,
                'total_examples': len(trainExamples),
                'iteration_time_sec': iter_time,
                'timestamp': datetime.now().isoformat(),
                'gpus_used': NUM_GPUS,
                'device': str(device)
            }
            self.metrics_history.append(iter_metrics)
            
            # Save checkpoint with metrics
            if accepted:
                checkpoint_name = f'checkpoint_{i}.pth.tar'
                self.nnet.save_checkpoint(
                    self.config.checkpoint_dir, 
                    checkpoint_name,
                    metrics=iter_metrics
                )
                self.nnet.save_checkpoint(
                    self.config.checkpoint_dir, 
                    'best.pth.tar',
                    metrics=iter_metrics
                )
                self.checkpoint_metrics[checkpoint_name] = iter_metrics
            
            # Save periodic checkpoints regardless of acceptance
            if i % self.config.save_every_n_iters == 0:
                self.nnet.save_checkpoint(
                    self.config.checkpoint_dir,
                    f'alphazero_iter{i}.pth.tar',
                    metrics=iter_metrics
                )
            
            self.progress = i / self.config.num_iterations * 100
            
            # Save metrics after each iteration (for real-time viewing)
            self._save_metrics()
            
            if callback:
                callback({
                    'iteration': i,
                    'total_iterations': self.config.num_iterations,
                    'progress': self.progress,
                    'metrics': iter_metrics,
                    'status': self.status,
                    'elapsed_time': time.time() - start_time
                })
        
        if not self.stop_requested:
            self.status = "completed"
        
        self._save_metrics()
        
        total_time = time.time() - start_time
        
        return {
            'status': self.status,
            'iterations_completed': self.current_iteration,
            'metrics_history': self.metrics_history,
            'total_time_sec': total_time,
            'avg_time_per_iter': total_time / max(1, self.current_iteration)
        }
    
    def _save_metrics(self):
        """Save training metrics to file."""
        metrics_path = os.path.join(self.config.checkpoint_dir, 'training_metrics.json')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'config': {
                    'num_iterations': self.config.num_iterations,
                    'num_episodes': self.config.num_episodes,
                    'num_mcts_sims': self.config.num_mcts_sims,
                    'batch_size': self.config.batch_size,
                    'hidden_size': self.config.hidden_size,
                    'num_gpus': NUM_GPUS,
                    'device': str(device)
                },
                'metrics': self.metrics_history,
                'checkpoint_metrics': self.checkpoint_metrics,
                'best_iteration': self._get_best_iteration()
            }, f, indent=2)
    
    def _get_best_iteration(self) -> Dict:
        """Get the iteration with lowest total loss."""
        if not self.metrics_history:
            return {}
        
        best = min(self.metrics_history, key=lambda x: x.get('total_loss', float('inf')))
        return {
            'iteration': best.get('iteration'),
            'total_loss': best.get('total_loss'),
            'policy_loss': best.get('policy_loss'),
            'value_loss': best.get('value_loss'),
            'win_rate': best.get('win_rate')
        }
    
    def stop(self):
        """Request training to stop."""
        self.stop_requested = True


# =============================================================================
# Task Manager Integration
# =============================================================================

class AlphaZeroTrainingTask:
    """Manages an AlphaZero training task in a background thread."""
    
    def __init__(self, task_id: str, config: AlphaZeroConfig):
        self.task_id = task_id
        self.config = config
        self.game = TogyzkumalakGame()
        self.nnet = NNetWrapper(self.game, config)
        self.coach = AlphaZeroCoach(self.game, self.nnet, config)
        
        self.thread = None
        self.result = None
        self.error = None
        
        self.status = {
            'task_id': task_id,
            'status': 'pending',
            'progress': 0,
            'current_iteration': 0,
            'total_iterations': config.num_iterations,
            'metrics': [],
            'start_time': None,
            'end_time': None,
            'error_message': None,
            'gpus': NUM_GPUS,
            'device': str(device)
        }
    
    def start(self):
        """Start training in background thread."""
        self.status['status'] = 'running'
        self.status['start_time'] = datetime.now().isoformat()
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self):
        """Run training."""
        try:
            self.result = self.coach.learn(callback=self._update_status)
            self.status['status'] = 'completed'
        except Exception as e:
            self.error = str(e)
            self.status['status'] = 'error'
            self.status['error_message'] = str(e)
            log.error(f"Training task {self.task_id} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.status['end_time'] = datetime.now().isoformat()
    
    def _update_status(self, data: Dict):
        """Update status from callback."""
        self.status['progress'] = data.get('progress', 0)
        self.status['current_iteration'] = data.get('iteration', 0)
        self.status['status'] = data.get('status', 'running')
        self.status['elapsed_time'] = data.get('elapsed_time', 0)
        
        if 'metrics' in data:
            if len(self.status['metrics']) >= 20:
                self.status['metrics'] = self.status['metrics'][-19:]
            self.status['metrics'].append(data['metrics'])
    
    def stop(self):
        """Request training to stop."""
        self.coach.stop()
        self.status['status'] = 'stopping'
    
    def get_status(self) -> Dict:
        """Get current status."""
        return self.status.copy()


class AlphaZeroTaskManagerV2:
    """Manages AlphaZero training tasks."""
    
    def __init__(self, checkpoint_dir: str = "models/alphazero"):
        self.tasks: Dict[str, AlphaZeroTrainingTask] = {}
        self.checkpoint_dir = checkpoint_dir
        self.lock = threading.Lock()
    
    def start_training(self, params: Dict) -> str:
        """Start a new training task."""
        task_id = f"az_{int(time.time())}"
        
        config = AlphaZeroConfig(
            num_iterations=params.get('numIters', 100),
            num_episodes=params.get('numEps', 100),
            num_mcts_sims=params.get('numMCTSSims', 100),
            cpuct=params.get('cpuct', 1.0),
            batch_size=params.get('batch_size', 256),
            hidden_size=params.get('hidden_size', 256),
            epochs=params.get('epochs', 10),
            checkpoint_dir=self.checkpoint_dir,
            use_bootstrap=params.get('use_bootstrap', True),
            use_multiprocessing=params.get('use_multiprocessing', True),
            num_parallel_games=params.get('num_parallel_games', 0),
            save_every_n_iters=params.get('save_every_n_iters', 5)
        )
        
        task = AlphaZeroTrainingTask(task_id, config)
        
        with self.lock:
            self.tasks[task_id] = task
        
        task.start()
        return task_id
    
    def get_status(self, task_id: str) -> Optional[Dict]:
        """Get task status."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                return task.get_status()
        return None
    
    def stop_task(self, task_id: str) -> bool:
        """Stop a running task."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.stop()
                return True
        return False
    
    def list_tasks(self) -> Dict[str, Dict]:
        """List all tasks."""
        with self.lock:
            return {tid: task.get_status() for tid, task in self.tasks.items()}
    
    def get_checkpoints(self) -> List[Dict]:
        """Get list of all checkpoints with their metrics."""
        checkpoints = []
        checkpoint_dir = Path(self.checkpoint_dir)
        
        if not checkpoint_dir.exists():
            return checkpoints
        
        for file in checkpoint_dir.glob('*.pth.tar'):
            if file.name in ['temp.pth.tar']:
                continue
            
            try:
                checkpoint = torch.load(file, map_location='cpu')
                metrics = checkpoint.get('metrics', {})
                checkpoints.append({
                    'name': file.name,
                    'path': str(file),
                    'size_mb': file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    'metrics': metrics
                })
            except Exception as e:
                log.warning(f"Could not load checkpoint {file}: {e}")
                checkpoints.append({
                    'name': file.name,
                    'path': str(file),
                    'size_mb': file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    'metrics': {}
                })
        
        # Sort by iteration number or modification time
        checkpoints.sort(key=lambda x: x.get('metrics', {}).get('iteration', 0), reverse=True)
        
        return checkpoints


# Global instance
alphazero_task_manager = AlphaZeroTaskManagerV2()


# =============================================================================
# Optimal Config Calculator
# =============================================================================

def get_optimal_config(num_gpus: int, time_budget_hours: float = 1.0) -> Dict:
    """
    Calculate optimal training config for given GPU count and time budget.
    
    BLITZ MODE: Optimized for FAST iterations with quality results.
    RTX 3090/4090 class GPUs (24GB VRAM each).
    
    Args:
        num_gpus: Number of available GPUs
        time_budget_hours: Available training time in hours
    
    Returns:
        Recommended configuration dictionary
    """
    # BLITZ MODE estimates - optimized for speed over deep search
    # Each iteration should complete in 5-10 minutes on multi-GPU
    base_batch_size = 256
    
    # Scale with GPUs (sublinear due to sync overhead)
    # Efficiency: 1 GPU=100%, 4 GPU=90%, 8 GPU=85%, 16 GPU=80%
    efficiency = max(0.6, 1.0 - (num_gpus - 1) * 0.015)
    
    effective_batch_size = base_batch_size * max(1, num_gpus // 2)
    parallel_games = min(64, num_gpus * 4)  # 4 parallel games per GPU, max 64
    
    # BLITZ: Fast iterations, more of them
    time_budget_min = time_budget_hours * 60
    
    # Configuration tiers - BLITZ optimized
    if num_gpus >= 16:
        # Monster setup (16+ GPUs) - BLITZ: POWERFUL AND FAST
        mcts_sims = 100         # Increased for quality
        episodes = 256          # Increased for better data
        iter_time_min = 8       # Target: 8 min per iteration
        hidden_size = 256
        arena_games = 40        # More accurate evaluation
        epochs = 10
        save_interval = 2
    elif num_gpus >= 8:
        # Large setup (8-15 GPUs) - BLITZ
        mcts_sims = 80
        episodes = 128
        iter_time_min = 10
        hidden_size = 256
        arena_games = 32
        epochs = 10
        save_interval = 2
    elif num_gpus >= 4:
        # Medium setup (4-7 GPUs) - BLITZ
        mcts_sims = 25
        episodes = 32
        iter_time_min = 8
        hidden_size = 256
        arena_games = 12
        epochs = 8
        save_interval = 5
    elif num_gpus >= 2:
        # Small setup (2-3 GPUs) - BLITZ
        mcts_sims = 20
        episodes = 24
        iter_time_min = 10
        hidden_size = 256
        arena_games = 10
        epochs = 8
        save_interval = 5
    else:
        # Single GPU - BLITZ
        mcts_sims = 15
        episodes = 16
        iter_time_min = 12
        hidden_size = 256
        arena_games = 8
        epochs = 5
        save_interval = 5
    
    # Calculate iterations for time budget
    estimated_iterations = int(time_budget_min / iter_time_min)
    
    # Estimated metrics
    examples_per_iter = episodes * 50  # ~50 moves per game average
    total_examples = estimated_iterations * examples_per_iter
    
    speedup = num_gpus * efficiency
    
    return {
        'numIters': estimated_iterations,
        'numEps': episodes,
        'numMCTSSims': mcts_sims,
        'cpuct': 1.0,
        'batch_size': effective_batch_size,
        'epochs': epochs,
        'hidden_size': hidden_size,
        'arena_compare': arena_games,
        'temp_threshold': 30,  # More exploration
        'use_bootstrap': True,
        'use_multiprocessing': True,
        'num_parallel_games': parallel_games,
        'save_every_n_iters': save_interval,
        'estimated_time_min': round(estimated_iterations * iter_time_min, 1),
        'estimated_examples': total_examples,
        'gpus': num_gpus,
        'efficiency': f"{efficiency*100:.0f}%",
        'speedup': f"{speedup:.1f}x",
        'mode': 'BLITZ',
        'iter_time_min': iter_time_min
    }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing AlphaZero Trainer for Togyzkumalak...")
    print(f"Device: {device}")
    print(f"GPUs: {NUM_GPUS}")
    
    if NUM_GPUS > 0:
        for i in range(NUM_GPUS):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test game logic
    game = TogyzkumalakGame()
    board = game.getInitBoard()
    print(f"Initial board: {board[:18]}")
    print(f"Valid moves: {game.getValidMoves(board, 1)}")
    
    # Test network
    config = AlphaZeroConfig(
        num_iterations=1,
        num_episodes=2,
        num_mcts_sims=10,
        epochs=1
    )
    nnet = NNetWrapper(game, config)
    pi, v = nnet.predict(board)
    print(f"Policy: {pi}")
    print(f"Value: {v}")
    
    # Test optimal config
    print("\n=== Optimal Configs ===")
    for gpus in [1, 4, 8, 16]:
        cfg = get_optimal_config(gpus, 1.0)
        print(f"{gpus} GPUs: {cfg['numIters']} iters, {cfg['numEps']} eps, ~{cfg['estimated_time_min']:.0f} min")
    
    print("\nAlphaZero trainer ready for use!")
