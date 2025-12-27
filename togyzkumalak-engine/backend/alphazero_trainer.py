"""
AlphaZero Trainer for Togyzkumalak - CPU Optimized Version

Self-contained AlphaZero implementation optimized for CPU training.
Uses PyTorch CPU-only, with reduced simulations and batch sizes for efficiency.
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
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import logging
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Ensure CPU-only execution
device = torch.device("cpu")
torch.set_num_threads(max(1, os.cpu_count() - 1))  # Leave one core free


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
    """Configuration for AlphaZero training - CPU optimized defaults."""
    # Training
    num_iterations: int = 10          # Number of training iterations
    num_episodes: int = 10            # Self-play games per iteration (reduced for CPU)
    num_mcts_sims: int = 25           # MCTS simulations per move (25-50 for CPU)
    
    # Neural Network
    batch_size: int = 32              # Batch size (32-64 for CPU to avoid OOM)
    epochs: int = 5                   # Training epochs per iteration
    learning_rate: float = 0.001
    hidden_size: int = 256            # Hidden layer size
    
    # MCTS
    cpuct: float = 1.0                # Exploration constant
    temp_threshold: int = 15          # Move count threshold for temperature
    
    # Arena
    arena_compare: int = 20           # Games to compare models (reduced for CPU)
    update_threshold: float = 0.55    # Win rate needed to accept new model
    
    # Memory
    max_queue_length: int = 50000     # Max training examples (reduced for memory)
    num_iters_for_history: int = 5    # Keep last N iterations of examples
    
    # Checkpoints
    checkpoint_dir: str = "models/alphazero"


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
        new_board = board.copy()
        color = 0 if player == 1 else 1  # Convert player format
        
        pit_index = action + (color * 9)
        num = int(new_board[pit_index])
        
        if num <= 0:
            return new_board, -player
        
        # Pick up kumalaks
        if num == 1:
            new_board[pit_index] = 0
            sow = 1
        else:
            new_board[pit_index] = 1
            sow = num - 1
        
        # Sow kumalaks
        current = pit_index
        for _ in range(sow):
            current = (current + 1) % 18
            if new_board[current] == self.TUZDUK:
                # Add to opponent's kazan
                if current < 9:
                    new_board[21] += 1
                else:
                    new_board[20] += 1
            else:
                new_board[current] += 1
        
        # Check for capture (even number on opponent's side)
        if new_board[current] != self.TUZDUK and new_board[current] % 2 == 0:
            if color == 0 and current > 8:
                new_board[20] += new_board[current]
                new_board[current] = 0
            elif color == 1 and current < 9:
                new_board[21] += new_board[current]
                new_board[current] = 0
        
        # Check for tuzduk (exactly 3)
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
        
        # Switch player
        new_board[22] = 1 - color
        
        # Check atsyrau
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
        return valid
    
    def getGameEnded(self, board: np.ndarray, player: int) -> float:
        """
        Returns 0 if not ended, 1 if player won, -1 if lost, small value for draw.
        """
        white_kazan = board[20]
        black_kazan = board[21]
        
        if white_kazan > 81:
            return 1.0 if player == 1 else -1.0
        elif black_kazan > 81:
            return -1.0 if player == 1 else 1.0
        elif white_kazan == 81 and black_kazan == 81:
            return 1e-4  # Draw
        
        return 0
    
    def getCanonicalForm(self, board: np.ndarray, player: int) -> np.ndarray:
        """Returns board from player's perspective."""
        if player == 1:
            return board.copy()
        
        # Swap for black player
        canonical = np.zeros_like(board)
        canonical[0:9] = board[9:18]
        canonical[9:18] = board[0:9]
        canonical[18] = board[19]
        canonical[19] = board[18]
        canonical[20] = board[21]
        canonical[21] = board[20]
        canonical[22] = 1 - board[22]
        
        return canonical
    
    def getSymmetries(self, board: np.ndarray, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Togyzkumalak has no symmetries."""
        return [(board, pi)]
    
    def stringRepresentation(self, board: np.ndarray) -> str:
        """String representation for MCTS hashing."""
        return board.tobytes()
    
    def boardToObservation(self, board: np.ndarray) -> np.ndarray:
        """Convert board to neural network input (128-dim)."""
        obs = np.zeros(128, dtype=np.float32)
        
        # Pits (normalized)
        for i in range(18):
            val = board[i] if board[i] != self.TUZDUK else 0
            obs[i] = val / 81.0
        
        # Kazans (normalized)
        obs[18] = board[20] / 162.0
        obs[19] = board[21] / 162.0
        
        # Tuzduk positions (one-hot)
        if board[18] > 0:
            obs[20 + int(board[18]) - 1] = 1.0
        if board[19] > 0:
            obs[29 + int(board[19]) - 1] = 1.0
        
        # Current player
        obs[38] = 1.0 if board[22] == 0 else 0.0
        obs[39] = 1.0 if board[22] == 1 else 0.0
        
        return obs


# =============================================================================
# Neural Network - Dual Head (Policy + Value)
# =============================================================================

class AlphaZeroNetwork(nn.Module):
    """
    Dual-head neural network for AlphaZero.
    Policy head: outputs move probabilities (9 actions)
    Value head: outputs position evaluation [-1, 1]
    """
    
    def __init__(self, input_size: int = 128, hidden_size: int = 256, action_size: int = 9):
        super().__init__()
        
        # Shared backbone
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        # Policy head
        self.policy_fc = nn.Linear(hidden_size // 2, action_size)
        
        # Value head
        self.value_fc1 = nn.Linear(hidden_size // 2, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle single sample (no batch dimension)
        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)
        
        # Shared features
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        
        # Policy head (log probabilities)
        pi = self.policy_fc(x)
        pi = F.log_softmax(pi, dim=1)
        
        # Value head
        v = F.relu(self.value_fc1(x))
        v = torch.tanh(self.value_fc2(v))
        
        if single_sample:
            return pi.squeeze(0), v.squeeze(0)
        
        return pi, v


class NNetWrapper:
    """Wrapper for neural network with training and prediction methods."""
    
    def __init__(self, game: TogyzkumalakGame, config: AlphaZeroConfig):
        self.game = game
        self.config = config
        self.nnet = AlphaZeroNetwork(
            input_size=128,
            hidden_size=config.hidden_size,
            action_size=game.getActionSize()
        ).to(device)
    
    def train(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Train network on examples (board, pi, v)."""
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.config.learning_rate)
        
        self.nnet.train()
        
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        
        for epoch in range(self.config.epochs):
            batch_count = max(1, len(examples) // self.config.batch_size)
            
            for _ in range(batch_count):
                # Sample batch
                sample_ids = np.random.randint(len(examples), size=min(self.config.batch_size, len(examples)))
                boards, pis, vs = zip(*[examples[i] for i in sample_ids])
                
                # Convert to observations
                observations = [self.game.boardToObservation(b) for b in boards]
                
                # Convert to tensors
                obs_tensor = torch.FloatTensor(np.array(observations)).to(device)
                target_pis = torch.FloatTensor(np.array(pis)).to(device)
                target_vs = torch.FloatTensor(np.array(vs)).to(device)
                
                # Forward
                out_pi, out_v = self.nnet(obs_tensor)
                
                # Loss
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                l_v = torch.sum((target_vs - out_v.squeeze()) ** 2) / target_vs.size(0)
                total_loss = l_pi + l_v
                
                # Backward
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                pi_losses.update(l_pi.item(), obs_tensor.size(0))
                v_losses.update(l_v.item(), obs_tensor.size(0))
            
            log.info(f'Epoch {epoch+1}/{self.config.epochs} - Policy Loss: {pi_losses} Value Loss: {v_losses}')
        
        return {'policy_loss': pi_losses.avg, 'value_loss': v_losses.avg}
    
    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for board."""
        obs = self.game.boardToObservation(board)
        obs_tensor = torch.FloatTensor(obs).to(device)
        
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(obs_tensor)
            pi = torch.exp(pi)  # Convert log-prob to prob
        
        return pi.cpu().numpy(), float(v.cpu().numpy())
    
    def save_checkpoint(self, folder: str, filename: str):
        """Save model checkpoint."""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'config': {
                'hidden_size': self.config.hidden_size,
                'action_size': self.game.getActionSize()
            }
        }, filepath)
        log.info(f'Checkpoint saved: {filepath}')
    
    def load_checkpoint(self, folder: str, filename: str):
        """Load model checkpoint."""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint at {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        log.info(f'Checkpoint loaded: {filepath}')


# =============================================================================
# MCTS - Monte Carlo Tree Search
# =============================================================================

EPS = 1e-8

class MCTS:
    """Monte Carlo Tree Search for AlphaZero."""
    
    def __init__(self, game: TogyzkumalakGame, nnet: NNetWrapper, config: AlphaZeroConfig):
        self.game = game
        self.nnet = nnet
        self.config = config
        
        # Tree statistics
        self.Qsa = {}  # Q values for (s, a)
        self.Nsa = {}  # Visit counts for (s, a)
        self.Ns = {}   # Visit counts for s
        self.Ps = {}   # Policy from neural net for s
        
        self.Es = {}   # Game ended for s
        self.Vs = {}   # Valid moves for s
    
    def getActionProb(self, canonicalBoard: np.ndarray, temp: float = 1) -> np.ndarray:
        """
        Run MCTS simulations and return action probabilities.
        """
        for _ in range(self.config.num_mcts_sims):
            self.search(canonicalBoard)
        
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]
        
        if temp == 0:
            # Deterministic: pick best action
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_actions)
            probs = np.zeros(len(counts))
            probs[best_a] = 1
            return probs
        
        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum if counts_sum > 0 else 1.0/len(counts) for x in counts]
        return np.array(probs)
    
    def search(self, canonicalBoard: np.ndarray) -> float:
        """One iteration of MCTS."""
        s = self.game.stringRepresentation(canonicalBoard)
        
        # Check if terminal
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]
        
        # Leaf node - expand
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids
            
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                log.warning("All valid moves masked, using uniform policy")
                self.Ps[s] = valids / np.sum(valids)
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        
        # Select action with highest UCB
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
        
        v = self.search(next_s)
        
        # Backpropagate
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
        
        while self.game.getGameEnded(board, curPlayer) == 0:
            canonical = self.game.getCanonicalForm(board, curPlayer)
            action = players[curPlayer + 1](canonical)
            
            valids = self.game.getValidMoves(canonical, 1)
            if valids[action] == 0:
                log.error(f"Invalid action {action}")
                action = np.argmax(valids)
            
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        
        return curPlayer * self.game.getGameEnded(board, curPlayer)
    
    def playGames(self, num: int) -> Tuple[int, int, int]:
        """Play num games. Returns (player1 wins, player2 wins, draws)."""
        num = max(2, num)
        num_each = num // 2
        
        oneWon = 0
        twoWon = 0
        draws = 0
        
        # Player 1 starts first half
        for _ in range(num_each):
            result = self.playGame()
            if result == 1:
                oneWon += 1
            elif result == -1:
                twoWon += 1
            else:
                draws += 1
        
        # Swap and play second half
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
# Coach - Main Training Loop
# =============================================================================

class AlphaZeroCoach:
    """
    AlphaZero training coach for Togyzkumalak.
    Manages the self-play -> train -> evaluate cycle.
    """
    
    def __init__(self, game: TogyzkumalakGame, nnet: NNetWrapper, config: AlphaZeroConfig):
        self.game = game
        self.nnet = nnet
        self.pnet = NNetWrapper(game, config)  # Previous network for comparison
        self.config = config
        self.mcts = MCTS(game, nnet, config)
        
        self.trainExamplesHistory = []
        self.metrics_history = []
        
        # Status tracking
        self.current_iteration = 0
        self.status = "initialized"
        self.progress = 0.0
        self.stop_requested = False
    
    def executeEpisode(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Execute one episode of self-play."""
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0
        
        while True:
            episodeStep += 1
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
    
    def learn(self, callback=None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            callback: Optional function to call after each iteration with status dict
        
        Returns:
            Final training metrics
        """
        self.status = "running"
        
        for i in range(1, self.config.num_iterations + 1):
            if self.stop_requested:
                self.status = "stopped"
                break
            
            self.current_iteration = i
            self.progress = (i - 1) / self.config.num_iterations * 100
            
            log.info(f'=== Iteration {i}/{self.config.num_iterations} ===')
            
            # Self-play
            iterationExamples = deque([], maxlen=self.config.max_queue_length)
            
            for ep in range(self.config.num_episodes):
                if self.stop_requested:
                    break
                self.mcts = MCTS(self.game, self.nnet, self.config)  # Reset tree
                examples = self.executeEpisode()
                iterationExamples.extend(examples)
                log.info(f'  Self-play episode {ep+1}/{self.config.num_episodes}, examples: {len(examples)}')
            
            self.trainExamplesHistory.append(iterationExamples)
            
            # Trim history
            while len(self.trainExamplesHistory) > self.config.num_iters_for_history:
                self.trainExamplesHistory.pop(0)
            
            # Aggregate training examples
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
                self.nnet.save_checkpoint(self.config.checkpoint_dir, f'checkpoint_{i}.pth.tar')
                self.nnet.save_checkpoint(self.config.checkpoint_dir, 'best.pth.tar')
                accepted = True
            
            # Record metrics
            iter_metrics = {
                'iteration': i,
                'policy_loss': train_metrics['policy_loss'],
                'value_loss': train_metrics['value_loss'],
                'new_wins': nwins,
                'old_wins': pwins,
                'draws': draws,
                'win_rate': win_rate,
                'accepted': accepted,
                'total_examples': len(trainExamples),
                'timestamp': datetime.now().isoformat()
            }
            self.metrics_history.append(iter_metrics)
            
            self.progress = i / self.config.num_iterations * 100
            
            if callback:
                callback({
                    'iteration': i,
                    'total_iterations': self.config.num_iterations,
                    'progress': self.progress,
                    'metrics': iter_metrics,
                    'status': self.status
                })
        
        if not self.stop_requested:
            self.status = "completed"
        
        # Save final metrics
        self._save_metrics()
        
        return {
            'status': self.status,
            'iterations_completed': self.current_iteration,
            'metrics_history': self.metrics_history
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
                    'hidden_size': self.config.hidden_size
                },
                'metrics': self.metrics_history
            }, f, indent=2)
    
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
            'error_message': None
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
        finally:
            self.status['end_time'] = datetime.now().isoformat()
    
    def _update_status(self, data: Dict):
        """Update status from callback."""
        self.status['progress'] = data.get('progress', 0)
        self.status['current_iteration'] = data.get('iteration', 0)
        self.status['status'] = data.get('status', 'running')
        
        if 'metrics' in data:
            # Keep last 10 metrics for status
            if len(self.status['metrics']) >= 10:
                self.status['metrics'] = self.status['metrics'][-9:]
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
            num_iterations=params.get('numIters', 10),
            num_episodes=params.get('numEps', 10),
            num_mcts_sims=params.get('numMCTSSims', 25),
            cpuct=params.get('cpuct', 1.0),
            batch_size=params.get('batch_size', 32),
            hidden_size=params.get('hidden_size', 256),
            checkpoint_dir=self.checkpoint_dir
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


# Global instance
alphazero_task_manager = AlphaZeroTaskManagerV2()


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing AlphaZero Trainer for Togyzkumalak...")
    
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
    
    # Quick self-play test
    coach = AlphaZeroCoach(game, nnet, config)
    examples = coach.executeEpisode()
    print(f"Episode generated {len(examples)} examples")
    
    print("\nAlphaZero trainer ready for use!")
