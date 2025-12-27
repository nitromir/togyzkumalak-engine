"""
AI Engine for Togyzkumalak.

Provides multiple AI levels from random to neural network-based.
Level 6 uses Google Gemini LLM for gameplay.
"""

import asyncio
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import ai_config
from .game_manager import TogyzkumalakBoard


class PolicyNetwork(nn.Module):
    """Neural network policy for Togyzkumalak."""
    
    def __init__(self, input_size: int = 128, hidden_size: int = 256, output_size: int = 9):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AIEngine:
    """
    AI Engine with multiple difficulty levels.
    
    Levels:
    1. Random - Completely random legal moves
    2. Heuristic - Simple rule-based strategy
    3. Basic NN - Simple neural network
    4. Advanced NN - Deeper network with better training
    5. Expert NN - Best available model
    6. Gemini AI - Google Gemini LLM opponent
    """
    
    def __init__(self):
        self.models: Dict[int, PolicyNetwork] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model_name: str = "default"
        self.gemini_player = None  # Lazy-loaded Gemini player
        self._load_models()
    
    def _load_models(self):
        """Load available neural network models."""
        # Initialize models for levels 3-5
        for level in [3, 4, 5]:
            model = PolicyNetwork(
                input_size=128,
                hidden_size=64 * level,  # Larger for higher levels
                output_size=9
            )
            model.to(self.device)
            model.eval()
            
            # Try to load saved weights
            model_path = os.path.join(
                ai_config.model_dir,
                f"policy_level_{level}.pth"
            )
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    if level == 5:
                        self.current_model_name = f"policy_level_5"
                except Exception as e:
                    print(f"Failed to load model for level {level}: {e}")
            
            self.models[level] = model
        
        # Auto-load latest human-trained model for level 5
        self._load_latest_human_model()
    
    def _load_latest_human_model(self):
        """Auto-load the latest human-trained model for level 5."""
        try:
            models_dir = ai_config.model_dir
            if not os.path.exists(models_dir):
                return
            
            # Find latest policy_net_human_v*.pt file
            human_models = []
            for f in os.listdir(models_dir):
                if f.startswith('policy_net_human_v') and f.endswith('.pt'):
                    try:
                        version = int(f.replace('policy_net_human_v', '').replace('.pt', ''))
                        human_models.append((version, f))
                    except ValueError:
                        continue
            
            if not human_models:
                return
            
            # Sort by version and get latest
            human_models.sort(key=lambda x: x[0], reverse=True)
            latest_version, latest_file = human_models[0]
            model_path = os.path.join(models_dir, latest_file)
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.models[5].load_state_dict(checkpoint['model_state_dict'])
                accuracy = checkpoint.get('final_accuracy', 'N/A')
                print(f"[OK] Auto-loaded human model v{latest_version} (Accuracy: {accuracy}%)")
            else:
                self.models[5].load_state_dict(checkpoint)
                print(f"[OK] Auto-loaded human model v{latest_version}")
            
            self.current_model_name = latest_file.replace('.pt', '')
                
        except Exception as e:
            print(f"[WARNING] Could not auto-load human model: {e}")
    
    def get_move(
        self,
        board: TogyzkumalakBoard,
        level: int = 3,
        thinking_time_ms: int = None
    ) -> Tuple[int, int]:
        """
        Get AI move for the current position.
        
        Args:
            board: Current board state
            level: AI difficulty level (1-5)
            thinking_time_ms: Optional thinking time limit
        
        Returns:
            Tuple of (move (1-9), actual_thinking_time_ms)
        """
        start_time = time.time()
        
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return 0, 0
        
        if level == 1:
            move = self._random_move(legal_moves)
        elif level == 2:
            move = self._heuristic_move(board, legal_moves)
        elif level == 6:
            move = self._gemini_move(board, legal_moves)
        else:
            move = self._neural_move(board, legal_moves, level)
        
        # Add small delay to simulate thinking
        elapsed = (time.time() - start_time) * 1000
        if thinking_time_ms and elapsed < thinking_time_ms:
            time.sleep((thinking_time_ms - elapsed) / 1000)
            elapsed = thinking_time_ms
        
        return move + 1, int(elapsed)  # Convert to 1-based
    
    def _random_move(self, legal_moves: List[int]) -> int:
        """Level 1: Random move selection."""
        return random.choice(legal_moves)
    
    def _heuristic_move(self, board: TogyzkumalakBoard, legal_moves: List[int]) -> int:
        """
        Level 2: Heuristic-based move selection.
        
        Strategy:
        1. Prefer moves that create tuzduk
        2. Prefer moves that capture
        3. Prefer moves with more kumalaks
        4. Avoid moves that leave opponent with capture
        """
        current_player = board.current_player
        scores = {}
        
        for move in legal_moves:
            score = 0
            pit_index = move + (0 if current_player == "white" else 9)
            kumalaks = board.fields[pit_index]
            
            # More kumalaks = generally better control
            score += kumalaks * 0.1
            
            # Simulate the move to see outcome
            test_board = TogyzkumalakBoard(board.fields.copy())
            success, notation = test_board.make_move(move + 1)
            
            if success:
                # Check for tuzduk creation
                if 'x' in notation:
                    score += 50  # Tuzduk is very valuable
                
                # Check kazan gain
                if current_player == "white":
                    gain = test_board.white_kazan - board.white_kazan
                else:
                    gain = test_board.black_kazan - board.black_kazan
                score += gain * 2
                
                # Penalize if opponent can capture after
                opponent_legal = test_board.get_legal_moves()
                for opp_move in opponent_legal:
                    opp_test = TogyzkumalakBoard(test_board.fields.copy())
                    opp_test.make_move(opp_move + 1)
                    if current_player == "white":
                        opp_gain = opp_test.black_kazan - test_board.black_kazan
                    else:
                        opp_gain = opp_test.white_kazan - test_board.white_kazan
                    if opp_gain > 10:
                        score -= opp_gain * 0.5
            
            scores[move] = score
        
        # Choose move with highest score (with small randomness)
        best_moves = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(best_moves) > 1 and random.random() < 0.1:
            return best_moves[1][0]  # Occasionally pick second best
        return best_moves[0][0]
    
    def _neural_move(
        self,
        board: TogyzkumalakBoard,
        legal_moves: List[int],
        level: int
    ) -> int:
        """Level 3-5: Neural network-based move selection."""
        model = self.models.get(level)
        if not model:
            return self._heuristic_move(board, legal_moves)
        
        # Get observation
        obs = board.to_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = model(obs_tensor).squeeze().cpu().numpy()
        
        # Mask illegal moves
        mask = np.zeros(9)
        for move in legal_moves:
            mask[move] = 1
        
        masked_probs = action_probs * mask
        
        if masked_probs.sum() == 0:
            # Fallback to random if all legal moves have 0 probability
            return random.choice(legal_moves)
        
        masked_probs = masked_probs / masked_probs.sum()
        
        # Epsilon-greedy based on level
        epsilon = ai_config.epsilon / level  # Lower epsilon for higher levels
        
        if random.random() < epsilon:
            return random.choice(legal_moves)
        
        # Sample from distribution (stochastic) or argmax (deterministic)
        if level < 5:
            return int(np.random.choice(9, p=masked_probs))
        else:
            return int(np.argmax(masked_probs))
    
    def _gemini_move(
        self,
        board: TogyzkumalakBoard,
        legal_moves: List[int]
    ) -> int:
        """Level 6: Gemini LLM-based move selection."""
        try:
            # Lazy-load Gemini player
            if self.gemini_player is None:
                from .gemini_battle import GeminiPlayer
                self.gemini_player = GeminiPlayer()
            
            if not self.gemini_player.is_available():
                print("[WARNING] Gemini not available, falling back to heuristic")
                return self._heuristic_move(board, legal_moves)
            
            # Determine current player color
            current_player = board.current_player
            
            # Get move from Gemini (async call in sync context)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # We're already in an async context, use nest_asyncio pattern
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        lambda: asyncio.run(
                            self.gemini_player.get_move(board, current_player, timeout=30)
                        )
                    )
                    move, explanation = future.result(timeout=35)
            else:
                # Simple case - run async in new loop
                move, explanation = loop.run_until_complete(
                    self.gemini_player.get_move(board, current_player, timeout=30)
                )
            
            print(f"[Gemini] Move: {move}, Explanation: {explanation}")
            return move - 1  # Convert to 0-indexed
            
        except Exception as e:
            print(f"[ERROR] Gemini move failed: {e}, falling back to heuristic")
            return self._heuristic_move(board, legal_moves)
    
    def get_move_probabilities(
        self,
        board: TogyzkumalakBoard,
        level: int = 3
    ) -> Dict[int, float]:
        """Get move probabilities for visualization."""
        model = self.models.get(level)
        if not model:
            legal_moves = board.get_legal_moves()
            return {m: 1.0 / len(legal_moves) for m in legal_moves}
        
        obs = board.to_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = model(obs_tensor).squeeze().cpu().numpy()
        
        legal_moves = board.get_legal_moves()
        mask = np.zeros(9)
        for move in legal_moves:
            mask[move] = 1
        
        masked_probs = action_probs * mask
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        
        return {i: float(masked_probs[i]) for i in range(9)}
    
    def evaluate_position(self, board: TogyzkumalakBoard) -> float:
        """
        Evaluate current position.
        
        Returns a value between -1 and 1:
        - Positive: White advantage
        - Negative: Black advantage
        - 0: Equal
        """
        white_kazan = board.white_kazan
        black_kazan = board.black_kazan
        
        # Simple material-based evaluation
        white_pits_total = sum(
            board.fields[i] for i in range(9) 
            if board.fields[i] != board.TUZDUK
        )
        black_pits_total = sum(
            board.fields[i] for i in range(9, 18)
            if board.fields[i] != board.TUZDUK
        )
        
        # Kazan score (most important)
        kazan_diff = (white_kazan - black_kazan) / 162.0
        
        # Tuzduk bonus
        tuzduk_bonus = 0
        if board.fields[18] > 0:  # White has tuzduk
            tuzduk_bonus += 0.1
        if board.fields[19] > 0:  # Black has tuzduk
            tuzduk_bonus -= 0.1
        
        # Material on board
        material_diff = (white_pits_total - black_pits_total) / 162.0 * 0.3
        
        evaluation = kazan_diff + tuzduk_bonus + material_diff
        return max(-1.0, min(1.0, evaluation))
    
    def save_model(self, level: int, path: str = None):
        """Save model weights."""
        model = self.models.get(level)
        if not model:
            return
        
        if not path:
            os.makedirs(ai_config.model_dir, exist_ok=True)
            path = os.path.join(ai_config.model_dir, f"policy_level_{level}.pth")
        
        torch.save(model.state_dict(), path)
    
    def get_policy_output(
        self,
        board: TogyzkumalakBoard,
        level: int = 5
    ) -> Tuple[List[float], List[float], float]:
        """
        Get raw policy output for training data collection.
        
        Returns:
            Tuple of (raw_logits, action_probs, value_estimate)
            - raw_logits: Raw network output before softmax (for KL regularization)
            - action_probs: Probabilities after softmax
            - value_estimate: Position evaluation [-1, 1]
        """
        model = self.models.get(level)
        
        if not model:
            # Fallback uniform distribution
            legal_moves = board.get_legal_moves()
            uniform_logits = [0.0] * 9
            uniform_probs = [0.0] * 9
            for m in legal_moves:
                uniform_probs[m] = 1.0 / len(legal_moves)
            value = self.evaluate_position(board)
            return uniform_logits, uniform_probs, value
        
        # Get observation
        obs = board.to_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get intermediate activations before softmax for logits
            # Since our network has softmax built-in, we need to extract before that
            x = obs_tensor
            for i, layer in enumerate(model.network):
                if isinstance(layer, nn.Softmax):
                    # This is the layer before softmax - raw logits
                    raw_logits = x.squeeze().cpu().numpy().tolist()
                    action_probs = layer(x).squeeze().cpu().numpy().tolist()
                    break
                x = layer(x)
            else:
                # Fallback if no softmax found
                action_probs = model(obs_tensor).squeeze().cpu().numpy().tolist()
                raw_logits = [0.0] * 9
        
        # Get position evaluation
        value = self.evaluate_position(board)
        
        return raw_logits, action_probs, value
    
    def get_move_with_policy(
        self,
        board: TogyzkumalakBoard,
        level: int = 5,
        thinking_time_ms: int = None
    ) -> Tuple[int, int, List[float], List[float], float]:
        """
        Get AI move along with full policy output for training.
        
        Returns:
            Tuple of (move, thinking_time_ms, raw_logits, action_probs, value_estimate)
        """
        start_time = time.time()
        
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return 0, 0, [], [], 0.0
        
        # Get policy output first
        raw_logits, action_probs, value = self.get_policy_output(board, level)
        
        # Select move based on level
        if level == 1:
            move = self._random_move(legal_moves)
        elif level == 2:
            move = self._heuristic_move(board, legal_moves)
        else:
            # Use action_probs directly
            mask = np.zeros(9)
            for m in legal_moves:
                mask[m] = 1
            
            masked_probs = np.array(action_probs) * mask
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                masked_probs = mask / mask.sum()
            
            epsilon = ai_config.epsilon / level
            if random.random() < epsilon:
                move = random.choice(legal_moves)
            elif level < 5:
                move = int(np.random.choice(9, p=masked_probs))
            else:
                move = int(np.argmax(masked_probs))
        
        elapsed = (time.time() - start_time) * 1000
        if thinking_time_ms and elapsed < thinking_time_ms:
            time.sleep((thinking_time_ms - elapsed) / 1000)
            elapsed = thinking_time_ms
        
        return move + 1, int(elapsed), raw_logits, action_probs, value


# Global AI engine instance
ai_engine = AIEngine()

