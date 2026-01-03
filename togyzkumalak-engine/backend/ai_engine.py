"""
AI Engine for Togyzkumalak.

Provides multiple AI levels from random to neural network-based.
Level 6 uses Google Gemini LLM for gameplay.
Level 7 uses PROBS (Predict Result of Beam Search) algorithm.
"""

import asyncio
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import ai_config
from .game_manager import TogyzkumalakBoard

# Add PROBS path for imports
PROBS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 
    "../../../probs-main/python_impl_generic"))
if not os.path.exists(PROBS_PATH):
    # Try one more level just in case of different deploy structures
    PROBS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 
        "../../../../probs-main/python_impl_generic"))

if PROBS_PATH not in sys.path:
    sys.path.insert(0, PROBS_PATH)


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
    7. PROBS AI - Predict Result of Beam Search algorithm
    """
    
    def __init__(self):
        self.models: Dict[int, PolicyNetwork] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model_name: str = "default"
        self.gemini_player = None  # Lazy-loaded Gemini player
        self.alphazero_model = None  # AlphaZero network for MCTS-based play
        self.use_mcts = False  # Whether to use MCTS with AlphaZero model
        self.mcts_cache = {}  # Cache for MCTS objects per level
        self.probs_agent = None  # PROBS agent for level 7
        self.probs_model_keeper = None  # PROBS model keeper
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
        
        # Load AlphaZero model if available (only if not already loaded via training_manager)
        # This is a fallback - the model should be loaded via training_manager when user selects a checkpoint
        az_path = os.path.join(ai_config.model_dir, "alphazero", "best.pth.tar")
        if os.path.exists(az_path) and self.alphazero_model is None:
            try:
                from .alphazero_trainer import TogyzkumalakGame, AlphaZeroNetwork
                game = TogyzkumalakGame()
                az_net = AlphaZeroNetwork(128, 256, 9) # Default params
                checkpoint = torch.load(az_path, map_location=self.device)
                az_net.load_state_dict(checkpoint['state_dict'])
                az_net.to(self.device)
                az_net.eval()
                self.alphazero_model = az_net
                print("[AI] Loaded AlphaZero model from best.pth.tar (fallback)")
            except Exception as e:
                print(f"[AI] Failed to load AlphaZero model: {e}")

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
    
    def get_model_info(self, level: int = 5) -> Dict:
        """Get information about the currently active model for a given level."""
        # Special case for PROBS
        if level == 7:
            return {
                "name": "PROBS",
                "level": 7,
                "type": "probs",
                "architecture": "PROBS (Value + Q-Value)",
                "use_mcts": False,
                "has_probs": self.probs_model_keeper is not None
            }
        
        model = self.models.get(level)
        model_type = "default"
        architecture = "PolicyNetwork"
        
        if model:
            # Check if it's an AlphaZero model
            if hasattr(model, 'net') and hasattr(model.net, 'policy_fc'):
                model_type = "alphazero"
                architecture = "AlphaZeroNetwork"
            elif hasattr(model, 'network'):
                model_type = "gym"
                architecture = "PolicyNetwork"
        
        return {
            "name": self.current_model_name,
            "level": level,
            "type": model_type,
            "architecture": architecture,
            "use_mcts": self.use_mcts and model_type == "alphazero",
            "has_alphazero": self.alphazero_model is not None,
            "has_probs": self.probs_model_keeper is not None
        }
    
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
        elif level == 7:
            move = self._probs_move(board, legal_moves)
        else:
            move = self._neural_move(board, legal_moves, level)
        
        # Add small delay to simulate thinking
        elapsed = (time.time() - start_time) * 1000
        if thinking_time_ms and elapsed < thinking_time_ms:
            time.sleep((thinking_time_ms - elapsed) / 1000)
            elapsed = thinking_time_ms
        
        return int(move) + 1, int(elapsed)  # Convert to 1-based, ensure Python int
    
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
        # Use AlphaZero MCTS for any neural level if model is loaded and MCTS is enabled
        # Level 5 ALWAYS uses the selected AlphaZero checkpoint from Training tab
        if self.alphazero_model is not None and self.use_mcts:
            if level == 5:
                # Log which model is being used for Level 5
                model_name = getattr(self, 'current_model_name', 'unknown')
                if not hasattr(self, '_last_logged_model') or self._last_logged_model != model_name:
                    print(f"[AI Level 5] Using AlphaZero model: {model_name}")
                    self._last_logged_model = model_name
            return self._alphazero_mcts_move(board, legal_moves, level)
            
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
    
    def _alphazero_mcts_move(
        self,
        board: TogyzkumalakBoard,
        legal_moves: List[int],
        level: int = 5
    ) -> int:
        """AlphaZero move selection with MCTS."""
        try:
            from .alphazero_trainer import MCTS, TogyzkumalakGame, AlphaZeroConfig
            
            # Use cached MCTS or create new one
            if level not in self.mcts_cache:
                game = TogyzkumalakGame()
                
                # Scale simulations by level and GPU availability
                if level == 3:
                    num_sims = 50 if torch.cuda.is_available() else 20
                elif level == 4:
                    num_sims = 150 if torch.cuda.is_available() else 50
                else:  # Level 5
                    num_sims = 400 if torch.cuda.is_available() else 100
                
                print(f"[AI] Initializing MCTS for Level {level} with {num_sims} simulations")
                config = AlphaZeroConfig(num_mcts_sims=num_sims)  
                
                # Create wrapper for existing model
                class QuickNNetWrapper:
                    def __init__(self, model, game, device):
                        self.model = model
                        self.game = game
                        self.device = device
                    
                    def predict(self, board):
                        obs = self.game.boardToObservation(board)
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            pi, v = self.model(obs_tensor)
                            pi = torch.exp(pi)
                        return pi.squeeze().cpu().numpy(), float(v.squeeze().cpu().numpy())
                
                nnet = QuickNNetWrapper(self.alphazero_model, game, self.device)
                self.mcts_cache[level] = MCTS(game, nnet, config)
            
            mcts = self.mcts_cache[level]
            game = mcts.game
            
            # Convert board to AlphaZero format
            az_board = np.array(board.fields, dtype=np.float32)
            player = 1 if board.current_player == "white" else -1
            canonical = game.getCanonicalForm(az_board, player)
            
            # Get action probabilities from MCTS
            # Use small temperature for some variety in Level 3-4, 0 for Level 5
            temp = 0.5 if level < 5 else 0
            action_probs = mcts.getActionProb(canonical, temp=temp)
            
            # Select best legal move
            best_move = -1
            best_prob = -1
            for move in legal_moves:
                if action_probs[move] > best_prob:
                    best_prob = action_probs[move]
                    best_move = move
            
            return best_move if best_move >= 0 else random.choice(legal_moves)
            
        except Exception as e:
            print(f"[WARNING] AlphaZero MCTS failed: {e}, using policy directly")
            import traceback
            traceback.print_exc()
            # Fallback to policy-only
            model = self.models.get(level) or self.models.get(5)
            if model:
                obs = board.to_observation()
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs = model(obs_tensor).squeeze().cpu().numpy()
                mask = np.zeros(9)
                for move in legal_moves:
                    mask[move] = 1
                masked_probs = action_probs * mask
                if masked_probs.sum() > 0:
                    return int(np.argmax(masked_probs))
            return random.choice(legal_moves)
    
    def _gemini_move(
        self,
        board: TogyzkumalakBoard,
        legal_moves: List[int]
    ) -> int:
        """Level 6: Gemini LLM-based move selection.
        ALWAYS uses Google Gemini API (not a checkpoint)."""
        try:
            # Lazy-load Gemini player
            if self.gemini_player is None:
                from .gemini_battle import GeminiPlayer
                self.gemini_player = GeminiPlayer()
            
            if not self.gemini_player.is_available():
                print("[WARNING] Level 6: Gemini API not available (GEMINI_API_KEY not set), falling back to heuristic")
                return self._heuristic_move(board, legal_moves)
            
            # Log that we're using Gemini API (only once)
            if not hasattr(self, '_gemini_logged'):
                print("[AI Level 6] Using Google Gemini API for move selection")
                self._gemini_logged = True
            
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
    
    def _probs_move(
        self,
        board: TogyzkumalakBoard,
        legal_moves: List[int]
    ) -> int:
        """Level 7: PROBS (Predict Result of Beam Search) move selection with Beam Search.
        ALWAYS uses the selected PROBS checkpoint from Training tab."""
        try:
            # Lazy-load PROBS components
            if self.probs_model_keeper is None:
                self._load_probs_model()
            
            if self.probs_model_keeper is None:
                print("[WARNING] PROBS model not available, falling back to heuristic")
                return self._heuristic_move(board, legal_moves)
            
            # Log which PROBS checkpoint is being used (if available)
            if hasattr(self.probs_model_keeper, '_checkpoint_name'):
                checkpoint_name = getattr(self.probs_model_keeper, '_checkpoint_name', 'unknown')
                if not hasattr(self, '_last_logged_probs') or self._last_logged_probs != checkpoint_name:
                    print(f"[AI Level 7] Using PROBS checkpoint: {checkpoint_name}")
                    self._last_logged_probs = checkpoint_name
            
            # Import PROBS components
            original_cwd = os.getcwd()
            os.chdir(PROBS_PATH)
            
            try:
                from probs_impl import probs_impl_common
                from environments.togyzkumalak_env import TogyzkumalakEnv
                
                # Check if we need to create/update the agent
                # We use SelfLearningAgent_TreeScan for deep search (Level 7)
                if self.probs_agent is None or not isinstance(self.probs_agent, probs_impl_common.SelfLearningAgent_TreeScan):
                    # Configure search parameters - make Level 7 strong
                    # 500 nodes and 2.0s for maximum strength
                    self.probs_agent = probs_impl_common.SelfLearningAgent_TreeScan(
                        "PROBS_Level7",
                        model_keeper=self.probs_model_keeper,
                        device=str(self.device),
                        action_time_budget=2.0,  # 2.0 seconds per move
                        expand_tree_budget=500,  # Expand up to 500 nodes
                        batch_size=32            # Batch size for NN inference
                    )
                
                # Create PROBS environment from current board state
                env = TogyzkumalakEnv()
                env.board = np.array(board.fields, dtype=np.float32)
                # Ensure correct player encoding (1 for white, -1 for black)
                env.player = 1 if board.current_player == "white" else -1
                env.done = False
                
                # Use the agent to get action via Beam Search
                move = self.probs_agent.get_action(env)
                
                # Validate move
                if move not in legal_moves:
                    print(f"[PROBS] Agent suggested illegal move {move}, choosing best legal")
                    # Fallback to one-shot if search failed or suggested illegal
                    q_values = probs_impl_common.get_q_a_single_state(
                        self.probs_model_keeper.models['self_learner'],
                        env,
                        str(self.device)
                    )
                    mask = np.zeros(9)
                    for m in legal_moves: mask[m] = 1
                    move = int(np.argmax(q_values * mask + (1 - mask) * (-1e9)))

                print(f"[PROBS] Search Move: {move} (Nodes: {self.probs_agent.last_search_nodes_cnt}, Time: {self.probs_agent.last_search_time:.2f}s)")
                return int(move)  # Ensure Python int for JSON serialization
                
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"[ERROR] PROBS move search failed: {e}, falling back to heuristic")
            import traceback
            traceback.print_exc()
            return self._heuristic_move(board, legal_moves)
            
        except Exception as e:
            print(f"[ERROR] PROBS move failed: {e}, falling back to heuristic")
            import traceback
            traceback.print_exc()
            return self._heuristic_move(board, legal_moves)
    
    def _load_probs_model(self):
        """Load PROBS model from checkpoint."""
        try:
            from .probs_task_manager import probs_task_manager
            
            # Check if model is already loaded in task manager
            if probs_task_manager.is_model_loaded():
                self.probs_model_keeper = probs_task_manager.get_loaded_model()
                print("[PROBS] Using model from task manager")
                return
            
            # Try to load best or latest checkpoint
            checkpoints = probs_task_manager.get_checkpoints()
            if checkpoints:
                # Always take the first one (latest) as it is most likely to exist and be valid
                target = checkpoints[0]
                
                if target and os.path.exists(target['path']):
                    if probs_task_manager.load_checkpoint(target['path'], str(self.device)):
                        self.probs_model_keeper = probs_task_manager.get_loaded_model()
                        print(f"[PROBS] Loaded checkpoint: {target['filename']}")
                        return
            
            print("[PROBS] No valid checkpoint available")
            
        except Exception as e:
            print(f"[ERROR] Failed to load PROBS model: {e}")
            import traceback
            traceback.print_exc()
    
    def get_move_probabilities(
        self,
        board: TogyzkumalakBoard,
        level: int = 3,
        model_type: str = None
    ) -> Dict[int, float]:
        """Get move probabilities for visualization."""

        # ENSEMBLE MODE: Combine all available models
        if model_type == 'ensemble' or model_type == 'combined':
            return self._get_ensemble_probabilities(board)

        # 1. GEMINI
        if model_type == 'gemini' or (model_type is None and level == 6):
            try:
                from .gemini_analyzer import gemini_analyzer
                if not gemini_analyzer.is_available():
                    legal_moves = board.get_legal_moves()
                    return {m: 1.0 / len(legal_moves) if len(legal_moves) > 0 else 0.0 for m in range(9)}
                
                # Use current event loop if available
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                board_state = board.get_state_dict()
                board_state["legal_moves"] = board.get_legal_moves()
                
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            lambda: asyncio.run(gemini_analyzer.get_move_probabilities(board_state))
                        )
                        return future.result(timeout=30)
                else:
                    return loop.run_until_complete(gemini_analyzer.get_move_probabilities(board_state))
            except Exception as e:
                print(f"[ERROR] Gemini probabilities failed: {e}")
                legal_moves = board.get_legal_moves()
                return {m: 1.0 / len(legal_moves) if len(legal_moves) > 0 else 0.0 for m in range(9)}

        # 2. PROBS
        if model_type == 'probs' or (model_type is None and level == 7):
            try:
                if self.probs_model_keeper is None:
                    self._load_probs_model()
                
                if self.probs_model_keeper is not None:
                    original_cwd = os.getcwd()
                    os.chdir(PROBS_PATH)
                    try:
                        from probs_impl import probs_impl_common
                        from environments.togyzkumalak_env import TogyzkumalakEnv
                        
                        env = TogyzkumalakEnv()
                        env.board = np.array(board.fields, dtype=np.float32)
                        env.player = 1 if board.current_player == "white" else -1
                        
                        q_values = probs_impl_common.get_q_a_single_state(
                            self.probs_model_keeper.models['self_learner'],
                            env,
                            str(self.device)
                        )
                        
                        legal_moves = board.get_legal_moves()
                        mask = np.zeros(9)
                        for m in legal_moves:
                            mask[m] = 1
                        
                        # Softmax for probabilities
                        # Using lower temperature to make differences more visible
                        # Standard deviation of Q-values is often small, so we need high sensitivity
                        temperature = 0.05
                        exp_q = np.exp((q_values - np.max(q_values)) / temperature)
                        masked_exp_q = exp_q * mask
                        
                        if masked_exp_q.sum() > 0:
                            probs = masked_exp_q / masked_exp_q.sum()
                            return {i: float(probs[i]) for i in range(9)}
                    finally:
                        os.chdir(original_cwd)
            except Exception as e:
                print(f"[ERROR] PROBS probabilities failed: {e}")
            
            # Fallback if specific PROBS request failed
            if model_type == 'probs':
                legal_moves = board.get_legal_moves()
                return {m: 1.0 / len(legal_moves) if len(legal_moves) > 0 else 0.0 for m in range(9)}

        # 3. ALPHAZERO (MCTS)
        if model_type == 'alphazero' or (self.alphazero_model is not None and self.use_mcts and model_type is None):
            try:
                from .alphazero_trainer import TogyzkumalakGame
                game = TogyzkumalakGame()
                az_board = np.array(board.fields, dtype=np.float32)
                player = 1 if board.current_player == "white" else -1
                canonical = game.getCanonicalForm(az_board, player)
                
                # Use level 5 sims for analysis
                self._alphazero_mcts_move(board, board.get_legal_moves(), 5)
                mcts = self.mcts_cache.get(5)
                
                if mcts:
                    action_probs = mcts.getActionProb(canonical, temp=1.0)
                    return {i: float(action_probs[i]) for i in range(9)}
            except Exception as e:
                print(f"[WARNING] AlphaZero MCTS probabilities failed: {e}")

        # 4. POLYNET / GYM MODELS (Default)
        target_level = level
        # If model_type is explicitly polynet, use level 5 if target level not found
        # Or if we fell through from level 7
        if model_type == 'polynet' or level == 7:
            target_level = 5
            
        model = self.models.get(target_level) or self.models.get(5)
        
        if not model:
            legal_moves = board.get_legal_moves()
            return {m: 1.0 / len(legal_moves) if len(legal_moves) > 0 else 0.0 for m in range(9)}
        
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
        else:
            return {m: 1.0 / len(legal_moves) if len(legal_moves) > 0 else 0.0 for m in range(9)}
        
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
            # Check if this is an AlphaZero model (has 'net' attribute instead of 'network')
            if hasattr(model, 'net') and hasattr(model.net, 'policy_fc'):
                # AlphaZero dual-head model
                pi, v = model.net(obs_tensor)
                action_probs = torch.exp(pi).squeeze().cpu().numpy().tolist()
                raw_logits = pi.squeeze().cpu().numpy().tolist()  # Log-probs
                value = float(v.squeeze().cpu().numpy())
                return raw_logits, action_probs, value
            elif hasattr(model, 'network'):
                # Standard PolicyNetwork with sequential layers
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
            else:
                # Generic model - just use forward pass
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

    def _get_ensemble_probabilities(self, board: TogyzkumalakBoard) -> Dict[int, float]:
        """
        Combine predictions from multiple models using ensemble learning.

        Strategy:
        - Get probabilities from Polynet, AlphaZero, PROBS
        - Use weighted average with confidence-based weights
        - Polynet: 0.3 (fast baseline)
        - AlphaZero: 0.4 (strategic depth)
        - PROBS: 0.3 (tactical search)
        """
        probabilities = {}
        weights = {}
        total_weight = 0

        # 1. Get Polynet probabilities (fast baseline)
        try:
            polynet_probs = self.get_move_probabilities(board, model_type='polynet')
            if polynet_probs and any(v > 0.1 for v in polynet_probs.values()):
                probabilities['polynet'] = polynet_probs
                weights['polynet'] = 0.3
                total_weight += 0.3
        except Exception as e:
            print(f"[ENSEMBLE] Polynet failed: {e}")

        # 2. Get AlphaZero probabilities (strategic)
        try:
            if self.alphazero_model is not None:
                alphazero_probs = self.get_move_probabilities(board, model_type='alphazero')
                if alphazero_probs and any(v > 0.1 for v in alphazero_probs.values()):
                    probabilities['alphazero'] = alphazero_probs
                    weights['alphazero'] = 0.4
                    total_weight += 0.4
        except Exception as e:
            print(f"[ENSEMBLE] AlphaZero failed: {e}")

        # 3. Get PROBS probabilities (tactical)
        try:
            probs_probs = self.get_move_probabilities(board, model_type='probs')
            if probs_probs and any(v > 0.1 for v in probs_probs.values()):
                probabilities['probs'] = probs_probs
                weights['probs'] = 0.3
                total_weight += 0.3
        except Exception as e:
            print(f"[ENSEMBLE] PROBS failed: {e}")

        # If no models available, return uniform distribution
        if not probabilities:
            legal_moves = board.get_legal_moves()
            return {m: 1.0 / len(legal_moves) if len(legal_moves) > 0 else 0.0 for m in range(9)}

        # Combine probabilities using weighted average
        combined_probs = {i: 0.0 for i in range(9)}

        for model_name, probs in probabilities.items():
            weight = weights[model_name]
            for move_idx in range(9):
                combined_probs[move_idx] += probs.get(move_idx, 0.0) * weight

        # Normalize to ensure sum = 1
        total = sum(combined_probs.values())
        if total > 0:
            combined_probs = {k: v / total for k, v in combined_probs.items()}

        print(f"[ENSEMBLE] Combined {len(probabilities)} models: {list(probabilities.keys())}")
        return combined_probs


# Global AI engine instance
ai_engine = AIEngine()
