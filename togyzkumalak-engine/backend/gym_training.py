"""
Gym Training Manager for Togyzkumalak
Handles training sessions, model management, and replay generation.
"""

import gym
import gym_togyzkumalak
import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TrainingConfig:
    """Configuration for a training session."""
    num_games: int = 10
    epsilon: float = 0.2
    hidden_size: int = 64
    learning_rate: float = 0.001
    save_replays: bool = True
    model_name: str = "policy_net"


@dataclass
class TrainingProgress:
    """Progress information for a training session."""
    session_id: str
    status: str  # "running", "paused", "completed", "error"
    games_completed: int
    total_games: int
    white_wins: int
    black_wins: int
    draws: int
    avg_steps: float
    current_game_step: int = 0
    error_message: Optional[str] = None
    model_info: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class PolicyNetwork(nn.Module):
    """MLP Policy Network for Togyzkumalak."""
    def __init__(self, input_size=128, hidden_size=64, output_size=9):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.network(x)


class GymTrainingManager:
    """Manages gym training sessions and model persistence."""
    
    def __init__(self, models_dir: str = "models", replays_dir: str = "../visualizer"):
        self.models_dir = Path(models_dir)
        self.replays_dir = Path(replays_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.replays_dir.mkdir(exist_ok=True)
        
        self.active_sessions: Dict[str, TrainingProgress] = {}
        self.env = None
        self.policy_net = None
    
    def create_session(self, config: TrainingConfig) -> str:
        """Create a new training session."""
        session_id = f"train_{int(time.time())}"
        
        progress = TrainingProgress(
            session_id=session_id,
            status="initialized",
            games_completed=0,
            total_games=config.num_games,
            white_wins=0,
            black_wins=0,
            draws=0,
            avg_steps=0.0,
            start_time=datetime.now().isoformat()
        )
        
        self.active_sessions[session_id] = progress
        return session_id
    
    def get_session_progress(self, session_id: str) -> Optional[Dict]:
        """Get progress of a training session."""
        progress = self.active_sessions.get(session_id)
        return asdict(progress) if progress else None
    
    def list_sessions(self) -> List[Dict]:
        """List all active and recent sessions."""
        return [asdict(p) for p in self.active_sessions.values()]
    
    def _init_environment(self):
        """Initialize gym environment."""
        if self.env is None:
            self.env = gym.make('Togyzkumalak-v0')
    
    def _init_network(self, hidden_size: int = 64):
        """Initialize or reset policy network."""
        self.policy_net = PolicyNetwork(
            input_size=128,
            hidden_size=hidden_size,
            output_size=9
        )
    
    def _select_action(self, observation, available_actions, epsilon=0.2):
        """Select action using policy network with epsilon-greedy."""
        obs_flat = observation.flatten().astype(np.float32)
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(obs_tensor).squeeze().numpy()
        
        masked_probs = action_probs * np.array(available_actions)
        
        if masked_probs.sum() == 0:
            masked_probs = np.array(available_actions, dtype=np.float32)
        
        masked_probs = masked_probs / masked_probs.sum()
        
        if np.random.random() < epsilon:
            valid_actions = [i for i, a in enumerate(available_actions) if a == 1]
            return np.random.choice(valid_actions) if valid_actions else 0
        
        action = np.random.choice(9, p=masked_probs)
        return action
    
    def _get_board_state(self):
        """Extract board state from environment."""
        board = self.env.board
        
        # Find tuzduk positions by checking which otau has tuzduk=True
        white_tuzduk = None
        black_tuzduk = None
        for i in range(9):
            if board.gamers['black'].home[i].tuzduk:  # White's tuzduk is on black's side
                white_tuzduk = i
            if board.gamers['white'].home[i].tuzduk:  # Black's tuzduk is on white's side
                black_tuzduk = i
        
        return {
            "white_pits": [int(board.gamers['white'].home[i].kumalaks) for i in range(9)],
            "black_pits": [int(board.gamers['black'].home[i].kumalaks) for i in range(9)],
            "white_kazan": int(board.gamers['white'].kazan.score),
            "black_kazan": int(board.gamers['black'].kazan.score),
            "white_tuzduk": white_tuzduk,
            "black_tuzduk": black_tuzduk,
            "current_player": board.run.name
        }
    
    def _run_single_game(self, game_id: int, epsilon: float, record_states: bool = False):
        """Run a single game and optionally record states."""
        obs = self.env.reset()
        done = False
        steps = 0
        states = []
        
        if record_states:
            initial_state = self._get_board_state()
            initial_state["step"] = 0
            initial_state["action"] = None
            initial_state["player"] = initial_state["current_player"]
            states.append(initial_state)
        
        while not done:
            available = self.env.available_action()
            current_player = self.env.board.run.name
            action = self._select_action(obs, available, epsilon)
            
            try:
                obs, reward, done, info = self.env.step(action)
                steps += 1
                
                if record_states:
                    state = self._get_board_state()
                    state["step"] = steps
                    state["action"] = int(action)
                    state["player"] = current_player
                    states.append(state)
                    
            except Exception as e:
                print(f"Error in game {game_id} at step {steps}: {e}")
                done = True
                break
        
        # Determine winner
        white_kazan = self.env.board.gamers['white'].kazan.score
        black_kazan = self.env.board.gamers['black'].kazan.score
        
        if white_kazan > 81:
            winner = "WHITE"
        elif black_kazan > 81:
            winner = "BLACK"
        elif white_kazan == 81 and black_kazan == 81:
            winner = "DRAW"
        else:
            winner = "WHITE" if white_kazan > black_kazan else "BLACK"
        
        result = {
            "game_id": game_id,
            "steps": steps,
            "winner": winner,
            "final_score": {
                "white": white_kazan,
                "black": black_kazan
            }
        }
        
        if record_states:
            result["timestamp"] = datetime.now().isoformat()
            result["total_steps"] = steps
            result["states"] = states
        
        return result
    
    async def run_training_session(self, session_id: str, config: TrainingConfig):
        """Run a complete training session."""
        progress = self.active_sessions.get(session_id)
        if not progress:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            progress.status = "running"
            
            # Initialize environment and network
            self._init_environment()
            
            # Check if we should use existing network or create new one
            # If current network exists and its hidden_size matches config, keep it
            should_init_new = True
            if self.policy_net is not None:
                # Get current hidden_size from the first layer
                try:
                    current_hidden = next(self.policy_net.parameters()).shape[0]
                    if current_hidden == config.hidden_size:
                        msg = f"Используется существующая сеть (hidden_size={current_hidden})"
                        print(f"[Training] {msg}")
                        progress.model_info = msg
                        should_init_new = False
                    else:
                        msg = f"Размер не совпал (текущий {current_hidden} != {config.hidden_size}), создана новая сеть"
                        print(f"[Training] {msg}")
                        progress.model_info = msg
                except Exception as e:
                    print(f"[Training] Could not determine current hidden size: {e}")
            
            if should_init_new:
                if progress.model_info is None:
                    progress.model_info = f"Создана новая сеть (hidden_size={config.hidden_size})"
                self._init_network(config.hidden_size)
            
            results = []
            recorded_games = []
            
            for i in range(config.num_games):
                # Record detailed states for first 3 games
                record_states = config.save_replays and i < 3
                
                result = self._run_single_game(
                    game_id=i + 1,
                    epsilon=config.epsilon,
                    record_states=record_states
                )
                
                results.append(result)
                
                if record_states:
                    recorded_games.append(result)
                
                # Update progress
                progress.games_completed = i + 1
                if result["winner"] == "WHITE":
                    progress.white_wins += 1
                elif result["winner"] == "BLACK":
                    progress.black_wins += 1
                else:
                    progress.draws += 1
                
                progress.avg_steps = np.mean([r["steps"] for r in results])
            
            # Save replays if requested
            if config.save_replays and recorded_games:
                replay_path = self.replays_dir / "replay.json"
                with open(replay_path, "w", encoding="utf-8") as f:
                    json.dump(recorded_games, f, indent=2, ensure_ascii=False)
            
            # Save model
            model_path = self.models_dir / f"{config.model_name}_{session_id}.pt"
            torch.save(self.policy_net.state_dict(), model_path)
            
            progress.status = "completed"
            progress.end_time = datetime.now().isoformat()
            
        except Exception as e:
            progress.status = "error"
            progress.error_message = str(e)
            progress.end_time = datetime.now().isoformat()
            raise
    
    def load_model(self, model_path: str) -> bool:
        """Load a saved model - auto-detects architecture (Gym or AlphaZero)."""
        try:
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            checkpoint = torch.load(model_path, map_location=device)
            
            from .ai_engine import ai_engine
            
            # Check if this is an AlphaZero model (has 'state_dict' key and policy/value heads)
            is_alphazero = (
                isinstance(checkpoint, dict) and 
                'state_dict' in checkpoint and
                any('policy_fc' in k or 'value_fc' in k for k in checkpoint.get('state_dict', {}).keys())
            )
            
            if is_alphazero:
                # Load AlphaZero model
                return self._load_alphazero_model(model_path, checkpoint, device, ai_engine)
            else:
                # Load standard Gym model
                return self._load_gym_model(model_path, checkpoint, device, ai_engine)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_alphazero_model(self, model_path: str, checkpoint: dict, device, ai_engine) -> bool:
        """Load an AlphaZero dual-head model."""
        from .alphazero_trainer import AlphaZeroNetwork, TogyzkumalakGame, NNetWrapper, AlphaZeroConfig
        
        state_dict = checkpoint['state_dict']
        # Remove 'module.' prefix if it exists (from DataParallel)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        config_data = checkpoint.get('config', {})
        
        # Get hidden_size from config or detect from weights
        hidden_size = config_data.get('hidden_size', 256)
        if 'fc1.weight' in clean_state_dict:
            hidden_size = clean_state_dict['fc1.weight'].shape[0]
        
        # Create AlphaZero network
        alphazero_net = AlphaZeroNetwork(
            input_size=128,
            hidden_size=hidden_size,
            action_size=9
        ).to(device)
        
        alphazero_net.load_state_dict(clean_state_dict)
        alphazero_net.eval()
        
        # Create a wrapper that provides policy-only interface for game play
        class AlphaZeroPolicyWrapper(nn.Module):
            """Wrapper to use AlphaZero network as a policy-only model for gameplay."""
            def __init__(self, alphazero_net, game):
                super().__init__()
                self.net = alphazero_net
                self.game = game
            
            def forward(self, x):
                """Return only policy probabilities for compatibility with existing AI."""
                # x is board observation (128-dim)
                with torch.no_grad():
                    pi, _ = self.net(x)
                    return torch.exp(pi)  # Convert log-prob to prob
        
        game = TogyzkumalakGame()
        wrapper = AlphaZeroPolicyWrapper(alphazero_net, game)
        
        # Apply to all neural levels for consistency when a model is loaded
        for level in [3, 4, 5]:
            ai_engine.models[level] = wrapper
            
        ai_engine.current_model_name = Path(model_path).stem
        ai_engine.alphazero_model = alphazero_net  # Store reference for MCTS usage
        
        print(f"[OK] AlphaZero model loaded to all levels (hidden_size={hidden_size})")
        return True
    
    def _load_gym_model(self, model_path: str, checkpoint, device, ai_engine) -> bool:
        """Load a standard Gym training model."""
        # Handle both formats: direct state_dict or {'model_state_dict': ...}
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            accuracy = checkpoint.get('final_accuracy', 'N/A')
            print(f"[OK] Loading model with accuracy: {accuracy}%")
        else:
            state_dict = checkpoint
        
        # Detect architecture from state dict
        layer_keys = [k for k in state_dict.keys() if 'weight' in k]
        num_layers = len(layer_keys)
        
        # Get dimensions from the first layer
        first_layer_key = layer_keys[0]
        hidden_size = state_dict[first_layer_key].shape[0]
        input_size = state_dict[first_layer_key].shape[1]
        
        # Dynamically create matching network
        if num_layers == 4:
            # 4-layer architecture (usually from ai_engine.PolicyNetwork)
            from .ai_engine import PolicyNetwork as AIPolicy
            target_model = AIPolicy(input_size=input_size, hidden_size=hidden_size, output_size=9)
        elif num_layers == 3:
            # 3-layer architecture (usually from gym_training.PolicyNetwork)
            target_model = PolicyNetwork(input_size=input_size, hidden_size=hidden_size, output_size=9)
        else:
            # Fallback for any other number of layers
            target_model = PolicyNetwork(input_size=input_size, hidden_size=hidden_size, output_size=9)
        
        # Apply to ai_engine level 5 (Expert/Self-Play/Training level)
        ai_engine.models[5] = target_model
        ai_engine.current_model_name = Path(model_path).stem
        
        target_model.load_state_dict(state_dict)
        
        # Also update self.policy_net reference
        self.policy_net = target_model
        
        print(f"[OK] Gym model loaded (hidden_size={hidden_size}, layers={num_layers})")
        return True
    
    def list_models(self) -> List[Dict]:
        """List all saved models including AlphaZero models."""
        models = []
        
        # 1. Standard gym training models (*.pt)
        for model_file in self.models_dir.glob("*.pt"):
            stat = model_file.stat()
            models.append({
                "name": model_file.stem,
                "path": str(model_file),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "type": "gym",
                "architecture": "PolicyNetwork"
            })
        
        # 2. AlphaZero models (*.pth.tar in models/alphazero/)
        alphazero_dir = self.models_dir / "alphazero"
        if alphazero_dir.exists():
            for model_file in alphazero_dir.glob("*.pth.tar"):
                # Skip temp files
                if model_file.stem == "temp":
                    continue
                stat = model_file.stat()
                
                # Determine model name
                name = model_file.stem
                if name == "best":
                    display_name = "alphazero_best"
                elif name.startswith("checkpoint_"):
                    iter_num = name.replace("checkpoint_", "")
                    display_name = f"alphazero_iter{iter_num}"
                else:
                    display_name = f"alphazero_{name}"
                
                models.append({
                    "name": display_name,
                    "path": str(model_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "type": "alphazero",
                    "architecture": "AlphaZeroNetwork"
                })
        
        return sorted(models, key=lambda x: x["created"], reverse=True)


# Global training manager instance
training_manager = GymTrainingManager()

