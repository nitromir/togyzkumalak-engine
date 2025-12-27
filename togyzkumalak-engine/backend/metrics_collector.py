"""
Real Metrics Collector for Togyzkumalak RL Training.

Collects REAL data from actual game files - NO MOCKS, NO FAKE DATA.
All metrics are computed from actual logs stored on disk.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrainingMetrics:
    """Real training metrics from actual data."""
    total_games: int = 0
    total_moves: int = 0
    
    # By source
    self_play_games: int = 0
    gemini_battle_games: int = 0
    human_games: int = 0
    
    # Results
    model_wins: int = 0
    model_losses: int = 0
    draws: int = 0
    
    # ELO tracking
    current_elo: int = 1500
    elo_history: List[Dict] = field(default_factory=list)
    peak_elo: int = 1500
    
    # Game stats
    avg_game_length: float = 0.0
    avg_score_diff: float = 0.0
    
    # Time tracking
    total_training_time_sec: float = 0.0
    last_updated: str = ""


class RealMetricsCollector:
    """
    Collects REAL metrics from actual game files.
    
    No mocks. No fake data. All numbers come from actual logs.
    """
    
    def __init__(self):
        self.logs_base = "logs"
        self.gemini_battles_dir = os.path.join(self.logs_base, "gemini_battles")
        self.self_play_dir = os.path.join(self.logs_base, "self_play")
        self.training_dir = os.path.join(self.logs_base, "training")
        
        # Create directories if needed
        for dir_path in [self.gemini_battles_dir, self.self_play_dir, self.training_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Cache for performance
        self._cache: Dict = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=5)  # Refresh every 5 seconds
    
    def get_all_metrics(self, force_refresh: bool = False) -> Dict:
        """
        Get all real metrics from actual files.
        
        Returns comprehensive metrics dict with NO mock data.
        """
        now = datetime.now()
        
        # Use cache if valid
        if not force_refresh and self._cache_time and (now - self._cache_time) < self._cache_ttl:
            return self._cache
        
        metrics = {
            "timestamp": now.isoformat(),
            "gemini_battles": self._collect_gemini_battle_metrics(),
            "self_play": self._collect_self_play_metrics(),
            "training": self._collect_training_metrics(),
            "elo": self._collect_elo_data(),
            "dataset": self._collect_dataset_stats(),
            "convergence": self._check_convergence(),
        }
        
        # Update cache
        self._cache = metrics
        self._cache_time = now
        
        return metrics
    
    def _collect_gemini_battle_metrics(self) -> Dict:
        """Collect REAL metrics from Gemini battle game files."""
        games_dir = os.path.join(self.gemini_battles_dir, "games")
        sessions_dir = os.path.join(self.gemini_battles_dir, "sessions")
        
        metrics = {
            "total_games": 0,
            "total_sessions": 0,
            "model_wins": 0,
            "gemini_wins": 0,
            "draws": 0,
            "total_moves": 0,
            "avg_game_length": 0.0,
            "avg_score_diff": 0.0,
            "winrate": 0.0,
            "games_by_session": {},
            "recent_games": [],
        }
        
        if not os.path.exists(games_dir):
            return metrics
        
        game_lengths = []
        score_diffs = []
        
        # Read all game files
        for filename in os.listdir(games_dir):
            if not filename.endswith('.json'):
                continue
            
            try:
                filepath = os.path.join(games_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    game = json.load(f)
                
                metrics["total_games"] += 1
                metrics["total_moves"] += game.get("total_moves", 0)
                game_lengths.append(game.get("total_moves", 0))
                
                # Score diff
                final_score = game.get("final_score", {})
                white_kazan = final_score.get("white_kazan", 0)
                black_kazan = final_score.get("black_kazan", 0)
                score_diffs.append(abs(white_kazan - black_kazan))
                
                # Winner tracking
                model_color = game.get("model_color", "white")
                winner = game.get("winner")
                
                if winner == model_color:
                    metrics["model_wins"] += 1
                elif winner == "draw":
                    metrics["draws"] += 1
                else:
                    metrics["gemini_wins"] += 1
                
                # Session tracking
                session_id = game.get("session_id", "unknown")
                if session_id not in metrics["games_by_session"]:
                    metrics["games_by_session"][session_id] = 0
                metrics["games_by_session"][session_id] += 1
                
                # Recent games
                if len(metrics["recent_games"]) < 10:
                    metrics["recent_games"].append({
                        "game_id": game.get("game_id"),
                        "session_id": session_id,
                        "winner": winner,
                        "model_color": model_color,
                        "total_moves": game.get("total_moves", 0),
                        "timestamp": game.get("timestamp", "")
                    })
            
            except Exception as e:
                print(f"Error reading game file {filename}: {e}")
                continue
        
        # Calculate averages
        if game_lengths:
            metrics["avg_game_length"] = float(np.mean(game_lengths))
        if score_diffs:
            metrics["avg_score_diff"] = float(np.mean(score_diffs))
        
        total_decided = metrics["model_wins"] + metrics["gemini_wins"]
        if total_decided > 0:
            metrics["winrate"] = metrics["model_wins"] / (metrics["total_games"]) * 100
        
        # Count sessions
        if os.path.exists(sessions_dir):
            metrics["total_sessions"] = len([f for f in os.listdir(sessions_dir) if f.endswith('.json')])
        
        # Sort recent games by timestamp
        metrics["recent_games"].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return metrics
    
    def _collect_self_play_metrics(self) -> Dict:
        """Collect REAL metrics from self-play game files."""
        games_dir = os.path.join(self.self_play_dir, "games")
        
        metrics = {
            "total_games": 0,
            "total_moves": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "avg_game_length": 0.0,
            "win_balance": 0.0,  # Should be close to 0 for balanced self-play
        }
        
        if not os.path.exists(games_dir):
            return metrics
        
        game_lengths = []
        
        for filename in os.listdir(games_dir):
            if not filename.endswith('.json'):
                continue
            
            try:
                filepath = os.path.join(games_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    game = json.load(f)
                
                metrics["total_games"] += 1
                metrics["total_moves"] += game.get("total_moves", 0)
                game_lengths.append(game.get("total_moves", 0))
                
                winner = game.get("winner")
                if winner == "white":
                    metrics["white_wins"] += 1
                elif winner == "black":
                    metrics["black_wins"] += 1
                else:
                    metrics["draws"] += 1
            
            except Exception:
                continue
        
        if game_lengths:
            metrics["avg_game_length"] = float(np.mean(game_lengths))
        
        if metrics["total_games"] > 0:
            metrics["win_balance"] = abs(metrics["white_wins"] - metrics["black_wins"]) / metrics["total_games"]
        
        return metrics
    
    def _collect_training_metrics(self) -> Dict:
        """Collect REAL training metrics from log files."""
        metrics_file = os.path.join(self.training_dir, "metrics.jsonl")
        
        metrics = {
            "total_training_steps": 0,
            "last_loss": None,
            "last_policy_loss": None,
            "last_value_loss": None,
            "last_entropy": None,
            "loss_history": [],
            "is_training": False,
        }
        
        if not os.path.exists(metrics_file):
            return metrics
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines[-1000:]:  # Last 1000 entries
                try:
                    entry = json.loads(line.strip())
                    metrics["total_training_steps"] = entry.get("step", metrics["total_training_steps"])
                    
                    if "total_loss" in entry:
                        metrics["last_loss"] = entry["total_loss"]
                        metrics["loss_history"].append({
                            "step": entry.get("step", 0),
                            "loss": entry["total_loss"]
                        })
                    
                    if "policy_loss" in entry:
                        metrics["last_policy_loss"] = entry["policy_loss"]
                    if "value_loss" in entry:
                        metrics["last_value_loss"] = entry["value_loss"]
                    if "entropy" in entry:
                        metrics["last_entropy"] = entry["entropy"]
                
                except json.JSONDecodeError:
                    continue
            
            # Keep only last 100 for chart
            metrics["loss_history"] = metrics["loss_history"][-100:]
        
        except Exception as e:
            print(f"Error reading training metrics: {e}")
        
        return metrics
    
    def _collect_elo_data(self) -> Dict:
        """Collect REAL ELO data from session files."""
        sessions_dir = os.path.join(self.gemini_battles_dir, "sessions")
        
        elo_data = {
            "current_elo": 1500,
            "peak_elo": 1500,
            "lowest_elo": 1500,
            "total_games": 0,
            "history": [],
            "category": "Начинающий",
        }
        
        if not os.path.exists(sessions_dir):
            return elo_data
        
        # Collect all ELO history from all sessions
        all_history = []
        
        for filename in os.listdir(sessions_dir):
            if not filename.endswith('.json'):
                continue
            
            try:
                filepath = os.path.join(sessions_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    session = json.load(f)
                
                session_history = session.get("elo_history", [])
                session_id = session.get("session_id", "unknown")
                started_at = session.get("started_at", "")
                
                for entry in session_history:
                    all_history.append({
                        "session_id": session_id,
                        "game": entry.get("game", 0),
                        "elo": entry.get("elo", 1500),
                        "change": entry.get("change", 0),
                        "result": entry.get("result", ""),
                        "timestamp": started_at
                    })
                
                # Track current ELO from latest session
                if session.get("model_elo"):
                    current = session["model_elo"]
                    if len(all_history) == 0 or current != 1500:
                        elo_data["current_elo"] = current
            
            except Exception:
                continue
        
        # Sort by timestamp and game number
        all_history.sort(key=lambda x: (x.get("timestamp", ""), x.get("game", 0)))
        
        if all_history:
            elos = [h["elo"] for h in all_history]
            elo_data["peak_elo"] = max(elos)
            elo_data["lowest_elo"] = min(elos)
            elo_data["total_games"] = len(all_history)
            elo_data["history"] = all_history[-100:]  # Last 100 for chart
            elo_data["current_elo"] = all_history[-1]["elo"]
        
        # Determine category
        current = elo_data["current_elo"]
        if current >= 2400:
            elo_data["category"] = "Гроссмейстер"
        elif current >= 2200:
            elo_data["category"] = "Международный мастер"
        elif current >= 2000:
            elo_data["category"] = "Мастер"
        elif current >= 1800:
            elo_data["category"] = "Кандидат в мастера"
        elif current >= 1600:
            elo_data["category"] = "Сильный клубный"
        elif current >= 1400:
            elo_data["category"] = "Клубный игрок"
        elif current >= 1200:
            elo_data["category"] = "Любитель"
        elif current >= 1000:
            elo_data["category"] = "Начинающий"
        else:
            elo_data["category"] = "Новичок"
        
        return elo_data
    
    def _collect_dataset_stats(self) -> Dict:
        """Collect REAL dataset statistics."""
        games_gemini = os.path.join(self.gemini_battles_dir, "games")
        games_self_play = os.path.join(self.self_play_dir, "games")
        
        stats = {
            "gemini_games": 0,
            "self_play_games": 0,
            "human_games": 0,  # Future: from tournament imports
            "total_transitions": 0,
            "gemini_pct": 0.0,
            "self_play_pct": 0.0,
            "human_pct": 0.0,
            "total_disk_size_mb": 0.0,
        }
        
        # Count Gemini battle games and transitions
        if os.path.exists(games_gemini):
            for filename in os.listdir(games_gemini):
                if filename.endswith('.json'):
                    stats["gemini_games"] += 1
                    filepath = os.path.join(games_gemini, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            game = json.load(f)
                        stats["total_transitions"] += game.get("total_moves", 0)
                        stats["total_disk_size_mb"] += os.path.getsize(filepath) / (1024 * 1024)
                    except:
                        pass
        
        # Count self-play games
        if os.path.exists(games_self_play):
            for filename in os.listdir(games_self_play):
                if filename.endswith('.json'):
                    stats["self_play_games"] += 1
                    filepath = os.path.join(games_self_play, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            game = json.load(f)
                        stats["total_transitions"] += game.get("total_moves", 0)
                        stats["total_disk_size_mb"] += os.path.getsize(filepath) / (1024 * 1024)
                    except:
                        pass
        
        # Calculate percentages
        total = stats["gemini_games"] + stats["self_play_games"] + stats["human_games"]
        if total > 0:
            stats["gemini_pct"] = stats["gemini_games"] / total * 100
            stats["self_play_pct"] = stats["self_play_games"] / total * 100
            stats["human_pct"] = stats["human_games"] / total * 100
        
        return stats
    
    def _check_convergence(self) -> Dict:
        """Check if training has converged based on REAL data."""
        elo_data = self._collect_elo_data()
        history = elo_data.get("history", [])
        
        result = {
            "status": "insufficient_data",
            "variance": None,
            "trend": None,
            "recommendation": "Нужно больше данных для анализа",
        }
        
        if len(history) < 10:
            return result
        
        # Get last 20 ELO values
        recent_elos = [h["elo"] for h in history[-20:]]
        
        variance = float(np.var(recent_elos))
        mean = float(np.mean(recent_elos))
        trend = (recent_elos[-1] - recent_elos[0]) / len(recent_elos)
        
        result["variance"] = round(variance, 2)
        result["trend"] = round(trend, 2)
        
        if variance < 100 and abs(trend) < 1:
            result["status"] = "converged"
            result["recommendation"] = "Модель сошлась. Попробуйте увеличить diversity оппонентов."
        elif trend > 2:
            result["status"] = "improving"
            result["recommendation"] = "Модель улучшается. Продолжайте обучение."
        elif trend < -2:
            result["status"] = "degrading"
            result["recommendation"] = "Производительность падает. Проверьте гиперпараметры."
        else:
            result["status"] = "stable"
            result["recommendation"] = "Стабильная производительность. Можно начать эксперименты."
        
        return result
    
    def log_training_step(
        self,
        step: int,
        total_loss: float,
        policy_loss: float = None,
        value_loss: float = None,
        entropy: float = None,
        learning_rate: float = None
    ):
        """Log a REAL training step to file."""
        os.makedirs(self.training_dir, exist_ok=True)
        metrics_file = os.path.join(self.training_dir, "metrics.jsonl")
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "total_loss": total_loss,
        }
        
        if policy_loss is not None:
            entry["policy_loss"] = policy_loss
        if value_loss is not None:
            entry["value_loss"] = value_loss
        if entropy is not None:
            entry["entropy"] = entropy
        if learning_rate is not None:
            entry["learning_rate"] = learning_rate
        
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
        
        # Invalidate cache
        self._cache_time = None
    
    def get_training_data_for_gym(self) -> List[Dict]:
        """
        Get REAL training data from Gemini battles for Gym training.
        
        Returns list of transitions in format:
        {state, action, reward, next_state, done, policy_logits, value_estimate}
        """
        games_dir = os.path.join(self.gemini_battles_dir, "games")
        
        if not os.path.exists(games_dir):
            return []
        
        all_transitions = []
        
        for filename in os.listdir(games_dir):
            if not filename.endswith('.json'):
                continue
            
            try:
                filepath = os.path.join(games_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    game = json.load(f)
                
                transitions = self._game_to_transitions(game)
                all_transitions.extend(transitions)
            
            except Exception as e:
                print(f"Error processing game file {filename}: {e}")
                continue
        
        return all_transitions
    
    def _game_to_transitions(self, game: Dict) -> List[Dict]:
        """Convert a game to training transitions."""
        transitions = []
        
        states = game.get("states", [])
        moves = game.get("moves", [])
        training_data = game.get("training_data", {})
        
        if not states or not moves:
            return []
        
        model_color = game.get("model_color", "white")
        final_reward = training_data.get("reward", 0.0)
        
        for i, move in enumerate(moves):
            # Only use model's moves for training
            if move.get("player") != model_color:
                continue
            
            # Get states before and after
            step = move.get("number", i + 1)
            state_before = states[step - 1] if step - 1 < len(states) else None
            state_after = states[step] if step < len(states) else None
            
            if not state_before or not state_after:
                continue
            
            # Compute intermediate reward
            my_kazan_key = f"{model_color}_kazan"
            opp_color = "black" if model_color == "white" else "white"
            opp_kazan_key = f"{opp_color}_kazan"
            
            my_score_before = state_before.get(my_kazan_key, 0)
            my_score_after = state_after.get(my_kazan_key, 0)
            opp_score_before = state_before.get(opp_kazan_key, 0)
            opp_score_after = state_after.get(opp_kazan_key, 0)
            
            intermediate_reward = 0.01 * ((my_score_after - my_score_before) - (opp_score_after - opp_score_before))
            
            # Add final reward to last move
            is_last = (i == len(moves) - 1) or (i == len(moves) - 2 and moves[-1].get("player") != model_color)
            if is_last:
                intermediate_reward += final_reward
            
            transition = {
                "state": move.get("observation_before", []),
                "action": move.get("action_index", 0),
                "reward": intermediate_reward,
                "next_state": state_after.get("observation", []),
                "done": is_last,
                "legal_moves": move.get("legal_moves_before", []),
                # Policy data for distillation
                "policy_logits": move.get("policy_logits"),
                "action_probs": move.get("action_probs"),
                "value_estimate": move.get("value_estimate"),
            }
            
            transitions.append(transition)
        
        return transitions


# Global collector instance
metrics_collector = RealMetricsCollector()
