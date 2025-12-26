"""
Weights & Biases Integration for Togyzkumalak RL Training.

REAL logging - all data comes from actual training runs and game files.
NO MOCKS, NO FAKE DATA.

If W&B is not available, falls back to local JSON logging.
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np

# Try to import wandb, gracefully handle if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[INFO] wandb not installed. Using local logging. Install with: pip install wandb")


@dataclass
class WandbConfig:
    """Configuration for W&B integration."""
    project: str = "togyzkumalak-rl"
    entity: Optional[str] = None  # Your W&B username/team
    run_name: Optional[str] = None
    tags: List[str] = None
    notes: str = ""
    
    # Logging settings
    log_frequency: int = 10  # Every N steps
    eval_frequency: int = 100  # Evaluation every N steps
    save_model_frequency: int = 500  # Save checkpoint every N steps


class WandbTracker:
    """
    REAL tracking system for Togyzkumalak RL training.
    
    All metrics are REAL - either from W&B or local JSON files.
    No fake data, no mocks.
    """
    
    def __init__(self, config: WandbConfig = None):
        self.config = config or WandbConfig()
        self.run = None
        self.step = 0
        
        # Local fallback storage
        self.logs_dir = "logs/wandb_local"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.local_log_file = os.path.join(self.logs_dir, "metrics.jsonl")
        self.is_active = False
    
    def start_run(self, run_name: str = None, resume: bool = False) -> bool:
        """
        Start a REAL W&B run or local logging session.
        
        Returns True if W&B is available, False if using local fallback.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = run_name or self.config.run_name or f"run_{timestamp}"
        
        if WANDB_AVAILABLE:
            try:
                self.run = wandb.init(
                    project=self.config.project,
                    entity=self.config.entity,
                    name=name,
                    tags=self.config.tags or ["togyzkumalak", "rl"],
                    notes=self.config.notes,
                    resume=resume,
                    config={
                        "framework": "pytorch",
                        "game": "togyzkumalak",
                        "timestamp": timestamp
                    }
                )
                print(f"[OK] W&B run started: {self.run.url}")
                self.is_active = True
                return True
            except Exception as e:
                print(f"[WARNING] Failed to start W&B run: {e}")
                print("[INFO] Using local logging instead")
        
        # Local fallback
        self.is_active = True
        self._log_local({
            "event": "run_started",
            "run_name": name,
            "timestamp": timestamp
        })
        print(f"[OK] Local logging started: {self.local_log_file}")
        return False
    
    def finish_run(self):
        """Finish the current run."""
        if self.run:
            self.run.finish()
            self.run = None
        
        self._log_local({
            "event": "run_finished",
            "total_steps": self.step,
            "timestamp": datetime.now().isoformat()
        })
        self.is_active = False
    
    # =========================================================================
    # TRAINING METRICS - REAL DATA ONLY
    # =========================================================================
    
    def log_training_step(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        entropy: float = None,
        kl_divergence: float = None,
        learning_rate: float = None,
        gradient_norm: float = None
    ):
        """Log REAL training metrics from actual training step."""
        self.step = step
        
        metrics = {
            "training/step": step,
            "training/policy_loss": policy_loss,
            "training/value_loss": value_loss,
            "training/total_loss": total_loss,
        }
        
        if entropy is not None:
            metrics["training/entropy"] = entropy
        if kl_divergence is not None:
            metrics["training/kl_divergence"] = kl_divergence
        if learning_rate is not None:
            metrics["training/learning_rate"] = learning_rate
        if gradient_norm is not None:
            metrics["training/gradient_norm"] = gradient_norm
        
        self._log(metrics, step)
    
    # =========================================================================
    # SELF-PLAY METRICS - REAL DATA ONLY
    # =========================================================================
    
    def log_self_play_batch(self, games: List[Dict]):
        """Log REAL self-play game statistics."""
        if not games:
            return
        
        game_lengths = [g.get("total_moves", 0) for g in games]
        score_diffs = []
        white_wins = 0
        black_wins = 0
        draws = 0
        
        for g in games:
            score = g.get("final_score", {})
            white_kazan = score.get("white_kazan", 0)
            black_kazan = score.get("black_kazan", 0)
            score_diffs.append(abs(white_kazan - black_kazan))
            
            winner = g.get("winner")
            if winner == "white":
                white_wins += 1
            elif winner == "black":
                black_wins += 1
            else:
                draws += 1
        
        n = len(games)
        metrics = {
            "self_play/games_count": n,
            "self_play/avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0,
            "self_play/std_game_length": float(np.std(game_lengths)) if game_lengths else 0,
            "self_play/avg_score_diff": float(np.mean(score_diffs)) if score_diffs else 0,
            "self_play/white_winrate": white_wins / n if n > 0 else 0,
            "self_play/black_winrate": black_wins / n if n > 0 else 0,
            "self_play/draw_rate": draws / n if n > 0 else 0,
            "self_play/win_balance": abs(white_wins - black_wins) / n if n > 0 else 0,
        }
        
        self._log(metrics, self.step)
        
        # Log histogram if W&B is available
        if self.run and game_lengths:
            try:
                self.run.log({
                    "self_play/game_length_hist": wandb.Histogram(game_lengths)
                }, step=self.step)
            except:
                pass
    
    # =========================================================================
    # GEMINI BATTLE METRICS - REAL DATA ONLY
    # =========================================================================
    
    def log_gemini_battle_game(self, game: Dict):
        """Log REAL metrics from a single Gemini battle game."""
        metrics = {
            "gemini_battle/game_id": game.get("game_id"),
            "gemini_battle/total_moves": game.get("total_moves", 0),
            "gemini_battle/model_color": game.get("model_color"),
            "gemini_battle/winner": game.get("winner"),
        }
        
        training_data = game.get("training_data", {})
        metrics["gemini_battle/result_for_model"] = training_data.get("result_for_model")
        
        score = game.get("final_score", {})
        metrics["gemini_battle/score_diff"] = abs(
            score.get("white_kazan", 0) - score.get("black_kazan", 0)
        )
        
        self._log(metrics, self.step)
    
    def log_gemini_battle_session(self, session: Dict):
        """Log REAL metrics from a complete Gemini battle session."""
        metrics = {
            "gemini_session/session_id": session.get("session_id"),
            "gemini_session/games_played": session.get("games_played", 0),
            "gemini_session/model_wins": session.get("model_wins", 0),
            "gemini_session/gemini_wins": session.get("gemini_wins", 0),
            "gemini_session/draws": session.get("draws", 0),
            "gemini_session/final_elo": session.get("model_elo", 1500),
        }
        
        total = session.get("games_played", 0)
        if total > 0:
            metrics["gemini_session/winrate"] = session.get("model_wins", 0) / total
        
        self._log(metrics, self.step)
        
        # Log ELO chart if we have history
        elo_history = session.get("elo_history", [])
        if self.run and elo_history:
            try:
                data = [[h.get("game", i), h.get("elo", 1500)] for i, h in enumerate(elo_history)]
                table = wandb.Table(data=data, columns=["game", "elo"])
                self.run.log({
                    "gemini_session/elo_chart": wandb.plot.line(
                        table, "game", "elo",
                        title="Model ELO vs Gemini"
                    )
                }, step=self.step)
            except:
                pass
    
    # =========================================================================
    # A/B TEST METRICS - REAL DATA ONLY
    # =========================================================================
    
    def log_ab_test_result(self, result: Dict):
        """Log REAL A/B test result."""
        exp_id = result.get("experiment_id", "unknown")
        
        metrics = {
            f"ab_test/{exp_id}/variant": result.get("variant"),
            f"ab_test/{exp_id}/response_time_ms": result.get("gemini_response_time_ms"),
            f"ab_test/{exp_id}/response_length": result.get("response_length"),
        }
        
        if result.get("user_rating"):
            metrics[f"ab_test/{exp_id}/user_rating"] = result.get("user_rating")
        if result.get("was_helpful") is not None:
            metrics[f"ab_test/{exp_id}/was_helpful"] = 1 if result.get("was_helpful") else 0
        
        self._log(metrics, self.step)
    
    def log_ab_test_summary(self, experiment_id: str, stats: Dict):
        """Log REAL A/B test experiment summary."""
        for variant, data in stats.get("variants", {}).items():
            prefix = f"ab_summary/{experiment_id}/{variant}"
            metrics = {
                f"{prefix}/sample_size": data.get("sample_size", 0),
                f"{prefix}/avg_response_time_ms": data.get("avg_response_time_ms"),
            }
            
            if data.get("avg_rating"):
                metrics[f"{prefix}/avg_rating"] = data["avg_rating"]
            if data.get("helpful_rate"):
                metrics[f"{prefix}/helpful_rate"] = data["helpful_rate"]
            if data.get("accuracy_rate"):
                metrics[f"{prefix}/accuracy_rate"] = data["accuracy_rate"]
            
            self._log(metrics, self.step)
        
        if stats.get("winner"):
            self._log({f"ab_summary/{experiment_id}/winner": stats["winner"]}, self.step)
    
    # =========================================================================
    # ELO TRACKING - REAL DATA ONLY
    # =========================================================================
    
    def log_elo_update(
        self,
        current_elo: int,
        elo_change: int,
        opponent: str = "gemini",
        result: str = None
    ):
        """Log REAL ELO update."""
        metrics = {
            "elo/current": current_elo,
            "elo/change": elo_change,
            "elo/opponent": opponent,
        }
        if result:
            metrics["elo/last_result"] = result
        
        self._log(metrics, self.step)
    
    # =========================================================================
    # MODEL CHECKPOINTS - REAL DATA ONLY
    # =========================================================================
    
    def log_model_checkpoint(
        self,
        version: str,
        elo: int,
        metrics: Dict[str, float],
        model_path: str
    ):
        """Log REAL model checkpoint with optional W&B artifact."""
        data = {
            "checkpoint/version": version,
            "checkpoint/elo": elo,
            "checkpoint/path": model_path,
            **{f"checkpoint/{k}": v for k, v in metrics.items()}
        }
        self._log(data, self.step)
        
        # Save as W&B artifact if available
        if self.run and os.path.exists(model_path):
            try:
                artifact = wandb.Artifact(
                    name=f"model-{version}",
                    type="model",
                    metadata={
                        "elo": elo,
                        "step": self.step,
                        **metrics
                    }
                )
                artifact.add_file(model_path)
                self.run.log_artifact(artifact)
                print(f"[OK] Model artifact saved: model-{version}")
            except Exception as e:
                print(f"[WARNING] Failed to save artifact: {e}")
    
    # =========================================================================
    # DATASET TRACKING - REAL DATA ONLY
    # =========================================================================
    
    def log_dataset_stats(
        self,
        gemini_games: int,
        self_play_games: int,
        human_games: int,
        total_transitions: int
    ):
        """Log REAL dataset composition."""
        total = gemini_games + self_play_games + human_games
        
        metrics = {
            "dataset/gemini_games": gemini_games,
            "dataset/self_play_games": self_play_games,
            "dataset/human_games": human_games,
            "dataset/total_games": total,
            "dataset/total_transitions": total_transitions,
        }
        
        if total > 0:
            metrics["dataset/gemini_pct"] = gemini_games / total * 100
            metrics["dataset/self_play_pct"] = self_play_games / total * 100
            metrics["dataset/human_pct"] = human_games / total * 100
        
        self._log(metrics, self.step)
    
    # =========================================================================
    # CONVERGENCE MONITORING - REAL DATA ONLY
    # =========================================================================
    
    def log_convergence_check(self, status: str, variance: float, trend: float):
        """Log REAL convergence check results."""
        metrics = {
            "convergence/status": status,
            "convergence/variance": variance,
            "convergence/trend": trend,
        }
        self._log(metrics, self.step)
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _log(self, metrics: Dict[str, Any], step: int):
        """Log to W&B or local file."""
        if self.run:
            try:
                self.run.log(metrics, step=step)
            except Exception as e:
                print(f"[WARNING] W&B log failed: {e}")
                self._log_local(metrics)
        else:
            self._log_local(metrics)
    
    def _log_local(self, metrics: Dict[str, Any]):
        """Log to local JSON file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.step,
            **metrics
        }
        
        with open(self.local_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def get_local_metrics(self, last_n: int = 100) -> List[Dict]:
        """Get last N metrics from local log file."""
        if not os.path.exists(self.local_log_file):
            return []
        
        metrics = []
        try:
            with open(self.local_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines[-last_n:]:
                try:
                    metrics.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        return metrics
    
    def sync_from_files(self, metrics_collector):
        """
        Sync REAL metrics from file-based collector to W&B.
        
        This ensures all historical data is properly tracked.
        """
        all_metrics = metrics_collector.get_all_metrics(force_refresh=True)
        
        # Log Gemini battle metrics
        gemini_data = all_metrics.get("gemini_battles", {})
        if gemini_data.get("total_games", 0) > 0:
            self._log({
                "sync/gemini_total_games": gemini_data["total_games"],
                "sync/gemini_model_wins": gemini_data["model_wins"],
                "sync/gemini_winrate": gemini_data["winrate"],
            }, self.step)
        
        # Log ELO data
        elo_data = all_metrics.get("elo", {})
        if elo_data.get("current_elo", 1500) != 1500:
            self._log({
                "sync/current_elo": elo_data["current_elo"],
                "sync/peak_elo": elo_data["peak_elo"],
                "sync/elo_category": elo_data["category"],
            }, self.step)
        
        # Log dataset stats
        dataset = all_metrics.get("dataset", {})
        self._log({
            "sync/dataset_gemini_games": dataset.get("gemini_games", 0),
            "sync/dataset_self_play_games": dataset.get("self_play_games", 0),
            "sync/dataset_total_transitions": dataset.get("total_transitions", 0),
        }, self.step)
        
        print("[OK] Synced real metrics to tracker")


# Global tracker instance
wandb_tracker = WandbTracker()

