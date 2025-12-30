"""
Task Manager for AlphaZero Training

Manages background AlphaZero training tasks using the new CPU-optimized trainer.
"""

import os
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Any

# Import our new CPU-optimized AlphaZero trainer
from .alphazero_trainer import (
    AlphaZeroTaskManagerV2,
    AlphaZeroConfig,
    alphazero_task_manager as _az_manager
)


class TrainingStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class AlphaZeroTaskManager:
    """
    Manages background AlphaZero training tasks.
    
    This is a wrapper around the new AlphaZeroTaskManagerV2 for backward compatibility
    with the existing API.
    """
    
    def __init__(self, logs_dir: str = "logs/alphazero"):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        self.lock = threading.Lock()
        
        # Use the new manager internally
        self._manager = _az_manager

    def start_training(self, config: Dict[str, Any]) -> str:
        """
        Start an AlphaZero training session.
        
        Args:
            config: Training configuration with keys:
                - numIters: Number of iterations (default: 10)
                - numEps: Episodes per iteration (default: 10)
                - numMCTSSims: MCTS simulations per move (default: 25)
                - cpuct: Exploration constant (default: 1.0)
                - batch_size: Training batch size (default: 32)
                - hidden_size: Network hidden size (default: 256)
        
        Returns:
            task_id: Unique identifier for the training task
        """
        # Start training using the new manager
        task_id = self._manager.start_training(config)
        
        # Also track in local tasks dict for backward compatibility
        with self.lock:
            self.tasks[task_id] = {
                "task_id": task_id,
                "status": TrainingStatus.RUNNING,
                "progress": 0,
                "current_iteration": 0,
                "total_iterations": config.get("numIters", 10),
                "metrics": {
                    "loss": [],
                    "accuracy": [],
                    "elo": [],
                    "win_rate": []
                },
                "start_time": datetime.now().isoformat(),
                "config": config
            }
        
        return task_id

    def get_status(self, task_id: str) -> Optional[Dict]:
        """
        Get status of a training task.
        
        Returns the current status including progress, metrics, and any errors.
        """
        # Get status from new manager
        status = self._manager.get_status(task_id)
        
        if status:
            # Update local tasks dict
            with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id].update({
                        "status": status.get("status", TrainingStatus.RUNNING),
                        "progress": status.get("progress", 0),
                        "current_iteration": status.get("current_iteration", 0),
                        "error_message": status.get("error_message"),
                    })
                    
                    # Convert metrics to expected format
                    if status.get("metrics"):
                        for m in status["metrics"]:
                            self.tasks[task_id]["metrics"]["loss"].append({
                                "iter": m.get("iteration", 0),
                                "value": m.get("policy_loss", 0) + m.get("value_loss", 0)
                            })
                            self.tasks[task_id]["metrics"]["win_rate"].append({
                                "iter": m.get("iteration", 0),
                                "value": m.get("win_rate", 0) * 100
                            })
            
            return self.tasks.get(task_id, status)
        
        # Fallback to local tasks
        with self.lock:
            return self.tasks.get(task_id)

    def stop_task(self, task_id: str) -> bool:
        """Stop a running training task."""
        success = self._manager.stop_task(task_id)
        
        if success:
            with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id]["status"] = TrainingStatus.STOPPED
        
        return success
    
    def list_tasks(self) -> Dict[str, Dict]:
        """List all training tasks."""
        # Get from new manager
        new_tasks = self._manager.list_tasks()
        
        # Merge with local tasks
        with self.lock:
            for task_id, status in new_tasks.items():
                if task_id not in self.tasks:
                    self.tasks[task_id] = status
                else:
                    self.tasks[task_id].update({
                        "status": status.get("status"),
                        "progress": status.get("progress"),
                        "current_iteration": status.get("current_iteration")
                    })
            
            return dict(self.tasks)

    def get_checkpoints(self):
        """Get list of all checkpoints from the underlying manager."""
        return self._manager.get_checkpoints()

    def start_tournament(self, num_games: int = 20) -> str:
        """Start a tournament between all checkpoints."""
        return self._manager.start_tournament(num_games)
        
    def get_tournament_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a tournament."""
        return self._manager.get_tournament_status(task_id)
        
    def list_tournaments(self) -> Dict[str, Dict]:
        """List all tournaments."""
        return self._manager.list_tournaments()

    def rename_checkpoint(self, old_name: str, new_name: str) -> bool:
        """Rename a checkpoint file."""
        return self._manager.rename_checkpoint(old_name, new_name)


# Global instance - backward compatible with existing code
az_task_manager = AlphaZeroTaskManager()
