"""
ELO Rating System for Togyzkumalak.

Calculates and tracks ELO ratings for players and AI.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import elo_config


@dataclass
class ELORecord:
    """Single ELO rating record."""
    elo: int
    timestamp: str
    opponent_elo: int
    result: str  # "win", "loss", "draw"
    elo_change: int


@dataclass
class PlayerELO:
    """Player ELO tracking."""
    player_id: str
    current_elo: int = 1500
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    history: List[ELORecord] = field(default_factory=list)


class ELOSystem:
    """
    ELO Rating System implementation.
    
    Uses standard ELO formula:
    - Expected score: E = 1 / (1 + 10^((R_opponent - R_player) / 400))
    - New rating: R_new = R_old + K * (S - E)
    
    Where S is actual score (1 for win, 0 for loss, 0.5 for draw)
    """
    
    def __init__(self, data_file: str = "logs/elo_data.json"):
        self.data_file = data_file
        self.players: Dict[str, PlayerELO] = {}
        self.k_factor = elo_config.k_factor
        self.load_data()
    
    def load_data(self):
        """Load ELO data from file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for player_id, player_data in data.items():
                        self.players[player_id] = PlayerELO(
                            player_id=player_id,
                            current_elo=player_data['current_elo'],
                            games_played=player_data['games_played'],
                            wins=player_data['wins'],
                            losses=player_data['losses'],
                            draws=player_data['draws'],
                            history=[ELORecord(**h) for h in player_data.get('history', [])]
                        )
            except (json.JSONDecodeError, KeyError):
                pass
    
    def save_data(self):
        """Save ELO data to file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        data = {}
        for player_id, player in self.players.items():
            data[player_id] = {
                'current_elo': player.current_elo,
                'games_played': player.games_played,
                'wins': player.wins,
                'losses': player.losses,
                'draws': player.draws,
                'history': [
                    {
                        'elo': h.elo,
                        'timestamp': h.timestamp,
                        'opponent_elo': h.opponent_elo,
                        'result': h.result,
                        'elo_change': h.elo_change
                    }
                    for h in player.history[-100:]  # Keep last 100 games
                ]
            }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_or_create_player(self, player_id: str, initial_elo: int = None) -> PlayerELO:
        """Get existing player or create new one."""
        if player_id not in self.players:
            self.players[player_id] = PlayerELO(
                player_id=player_id,
                current_elo=initial_elo or elo_config.initial_elo
            )
        return self.players[player_id]
    
    def expected_score(self, player_elo: int, opponent_elo: int) -> float:
        """Calculate expected score based on ELO difference."""
        return 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))
    
    def calculate_elo_change(
        self,
        player_elo: int,
        opponent_elo: int,
        result: str  # "win", "loss", "draw"
    ) -> int:
        """Calculate ELO change for a single game."""
        expected = self.expected_score(player_elo, opponent_elo)
        
        if result == "win":
            actual = 1.0
        elif result == "loss":
            actual = 0.0
        else:  # draw
            actual = 0.5
        
        change = round(self.k_factor * (actual - expected))
        return change
    
    def update_ratings(
        self,
        player_id: str,
        opponent_id: str,
        result: str  # "win", "loss", "draw"
    ) -> Tuple[int, int]:
        """
        Update ratings for both players after a game.
        
        Returns:
            Tuple of (player_elo_change, opponent_elo_change)
        """
        player = self.get_or_create_player(player_id)
        opponent = self.get_or_create_player(opponent_id)
        
        # Calculate changes
        player_change = self.calculate_elo_change(
            player.current_elo, opponent.current_elo, result
        )
        
        # Opponent has opposite result
        opponent_result = "loss" if result == "win" else ("win" if result == "loss" else "draw")
        opponent_change = self.calculate_elo_change(
            opponent.current_elo, player.current_elo, opponent_result
        )
        
        # Update player
        old_player_elo = player.current_elo
        player.current_elo = max(
            elo_config.min_elo,
            min(elo_config.max_elo, player.current_elo + player_change)
        )
        player.games_played += 1
        if result == "win":
            player.wins += 1
        elif result == "loss":
            player.losses += 1
        else:
            player.draws += 1
        
        player.history.append(ELORecord(
            elo=player.current_elo,
            timestamp=datetime.now().isoformat(),
            opponent_elo=opponent.current_elo,
            result=result,
            elo_change=player_change
        ))
        
        # Update opponent
        old_opponent_elo = opponent.current_elo
        opponent.current_elo = max(
            elo_config.min_elo,
            min(elo_config.max_elo, opponent.current_elo + opponent_change)
        )
        opponent.games_played += 1
        if opponent_result == "win":
            opponent.wins += 1
        elif opponent_result == "loss":
            opponent.losses += 1
        else:
            opponent.draws += 1
        
        opponent.history.append(ELORecord(
            elo=opponent.current_elo,
            timestamp=datetime.now().isoformat(),
            opponent_elo=player.current_elo,
            result=opponent_result,
            elo_change=opponent_change
        ))
        
        self.save_data()
        return player_change, opponent_change
    
    def get_ai_elo(self, level: int) -> int:
        """Get AI ELO for a specific level."""
        return elo_config.level_elos.get(level, elo_config.initial_elo)
    
    def estimate_player_elo(self, results_vs_ai: List[Tuple[int, str]]) -> int:
        """
        Estimate player ELO based on results against AI of known strength.
        
        Args:
            results_vs_ai: List of (ai_level, result) tuples
        
        Returns:
            Estimated player ELO
        """
        if not results_vs_ai:
            return elo_config.initial_elo
        
        estimated_elo = elo_config.initial_elo
        
        for ai_level, result in results_vs_ai:
            ai_elo = self.get_ai_elo(ai_level)
            change = self.calculate_elo_change(estimated_elo, ai_elo, result)
            estimated_elo += change
        
        return max(elo_config.min_elo, min(elo_config.max_elo, estimated_elo))
    
    def get_player_stats(self, player_id: str) -> Optional[Dict]:
        """Get player statistics."""
        player = self.players.get(player_id)
        if not player:
            return None
        
        return {
            'player_id': player.player_id,
            'current_elo': player.current_elo,
            'games_played': player.games_played,
            'wins': player.wins,
            'losses': player.losses,
            'draws': player.draws,
            'win_rate': player.wins / player.games_played if player.games_played > 0 else 0,
            'recent_history': [
                {
                    'elo': h.elo,
                    'change': h.elo_change,
                    'result': h.result,
                    'opponent_elo': h.opponent_elo
                }
                for h in player.history[-10:]
            ]
        }


# Global ELO system instance
elo_system = ELOSystem()

