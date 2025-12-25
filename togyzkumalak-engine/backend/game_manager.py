"""
Game Manager for Togyzkumalak.

Manages game state, move validation, and game flow.
Uses the togyz_py board logic for clean notation support.
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class GameStatus(Enum):
    """Game status enumeration."""
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    ABANDONED = "abandoned"


class PlayerColor(Enum):
    """Player color enumeration."""
    WHITE = "white"  # First player (Бастаушы)
    BLACK = "black"  # Second player (Қостаушы)


@dataclass
class MoveRecord:
    """Record of a single move."""
    move_number: int
    player: str  # "white" or "black"
    action: int  # 0-8 (pit index)
    notation: str  # e.g., "15", "23x"
    timestamp: str
    board_state: Dict
    thinking_time_ms: Optional[int] = None


@dataclass
class GameState:
    """Complete game state."""
    game_id: str
    status: GameStatus
    human_color: str
    ai_level: int
    
    # Board state (togyz_py format)
    # [0-8]: White pits, [9-17]: Black pits
    # [18]: White tuzduk position (0 = none, 1-9 = pit number)
    # [19]: Black tuzduk position
    # [20]: White kazan (score)
    # [21]: Black kazan (score)
    # [22]: Current player (0 = white, 1 = black)
    fields: List[int] = field(default_factory=list)
    
    moves: List[MoveRecord] = field(default_factory=list)
    winner: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.fields:
            # Initialize standard starting position
            self.fields = [9] * 18 + [0, 0, 0, 0, 0]  # 18 pits + tuzduk pos + kazans + current
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class TogyzkumalakBoard:
    """
    Togyzkumalak board logic.
    Based on togyz_py implementation with enhancements.
    """
    
    TUZDUK = -1
    
    def __init__(self, fields: List[int] = None):
        """Initialize board with optional state."""
        if fields:
            self.fields = fields.copy()
        else:
            # Standard starting position
            # [0-8]: White pits (9 each)
            # [9-17]: Black pits (9 each)
            # [18]: White tuzduk position (0 = none)
            # [19]: Black tuzduk position (0 = none)
            # [20]: White kazan
            # [21]: Black kazan
            # [22]: Current player (0 = white, 1 = black)
            self.fields = [9] * 18 + [0, 0, 0, 0, 0]
    
    @property
    def current_player(self) -> str:
        """Get current player color."""
        return "white" if self.fields[22] == 0 else "black"
    
    @property
    def white_kazan(self) -> int:
        """Get white player's kazan (score)."""
        return self.fields[20]
    
    @property
    def black_kazan(self) -> int:
        """Get black player's kazan (score)."""
        return self.fields[21]
    
    @property
    def is_finished(self) -> bool:
        """Check if game is finished."""
        return self.white_kazan > 81 or self.black_kazan > 81 or \
               (self.white_kazan == 81 and self.black_kazan == 81)
    
    @property
    def winner(self) -> Optional[str]:
        """Get winner if game is finished."""
        if self.white_kazan > 81:
            return "white"
        elif self.black_kazan > 81:
            return "black"
        elif self.white_kazan == 81 and self.black_kazan == 81:
            return "draw"
        return None
    
    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices (0-8)."""
        color = self.fields[22]
        legal = []
        
        for i in range(9):
            pit_index = i + (color * 9)
            if self.fields[pit_index] > 0:
                legal.append(i)
        
        return legal
    
    def get_pit_value(self, player: str, pit: int) -> int:
        """Get kumalak count in a pit."""
        offset = 0 if player == "white" else 9
        value = self.fields[offset + pit]
        return 0 if value == self.TUZDUK else value
    
    def get_tuzduk_position(self, player: str) -> int:
        """Get tuzduk position for a player (0 = none, 1-9 = pit number)."""
        return self.fields[18 if player == "white" else 19]
    
    def make_move(self, move: int) -> Tuple[bool, str]:
        """
        Make a move on the board.
        
        Args:
            move: Pit number (1-9) to move from
        
        Returns:
            Tuple of (success, notation)
        """
        color = self.fields[22]
        pit_index = move + (color * 9) - 1  # Convert to 0-based index
        num = self.fields[pit_index]
        tuzduk_captured = False
        
        # Validate move
        if num <= 0:
            return False, ""
        
        # Pick up kumalaks
        if num == 1:
            self.fields[pit_index] = 0
            sow = 1
        else:
            self.fields[pit_index] = 1  # Leave one behind
            sow = num - 1
        
        # Sow kumalaks
        current = pit_index
        for i in range(sow):
            current = (current + 1) % 18
            
            if self.fields[current] == self.TUZDUK:
                # Add to opponent's kazan
                if current < 9:
                    self.fields[21] += 1  # Black's kazan
                else:
                    self.fields[20] += 1  # White's kazan
            else:
                self.fields[current] += 1
        
        # Check for capture (even number)
        if self.fields[current] % 2 == 0 and self.fields[current] != self.TUZDUK:
            if color == 0 and current > 8:  # White captures from black's side
                self.fields[20] += self.fields[current]
                self.fields[current] = 0
            elif color == 1 and current < 9:  # Black captures from white's side
                self.fields[21] += self.fields[current]
                self.fields[current] = 0
        
        # Check for tuzduk (exactly 3)
        elif self.fields[current] == 3:
            if color == 0 and self.fields[18] == 0 and current > 8 and current < 17:
                # White creates tuzduk (can't be 9th pit, can't be symmetric)
                if self.fields[19] != current - 8:
                    self.fields[18] = current - 8
                    self.fields[current] = self.TUZDUK
                    self.fields[20] += 3
                    tuzduk_captured = True
            elif color == 1 and self.fields[19] == 0 and current < 8:
                # Black creates tuzduk (can't be 9th pit, can't be symmetric)
                if self.fields[18] != current + 1:
                    self.fields[19] = current + 1
                    self.fields[current] = self.TUZDUK
                    self.fields[21] += 3
                    tuzduk_captured = True
        
        # Switch player
        self.fields[22] = 1 - color
        
        # Check for atsyrau (opponent has no moves)
        self._check_atsyrau()
        
        # Generate notation (just the pit number, with x for tuzduk)
        notation = str(move)
        if tuzduk_captured:
            notation += "x"
        
        return True, notation
    
    def _check_atsyrau(self):
        """Check if current player has no moves (atsyrau)."""
        color = self.fields[22]
        
        has_moves = False
        for i in range(9):
            pit_index = i + (color * 9)
            if self.fields[pit_index] > 0:
                has_moves = True
                break
        
        if not has_moves:
            # Opponent collects remaining kumalaks
            opponent_color = 1 - color
            for i in range(9):
                pit_index = i + (opponent_color * 9)
                if self.fields[pit_index] > 0:
                    if opponent_color == 0:
                        self.fields[20] += self.fields[pit_index]
                    else:
                        self.fields[21] += self.fields[pit_index]
                    self.fields[pit_index] = 0
    
    def get_state_dict(self) -> Dict:
        """Get board state as dictionary with JSON-serializable types."""
        # Convert numpy types to native Python types
        def to_int(val):
            return int(val) if hasattr(val, 'item') else val
        
        return {
            "white_pits": [to_int(v) for v in self.fields[0:9]],
            "black_pits": [to_int(v) for v in self.fields[9:18]],
            "white_tuzduk": to_int(self.fields[18]),
            "black_tuzduk": to_int(self.fields[19]),
            "white_kazan": to_int(self.fields[20]),
            "black_kazan": to_int(self.fields[21]),
            "current_player": self.current_player,
            "legal_moves": [to_int(m) for m in self.get_legal_moves()],
            "is_finished": self.is_finished,
            "winner": self.winner
        }
    
    def to_observation(self) -> List[float]:
        """Convert to neural network observation format (128-dim)."""
        # Similar to gym environment observation
        obs = []
        
        def to_float(val):
            return float(val) if hasattr(val, 'item') else float(val)
        
        # White pits (9 values, normalized)
        for i in range(9):
            val = self.fields[i] if self.fields[i] != self.TUZDUK else 0
            obs.append(to_float(val) / 81.0)
        
        # Black pits (9 values, normalized)
        for i in range(9, 18):
            val = self.fields[i] if self.fields[i] != self.TUZDUK else 0
            obs.append(to_float(val) / 81.0)
        
        # Kazans (normalized)
        obs.append(to_float(self.fields[20]) / 162.0)
        obs.append(to_float(self.fields[21]) / 162.0)
        
        # Tuzduk positions (one-hot encoded, 9 positions each)
        for i in range(9):
            obs.append(1.0 if int(self.fields[18]) == i + 1 else 0.0)
        for i in range(9):
            obs.append(1.0 if int(self.fields[19]) == i + 1 else 0.0)
        
        # Current player (one-hot)
        obs.append(1.0 if int(self.fields[22]) == 0 else 0.0)
        obs.append(1.0 if int(self.fields[22]) == 1 else 0.0)
        
        # Pad to 128 dimensions
        while len(obs) < 128:
            obs.append(0.0)
        
        return obs[:128]


class GameManager:
    """Manages multiple games and game flow."""
    
    def __init__(self, logs_dir: str = "logs/games"):
        self.games: Dict[str, GameState] = {}
        self.boards: Dict[str, TogyzkumalakBoard] = {}
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
    
    def create_game(
        self,
        human_color: str = "white",
        ai_level: int = 3
    ) -> GameState:
        """Create a new game."""
        game_id = str(uuid.uuid4())[:8]
        
        game = GameState(
            game_id=game_id,
            status=GameStatus.IN_PROGRESS,
            human_color=human_color,
            ai_level=ai_level
        )
        
        board = TogyzkumalakBoard()
        
        self.games[game_id] = game
        self.boards[game_id] = board
        
        return game
    
    def get_game(self, game_id: str) -> Optional[GameState]:
        """Get game by ID."""
        return self.games.get(game_id)
    
    def get_board(self, game_id: str) -> Optional[TogyzkumalakBoard]:
        """Get board by game ID."""
        return self.boards.get(game_id)
    
    def make_move(
        self,
        game_id: str,
        move: int,
        thinking_time_ms: int = 0
    ) -> Tuple[bool, Dict]:
        """
        Make a move in a game.
        
        Args:
            game_id: Game ID
            move: Pit number (1-9)
            thinking_time_ms: Time spent thinking (for logging)
        
        Returns:
            Tuple of (success, game_state_dict)
        """
        game = self.games.get(game_id)
        board = self.boards.get(game_id)
        
        if not game or not board:
            return False, {"error": "Game not found"}
        
        if game.status != GameStatus.IN_PROGRESS:
            return False, {"error": "Game is not in progress"}
        
        # Remember who made the move BEFORE switching
        current_player = board.current_player
        
        # Make the move
        success, notation = board.make_move(move)
        
        if not success:
            return False, {"error": "Invalid move"}
        
        # Record the move with the player who made it
        move_record = MoveRecord(
            move_number=len(game.moves) + 1,
            player=current_player,  # Player who made this move
            action=move - 1,  # Convert to 0-based
            notation=notation,
            timestamp=datetime.now().isoformat(),
            board_state=board.get_state_dict(),
            thinking_time_ms=thinking_time_ms
        )
        game.moves.append(move_record)
        game.fields = board.fields.copy()
        game.updated_at = datetime.now().isoformat()
        
        # Check for game end
        if board.is_finished:
            game.status = GameStatus.FINISHED
            game.winner = board.winner
            self._save_game_log(game)
        
        return True, self.get_game_state(game_id)
    
    def get_game_state(self, game_id: str) -> Dict:
        """Get complete game state as dictionary."""
        game = self.games.get(game_id)
        board = self.boards.get(game_id)
        
        if not game or not board:
            return {"error": "Game not found"}
        
        return {
            "game_id": game.game_id,
            "status": game.status.value,
            "human_color": game.human_color,
            "ai_level": game.ai_level,
            "board": board.get_state_dict(),
            "move_count": len(game.moves),
            "last_move": game.moves[-1].notation if game.moves else None,
            "winner": game.winner,
            "observation": board.to_observation()
        }
    
    def get_move_history(self, game_id: str) -> List[Dict]:
        """Get move history for a game."""
        game = self.games.get(game_id)
        if not game:
            return []
        
        history = []
        for i, move in enumerate(game.moves):
            history.append({
                "number": move.move_number,
                "player": move.player,
                "notation": move.notation,
                "thinking_time": move.thinking_time_ms
            })
        
        return history
    
    def _save_game_log(self, game: GameState):
        """Save completed game to log file."""
        filename = f"{game.game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.logs_dir, filename)
        
        log_data = {
            "game_id": game.game_id,
            "human_color": game.human_color,
            "ai_level": game.ai_level,
            "winner": game.winner,
            "total_moves": len(game.moves),
            "created_at": game.created_at,
            "finished_at": game.updated_at,
            "moves": [
                {
                    "number": m.move_number,
                    "player": m.player,
                    "notation": m.notation,
                    "action": m.action
                }
                for m in game.moves
            ],
            "final_score": {
                "white": game.fields[20],
                "black": game.fields[21]
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def resign(self, game_id: str, player: str) -> Dict:
        """Player resigns the game."""
        game = self.games.get(game_id)
        if not game:
            return {"error": "Game not found"}
        
        game.status = GameStatus.FINISHED
        game.winner = "black" if player == "white" else "white"
        game.updated_at = datetime.now().isoformat()
        
        self._save_game_log(game)
        return self.get_game_state(game_id)


# Global game manager instance
game_manager = GameManager()

