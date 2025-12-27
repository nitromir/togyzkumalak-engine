"""
Gemini Battle System - Model vs LLM Training.

Enables automated training of the neural network against Google Gemini API.
Features:
- Deterministic Gemini responses (temperature=0)
- ELO rating tracking with international classification
- Automatic game logging
- Russian language game summaries for future LLM training
"""

import asyncio
import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import gemini_config
from .game_manager import TogyzkumalakBoard
from .ai_engine import ai_engine


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BattleConfig:
    """Battle session configuration."""
    num_games: int = 10                    # Number of games to play
    model_level: int = 5                   # Model level (1-5)
    gemini_timeout: int = 30               # Timeout for Gemini requests (seconds)
    gemini_temperature: float = 0.0        # Deterministic responses
    save_replays: bool = True              # Save game replays
    generate_summaries: bool = True        # Generate game summaries
    delay_between_games: float = 2.0       # Delay between games (seconds)
    delay_between_moves: float = 0.5       # Delay between moves (seconds)


@dataclass
class BattleSession:
    """Battle session state."""
    session_id: str
    config: BattleConfig
    status: str = "pending"  # pending, running, completed, stopped, error
    games_played: int = 0
    model_wins: int = 0
    gemini_wins: int = 0
    draws: int = 0
    model_elo: int = 1500
    gemini_elo: int = 2000  # Gemini starts as "Master" level
    elo_history: List[Dict] = field(default_factory=list)
    current_game_moves: List[Dict] = field(default_factory=list)
    last_game_summary: str = ""
    started_at: str = ""
    finished_at: str = ""
    error_message: str = ""


# ELO Rating Categories (International Classification)
ELO_CATEGORIES = {
    (0, 1000): "Новичок",
    (1000, 1200): "Начинающий",
    (1200, 1400): "Любитель",
    (1400, 1600): "Клубный игрок",
    (1600, 1800): "Сильный клубный",
    (1800, 2000): "Кандидат в мастера",
    (2000, 2200): "Мастер",
    (2200, 2400): "Международный мастер",
    (2400, 9999): "Гроссмейстер",
}


def get_elo_category(elo: int) -> str:
    """Get ELO category name by rating."""
    for (low, high), name in ELO_CATEGORIES.items():
        if low <= elo < high:
            return name
    return "Unknown"


def calculate_k_factor(games_played: int) -> int:
    """Calculate K-factor based on games played."""
    if games_played < 30:
        return 40  # Quick changes for new players
    elif games_played < 100:
        return 32  # Standard
    else:
        return 20  # Stabilized


def calculate_elo_change(player_elo: int, opponent_elo: int, result: str, k: int) -> int:
    """Calculate ELO change for a single game."""
    expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))
    
    if result == "win":
        actual = 1.0
    elif result == "loss":
        actual = 0.0
    else:  # draw
        actual = 0.5
    
    return round(k * (actual - expected))


# =============================================================================
# Gemini Player
# =============================================================================

class GeminiPlayer:
    """Gemini API player for Togyzkumalak."""
    
    HOLE_NAMES = {
        1: "Арт", 2: "Тектұрмас", 3: "Ат өтпес",
        4: "Атсыратар", 5: "Бел", 6: "Белбасар",
        7: "Қандықақпан", 8: "Көкмойын", 9: "Маңдай"
    }
    
    def __init__(self):
        self.client = None
        self.model = gemini_config.model
        self._init_client()
    
    def _init_client(self):
        """Initialize Gemini client."""
        api_key = gemini_config.api_key or os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            print("[WARNING] Gemini API key not configured for battle mode.")
            return
        
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            print(f"[OK] GeminiPlayer initialized (model: {self.model})")
        except ImportError:
            print("[WARNING] google-genai package not installed.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Gemini client: {e}")
    
    def _format_board_for_move(self, board: TogyzkumalakBoard, color: str) -> str:
        """Format board state for move request."""
        state = board.get_state_dict()
        
        lines = []
        lines.append("=== ПОЗИЦИЯ НА ДОСКЕ ===")
        
        # Black side
        lines.append(f"ЧЁРНЫЕ (Қара): лунки [9←1] = {list(reversed(state['black_pits']))}")
        lines.append(f"  Казан чёрных: {state['black_kazan']}")
        
        # White side
        lines.append(f"БЕЛЫЕ (Ақ): лунки [1→9] = {state['white_pits']}")
        lines.append(f"  Казан белых: {state['white_kazan']}")
        
        # Tuzduk
        if state['white_tuzduk'] > 0:
            lines.append(f"* Белые имеют тузду на лунке {state['white_tuzduk']} чёрных")
        if state['black_tuzduk'] > 0:
            lines.append(f"* Чёрные имеют тузду на лунке {state['black_tuzduk']} белых")
        
        return "\n".join(lines)
    
    async def get_move(
        self,
        board: TogyzkumalakBoard,
        color: str,
        timeout: int = 30
    ) -> Tuple[int, str]:
        """
        Get a move from Gemini.
        
        Returns:
            Tuple of (move (1-9), explanation or error)
        """
        if not self.client:
            # Fallback to random move
            legal = board.get_legal_moves()
            return random.choice(legal) + 1, "Gemini unavailable, random move"
        
        legal_moves = [m + 1 for m in board.get_legal_moves()]
        position_text = self._format_board_for_move(board, color)
        
        prompt = f"""Вы играете в Тогыз Кумалак за {color.upper()} ({"Белые" if color == "white" else "Чёрные"}).

{position_text}

ДОСТУПНЫЕ ХОДЫ: {legal_moves}

ПРАВИЛА НАПОМИНАНИЕ:
- Выбираете одну из своих лунок (1-9)
- Берёте все кумалаки из лунки и сеете по одному против часовой стрелки
- Если последний кумалак падает в лунку противника с чётным числом - захват
- Если получается 3 - можно создать тузду (постоянный захват)

КРИТИЧЕСКИ ВАЖНО: Ответьте ТОЛЬКО ОДНОЙ ЦИФРОЙ от 1 до 9.
Никаких объяснений, никакого текста - только цифра.
Пример правильного ответа: 5"""

        try:
            loop = asyncio.get_event_loop()
            
            async def make_request():
                return await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config={
                            "max_output_tokens": 50,
                            "temperature": 0.0,  # Deterministic
                            "top_p": 1.0,
                            "top_k": 1
                        }
                    )
                )
            
            response = await asyncio.wait_for(make_request(), timeout=timeout)
            text = response.text.strip()
            
            # Extract move number
            match = re.search(r'(\d)', text)
            if match:
                move = int(match.group(1))
                if move in legal_moves:
                    return move, f"Gemini выбрал ход {move}"
            
            # If invalid, pick first legal move
            return legal_moves[0], f"Invalid response '{text}', using {legal_moves[0]}"
            
        except asyncio.TimeoutError:
            move = random.choice(legal_moves)
            return move, f"Timeout, random move {move}"
        except Exception as e:
            move = random.choice(legal_moves)
            return move, f"Error: {e}, random move {move}"
    
    async def generate_game_summary(
        self,
        game_log: Dict,
        timeout: int = 60
    ) -> str:
        """Generate a Russian language summary of the game."""
        if not self.client:
            return "Резюме недоступно (Gemini не настроен)."
        
        # Format move history
        moves_text = []
        for move in game_log.get("moves", [])[:50]:  # Limit to first 50 moves
            player = "Б" if move["player"] == "white" else "Ч"
            moves_text.append(f"{move['number']}.{player}:{move['notation']}")
        
        history = " ".join(moves_text)
        
        winner_text = {
            "white": "Белые (Модель)",
            "black": "Чёрные (Gemini)",
            "draw": "Ничья"
        }.get(game_log.get("winner", ""), "Неизвестно")
        
        prompt = f"""Вы - профессиональный комментатор турниров по Тогыз Кумалак.
Только что завершилась партия № {game_log.get('game_id', '?')}:

РЕЗУЛЬТАТ:
- Белые (Нейросеть): {game_log.get('final_score', {}).get('white_kazan', 0)} кумалаков
- Чёрные (Gemini AI): {game_log.get('final_score', {}).get('black_kazan', 0)} кумалаков
- Победитель: {winner_text}
- Количество ходов: {game_log.get('total_moves', 0)}

ИСТОРИЯ ХОДОВ (сокращённо):
{history}

ИЗМЕНЕНИЕ ELO:
- Модель: {game_log.get('model_elo_before', 1500)} → {game_log.get('model_elo_after', 1500)} ({'+' if game_log.get('elo_change', 0) >= 0 else ''}{game_log.get('elo_change', 0)})

Напишите краткое резюме этой партии на РУССКОМ ЯЗЫКЕ (4-6 предложений).
Стиль: как будто вы комментируете турнир для телевизионной аудитории.
Упомяните:
1. Общую оценку игры (острая, позиционная, ошибочная)
2. Решающий момент или перелом (если был)
3. Качество игры обеих сторон
4. Итоговую оценку"""

        try:
            loop = asyncio.get_event_loop()
            
            async def make_request():
                return await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config={
                            "max_output_tokens": 500,
                            "temperature": 0.7  # Some creativity for commentary
                        }
                    )
                )
            
            response = await asyncio.wait_for(make_request(), timeout=timeout)
            return response.text.strip()
            
        except asyncio.TimeoutError:
            return "Резюме недоступно (таймаут запроса)."
        except Exception as e:
            return f"Резюме недоступно (ошибка: {str(e)[:100]})."
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self.client is not None


# =============================================================================
# Battle Manager
# =============================================================================

class GeminiBattleManager:
    """Manages battle sessions between model and Gemini."""
    
    def __init__(self, logs_dir: str = "logs/gemini_battles"):
        self.sessions: Dict[str, BattleSession] = {}
        self.logs_dir = logs_dir
        self.gemini_player = GeminiPlayer()
        self.stop_flags: Dict[str, bool] = {}
        
        # Create directories
        os.makedirs(os.path.join(logs_dir, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(logs_dir, "games"), exist_ok=True)
        os.makedirs(os.path.join(logs_dir, "summaries"), exist_ok=True)
        
        # Load existing sessions
        self._load_sessions()
    
    def _load_sessions(self):
        """Load existing sessions from disk."""
        sessions_dir = os.path.join(self.logs_dir, "sessions")
        if os.path.exists(sessions_dir):
            for filename in os.listdir(sessions_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(sessions_dir, filename), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            session_id = data.get("session_id")
                            if session_id:
                                config = BattleConfig(**data.get("config", {}))
                                session = BattleSession(
                                    session_id=session_id,
                                    config=config,
                                    status=data.get("status", "completed"),
                                    games_played=data.get("games_played", 0),
                                    model_wins=data.get("model_wins", 0),
                                    gemini_wins=data.get("gemini_wins", 0),
                                    draws=data.get("draws", 0),
                                    model_elo=data.get("model_elo", 1500),
                                    gemini_elo=data.get("gemini_elo", 2000),
                                    elo_history=data.get("elo_history", []),
                                    started_at=data.get("started_at", ""),
                                    finished_at=data.get("finished_at", "")
                                )
                                self.sessions[session_id] = session
                    except Exception as e:
                        print(f"Failed to load session {filename}: {e}")
    
    def _save_session(self, session: BattleSession):
        """Save session to disk."""
        filepath = os.path.join(self.logs_dir, "sessions", f"session_{session.session_id}.json")
        data = {
            "session_id": session.session_id,
            "config": {
                "num_games": session.config.num_games,
                "model_level": session.config.model_level,
                "gemini_timeout": session.config.gemini_timeout,
                "gemini_temperature": session.config.gemini_temperature,
                "save_replays": session.config.save_replays,
                "generate_summaries": session.config.generate_summaries
            },
            "status": session.status,
            "games_played": session.games_played,
            "model_wins": session.model_wins,
            "gemini_wins": session.gemini_wins,
            "draws": session.draws,
            "model_elo": session.model_elo,
            "gemini_elo": session.gemini_elo,
            "elo_history": session.elo_history,
            "started_at": session.started_at,
            "finished_at": session.finished_at,
            "error_message": session.error_message
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_game_log(self, session_id: str, game_log: Dict):
        """Save individual game log."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_id = game_log.get("game_id", 0)
        filename = f"game_{session_id}_{game_id:03d}_{timestamp}.json"
        filepath = os.path.join(self.logs_dir, "games", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game_log, f, indent=2, ensure_ascii=False)
    
    def _save_summary(self, session_id: str, game_id: int, summary: str):
        """Save game summary."""
        filename = f"summary_{session_id}_{game_id:03d}.txt"
        filepath = os.path.join(self.logs_dir, "summaries", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== Партия #{game_id} ===\n\n")
            f.write(summary)
            f.write(f"\n\n[Сгенерировано: {datetime.now().isoformat()}]")
    
    def create_session(self, config: BattleConfig) -> str:
        """Create a new battle session."""
        session_id = str(uuid.uuid4())[:8]
        
        session = BattleSession(
            session_id=session_id,
            config=config,
            status="pending",
            model_elo=1500,
            gemini_elo=2000,
            elo_history=[{"game": 0, "elo": 1500, "change": 0}],
            started_at=datetime.now().isoformat()
        )
        
        self.sessions[session_id] = session
        self.stop_flags[session_id] = False
        self._save_session(session)
        
        return session_id
    
    async def run_battle_session(self, session_id: str):
        """Run the battle session (called as background task)."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        session.status = "running"
        self._save_session(session)
        
        try:
            for game_num in range(1, session.config.num_games + 1):
                # Check stop flag
                if self.stop_flags.get(session_id, False):
                    session.status = "stopped"
                    break
                
                # Alternate colors - model plays white first, then black
                model_color = "white" if game_num % 2 == 1 else "black"
                
                # Play single game
                game_log = await self._play_single_game(session, game_num, model_color)
                
                # Update session stats
                session.games_played = game_num
                
                winner = game_log.get("winner")
                if winner == model_color:
                    session.model_wins += 1
                    result = "win"
                elif winner == "draw":
                    session.draws += 1
                    result = "draw"
                else:
                    session.gemini_wins += 1
                    result = "loss"
                
                # Update ELO
                k = calculate_k_factor(session.games_played)
                elo_change = calculate_elo_change(
                    session.model_elo, session.gemini_elo, result, k
                )
                session.model_elo += elo_change
                session.model_elo = max(100, min(3000, session.model_elo))
                
                # Update game log with ELO info
                game_log["model_elo_before"] = session.elo_history[-1]["elo"]
                game_log["model_elo_after"] = session.model_elo
                game_log["elo_change"] = elo_change
                
                session.elo_history.append({
                    "game": game_num,
                    "elo": session.model_elo,
                    "change": elo_change,
                    "result": result
                })
                
                # Save game log
                if session.config.save_replays:
                    self._save_game_log(session_id, game_log)
                
                # Generate and save summary
                if session.config.generate_summaries:
                    summary = await self.gemini_player.generate_game_summary(game_log)
                    session.last_game_summary = summary
                    self._save_summary(session_id, game_num, summary)
                
                # Save session progress
                self._save_session(session)
                
                # Delay between games
                await asyncio.sleep(session.config.delay_between_games)
            
            if session.status != "stopped":
                session.status = "completed"
            
        except Exception as e:
            session.status = "error"
            session.error_message = str(e)
        
        session.finished_at = datetime.now().isoformat()
        self._save_session(session)
    
    async def _play_single_game(
        self,
        session: BattleSession,
        game_num: int,
        model_color: str
    ) -> Dict:
        """Play a single game between model and Gemini with full state tracking."""
        board = TogyzkumalakBoard()
        gemini_color = "black" if model_color == "white" else "white"
        
        moves = []
        states = []  # Full board states for replay/training
        move_number = 0
        
        session.current_game_moves = []
        
        # Save initial state
        initial_state = board.get_state_dict()
        initial_state["step"] = 0
        initial_state["action"] = None
        initial_state["player"] = "white"
        initial_state["observation"] = board.to_observation()
        states.append(initial_state)
        
        while not board.is_finished:
            current_player = board.current_player
            move_number += 1
            
            # Check stop flag
            if self.stop_flags.get(session.session_id, False):
                break
            
            timestamp = datetime.now().isoformat()
            
            # Capture state BEFORE the move for training
            observation_before = board.to_observation()
            legal_moves_before = board.get_legal_moves()
            
            # Policy data (only for model's moves)
            raw_logits = None
            action_probs = None
            value_estimate = None
            
            if current_player == model_color:
                # Model's turn - get full policy output for training
                move, thinking_time, raw_logits, action_probs, value_estimate = ai_engine.get_move_with_policy(
                    board, 
                    session.config.model_level,
                    thinking_time_ms=100
                )
                explanation = f"Model move {move}"
            else:
                # Gemini's turn
                move, explanation = await self.gemini_player.get_move(
                    board,
                    gemini_color,
                    timeout=session.config.gemini_timeout
                )
                thinking_time = 0
            
            # Make the move
            success, notation = board.make_move(move)
            
            if not success:
                # Invalid move - try random legal move
                legal = board.get_legal_moves()
                if legal:
                    move = random.choice(legal) + 1
                    success, notation = board.make_move(move)
                else:
                    break
            
            # Capture state AFTER the move
            state_after = board.get_state_dict()
            state_after["step"] = move_number
            state_after["action"] = move - 1  # 0-indexed for gym
            state_after["player"] = current_player
            state_after["observation"] = board.to_observation()
            states.append(state_after)
            
            # Build comprehensive move record
            move_record = {
                "number": move_number,
                "player": current_player,
                "move": move,
                "action_index": move - 1,  # 0-indexed for gym
                "notation": notation,
                "timestamp": timestamp,
                "explanation": explanation,
                "thinking_time_ms": thinking_time,
                # State data for training
                "observation_before": observation_before,
                "legal_moves_before": legal_moves_before,
                "board_state_before": states[-2] if len(states) >= 2 else None,
            }
            
            # Add policy data if available (model's moves only)
            if raw_logits is not None:
                move_record["policy_logits"] = raw_logits
                move_record["action_probs"] = action_probs
                move_record["value_estimate"] = value_estimate
            
            moves.append(move_record)
            session.current_game_moves = moves.copy()
            
            # Small delay between moves
            await asyncio.sleep(session.config.delay_between_moves)
        
        # Determine winner
        winner = board.winner
        
        # Compute game result for training
        if winner == model_color:
            result_for_model = "win"
            reward_for_model = 1.0
        elif winner == "draw":
            result_for_model = "draw"
            reward_for_model = 0.0
        else:
            result_for_model = "loss"
            reward_for_model = -1.0
        
        # Build comprehensive game log with training data
        game_log = {
            "game_id": game_num,
            "session_id": session.session_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": ai_engine.current_model_name,
            "model_color": model_color,
            "gemini_color": gemini_color,
            "winner": winner,
            "final_score": {
                "white_kazan": board.white_kazan,
                "black_kazan": board.black_kazan
            },
            "total_moves": len(moves),
            "moves": moves,
            # Full states for replay visualization
            "states": states,
            "total_steps": len(states),
            # Training metadata
            "training_data": {
                "source": "gemini_battle",
                "model_level": session.config.model_level,
                "opponent": "gemini-3-flash",
                "result_for_model": result_for_model,
                "reward": reward_for_model,
                "model_moves_count": sum(1 for m in moves if m["player"] == model_color),
                "gemini_moves_count": sum(1 for m in moves if m["player"] == gemini_color),
            }
        }
        
        return game_log
    
    def stop_session(self, session_id: str) -> bool:
        """Signal a session to stop."""
        if session_id in self.sessions:
            self.stop_flags[session_id] = True
            return True
        return False
    
    def get_session_progress(self, session_id: str) -> Optional[Dict]:
        """Get current progress of a session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "status": session.status,
            "games_played": session.games_played,
            "total_games": session.config.num_games,
            "progress_percent": (session.games_played / session.config.num_games) * 100,
            "model_wins": session.model_wins,
            "gemini_wins": session.gemini_wins,
            "draws": session.draws,
            "model_elo": session.model_elo,
            "model_category": get_elo_category(session.model_elo),
            "gemini_elo": session.gemini_elo,
            "elo_history": session.elo_history,
            "current_game_moves": session.current_game_moves,
            "last_game_summary": session.last_game_summary,
            "started_at": session.started_at,
            "finished_at": session.finished_at,
            "error_message": session.error_message
        }
    
    def list_sessions(self) -> List[Dict]:
        """List all sessions with summary info."""
        sessions = []
        for session_id, session in self.sessions.items():
            sessions.append({
                "session_id": session_id,
                "status": session.status,
                "games_played": session.games_played,
                "total_games": session.config.num_games,
                "model_wins": session.model_wins,
                "gemini_wins": session.gemini_wins,
                "draws": session.draws,
                "final_elo": session.model_elo,
                "started_at": session.started_at,
                "finished_at": session.finished_at
            })
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return sessions
    
    def get_elo_chart_data(self, session_id: str) -> Optional[Dict]:
        """Get ELO history formatted for chart display."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Format for chart
        labels = [str(h["game"]) for h in session.elo_history]
        values = [h["elo"] for h in session.elo_history]
        
        # Category thresholds for horizontal lines
        thresholds = [
            {"value": 1000, "label": "Начинающий"},
            {"value": 1200, "label": "Любитель"},
            {"value": 1400, "label": "Клубный"},
            {"value": 1600, "label": "Сильный клубный"},
            {"value": 1800, "label": "КМС"},
            {"value": 2000, "label": "Мастер"},
            {"value": 2200, "label": "ММ"},
        ]
        
        return {
            "labels": labels,
            "values": values,
            "thresholds": thresholds,
            "min_elo": min(values) - 50 if values else 1400,
            "max_elo": max(values) + 50 if values else 1600
        }
    
    def get_summaries(self, session_id: str) -> List[Dict]:
        """Get all summaries for a session."""
        summaries = []
        summaries_dir = os.path.join(self.logs_dir, "summaries")
        
        for filename in os.listdir(summaries_dir):
            if filename.startswith(f"summary_{session_id}_"):
                try:
                    filepath = os.path.join(summaries_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract game number from filename
                    parts = filename.replace(".txt", "").split("_")
                    game_num = int(parts[2]) if len(parts) >= 3 else 0
                    
                    summaries.append({
                        "game_id": game_num,
                        "content": content
                    })
                except Exception:
                    pass
        
        summaries.sort(key=lambda x: x["game_id"])
        return summaries


# Global battle manager instance
gemini_battle_manager = GeminiBattleManager()

