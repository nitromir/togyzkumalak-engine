"""
Gemini Battle System - Model vs LLM Training.
Enables automated training of the neural network against Google Gemini API or Model vs Model.
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


@dataclass
class BattleConfig:
    """Battle session configuration."""
    num_games: int = 10
    player1: str = "active"
    player2: str = "gemini"
    gemini_timeout: int = 30
    gemini_temperature: float = 0.0
    save_replays: bool = True
    delay_between_games: float = 2.0
    delay_between_moves: float = 0.5
    model_level: int = 5 # Legacy


@dataclass
class BattleSession:
    """Battle session state."""
    session_id: str
    config: BattleConfig
    status: str = "pending"
    games_played: int = 0
    model_wins: int = 0
    gemini_wins: int = 0
    draws: int = 0
    model_elo: int = 1500
    gemini_elo: int = 2000
    elo_history: List[Dict] = field(default_factory=list)
    current_game_moves: List[Dict] = field(default_factory=list)
    last_game_summary: str = ""
    started_at: str = ""
    finished_at: str = ""
    error_message: str = ""


ELO_CATEGORIES = {
    (0, 1000): "Новичок", (1000, 1200): "Начинающий", (1200, 1400): "Любитель",
    (1400, 1600): "Клубный игрок", (1600, 1800): "Сильный клубный",
    (1800, 2000): "КМС", (2000, 2200): "Мастер",
    (2200, 2400): "Международный мастер", (2400, 9999): "Гроссмейстер",
}

def get_elo_category(elo: int) -> str:
    for (low, high), name in ELO_CATEGORIES.items():
        if low <= elo < high: return name
    return "Unknown"

def calculate_k_factor(games_played: int) -> int:
    if games_played < 30: return 40
    elif games_played < 100: return 32
    else: return 20

def calculate_elo_change(player_elo: int, opponent_elo: int, result: str, k: int) -> int:
    expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))
    score = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(result, 0.5)
    return int(k * (score - expected))

class GeminiPlayer:
    def __init__(self):
        self.client = None
        self.model = "gemini-3-flash-preview"
        if gemini_config.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=gemini_config.api_key)
                print(f"[Gemini] Initialized with model: {self.model}")
            except Exception as e: print(f"Failed to init Gemini: {e}")

    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        return self.client is not None

    def _format_board_for_move(self, board, color):
        state = board.get_state_dict()
        lines = ["=== РџРћР—РР¦РРЇ РќРђ Р”РћРЎРљР• ==="]
        lines.append(f"Р§РЃР РќР«Р• (ТљР°СЂР°): Р»СѓРЅРєРё [9в†ђ1] = {list(reversed(state['black_pits']))}")
        lines.append(f"  РљР°Р·Р°РЅ С‡С‘СЂРЅС‹С…: {state['black_kazan']}")
        lines.append(f"Р‘Р•Р›Р«Р• (РђТ›): Р»СѓРЅРєРё [1в†’9] = {state['white_pits']}")
        lines.append(f"  РљР°Р·Р°РЅ Р±РµР»С‹С…: {state['white_kazan']}")
        if state['white_tuzduk'] > 0: lines.append(f"* Р‘РµР»С‹Рµ РёРјРµСЋС‚ С‚СѓР·РґСѓ РЅР° Р»СѓРЅРєРµ {state['white_tuzduk']} С‡С‘СЂРЅС‹С…")
        if state['black_tuzduk'] > 0: lines.append(f"* Р§С‘СЂРЅС‹Рµ РёРјРµСЋС‚ С‚СѓР·РґСѓ РЅР° Р»СѓРЅРєРµ {state['black_tuzduk']} Р±РµР»С‹С…")
        return "\n".join(lines)

    async def get_move(self, board, color, timeout=30):
        if not self.client:
            legal = board.get_legal_moves()
            return random.choice(legal) + 1, "Gemini unavailable, random move"
        legal_moves = [m + 1 for m in board.get_legal_moves()]
        position_text = self._format_board_for_move(board, color)
        prompt = f"Р’С‹ РёРіСЂР°РµС‚Рµ РІ РўРѕРіС‹Р· РљСѓРјР°Р»Р°Рє Р·Р° {color.upper()}. {position_text}\nР”РћРЎРўРЈРџРќР«Р• РҐРћР”Р«: {legal_moves}\nРћС‚РІРµС‚СЊС‚Рµ РўРћР›Р¬РљРћ РћР”РќРћР™ Р¦РР¤Р РћР™ РѕС‚ 1 РґРѕ 9."
        try:
            loop = asyncio.get_event_loop()
            async def make_req():
                return await loop.run_in_executor(None, lambda: self.client.models.generate_content(model=self.model, contents=prompt))
            response = await asyncio.wait_for(make_req(), timeout=timeout)
            text = response.text.strip()
            match = re.search(r"(\d)", text)
            if match:
                move = int(match.group(1))
                if move in legal_moves: return move, f"Gemini РІС‹Р±СЂР°Р» {move}"
            return legal_moves[0], f"Invalid response '{text}', using {legal_moves[0]}"
        except Exception as e:
            legal = board.get_legal_moves()
            return random.choice(legal) + 1, f"Error: {e}"

class GeminiBattleManager:
    def __init__(self, logs_dir: str = "logs/gemini_battles"):
        self.sessions: Dict[str, BattleSession] = {}
        self.logs_dir = logs_dir
        self.gemini_player = GeminiPlayer()
        self.stop_flags: Dict[str, bool] = {}
        os.makedirs(os.path.join(logs_dir, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(logs_dir, "games"), exist_ok=True)
        self._load_sessions()

    def _load_sessions(self):
        s_dir = os.path.join(self.logs_dir, "sessions")
        if os.path.exists(s_dir):
            for f_name in os.listdir(s_dir):
                if f_name.endswith(".json"):
                    try:
                        with open(os.path.join(s_dir, f_name), "r", encoding="utf-8") as f:
                            d = json.load(f)
                            s = BattleSession(session_id=d["session_id"], config=BattleConfig(**d["config"]), status=d["status"], games_played=d["games_played"], model_wins=d["model_wins"], gemini_wins=d["gemini_wins"], draws=d["draws"], model_elo=d["model_elo"], gemini_elo=d["gemini_elo"], elo_history=d["elo_history"], started_at=d["started_at"], finished_at=d.get("finished_at", ""))
                            self.sessions[s.session_id] = s
                    except: pass

    def create_session(self, config: BattleConfig) -> str:
        s_id = str(uuid.uuid4())[:8]
        s = BattleSession(session_id=s_id, config=config)
        self.sessions[s_id] = s
        self.stop_flags[s_id] = False
        self._save_session(s)
        return s_id

    def _save_session(self, session: BattleSession):
        path = os.path.join(self.logs_dir, "sessions", f"{session.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**session.__dict__, "config": session.config.__dict__}, f, indent=4)

    def _save_game_log(self, session_id, game_log):
        path = os.path.join(self.logs_dir, "games", f"{session_id}_game_{game_log['game_id']}.json")
        with open(path, "w", encoding="utf-8") as f: json.dump(game_log, f, indent=4)

    async def _play_single_game(self, session, game_num, p1_color):
        board = TogyzkumalakBoard()
        p1_type, p2_type = session.config.player1, session.config.player2
        p2_color = "black" if p1_color == "white" else "white"
        moves, states = [], []
        initial_state = board.get_state_dict()
        initial_state.update({"step": 0, "player": "white", "observation": board.to_observation()})
        states.append(initial_state)
        
        while not board.is_finished:
            if self.stop_flags.get(session.session_id, False): break
            current_player = board.current_player
            active_type = p1_type if current_player == p1_color else p2_type
            move, explanation, thinking_time = 0, "", 0
            
            if active_type == "gemini":
                move, explanation = await self.gemini_player.get_move(board, current_player, timeout=session.config.gemini_timeout)
            elif active_type == "active" or active_type == "default":
                move, thinking_time = ai_engine.get_move(board, 5)
                explanation = f"Active model move {move}"
            elif ":" in active_type:
                m_type, f_name = active_type.split(":", 1)
                if m_type == "probs":
                    orig = ai_engine.probs_model_keeper
                    ckpt = os.path.join("models", "probs", "checkpoints", f_name)
                    if not os.path.exists(ckpt): ckpt = os.path.join("models", "probs", f_name)
                    from .probs_task_manager import probs_task_manager
                    if probs_task_manager.load_checkpoint(ckpt):
                        ai_engine.probs_model_keeper = probs_task_manager.get_loaded_model()
                        move, thinking_time = ai_engine.get_move(board, 7)
                        explanation = f"PROBS {f_name} move {move}"
                    else: move, thinking_time = ai_engine.get_move(board, 2)
                    ai_engine.probs_model_keeper = orig
                elif m_type == "az":
                    orig = ai_engine.alphazero_model
                    ckpt = os.path.join("models", "alphazero", f_name)
                    from .gym_training import training_manager
                    if training_manager.load_model(ckpt):
                        ai_engine.alphazero_model = training_manager.policy_net
                        ai_engine.use_mcts = True
                        move, thinking_time = ai_engine.get_move(board, 5)
                        explanation = f"AZ {f_name} move {move}"
                    else: move, thinking_time = ai_engine.get_move(board, 2)
                    ai_engine.alphazero_model = orig
            
            success, notation = board.make_move(move)
            if not success:
                legal = board.get_legal_moves()
                if legal:
                    move = random.choice(legal) + 1
                    success, notation = board.make_move(move)
                else: break
            
            move_rec = {"number": len(moves)+1, "player": current_player, "move": move, "notation": notation, "explanation": explanation, "thinking_time_ms": thinking_time}
            moves.append(move_rec)
            session.current_game_moves = moves.copy()
            await asyncio.sleep(session.config.delay_between_moves)
            
        return {"game_id": game_num, "winner": board.winner, "final_score": {"white": board.white_kazan, "black": board.black_kazan}, "moves": moves}

    async def run_battle_session(self, s_id):
        s = self.sessions.get(s_id)
        if not s: return
        s.status, s.started_at = "running", datetime.now().isoformat()
        s.elo_history.append({"game": 0, "elo": s.model_elo, "change": 0, "result": "start"})
        try:
            for g_num in range(1, s.config.num_games + 1):
                if self.stop_flags.get(s_id, False):
                    s.status = "stopped"
                    break
                p1_color = "white" if g_num % 2 == 1 else "black"
                game_log = await self._play_single_game(s, g_num, p1_color)
                s.games_played = g_num
                winner = game_log.get("winner")
                if winner == "draw": result, s.draws = "draw", s.draws + 1
                elif winner == p1_color: result, s.model_wins = "win", s.model_wins + 1
                else: result, s.gemini_wins = "loss", s.gemini_wins + 1
                change = calculate_elo_change(s.model_elo, s.gemini_elo, result, calculate_k_factor(s.games_played))
                s.model_elo += change
                s.elo_history.append({"game": g_num, "elo": s.model_elo, "change": change, "result": result})
                if s.config.save_replays: self._save_game_log(s_id, game_log)
                self._save_session(s)
                await asyncio.sleep(s.config.delay_between_games)
            if s.status != "stopped": s.status = "completed"
        except Exception as e:
            s.status, s.error_message = "error", str(e)
            import traceback; traceback.print_exc()
        s.finished_at = datetime.now().isoformat()
        self._save_session(s)

    def stop_session(self, s_id):
        if s_id in self.sessions: self.stop_flags[s_id] = True; return True
        return False

    def get_session_progress(self, s_id):
        s = self.sessions.get(s_id)
        if not s: return None
        return {**s.__dict__, "progress_percent": (s.games_played/s.config.num_games)*100, "model_category": get_elo_category(s.model_elo), "total_games": s.config.num_games}

    def list_sessions(self):
        lst = [{"session_id": s.session_id, "status": s.status, "games_played": s.games_played, "total_games": s.config.num_games, "model_wins": s.model_wins, "gemini_wins": s.gemini_wins, "draws": s.draws, "final_elo": s.model_elo, "started_at": s.started_at} for s in self.sessions.values()]
        lst.sort(key=lambda x: x["started_at"], reverse=True)
        return lst

    def get_elo_chart_data(self, s_id):
        s = self.sessions.get(s_id)
        if not s: return None
        vals = [h["elo"] for h in s.elo_history]
        thresholds = [
            {"value": 1400, "label": "Клубный"},
            {"value": 1800, "label": "КМС"},
            {"value": 2000, "label": "Мастер"},
            {"value": 2200, "label": "ММ"},
            {"value": 2400, "label": "Гроссмейстер"}
        ]
        return {
            "labels": [str(h["game"]) for h in s.elo_history], 
            "values": vals, 
            "thresholds": thresholds,
            "min_elo": min(vals + [1400]) - 50 if vals else 1350, 
            "max_elo": max(vals + [2400]) + 50 if vals else 2450
        }

gemini_battle_manager = GeminiBattleManager()