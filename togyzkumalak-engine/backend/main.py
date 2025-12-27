"""
Togyzkumalak Engine - FastAPI Backend

Main server with REST API and WebSocket support for real-time gameplay.
"""

import asyncio
import datetime
import json
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import server_config, elo_config
from .game_manager import game_manager, TogyzkumalakBoard, GameStatus
from .ai_engine import ai_engine
from .elo_system import elo_system
from .gemini_analyzer import gemini_analyzer
from .gym_training import training_manager, TrainingConfig
from .gemini_battle import gemini_battle_manager, BattleConfig
from .metrics_collector import metrics_collector
from .schema_ab_testing import ab_test_manager
from .wandb_integration import wandb_tracker


# FastAPI app
app = FastAPI(
    title="Togyzkumalak Engine",
    description="AI-powered Togyzkumalak (Toguz Kumalak) game engine",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic Models
# =============================================================================

class NewGameRequest(BaseModel):
    player_color: str = "white"
    ai_level: int = 3
    player_id: Optional[str] = "human"


class MoveRequest(BaseModel):
    move: int  # 1-9


class AnalysisRequest(BaseModel):
    include_history: bool = True


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time gameplay."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        self.active_connections[game_id] = websocket
    
    def disconnect(self, game_id: str):
        if game_id in self.active_connections:
            del self.active_connections[game_id]
    
    async def send_json(self, game_id: str, data: dict):
        websocket = self.active_connections.get(game_id)
        if websocket:
            await websocket.send_json(data)
    
    async def broadcast(self, data: dict):
        for websocket in self.active_connections.values():
            await websocket.send_json(data)


manager = ConnectionManager()


# =============================================================================
# REST API Endpoints
# =============================================================================

import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Togyzkumalak Engine API", "version": "1.0.0"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gemini_available": gemini_analyzer.is_available()
    }


@app.post("/api/games")
async def create_game(request: NewGameRequest):
    """Create a new game."""
    if request.player_color not in ["white", "black"]:
        raise HTTPException(status_code=400, detail="Invalid player color")
    
    if request.ai_level not in range(1, 6):
        raise HTTPException(status_code=400, detail="AI level must be 1-5")
    
    game = game_manager.create_game(
        human_color=request.player_color,
        ai_level=request.ai_level
    )
    
    state = game_manager.get_game_state(game.game_id)
    state["ai_elo"] = elo_system.get_ai_elo(request.ai_level)
    state["player_elo"] = elo_system.get_or_create_player(request.player_id).current_elo
    
    # If human is black, AI makes first move
    if request.player_color == "black":
        board = game_manager.get_board(game.game_id)
        ai_move, thinking_time = ai_engine.get_move(board, request.ai_level)
        success, state = game_manager.make_move(game.game_id, ai_move, thinking_time)
    
    return convert_numpy(state)


@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    """Get game state."""
    state = game_manager.get_game_state(game_id)
    if "error" in state:
        raise HTTPException(status_code=404, detail=state["error"])
    return convert_numpy(state)


@app.post("/api/games/{game_id}/move")
async def make_move(game_id: str, request: MoveRequest):
    """Make a move in a game."""
    game = game_manager.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if game.status != GameStatus.IN_PROGRESS:
        raise HTTPException(status_code=400, detail="Game is not in progress")
    
    board = game_manager.get_board(game_id)
    
    # Validate it's human's turn
    current_player = board.current_player
    if current_player != game.human_color:
        raise HTTPException(status_code=400, detail="Not your turn")
    
    # Validate move is legal
    legal_moves = [m + 1 for m in board.get_legal_moves()]
    if request.move not in legal_moves:
        raise HTTPException(status_code=400, detail=f"Invalid move. Legal moves: {legal_moves}")
    
    # Make human move
    success, state = game_manager.make_move(game_id, request.move)
    if not success:
        raise HTTPException(status_code=400, detail="Move failed")
    
    # If game continues and it's AI's turn, make AI move
    if state["status"] == "in_progress":
        board = game_manager.get_board(game_id)
        if board.current_player != game.human_color:
            ai_move, thinking_time = ai_engine.get_move(board, game.ai_level)
            success, state = game_manager.make_move(game_id, ai_move, thinking_time)
            state["ai_move"] = {
                "move": int(ai_move),
                "thinking_time_ms": int(thinking_time)
            }
    
    # Handle game end
    if state["status"] == "finished":
        result = "win" if state["winner"] == game.human_color else (
            "draw" if state["winner"] == "draw" else "loss"
        )
        player_change, ai_change = elo_system.update_ratings(
            "human",
            f"ai_level_{game.ai_level}",
            result
        )
        state["elo_change"] = player_change
        state["new_elo"] = elo_system.get_or_create_player("human").current_elo
    
    return convert_numpy(state)


@app.get("/api/games/{game_id}/history")
async def get_move_history(game_id: str):
    """Get move history for a game."""
    history = game_manager.get_move_history(game_id)
    if not history and not game_manager.get_game(game_id):
        raise HTTPException(status_code=404, detail="Game not found")
    return {"moves": history}


@app.post("/api/games/{game_id}/resign")
async def resign_game(game_id: str):
    """Resign the current game."""
    game = game_manager.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    state = game_manager.resign(game_id, game.human_color)
    
    # Update ELO
    player_change, _ = elo_system.update_ratings(
        "human",
        f"ai_level_{game.ai_level}",
        "loss"
    )
    state["elo_change"] = player_change
    
    return state


@app.post("/api/games/{game_id}/analyze")
async def analyze_position(game_id: str, request: AnalysisRequest):
    """Get Gemini analysis of current position."""
    board = game_manager.get_board(game_id)
    if not board:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Convert board to dictionary for Gemini analyzer
    board_state = board.get_state_dict()
    history = game_manager.get_move_history(game_id) if request.include_history else None
    
    analysis = await gemini_analyzer.analyze_position(board_state, history)
    return analysis


@app.post("/api/games/{game_id}/suggest")
async def suggest_move(game_id: str):
    """Get Gemini move suggestion."""
    board = game_manager.get_board(game_id)
    if not board:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Convert board to dictionary for Gemini analyzer
    board_state = board.get_state_dict()
    history = game_manager.get_move_history(game_id)
    suggestion = await gemini_analyzer.suggest_move(board_state, history)
    return suggestion


@app.get("/api/elo")
async def get_elo_stats():
    """Get ELO statistics."""
    human_stats = elo_system.get_player_stats("human")
    ai_elos = {f"level_{i}": elo_config.level_elos[i] for i in range(1, 6)}
    
    return {
        "human": human_stats,
        "ai_levels": ai_elos
    }


@app.get("/api/ai/levels")
async def get_ai_levels():
    """Get available AI levels with descriptions."""
    return {
        "levels": [
            {"level": 1, "name": "Random", "elo": 800, "description": "Plays random legal moves"},
            {"level": 2, "name": "Beginner", "elo": 1200, "description": "Simple heuristic-based play"},
            {"level": 3, "name": "Intermediate", "elo": 1500, "description": "Neural network (basic)"},
            {"level": 4, "name": "Advanced", "elo": 1800, "description": "Neural network (deep)"},
            {"level": 5, "name": "Expert", "elo": 2100, "description": "Best available model"}
        ]
    }


@app.get("/api/ai/probabilities/{game_id}")
async def get_move_probabilities(game_id: str, level: int = 3):
    """Get AI move probability distribution for visualization."""
    board = game_manager.get_board(game_id)
    if not board:
        raise HTTPException(status_code=404, detail="Game not found")
    
    probs = ai_engine.get_move_probabilities(board, level)
    evaluation = ai_engine.evaluate_position(board)
    
    return {
        "probabilities": probs,
        "evaluation": evaluation,
        "legal_moves": board.get_legal_moves()
    }


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time gameplay."""
    await manager.connect(websocket, game_id)
    
    try:
        # Send initial game state
        state = game_manager.get_game_state(game_id)
        await websocket.send_json({"type": "game_state", "data": state})
        
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "make_move":
                game = game_manager.get_game(game_id)
                board = game_manager.get_board(game_id)
                
                if not game or not board:
                    await websocket.send_json({"type": "error", "message": "Game not found"})
                    continue
                
                move = data.get("move")
                legal_moves = [m + 1 for m in board.get_legal_moves()]
                
                if move not in legal_moves:
                    await websocket.send_json({
                        "type": "error", 
                        "message": f"Invalid move. Legal: {legal_moves}"
                    })
                    continue
                
                # Make human move
                success, state = game_manager.make_move(game_id, move)
                await websocket.send_json({"type": "move_made", "data": state})
                
                # AI response if needed
                if state["status"] == "in_progress":
                    board = game_manager.get_board(game_id)
                    if board.current_player != game.human_color:
                        # Send "thinking" indicator
                        await websocket.send_json({"type": "ai_thinking"})
                        
                        ai_move, thinking_time = ai_engine.get_move(
                            board, game.ai_level, thinking_time_ms=500
                        )
                        success, state = game_manager.make_move(game_id, ai_move, thinking_time)
                        
                        await websocket.send_json({
                            "type": "ai_move",
                            "data": state,
                            "move": ai_move,
                            "thinking_time": thinking_time
                        })
                
                # Game over
                if state["status"] == "finished":
                    result = "win" if state["winner"] == game.human_color else (
                        "draw" if state["winner"] == "draw" else "loss"
                    )
                    player_change, _ = elo_system.update_ratings(
                        "human", f"ai_level_{game.ai_level}", result
                    )
                    await websocket.send_json({
                        "type": "game_over",
                        "winner": state["winner"],
                        "elo_change": player_change
                    })
            
            elif data["type"] == "request_analysis":
                board = game_manager.get_board(game_id)
                if board:
                    board_state = board.get_state_dict()
                    history = game_manager.get_move_history(game_id)
                    analysis = await gemini_analyzer.analyze_position(board_state, history)
                    await websocket.send_json({"type": "analysis", "data": analysis})
            
            elif data["type"] == "request_suggestion":
                board = game_manager.get_board(game_id)
                if board:
                    board_state = board.get_state_dict()
                    history = game_manager.get_move_history(game_id)
                    suggestion = await gemini_analyzer.suggest_move(board_state, history)
                    await websocket.send_json({"type": "suggestion", "data": suggestion})
            
            elif data["type"] == "resign":
                game = game_manager.get_game(game_id)
                if game:
                    state = game_manager.resign(game_id, game.human_color)
                    player_change, _ = elo_system.update_ratings(
                        "human", f"ai_level_{game.ai_level}", "loss"
                    )
                    await websocket.send_json({
                        "type": "game_over",
                        "winner": state["winner"],
                        "elo_change": player_change,
                        "resigned": True
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(game_id)


# =============================================================================
# Replay API
# =============================================================================

@app.get("/api/replays")
async def get_replays():
    """Get list of available replays from gym simulations."""
    replay_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "..", "..", "visualizer", "replay.json"
    )
    
    if not os.path.exists(replay_file):
        return {"replays": [], "error": "No replay file found"}
    
    try:
        with open(replay_file, 'r', encoding='utf-8') as f:
            replays = json.load(f)
        
        # Return summary of each replay
        summary = []
        for replay in replays:
            summary.append({
                "game_id": replay.get("game_id"),
                "timestamp": replay.get("timestamp"),
                "total_steps": replay.get("total_steps"),
                "winner": replay.get("winner"),
                "final_score": replay.get("final_score")
            })
        
        return {"replays": summary, "total": len(summary)}
    except Exception as e:
        return {"replays": [], "error": str(e)}


@app.get("/api/replays/{game_id}")
async def get_replay(game_id: int):
    """Get full replay data for a specific game."""
    replay_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "..", "..", "visualizer", "replay.json"
    )
    
    if not os.path.exists(replay_file):
        raise HTTPException(status_code=404, detail="No replay file found")
    
    try:
        with open(replay_file, 'r', encoding='utf-8') as f:
            replays = json.load(f)
        
        for replay in replays:
            if replay.get("game_id") == game_id:
                return replay
        
        raise HTTPException(status_code=404, detail=f"Replay {game_id} not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse replay file")


# =============================================================================
# Gym Training API Endpoints
# =============================================================================

class TrainingConfigRequest(BaseModel):
    num_games: int = 10
    epsilon: float = 0.2
    hidden_size: int = 64
    learning_rate: float = 0.001
    save_replays: bool = True
    model_name: str = "policy_net"


class GeminiBattleRequest(BaseModel):
    """Request to start a Gemini battle session."""
    num_games: int = 10
    model_level: int = 5
    gemini_timeout: int = 30
    save_replays: bool = True
    generate_summaries: bool = True


@app.post("/api/training/start")
async def start_training(config: TrainingConfigRequest):
    """Start a new gym training session."""
    try:
        training_config = TrainingConfig(
            num_games=config.num_games,
            epsilon=config.epsilon,
            hidden_size=config.hidden_size,
            learning_rate=config.learning_rate,
            save_replays=config.save_replays,
            model_name=config.model_name
        )
        
        session_id = training_manager.create_session(training_config)
        
        # Run training in background
        asyncio.create_task(
            training_manager.run_training_session(session_id, training_config)
        )
        
        return {
            "session_id": session_id,
            "status": "started",
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/sessions")
async def list_training_sessions():
    """List all training sessions."""
    try:
        sessions = training_manager.list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/sessions/{session_id}")
async def get_training_progress(session_id: str):
    """Get progress of a specific training session."""
    try:
        progress = training_manager.get_session_progress(session_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Session not found")
        return progress
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/models")
async def list_models():
    """List all saved models."""
    try:
        models = training_manager.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a saved model for use in gameplay."""
    try:
        models = training_manager.list_models()
        model_path = None
        
        for model in models:
            if model["name"] == model_name:
                model_path = model["path"]
                break
        
        if not model_path:
            raise HTTPException(status_code=404, detail="Model not found")
        
        success = training_manager.load_model(model_path)
        
        if success:
            return {"status": "loaded", "model": model_name}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Gemini Battle API Endpoints
# =============================================================================

@app.post("/api/gemini-battle/start")
async def start_gemini_battle(request: GeminiBattleRequest):
    """Start a new Gemini battle session."""
    try:
        config = BattleConfig(
            num_games=request.num_games,
            model_level=request.model_level,
            gemini_timeout=request.gemini_timeout,
            save_replays=request.save_replays,
            generate_summaries=request.generate_summaries
        )
        
        session_id = gemini_battle_manager.create_session(config)
        
        # Run battle session in background
        asyncio.create_task(
            gemini_battle_manager.run_battle_session(session_id)
        )
        
        return {
            "session_id": session_id,
            "status": "started",
            "config": {
                "num_games": config.num_games,
                "model_level": config.model_level,
                "gemini_timeout": config.gemini_timeout
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gemini-battle/sessions")
async def list_gemini_battle_sessions():
    """List all Gemini battle sessions."""
    try:
        sessions = gemini_battle_manager.list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gemini-battle/sessions/{session_id}")
async def get_gemini_battle_progress(session_id: str):
    """Get progress of a specific Gemini battle session."""
    try:
        progress = gemini_battle_manager.get_session_progress(session_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Session not found")
        return progress
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gemini-battle/sessions/{session_id}/elo-chart")
async def get_gemini_battle_elo_chart(session_id: str):
    """Get ELO chart data for a session."""
    try:
        chart_data = gemini_battle_manager.get_elo_chart_data(session_id)
        if not chart_data:
            raise HTTPException(status_code=404, detail="Session not found")
        return chart_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gemini-battle/sessions/{session_id}/summaries")
async def get_gemini_battle_summaries(session_id: str):
    """Get all game summaries for a session."""
    try:
        summaries = gemini_battle_manager.get_summaries(session_id)
        return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gemini-battle/sessions/{session_id}/stop")
async def stop_gemini_battle(session_id: str):
    """Stop a running Gemini battle session."""
    try:
        success = gemini_battle_manager.stop_session(session_id)
        if success:
            return {"status": "stopping", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gemini-battle/replays")
async def list_gemini_battle_replays():
    """Get list of all Gemini battle game replays."""
    try:
        games_dir = os.path.join(gemini_battle_manager.logs_dir, "games")
        replays = []
        
        if os.path.exists(games_dir):
            for filename in sorted(os.listdir(games_dir), reverse=True):
                if filename.endswith('.json'):
                    filepath = os.path.join(games_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            game_data = json.load(f)
                        replays.append({
                            "filename": filename,
                            "game_id": game_data.get("game_id"),
                            "session_id": game_data.get("session_id"),
                            "timestamp": game_data.get("timestamp"),
                            "winner": game_data.get("winner"),
                            "model_color": game_data.get("model_color"),
                            "total_moves": game_data.get("total_moves"),
                            "final_score": game_data.get("final_score"),
                            "elo_change": game_data.get("elo_change", 0)
                        })
                    except Exception as e:
                        print(f"Error loading game {filename}: {e}")
        
        return {"replays": replays, "total": len(replays)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gemini-battle/replays/{filename}")
async def get_gemini_battle_replay(filename: str):
    """Get full data for a specific Gemini battle game."""
    try:
        filepath = os.path.join(gemini_battle_manager.logs_dir, "games", filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Replay not found")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        return game_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gemini-battle/summaries")
async def list_gemini_battle_summaries():
    """Get list of all Gemini battle game summaries."""
    try:
        summaries_dir = os.path.join(gemini_battle_manager.logs_dir, "summaries")
        summaries = []
        
        if os.path.exists(summaries_dir):
            for filename in sorted(os.listdir(summaries_dir), reverse=True):
                if filename.endswith('.txt'):
                    filepath = os.path.join(summaries_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # Parse filename: summary_SESSIONID_GAMEID.txt
                        parts = filename.replace('.txt', '').split('_')
                        session_id = parts[1] if len(parts) > 1 else ""
                        game_id = parts[2] if len(parts) > 2 else ""
                        
                        summaries.append({
                            "filename": filename,
                            "session_id": session_id,
                            "game_id": game_id,
                            "content": content[:500] + "..." if len(content) > 500 else content
                        })
                    except Exception as e:
                        print(f"Error loading summary {filename}: {e}")
        
        return {"summaries": summaries, "total": len(summaries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gemini-battle/summaries/{filename}")
async def get_gemini_battle_summary(filename: str):
    """Get full summary for a specific game."""
    try:
        filepath = os.path.join(gemini_battle_manager.logs_dir, "summaries", filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Summary not found")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {"filename": filename, "content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Real Metrics API - NO MOCKS
# =============================================================================

@app.get("/api/metrics/all")
async def get_all_real_metrics():
    """
    Get ALL real metrics from actual files.
    NO MOCKS - all data is computed from real game logs.
    """
    try:
        metrics = metrics_collector.get_all_metrics(force_refresh=True)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/gemini-battles")
async def get_gemini_battle_metrics():
    """Get real Gemini battle metrics from game files."""
    try:
        all_metrics = metrics_collector.get_all_metrics()
        return all_metrics.get("gemini_battles", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/elo")
async def get_real_elo_data():
    """Get real ELO data from session files."""
    try:
        all_metrics = metrics_collector.get_all_metrics()
        return all_metrics.get("elo", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/dataset")
async def get_dataset_stats():
    """Get real dataset composition statistics."""
    try:
        all_metrics = metrics_collector.get_all_metrics()
        return all_metrics.get("dataset", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/convergence")
async def get_convergence_status():
    """Get real convergence status based on ELO history."""
    try:
        all_metrics = metrics_collector.get_all_metrics()
        return all_metrics.get("convergence", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/training-data")
async def get_training_data_from_battles():
    """
    Get real training data from Gemini battle games.
    Returns transitions in format usable for Gym training.
    """
    try:
        transitions = metrics_collector.get_training_data_for_gym()
        return {
            "total_transitions": len(transitions),
            "transitions": transitions[:100],  # First 100 for preview
            "has_more": len(transitions) > 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# A/B Testing API - Real Experiments
# =============================================================================

class ABTestCreateRequest(BaseModel):
    name: str
    variants: List[str] = ["structured", "tactical", "beginner"]
    description: str = ""


class ABTestFeedbackRequest(BaseModel):
    experiment_id: str
    variant: str
    game_id: str
    move_number: int
    user_rating: Optional[int] = None
    was_helpful: Optional[bool] = None
    was_accurate: Optional[bool] = None


@app.post("/api/ab-test/experiments")
async def create_ab_experiment(request: ABTestCreateRequest):
    """Create a new A/B test experiment."""
    try:
        experiment = ab_test_manager.create_experiment(
            name=request.name,
            variants=request.variants,
            description=request.description
        )
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "variants": experiment.variants,
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ab-test/experiments")
async def list_ab_experiments():
    """List all A/B test experiments."""
    try:
        experiments = ab_test_manager.list_experiments()
        return {"experiments": experiments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ab-test/experiments/{experiment_id}/stats")
async def get_ab_experiment_stats(experiment_id: str):
    """Get REAL statistics for an A/B test experiment."""
    try:
        stats = ab_test_manager.get_experiment_stats(experiment_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ab-test/feedback")
async def submit_ab_feedback(request: ABTestFeedbackRequest):
    """Submit user feedback for A/B test (real data collection)."""
    try:
        ab_test_manager.record_feedback(
            experiment_id=request.experiment_id,
            variant=request.variant,
            game_id=request.game_id,
            move_number=request.move_number,
            user_rating=request.user_rating,
            was_helpful=request.was_helpful,
            was_accurate=request.was_accurate
        )
        return {"status": "recorded", "experiment_id": request.experiment_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ab-test/experiments/{experiment_id}/stop")
async def stop_ab_experiment(experiment_id: str):
    """Stop an A/B test experiment."""
    try:
        success = ab_test_manager.stop_experiment(experiment_id)
        if success:
            return {"status": "stopped", "experiment_id": experiment_id}
        raise HTTPException(status_code=404, detail="Experiment not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# W&B / Analytics API
# =============================================================================

@app.post("/api/wandb/start")
async def start_wandb_run(run_name: Optional[str] = None):
    """Start a W&B tracking run."""
    try:
        is_wandb = wandb_tracker.start_run(run_name=run_name)
        return {
            "status": "started",
            "using_wandb": is_wandb,
            "fallback_log": not is_wandb
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/wandb/stop")
async def stop_wandb_run():
    """Stop the current W&B run."""
    try:
        wandb_tracker.finish_run()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/wandb/sync")
async def sync_metrics_to_wandb():
    """Sync all real metrics from files to W&B."""
    try:
        wandb_tracker.sync_from_files(metrics_collector)
        return {"status": "synced"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/wandb/local-metrics")
async def get_local_wandb_metrics(last_n: int = 100):
    """Get local W&B metrics (fallback when W&B unavailable)."""
    try:
        metrics = wandb_tracker.get_local_metrics(last_n=last_n)
        return {"metrics": metrics, "count": len(metrics)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Human Data Import API
# =============================================================================

from fastapi import UploadFile, File
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class ImportDataRequest(BaseModel):
    """Request to import human data for training."""
    opening_book_paths: Optional[List[str]] = None
    championship_path: Optional[str] = None
    playok_path: Optional[str] = None
    output_dir: str = "training_data"


class ParseStatsResponse(BaseModel):
    """Statistics from data parsing."""
    opening_book: int = 0
    human_tournament: int = 0
    playok: int = 0
    total_games: int = 0
    total_moves: int = 0
    total_transitions: int = 0


@app.post("/api/data/parse-all")
async def parse_all_human_data(request: ImportDataRequest):
    """
    Parse all available human training data sources.
    
    This will:
    1. Parse opening book files (open_tree*.txt)
    2. Parse championship games (games.txt)
    3. Parse PlayOK PGN games (all_results_combined.txt)
    4. Convert to unified training format
    5. Save to training_data/ directory
    """
    try:
        from scripts.data_parsers import parse_all_data
        
        stats = parse_all_data(
            opening_book_paths=request.opening_book_paths,
            championship_path=request.championship_path,
            playok_path=request.playok_path,
            output_dir=request.output_dir
        )
        
        return {
            "status": "success",
            "stats": stats,
            "output_dir": request.output_dir
        }
    except ImportError as e:
        # Try alternative import
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "data_parsers",
                os.path.join(os.path.dirname(__file__), "..", "scripts", "data_parsers.py")
            )
            data_parsers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_parsers)
            
            stats = data_parsers.parse_all_data(
                opening_book_paths=request.opening_book_paths,
                championship_path=request.championship_path,
                playok_path=request.playok_path,
                output_dir=request.output_dir
            )
            
            return {
                "status": "success",
                "stats": stats,
                "output_dir": request.output_dir
            }
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Import error: {str(e)} / {str(e2)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/parse-auto")
async def parse_auto_discover():
    """
    Automatically discover and parse all training data from standard locations.
    
    Looks for:
    - Android-APK/assets/internal/open_tree*.txt
    - games.txt
    - all_results_combined.txt
    """
    try:
        from pathlib import Path
        import importlib.util
        
        # Find project root
        backend_dir = Path(__file__).parent
        engine_dir = backend_dir.parent
        project_root = engine_dir.parent.parent  # gym-togyzkumalak-master parent
        
        # Look for files
        opening_books = list(project_root.glob('**/Android-APK/assets/internal/open_tree*.txt'))
        championship = list(project_root.glob('**/games.txt'))
        playok = list(project_root.glob('**/all_results_combined.txt'))
        
        opening_book_paths = [str(p) for p in opening_books] if opening_books else None
        championship_path = str(championship[0]) if championship else None
        playok_path = str(playok[0]) if playok else None
        
        # Import and run parser
        spec = importlib.util.spec_from_file_location(
            "data_parsers",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "data_parsers.py")
        )
        data_parsers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_parsers)
        
        output_dir = os.path.join(str(engine_dir), "training_data")
        
        stats = data_parsers.parse_all_data(
            opening_book_paths=opening_book_paths,
            championship_path=championship_path,
            playok_path=playok_path,
            output_dir=output_dir
        )
        
        return {
            "status": "success",
            "stats": stats,
            "found_files": {
                "opening_books": opening_book_paths or [],
                "championship": championship_path,
                "playok": playok_path
            },
            "output_dir": output_dir
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.get("/api/data/training-files")
async def list_training_files():
    """List available training data files."""
    try:
        engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        training_dir = os.path.join(engine_dir, "training_data")
        
        if not os.path.exists(training_dir):
            return {"files": [], "directory": training_dir, "exists": False}
        
        files = []
        for f in os.listdir(training_dir):
            filepath = os.path.join(training_dir, f)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                # Count lines for jsonl files
                lines = 0
                if f.endswith('.jsonl'):
                    with open(filepath, 'r', encoding='utf-8') as file:
                        lines = sum(1 for _ in file)
                
                files.append({
                    "name": f,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "lines": lines
                })
        
        return {
            "files": files,
            "directory": training_dir,
            "exists": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/stats")
async def get_data_stats():
    """Get statistics about parsed training data."""
    try:
        engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stats_file = os.path.join(engine_dir, "training_data", "parse_stats.json")
        
        if not os.path.exists(stats_file):
            return {
                "parsed": False,
                "message": "No parsed data. Call POST /api/data/parse-auto first."
            }
        
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        return {
            "parsed": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class HumanTrainingRequest(BaseModel):
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    model_name: str = "policy_net_human"
    use_compact: bool = True  # Use compact format (smaller, faster)


@app.post("/api/training/human-data")
async def train_on_human_data(request: HumanTrainingRequest, background_tasks: BackgroundTasks):
    """
    Start training the model on parsed human game data.
    
    This uses behavioral cloning (supervised learning) to train
    the policy network on human expert moves.
    """
    try:
        engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check for training data - prefer compact format
        if request.use_compact:
            data_file = os.path.join(engine_dir, "training_data", "transitions_compact.jsonl")
            if not os.path.exists(data_file):
                # Fall back to full format
                data_file = os.path.join(engine_dir, "training_data", "human_transitions.jsonl")
        else:
            data_file = os.path.join(engine_dir, "training_data", "human_games.jsonl")
        
        if not os.path.exists(data_file):
            raise HTTPException(
                status_code=400, 
                detail=f"Training data not found: {data_file}. Run POST /api/data/parse-auto first."
            )
        
        # Create training session
        session_id = f"human_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start training in background
        background_tasks.add_task(
            run_human_data_training,
            session_id=session_id,
            data_file=data_file,
            batch_size=request.batch_size,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            model_name=request.model_name,
            use_compact=request.use_compact
        )
        
        return {
            "status": "started",
            "session_id": session_id,
            "data_file": data_file,
            "config": {
                "batch_size": request.batch_size,
                "epochs": request.epochs,
                "learning_rate": request.learning_rate,
                "model_name": request.model_name
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Store human training progress
human_training_sessions = {}


async def run_human_data_training(
    session_id: str,
    data_file: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    model_name: str,
    use_compact: bool
):
    """Background task for human data training."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    try:
        human_training_sessions[session_id] = {
            "status": "loading",
            "progress": 0,
            "epoch": 0,
            "total_epochs": epochs,
            "loss": 0,
            "samples_trained": 0,
            "total_samples": 0,
            "start_time": datetime.datetime.now().isoformat()
        }
        
        # Load training data
        print(f"[Human Training] Loading data from {data_file}")
        states = []
        actions = []
        
        is_compact = 'compact' in data_file
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    
                    if is_compact:
                        # Compact format: {"s": [...], "a": N, "r": N, "d": 0/1, "p": 0/1}
                        state = record.get('s', [])
                        action = record.get('a', 0)
                        if len(state) >= 18 and 0 <= action <= 8:
                            states.append(state[:20])
                            actions.append(action)
                    elif 'state' in record:
                        # Full transitions format: {"state": [...], "action": N, ...}
                        state = record.get('state', [])
                        action = record.get('action', 0)
                        if len(state) >= 18 and 0 <= action <= 8:
                            states.append(state[:20])
                            actions.append(action)
                    elif 'moves' in record:
                        # Games format: {"moves": [...]}
                        for move in record.get('moves', []):
                            state = move.get('board_before', {})
                            action = move.get('action', 0)
                            if isinstance(state, dict):
                                white_pits = state.get('white_pits', [0]*9)
                                black_pits = state.get('black_pits', [0]*9)
                                flat_state = white_pits + black_pits
                                flat_state.append(state.get('white_kazan', 0))
                                flat_state.append(state.get('black_kazan', 0))
                            elif isinstance(state, list):
                                flat_state = state
                            else:
                                continue
                            if len(flat_state) >= 18 and 0 <= action <= 8:
                                states.append(flat_state[:20])
                                actions.append(action)
                except json.JSONDecodeError:
                    continue
        
        if len(states) < 100:
            human_training_sessions[session_id]["status"] = "error"
            human_training_sessions[session_id]["error"] = f"Not enough samples: {len(states)}"
            return
        
        print(f"[Human Training] Loaded {len(states)} samples")
        human_training_sessions[session_id]["total_samples"] = len(states)
        human_training_sessions[session_id]["status"] = "training"
        
        # Prepare tensors - pad to 20 features
        padded_states = []
        for s in states:
            if len(s) < 20:
                s = list(s) + [0] * (20 - len(s))
            padded_states.append(s[:20])
        
        X = torch.FloatTensor(padded_states)
        y = torch.LongTensor(actions)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create or load model
        from .ai_engine import ai_engine
        model = ai_engine.policy_net
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        total_batches = len(dataloader) * epochs
        current_batch = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                current_batch += 1
                
                # Update progress
                human_training_sessions[session_id].update({
                    "progress": (current_batch / total_batches) * 100,
                    "epoch": epoch + 1,
                    "loss": epoch_loss / (batch_idx + 1),
                    "accuracy": (correct / total) * 100,
                    "samples_trained": min(total, len(states))
                })
            
            print(f"[Human Training] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}, Acc: {correct/total*100:.2f}%")
        
        # Save model
        engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(engine_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f"{model_name}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_samples': len(states),
            'epochs': epochs,
            'final_loss': human_training_sessions[session_id]["loss"],
            'final_accuracy': human_training_sessions[session_id]["accuracy"],
            'timestamp': datetime.datetime.now().isoformat()
        }, model_path)
        
        human_training_sessions[session_id].update({
            "status": "completed",
            "progress": 100,
            "model_path": model_path,
            "end_time": datetime.datetime.now().isoformat()
        })
        
        print(f"[Human Training] Complete! Model saved to {model_path}")
        
    except Exception as e:
        import traceback
        human_training_sessions[session_id] = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"[Human Training] Error: {e}")


@app.get("/api/training/human-data/{session_id}")
async def get_human_training_progress(session_id: str):
    """Get progress of a human data training session."""
    if session_id not in human_training_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return human_training_sessions[session_id]


@app.get("/api/training/human-data/sessions")
async def list_human_training_sessions():
    """List all human data training sessions."""
    return {"sessions": list(human_training_sessions.keys())}


# =============================================================================
# Static Files (Frontend)
# =============================================================================

# Get frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

# Mount static subdirectories
css_dir = os.path.join(frontend_dir, "css")
js_dir = os.path.join(frontend_dir, "js")
assets_dir = os.path.join(frontend_dir, "assets")

if os.path.exists(css_dir):
    app.mount("/css", StaticFiles(directory=css_dir), name="css")
if os.path.exists(js_dir):
    app.mount("/js", StaticFiles(directory=js_dir), name="js")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


# =============================================================================
# Entry Point
# =============================================================================

def run_server():
    """Run the server."""
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=server_config.host,
        port=server_config.port,
        reload=server_config.debug
    )


if __name__ == "__main__":
    run_server()

