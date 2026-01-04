"""
Togyzkumalak Engine - FastAPI Backend

Main server with REST API and WebSocket support for real-time gameplay.
"""

import asyncio
import datetime
import json
import os
import subprocess
import sys
import signal
import psutil
import psutil
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import server_config, elo_config
from .game_manager import game_manager, TogyzkumalakBoard, GameStatus
from .ai_engine import ai_engine
from .elo_system import elo_system
from .gemini_analyzer import gemini_analyzer
from .voice_service import voice_service
from .gym_training import training_manager, TrainingConfig
from .gemini_battle import gemini_battle_manager, BattleConfig
from .metrics_collector import metrics_collector
from .schema_ab_testing import ab_test_manager
from .wandb_integration import wandb_tracker
from .task_manager import az_task_manager
from .probs_task_manager import probs_task_manager


# Engine directory (for file paths)
engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    
    if request.ai_level not in range(1, 9):
        raise HTTPException(status_code=400, detail="AI level must be 1-8")
    
    game = game_manager.create_game(
        human_color=request.player_color,
        ai_level=request.ai_level
    )
    
    state = game_manager.get_game_state(game.game_id)
    state["ai_elo"] = elo_system.get_ai_elo(request.ai_level)
    state["player_elo"] = elo_system.get_or_create_player(request.player_id).current_elo
    
    # Add model info
    model_info = ai_engine.get_model_info(request.ai_level)
    state["ai_model"] = model_info
    
    # If human is black, AI makes first move
    if request.player_color == "black":
        board = game_manager.get_board(game.game_id)
        ai_move, thinking_time = ai_engine.get_move(board, request.ai_level)
        success, state = game_manager.make_move(game.game_id, ai_move, thinking_time)
        state["ai_model"] = model_info
    
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
            try:
                ai_move, thinking_time = ai_engine.get_move(board, game.ai_level)
                success, state = game_manager.make_move(game_id, ai_move, thinking_time)
                state["ai_move"] = {
                    "move": int(ai_move),
                    "thinking_time_ms": int(thinking_time)
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"AI move failed: {str(e)}")
    
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
    ai_elos = {f"level_{i}": elo_config.level_elos[i] for i in range(1, 9)}
    
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
            {"level": 5, "name": "Expert", "elo": 2100, "description": "Best available model"},
            {"level": 6, "name": "Gemini AI", "elo": 2400, "description": "Google Gemini LLM opponent"},
            {"level": 7, "name": "PROBS AI", "elo": 2200, "description": "PROBS algorithm (Beam Search)"},
            {"level": 8, "name": "Ensemble AI", "elo": 2500, "description": "Combined neural networks + heuristics"}
        ]
    }


@app.get("/api/ai/probabilities/{game_id}")
async def get_move_probabilities(game_id: str, level: int = 3, model: Optional[str] = None):
    """Get AI move probability distribution for visualization."""
    # #region agent log
    import time
    try:
        with open(r"c:\Users\Admin\Documents\Toguzkumalak\debug_final.log", "a") as f:
            f.write(f"API hit for {model} at {time.time()}\n")
    except: pass
    # #endregion
    board = game_manager.get_board(game_id)
    if not board:
        raise HTTPException(status_code=404, detail="Game not found")
    
    probs = ai_engine.get_move_probabilities(board, level, model_type=model)
    evaluation = ai_engine.evaluate_position(board)

    # Ensure JSON-friendly types and stable shape (list[9]) for frontend
    try:
        if isinstance(probs, dict):
            probs_out = [float(probs.get(str(i), probs.get(i, 0.0))) for i in range(9)]
        else:
            probs_out = [float(probs[i]) for i in range(9)]
    except Exception:
        probs_out = [0.0] * 9

    return {
        "probabilities": probs_out,
        "evaluation": float(evaluation) if hasattr(evaluation, "__float__") else evaluation,
        "legal_moves": [int(m) for m in board.get_legal_moves()]
    }


# =============================================================================
# Voice API (TTS & STT)
# =============================================================================

from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
import base64

@app.get("/api/voice/status")
async def voice_status():
    """Check voice services availability."""
    return {
        "tts_available": voice_service.is_tts_available(),
        "stt_available": voice_service.is_stt_available(),
        "tts_model": voice_service.tts_model if voice_service.is_tts_available() else None,
        "stt_model": voice_service.stt_model if voice_service.is_stt_available() else None,
    }


@app.post("/api/voice/tts")
async def text_to_speech(text: str = Form(...)):
    """Convert text to speech. Returns base64-encoded PCM audio."""
    if not voice_service.is_tts_available():
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    audio_data = await voice_service.text_to_speech(text)
    if audio_data is None:
        raise HTTPException(status_code=500, detail="TTS generation failed")
    
    # Return base64-encoded audio
    return {
        "audio": base64.b64encode(audio_data).decode('utf-8'),
        "format": "pcm",
        "sample_rate": 24000,
        "channels": 1,
        "bit_depth": 16
    }


@app.post("/api/voice/tts/stream")
async def text_to_speech_stream(text: str = Form(...)):
    """Stream text-to-speech audio. Returns chunked PCM audio."""
    if not voice_service.is_tts_available():
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    async def generate():
        async for chunk in voice_service.text_to_speech_stream(text):
            # Yield base64 chunk with newline delimiter
            yield base64.b64encode(chunk).decode('utf-8') + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X-Audio-Format": "pcm;rate=24000;channels=1;bits=16"}
    )


@app.post("/api/voice/stt")
async def speech_to_text(file: UploadFile = File(...), language: str = "ru"):
    """Convert speech to text using Groq Whisper."""
    if not voice_service.is_stt_available():
        raise HTTPException(status_code=503, detail="STT service not available")
    
    # Read audio file
    audio_data = await file.read()
    
    # Transcribe
    text = await voice_service.speech_to_text(audio_data, language)
    if text is None:
        raise HTTPException(status_code=500, detail="STT transcription failed")
    
    return {"text": text, "language": language}


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
                        "elo_change": player_change,
                        "board": state["board"]
                    })
            
            elif data["type"] == "request_analysis":
                board = game_manager.get_board(game_id)
                if board:
                    board_state = board.get_state_dict()
                    board_state["legal_moves"] = board.get_legal_moves()
                    history = game_manager.get_move_history(game_id)
                    
                    # Get probabilities from all models
                    model_probs = {
                        "polynet": ai_engine.get_move_probabilities(board, model_type="polynet"),
                        "alphazero": ai_engine.get_move_probabilities(board, model_type="alphazero"),
                        "probs": ai_engine.get_move_probabilities(board, model_type="probs")
                    }
                    
                    await websocket.send_json({"type": "analysis_start"})
                    full_text = ""
                    chunk_count = 0
                    async for chunk in gemini_analyzer.analyze_position_stream(board_state, history, model_probs):
                        chunk_count += 1
                        full_text += chunk
                        print(f"[WebSocket] Sending analysis chunk {chunk_count}: '{chunk[:50]}...' (total_len: {len(full_text)})")
                        await websocket.send_json({"type": "analysis_chunk", "chunk": chunk})
                    print(f"[WebSocket] Analysis complete: {chunk_count} chunks, total length: {len(full_text)}")
                    await websocket.send_json({"type": "analysis_end", "full_text": full_text})
            
            elif data["type"] == "request_suggestion":
                board = game_manager.get_board(game_id)
                if board:
                    board_state = board.get_state_dict()
                    board_state["legal_moves"] = board.get_legal_moves()
                    history = game_manager.get_move_history(game_id)
                    
                    # Get probabilities from all models
                    model_probs = {
                        "polynet": ai_engine.get_move_probabilities(board, model_type="polynet"),
                        "alphazero": ai_engine.get_move_probabilities(board, model_type="alphazero"),
                        "probs": ai_engine.get_move_probabilities(board, model_type="probs")
                    }
                    
                    await websocket.send_json({"type": "suggestion_start"})
                    full_text = ""
                    async for chunk in gemini_analyzer.suggest_move_stream(board_state, history, model_probs):
                        full_text += chunk
                        await websocket.send_json({"type": "suggestion_chunk", "chunk": chunk})
                    await websocket.send_json({"type": "suggestion_end", "full_text": full_text})
            
            elif data["type"] == "voice_query":
                # Handle voice conversation with AI
                board = game_manager.get_board(game_id)
                if board:
                    board_state = board.get_state_dict()
                    board_state["legal_moves"] = board.get_legal_moves()
                    history = game_manager.get_move_history(game_id)
                    
                    user_query = data.get("query", "")
                    context = data.get("context", "")
                    
                    await websocket.send_json({"type": "analysis_start"})
                    full_text = ""
                    async for chunk in gemini_analyzer.voice_conversation_stream(
                        user_query, context, board_state, history
                    ):
                        full_text += chunk
                        await websocket.send_json({"type": "analysis_chunk", "chunk": chunk})
                    await websocket.send_json({"type": "analysis_end", "full_text": full_text})
            
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
                        "resigned": True,
                        "board": state["board"]
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(game_id)


# =============================================================================
# Replay API
# =============================================================================

@app.get("/api/replays")
async def get_replays():
    """Get list of available replays from gym simulations."""
    # Try multiple possible paths for the replay file
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "visualizer", "replay.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visualizer", "replay.json"),
        "visualizer/replay.json"
    ]

    replay_file = None
    for path in possible_paths:
        if os.path.exists(path):
            replay_file = path
            break

    if not replay_file:
        return {"replays": [], "error": "No replay file found"}
    
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
    player1: str = "active"
    player2: str = "gemini"
    gemini_timeout: int = 30
    save_replays: bool = True
    model_level: int = 5 # Legacy support
    generate_summaries: bool = True # Legacy support


class AlphaZeroTrainingRequest(BaseModel):
    numIters: int = 100
    numEps: int = 100
    numMCTSSims: int = 100
    cpuct: float = 1.0
    batch_size: int = 256
    hidden_size: int = 256
    epochs: int = 10
    use_bootstrap: bool = True
    use_multiprocessing: bool = True
    num_parallel_games: int = 0  # 0 = auto
    num_workers: int = 0         # 0 = auto
    save_every_n_iters: int = 5
    resume_from_checkpoint: bool = True
    initial_checkpoint: Optional[str] = None


@app.post("/api/training/alphazero/start")
async def start_alphazero_training(request: AlphaZeroTrainingRequest):
    """Start AlphaZero self-play training."""
    try:
        task_id = az_task_manager.start_training(request.dict())
        return {"task_id": task_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/alphazero/sessions")
async def list_alphazero_sessions():
    """List all AlphaZero sessions."""
    return {"sessions": az_task_manager.list_tasks()}


@app.get("/api/training/alphazero/sessions/{task_id}")
async def get_alphazero_status(task_id: str):
    """Get status of an AlphaZero session."""
    status = az_task_manager.get_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@app.post("/api/training/alphazero/sessions/{task_id}/stop")
async def stop_alphazero_training(task_id: str):
    """Stop an AlphaZero session."""
    success = az_task_manager.stop_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "stopping"}


@app.get("/api/training/alphazero/metrics")
async def get_alphazero_metrics():
    """Get AlphaZero training metrics and checkpoints."""
    try:
        metrics_file = os.path.join(engine_dir, "models", "alphazero", "training_metrics.json")
        if not os.path.exists(metrics_file):
            return {"metrics": [], "config": {}, "summary": {"status": "no_training"}, "checkpoints": []}
        
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        metrics = data.get("metrics", [])
        config = data.get("config", {})
        
        # Find available checkpoints
        checkpoints = []
        alphazero_dir = os.path.join(engine_dir, "models", "alphazero")
        
        for m in metrics:
            iter_num = m.get("iteration", 0)
            checkpoint_file = f"checkpoint_{iter_num}.pth.tar"
            checkpoint_path = os.path.join(alphazero_dir, checkpoint_file)
            
            if os.path.exists(checkpoint_path):
                stat = os.stat(checkpoint_path)
                checkpoints.append({
                    "iteration": iter_num,
                    "filename": checkpoint_file,
                    "policy_loss": m.get("policy_loss", 0),
                    "value_loss": m.get("value_loss", 0),
                    "win_rate": m.get("win_rate", 0),
                    "accepted": m.get("accepted", False),
                    "timestamp": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by policy_loss (best first)
        checkpoints_sorted = sorted(checkpoints, key=lambda x: x["policy_loss"])
        best_checkpoint = checkpoints_sorted[0] if checkpoints_sorted else None
        
        summary = {
            "status": "completed" if metrics else "no_metrics",
            "total_iterations": len(metrics),
            "latest_policy_loss": metrics[-1].get("policy_loss", 0) if metrics else 0,
            "latest_value_loss": metrics[-1].get("value_loss", 0) if metrics else 0,
            "latest_win_rate": metrics[-1].get("win_rate", 0) if metrics else 0,
            "total_examples": metrics[-1].get("total_examples", 0) if metrics else 0,
            "best_checkpoint": best_checkpoint
        }
        
        return {"metrics": metrics, "config": config, "summary": summary, "checkpoints": checkpoints_sorted}  # Показываем ВСЕ чекпойнты
    except Exception as e:
        return {"error": str(e), "metrics": [], "config": {}, "checkpoints": []}


@app.post("/api/training/models/alphazero/{checkpoint_name}/load")
async def load_alphazero_checkpoint(checkpoint_name: str, use_mcts: bool = True):
    """Load a specific AlphaZero checkpoint."""
    try:
        checkpoint_file = checkpoint_name if checkpoint_name.endswith('.pth.tar') else f"{checkpoint_name}.pth.tar"
        checkpoint_path = os.path.join(engine_dir, "models", "alphazero", checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_file}")
        
        success = training_manager.load_model(checkpoint_path)
        if success:
            from .ai_engine import ai_engine
            ai_engine.use_mcts = use_mcts
            return {
                "status": "success", 
                "loaded": checkpoint_file,
                "use_mcts": ai_engine.use_mcts
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load checkpoint")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/alphazero/checkpoints")
async def list_alphazero_checkpoints():
    """List all AlphaZero checkpoints with their metrics."""
    try:
        checkpoints = az_task_manager.get_checkpoints()
        
        # Also load metrics file for detailed info
        metrics_file = os.path.join(engine_dir, "models", "alphazero", "training_metrics.json")
        best_iteration = None
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                best_iteration = data.get("best_iteration", {})
        
        return {
            "checkpoints": checkpoints,
            "total": len(checkpoints),
            "best_iteration": best_iteration
        }
    except Exception as e:
        return {"checkpoints": [], "total": 0, "error": str(e)}


@app.post("/api/training/alphazero/checkpoints/{checkpoint_name}/rename")
async def rename_alphazero_checkpoint(checkpoint_name: str, new_name: str):
    """Rename a specific AlphaZero checkpoint file."""
    try:
        success = az_task_manager.rename_checkpoint(checkpoint_name, new_name)
        if success:
            return {"status": "success", "old_name": checkpoint_name, "new_name": new_name}
        else:
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/alphazero/tournament/start")
async def start_alphazero_tournament(num_games: int = 20):
    """Start a tournament between all checkpoints."""
    try:
        task_id = az_task_manager.start_tournament(num_games)
        return {"task_id": task_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/alphazero/tournament/sessions")
async def list_alphazero_tournaments():
    """List all AlphaZero tournament sessions."""
    return {"sessions": az_task_manager.list_tournaments()}


@app.get("/api/training/alphazero/tournament/sessions/{task_id}")
async def get_alphazero_tournament_status(task_id: str):
    """Get status of an AlphaZero tournament session."""
    status = az_task_manager.get_tournament_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Tournament task not found")
    return status


@app.get("/api/training/alphazero/checkpoints/{checkpoint_name}/download")
async def download_alphazero_checkpoint(checkpoint_name: str):
    """Download a specific AlphaZero checkpoint file."""
    try:
        checkpoint_file = checkpoint_name if checkpoint_name.endswith('.pth.tar') else f"{checkpoint_name}.pth.tar"
        checkpoint_path = os.path.join(engine_dir, "models", "alphazero", checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_file}")
        
        return FileResponse(
            path=checkpoint_path,
            filename=checkpoint_file,
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/alphazero/optimal-config")
async def get_optimal_alphazero_config(gpus: int = 1, hours: float = 1.0):
    """
    Get optimal AlphaZero training configuration based on available GPUs and time.
    
    Args:
        gpus: Number of available GPUs (default: 1)
        hours: Available training time in hours (default: 1.0)
    
    Returns:
        Recommended configuration with estimated training time
    """
    try:
        from .alphazero_trainer import get_optimal_config, NUM_GPUS, device
        
        # Use actual detected GPUs if not specified or if requesting auto-detect
        actual_gpus = gpus if gpus > 0 else max(1, NUM_GPUS)
        
        config = get_optimal_config(actual_gpus, hours)
        
        return {
            "recommended_config": config,
            "detected_gpus": NUM_GPUS,
            "device": str(device),
            "estimated_iterations": config.get("numIters", 100),
            "estimated_time_minutes": config.get("estimated_time_min", 60),
            "tips": [
                f"With {actual_gpus} GPU(s), you can run ~{config.get('numIters', 100)} iterations in {hours} hour(s)",
                f"Batch size scaled to {config.get('batch_size', 256)} for optimal GPU utilization",
                f"Using {config.get('num_parallel_games', 8)} parallel self-play games",
                "Enable 'use_bootstrap' for faster convergence with human game data",
                "Checkpoints are saved every few iterations - download them during training!"
            ]
        }
    except Exception as e:
        return {
            "error": str(e),
            "recommended_config": {
                "numIters": 100,
                "numEps": 100,
                "numMCTSSims": 100,
                "batch_size": 256,
                "use_bootstrap": True
            }
        }


@app.get("/api/training/alphazero/gpu-info")
async def get_gpu_info():
    """Get information about available GPUs."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "device": "cpu",
                "gpus": [],
                "message": "CUDA not available, training will use CPU"
            }
        
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": round(props.total_memory / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}"
            })
        
        return {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "device": "cuda",
            "gpu_count": len(gpus),
            "gpus": gpus
        }
    except Exception as e:
        return {
            "cuda_available": False,
            "error": str(e)
        }


@app.get("/api/system/gpu-utilization")
async def get_gpu_utilization():
    """Get real-time GPU utilization and memory using nvidia-smi."""
    try:
        import subprocess
        
        # Call nvidia-smi to get utilization and memory usage
        # index, utilization.gpu [%], memory.used [MiB], memory.total [MiB], name
        cmd = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,name", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line: continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "utilization": int(parts[1]),
                    "memory_used_mib": int(parts[2]),
                    "memory_total_mib": int(parts[3]),
                    "name": parts[4]
                })
        
        return {
            "status": "ok",
            "gpus": gpus,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/api/system/cpu-utilization")
async def get_cpu_utilization():
    """Get real-time CPU utilization and system memory."""
    try:
        return {
            "status": "ok",
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(),
            "memory_percent": psutil.virtual_memory().percent,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/api/ai/model-info")
async def get_ai_model_info(level: int = 5):
    """Get info about the currently active AI model."""
    return ai_engine.get_model_info(level)


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
async def load_model(model_name: str, use_mcts: bool = True):
    """Load a saved model for use in gameplay."""
    try:
        models = training_manager.list_models()
        model_info = None
        
        for model in models:
            if model["name"] == model_name:
                model_info = model
                break
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")
        
        success = training_manager.load_model(model_info["path"])
        
        if success:
            from .ai_engine import ai_engine
            
            # Get hidden_size from the loaded model
            hidden_size = 64
            if training_manager.policy_net is not None:
                try:
                    hidden_size = next(training_manager.policy_net.parameters()).shape[0]
                except:
                    pass
            
            # Set MCTS mode for AlphaZero models
            model_type = model_info.get("type", "alphazero")
            if model_type == "alphazero":
                # Default to MCTS if use_mcts is True or not specified
                ai_engine.use_mcts = use_mcts
            else:
                ai_engine.use_mcts = False
                    
            return {
                "status": "loaded", 
                "model": model_name,
                "hidden_size": hidden_size,
                "type": model_type,
                "use_mcts": ai_engine.use_mcts
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/training/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a saved model from disk."""
    try:
        models = training_manager.list_models()
        model_path = None
        
        for model in models:
            if model["name"] == model_name:
                model_path = model["path"]
                break
        
        if not model_path:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Delete file
        if os.path.exists(model_path):
            os.remove(model_path)
            return {"status": "deleted", "model": model_name}
        else:
            raise HTTPException(status_code=404, detail="Model file not found on disk")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/models/active")
async def get_active_model():
    """Get the name of the currently active model."""
    try:
        from .ai_engine import ai_engine
        return {
            "model": ai_engine.current_model_name,
            "use_mcts": ai_engine.use_mcts,
            "has_alphazero": ai_engine.alphazero_model is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/models/mcts")
async def toggle_mcts(enabled: bool = True):
    """Toggle MCTS mode for AlphaZero models."""
    try:
        from .ai_engine import ai_engine
        
        if enabled and ai_engine.alphazero_model is None:
            raise HTTPException(
                status_code=400, 
                detail="No AlphaZero model loaded. Load an AlphaZero model first."
            )
        
        ai_engine.use_mcts = enabled
        return {
            "status": "ok",
            "use_mcts": ai_engine.use_mcts,
            "model": ai_engine.current_model_name
        }
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
            player1=request.player1,
            player2=request.player2,
            gemini_timeout=request.gemini_timeout,
            save_replays=request.save_replays
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


@app.post("/api/gemini-battle/export-training-data")
async def export_gemini_training_data():
    """
    Export Gemini battle games to training data format.
    Converts all saved game logs to transitions for training.
    """
    try:
        # Games are in logs_dir/games subdirectory
        games_dir = os.path.join(gemini_battle_manager.logs_dir, "games")
        
        if not os.path.exists(games_dir):
            return {"status": "no_data", "message": "No Gemini battle games found", "games_exported": 0}
        
        transitions = []
        games_processed = 0
        
        # Initial board state for reconstruction
        initial_pits = [9] * 18  # 9 stones in each pit
        
        for filename in os.listdir(games_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(games_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    game_log = json.load(f)
                
                moves = game_log.get("moves", [])
                winner = game_log.get("winner")
                model_side = game_log.get("model_side", 0)
                
                # Skip games with no moves
                if not moves:
                    continue
                
                # Reconstruct states from moves
                # This is a simplified version - just record (action, reward)
                for i, move in enumerate(moves):
                    player = move.get("player", 0)
                    action = move.get("action", 0) - 1  # Convert 1-9 to 0-8
                    
                    # Calculate reward (1 for win, -1 for loss, 0 for ongoing)
                    if i == len(moves) - 1:  # Last move
                        if winner == model_side:
                            reward = 1.0
                        elif winner == 1 - model_side:
                            reward = -1.0
                        else:
                            reward = 0.0
                    else:
                        reward = 0.0
                    
                    # Create simple state representation (20 features)
                    state = [0.0] * 20
                    state[action] = 1.0  # One-hot for action taken
                    state[9 + player] = 1.0  # Player indicator
                    
                    transitions.append({
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "source": "gemini_battle"
                    })
                
                games_processed += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # Save to training data
        engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        training_dir = os.path.join(engine_dir, "training_data")
        os.makedirs(training_dir, exist_ok=True)
        
        output_file = os.path.join(training_dir, "gemini_transitions.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for t in transitions:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')
        
        return {
            "status": "success",
            "games_exported": games_processed,
            "transitions_created": len(transitions),
            "output_file": output_file
        }
        
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


@app.post("/api/data/upload-training-file")
async def upload_training_file(file: UploadFile = File(...)):
    """
    Upload a training data file (.jsonl or .json) directly to the server.
    Files are saved to training_data/ directory.
    """
    try:
        # Validate file extension
        if not file.filename.endswith(('.jsonl', '.json')):
            raise HTTPException(status_code=400, detail="Only .jsonl and .json files are allowed")
        
        # Create training_data directory if not exists
        training_dir = os.path.join(engine_dir, "training_data")
        os.makedirs(training_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(training_dir, file.filename)
        
        # Read and save content
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Count lines for .jsonl files
        line_count = 0
        file_size = len(content)
        if file.filename.endswith('.jsonl'):
            line_count = content.count(b'\n')
        
        return {
            "status": "success",
            "filename": file.filename,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "lines": line_count,
            "path": file_path
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.get("/api/data/training-files-status")
async def get_training_files_status():
    """
    Get status of all training data files including presence and sizes.
    """
    try:
        training_dir = os.path.join(engine_dir, "training_data")
        
        files_status = {
            "transitions_compact.jsonl": {"exists": False, "size_mb": 0, "lines": 0},
            "human_transitions.jsonl": {"exists": False, "size_mb": 0, "lines": 0},
            "human_games.jsonl": {"exists": False, "size_mb": 0, "lines": 0},
            "android_apk_openings.jsonl": {"exists": False, "size_mb": 0, "lines": 0},
            "gemini_transitions.jsonl": {"exists": False, "size_mb": 0, "lines": 0},
        }
        
        if os.path.exists(training_dir):
            for filename in files_status:
                filepath = os.path.join(training_dir, filename)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    lines = 0
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = sum(1 for _ in f)
                    except:
                        pass
                    files_status[filename] = {
                        "exists": True,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "lines": lines
                    }
        
        # Calculate totals
        total_size = sum(f["size_mb"] for f in files_status.values())
        total_lines = sum(f["lines"] for f in files_status.values())
        
        return {
            "training_dir": training_dir,
            "files": files_status,
            "total_size_mb": round(total_size, 2),
            "total_lines": total_lines,
            "bootstrap_ready": files_status["transitions_compact.jsonl"]["exists"]
        }
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
    base_model: str = ""  # If provided, load this model as starting point


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
            use_compact=request.use_compact,
            base_model=request.base_model
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
    use_compact: bool,
    base_model: str = ""
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
            "start_time": datetime.datetime.now().isoformat(),
            "base_model": base_model
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
        
        # Prepare tensors - pad to 128 features to match PolicyNetwork input
        padded_states = []
        for s in states:
            s = list(s)
            if len(s) < 128:
                s = s + [0.0] * (128 - len(s))
            padded_states.append(s[:128])
        
        X = torch.FloatTensor(padded_states)
        y = torch.LongTensor(actions)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create or load model - use the highest level model from AIEngine
        from .ai_engine import ai_engine, PolicyNetwork
        
        model_info = ""
        
        # Load base model if specified, otherwise create new
        if base_model:
            print(f"[Human Training] Loading base model: {base_model}")
            # Find the base model file
            engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(engine_dir, "models")
            base_path = os.path.join(models_dir, f"{base_model}.pt")
            
            if os.path.exists(base_path):
                checkpoint = torch.load(base_path, map_location="cpu")
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Detect architecture from state dict
                first_layer = state_dict.get('network.0.weight')
                if first_layer is not None:
                    hidden_size = first_layer.shape[0]
                else:
                    hidden_size = 256
                
                model = PolicyNetwork(input_size=128, hidden_size=hidden_size, output_size=9)
                model.load_state_dict(state_dict)
                ai_engine.models[5] = model
                model_info = f"Дообучение на базе {base_model} (hidden={hidden_size})"
                print(f"[Human Training] Loaded base model with hidden_size={hidden_size}")
            else:
                model_info = f"Базовая модель {base_model} не найдена, создана новая (hidden=256)"
                print(f"[Human Training] Base model not found: {base_path}, creating new")
                model = PolicyNetwork(input_size=128, hidden_size=256, output_size=9)
                ai_engine.models[5] = model
        elif 5 in ai_engine.models:
            model = ai_engine.models[5]
            try:
                current_hidden = next(model.parameters()).shape[0]
                model_info = f"Продолжение обучения текущей модели (hidden={current_hidden})"
            except:
                model_info = "Продолжение обучения текущей модели"
        else:
            # Create new model with standard input size
            model_info = "Создана новая модель (hidden=256)"
            model = PolicyNetwork(input_size=128, hidden_size=256, output_size=9)
            ai_engine.models[5] = model
        
        human_training_sessions[session_id]["model_info"] = model_info
        model.train()  # Set to training mode
        
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
        
        # Save model with auto-versioning
        engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(engine_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Find next version number
        version = 1
        while os.path.exists(os.path.join(models_dir, f"{model_name}_v{version}.pt")):
            version += 1
        
        versioned_name = f"{model_name}_v{version}"
        model_path = os.path.join(models_dir, f"{versioned_name}.pt")
        
        final_accuracy = human_training_sessions[session_id].get("accuracy", 0)
        final_loss = human_training_sessions[session_id].get("loss", 0)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_samples': len(states),
            'epochs': epochs,
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
            'version': version,
            'timestamp': datetime.datetime.now().isoformat()
        }, model_path)
        
        print(f"[Human Training] Model saved as version {version}")
        
        human_training_sessions[session_id].update({
            "status": "completed",
            "progress": 100,
            "model_path": model_path,
            "model_name": versioned_name,
            "version": version,
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
# System Management
# =============================================================================

@app.post("/api/system/update-and-restart")
async def update_and_restart():
    """
    Update code from GitHub and restart the server.
    This endpoint:
    1. Pulls latest changes from GitHub
    2. Restarts the server process
    """
    try:
        # Get the repository root directory - try multiple paths
        possible_roots = [
            os.path.dirname(os.path.dirname(engine_dir)),  # /workspace/togyzkumalak
            engine_dir,  # /workspace/togyzkumalak/togyzkumalak-engine
            os.path.dirname(engine_dir),  # /workspace/togyzkumalak (if engine_dir is wrong)
            '/workspace/togyzkumalak',  # Direct path
            os.getcwd(),  # Current working directory
        ]
        
        repo_root = None
        git_dir = None
        
        for root in possible_roots:
            if root and os.path.exists(root):
                test_git = os.path.join(root, ".git")
                if os.path.exists(test_git):
                    repo_root = root
                    git_dir = test_git
                    break
        
        if not repo_root or not os.path.exists(git_dir):
            # Last resort: search from engine_dir up
            current = engine_dir
            for _ in range(5):  # Max 5 levels up
                test_git = os.path.join(current, ".git")
                if os.path.exists(test_git):
                    repo_root = current
                    git_dir = test_git
                    break
                parent = os.path.dirname(current)
                if parent == current:  # Reached root
                    break
                current = parent
        
        if not repo_root or not os.path.exists(git_dir):
            return {
                "success": False,
                "error": f"Git repository not found. Checked paths: {possible_roots}. Current engine_dir: {engine_dir}",
                "repo_root": repo_root,
                "engine_dir": engine_dir,
                "cwd": os.getcwd()
            }
        
        # Change to repository root
        os.chdir(repo_root)
        
        # Check for uncommitted changes first
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        has_changes = bool(status_result.stdout.strip())
        if has_changes:
            # Reset any local changes to avoid conflicts
            subprocess.run(
                ["git", "checkout", "."],
                capture_output=True,
                timeout=10
            )
            # Also try to clean untracked files in deploy directory
            subprocess.run(
                ["git", "clean", "-fd", "togyzkumalak-engine/deploy/"],
                capture_output=True,
                timeout=10
            )
        
        # Pull latest changes
        result = subprocess.run(
            ["git", "pull", "origin", "master"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            # Try hard reset if pull fails
            subprocess.run(
                ["git", "reset", "--hard", "origin/master"],
                capture_output=True,
                timeout=10
            )
            # Try pull again
            result = subprocess.run(
                ["git", "pull", "origin", "master"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Git pull failed: {result.stderr}",
                    "stdout": result.stdout
                }
        
        # Schedule restart (give time for response to be sent)
        async def delayed_restart():
            await asyncio.sleep(2)  # Wait 2 seconds for response
            
            # Restart by replacing the current process with a new one
            try:
                # Use absolute path for the entry point script
                run_py_path = os.path.join(os.path.dirname(engine_dir), 'run.py')
                # Reconstruct full command with absolute path to run.py
                restart_args = [sys.executable, run_py_path]
                
                print(f"🔄 RESTARTING SERVER: {' '.join(restart_args)}")
                # Ensure we are in the engine directory for correct imports
                os.chdir(os.path.dirname(engine_dir))
                os.execv(sys.executable, restart_args)
            except Exception as e:
                print(f"❌ Restart failed: {e}")
                os._exit(1)
        
        # Always restart if requested, or if changes were detected
        asyncio.create_task(delayed_restart())
        
        return {
            "success": True,
            "message": "Code updated successfully." if has_changes else "System up to date. Restarting anyway to apply all changes...",
            "git_output": result.stdout,
            "restarting": True
        }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Git pull timed out after 60 seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Update failed: {str(e)}"
        }


@app.get("/api/training/alphazero/logs")
async def get_alphazero_logs(task_id: Optional[str] = None, lines: int = 200):
    """
    Get training logs for AlphaZero.
    
    Args:
        task_id: Optional task ID to filter logs
        lines: Number of lines to return (default: 200)
    
    Returns:
        Training logs from server_error.log and server.log
    """
    try:
        logs = {
            "errors": [],
            "output": [],
            "status": "ok"
        }
        
        # Get error logs
        error_log_path = os.path.join(engine_dir, "server_error.log")
        if os.path.exists(error_log_path):
            with open(error_log_path, 'r') as f:
                error_lines = f.readlines()
                # Filter for AlphaZero related lines
                relevant_errors = [
                    line.rstrip() for line in error_lines[-lines:]
                    if any(keyword in line.lower() for keyword in 
                          ['alphazero', 'training', 'error', 'iteration', 'episode', 'mcts', 'warning'])
                ]
                logs["errors"] = relevant_errors[-lines:]
        
        # Get output logs
        server_log_path = os.path.join(engine_dir, "server.log")
        if os.path.exists(server_log_path):
            with open(server_log_path, 'r') as f:
                output_lines = f.readlines()
                # Filter for AlphaZero related lines
                relevant_output = [
                    line.rstrip() for line in output_lines[-lines:]
                    if any(keyword in line.lower() for keyword in 
                          ['alphazero', 'training', 'iteration', 'self-play', 'episode', 'mcts'])
                ]
                logs["output"] = relevant_output[-lines:]
        
        # Get AlphaZero specific training log
        az_log_path = os.path.join(engine_dir, "alphazero_training.log")
        if os.path.exists(az_log_path):
            with open(az_log_path, 'r') as f:
                az_lines = f.readlines()
                # Include everything from the training log
                logs["training"] = [line.strip() for line in az_lines[-lines:]]
        
        # Get current task status if task_id provided
        if task_id:
            status = az_task_manager.get_status(task_id)
            if status:
                logs["task_status"] = status
        
        return logs
        
    except Exception as e:
        return {
            "errors": [],
            "output": [],
            "status": "error",
            "error": str(e)
        }


# =============================================================================
# PROBS Training API Endpoints
# =============================================================================

class PROBSTrainingRequest(BaseModel):
    n_high_level_iterations: int = 10
    v_train_episodes: int = 5
    q_train_episodes: int = 5
    mem_max_episodes: int = 500
    train_batch_size: int = 32
    self_learning_batch_size: int = 32
    num_q_s_a_calls: int = 10
    max_depth: int = 8
    self_play_threads: int = 2
    sub_processes_cnt: int = 0
    evaluate_n_games: int = 5
    device: str = "cpu"
    use_boost: bool = False
    initial_checkpoint: Optional[str] = None


@app.post("/api/training/probs/start")
async def start_probs_training(request: PROBSTrainingRequest):
    """Start PROBS training session."""
    try:
        task_id = probs_task_manager.start_training(request.dict())
        return {"task_id": task_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/probs/sessions")
async def list_probs_sessions():
    """List all PROBS training sessions."""
    return {"sessions": probs_task_manager.list_tasks()}


@app.get("/api/training/probs/sessions/{task_id}")
async def get_probs_status(task_id: str):
    """Get status of a PROBS training session."""
    status = probs_task_manager.get_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@app.post("/api/training/probs/sessions/{task_id}/stop")
async def stop_probs_training(task_id: str):
    """Stop a PROBS training session."""
    success = probs_task_manager.stop_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "stopping"}


@app.get("/api/training/probs/checkpoints")
async def list_probs_checkpoints():
    """List all PROBS checkpoints."""
    try:
        checkpoints = probs_task_manager.get_checkpoints()
        best_info = probs_task_manager.get_best_info()
        return {"checkpoints": checkpoints, "total": len(checkpoints), "best_checkpoint": best_info}
    except Exception as e:
        return {"checkpoints": [], "total": 0, "error": str(e)}


@app.post("/api/training/probs/tournament/start")
async def start_probs_tournament(num_games: int = 20):
    """Start a PROBS tournament between all checkpoints."""
    try:
        tournament_id = probs_task_manager.start_tournament(num_games)
        return {"task_id": tournament_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/probs/tournament/results")
async def get_probs_tournament_results():
    """Get latest PROBS tournament results."""
    try:
        results = probs_task_manager.get_tournament_results()
        if results:
            return results
        return {"error": "No tournament results found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/probs/checkpoints/{checkpoint_name}/load")
async def load_probs_checkpoint(checkpoint_name: str):
    """Load a specific PROBS checkpoint for gameplay."""
    try:
        checkpoint_file = checkpoint_name if checkpoint_name.endswith('.ckpt') else f"{checkpoint_name}.ckpt"
        checkpoint_path = os.path.join(engine_dir, "models", "probs", "checkpoints", checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_file}")
        
        success = probs_task_manager.load_checkpoint(checkpoint_path)
        if success:
            # Also update the ai_engine
            loaded_model = probs_task_manager.get_loaded_model()
            ai_engine.probs_model_keeper = loaded_model
            # Store checkpoint name for logging
            if hasattr(loaded_model, '__dict__'):
                loaded_model._checkpoint_name = checkpoint_file
            print(f"[API] PROBS checkpoint loaded for Level 7: {checkpoint_file}")
            return {"status": "success", "loaded": checkpoint_file}
        else:
            raise HTTPException(status_code=500, detail="Failed to load checkpoint")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/probs/checkpoints/{checkpoint_name}/download")
async def download_probs_checkpoint(checkpoint_name: str):
    """Download a PROBS checkpoint file."""
    try:
        checkpoint_file = checkpoint_name if checkpoint_name.endswith('.ckpt') else f"{checkpoint_name}.ckpt"
        
        # Используем тот же путь, что и probs_task_manager
        checkpoint_path = os.path.join(probs_task_manager.models_dir, "checkpoints", checkpoint_file)
        
        # Альтернативный путь (на случай разных структур)
        if not os.path.exists(checkpoint_path):
            alt_path = os.path.join(engine_dir, "models", "probs", "checkpoints", checkpoint_file)
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
        
        if not os.path.exists(checkpoint_path):
            # Попробуем найти файл в любом месте
            import glob
            search_paths = [
                os.path.join(engine_dir, "models", "probs", "checkpoints", "*.ckpt"),
                os.path.join(probs_task_manager.models_dir, "checkpoints", "*.ckpt"),
                os.path.join("/workspace", "togyzkumalak-engine", "togyzkumalak-engine", "models", "probs", "checkpoints", "*.ckpt"),
            ]
            found_files = []
            for pattern in search_paths:
                found_files.extend(glob.glob(pattern))
            
            similar = [f for f in found_files if checkpoint_file.lower() in os.path.basename(f).lower()]
            error_msg = f"Checkpoint not found: {checkpoint_file}"
            if similar:
                error_msg += f"\nSimilar files found: {[os.path.basename(f) for f in similar[:5]]}"
            raise HTTPException(status_code=404, detail=error_msg)
        
        return FileResponse(
            path=checkpoint_path,
            filename=checkpoint_file,
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/probs/model-info")
async def get_probs_model_info():
    """Get info about currently loaded PROBS model."""
    return {
        "loaded": probs_task_manager.is_model_loaded(),
        "checkpoints_available": len(probs_task_manager.get_checkpoints()),
        "model_info": ai_engine.get_model_info(7)
    }


@app.get("/api/training/probs/logs")
async def get_probs_logs(lines: int = 200):
    """Get training logs for PROBS."""
    try:
        log_path = os.path.join(engine_dir, "probs_training.log")
        if not os.path.exists(log_path):
            return {"output": [], "status": "no_log"}
        
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            return {"output": [line.strip() for line in all_lines[-lines:]], "status": "ok"}
    except Exception as e:
        return {"output": [], "status": "error", "error": str(e)}


@app.get("/api/system/git-status")
async def get_git_status():
    """Get current git status and last commit info."""
# ... (rest of the function)

# =============================================================================
# Checkpoint Sync API
# =============================================================================

class SyncConfigRequest(BaseModel):
    remote_url: str = "http://localhost"
    ports: List[int] = [8000, 8080]
    interval: int = 30
    enabled: bool = True

@app.get("/api/sync/status")
async def get_sync_status():
    """Get current synchronization status."""
    status_file = os.path.join(os.path.dirname(engine_dir), "sync_status.json")
    config_file = os.path.join(os.path.dirname(engine_dir), "sync_config.json")
    
    status = {"status": "inactive", "last_sync": None, "server": None}
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                status = json.load(f)
        except: pass
        
    config = {"enabled": False, "remote_url": "", "ports": []}
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except: pass
        
    # Check if process is running
    is_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['cmdline'] and 'sync_checkpoints.py' in ' '.join(proc.info['cmdline']):
            is_running = True
            break
            
    return {
        "is_running": is_running,
        "config": config,
        "status": status
    }

@app.post("/api/sync/config")
async def update_sync_config(request: SyncConfigRequest):
    """Update sync configuration."""
    config_file = os.path.join(os.path.dirname(engine_dir), "sync_config.json")
    with open(config_file, "w") as f:
        json.dump(request.dict(), f, indent=4)
        
    # Restart sync process if needed
    if request.enabled:
        # Kill existing
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline'] and 'sync_checkpoints.py' in ' '.join(proc.info['cmdline']):
                os.kill(proc.info['pid'], signal.SIGTERM)
        
        # Start new
        sync_script = os.path.join(os.path.dirname(engine_dir), "sync_checkpoints.py")
        subprocess.Popen([sys.executable, sync_script], start_new_session=True)
        
    return {"status": "success", "config": request.dict()}

@app.post("/api/sync/restart")
async def restart_sync():
    """Force restart the sync process."""
    # Kill existing
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['cmdline'] and 'sync_checkpoints.py' in ' '.join(proc.info['cmdline']):
            try:
                os.kill(proc.info['pid'], signal.SIGTERM)
            except: pass
    
    # Start new
    sync_script = os.path.join(os.path.dirname(engine_dir), "sync_checkpoints.py")
    subprocess.Popen([sys.executable, sync_script], start_new_session=True)
    return {"status": "restarted"}

    try:
        repo_root = os.path.dirname(os.path.dirname(engine_dir))
        git_dir = os.path.join(repo_root, ".git")
        
        if not os.path.exists(git_dir):
            repo_root = engine_dir
            git_dir = os.path.join(repo_root, ".git")
        
        if not os.path.exists(git_dir):
            return {
                "is_git_repo": False,
                "error": "Not a git repository"
            }
        
        os.chdir(repo_root)
        
        # Get current branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True
        )
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
        
        # Get last commit
        commit_result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%H|%an|%ae|%ad|%s", "--date=iso"],
            capture_output=True,
            text=True
        )
        
        last_commit = None
        if commit_result.returncode == 0 and commit_result.stdout:
            parts = commit_result.stdout.split("|")
            if len(parts) >= 5:
                last_commit = {
                    "hash": parts[0][:8],
                    "author": parts[1],
                    "email": parts[2],
                    "date": parts[3],
                    "message": parts[4]
                }
        
        # Check if behind remote
        subprocess.run(["git", "fetch", "origin"], capture_output=True)
        behind_result = subprocess.run(
            ["git", "rev-list", "--count", f"HEAD..origin/{branch}"],
            capture_output=True,
            text=True
        )
        commits_behind = int(behind_result.stdout.strip()) if behind_result.returncode == 0 else 0
        
        return {
            "is_git_repo": True,
            "branch": branch,
            "last_commit": last_commit,
            "commits_behind": commits_behind,
            "needs_update": commits_behind > 0
        }
        
    except Exception as e:
        return {
            "is_git_repo": False,
            "error": str(e)
        }


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

