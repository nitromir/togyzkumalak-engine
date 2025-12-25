"""
Togyzkumalak Engine - FastAPI Backend

Main server with REST API and WebSocket support for real-time gameplay.
"""

import asyncio
import json
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import server_config, elo_config
from .game_manager import game_manager, TogyzkumalakBoard, GameStatus
from .ai_engine import ai_engine
from .elo_system import elo_system
from .gemini_analyzer import gemini_analyzer


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
    
    return state


@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    """Get game state."""
    state = game_manager.get_game_state(game_id)
    if "error" in state:
        raise HTTPException(status_code=404, detail=state["error"])
    return state


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
                "move": ai_move,
                "thinking_time_ms": thinking_time
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
    
    return state


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

