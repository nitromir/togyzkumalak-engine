# Togyzkumalak AI Engine - Project Plan

## Executive Summary

This document outlines the development plan for a comprehensive Togyzkumalak AI Engine with:
- Human vs AI gameplay interface
- ELO rating system for AI evaluation
- RL training quality control
- LLM integration for move analysis (Gemini SDK)
- Classic board visualization

---

## 1. Analysis of Existing Resources

### 1.1 Available Components

| Source | Language | Purpose | Usable Assets |
|--------|----------|---------|---------------|
| `gym-togyzkumalak` | Python | RL Environment | Game logic, observation space, reward system |
| `9qumalaq` | JavaScript | Canvas Game | Board images, cell sprites, tuzduk graphics |
| `togyz_py` | Python | Game Logic | Clean notation system (e.g., "15", "23x"), move validation |

### 1.2 Key Technical Findings

**gym-togyzkumalak/board.py:**
- 128-dimensional observation vector (TD-Gammon inspired)
- Reward: +1 (White win), -1 (Black win), 0 (Draw)
- Turn-based with automatic player switching

**togyz_py/togboard.py:**
- Clean notation: move "15" = from pit 1, lands on 5
- Tuzduk notation: "23x" = tuzduk captured
- Fields array: [0-8] = White, [9-17] = Black, [18-19] = tuzduk positions, [20-21] = kazans, [22] = current player

**9qumalaq:**
- Canvas-based rendering at 60fps
- Board image: 80x200px per cell
- Tuzduk image overlay
- Click listener per cell

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Browser)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Classic View │  │ Modern View  │  │ Gemini Analysis Panel  │ │
│  │ (Canvas 2D)  │  │ (CSS/HTML)   │  │ (Move comments, hints) │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│                           │                                      │
│                    ┌──────┴──────┐                               │
│                    │ WebSocket   │                               │
│                    └──────┬──────┘                               │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                         BACKEND (Python FastAPI)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Game Manager │  │ AI Engine    │  │ Gemini Integration     │ │
│  │ (State, Log) │  │ (PyTorch)    │  │ (google-genai SDK)     │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│         │                  │                    │                │
│  ┌──────┴──────────────────┴────────────────────┴──────────────┐│
│  │                    Gym Environment                           ││
│  │               (togyzkumalak_env.py + board.py)              ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                    ELO Rating System                         ││
│  │         (Track AI performance across games)                  ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature Breakdown

### 3.1 Core Features (Phase 1)

1. **Human vs AI Mode**
   - Player selects color (White/Black)
   - AI responds with move within configurable time
   - Move validation and visualization

2. **Classic Board View**
   - Canvas-based rendering using 9qumalaq assets
   - Animated kumalak movement
   - Tuzduk highlighting
   - Kazan score display

3. **Game Logging**
   - Full game state per move
   - Standard notation (togyz_py format)
   - JSON export for analysis

### 3.2 AI Evaluation (Phase 2)

4. **ELO Rating System**
   - Initial ELO: 1500
   - K-factor: 32 (adjustable)
   - Track AI ELO across games
   - Human ELO estimation based on results

5. **AI Strength Levels**
   - Level 1: Random moves (ELO ~800)
   - Level 2: Simple heuristics (ELO ~1200)
   - Level 3: Neural Network (current MLP)
   - Level 4: Advanced RL model (future)

### 3.3 Training Control (Phase 3)

6. **RL Training Monitor**
   - Live game during training
   - Win/loss statistics
   - Policy visualization
   - Manual intervention option

7. **Supervised Learning from Logs**
   - Import game logs (PGN-like format)
   - Convert to training data
   - Fine-tune model on expert games

### 3.4 LLM Integration (Phase 4)

8. **Gemini Analysis Panel**
   - Send current position to Gemini
   - Receive move recommendations with explanations
   - Position evaluation commentary
   - Strategic insights

---

## 4. User Goals Analysis

| User Goal | Solution | Implementation |
|-----------|----------|----------------|
| Evaluate AI ELO during play | Real-time ELO calculation | ELO tracker with K-factor adjustment |
| Control AI quality after RL | Live game monitoring | WebSocket connection to training loop |
| Train on quality game logs | Log import system | Parser + supervised learning pipeline |
| LLM move analysis | Gemini integration | Prompt engineering for Togyzkumalak |

### 4.1 Additional Goals (Inferred)

9. **Opening Book**
   - Database of known openings
   - Named strategies (like Chess openings)

10. **Position Database**
    - Store interesting positions
    - Endgame tablebase (if computationally feasible)

11. **Tournament Mode**
    - AI vs AI matches
    - Round-robin tournaments
    - ELO leaderboard

12. **Mobile Support**
    - Responsive design
    - Touch controls

13. **Replay Analysis**
    - Step through recorded games
    - Alternative move suggestions
    - Blunder detection

---

## 5. Directory Structure

```
togyzkumalak-engine/
├── docs/
│   ├── PROJECT_PLAN.md          # This file
│   ├── API_REFERENCE.md         # API documentation
│   └── NOTATION_GUIDE.md        # Game notation reference
│
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── game_manager.py          # Game state management
│   ├── ai_engine.py             # Neural network inference
│   ├── elo_system.py            # ELO rating calculations
│   ├── gemini_analyzer.py       # Gemini SDK integration
│   ├── websocket_handler.py     # WebSocket connections
│   └── config.py                # Configuration settings
│
├── frontend/
│   ├── index.html               # Main game page
│   ├── css/
│   │   └── styles.css           # Modern styling
│   ├── js/
│   │   ├── app.js               # Main application
│   │   ├── board-classic.js     # Canvas board renderer
│   │   ├── board-modern.js      # CSS/HTML board renderer
│   │   ├── websocket.js         # Server communication
│   │   ├── elo-display.js       # ELO rating UI
│   │   └── gemini-panel.js      # LLM analysis panel
│   └── assets/
│       └── images/              # Copied from 9qumalaq
│
├── models/
│   ├── policy_network.py        # PyTorch model definition
│   ├── checkpoints/             # Saved model weights
│   └── training/
│       ├── trainer.py           # RL training loop
│       └── supervised.py        # Supervised learning from logs
│
├── logs/
│   ├── games/                   # Game logs (JSON)
│   └── training/                # Training logs
│
├── requirements.txt
└── run.py                       # Entry point
```

---

## 6. UX/UI Design

### 6.1 Layout (Desktop)

```
┌────────────────────────────────────────────────────────────────────┐
│  🎮 TOGYZKUMALAK ENGINE                    [Settings] [About]      │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────┐  ┌────────────────────────────┐  │
│  │                              │  │ GAME INFO                  │  │
│  │                              │  │ ─────────────────────────  │  │
│  │                              │  │ Player: WHITE (You)        │  │
│  │      GAME BOARD              │  │ AI Level: Advanced (1650)  │  │
│  │      (Canvas/HTML)           │  │ Your ELO: 1850             │  │
│  │                              │  │ Time: 05:23                │  │
│  │                              │  │                            │  │
│  │                              │  ├────────────────────────────┤  │
│  │                              │  │ MOVE HISTORY               │  │
│  │                              │  │ ─────────────────────────  │  │
│  └──────────────────────────────┘  │ 1. 15  27                  │  │
│                                    │ 2. 34  81                  │  │
│  ┌──────────────────────────────┐  │ 3. 62  ...                 │  │
│  │ 🤖 GEMINI ANALYSIS           │  │                            │  │
│  │ ─────────────────────────────│  ├────────────────────────────┤  │
│  │ Current position evaluation: │  │ CONTROLS                   │  │
│  │ +0.3 (slight White advantage)│  │ ─────────────────────────  │  │
│  │                              │  │ [New Game] [Undo] [Resign] │  │
│  │ Best move: Pit 5 → "56"      │  │                            │  │
│  │ Reasoning: Controls center...│  │ [📊 View Classic Board]    │  │
│  └──────────────────────────────┘  └────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 6.2 Classic Board View

Using Canvas 2D with 9qumalaq-style graphics:
- Wooden board texture
- Circular pits with shadow
- Golden tuzduk overlay
- Animated kumalak stones
- Kazan bowls on sides

### 6.3 Modern Board View

Using CSS Grid with styled elements:
- Gradient backgrounds
- Hover effects on playable pits
- Last move highlighting
- Responsive sizing

### 6.4 Color Scheme

```css
:root {
  --bg-primary: #1a1a2e;
  --bg-secondary: #16213e;
  --accent: #e94560;
  --gold: #d4af37;
  --text-primary: #e8e8e8;
  --board-wood: #5d4e37;
}
```

---

## 7. API Design

### 7.1 WebSocket Events

**Client → Server:**
```json
{ "type": "new_game", "player_color": "white", "ai_level": 3 }
{ "type": "make_move", "action": 4 }
{ "type": "request_analysis" }
{ "type": "undo" }
{ "type": "resign" }
```

**Server → Client:**
```json
{ "type": "game_state", "state": {...}, "legal_moves": [0,1,3,5,7,8] }
{ "type": "ai_move", "action": 6, "notation": "78", "thinking_time": 0.5 }
{ "type": "game_over", "winner": "white", "elo_change": +15 }
{ "type": "analysis", "evaluation": 0.3, "best_move": 5, "explanation": "..." }
```

### 7.2 REST Endpoints

```
GET  /api/models           - List available AI models
POST /api/games            - Create new game
GET  /api/games/{id}       - Get game state
POST /api/games/{id}/move  - Make a move
GET  /api/elo              - Get ELO statistics
POST /api/import-log       - Import game log for training
```

---

## 8. Implementation Phases

### Phase 1: Core Game (Week 1) ✅ COMPLETED
- [x] FastAPI backend setup
- [x] WebSocket game communication
- [x] Modern board UI (HTML/CSS)
- [x] Human vs AI gameplay
- [x] Move validation and execution

### Phase 2: Classic View + ELO (Week 2) ✅ COMPLETED
- [x] Canvas board renderer
- [x] Port 9qumalaq graphics (adapted style)
- [x] ELO rating system
- [x] Game logging

### Phase 3: Advanced AI (Week 3) 🔄 PARTIAL
- [x] Multiple AI difficulty levels (5 levels: Random to Expert)
- [ ] Model checkpoint management
- [ ] Training monitor UI

### Phase 4: Gemini Integration (Week 4) ✅ COMPLETED
- [x] Gemini SDK setup (google-genai + gemini-3-flash-preview)
- [x] Position encoding for LLM
- [x] Move explanation prompts
- [x] Analysis panel UI

---

## 9. Technical Dependencies

```
# Backend
fastapi>=0.104.0
uvicorn>=0.24.0
websockets>=12.0
torch>=2.0.0
numpy<2.0
gym>=0.26.0
google-genai>=1.0.0  # New SDK for Gemini API

# Frontend
# Vanilla JS (no framework)
# Canvas 2D API
# WebSocket API
```

---

## 10. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Gemini API rate limits | Implement caching, throttling |
| AI too slow for real-time | Pre-compute moves, time limits |
| Browser compatibility | Test on Chrome, Firefox, Edge |
| WebSocket disconnects | Auto-reconnect with state recovery |

---

## 11. Success Metrics

1. **Playability**: Complete game vs AI without crashes
2. **ELO Accuracy**: ELO correlates with AI strength
3. **Analysis Quality**: Gemini provides useful insights
4. **Training Control**: Can observe and intervene in RL
5. **User Satisfaction**: Champion player finds it useful

---

*Document Version: 1.0*
*Created: December 25, 2025*

