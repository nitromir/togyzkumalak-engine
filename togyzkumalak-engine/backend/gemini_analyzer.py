"""
Gemini Integration for Togyzkumalak Analysis.

Uses the new google-genai SDK with gemini-3-flash-preview model.
Provides LLM-powered move analysis and position evaluation.
"""

import os
import asyncio
from typing import Dict, List, Optional

from .config import gemini_config


class GeminiAnalyzer:
    """
    Analyzes Togyzkumalak positions using Google's Gemini API.
    
    Features:
    - Position evaluation with explanation
    - Best move suggestion with reasoning
    - Game commentary
    - Strategic insights
    """
    
    # Kazakh hole names for better explanation
    HOLE_NAMES = {
        1: "Арт",
        2: "Тектұрмас", 
        3: "Ат өтпес",
        4: "Атсыратар",
        5: "Бел",
        6: "Белбасар",
        7: "Қандықақпан",
        8: "Көкмойын",
        9: "Маңдай"
    }
    
    def __init__(self):
        self.client = None
        self.model = gemini_config.model
        self._init_client()
    
    def _init_client(self):
        """Initialize Gemini client if API key is available."""
        api_key = gemini_config.api_key or os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            print("[WARNING] Gemini API key not configured. Analysis will be unavailable.")
            return
        
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            print(f"[OK] Gemini client initialized successfully (model: {self.model})")
        except ImportError:
            print("[WARNING] google-genai package not installed. Run: pip install google-genai")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Gemini client: {e}")
    
    def _format_position(self, board_state: Dict) -> str:
        """Format board position for LLM consumption."""
        lines = []
        lines.append("=== TOGYZKUMALAK POSITION ===")
        lines.append("")
        
        # Black side (top) - reverse order for display
        black_pits = board_state.get("black_pits", [9]*9)
        lines.append(f"ҚАРА (Black/Қостаушы):")
        lines.append(f"  Pits: [9←1] {list(reversed(black_pits))}")
        lines.append(f"  Қазан: {board_state.get('black_kazan', 0)}")
        
        # White side (bottom)
        white_pits = board_state.get("white_pits", [9]*9)
        lines.append("")
        lines.append(f"АҚ (White/Бастаушы):")
        lines.append(f"  Pits: [1→9] {white_pits}")
        lines.append(f"  Қазан: {board_state.get('white_kazan', 0)}")
        
        # Tuzduk info
        lines.append("")
        white_tuzduk = board_state.get("white_tuzduk", 0)
        black_tuzduk = board_state.get("black_tuzduk", 0)
        if white_tuzduk > 0:
            lines.append(f"* White has TUZDUK at Black's pit {white_tuzduk} ({self.HOLE_NAMES.get(white_tuzduk, '')})")
        if black_tuzduk > 0:
            lines.append(f"* Black has TUZDUK at White's pit {black_tuzduk} ({self.HOLE_NAMES.get(black_tuzduk, '')})")
        
        # Current player and legal moves
        lines.append("")
        current = board_state.get("current_player", "white")
        lines.append(f"Current player: {current.upper()}")
        
        legal = board_state.get("legal_moves", list(range(9)))
        lines.append(f"Legal moves: {[m+1 for m in legal]}")
        
        return "\n".join(lines)
    
    def _format_move_history(self, moves: List[Dict]) -> str:
        """Format move history for context."""
        if not moves:
            return "No moves played yet."
        
        lines = ["Move history:"]
        for i, move in enumerate(moves):
            move_num = i + 1
            player = move.get("player", "?")
            notation = move.get("notation", str(move.get("move", "?")))
            lines.append(f"  {move_num}. {player[0].upper()}: {notation}")
        
        return "\n".join(lines[-20:])  # Last 20 moves
    
    def _build_analysis_prompt(self, position_text: str, history_text: str) -> str:
        """Build the analysis prompt."""
        return f"""You are an expert Togyzkumalak (Toguz Kumalak / Тоғыз Құмалақ) player and analyst.
Analyze the following position and provide strategic advice.

RULES REMINDER:
- Togyzkumalak is a two-player mancala game with 9 pits per side
- Each pit starts with 9 kumalaks (stones), totaling 162
- Players move counterclockwise, sowing stones one per pit
- Capture: If last stone lands in opponent's pit making it even, capture all
- Tuzduk: If last stone lands making exactly 3 in opponent's pit (not 9th pit), it becomes tuzduk (marked X)
- Tuzduk belongs to you permanently, opponent's stones falling there go to your kazan
- Each player can have only one tuzduk, and they cannot be symmetric
- Win: First to get more than 81 in kazan (or opponent has no moves)
- Draw: Both have exactly 81

{position_text}

{history_text}

Please provide analysis in the following format:

**EVALUATION:** Rate the position from -10 (Black winning) to +10 (White winning). Example: "+2.5 (slight White advantage)"

**BEST MOVE:** Recommend the best move (pit number 1-9) for the current player with reasoning.

**KEY FACTORS:** List 2-3 key strategic factors in this position.

**THREATS:** Any tactical threats the current player should be aware of.

Keep your response concise but insightful. Use Kazakh pit names where helpful."""
    
    def _build_suggest_prompt(self, position_text: str, legal_moves: List[int]) -> str:
        """Build the move suggestion prompt."""
        return f"""You are an expert Togyzkumalak player. Recommend the best move for this position.

{position_text}

Legal moves: {legal_moves}

Analyze each legal move briefly:
- Where does the last stone land?
- Any captures possible?
- Any tuzduk opportunities?
- Does it leave you vulnerable?

Then provide your recommendation:

**RECOMMENDED MOVE:** [single number 1-9]
**REASONING:** [2-3 sentences explaining why]"""

    async def analyze_position(
        self,
        board_state: Dict,
        move_history: List[Dict] = None
    ) -> Dict:
        """
        Analyze the current position.
        
        Args:
            board_state: Dictionary with white_pits, black_pits, kazans, tuzduk, etc.
            move_history: List of previous moves
        
        Returns:
            Dictionary with evaluation, analysis text, etc.
        """
        if not self.client:
            return {
                "available": False,
                "error": "Gemini not configured. Set GEMINI_API_KEY environment variable."
            }
        
        position_text = self._format_position(board_state)
        history_text = self._format_move_history(move_history or [])
        prompt = self._build_analysis_prompt(position_text, history_text)
        
        try:
            # Use run_in_executor for sync SDK call in async context
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "max_output_tokens": gemini_config.max_tokens,
                        "temperature": gemini_config.temperature,
                    }
                )
            )
            
            return {
                "available": True,
                "analysis": response.text,
                "position": position_text,
                "model": self.model
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    async def suggest_move(
        self,
        board_state: Dict,
        move_history: List[Dict] = None
    ) -> Dict:
        """
        Get a move suggestion with explanation.
        
        Returns:
            Dictionary with suggested_move (1-9) and explanation
        """
        if not self.client:
            return {
                "available": False,
                "error": "Gemini not configured"
            }
        
        position_text = self._format_position(board_state)
        legal_moves = [m + 1 for m in board_state.get("legal_moves", list(range(9)))]
        prompt = self._build_suggest_prompt(position_text, legal_moves)
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "max_output_tokens": 500,
                        "temperature": 0.3  # Lower for focused response
                    }
                )
            )
            
            text = response.text
            suggested_move = None
            
            # Try to extract move number
            for move in legal_moves:
                if f"RECOMMENDED MOVE:** {move}" in text or f"RECOMMENDED MOVE: {move}" in text:
                    suggested_move = move
                    break
            
            # Fallback extraction
            if not suggested_move:
                import re
                match = re.search(r'RECOMMENDED MOVE[:\*\s]+(\d)', text)
                if match:
                    num = int(match.group(1))
                    if num in legal_moves:
                        suggested_move = num
            
            return {
                "available": True,
                "suggested_move": suggested_move,
                "explanation": text,
                "legal_moves": legal_moves
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    async def comment_move(
        self,
        board_before: Dict,
        board_after: Dict,
        move: int,
        player: str
    ) -> Dict:
        """
        Provide commentary on a move that was just played.
        """
        if not self.client:
            return {
                "available": False,
                "error": "Gemini not configured"
            }
        
        # Calculate changes
        if player == "white":
            kazan_gain = board_after.get("white_kazan", 0) - board_before.get("white_kazan", 0)
        else:
            kazan_gain = board_after.get("black_kazan", 0) - board_before.get("black_kazan", 0)
        
        before_text = self._format_position(board_before)
        after_text = self._format_position(board_after)
        
        prompt = f"""{player.upper()} played move {move} (pit {self.HOLE_NAMES.get(move, move)}).

BEFORE:
{before_text}

AFTER:
{after_text}

Kazan gained: {kazan_gain}

Provide brief commentary (2-3 sentences):
- Was it a good move?
- What did it accomplish?
- Any missed opportunities?"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "max_output_tokens": 200,
                        "temperature": 0.5
                    }
                )
            )
            
            return {
                "available": True,
                "commentary": response.text,
                "move": move,
                "kazan_gain": kazan_gain
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self.client is not None


# Global analyzer instance
gemini_analyzer = GeminiAnalyzer()
