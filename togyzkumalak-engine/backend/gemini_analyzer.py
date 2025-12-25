"""
Gemini Integration for Togyzkumalak Analysis.

Provides LLM-powered move analysis and position evaluation.
"""

import os
from typing import Dict, List, Optional

from .config import gemini_config
from .game_manager import TogyzkumalakBoard


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
        self.model = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Gemini client if API key is available."""
        api_key = gemini_config.api_key or os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            print("Gemini API key not configured. Analysis will be unavailable.")
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(gemini_config.model)
            print("Gemini client initialized successfully.")
        except ImportError:
            print("google-generativeai package not installed.")
        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
    
    def _format_position(self, board: TogyzkumalakBoard) -> str:
        """Format board position for LLM consumption."""
        lines = []
        lines.append("=== TOGYZKUMALAK POSITION ===")
        lines.append("")
        
        # Black side (top)
        black_pits = []
        for i in range(8, -1, -1):
            val = board.fields[9 + i]
            if val == board.TUZDUK:
                black_pits.append("X")
            else:
                black_pits.append(str(val))
        lines.append(f"Black (Қостаушы): [{'] ['.join(black_pits)}] Kazan: {board.black_kazan}")
        
        # Pit numbers
        lines.append(f"Pit numbers:       [9] [8] [7] [6] [5] [4] [3] [2] [1]")
        lines.append(f"Pit numbers:       [1] [2] [3] [4] [5] [6] [7] [8] [9]")
        
        # White side (bottom)
        white_pits = []
        for i in range(9):
            val = board.fields[i]
            if val == board.TUZDUK:
                white_pits.append("X")
            else:
                white_pits.append(str(val))
        lines.append(f"White (Бастаушы): [{'] ['.join(white_pits)}] Kazan: {board.white_kazan}")
        
        # Tuzduk info
        lines.append("")
        if board.fields[18] > 0:
            lines.append(f"White has tuzduk at Black's pit {board.fields[18]} ({self.HOLE_NAMES[board.fields[18]]})")
        if board.fields[19] > 0:
            lines.append(f"Black has tuzduk at White's pit {board.fields[19]} ({self.HOLE_NAMES[board.fields[19]]})")
        
        # Current player and legal moves
        lines.append("")
        lines.append(f"Current player: {board.current_player.upper()}")
        lines.append(f"Legal moves: {[m+1 for m in board.get_legal_moves()]}")
        
        return "\n".join(lines)
    
    def _format_move_history(self, moves: List[Dict]) -> str:
        """Format move history for context."""
        if not moves:
            return "No moves played yet."
        
        lines = ["Move history:"]
        for i in range(0, len(moves), 2):
            move_num = i // 2 + 1
            white_move = moves[i]["notation"] if i < len(moves) else ""
            black_move = moves[i+1]["notation"] if i+1 < len(moves) else ""
            lines.append(f"{move_num}. {white_move} {black_move}")
        
        return "\n".join(lines)
    
    async def analyze_position(
        self,
        board: TogyzkumalakBoard,
        move_history: List[Dict] = None
    ) -> Dict:
        """
        Analyze the current position.
        
        Returns:
            Dictionary with evaluation, best_move, and explanation
        """
        if not self.model:
            return {
                "available": False,
                "error": "Gemini not configured. Set GEMINI_API_KEY environment variable."
            }
        
        position_text = self._format_position(board)
        history_text = self._format_move_history(move_history or [])
        
        prompt = f"""You are an expert Togyzkumalak (Toguz Kumalak) player and analyst. 
Analyze the following position and provide strategic advice.

RULES REMINDER:
- Togyzkumalak is a two-player mancala game with 9 pits per side
- Each pit starts with 9 kumalaks (stones), totaling 162
- Players move counterclockwise, sowing stones one per pit
- Capture: If last stone lands in opponent's pit making it even, capture all
- Tuzduk: If last stone lands making exactly 3 in opponent's pit (not 9th), it becomes tuzduk (marked X)
- Tuzduk belongs to you permanently, opponent's stones falling there go to your kazan
- Each player can have only one tuzduk, and they cannot be symmetric
- Win: First to get more than 81 in kazan (or opponent has no moves)
- Draw: Both have exactly 81

{position_text}

{history_text}

Please provide:
1. EVALUATION: Rate the position from -10 (Black winning) to +10 (White winning) with explanation
2. BEST MOVE: Recommend the best move for the current player with detailed reasoning
3. KEY FACTORS: List 2-3 key strategic factors in this position
4. WARNING: Any tactical threats the current player should be aware of

Format your response clearly with these sections."""

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": gemini_config.max_tokens,
                    "temperature": gemini_config.temperature
                }
            )
            
            return {
                "available": True,
                "analysis": response.text,
                "position": position_text
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    async def suggest_move(
        self,
        board: TogyzkumalakBoard,
        move_history: List[Dict] = None
    ) -> Dict:
        """
        Get a move suggestion with explanation.
        
        Returns:
            Dictionary with suggested_move (1-9) and explanation
        """
        if not self.model:
            return {
                "available": False,
                "error": "Gemini not configured"
            }
        
        position_text = self._format_position(board)
        legal_moves = [m + 1 for m in board.get_legal_moves()]
        
        prompt = f"""You are an expert Togyzkumalak player. Given this position, recommend the best move.

{position_text}

Legal moves available: {legal_moves}

IMPORTANT: You must choose ONE move from the legal moves list.

For each legal move, briefly evaluate what happens:
- Where does the last stone land?
- Any captures possible?
- Any tuzduk opportunities?
- Does it leave you vulnerable?

Then state your recommended move as a single number (1-9) and explain why in 2-3 sentences.

Format:
RECOMMENDED MOVE: [number]
REASONING: [your explanation]"""

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 0.3  # Lower temperature for more focused response
                }
            )
            
            # Try to extract the move number from response
            text = response.text
            suggested_move = None
            
            for move in legal_moves:
                if f"RECOMMENDED MOVE: {move}" in text or f"RECOMMENDED MOVE:{move}" in text:
                    suggested_move = move
                    break
            
            if not suggested_move:
                # Fallback: look for any number in legal moves
                for move in legal_moves:
                    if str(move) in text.split("RECOMMENDED MOVE")[-1][:20]:
                        suggested_move = move
                        break
            
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
        board_before: TogyzkumalakBoard,
        board_after: TogyzkumalakBoard,
        move: int,
        notation: str,
        player: str
    ) -> Dict:
        """
        Provide commentary on a move that was just played.
        
        Returns:
            Dictionary with commentary
        """
        if not self.model:
            return {
                "available": False,
                "error": "Gemini not configured"
            }
        
        before_text = self._format_position(board_before)
        after_text = self._format_position(board_after)
        
        # Calculate changes
        if player == "white":
            kazan_gain = board_after.white_kazan - board_before.white_kazan
        else:
            kazan_gain = board_after.black_kazan - board_before.black_kazan
        
        tuzduk_created = 'x' in notation
        
        prompt = f"""{player.upper()} played move {move} (notation: {notation}).

BEFORE:
{before_text}

AFTER:
{after_text}

Changes:
- Kazan gained: {kazan_gain}
- Tuzduk created: {"Yes" if tuzduk_created else "No"}

Provide a brief (2-3 sentences) commentary on this move:
- Was it a good move?
- What did it accomplish?
- Any missed opportunities?"""

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": 200,
                    "temperature": 0.5
                }
            )
            
            return {
                "available": True,
                "commentary": response.text,
                "move": move,
                "notation": notation,
                "kazan_gain": kazan_gain,
                "tuzduk_created": tuzduk_created
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self.model is not None


# Global analyzer instance
gemini_analyzer = GeminiAnalyzer()

