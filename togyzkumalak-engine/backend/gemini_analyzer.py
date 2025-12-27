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

    def _build_generate_config(self, max_output_tokens: int, temperature: float):
        """
        Build a GenerateContentConfig for google-genai.
        Passing a plain dict may be ignored by some SDK versions, leading to short/truncated outputs.
        """
        try:
            from google.genai import types
            return types.GenerateContentConfig(
                max_output_tokens=int(max_output_tokens),
                temperature=float(temperature),
            )
        except Exception:
            # Fallback to dict for older SDKs
            return {
                "max_output_tokens": int(max_output_tokens),
                "temperature": float(temperature),
            }

    def _response_to_text(self, response) -> str:
        """
        Extract full text from google-genai response across SDK variants.
        Some versions expose `.text`, others require joining candidate parts.
        """
        if response is None:
            return ""

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        # Try candidates/parts structure
        try:
            candidates = getattr(response, "candidates", None) or []
            chunks: List[str] = []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text:
                        chunks.append(part_text)
            joined = "".join(chunks)
            return joined
        except Exception:
            return ""
    
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
        """Build the analysis prompt in Russian."""
        return f"""Вы — эксперт по игре Тогыз Кумалак (Тоғыз Құмалақ).
Проанализируйте следующую позицию и дайте стратегический совет.

ОТВЕЧАЙТЕ СТРОГО НА РУССКОМ ЯЗЫКЕ.

{position_text}

{history_text}

Пожалуйста, предоставьте анализ в следующем формате:

**ОЦЕНКА:** Оцените позицию от -10 (черные выигрывают) до +10 (белые выигрывают). Пример: "+2.5 (небольшое преимущество белых)"

**ЛУЧШИЙ ХОД:** Рекомендуйте лучший ход (номер лунки 1-9) для текущего игрока с обоснованием.

**КЛЮЧЕВЫЕ ФАКТОРЫ:** Перечислите 2-3 ключевых стратегических фактора в этой позиции.

**УГРОЗЫ:** Любые тактические угрозы, о которых должен знать текущий игрок.

Пишите кратко, но содержательно. Используйте казахские названия лунок, где это уместно."""

    def _build_suggest_prompt(self, position_text: str, legal_moves: List[int]) -> str:
        """Build the move suggestion prompt in Russian - optimized for quick response."""
        return f"""Вы — эксперт по Тогыз Кумалак. 

ВАЖНО: Начните ответ СРАЗУ с рекомендации в формате ниже, без вступления!

{position_text}

Доступные ходы: {legal_moves}

ОТВЕТЬТЕ СТРОГО В ЭТОМ ФОРМАТЕ:

**РЕКОМЕНДУЕМЫЙ ХОД:** [число 1-9]

**ОБОСНОВАНИЕ:** Краткое объяснение (2-3 предложения).

**АНАЛИЗ ХОДОВ:**
- Ход X: куда приземлится, захват?
- Ход Y: ...

Отвечайте на русском языке."""

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
            gen_cfg = self._build_generate_config(
                max_output_tokens=gemini_config.max_tokens,
                temperature=gemini_config.temperature,
            )

            # Use run_in_executor for sync SDK call in async context
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=gen_cfg
                )
            )

            text = self._response_to_text(response)
            
            # #region agent log
            import json, time
            with open(r'c:\Users\Admin\Documents\Toguzkumalak\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location":"gemini_analyzer.py:202", "message":"Gemini Analysis Raw Output", "data":{"raw_text":text, "len":len(text)}, "timestamp":int(time.time()*1000), "sessionId":"debug-session", "hypothesisId":"G"}) + "\n")
            # #endregion

            return {
                "available": True,
                "analysis": text,
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
            gen_cfg = self._build_generate_config(
                max_output_tokens=1200,
                temperature=0.3,
            )
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=gen_cfg
                )
            )
            
            text = self._response_to_text(response)
            
            # #region agent log
            import json, time
            with open(r'c:\Users\Admin\Documents\Toguzkumalak\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location":"gemini_analyzer.py:250", "message":"Gemini Suggestion Raw Output", "data":{"raw_text":text, "len":len(text), "legal_moves":legal_moves}, "timestamp":int(time.time()*1000), "sessionId":"debug-session", "hypothesisId":"E"}) + "\n")
            # #endregion

            suggested_move = None
            
            # Try to extract move number (English and Russian keys)
            search_keys = ["RECOMMENDED MOVE:", "РЕКОМЕНДУЕМЫЙ ХОД:"]
            for key in search_keys:
                if f"{key}**" in text:
                    parts = text.split(f"{key}**")
                    if len(parts) > 1:
                        import re
                        match = re.search(r'(\d)', parts[1])
                        if match:
                            num = int(match.group(1))
                            if num in legal_moves:
                                suggested_move = num
                                break
                if key in text:
                    parts = text.split(key)
                    if len(parts) > 1:
                        import re
                        match = re.search(r'(\d)', parts[1])
                        if match:
                            num = int(match.group(1))
                            if num in legal_moves:
                                suggested_move = num
                                break
            
            # Fallback extraction if suggested_move still None
            if not suggested_move:
                import re
                match = re.search(r'(?:RECOMMENDED MOVE|РЕКОМЕНДУЕМЫЙ ХОД)[:\*\s]+(\d)', text)
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
            gen_cfg = self._build_generate_config(
                max_output_tokens=500,
                temperature=0.5,
            )
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=gen_cfg
                )
            )
            
            text = self._response_to_text(response)
            
            return {
                "available": True,
                "commentary": text,
                "move": move,
                "kazan_gain": kazan_gain
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    async def get_move_probabilities(
        self,
        board_state: Dict
    ) -> Dict[int, float]:
        """
        Get move probabilities (confidence levels) for all legal moves using Gemini.
        Returns a dictionary mapping move (0-8) to probability (0.0-1.0).
        """
        if not self.client:
            return {i: 0.0 for i in range(9)}

        position_text = self._format_position(board_state)
        legal_moves = [m + 1 for m in board_state.get("legal_moves", list(range(9)))]
        
        prompt = f"""{position_text}

Вы — гроссмейстер Тогыз Кумалак. Оцените вероятность того, что каждый из доступных ходов является наилучшим в данной позиции.
ДОСТУПНЫЕ ХОДЫ: {legal_moves}

ОТВЕТЬТЕ СТРОГО В ФОРМАТЕ JSON, где ключи — номера лунок (1-9), а значения — вероятность (от 0 до 1). 
Сумма всех вероятностей должна быть равна 1.0.

Пример ответа:
{{
  "1": 0.1,
  "2": 0.8,
  "5": 0.1
}}"""

        try:
            from google.genai import types
            gen_cfg = types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.1,
                response_mime_type="application/json"
            )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=gen_cfg
                )
            )
            
            text = self._response_to_text(response)
            import json
            probs_data = json.loads(text)
            
            # Map back to 0-8 indexing and ensure all 9 pits are covered
            result = {i: 0.0 for i in range(9)}
            for move_str, prob in probs_data.items():
                try:
                    move_idx = int(move_str) - 1
                    if 0 <= move_idx < 9:
                        result[move_idx] = float(prob)
                except (ValueError, TypeError):
                    continue
            
            # Re-normalize if necessary
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
            else:
                # Fallback to uniform if something went wrong
                if legal_moves:
                    val = 1.0 / len(legal_moves)
                    for m in legal_moves:
                        result[m-1] = val
                        
            return result
            
        except Exception as e:
            print(f"[ERROR] Gemini probabilities failed: {e}")
            # Fallback to uniform
            result = {i: 0.0 for i in range(9)}
            if legal_moves:
                val = 1.0 / len(legal_moves)
                for m in legal_moves:
                    result[m-1] = val
            return result

    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self.client is not None


# Global analyzer instance
gemini_analyzer = GeminiAnalyzer()
