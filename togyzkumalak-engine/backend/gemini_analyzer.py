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
    
    # Model display names in Russian
    MODEL_NAMES = {
        "polynet": "ПолиНет (базовая сеть)",
        "alphazero": "АльфаЗеро (MCTS)",
        "probs": "ПРОБС (Beam Search)"
    }
    
    # System prompt for consistent persona
    SYSTEM_PROMPT = """Ты — гроссмейстер и тренер по Тогыз Кумалаку (Тоғыз Құмалақ) с 20-летним опытом.
Твоя задача — давать глубокий стратегический анализ позиций и объяснять тактические нюансы игры.

ПРАВИЛА:
- Отвечай ТОЛЬКО на русском языке
- Используй казахские названия лунок (Арт, Тектұрмас, Ат өтпес, Атсыратар, Бел, Белбасар, Қандықақпан, Көкмойын, Маңдай)
- Будь конкретен: указывай номера лунок и точные расчёты
- Объясняй логику, а не просто констатируй факты
- Учитывай данные нейросетей, но критически их оценивай"""

    
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
        """Format board position for LLM consumption - fully in Russian."""
        lines = []
        lines.append("═══ ТЕКУЩАЯ ПОЗИЦИЯ ═══")
        lines.append("")
        
        # Black side (top) - reverse order for display
        black_pits = board_state.get("black_pits", [9]*9)
        black_kazan = board_state.get('black_kazan', 0)
        lines.append(f"ЧЁРНЫЕ (Қара/Қостаушы):")
        lines.append(f"  Лунки [9←1]: {list(reversed(black_pits))}")
        lines.append(f"  Казан: {black_kazan} кумалаков")
        
        # White side (bottom)
        white_pits = board_state.get("white_pits", [9]*9)
        white_kazan = board_state.get('white_kazan', 0)
        lines.append("")
        lines.append(f"БЕЛЫЕ (Ақ/Бастаушы):")
        lines.append(f"  Лунки [1→9]: {white_pits}")
        lines.append(f"  Казан: {white_kazan} кумалаков")
        
        # Material balance
        lines.append("")
        diff = white_kazan - black_kazan
        if diff > 0:
            lines.append(f"📊 Материал: белые +{diff}")
        elif diff < 0:
            lines.append(f"📊 Материал: чёрные +{abs(diff)}")
        else:
            lines.append("📊 Материал: равенство")
        
        # Victory progress
        lines.append(f"   До победы: белым нужно {82 - white_kazan}, чёрным нужно {82 - black_kazan}")
        
        # Tuzduk info
        white_tuzduk = board_state.get("white_tuzduk", 0)
        black_tuzduk = board_state.get("black_tuzduk", 0)
        if white_tuzduk > 0 or black_tuzduk > 0:
            lines.append("")
            lines.append("🏴 ТУЗДЫКИ:")
        if white_tuzduk > 0:
            lines.append(f"  • У белых туздык на лунке {white_tuzduk} чёрных ({self.HOLE_NAMES.get(white_tuzduk, '')})")
        if black_tuzduk > 0:
            lines.append(f"  • У чёрных туздык на лунке {black_tuzduk} белых ({self.HOLE_NAMES.get(black_tuzduk, '')})")
        
        # Current player and legal moves
        lines.append("")
        current = board_state.get("current_player", "white")
        current_ru = "БЕЛЫЕ" if current == "white" else "ЧЁРНЫЕ"
        lines.append(f"🎯 Ход: {current_ru}")
        
        legal = board_state.get("legal_moves", list(range(9)))
        legal_with_names = [f"{m+1} ({self.HOLE_NAMES.get(m+1, '')})" for m in legal]
        lines.append(f"   Доступные ходы: {', '.join(legal_with_names)}")
        
        return "\n".join(lines)
    
    def _format_move_history(self, moves: List[Dict]) -> str:
        """Format move history for context - in Russian."""
        if not moves:
            return "История ходов: партия только началась."
        
        lines = ["📜 ИСТОРИЯ ХОДОВ (последние 20):"]
        # Get last 20 moves
        recent_moves = moves[-20:] if len(moves) > 20 else moves
        start_idx = len(moves) - len(recent_moves)
        
        for i, move in enumerate(recent_moves):
            move_num = start_idx + i + 1
            player = move.get("player", "?")
            player_ru = "Б" if player.lower().startswith("w") else "Ч"
            notation = move.get("notation", str(move.get("move", "?")))
            lines.append(f"  {move_num}. {player_ru}: лунка {notation}")
        
        return "\n".join(lines)
    
    def _format_ai_data(self, model_probs: Dict[str, Dict[int, float]]) -> str:
        """Format AI model probabilities for LLM consumption - in Russian."""
        if not model_probs:
            return ""
        
        lines = ["", "🤖 ОЦЕНКИ НЕЙРОСЕТЕЙ:"]
        for model_name, probs in model_probs.items():
            if not probs:
                continue
            
            # Get display name in Russian
            display_name = self.MODEL_NAMES.get(model_name, model_name)
            
            # Get top 3 moves for each model
            sorted_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            if not sorted_moves or sorted_moves[0][1] < 0.01:
                lines.append(f"  • {display_name}: нет данных")
                continue
                
            moves_parts = []
            for m, p in sorted_moves:
                if p > 0.01:  # Only show moves with >1% probability
                    hole_name = self.HOLE_NAMES.get(m + 1, "")
                    moves_parts.append(f"лунка {m+1} ({hole_name}) — {p*100:.0f}%")
            
            if moves_parts:
                lines.append(f"  • {display_name}:")
                lines.append(f"    Топ ходы: {', '.join(moves_parts)}")
        
        # Add consensus note if models agree
        if len(model_probs) >= 2:
            top_moves = []
            for probs in model_probs.values():
                if probs:
                    best = max(probs.items(), key=lambda x: x[1])
                    if best[1] > 0.2:  # Only count if confident
                        top_moves.append(best[0])
            
            if len(top_moves) >= 2 and len(set(top_moves)) == 1:
                agreed_move = top_moves[0] + 1
                lines.append(f"  ⚡ Консенсус: все сети выбирают лунку {agreed_move} ({self.HOLE_NAMES.get(agreed_move, '')})")
        
        return "\n".join(lines)

    def _build_analysis_prompt(self, position_text: str, history_text: str, ai_data_text: str = "") -> str:
        """Build the analysis prompt - optimized for CONCISE Russian output (max 3 paragraphs)."""
        
        # Build the user message with all context
        user_message = f"""{position_text}
{ai_data_text}

═══ ЗАДАНИЕ ═══
Дай КРАТКИЙ анализ позиции (МАКСИМУМ 3 коротких абзаца!).

ФОРМАТ ОТВЕТА:

**Оценка:** [число от -5 до +5] — [одно предложение почему]

**Позиция:** [2-3 предложения о том, кто владеет инициативой, главные угрозы и ключевые лунки]

**Совет:** [1-2 предложения о тактике для ходящего игрока, НО без конкретного хода]

⛔ СТРОГО: Не давай конкретный ход! Не используй списки и заголовки ##. Пиши живым языком, как комментатор матча."""

        return f"{self.SYSTEM_PROMPT}\n\n{user_message}"

    def _build_suggest_prompt(self, position_text: str, legal_moves: List[int], ai_data_text: str = "") -> str:
        """Build the move suggestion prompt - CONCISE (max 3 paragraphs)."""
        
        user_message = f"""{position_text}
{ai_data_text}

═══ ЗАДАНИЕ ═══
Порекомендуй лучший ход. МАКСИМУМ 3 коротких абзаца!

ФОРМАТ:

**Лучший ход: лунка [N]** — [одно предложение почему это сильнейший ход]

**Идея:** [2-3 предложения: куда приземлится последний кумалак, будет ли захват, что это даёт]

**Альтернатива:** Лунка [X] тоже неплоха — [одно предложение]

⛔ СТРОГО: Не используй списки и заголовки ##. Пиши живым языком, как тренер даёт совет ученику."""

        return f"{self.SYSTEM_PROMPT}\n\n{user_message}"

    async def analyze_position_stream(
        self,
        board_state: Dict,
        move_history: List[Dict] = None,
        model_probs: Dict[str, Dict[int, float]] = None
    ):
        """
        Analyze the current position and yield text chunks (streaming).
        Uses a queue to properly async iterate over synchronous Gemini stream.
        """
        print(f"[Gemini Analysis] Starting position analysis")
        if not self.client:
            print(f"[Gemini Analysis] Client not available")
            yield "Gemini not configured."
            return
        
        position_text = self._format_position(board_state)
        history_text = self._format_move_history(move_history or [])
        ai_data_text = self._format_ai_data(model_probs or {})
        prompt = self._build_analysis_prompt(position_text, history_text, ai_data_text)
        
        try:
            gen_cfg = self._build_generate_config(
                max_output_tokens=4000,  # Increased for comprehensive analysis: ~20 paragraphs
                temperature=0.6,
            )

            # Use async queue to bridge sync stream to async generator
            import queue
            import threading
            
            chunk_queue: queue.Queue = queue.Queue()
            error_container = {"error": None}
            
            def stream_worker():
                """Worker thread to consume sync stream and put chunks in queue."""
                try:
                    response_stream = self.client.models.generate_content_stream(
                        model=self.model,
                        contents=prompt,
                        config=gen_cfg
                    )
                    for chunk in response_stream:
                        text = self._response_to_text(chunk)
                        if text:
                            chunk_queue.put(text)
                except Exception as e:
                    error_container["error"] = str(e)
                finally:
                    chunk_queue.put(None)  # Signal end of stream
            
            # Start worker thread
            worker = threading.Thread(target=stream_worker, daemon=True)
            worker.start()
            
            # Async consume from queue
            while True:
                # Non-blocking check with short sleep to yield control
                try:
                    chunk = chunk_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)  # Yield control to event loop
                    continue
                
                if chunk is None:  # End of stream
                    if error_container["error"]:
                        yield f"Error: {error_container['error']}"
                    break
                
                yield chunk
                    
        except Exception as e:
            yield f"Error during analysis: {str(e)}"

    async def suggest_move_stream(
        self,
        board_state: Dict,
        move_history: List[Dict] = None,
        model_probs: Dict[str, Dict[int, float]] = None
    ):
        """
        Get a move suggestion with explanation (streaming).
        Uses a queue to properly async iterate over synchronous Gemini stream.
        """
        if not self.client:
            yield "Gemini not configured."
            return
        
        position_text = self._format_position(board_state)
        legal_moves = [m + 1 for m in board_state.get("legal_moves", list(range(9)))]
        ai_data_text = self._format_ai_data(model_probs or {})
        prompt = self._build_suggest_prompt(position_text, legal_moves, ai_data_text)
        
        try:
            gen_cfg = self._build_generate_config(
                max_output_tokens=3000,  # Increased for detailed move suggestions: ~15 paragraphs
                temperature=0.4,
            )
            
            # Use async queue to bridge sync stream to async generator
            import queue
            import threading
            
            chunk_queue: queue.Queue = queue.Queue()
            error_container = {"error": None}
            
            def stream_worker():
                """Worker thread to consume sync stream and put chunks in queue."""
                try:
                    response_stream = self.client.models.generate_content_stream(
                        model=self.model,
                        contents=prompt,
                        config=gen_cfg
                    )
                    for chunk in response_stream:
                        text = self._response_to_text(chunk)
                        if text:
                            chunk_queue.put(text)
                except Exception as e:
                    error_container["error"] = str(e)
                finally:
                    chunk_queue.put(None)  # Signal end of stream
            
            # Start worker thread
            worker = threading.Thread(target=stream_worker, daemon=True)
            worker.start()
            
            # Async consume from queue
            while True:
                try:
                    chunk = chunk_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)  # Yield control to event loop
                    continue
                
                if chunk is None:  # End of stream
                    if error_container["error"]:
                        yield f"Error: {error_container['error']}"
                    break
                
                yield chunk
                    
        except Exception as e:
            yield f"Error during suggestion: {str(e)}"

    async def voice_conversation_stream(
        self,
        user_query: str,
        previous_analysis: str,
        board_state: Dict,
        move_history: List[Dict] = None
    ):
        """
        Handle voice conversation - user asks a follow-up question about the game.
        Streams response with context of previous analysis.
        """
        if not self.client:
            yield "Gemini не настроен."
            return
        
        position_text = self._format_position(board_state)
        history_text = self._format_move_history(move_history or [])
        
        prompt = f"""{self.SYSTEM_PROMPT}

═══ ТЕКУЩАЯ ПОЗИЦИЯ ═══
{position_text}

{history_text}

═══ ПРЕДЫДУЩИЙ АНАЛИЗ ═══
{previous_analysis if previous_analysis else "Анализ ещё не проводился."}

═══ ВОПРОС ПОЛЬЗОВАТЕЛЯ ═══
🎤 {user_query}

═══ ЗАДАНИЕ ═══
Ответь на вопрос пользователя, учитывая контекст позиции и предыдущего анализа.
Отвечай кратко и по существу. Если вопрос касается конкретного хода - объясни его.
Если вопрос общий о стратегии - дай краткий совет."""
        
        try:
            gen_cfg = self._build_generate_config(
                max_output_tokens=1500,
                temperature=0.5,
            )
            
            import queue
            import threading
            
            chunk_queue: queue.Queue = queue.Queue()
            error_container = {"error": None}
            
            def stream_worker():
                try:
                    response_stream = self.client.models.generate_content_stream(
                        model=self.model,
                        contents=prompt,
                        config=gen_cfg
                    )
                    for chunk in response_stream:
                        text = self._response_to_text(chunk)
                        if text:
                            chunk_queue.put(text)
                except Exception as e:
                    error_container["error"] = str(e)
                finally:
                    chunk_queue.put(None)
            
            worker = threading.Thread(target=stream_worker, daemon=True)
            worker.start()
            
            while True:
                try:
                    chunk = chunk_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                
                if chunk is None:
                    if error_container["error"]:
                        yield f"Ошибка: {error_container['error']}"
                    break
                
                yield chunk
                    
        except Exception as e:
            yield f"Ошибка: {str(e)}"

    
    async def comment_move(
        self,
        board_before: Dict,
        board_after: Dict,
        move: int,
        player: str
    ) -> Dict:
        """
        Provide commentary on a move that was just played - in Russian.
        """
        if not self.client:
            return {
                "available": False,
                "error": "Gemini not configured"
            }
        
        # Calculate changes
        if player == "white":
            kazan_before = board_before.get("white_kazan", 0)
            kazan_after = board_after.get("white_kazan", 0)
            player_ru = "Белые"
        else:
            kazan_before = board_before.get("black_kazan", 0)
            kazan_after = board_after.get("black_kazan", 0)
            player_ru = "Чёрные"
        
        kazan_gain = kazan_after - kazan_before
        hole_name = self.HOLE_NAMES.get(move, "")
        
        before_text = self._format_position(board_before)
        after_text = self._format_position(board_after)
        
        prompt = f"""{self.SYSTEM_PROMPT}

═══ КОММЕНТАРИЙ К ХОДУ ═══

{player_ru} сыграли лунку {move} ({hole_name}).

ПОЗИЦИЯ ДО ХОДА:
{before_text}

ПОЗИЦИЯ ПОСЛЕ ХОДА:
{after_text}

Захвачено кумалаков: {kazan_gain}

ЗАДАНИЕ: Дай краткий комментарий (2-3 предложения):
1. Насколько хорош этот ход? (отлично / хорошо / нормально / сомнительно / ошибка)
2. Чего добился игрок этим ходом?
3. Был ли лучший вариант?"""

        try:
            gen_cfg = self._build_generate_config(
                max_output_tokens=500,
                temperature=0.4,
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
        legal_with_names = [f"{m} ({self.HOLE_NAMES.get(m, '')})" for m in legal_moves]
        
        prompt = f"""Ты — гроссмейстер Тогыз Кумалака. Оцени вероятность того, что каждый ход является лучшим.

{position_text}

ДОСТУПНЫЕ ХОДЫ: {', '.join(legal_with_names)}

ОТВЕТЬ СТРОГО В JSON-ФОРМАТЕ. Ключи — номера лунок, значения — вероятности (сумма = 1.0).

Пример:
{{"3": 0.6, "5": 0.25, "7": 0.15}}

Только JSON, без пояснений."""

        try:
            from google.genai import types
            gen_cfg = types.GenerateContentConfig(
                max_output_tokens=200,
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
