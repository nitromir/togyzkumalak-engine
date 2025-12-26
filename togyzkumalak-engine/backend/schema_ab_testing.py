"""
A/B Testing for Gemini Schema Formats.

Tests REAL different schema formats for Gemini commentary.
All results are stored on disk and can be analyzed.
NO MOCKS - all data comes from real Gemini API calls.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .config import gemini_config


class SchemaVariant(Enum):
    """Available schema variants for A/B testing."""
    MINIMAL = "minimal"           # Just move + score
    STRUCTURED = "structured"     # JSON with categories
    NARRATIVE = "narrative"       # Natural language context
    KAZAKH_NAMES = "kazakh_names" # Uses traditional hole names
    TACTICAL = "tactical"         # Focus on threats/opportunities
    BEGINNER = "beginner"         # Simplified for new players


@dataclass
class SchemaExperiment:
    """A/B test experiment configuration."""
    experiment_id: str
    name: str
    variants: List[str]  # Variant names
    traffic_split: Dict[str, float]  # variant -> percentage (0.0-1.0)
    start_date: str
    end_date: Optional[str] = None
    is_active: bool = True
    description: str = ""


@dataclass
class SchemaTestResult:
    """Result of a single schema test from REAL Gemini API call."""
    experiment_id: str
    variant: str
    game_id: str
    move_number: int
    
    # Quality metrics (collected via user feedback)
    user_rating: Optional[int] = None  # 1-5 stars
    was_helpful: Optional[bool] = None
    was_accurate: Optional[bool] = None
    
    # Real API metrics
    gemini_response_time_ms: int = 0
    gemini_tokens_used: int = 0
    gemini_response_text: str = ""
    
    # Automated quality checks
    hallucination_detected: bool = False
    response_length: int = 0
    
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "variant": self.variant,
            "game_id": self.game_id,
            "move_number": self.move_number,
            "user_rating": self.user_rating,
            "was_helpful": self.was_helpful,
            "was_accurate": self.was_accurate,
            "gemini_response_time_ms": self.gemini_response_time_ms,
            "gemini_tokens_used": self.gemini_tokens_used,
            "gemini_response_text": self.gemini_response_text,
            "hallucination_detected": self.hallucination_detected,
            "response_length": self.response_length,
            "timestamp": self.timestamp
        }


class SchemaGenerator:
    """Generates different REAL schema formats for Gemini."""
    
    HOLE_NAMES = {
        1: "Арт", 2: "Тектұрмас", 3: "Ат өтпес",
        4: "Атсыратар", 5: "Бел", 6: "Белбасар",
        7: "Қандықақпан", 8: "Көкмойын", 9: "Маңдай"
    }
    
    HOLE_MEANINGS = {
        1: "Начало пути",
        2: "Упрямый конь",
        3: "Конь не пройдёт",
        4: "Конь споткнётся",
        5: "Пояс (центр)",
        6: "Переход пояса",
        7: "Кровавая ловушка",
        8: "Синяя шея",
        9: "Лоб (главная)"
    }
    
    def generate(
        self, 
        variant: SchemaVariant,
        board_state: Dict,
        move: int,
        policy_data: Optional[Dict] = None
    ) -> Tuple[Dict, str]:
        """
        Generate schema in specified variant format.
        
        Returns:
            Tuple of (schema_dict, prompt_for_gemini)
        """
        generators = {
            SchemaVariant.MINIMAL: self._minimal_schema,
            SchemaVariant.STRUCTURED: self._structured_schema,
            SchemaVariant.NARRATIVE: self._narrative_schema,
            SchemaVariant.KAZAKH_NAMES: self._kazakh_schema,
            SchemaVariant.TACTICAL: self._tactical_schema,
            SchemaVariant.BEGINNER: self._beginner_schema,
        }
        
        generator = generators.get(variant, self._structured_schema)
        return generator(board_state, move, policy_data)
    
    def _minimal_schema(self, board_state: Dict, move: int, policy_data: Dict) -> Tuple[Dict, str]:
        """Minimal schema - just essential data."""
        schema = {
            "format": "minimal",
            "move": move,
            "current_player": board_state.get("current_player"),
            "score": {
                "white": board_state.get("white_kazan", 0),
                "black": board_state.get("black_kazan", 0)
            }
        }
        
        prompt = f"""Ход {move}. Счёт: {schema['score']['white']}:{schema['score']['black']}.
Дай краткий комментарий (1-2 предложения)."""
        
        return schema, prompt
    
    def _structured_schema(self, board_state: Dict, move: int, policy_data: Dict) -> Tuple[Dict, str]:
        """Structured JSON with all categories."""
        move_quality = "unknown"
        confidence = 0
        
        if policy_data:
            action_probs = policy_data.get("action_probs", [])
            if action_probs and len(action_probs) > move - 1:
                confidence = action_probs[move - 1] * 100
                max_prob = max(action_probs) if action_probs else 0
                if action_probs[move - 1] >= max_prob * 0.95:
                    move_quality = "best"
                elif action_probs[move - 1] >= max_prob * 0.7:
                    move_quality = "good"
                elif action_probs[move - 1] >= max_prob * 0.4:
                    move_quality = "inaccuracy"
                else:
                    move_quality = "mistake"
        
        schema = {
            "format": "structured",
            "move": {
                "pit": move,
                "quality": move_quality,
                "confidence_percent": round(confidence, 1)
            },
            "position": {
                "white_pits": board_state.get("white_pits", []),
                "black_pits": board_state.get("black_pits", []),
                "white_kazan": board_state.get("white_kazan", 0),
                "black_kazan": board_state.get("black_kazan", 0),
                "white_tuzduk": board_state.get("white_tuzduk"),
                "black_tuzduk": board_state.get("black_tuzduk"),
                "current_player": board_state.get("current_player")
            },
            "evaluation": {
                "value_estimate": policy_data.get("value_estimate") if policy_data else None,
                "phase": self._detect_phase(board_state)
            }
        }
        
        prompt = f"""Анализ позиции Тогыз Кумалак:

Ход: {move} (качество: {move_quality}, уверенность: {confidence:.0f}%)
Счёт: {schema['position']['white_kazan']}:{schema['position']['black_kazan']}
Фаза: {schema['evaluation']['phase']}

Белые лунки: {schema['position']['white_pits']}
Чёрные лунки: {schema['position']['black_pits']}

Дай развёрнутый комментарий (3-4 предложения):
- Оценка хода
- Почему это хороший/плохой выбор
- Что следует учитывать"""
        
        return schema, prompt
    
    def _narrative_schema(self, board_state: Dict, move: int, policy_data: Dict) -> Tuple[Dict, str]:
        """Natural language context for Gemini."""
        white_kazan = board_state.get("white_kazan", 0)
        black_kazan = board_state.get("black_kazan", 0)
        
        score_diff = white_kazan - black_kazan
        if score_diff > 15:
            balance = "Белые доминируют"
        elif score_diff > 5:
            balance = "Белые впереди"
        elif score_diff > -5:
            balance = "Позиция равная"
        elif score_diff > -15:
            balance = "Чёрные впереди"
        else:
            balance = "Чёрные доминируют"
        
        player = board_state.get("current_player", "white")
        player_ru = "Белые" if player == "white" else "Чёрные"
        
        schema = {
            "format": "narrative",
            "context": f"{player_ru} делают ход {move}. Счёт: {white_kazan}:{black_kazan}. {balance}.",
            "move": move,
            "phase": self._detect_phase(board_state)
        }
        
        prompt = f"""{schema['context']}

Стадия игры: {schema['phase']}.

Прокомментируй этот ход как профессиональный комментатор турнира.
Стиль: живой, эмоциональный, понятный зрителям."""
        
        return schema, prompt
    
    def _kazakh_schema(self, board_state: Dict, move: int, policy_data: Dict) -> Tuple[Dict, str]:
        """Schema with traditional Kazakh hole names."""
        hole_name = self.HOLE_NAMES.get(move, str(move))
        meaning = self.HOLE_MEANINGS.get(move, "")
        
        schema = {
            "format": "kazakh_names",
            "move": {
                "pit_number": move,
                "kazakh_name": hole_name,
                "meaning": meaning
            },
            "position": {
                "white_pits_named": [(i+1, self.HOLE_NAMES[i+1], v) 
                              for i, v in enumerate(board_state.get("white_pits", [9]*9))],
                "black_pits_named": [(i+1, self.HOLE_NAMES[i+1], v) 
                              for i, v in enumerate(board_state.get("black_pits", [9]*9))],
                "score": f"{board_state.get('white_kazan', 0)}:{board_state.get('black_kazan', 0)}"
            }
        }
        
        prompt = f"""Ход из лунки "{hole_name}" ({meaning}) - лунка №{move}.

Счёт: {schema['position']['score']}

Используя традиционные казахские названия лунок, объясни этот ход.
Упомяни культурное значение выбранной лунки если уместно."""
        
        return schema, prompt
    
    def _tactical_schema(self, board_state: Dict, move: int, policy_data: Dict) -> Tuple[Dict, str]:
        """Focus on tactical elements."""
        white_pits = board_state.get("white_pits", [9]*9)
        black_pits = board_state.get("black_pits", [9]*9)
        
        # Detect threats
        threats = []
        opportunities = []
        
        for i, count in enumerate(white_pits):
            if count == 3:
                threats.append(f"Лунка {i+1} белых угрожает туздуком")
        for i, count in enumerate(black_pits):
            if count == 3:
                threats.append(f"Лунка {i+1} чёрных угрожает туздуком")
        
        # Check move destination
        pit_count = white_pits[move - 1] if board_state.get("current_player") == "white" else black_pits[move - 1]
        if pit_count > 0:
            opportunities.append(f"Ход распределяет {pit_count} кумалаков")
        
        schema = {
            "format": "tactical",
            "move": move,
            "tactics": {
                "threats": threats,
                "opportunities": opportunities,
                "pit_count": pit_count
            },
            "position_summary": {
                "score_diff": board_state.get("white_kazan", 0) - board_state.get("black_kazan", 0),
                "total_on_board": sum(white_pits) + sum(black_pits)
            }
        }
        
        threats_text = "; ".join(threats) if threats else "Нет явных угроз"
        
        prompt = f"""ТАКТИЧЕСКИЙ АНАЛИЗ хода {move}:

Угрозы на доске: {threats_text}
Кумалаков в выбранной лунке: {pit_count}
Разница в счёте: {schema['position_summary']['score_diff']:+d}

Проанализируй тактические последствия этого хода.
Какие варианты открываются? Какие угрозы создаются или нейтрализуются?"""
        
        return schema, prompt
    
    def _beginner_schema(self, board_state: Dict, move: int, policy_data: Dict) -> Tuple[Dict, str]:
        """Simplified schema for new players."""
        white_kazan = board_state.get("white_kazan", 0)
        black_kazan = board_state.get("black_kazan", 0)
        
        # Simple tip based on phase
        phase = self._detect_phase(board_state)
        if phase == "opening":
            tip = "В начале игры важно равномерно распределять камни"
        elif phase == "midgame":
            tip = "Ищите возможности для захвата и создания туздука"
        else:
            tip = "В конце игры каждый камень на счету"
        
        schema = {
            "format": "beginner",
            "move": {
                "pit": move,
                "simple_tip": tip
            },
            "score": {
                "you": white_kazan,
                "opponent": black_kazan
            },
            "phase": phase
        }
        
        prompt = f"""Объясни ход {move} ПРОСТЫМИ СЛОВАМИ для начинающего игрока.

Счёт: {white_kazan}:{black_kazan}
Стадия игры: {phase}

Совет: {tip}

Используй простой язык. Избегай сложных терминов.
Объясни ПОЧЕМУ этот ход хороший или плохой.
2-3 коротких предложения максимум."""
        
        return schema, prompt
    
    def _detect_phase(self, board_state: Dict) -> str:
        """Detect game phase based on captured stones."""
        total_captured = board_state.get("white_kazan", 0) + board_state.get("black_kazan", 0)
        if total_captured < 20:
            return "opening"
        elif total_captured < 100:
            return "midgame"
        else:
            return "endgame"


class ABTestManager:
    """
    Manages REAL A/B testing experiments.
    
    All data is stored on disk. No mocks.
    """
    
    def __init__(self, storage_path: str = "logs/ab_tests"):
        self.storage_path = storage_path
        self.experiments: Dict[str, SchemaExperiment] = {}
        self.schema_generator = SchemaGenerator()
        self.gemini_client = None
        
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "results"), exist_ok=True)
        
        self._load_experiments()
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini client for real API calls."""
        api_key = gemini_config.api_key or os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"[WARNING] Failed to init Gemini for A/B testing: {e}")
    
    def _load_experiments(self):
        """Load experiments from disk."""
        experiments_file = os.path.join(self.storage_path, "experiments.json")
        if os.path.exists(experiments_file):
            try:
                with open(experiments_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for exp_id, exp_data in data.items():
                    self.experiments[exp_id] = SchemaExperiment(
                        experiment_id=exp_data["experiment_id"],
                        name=exp_data["name"],
                        variants=exp_data["variants"],
                        traffic_split=exp_data["traffic_split"],
                        start_date=exp_data["start_date"],
                        end_date=exp_data.get("end_date"),
                        is_active=exp_data.get("is_active", True),
                        description=exp_data.get("description", "")
                    )
            except Exception as e:
                print(f"Failed to load experiments: {e}")
    
    def _save_experiments(self):
        """Save experiments to disk."""
        experiments_file = os.path.join(self.storage_path, "experiments.json")
        data = {
            exp_id: {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "variants": exp.variants,
                "traffic_split": exp.traffic_split,
                "start_date": exp.start_date,
                "end_date": exp.end_date,
                "is_active": exp.is_active,
                "description": exp.description
            }
            for exp_id, exp in self.experiments.items()
        }
        with open(experiments_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_experiment(
        self,
        name: str,
        variants: List[str] = None,
        traffic_split: Optional[Dict[str, float]] = None,
        description: str = ""
    ) -> SchemaExperiment:
        """Create a new REAL A/B test experiment."""
        if variants is None:
            variants = ["structured", "tactical", "beginner"]
        
        experiment_id = hashlib.md5(f"{name}{datetime.now()}".encode()).hexdigest()[:8]
        
        # Default: equal split
        if traffic_split is None:
            equal_split = 1.0 / len(variants)
            traffic_split = {v: equal_split for v in variants}
        
        experiment = SchemaExperiment(
            experiment_id=experiment_id,
            name=name,
            variants=variants,
            traffic_split=traffic_split,
            start_date=datetime.now().isoformat(),
            description=description
        )
        
        self.experiments[experiment_id] = experiment
        self._save_experiments()
        
        return experiment
    
    def get_variant_for_user(self, experiment_id: str, user_id: str) -> str:
        """Get consistent variant for user (deterministic assignment)."""
        experiment = self.experiments.get(experiment_id)
        if not experiment or not experiment.is_active:
            return "structured"  # Default
        
        # Deterministic assignment based on user_id hash
        hash_value = int(hashlib.md5(f"{user_id}{experiment_id}".encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000.0
        
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += experiment.traffic_split.get(variant, 0)
            if normalized < cumulative:
                return variant
        
        return experiment.variants[0]
    
    def generate_commentary_with_test(
        self,
        experiment_id: str,
        user_id: str,
        game_id: str,
        move_number: int,
        board_state: Dict,
        move: int,
        policy_data: Optional[Dict] = None,
        timeout: int = 30
    ) -> Tuple[str, SchemaTestResult]:
        """
        Generate commentary using A/B tested schema and REAL Gemini API.
        
        Returns:
            Tuple of (commentary_text, test_result)
        """
        variant_name = self.get_variant_for_user(experiment_id, user_id)
        
        try:
            variant = SchemaVariant(variant_name)
        except ValueError:
            variant = SchemaVariant.STRUCTURED
        
        # Generate schema and prompt
        schema, prompt = self.schema_generator.generate(variant, board_state, move, policy_data)
        
        # Make REAL Gemini API call
        start_time = time.time()
        commentary = ""
        tokens_used = 0
        
        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=gemini_config.model,
                    contents=prompt,
                    config={
                        "max_output_tokens": 300,
                        "temperature": 0.7
                    }
                )
                commentary = response.text.strip()
                # Estimate tokens (rough)
                tokens_used = len(prompt.split()) + len(commentary.split())
            except Exception as e:
                commentary = f"Ошибка генерации: {str(e)[:50]}"
        else:
            commentary = "Gemini API не настроен"
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Create result
        result = SchemaTestResult(
            experiment_id=experiment_id,
            variant=variant_name,
            game_id=game_id,
            move_number=move_number,
            gemini_response_time_ms=response_time,
            gemini_tokens_used=tokens_used,
            gemini_response_text=commentary,
            response_length=len(commentary),
            timestamp=datetime.now().isoformat()
        )
        
        # Save result
        self._save_result(result)
        
        return commentary, result
    
    def record_feedback(
        self,
        experiment_id: str,
        variant: str,
        game_id: str,
        move_number: int,
        user_rating: Optional[int] = None,
        was_helpful: Optional[bool] = None,
        was_accurate: Optional[bool] = None
    ):
        """Record user feedback for a test result."""
        # Find and update existing result
        results_file = os.path.join(
            self.storage_path, "results",
            f"results_{experiment_id}.jsonl"
        )
        
        # Append feedback as new entry
        feedback = {
            "type": "feedback",
            "experiment_id": experiment_id,
            "variant": variant,
            "game_id": game_id,
            "move_number": move_number,
            "user_rating": user_rating,
            "was_helpful": was_helpful,
            "was_accurate": was_accurate,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
    
    def _save_result(self, result: SchemaTestResult):
        """Save test result to disk."""
        results_file = os.path.join(
            self.storage_path, "results",
            f"results_{result.experiment_id}.jsonl"
        )
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    
    def get_experiment_stats(self, experiment_id: str) -> Dict:
        """Get REAL statistics for an experiment from actual data."""
        results_file = os.path.join(
            self.storage_path, "results",
            f"results_{experiment_id}.jsonl"
        )
        
        if not os.path.exists(results_file):
            return {"error": "No results yet", "experiment_id": experiment_id}
        
        # Read all results
        results = []
        feedback = []
        
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("type") == "feedback":
                        feedback.append(entry)
                    else:
                        results.append(entry)
                except json.JSONDecodeError:
                    continue
        
        # Match feedback to results
        feedback_by_key = {}
        for fb in feedback:
            key = f"{fb.get('game_id')}_{fb.get('move_number')}"
            feedback_by_key[key] = fb
        
        # Calculate stats by variant
        stats_by_variant = {}
        
        for result in results:
            variant = result.get("variant", "unknown")
            if variant not in stats_by_variant:
                stats_by_variant[variant] = {
                    "sample_size": 0,
                    "total_response_time_ms": 0,
                    "total_tokens": 0,
                    "total_response_length": 0,
                    "ratings": [],
                    "helpful_count": 0,
                    "helpful_total": 0,
                    "accurate_count": 0,
                    "accurate_total": 0,
                }
            
            stats = stats_by_variant[variant]
            stats["sample_size"] += 1
            stats["total_response_time_ms"] += result.get("gemini_response_time_ms", 0)
            stats["total_tokens"] += result.get("gemini_tokens_used", 0)
            stats["total_response_length"] += result.get("response_length", 0)
            
            # Match feedback
            key = f"{result.get('game_id')}_{result.get('move_number')}"
            if key in feedback_by_key:
                fb = feedback_by_key[key]
                if fb.get("user_rating"):
                    stats["ratings"].append(fb["user_rating"])
                if fb.get("was_helpful") is not None:
                    stats["helpful_total"] += 1
                    if fb["was_helpful"]:
                        stats["helpful_count"] += 1
                if fb.get("was_accurate") is not None:
                    stats["accurate_total"] += 1
                    if fb["was_accurate"]:
                        stats["accurate_count"] += 1
        
        # Compute final stats
        final_stats = {}
        for variant, data in stats_by_variant.items():
            n = data["sample_size"]
            final_stats[variant] = {
                "sample_size": n,
                "avg_response_time_ms": data["total_response_time_ms"] / n if n > 0 else 0,
                "avg_tokens": data["total_tokens"] / n if n > 0 else 0,
                "avg_response_length": data["total_response_length"] / n if n > 0 else 0,
                "avg_rating": sum(data["ratings"]) / len(data["ratings"]) if data["ratings"] else None,
                "helpful_rate": data["helpful_count"] / data["helpful_total"] if data["helpful_total"] > 0 else None,
                "accuracy_rate": data["accurate_count"] / data["accurate_total"] if data["accurate_total"] > 0 else None,
                "feedback_count": len(data["ratings"]),
            }
        
        # Determine winner
        winner = self._determine_winner(final_stats)
        
        return {
            "experiment_id": experiment_id,
            "total_samples": len(results),
            "total_feedback": len(feedback),
            "variants": final_stats,
            "winner": winner
        }
    
    def _determine_winner(self, stats: Dict) -> Optional[str]:
        """Determine winning variant based on REAL stats."""
        if not stats:
            return None
        
        scores = {}
        for variant, data in stats.items():
            score = 0
            
            # Rating weight: 40%
            if data.get("avg_rating"):
                score += (data["avg_rating"] / 5.0) * 40
            
            # Helpful rate weight: 30%
            if data.get("helpful_rate"):
                score += data["helpful_rate"] * 30
            
            # Accuracy weight: 20%
            if data.get("accuracy_rate"):
                score += data["accuracy_rate"] * 20
            
            # Response time penalty: -10% for slow responses
            avg_time = data.get("avg_response_time_ms", 0)
            if avg_time > 5000:  # >5 seconds is slow
                score -= 10
            
            # Sample size bonus (more data = more reliable)
            if data.get("sample_size", 0) >= 100:
                score += 5
            
            scores[variant] = score
        
        if scores:
            return max(scores, key=scores.get)
        return None
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments with their status."""
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "variants": exp.variants,
                "is_active": exp.is_active,
                "start_date": exp.start_date,
                "description": exp.description
            }
            for exp in self.experiments.values()
        ]
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].is_active = False
            self.experiments[experiment_id].end_date = datetime.now().isoformat()
            self._save_experiments()
            return True
        return False


# Global instance
ab_test_manager = ABTestManager()

