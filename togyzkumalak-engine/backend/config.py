"""
Configuration settings for the Togyzkumalak Engine.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GameConfig:
    """Game-related configuration."""
    initial_kumalaks: int = 9
    total_kumalaks: int = 162
    win_threshold: int = 82  # >81 wins
    draw_threshold: int = 81
    max_moves: int = 500  # Prevent infinite games


@dataclass
class AIConfig:
    """AI-related configuration."""
    default_level: int = 3
    thinking_time_ms: int = 1000
    epsilon: float = 0.1  # Exploration rate
    
    # Model paths - models are in 'models/' directory
    model_dir: str = "models"
    default_model: str = "policy_net_v1.pth"


@dataclass
class ELOConfig:
    """ELO rating configuration."""
    initial_elo: int = 1500
    k_factor: int = 32
    min_elo: int = 100
    max_elo: int = 3000
    
    # AI level ELO estimates
    level_elos: dict = None
    
    def __post_init__(self):
        self.level_elos = {
            1: 800,   # Random
            2: 1200,  # Heuristic
            3: 1500,  # Basic NN
            4: 1800,  # Advanced NN
            5: 2100,  # Expert NN
            6: 2400,  # Gemini AI
            7: 2200,  # PROBS AI
            8: 2500,  # Ensemble AI
        }


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: list = None
    
    def __post_init__(self):
        self.cors_origins = ["*"]


@dataclass
class GeminiConfig:
    """Gemini API configuration."""
    api_key: Optional[str] = None
    model: str = "gemini-2.0-flash"  # Use stable 2.0 Flash
    tts_model: str = "gemini-2.5-flash-preview-tts"  # TTS model for voice
    max_tokens: int = 8000  # Increased for longer analysis output
    temperature: float = 0.5  # Balanced: creative but focused (was 0.7)
    thinking_level: str = "HIGH"  # HIGH thinking for better analysis


@dataclass
class GroqConfig:
    """Groq API configuration for STT."""
    api_key: Optional[str] = None
    stt_model: str = "whisper-large-v3-turbo"
    api_url: str = "https://api.groq.com/openai/v1/audio/transcriptions"


# Global config instances
game_config = GameConfig()
ai_config = AIConfig()
elo_config = ELOConfig()
server_config = ServerConfig()

# Gemini API key from environment variable (REQUIRED)
import os
_gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not _gemini_api_key:
    print("[WARNING] GEMINI_API_KEY environment variable not set. Gemini features will be disabled.")
gemini_config = GeminiConfig(api_key=_gemini_api_key)

# Groq API key for STT
_groq_api_key = os.environ.get("GROQ_API_KEY")
if not _groq_api_key:
    print("[WARNING] GROQ_API_KEY environment variable not set. Voice input will be disabled.")
groq_config = GroqConfig(api_key=_groq_api_key)

