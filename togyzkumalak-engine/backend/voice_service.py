"""
Voice Service - TTS and STT integration.
"""

import asyncio
import base64
import httpx
import queue
import threading
from typing import Optional, AsyncGenerator
from google import genai
from google.genai import types

from .config import gemini_config, groq_config


class VoiceService:
    DEFAULT_VOICE = "Kore"
    
    def __init__(self):
        self.tts_client = None
        self.tts_model = gemini_config.tts_model
        self.groq_api_key = groq_config.api_key
        self.stt_model = groq_config.stt_model
        self.stt_url = groq_config.api_url
        
        if gemini_config.api_key:
            try:
                self.tts_client = genai.Client(api_key=gemini_config.api_key)
                print(f"[OK] Voice TTS: {self.tts_model}")
            except Exception as e:
                print(f"[ERROR] TTS: {e}")
        else:
            print("[WARNING] TTS not available")
        
        if not self.groq_api_key:
            print("[WARNING] STT not available")
        else:
            print(f"[OK] Voice STT: {self.stt_model}")
    
    def is_tts_available(self):
        return self.tts_client is not None
    
    def is_stt_available(self):
        return self.groq_api_key is not None
    
    async def text_to_speech_stream(self, text, voice=None):
        if not self.tts_client:
            return
        voice = voice or self.DEFAULT_VOICE
        prompt = f"Say clearly:\n\n{text}"
        
        # #region agent log
        import json, time
        with open(r'c:\Users\Admin\Documents\Toguzkumalak\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"location": "voice_service.py:53", "message": "text_to_speech_stream entry", "data": {"text": text[:50], "voice": voice}, "timestamp": int(time.time()*1000), "sessionId": "debug-session", "hypothesisId": "1"}) + "\n")
        # #endregion

        try:
            q = queue.Queue()
            err = {"e": None}
            def w():
                try:
                    # Use generate_content_stream for TRUE streaming
                    # IMPORTANT: For AUDIO output, many SDKs don't support true partial audio chunks
                    # but we can try to force smaller yields to the browser.
                    responses = self.tts_client.models.generate_content_stream(
                        model=self.tts_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                                )
                            ),
                        )
                    )
                    
                    chunk_count = 0
                    for r in responses:
                        if r.candidates and r.candidates[0].content and r.candidates[0].content.parts:
                            for p in r.candidates[0].content.parts:
                                if hasattr(p, "inline_data") and p.inline_data:
                                    d = p.inline_data.data
                                    if d:
                                        chunk_count += 1
                                        # Don't chunk the audio data - send as complete blob
                                        # Audio files must remain intact to avoid corruption/noise
                                        q.put(d)
                    
                    # #region agent log
                    with open(r'c:\Users\Admin\Documents\Toguzkumalak\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"location": "voice_service.py:88", "message": "TTS stream generation finished", "data": {"chunk_count": chunk_count}, "timestamp": int(time.time()*1000), "sessionId": "debug-session", "hypothesisId": "1"}) + "\n")
                    # #endregion

                except Exception as ex:
                    err["e"] = str(ex)
                    print(f"[TTS Stream Error] {ex}")
                    # #region agent log
                    with open(r'c:\Users\Admin\Documents\Toguzkumalak\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"location": "voice_service.py:96", "message": "TTS stream generation error", "data": {"err": str(ex)}, "timestamp": int(time.time()*1000), "sessionId": "debug-session", "hypothesisId": "1"}) + "\n")
                    # #endregion
                finally:
                    q.put(None)
            t = threading.Thread(target=w, daemon=True)
            t.start()
            while True:
                try:
                    c = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                if c is None:
                    break
                yield c
        except Exception as ex:
            print(f"[TTS Error] {ex}")
            # #region agent log
            with open(r'c:\Users\Admin\Documents\Toguzkumalak\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"location": "voice_service.py:115", "message": "TTS stream outer error", "data": {"err": str(ex)}, "timestamp": int(time.time()*1000), "sessionId": "debug-session", "hypothesisId": "1"}) + "\n")
            # #endregion
    
    async def text_to_speech(self, text, voice=None):
        if not self.tts_client:
            return None
        voice = voice or self.DEFAULT_VOICE
        prompt = f"Say clearly:\n\n{text}"
        try:
            r = self.tts_client.models.generate_content(
                model=self.tts_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        )
                    ),
                )
            )
            if r.candidates and r.candidates[0].content and r.candidates[0].content.parts:
                for p in r.candidates[0].content.parts:
                    if hasattr(p, "inline_data") and p.inline_data:
                        return p.inline_data.data
            return None
        except Exception as ex:
            print(f"[TTS Error] {ex}")
            return None
    
    async def speech_to_text(self, audio_data, language="ru"):
        if not self.groq_api_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("audio.wav", audio_data, "audio/wav")}
                data = {"model": self.stt_model, "language": language, "response_format": "json", "temperature": 0.0}
                headers = {"Authorization": f"Bearer {self.groq_api_key}"}
                resp = await client.post(self.stt_url, files=files, data=data, headers=headers)
                if resp.status_code == 200:
                    return resp.json().get("text", "")
                return None
        except Exception as ex:
            print(f"[STT Error] {ex}")
            return None


voice_service = VoiceService()