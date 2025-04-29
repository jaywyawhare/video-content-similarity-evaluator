import whisper
import os
import numpy as np
from typing import Optional
from ..core.logger import get_logger
from ..config import settings 

logger = get_logger(__name__) 


class TranscriptionService:
    def __init__(self):
        self.model = None

    async def load_model(self):
        """Load the whisper model with proper error handling"""
        if self.model is None:
            try:
                self.model = whisper.load_model(settings.WHISPER_MODEL)
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {str(e)}")
                raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    async def warmup(self):
        """Warm up the model with a dummy input"""
        try:
            await self.load_model()  # Ensure model is loaded first
            dummy_audio = np.zeros((16000,), dtype=np.float32) 
            if self.model is None:
                raise RuntimeError("Model failed to load during warmup")
            self.model.transcribe(dummy_audio)
            logger.info("Whisper model warmed up successfully")
        except Exception as e:
            logger.error(f"Error warming up Whisper model: {str(e)}")
            raise  

    async def validate_audio(self, audio_path: str) -> bool:
        """Validate audio file before transcription"""
        try:
            with open(audio_path, "rb") as f:
                if os.path.getsize(audio_path) == 0:
                    raise ValueError("Audio file is empty")

                header = f.read(4)
                if header != b"RIFF":
                    raise ValueError("Invalid audio format")

            return True
        except Exception as e:
            raise ValueError(f"Audio validation failed: {str(e)}")

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text"""
        if self.model is None:
            await self.load_model()
            if self.model is None:
                raise RuntimeError("Failed to initialize Whisper model")
        try:
            await self.validate_audio(audio_path)
            result = self.model.transcribe(audio_path)
            text = result.get("text", "").strip()

            if not text:
                logger.warning(f"Empty transcription for {audio_path}")
                raise ValueError("Transcription resulted in empty text")

            text = self._preprocess_text(text)
            return text

        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize transcribed text"""
        import re

        text = text.lower()
        text = " ".join(text.split())
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        text = re.sub(r"\b\d+\b", "", text)
        text = " ".join(text.split())

        return text.strip()
