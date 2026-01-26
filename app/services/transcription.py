"""
WhisperX Transcription Service.

Provides audio transcription using WhisperX with word-level alignment.
"""
import os
import logging
from pathlib import Path
from typing import Optional
import tempfile

import torch

# Fix for PyTorch 2.6+ compatibility with WhisperX/Pyannote
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import whisperx

from app.config import get_settings

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for transcribing audio files using WhisperX."""
    
    _instance: Optional['TranscriptionService'] = None
    _model = None
    _align_model = None
    _align_metadata = None
    
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.webm', '.ogg', '.flac'}
    
    def __new__(cls):
        """Singleton pattern for lazy model loading."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.settings = get_settings()
        self.device = self.settings.whisperx_device
        self.compute_type = self.settings.whisperx_compute_type
        self.model_name = self.settings.whisperx_model
        
        # Adjust compute type for CPU
        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "float32"
            logger.warning("Changed compute_type to float32 for CPU device")
    
    def _load_model(self):
        """Lazy load WhisperX model."""
        if self._model is None:
            logger.info(f"Loading WhisperX model: {self.model_name} on {self.device}")
            self._model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type
            )
            logger.info("WhisperX model loaded successfully")
        return self._model
    
    def _load_align_model(self, language_code: str):
        """Load alignment model for word-level timestamps."""
        if self._align_model is None or self._align_metadata is None:
            logger.info(f"Loading alignment model for language: {language_code}")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
        return self._align_model, self._align_metadata
    
    def transcribe(self, audio_path: str, language: str = "ru") -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, webm, etc.)
            language: Language code (default: "ru" for Russian)
            
        Returns:
            Transcribed text as a single string
        """
        # Validate file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Validate format
        ext = Path(audio_path).suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {ext}. Supported: {self.SUPPORTED_FORMATS}")
        
        logger.info(f"Starting transcription: {audio_path}")
        
        # Load model
        model = self._load_model()
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe
        result = model.transcribe(audio, batch_size=16, language=language)
        logger.info(f"Transcription complete. Detected language: {result.get('language', language)}")
        
        # Get detected language
        detected_language = result.get("language", language)
        
        # Align for word-level timestamps (optional, improves quality)
        try:
            align_model, metadata = self._load_align_model(detected_language)
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            logger.info("Word alignment complete")
        except Exception as e:
            logger.warning(f"Could not perform word alignment: {e}")
        
        # Extract text from segments
        segments = result.get("segments", [])
        text_parts = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if text:
                text_parts.append(text)
        
        full_text = " ".join(text_parts)
        logger.info(f"Transcription result: {len(full_text)} characters")
        
        return full_text
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported."""
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_FORMATS


# Singleton instance getter
_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Get or create transcription service instance."""
    global _service
    if _service is None:
        _service = TranscriptionService()
    return _service
