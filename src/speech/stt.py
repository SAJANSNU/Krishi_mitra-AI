import tempfile
import logging
import os
from typing import Dict, Any
from src.utils.logger import log

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

class SpeechToText:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.whisper_model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Whisper model, if available."""
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                log.info("Whisper model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load Whisper model: {e}")

    def transcribe_file(self, uploaded_file) -> Dict[str, Any]:
        """Transcribe uploaded audio file (WAV/MP3/M4A etc.) using Whisper."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            if self.whisper_model:
                result = self.whisper_model.transcribe(temp_path, language="hi")
                return {
                    "success": True,
                    "text": result["text"].strip(),
                    "method": "whisper_file"
                }
            return {
                "success": False,
                "text": "",
                "error": "No transcription models available"
            }
        except Exception as e:
            log.error(f"File transcription failed: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def transcribe_blob(self, audio_blob) -> Dict[str, Any]:
        """
        Transcribe audio from an in-memory bytes-like object (from browser recorder).
        """
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_blob.read())
                temp_path = temp_file.name
            if self.whisper_model:
                result = self.whisper_model.transcribe(temp_path, language="hi")
                return {
                    "success": True,
                    "text": result["text"].strip(),
                    "method": "whisper_blob"
                }
            return {
                "success": False,
                "text": "",
                "error": "No transcription models available"
            }
        except Exception as e:
            log.error(f"Blob transcription failed: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def is_available(self) -> bool:
        """Check if STT is available (True if Whisper loads)"""
        return self.whisper_model is not None
