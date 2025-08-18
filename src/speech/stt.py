"""
Advanced Speech-to-Text Engine with Vosk (offline) and Whisper (fallback)
Supports Hindi-English code-switching
"""

import os
import json
import tempfile
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize availability flags
VOSK_AVAILABLE = False
PYAUDIO_AVAILABLE = False
WHISPER_AVAILABLE = False
AUDIO_PROCESSING_AVAILABLE = False

# Try imports
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError as e:
    logger.warning("PyAudio not available. Microphone input will be disabled.")

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    logger.warning("Vosk not available. Some speech recognition features may be limited.")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import soundfile as sf
    import numpy as np
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

from src.utils.logger import log

class SpeechToText:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vosk_model = None
        self.whisper_model = None
        
        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize speech recognition models dynamically."""
        # Initialize Vosk (offline Hindi model)
        if VOSK_AVAILABLE:
            try:
                model_path = Path("models/vosk-hindi")
                if model_path.exists():
                    self.vosk_model = vosk.Model(str(model_path))
                    log.info("Vosk Hindi model loaded successfully")
                else:
                    log.warning("Vosk Hindi model not found, will download on first use")
            except Exception as e:
                log.error(f"Failed to load Vosk model: {e}")

        # Initialize Whisper (fallback)
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                log.info("Whisper model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load Whisper model: {e}")

    def listen_from_microphone(self, duration: int = 5) -> Dict[str, Any]:
        """Listen from microphone using Vosk with Whisper fallback."""
        try:
            if not AUDIO_PROCESSING_AVAILABLE:
                return {
                    "success": False,
                    "text": "",
                    "error": "Audio processing libraries not available"
                }

            # Try Vosk first (offline)
            if self.vosk_model and VOSK_AVAILABLE:
                result = self._listen_vosk(duration)
                if result["success"]:
                    return result

            # Fallback to Whisper
            if self.whisper_model:
                return self._listen_whisper(duration)

            return {
                "success": False,
                "text": "",
                "error": "No speech recognition models available"
            }

        except Exception as e:
            log.error(f"Microphone listening failed: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }

    def _listen_vosk(self, duration: int) -> Dict[str, Any]:
        """Listen using Vosk offline model."""
        try:
            if not PYAUDIO_AVAILABLE:
                raise ImportError("PyAudio not available")

            # Audio recording parameters
            RATE = 16000
            CHUNK = 4000

            # Initialize audio stream
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            # Initialize recognizer
            rec = vosk.KaldiRecognizer(self.vosk_model, RATE)
            rec.SetWords(True)

            # Record audio
            stream.start_stream()
            frames = []
            for _ in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        break

            # Final result
            final_result = json.loads(rec.FinalResult())

            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()

            text = final_result.get("text", "").strip()
            return {
                "success": bool(text),
                "text": text,
                "confidence": final_result.get("confidence", 0.0),
                "method": "vosk"
            }

        except Exception as e:
            log.error(f"Vosk listening failed: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }

    def _listen_whisper(self, timeout: int) -> Dict[str, Any]:
        """Listen using Whisper as fallback."""
        temp_path = None
        try:
            if not PYAUDIO_AVAILABLE:
                raise ImportError("PyAudio is not available. Please ensure it's properly installed.")
            
            if not WHISPER_AVAILABLE or not hasattr(self, 'whisper_model'):
                raise ImportError("Whisper model is not properly initialized.")

            # Record audio to temporary file
            RATE = 16000
            CHUNK = 1024
            
            p = pyaudio.PyAudio()

            # Get default input device info
            default_input = p.get_default_input_device_info()
            if not default_input:
                raise RuntimeError("No default input device found")

            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=default_input['index']
            )

            logger.info("Recording audio...")
            frames = []
            stream.start_stream()
            
            for _ in range(0, int(RATE / CHUNK * timeout)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except IOError as e:
                    logger.warning(f"Audio input overflow: {e}")
                    continue

            logger.info("Finished recording")
            stream.stop_stream()
            stream.close()
            p.terminate()

            if not frames:
                raise RuntimeError("No audio data recorded")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            # Convert to audio format
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            sf.write(temp_path, audio_data, RATE)

            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"Failed to create temporary audio file at {temp_path}")

            logger.info(f"Transcribing audio file: {temp_path}")

            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                temp_path,
                language="hi",
                fp16=False  # Disable FP16 for better compatibility
            )

            return {
                "success": True,
                "text": result.get("text", "").strip(),
                "confidence": 1.0,  # Whisper doesn't provide confidence
                "method": "whisper"
            }

        except Exception as e:
            log.error(f"Whisper listening failed: {e}")
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

    def transcribe_file(self, uploaded_file) -> Dict[str, Any]:
        """Transcribe uploaded audio file."""
        temp_path = None
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            # Try Whisper for file transcription (better for files)
            if self.whisper_model:
                result = self.whisper_model.transcribe(temp_path, language="hi")
                return {
                    "success": True,
                    "text": result["text"].strip(),
                    "method": "whisper_file"
                }

            # Fallback: return error if no models available
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

    def is_available(self) -> bool:
        """Check if STT is available."""
        return (self.vosk_model is not None) or (self.whisper_model is not None)
