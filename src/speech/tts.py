"""
Text-to-Speech engine using gTTS (Google Text-to-Speech)
Supports Hindi, English, and simple code-switch Hindi-English queries.
"""

import tempfile
import os
import logging
from typing import Optional

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

from src.utils.logger import log

class TextToSpeech:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def synthesize(self, text: str, language: str = "Hindi-English Mix") -> bytes:
        """
        Synthesize text to speech using gTTS with reliable temp file handling.
        Returns: bytes object for mp3 Streamlit playback.
        """
        if not GTTS_AVAILABLE:
            log.error("gTTS not available")
            return b""

        lang_code = self._get_language_code(language)
        temp_path = None
        
        try:
            # Create gTTS instance
            tts = gTTS(text=text, lang=lang_code, slow=False)

            # Create a temporary file with explicit cleanup
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp3')
            os.close(temp_fd)  # Close the file descriptor as gTTS will open it again

            try:
                # Save to the temp file
                tts.save(temp_path)

                # Read the file content
                with open(temp_path, 'rb') as f:
                    audio_bytes = f.read()

                return audio_bytes

            finally:
                # Ensure cleanup even if reading fails
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

        except Exception as e:
            self.logger.error(f"gTTS synthesis failed: {e}")
            return b""

    def _get_language_code(self, language: str) -> str:
        """Map language to gTTS language code."""
        mapping = {
            "Hindi-English Mix": "hi",  # gTTS will still handle basic English words in Hindi
            "Hindi Only": "hi",
            "English Only": "en"
        }
        return mapping.get(language, "hi")

    def is_available(self) -> bool:
        """Check if TTS is available."""
        return GTTS_AVAILABLE
