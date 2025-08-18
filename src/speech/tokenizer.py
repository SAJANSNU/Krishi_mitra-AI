"""
Advanced tokenizer for Hindi-English code-switching
Uses SentencePiece for robust multilingual tokenization
"""

import os
import re
import logging
from typing import List, Dict, Any
from pathlib import Path

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

from src.utils.logger import log

class CodeSwitchTokenizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sp_model = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize SentencePiece tokenizer."""
        if not SENTENCEPIECE_AVAILABLE:
            log.warning("SentencePiece not available, using simple tokenization")
            return

        try:
            model_path = Path("models/sentencepiece.model")
            if model_path.exists():
                self.sp_model = smp.SentencePieceProcessor(model_file=str(model_path))
                log.info("SentencePiece model loaded successfully")
            else:
                log.warning("SentencePiece model not found, will use simple tokenization")
                self._create_simple_model()
        except Exception as e:
            log.error(f"Failed to load SentencePiece model: {e}")

    def _create_simple_model(self):
        """Create a simple tokenization model if SentencePiece model is not available."""
        # This is a fallback - in production, you should train a proper SentencePiece model
        # on Hindi-English code-switched data
        pass

    def tokenize(self, text: str) -> List[str]:
        """Tokenize code-switched text."""
        try:
            if self.sp_model:
                return self.sp_model.encode(text, out_type=str)
            else:
                # Simple fallback tokenization
                return self._simple_tokenize(text)
        except Exception as e:
            log.error(f"Tokenization failed: {e}")
            return self._simple_tokenize(text)

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize tokens back to text."""
        try:
            if self.sp_model:
                return self.sp_model.decode(tokens)
            else:
                # Simple fallback detokenization
                return self._simple_detokenize(tokens)
        except Exception as e:
            log.error(f"Detokenization failed: {e}")
            return self._simple_detokenize(tokens)

    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization fallback."""
        # Basic word-level tokenization with language detection
        tokens = re.findall(r'\S+', text)
        
        # Further split mixed tokens if needed
        processed_tokens = []
        for token in tokens:
            # Check if token contains both Hindi and English characters
            has_hindi = bool(re.search(r'[\u0900-\u097F]', token))
            has_english = bool(re.search(r'[a-zA-Z]', token))
            
            if has_hindi and has_english:
                # Split mixed tokens more carefully
                subtokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+|\d+|[^\w\s]', token)
                processed_tokens.extend(subtokens)
            else:
                processed_tokens.append(token)
        
        return processed_tokens

    def _simple_detokenize(self, tokens: List[str]) -> str:
        """Simple detokenization fallback."""
        if not tokens:
            return ""
        
        # Join with spaces, but handle punctuation
        result = ""
        for i, token in enumerate(tokens):
            if i == 0:
                result = token
            elif token in ".,!?;:":
                result += token
            elif tokens[i-1] in "\"'(":
                result += token
            else:
                result += " " + token
        
        return result

    def analyze_code_switching(self, text: str) -> Dict[str, Any]:
        """Analyze code-switching patterns in text."""
        # Count Hindi and English characters
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'\S', text))

        if total_chars == 0:
            return {
                "is_code_switched": False,
                "hindi_ratio": 0.0,
                "english_ratio": 0.0,
                "dominant_language": "unknown"
            }

        hindi_ratio = hindi_chars / total_chars
        english_ratio = english_chars / total_chars

        # Determine if it's code-switched (both languages present significantly)
        is_code_switched = hindi_ratio > 0.1 and english_ratio > 0.1

        # Determine dominant language
        if hindi_ratio > english_ratio:
            dominant_language = "hindi"
        elif english_ratio > hindi_ratio:
            dominant_language = "english"
        else:
            dominant_language = "mixed"

        return {
            "is_code_switched": is_code_switched,
            "hindi_ratio": hindi_ratio,
            "english_ratio": english_ratio,
            "dominant_language": dominant_language,
            "switching_points": self._find_switching_points(text)
        }

    def _find_switching_points(self, text: str) -> List[int]:
        """Find positions where language switching occurs."""
        switching_points = []
        words = text.split()
        prev_lang = None
        pos = 0

        for word in words:
            has_hindi = bool(re.search(r'[\u0900-\u097F]', word))
            has_english = bool(re.search(r'[a-zA-Z]', word))

            if has_hindi and not has_english:
                current_lang = "hindi"
            elif has_english and not has_hindi:
                current_lang = "english"
            else:
                current_lang = "mixed"

            if prev_lang and prev_lang != current_lang and current_lang != "mixed":
                switching_points.append(pos)

            prev_lang = current_lang
            pos += len(word) + 1  # +1 for space

        return switching_points

    def get_language_stats(self, text: str) -> Dict[str, Any]:
        """Get detailed language statistics for the text."""
        analysis = self.analyze_code_switching(text)
        tokens = self.tokenize(text)
        
        return {
            "total_tokens": len(tokens),
            "total_characters": len(text),
            "language_analysis": analysis,
            "tokenization_method": "sentencepiece" if self.sp_model else "simple"
        }

    def is_available(self) -> bool:
        """Check if tokenizer is available."""
        return True  # Simple tokenizer is always available as fallback
