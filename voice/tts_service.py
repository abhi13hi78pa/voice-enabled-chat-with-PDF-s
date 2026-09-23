from gtts import gTTS
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List to track temp files for cleanup
_temp_files = []

def synthesize(text: str, language: str = 'en', slow: bool = False) -> str:
    """Convert text to speech and return path to audio file."""
    if not text:
        logger.warning("Empty text provided for TTS")
        return None
        
    try:
        # Clean the text if it has 'Sources:'
        clean_text = text.split("Sources:")[0].strip()
        
        if not clean_text:
            logger.warning("Text became empty after removing citations")
            return None
            
        tts = gTTS(text=clean_text, lang=language, slow=slow)
        
        # Create a temp file
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        
        tts.save(path)
        _temp_files.append(path)
        return path
        
    except Exception as e:
        logger.warning(f"TTS Error during synthesis: {e}")
        return None

def cleanup_temp_files():
    """Delete generated audio files."""
    for path in _temp_files[:]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {path}: {e}")
        _temp_files.remove(path)
