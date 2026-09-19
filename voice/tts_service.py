from gtts import gTTS
import tempfile
import os

def synthesize(text: str) -> str:
    """Convert text to speech and return path to audio file."""
    if not text:
        return None
        
    try:
        # We don't want to read the citations part in voice.
        # Let's clean the text if it has 'Sources:'
        clean_text = text.split("Sources:")[0].strip()
        
        tts = gTTS(text=clean_text, lang='en', slow=False)
        
        # Create a temp file
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        
        tts.save(path)
        return path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None
