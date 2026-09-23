import speech_recognition as sr
import os
import logging
import tempfile
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe(audio_file_path: str, language: str = 'en-US') -> str:
    """Convert audio file to text using SpeechRecognition."""
    if not audio_file_path:
        logger.warning("Empty audio file path provided")
        return ""
        
    if not os.path.exists(audio_file_path):
        logger.warning(f"Audio file not found: {audio_file_path}")
        return ""
        
    recognizer = sr.Recognizer()
    temp_wav_path = None
    
    try:
        # Audio format conversion
        ext = os.path.splitext(audio_file_path)[1].lower()
        if ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
            logger.warning(f"Unsupported audio format: {ext}")
            return ""
            
        if ext != '.wav':
            audio = AudioSegment.from_file(audio_file_path)
            fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            audio.export(temp_wav_path, format="wav")
            target_path = temp_wav_path
        else:
            target_path = audio_file_path
            
        with sr.AudioFile(target_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language)
            return text
            
    except sr.UnknownValueError:
        logger.warning("SpeechRecognition could not understand audio")
        return ""
    except sr.RequestError as e:
        logger.warning(f"Could not request results from Google Web Speech API: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Error processing audio: {e}")
        return ""
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_wav_path}: {e}")
