import speech_recognition as sr
import os

def transcribe(audio_file_path: str) -> str:
    """Convert audio file to text using SpeechRecognition."""
    if not audio_file_path or not os.path.exists(audio_file_path):
        return ""
        
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            # Using Google Web Speech API (free, but requires internet)
            # In a prod environment, this would be swapped for Whisper, Deepgram, etc.
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except Exception as e:
        return f"Error processing audio: {e}"
