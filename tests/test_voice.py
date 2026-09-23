import os
import pytest
from unittest.mock import patch, MagicMock
from voice.stt_service import transcribe
from voice.tts_service import synthesize, cleanup_temp_files, _temp_files

# --- STT Tests ---

@patch('voice.stt_service.os.path.exists')
@patch('voice.stt_service.AudioSegment')
@patch('voice.stt_service.sr.Recognizer')
@patch('voice.stt_service.sr.AudioFile')
def test_transcribe_success(mock_audio_file, mock_recognizer, mock_audio_segment, mock_exists):
    mock_exists.return_value = True
    
    # Mocking recognizer instance
    mock_rec_instance = MagicMock()
    mock_rec_instance.recognize_google.return_value = "hello world"
    mock_recognizer.return_value = mock_rec_instance

    # For a .wav file, AudioSegment isn't used. Let's use a .wav
    result = transcribe("test.wav", language="en-US")
    assert result == "hello world"
    mock_rec_instance.recognize_google.assert_called_once()

@patch('voice.stt_service.os.path.exists')
def test_transcribe_missing_file(mock_exists):
    mock_exists.return_value = False
    assert transcribe("missing.wav") == ""

def test_transcribe_empty_file_path():
    assert transcribe("") == ""
    assert transcribe(None) == ""

@patch('voice.stt_service.os.path.exists')
@patch('voice.stt_service.sr.Recognizer')
@patch('voice.stt_service.sr.AudioFile')
def test_transcribe_network_error(mock_audio_file, mock_recognizer, mock_exists):
    import speech_recognition as sr
    mock_exists.return_value = True
    
    mock_rec_instance = MagicMock()
    mock_rec_instance.recognize_google.side_effect = sr.RequestError("Network error")
    mock_recognizer.return_value = mock_rec_instance

    assert transcribe("test.wav") == ""

@patch('voice.stt_service.os.path.exists')
@patch('voice.stt_service.sr.Recognizer')
@patch('voice.stt_service.sr.AudioFile')
def test_transcribe_unrecognized_speech(mock_audio_file, mock_recognizer, mock_exists):
    import speech_recognition as sr
    mock_exists.return_value = True
    
    mock_rec_instance = MagicMock()
    mock_rec_instance.recognize_google.side_effect = sr.UnknownValueError()
    mock_recognizer.return_value = mock_rec_instance

    assert transcribe("test.wav") == ""

# --- TTS Tests ---

@patch('voice.tts_service.gTTS')
@patch('voice.tts_service.tempfile.mkstemp')
@patch('voice.tts_service.os.close')
def test_synthesize_success(mock_close, mock_mkstemp, mock_gtts):
    mock_mkstemp.return_value = (1, "temp/path.mp3")
    mock_gtts_instance = MagicMock()
    mock_gtts.return_value = mock_gtts_instance
    
    result = synthesize("hello world", language="en", slow=False)
    
    assert result == "temp/path.mp3"
    mock_gtts.assert_called_once_with(text="hello world", lang="en", slow=False)
    mock_gtts_instance.save.assert_called_once_with("temp/path.mp3")

def test_synthesize_empty_text():
    assert synthesize("") is None
    assert synthesize(None) is None

@patch('voice.tts_service.gTTS')
def test_synthesize_failure(mock_gtts):
    mock_gtts.side_effect = Exception("Synthesis failed")
    assert synthesize("hello world") is None

@patch('voice.tts_service.gTTS')
@patch('voice.tts_service.tempfile.mkstemp')
@patch('voice.tts_service.os.close')
def test_synthesize_citation_stripping(mock_close, mock_mkstemp, mock_gtts):
    mock_mkstemp.return_value = (1, "temp/path.mp3")
    
    synthesize("hello world Sources: doc1.pdf")
    
    mock_gtts.assert_called_once_with(text="hello world", lang="en", slow=False)

@patch('voice.tts_service.os.path.exists')
@patch('voice.tts_service.os.remove')
def test_cleanup_temp_files(mock_remove, mock_exists):
    # Setup some fake files in tracking list
    _temp_files.clear()
    _temp_files.append("fake/path1.mp3")
    _temp_files.append("fake/path2.mp3")
    
    mock_exists.return_value = True
    
    cleanup_temp_files()
    
    assert len(_temp_files) == 0
    assert mock_remove.call_count == 2
