import pytest
from unittest.mock import MagicMock, patch, mock_open
import app

@pytest.fixture
def mock_openai_client():
    return MagicMock()

@pytest.fixture
def mock_accent_classifier():
    return MagicMock()

@patch('app.open', new_callable=mock_open, read_data=b"fake audio data")
def test_english_language_happy_path(mock_file_open, mock_accent_classifier, mock_openai_client):
    """
    Tests the full pipeline for a standard English-speaking candidate.
    Mocks the file 'open' call and correctly mocks the chained calls on the
    SpeechBrain prediction object.
    """
    # --- Arrange: Configure the mock responses ---

    # 1. Mock Whisper's response
    mock_transcript_obj = MagicMock()
    mock_transcript_obj.text = "This is a test of the system."
    mock_openai_client.audio.transcriptions.create.return_value = mock_transcript_obj

    # 2. Mock SpeechBrain's response (THE CORRECTED PART)
    # Create a mock for the score tensor at prediction[1]
    mock_score_tensor = MagicMock()
    # Configure the chained calls: .exp().item() should return a float
    mock_score_tensor.exp.return_value.item.return_value = 0.95
    # Now, construct the full prediction tuple with the properly configured mock
    mock_prediction = (MagicMock(), mock_score_tensor, None, ["en-US: English (United States)"])
    mock_accent_classifier.classify_batch.return_value = mock_prediction

    # 3. Mock GPT-4o's final JSON synthesis
    mock_gpt_response = MagicMock()
    final_json = '{"accent_classification": "American (US)", "english_confidence_score": 95, "explanation": "Clear American English detected."}'
    mock_gpt_response.choices[0].message.content = final_json
    mock_openai_client.chat.completions.create.return_value = mock_gpt_response
    
    # --- Act: Run the function under test ---
    results = app.analyze_accent_with_ai("fake/path.mp3", mock_accent_classifier, mock_openai_client)

    # --- Assert: Verify the behavior and output ---
    mock_file_open.assert_called_once_with("fake/path.mp3", "rb")
    mock_accent_classifier.classify_batch.assert_called_once()
    mock_openai_client.chat.completions.create.assert_called_once()
    assert results["accent_classification"] == "American (US)"
    assert results["english_confidence_score"] == 95


@patch('app.yt_dlp.YoutubeDL')
def test_download_and_extract_audio(mock_youtube_dl):
    """
    Tests the audio downloader utility.
    Verifies that it calls the yt-dlp library with the correct parameters.
    """
    # --- Arrange ---
    mock_ydl_instance = MagicMock()
    mock_youtube_dl.return_value.__enter__.return_value = mock_ydl_instance
    test_url = "https://example.com/video.mp4"

    # --- Act ---
    result_path = app.download_and_extract_audio(test_url)

    # --- Assert ---
    mock_youtube_dl.assert_called_once()
    assert mock_youtube_dl.call_args[0][0]['postprocessors'][0]['key'] == 'FFmpegExtractAudio'
    mock_ydl_instance.download.assert_called_once_with([test_url])
    assert result_path == "audio.mp3"