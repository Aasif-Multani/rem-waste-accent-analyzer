import streamlit as st
import yt_dlp
import os
import openai
import json
from speechbrain.inference.classifiers import EncoderClassifier
import logging

# --- Setup and Configuration ---
st.set_page_config(
    page_title="Accent Analyzer",
    page_icon="🗣️",
    layout="centered"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_models():
    """
    Loads the SpeechBrain model and initializes the OpenAI client.
    """
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="pretrained_models/lang_id"
        )
        logger.info("SpeechBrain model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading SpeechBrain model: {e}")
        st.error(f"Could not load the accent classification model. Error: {e}")
        return None, None

    try:
        # Configure OpenAI client using secrets from Streamlit Cloud
        openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        logger.info("OpenAI client configured successfully.")
    except (KeyError, AttributeError):
        logger.error("OpenAI API key not found in secrets.")
        st.error("OpenAI API key not configured. Please add it in the app's settings on Streamlit Cloud.")
        openai_client = None

    return classifier, openai_client

# --- Core Functions ---
def download_and_extract_audio(url):
    """
    Downloads video and extracts audio as an MP3 file.
    """
    output_audio_path = 'audio.mp3'
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)

    # REMOVED the hardcoded 'ffmpeg_location' for deployment
    ydl_opts = {
        'format': 'bestaudio/best',
        # 'ffmpeg_location': 'ffmpeg/bin', # REMOVED for deployment
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'outtmpl': 'audio',
        'noplaylist': True,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info(f"Audio extracted to {output_audio_path}")
        return output_audio_path
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        st.error(f"Failed to process video URL. It might be private or an unsupported format. Error: {e}")
        return None

def analyze_accent_with_ai(audio_path, accent_classifier, openai_client):
    """
    The main analysis pipeline.
    """
    try:
        # NOTE: This is the single-stage analysis from your final working code.
        # It assumes the user will input English audio.
        logger.info("Analyzing speech...")
        with open(audio_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            ).text
        logger.info("Transcription successful.")

        signal = accent_classifier.load_audio(audio_path)
        prediction = accent_classifier.classify_batch(signal)
        
        accent_label = prediction[3][0] # e.g., 'en-US: English (United States)'
        model_confidence = f"{prediction[1].exp().item() * 100:.1f}%"
        logger.info(f"SpeechBrain prediction: {accent_label} with {model_confidence} confidence.")

        prompt = f"""
        You are an expert hiring assistant. Analyze a candidate's spoken English based on AI-generated data.
        **Input Data:**
        - **Transcript:** "{transcript}"
        - **AI Accent Prediction:** "{accent_label}" (Model Confidence: {model_confidence})
        **Your Task:**
        Provide a final analysis in a valid JSON object format ONLY. The JSON must contain three keys:
        1. "accent_classification": Interpret the technical label (e.g., 'en-US: English (United States)') into a user-friendly classification (e.g., "American (US)", "British (UK)"). If the model detects a non-English language, state what it is.
        2. "english_confidence_score": An integer (0-100) representing confidence in the speaker's clarity and fluency for a professional setting. If a non-English language is detected, this score should be 0.
        3. "explanation": A one-sentence summary explaining your reasoning.
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        results = json.loads(response.choices[0].message.content)
        logger.info("GPT-4o analysis successful.")
        return results

    except Exception as e:
        logger.error(f"Error during AI analysis: {e}")
        st.error(f"An error occurred during the analysis pipeline. Error: {e}")
        return None

# --- Streamlit User Interface ---
st.title("🗣️ REM Waste Accent Analyzer")
st.markdown(
    "This tool analyzes a speaker's English accent from a video. "
    "Paste a public URL (e.g., Loom, MP4) to begin."
)

accent_classifier, openai_client = load_models()

if accent_classifier and openai_client:
    video_url = st.text_input("Enter a public video URL", placeholder="https://www.loom.com/share/...")

    if st.button("Analyze Accent", type="primary", use_container_width=True):
        if video_url:
            with st.spinner("Step 1/3: Preparing audio..."):
                audio_file = download_and_extract_audio(video_url)

            if audio_file:
                with st.spinner("Step 2/3: Analyzing speech... (This may take a moment for the first run)"):
                    results = analyze_accent_with_ai(audio_file, accent_classifier, openai_client)
                    os.remove(audio_file)

                st.spinner("Step 3/3: Finalizing results...")
                if results:
                    st.success("✅ Analysis Complete!")
                    st.divider()
                    col1, col2 = st.columns(2)
                    col1.metric("Accent Classification", results.get("accent_classification", "N/A"))
                    col2.metric("Confidence in English Score", f"{results.get('english_confidence_score', 'N/A')}%")
                    st.info(f"**Summary:** {results.get('explanation', 'No explanation provided.')}")
        else:
            st.warning("Please enter a video URL.")