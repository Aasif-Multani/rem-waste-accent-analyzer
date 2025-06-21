# REM Waste - Accent Analyzer

This project is a submission for the REM Waste practical interview challenge. It is a web application that takes a public video URL, analyzes the speaker's English accent, and provides a classification, a confidence score, and a summary.

**Live Demo:** https://aasif-multani-rem-waste-accent-analyzer-app-yf8elf.streamlit.app/](https://rem-waste-accent-analyzer.streamlit.app/)

## Features

-   **Accepts Public Video URLs**: Works with a wide variety of sources, including Loom and direct MP4 links, thanks to `yt-dlp`.
-   **Audio Extraction**: Automatically extracts the audio from the video into a usable format.
-   **Accent Analysis (Hybrid AI Approach)**:
    1.  **Transcription**: Uses **OpenAI's Whisper** to get a text transcript, implicitly verifying the language is English.
    2.  **Classification**: Uses **SpeechBrain's** specialized accent classification model for a technical, data-driven analysis.
    3.  **Synthesis & Scoring**: Uses **OpenAI's GPT-4o** to interpret the technical results, provide a human-readable classification, generate a confidence score, and write a concise explanation.
-   **Clear Output**: Displays the accent classification, a confidence score (0-100%), and a short summary.

## Technical Approach

The core of this solution is a **hybrid AI model approach**. Instead of relying on a single tool, it leverages the best tool for each part of the task:

-   **SpeechBrain**: A specialized, open-source model trained specifically for accent classification. This provides an objective, foundational analysis.
-   **OpenAI Whisper**: State-of-the-art for transcription, providing crucial text context.
-   **OpenAI GPT-4o**: A powerful Large Language Model used to synthesize the outputs from the other models into a final, polished result. It excels at the nuanced task of generating a confidence score and a human-friendly explanation, which a purely technical model cannot do.

This approach is resourceful and robust, directly addressing the challenge's request for an "intelligent tool."

## How to Run Locally

### Prerequisites

-   Python 3.9+
-   `pip` for package installation
-   `ffmpeg` installed on your system. (On macOS: `brew install ffmpeg`; on Debian/Ubuntu: `sudo apt-get install ffmpeg`)

### Setup Instructions

1.  **Create the Project Files**:
    Create a directory and save the files provided (`app.py`, `requirements.txt`, `packages.txt`) into it.

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key**:
    You need an OpenAI API key for this project to work.
    -   Create a directory named `.streamlit` inside your project folder.
    -   Inside `.streamlit`, create a file named `secrets.toml`.
    -   Add your API key to this file like so:
        ```toml
        # .streamlit/secrets.toml
        OPENAI_API_KEY = "sk-your-secret-key-here"
        ```

5.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```
    Your browser should open with the application running locally.