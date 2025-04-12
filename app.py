import streamlit as st
import numpy as np
import soundfile as sf
import tempfile
import os
import matplotlib.pyplot as plt
import sounddevice as sd

from src.model.diarization import SpeakerDiarizer
from src.model.transcription import Transcriber
from src.model.summarization import Summarizer
from src.utils.audio_processor import AudioProcessor as MyAudioProcessor
from src.utils.formatter import TimeFormatter

st.set_page_config(page_title="Multi-Speaker Audio Analyzer", layout="wide")
st.title("Multi-Speaker Audio Analyzer")

st.write("Upload or record audio for speaker diarization, transcription, and summarization.")
input_method = st.radio("Choose input method:", ["Upload File", "Record Live Audio"], horizontal=True)

@st.cache_resource
def load_models():
    try:
        diarizer = SpeakerDiarizer(st.secrets["hf_token"])
        diarizer_model = diarizer.load_model()

        transcriber = Transcriber()
        transcriber_model = transcriber.load_model()

        summarizer = Summarizer()
        summarizer_model = summarizer.load_model()

        if not all([diarizer_model, transcriber_model, summarizer_model]):
            raise ValueError("One or more models failed to load")

        return diarizer, transcriber, summarizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def process_audio(audio_input):
    try:
        audio_processor = MyAudioProcessor()

        standardized_path = audio_processor.standardize_audio(audio_input)

        diarizer, transcriber, summarizer = load_models()
        if not all([diarizer, transcriber, summarizer]):
            return None

        with st.spinner("🧠 Identifying speakers..."):
            diarization_result = diarizer.process(standardized_path)

        with st.spinner("✍️ Transcribing audio..."):
            transcription = transcriber.process(standardized_path)

        with st.spinner("📝 Generating summary..."):
            summary = summarizer.process(transcription["text"])

        if os.path.exists(standardized_path):
            os.unlink(standardized_path)

        return {
            "diarization": diarization_result,
            "transcription": transcription,
            "summary": summary[0]["summary_text"]
        }

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def display_results(results):
    tab1, tab2, tab3 = st.tabs(["🗣️Speakers", "📝Transcription", "🧾Summary"])

    with tab1:
        st.write("Speaker Timeline:")
        segments = TimeFormatter.format_speaker_segments(
            results["diarization"], results["transcription"]
        )
        if segments:
            for segment in segments:
                col1, col2, col3 = st.columns([2, 3, 5])
                with col1:
                    speaker_num = int(segment['speaker'].split('_')[1])
                    st.write(f"Speaker {speaker_num}")
                with col2:
                    st.write(f"{TimeFormatter.format_timestamp(segment['start'])} -> {TimeFormatter.format_timestamp(segment['end'])}")
                with col3:
                    st.write(f"\"{segment['text']}\"" if segment['text'] else "(no speech detected)")
                st.markdown("---")
        else:
            st.warning("No speaker segments detected")

    with tab2:
        st.write("Transcription:")
        st.write(results["transcription"].get("text", "No transcription available."))

    with tab3:
        st.write("Summary:")
        st.write(results.get("summary", "No summary available."))

if input_method == "Upload File":
    uploaded_file = st.file_uploader("📁 Choose a file", type=["mp3", "wav"])

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("Analyze Uploaded Audio"):
            results = process_audio(uploaded_file)
            if results:
                display_results(results)

elif input_method == "Record Live Audio":
    st.info("🎙️ Click 'Start Recording' to record audio. Then click 'Analyze Recorded Audio'.")

    duration = st.slider("Select recording duration (seconds):", min_value=5, max_value=150, value=10)

    if "recording_path" not in st.session_state:
        st.session_state.recording_path = None

    if st.button("Start Recording"):
        samplerate = 44100
        try:
            st.write(f"🎙️ Recording for {duration} seconds... Speak now!")
            recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
            sd.wait()
            st.success("✅ Recording completed!")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                sf.write(tmpfile.name, recording, samplerate)
                st.session_state.recording_path = tmpfile.name
                st.audio(tmpfile.name, format="audio/wav")

                fig, ax = plt.subplots()
                ax.plot(recording)
                ax.set_title("Recorded Audio Waveform")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error during recording: {e}")

    if st.session_state.recording_path and st.button("Analyze Recorded Audio"):
        results = process_audio(st.session_state.recording_path)
        if results:
            display_results(results)
            os.unlink(st.session_state.recording_path)
            st.session_state.recording_path = None
