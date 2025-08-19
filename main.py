import streamlit as st
import os
import logging
from src.core.cerebrum_engine import CerebrumEngine
from src.speech.stt import SpeechToText
from src.speech.tts import TextToSpeech

# Import the Streamlit audio recorder component
from streamlit_audio_recorder import audio_recorder  # pip install streamlit-audio-recorder

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="🌾 Krishi Mitra AI",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def initialize_system():
    try:
        cerebrum = CerebrumEngine()
        stt = SpeechToText()
        tts = TextToSpeech()
        return cerebrum, stt, tts
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, None

cerebrum_engine, speech_to_text, text_to_speech = initialize_system()
if not cerebrum_engine:
    st.error("⚠️ System initialization failed. Please check your configuration.")
    st.stop()

st.title("🌾 Krishi Mitra AI")
st.subheader("Advanced Agricultural Intelligence Assistant")

with st.sidebar:
    st.header("🔧 System Controls")
    st.subheader("📍 Location Settings")
    city = st.selectbox(
        "Select City:",
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Lucknow", "Jaipur"],
        key="city_selector"
    )
    if st.button("🔍 Check System Status"):
        with st.spinner("Checking system status..."):
            status = cerebrum_engine.get_system_status()
            st.json(status)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

st.header("💬 Chat Interface")
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "audio" in message and message["audio"]:
                    st.audio(message["audio"], format="audio/mp3")

input_method = st.radio(
    "Choose input method:",
    ["Text", "Voice (Record)", "Voice (Upload)"],
    horizontal=True,
    key="input_method"
)
user_input = None

if input_method == "Text":
    user_input = st.chat_input("Ask me about farming, weather, prices, or government schemes...")

elif input_method == "Voice (Record)":
    st.info("Click 'Start' to record your question and then 'Stop' to finish. You will see the waveform and can playback before submitting.")
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=16000,
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_size="2x",
    )
    if audio_bytes and not st.session_state.processing:
        # bytesIO interface compatibility for STT
        import io
        audio_file = io.BytesIO(audio_bytes)
        # Whisper transcribes from file-like object
        with st.spinner("🔄 Transcribing your question..."):
            result = speech_to_text.transcribe_blob(audio_file)
        if result["success"]:
            user_input = result["text"]
            st.success(f"Transcribed: {user_input}")
        else:
            st.error(f"Transcription failed: {result.get('error', 'Unknown error')}")

elif input_method == "Voice (Upload)":
    uploaded_audio = st.file_uploader(
        "Upload an audio file recorded on your device (WAV/MP3/M4A, <2min is best).",
        type=['wav', 'mp3', 'm4a'],
        key="audio_upload"
    )
    if uploaded_audio and not st.session_state.processing:
        with st.spinner("🔄 Processing audio file..."):
            result = speech_to_text.transcribe_file(uploaded_audio)
        if result["success"]:
            user_input = result["text"]
            st.success(f"Transcribed: {user_input}")
        else:
            st.error(f"Transcription failed: {result.get('error', 'Unknown error')}")

# Process user input
if user_input and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("🧠 Processing with Cerebrum Engine..."):
            try:
                enhanced_query = f"{user_input} (Location: {city})"
                response = cerebrum_engine.process_query(
                    query=enhanced_query,
                    session_id="streamlit_session",
                    include_context=True
                )
                bot_text = response["response"]
                st.write(bot_text)
                bot_audio_bytes = None
                if text_to_speech and text_to_speech.is_available():
                    try:
                        bot_audio_bytes = text_to_speech.synthesize(bot_text)
                        if bot_audio_bytes:
                            st.audio(bot_audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.warning(f"Audio generation failed: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_text,
                    "audio": bot_audio_bytes,
                    "metadata": response.get("metadata", {})
                })
                with st.expander("🔍 Processing Details"):
                    st.json(response.get("metadata", {}))
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
    st.session_state.processing = False
    st.rerun()

# Chat management
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🧹 Clear Chat") and not st.session_state.processing:
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("📊 Show Analytics") and st.session_state.messages:
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        st.info(f"Conversation: {user_msgs} questions, {bot_msgs} responses")

st.markdown("---")
st.info("💡 **Tips**: Ask about crop prices, weather forecasts, farming advice, or government schemes. Record or upload your voice for transcription!")
if st.checkbox("Show System Information"):
    st.subheader("🖥️ System Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Engine Status", "Active" if cerebrum_engine else "Inactive")
        st.metric("🎤 Speech Input", "Recording & File Upload")
    with col2:
        st.metric("🔊 Speech Output", "Available" if text_to_speech and text_to_speech.is_available() else "Unavailable")
        st.metric("💬 Total Messages", len(st.session_state.messages))
