import streamlit as st
import os
import io
import logging
from src.core.cerebrum_engine import CerebrumEngine
from src.speech.stt import SpeechToText
from src.speech.tts import TextToSpeech

# Import the audio recorder component (same as ledger.py)
# pip install streamlit-audiorec
from st_audiorec import st_audiorec
import soundfile as sf
import numpy as np

ENABLE_WHISPER = True  # Set True to enable server-side Whisper transcription

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
    
    # Navigation buttons
    st.subheader("🚀 Quick Access")
    
    # Use st.link_button for external links
    st.link_button(
        "🎙️ Voice Ledger",
        "https://voice-powered-digital-ledger-krishi-mitra-ai.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "🚨 SOS Alert System", 
        "https://emotion-detection-sos-allert-krishi-mitra-ai.streamlit.app/",
        use_container_width=True
    )
    
    st.divider()
    
    st.subheader("📍 Location Settings")
    city = st.selectbox(
        "Select City:",
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Lucknow", "Jaipur"],
        key="city_selector"
    )
    
    # Add error handling for system status
    if st.button("🔍 Check System Status"):
        with st.spinner("Checking system status..."):
            try:
                # Check if cerebrum_engine exists
                if 'cerebrum_engine' in globals():
                    status = cerebrum_engine.get_system_status()
                    st.json(status)
                else:
                    st.warning("cerebrum_engine not available")
            except Exception as e:
                st.error(f"Error checking system status: {e}")

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
    # Using the same logic as ledger.py
    st.info("Click the mic button, speak for 5–10s, then stop. The recorder returns WAV bytes directly.")
    
    wav_audio_bytes = st_audiorec()  # returns WAV bytes or None
    
    if wav_audio_bytes and not st.session_state.processing:
        if st.button("🔄 Process Recording"):
            try:
                # Validate/normalize audio using soundfile (same as ledger.py)
                data, sr = sf.read(io.BytesIO(wav_audio_bytes), dtype="float32", always_2d=False)
                if isinstance(data, np.ndarray) and data.ndim > 1:
                    data = data.mean(axis=1)  # mono
                
                # Re-encode to a clean WAV buffer
                buf = io.BytesIO()
                sf.write(buf, data, sr, subtype="PCM_16", format="WAV")
                wav_clean_bytes = buf.getvalue()
                st.success(f"Audio captured (sr={sr} Hz).")
                
                transcript = ""
                if ENABLE_WHISPER:
                    with st.spinner("🔄 Transcribing with Whisper..."):
                        # Use the same transcription logic as ledger.py
                        result = speech_to_text.transcribe_blob(io.BytesIO(wav_clean_bytes))
                        if result["success"]:
                            transcript = result["text"]
                
                if transcript:
                    user_input = transcript
                    st.success(f"Transcribed: {user_input}")
                else:
                    st.error("Transcription failed or returned empty text.")
                    
            except Exception as e:
                st.error(f"Audio processing failed: {e}")

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

# Process user input (rest remains the same)
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

# Chat management (rest remains the same)
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


