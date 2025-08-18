import streamlit as st
import os
from src.core.cerebrum_engine import CerebrumEngine
from src.speech.stt import SpeechToText
from src.speech.tts import TextToSpeech
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="🌾 Krishi Mitra AI",
    page_icon="🌾",
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_system():
    """Initialize the Cerebrum Engine and speech components"""
    try:
        cerebrum = CerebrumEngine()
        stt = SpeechToText()
        tts = TextToSpeech()
        return cerebrum, stt, tts
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, None

# Load system
cerebrum_engine, speech_to_text, text_to_speech = initialize_system()

if not cerebrum_engine:
    st.error("⚠️ System initialization failed. Please check your configuration.")
    st.stop()

# Title and description
st.title("🌾 Krishi Mitra AI")
st.subheader("Advanced Agricultural Intelligence Assistant")

# Sidebar for system status and location
with st.sidebar:
    st.header("🔧 System Controls")
    
    # Location input
    st.subheader("📍 Location Settings")
    city = st.selectbox(
        "Select City:", 
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Lucknow", "Jaipur"],
        key="city_selector"
    )
    
    # System status
    if st.button("🔍 Check System Status"):
        with st.spinner("Checking system status..."):
            status = cerebrum_engine.get_system_status()
            st.json(status)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "processing" not in st.session_state:
    st.session_state.processing = False

# Chat interface
st.header("💬 Chat Interface")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                # Add audio playback if available
                if "audio" in message and message["audio"]:
                    st.audio(message["audio"], format="audio/mp3")

# Input methods
input_method = st.radio(
    "Choose input method:",
    ["Text", "Voice"],
    horizontal=True,
    key="input_method"
)

user_input = None

if input_method == "Text":
    # Text input
    user_input = st.chat_input("Ask me about farming, weather, prices, or government schemes...")
    
elif input_method == "Voice":
    # Voice input
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🎤 Start Recording", disabled=st.session_state.processing):
            if speech_to_text and speech_to_text.is_available():
                with st.spinner("🎙️ Listening... Speak now!"):
                    result = speech_to_text.listen_from_microphone(duration=5)
                    
                if result["success"]:
                    user_input = result["text"]
                    st.success(f"Heard: {user_input}")
                else:
                    st.error(f"Speech recognition failed: {result.get('error', 'Unknown error')}")
            else:
                st.error("Speech recognition not available")
    
    with col2:
        # File upload for audio
        uploaded_audio = st.file_uploader(
            "Or upload an audio file:",
            type=['wav', 'mp3', 'm4a'],
            key="audio_upload"
        )
        
        if uploaded_audio and not st.session_state.processing:
            if speech_to_text:
                with st.spinner("🔄 Processing audio file..."):
                    result = speech_to_text.transcribe_file(uploaded_audio)
                    
                if result["success"]:
                    user_input = result["text"]
                    st.success(f"Transcribed: {user_input}")
                else:
                    st.error(f"Transcription failed: {result.get('error', 'Unknown error')}")

# Process user input
if user_input and not st.session_state.processing:
    # Prevent multiple simultaneous processing
    st.session_state.processing = True
    
    # Add user message to history
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input
    })
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Process query with Cerebrum Engine
    with st.chat_message("assistant"):
        with st.spinner("🧠 Processing with Cerebrum Engine..."):
            try:
                # Create enhanced query with location context
                enhanced_query = f"{user_input} (Location: {city})"
                
                # Process through Cerebrum Engine
                response = cerebrum_engine.process_query(
                    query=enhanced_query,
                    session_id="streamlit_session",
                    include_context=True
                )
                
                bot_text = response["response"]
                
                # Display text response
                st.write(bot_text)
                
                # Generate audio response if TTS is available
                bot_audio_bytes = None
                if text_to_speech and text_to_speech.is_available():
                    try:
                        bot_audio_bytes = text_to_speech.synthesize(bot_text)
                        if bot_audio_bytes:
                            st.audio(bot_audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.warning(f"Audio generation failed: {e}")
                
                # Add bot response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_text,
                    "audio": bot_audio_bytes,
                    "metadata": response.get("metadata", {})
                })
                
                # Display processing metadata in expander
                with st.expander("🔍 Processing Details"):
                    st.json(response.get("metadata", {}))
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
    
    # Reset processing flag
    st.session_state.processing = False
    
    # Use st.rerun() instead of deprecated st.experimental_rerun()
    st.rerun()

# Chat management
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🧹 Clear Chat") and not st.session_state.processing:
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("📊 Show Analytics") and st.session_state.messages:
        # Show conversation analytics
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        st.info(f"Conversation: {user_msgs} questions, {bot_msgs} responses")

# Footer information
st.markdown("---")
st.info("💡 **Tips**: Ask about crop prices, weather forecasts, farming advice, or government schemes. Use voice input for hands-free interaction!")

# Display system information
if st.checkbox("Show System Information"):
    st.subheader("🖥️ System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🧠 Engine Status", "Active" if cerebrum_engine else "Inactive")
        st.metric("🎤 Speech Input", "Available" if speech_to_text and speech_to_text.is_available() else "Unavailable")
    
    with col2:
        st.metric("🔊 Speech Output", "Available" if text_to_speech and text_to_speech.is_available() else "Unavailable")
        st.metric("💬 Total Messages", len(st.session_state.messages))
