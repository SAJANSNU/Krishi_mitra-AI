import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "579b464db66ec23bdd000001b7da63770362465654d73aa5ac2abd1e")

# LLM Model Configuration
LLM_MODELS = {
    "simple": "meta-llama/llama-guard-4-12b",
    "orchestration": "moonshotai/kimi-k2-instruct", 
    "complex": "models/gemini-2.5-flash"
}

# API Endpoints
MARKET_API_ENDPOINT = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
WEATHER_API_BASE = "http://api.openweathermap.org/data/2.5"

# Speech Settings
SPEECH_RATE = 16000
SPEECH_CHUNK = 4000
DEFAULT_RECORDING_DURATION = 20

# RAG Settings
PINECONE_INDEX_NAME = "farmer-semantic-index"
EMBEDDING_MODEL = "models/embedding-001"
