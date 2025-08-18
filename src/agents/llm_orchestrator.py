import os
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from groq import Groq
from config.settings import GOOGLE_API_KEY, GROQ_API_KEY, LLM_MODELS
from src.utils.logger import log

class LLMOrchestrator:

    def __init__(self):
        self.setup_llm_clients()

    def setup_llm_clients(self):
        """Initialize LLM API clients"""
        try:
            # Load environment variables explicitly
            from dotenv import load_dotenv
            load_dotenv(override=True)

            # Configure Gemini
            google_key = os.getenv("GOOGLE_API_KEY") or GOOGLE_API_KEY
            if google_key:
                genai.configure(api_key=google_key)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                log.info("Gemini client initialized")
            else:
                self.gemini_client = None
                log.warning("Gemini API key not provided")

            # Configure Groq with working models
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key and groq_key.startswith('gsk_'):
                try:
                    # Set environment variable and create client
                    os.environ["GROQ_API_KEY"] = groq_key
                    self.groq_client = Groq(api_key=groq_key)
                    log.info(f"Groq client initialized with key: {groq_key[:10]}...")
                except Exception as e:
                    self.groq_client = None
                    log.error(f"Failed to initialize Groq client: {e}")
            else:
                self.groq_client = None
                log.error(f"GROQ_API_KEY invalid or missing")

        except Exception as e:
            log.error(f"Failed to initialize LLM clients: {e}")
            self.gemini_client = None
            self.groq_client = None

    def generate_response(self, query: str, model_type: str, context: Dict[str, Any] = None, intent: str = None) -> Dict[str, Any]:
        """Generate response using specified model type with PROPER CONTEXT INTEGRATION"""
        try:
            # Updated model mapping with ACTUAL available models
            groq_model_map = {
                "simple": "llama3-8b-8192",
                "complex": "mixtral-8x7b-32768", 
                "orchestration": "gemma2-9b-it",
            }

            # Prepare the prompt with enhanced context
            enhanced_prompt = self._prepare_prompt(query, context, intent)

            # Route to appropriate model based on availability and complexity
            if model_type == "simple" and self.groq_client:
                model_name = groq_model_map["simple"]
                response = self._call_groq(enhanced_prompt, model_name)
                model_used = model_name

            elif model_type == "complex" and self.gemini_client:
                response = self._call_gemini(enhanced_prompt)
                model_used = "gemini-1.5-flash"

            elif model_type == "orchestration" and self.groq_client:
                model_name = groq_model_map["orchestration"]
                response = self._call_groq(enhanced_prompt, model_name)
                model_used = model_name

            else:
                # Fallback to available model
                if self.gemini_client:
                    response = self._call_gemini(enhanced_prompt)
                    model_used = "gemini-1.5-flash"
                elif self.groq_client:
                    model_name = groq_model_map["simple"]
                    response = self._call_groq(enhanced_prompt, model_name)
                    model_used = model_name
                else:
                    return self._fallback_response(query, intent)

            return {
                "response": response,
                "model_used": model_used,
                "success": True,
                "context_used": bool(context)
            }

        except Exception as e:
            log.error(f"LLM generation failed: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing your request right now. Please try again.",
                "model_used": "fallback",
                "success": False,
                "error": str(e)
            }

    def _prepare_prompt(self, query: str, context: Dict[str, Any] = None, intent: str = None) -> str:
        """Prepare enhanced prompt with proper context integration"""
        base_prompt = f"""You are Krishi-Mitra AI, an intelligent agricultural assistant for Indian farmers.
You provide accurate, practical advice based on scientific knowledge and local agricultural practices.
Always respond in a helpful, respectful manner. If the query is in Hindi or contains Hindi words,
feel free to respond with appropriate Hindi terms while maintaining English as the primary language.

User Query: {query}"""

        if context:
            context_parts = []
            
            # Handle weather context
            if context.get("type") == "weather":
                weather_data = context.get("data", {})
                if "current" in weather_data and weather_data["current"].get("success"):
                    current = weather_data["current"]
                    location = context.get("location", "Unknown")
                    context_parts.append(f"Current Weather in {location}:")
                    context_parts.append(f"- Temperature: {current.get('temperature', 'N/A')}°C")
                    context_parts.append(f"- Humidity: {current.get('humidity', 'N/A')}%")
                    context_parts.append(f"- Conditions: {current.get('description', 'N/A')}")
                    context_parts.append(f"- Wind Speed: {current.get('wind_speed', 'N/A')} m/s")
                    
                    # Add forecast if available
                    if "forecast" in weather_data and weather_data["forecast"].get("success"):
                        forecast_data = weather_data["forecast"].get("forecast", [])
                        if forecast_data:
                            context_parts.append("Weather Forecast:")
                            for i, item in enumerate(forecast_data[:5]):  # Next 5 periods
                                context_parts.append(f"- {item.get('datetime', 'Time N/A')}: {item.get('temperature', 'N/A')}°C, {item.get('description', 'N/A')}")

            # Handle market context
            elif context.get("type") == "market":
                market_data = context.get("data", {})
                commodity = context.get("commodity", "Unknown")
                if market_data.get("success"):
                    context_parts.append(f"Market Price Information for {commodity}:")
                    context_parts.append(f"- Average Price: ₹{market_data.get('average_price', 'N/A')}/quintal")
                    context_parts.append(f"- Price Range: ₹{market_data.get('min_price', 'N/A')} - ₹{market_data.get('max_price', 'N/A')}/quintal")
                    context_parts.append(f"- Market Trend: {market_data.get('trend', 'N/A')}")

            # Handle RAG context
            elif context.get("vector_results") and context["vector_results"].get("documents"):
                documents = context["vector_results"]["documents"][0]
                if documents:
                    context_parts.append("Relevant Information:")
                    for doc in documents[:3]:  # Use top 3 results
                        context_parts.append(f"- {doc}")

            if context_parts:
                base_prompt += f"\n\nContext:\n" + "\n".join(context_parts)

        base_prompt += "\n\nPlease provide a helpful and accurate response:"
        return base_prompt

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API"""
        try:
            response = self.gemini_client.generate_content(prompt)
            return response.text
        except Exception as e:
            log.error(f"Gemini API call failed: {e}")
            raise

    def _call_groq(self, prompt: str, model: str) -> str:
        """Call Groq API with WORKING model names"""
        try:
            completion = self.groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            return completion.choices[0].message.content
        except Exception as e:
            log.error(f"Groq API call failed with model {model}: {e}")
            raise

    def _fallback_response(self, query: str, intent: str = None) -> Dict[str, Any]:
        """Provide fallback response when APIs are unavailable"""
        if intent == "weather_query":
            response = "For weather information, please check the IMD website or local weather reports."
        elif intent == "price_query":
            response = "For current market prices, please check AgMarkNet or visit your local mandi."
        elif intent in ["crop_info", "fertilizer_query", "pest_disease"]:
            response = "For crop-related advice, consult your local agricultural extension officer or KVK."
        else:
            response = "I apologize, but I'm currently unable to process your request. Please try again later or consult local agricultural experts."

        return {
            "response": response,
            "model_used": "fallback",
            "success": False,
            "context_used": False
        }

    def test_llm_connections(self) -> Dict[str, bool]:
        """Test all LLM API connections with WORKING models"""
        results = {}

        # Test Gemini
        if self.gemini_client:
            try:
                test_response = self.gemini_client.generate_content("Hello, this is a test.")
                results["gemini"] = bool(test_response.text)
                log.info("Gemini test: SUCCESS")
            except Exception as e:
                results["gemini"] = False
                log.error(f"Gemini test failed: {e}")
        else:
            results["gemini"] = False

        # Test Groq with WORKING model
        if self.groq_client:
            try:
                test_completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello, this is a test."}],
                    model="llama3-8b-8192",  # Working model
                    max_tokens=50,
                    temperature=0.1
                )
                results["groq"] = bool(test_completion.choices[0].message.content)
                log.info("Groq test: SUCCESS")
            except Exception as e:
                results["groq"] = False
                log.error(f"Groq test failed: {e}")
        else:
            results["groq"] = False

        return results
