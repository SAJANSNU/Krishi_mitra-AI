from typing import Dict, Any, Optional
from src.agents.intent_classifier import IntentClassifier
from src.agents.llm_orchestrator import LLMOrchestrator
from src.data.rag_engine import RAGEngine
from src.api.weather_api import WeatherAPI
from src.api.market_api import MarketAPI
from src.utils.logger import log

class CerebrumEngine:
    """
    The Cerebrum Engine - Core AI orchestrator that routes queries to appropriate
    models and agents based on intent and complexity analysis WITH PROPER DATA INTEGRATION
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.llm_orchestrator = LLMOrchestrator()
        self.rag_engine = RAGEngine()
        
        # Initialize API connectors
        self.weather_api = WeatherAPI()
        self.market_api = MarketAPI()
        
        self.session_history = {}  # For future session management
        log.info("Cerebrum Engine initialized with all APIs")

    def process_query(self, query: str, session_id: str = "default",
                     include_context: bool = True) -> Dict[str, Any]:
        """
        Process user query through the complete Cerebrum pipeline WITH DATA SOURCES
        """
        try:
            log.info(f"Processing query: {query[:100]}...")

            # Extract location from query if present
            location = self._extract_location(query)
            
            # Step 1: Intent and complexity analysis
            analysis = self.intent_classifier.analyze_query(query)
            log.info(f"Query classified - Intent: {analysis.intent}, Complexity: {analysis.complexity}")

            # Step 2: Get actual data based on intent
            context_data = None
            if analysis.intent == "weather_query":
                context_data = self._get_weather_context(location or "Mumbai")
            elif analysis.intent == "price_query":
                crop = self._extract_crop_from_query(query)
                context_data = self._get_market_context(crop)
            elif include_context:
                context_data = self.rag_engine.search_agricultural_knowledge(
                    query=query,
                    context_type=self._map_intent_to_context_type(analysis.intent)
                )

            log.info(f"Context retrieved: {type(context_data)} with data")

            # Step 3: Route to appropriate LLM based on complexity WITH CONTEXT
            llm_response = self.llm_orchestrator.generate_response(
                query=query,
                model_type=analysis.complexity,
                context=context_data,
                intent=analysis.intent
            )

            # Step 4: Compile complete response
            cerebrum_response = {
                "query": query,
                "response": llm_response["response"],
                "metadata": {
                    "intent": analysis.intent,
                    "complexity": analysis.complexity,
                    "confidence": analysis.confidence,
                    "urgency": analysis.urgency,
                    "domain": analysis.domain,
                    "entities": analysis.entities,
                    "multimodal_required": analysis.requires_multimodal,
                    "model_used": llm_response["model_used"],
                    "context_type": "weather" if analysis.intent == "weather_query" else "market" if analysis.intent == "price_query" else "rag",
                    "processing_success": llm_response["success"]
                },
                "context_used": context_data is not None,
                "session_id": session_id,
                "timestamp": self._get_timestamp()
            }

            # Store in session history
            self._update_session_history(session_id, cerebrum_response)
            log.info(f"Query processed successfully using {llm_response['model_used']}")
            return cerebrum_response

        except Exception as e:
            log.error(f"Cerebrum processing failed: {e}")
            return self._error_response(query, str(e), session_id)

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query"""
        import re
        cities = ["mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "lucknow", "jaipur"]
        query_lower = query.lower()
        for city in cities:
            if city in query_lower:
                return city.title()
        
        # Check for "here" and use default location logic
        if "here" in query_lower:
            return "Mumbai"  # Default location
        return None

    def _extract_crop_from_query(self, query: str) -> Optional[str]:
        """Extract crop/commodity from query"""
        import re
        crops = ["wheat", "rice", "tomato", "onion", "potato", "cotton", "sugarcane", "maize", "soybean"]
        query_lower = query.lower()
        for crop in crops:
            if crop in query_lower:
                return crop.title()
        return None

    def _get_weather_context(self, location: str) -> Dict[str, Any]:
        """Get weather data as context"""
        try:
            weather_data = self.weather_api.get_weather_bundle(location, "IN")
            return {
                "type": "weather",
                "data": weather_data,
                "location": location
            }
        except Exception as e:
            log.error(f"Weather context fetch failed: {e}")
            return {"type": "weather", "error": str(e)}

    def _get_market_context(self, crop: str) -> Dict[str, Any]:
        """Get market price data as context"""
        try:
            if not crop:
                return {"type": "market", "error": "No crop specified"}
            
            market_data = self.market_api.get_commodity_prices(crop)
            return {
                "type": "market",
                "data": market_data,
                "commodity": crop
            }
        except Exception as e:
            log.error(f"Market context fetch failed: {e}")
            return {"type": "market", "error": str(e)}

    def _map_intent_to_context_type(self, intent: str) -> Optional[str]:
        """Map intent to ChromaDB context type for filtering"""
        intent_mapping = {
            "crop_info": "crop_info",
            "price_query": "market_price",
            "weather_query": "weather_data",
            "fertilizer_query": "crop_info",
            "pest_disease": "crop_info",
            "government_scheme": "schemes"
        }
        return intent_mapping.get(intent)

    def _update_session_history(self, session_id: str, response: Dict[str, Any]):
        """Update session history for context awareness"""
        if session_id not in self.session_history:
            self.session_history[session_id] = []
        
        # Keep only last 5 interactions per session
        self.session_history[session_id].append({
            "query": response["query"],
            "intent": response["metadata"]["intent"],
            "timestamp": response["timestamp"]
        })
        
        if len(self.session_history[session_id]) > 5:
            self.session_history[session_id] = self.session_history[session_id][-5:]

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _error_response(self, query: str, error: str, session_id: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "query": query,
            "response": "I apologize, but I encountered an error processing your request. Please try again or rephrase your question.",
            "metadata": {
                "intent": "unknown",
                "complexity": "simple",
                "confidence": 0.0,
                "urgency": "low",
                "domain": "general",
                "entities": [],
                "multimodal_required": False,
                "model_used": "error_handler",
                "context_type": "error",
                "processing_success": False,
                "error": error
            },
            "context_used": False,
            "session_id": session_id,
            "timestamp": self._get_timestamp()
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        try:
            # Test LLM connections
            llm_status = self.llm_orchestrator.test_llm_connections()
            
            # Test API connections
            weather_status = self.weather_api.healthcheck()
            market_status = self.market_api.healthcheck()
            
            # Get knowledge base stats
            knowledge_stats = self.rag_engine.get_knowledge_stats()
            
            # Test RAG retrieval
            rag_test = self.rag_engine.test_rag_retrieval(["What is wheat?"])

            return {
                "cerebrum_engine": "operational",
                "llm_models": llm_status,
                "weather_api": weather_status,
                "market_api": market_status,
                "knowledge_base": knowledge_stats,
                "rag_system": {
                    "operational": any(result["success"] for result in rag_test.values()),
                    "test_results": len(rag_test)
                },
                "active_sessions": len(self.session_history),
                "timestamp": self._get_timestamp()
            }

        except Exception as e:
            log.error(f"System status check failed: {e}")
            return {
                "cerebrum_engine": "error",
                "error": str(e),
                "timestamp": self._get_timestamp()
            }

    def process_batch_queries(self, queries: list, session_id: str = "batch") -> list:
        """Process multiple queries in batch"""
        results = []
        for i, query in enumerate(queries):
            batch_session_id = f"{session_id}_{i}"
            result = self.process_query(query, batch_session_id)
            results.append(result)
        return results

    def get_intent_classification(self, query: str) -> Dict[str, Any]:
        """Get only intent classification without full processing"""
        analysis = self.intent_classifier.analyze_query(query)
        return self.intent_classifier.get_classification_summary(analysis)
