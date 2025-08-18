import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from src.utils.logger import log

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    log.warning("spaCy not available, using simple classification")

@dataclass
class QueryAnalysis:
    intent: str
    complexity: str
    confidence: float
    entities: List[str]
    urgency: str
    domain: str
    requires_multimodal: bool

class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            "weather_query": [
                r"weather", r"temperature", r"rain", r"forecast", r"climate",
                r"मौसम", r"बारिश", r"तापमान", r"today", r"tomorrow"
            ],
            "price_query": [
                r"price", r"cost", r"rate", r"market", r"sell", r"buy", r"selling",
                r"कीमत", r"भाव", r"दाम", r"बाज़ार", r"mandi"
            ],
            "crop_info": [
                r"crop", r"farming", r"cultivation", r"growing", r"harvest", r"plant",
                r"फसल", r"खेती", r"उगाना", r"grow", r"sow"
            ],
            "fertilizer_query": [
                r"fertilizer", r"nutrient", r"manure", r"compost", r"urea",
                r"खाद", r"उर्वरक"
            ],
            "pest_disease": [
                r"pest", r"disease", r"insect", r"bug", r"infection", r"yellow", r"spots",
                r"कीट", r"बीमारी", r"रोग"
            ],
            "government_scheme": [
                r"scheme", r"subsidy", r"loan", r"insurance", r"government", r"pm", r"kisan",
                r"योजना", r"सब्सिडी", r"कर्ज", r"help", r"assistance"
            ],
            "emergency": [
                r"urgent", r"emergency", r"help", r"crisis", r"problem", r"dying",
                r"तुरंत", r"मदद", r"समस्या"
            ]
        }

        self.complexity_indicators = {
            "simple": [
                r"what is", r"current", r"today", r"now", r"price of",
                r"क्या है", r"आज", r"अभी"
            ],
            "complex": [
                r"how to", r"best way", r"recommend", r"suggest", r"strategy", r"should i",
                r"कैसे", r"सबसे अच्छा", r"सुझाव"
            ],
            "orchestration": [
                r"multiple", r"compare", r"analyze", r"predict", r"forecast", r"and",
                r"तुलना", r"विश्लेषण", r"भविष्यवाणी"
            ]
        }

        self.urgency_patterns = {
            "high": [
                r"urgent", r"emergency", r"immediately", r"critical", r"dying", r"help",
                r"तुरंत", r"आपातकाल", r"मर रहा"
            ],
            "medium": [
                r"soon", r"quickly", r"fast", r"within.*day",
                r"जल्दी", r"तेज़"
            ],
            "low": [
                r"when", r"sometime", r"planning", r"future",
                r"कब", r"योजना"
            ]
        }

        self.domain_patterns = {
            "agriculture": [r"crop", r"farm", r"soil", r"seed", r"harvest"],
            "weather": [r"weather", r"rain", r"temperature", r"climate"],
            "market": [r"price", r"market", r"sell", r"buy", r"trade"],
            "finance": [r"loan", r"insurance", r"subsidy", r"cost"],
            "health": [r"disease", r"pest", r"treatment", r"medicine"]
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis with dynamic processing."""
        try:
            query_lower = query.lower().strip()
            
            # Extract entities
            entities = self._extract_entities(query)
            
            # Classify intent
            intent = self._classify_intent(query_lower)
            
            # Assess complexity
            complexity = self._assess_complexity(query_lower)
            
            # Determine urgency
            urgency = self._determine_urgency(query_lower)
            
            # Identify domain
            domain = self._identify_domain(query_lower)
            
            # Check for multimodal requirements
            requires_multimodal = self._check_multimodal_need(query_lower)
            
            # Calculate confidence
            confidence = self._calculate_confidence(query_lower, intent, complexity)

            return QueryAnalysis(
                intent=intent,
                complexity=complexity,
                confidence=confidence,
                entities=entities,
                urgency=urgency,
                domain=domain,
                requires_multimodal=requires_multimodal
            )

        except Exception as e:
            log.error(f"Query analysis failed: {e}")
            return QueryAnalysis(
                intent="general",
                complexity="simple",
                confidence=0.5,
                entities=[],
                urgency="low",
                domain="agriculture",
                requires_multimodal=False
            )

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query."""
        if SPACY_AVAILABLE:
            try:
                doc = nlp(query)
                return [ent.text for ent in doc.ents]
            except:
                pass
        
        # Fallback entity extraction
        entities = []
        common_crops = ["wheat", "rice", "tomato", "onion", "potato", "cotton", "sugarcane"]
        query_lower = query.lower()
        
        for crop in common_crops:
            if crop in query_lower:
                entities.append(crop.title())
        
        return entities

    def _classify_intent(self, query: str) -> str:
        """Classify query intent dynamically."""
        max_score = 0
        best_intent = "general"

        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query, re.IGNORECASE))
            if score > max_score:
                max_score = score
                best_intent = intent

        return best_intent

    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity dynamically."""
        complexity_scores = {}
        
        for complexity, patterns in self.complexity_indicators.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query, re.IGNORECASE))
            complexity_scores[complexity] = score

        # Factor in query characteristics
        word_count = len(query.split())
        question_marks = query.count('?')
        conjunctions = len(re.findall(r'\b(and|or|but|if|when|how)\b', query, re.IGNORECASE))

        # Dynamic complexity assessment
        if word_count > 20 or question_marks > 1 or conjunctions > 2:
            base_complexity = "orchestration"
        elif word_count > 10 or complexity_scores.get("complex", 0) > 0:
            base_complexity = "complex"
        else:
            base_complexity = "simple"

        # Override with pattern matching if strong signal
        if max(complexity_scores.values()) > 0:
            pattern_complexity = max(complexity_scores, key=complexity_scores.get)
            return pattern_complexity

        return base_complexity

    def _determine_urgency(self, query: str) -> str:
        """Determine query urgency."""
        for urgency, patterns in self.urgency_patterns.items():
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
                return urgency
        return "low"

    def _identify_domain(self, query: str) -> str:
        """Identify primary domain."""
        domain_scores = {}
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query, re.IGNORECASE))
            domain_scores[domain] = score

        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return "agriculture"

    def _check_multimodal_need(self, query: str) -> bool:
        """Check if query requires multimodal processing."""
        multimodal_indicators = [
            r"image", r"photo", r"picture", r"see", r"show", r"identify",
            r"तस्वीर", r"फोटो", r"दिखाना", r"look", r"visual"
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in multimodal_indicators)

    def _calculate_confidence(self, query: str, intent: str, complexity: str) -> float:
        """Calculate classification confidence."""
        base_confidence = 0.7

        # Boost confidence for clear patterns
        if intent != "general":
            intent_patterns = self.intent_patterns.get(intent, [])
            matches = sum(1 for pattern in intent_patterns if re.search(pattern, query, re.IGNORECASE))
            base_confidence += min(matches * 0.1, 0.25)

        # Adjust for query clarity
        word_count = len(query.split())
        if word_count < 3:
            base_confidence -= 0.2
        elif word_count > 15:
            base_confidence += 0.1

        return min(max(base_confidence, 0.1), 1.0)

    def get_classification_summary(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Get human-readable classification summary."""
        return {
            "intent": analysis.intent,
            "complexity": analysis.complexity,
            "confidence": f"{analysis.confidence:.2f}",
            "urgency": analysis.urgency,
            "domain": analysis.domain,
            "multimodal": analysis.requires_multimodal,
            "entities": analysis.entities
        }
