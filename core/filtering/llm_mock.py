"""
Mock LLM Module for Testing and Development

This module provides deterministic canned responses to support unit/integration tests
and local development without using OpenRouter API calls.
"""

import json
import logging
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MockLLMResponse:
    """Mock LLM API response structure"""
    success: bool
    raw_output: str
    http_status: int
    error: Optional[str]
    meta: Dict[str, Any]

class MockLLMValidator:
    """
    Mock LLM validator that provides deterministic responses for testing.
    
    This allows testing the hybrid filter pipeline without hitting rate limits
    or requiring real API keys.
    """
    
    def __init__(self):
        # Predefined responses for different query types
        self.query_responses = {
            "japanese": {
                "recommendations": ["sushi", "ramen", "tempura", "udon", "gyoza"],
                "intent": "japanese_cuisine",
                "reasoning": "User specifically requested Japanese food"
            },
            "italian": {
                "recommendations": ["pizza", "pasta", "risotto", "bruschetta", "tiramisu"],
                "intent": "italian_cuisine", 
                "reasoning": "User requested Italian cuisine"
            },
            "spicy": {
                "recommendations": ["curry", "hot wings", "jalapeno poppers", "szechuan chicken", "vindaloo"],
                "intent": "spicy_food",
                "reasoning": "User wants spicy food"
            },
            "comfort": {
                "recommendations": ["mac and cheese", "chicken soup", "grilled cheese", "mashed potatoes", "hot chocolate"],
                "intent": "comfort_food",
                "reasoning": "User needs comforting food"
            },
            "healthy": {
                "recommendations": ["salad", "grilled chicken", "quinoa bowl", "smoothie", "vegetable stir fry"],
                "intent": "healthy_food",
                "reasoning": "User wants healthy options"
            },
            "quick": {
                "recommendations": ["sandwich", "wrap", "soup", "salad", "fruit"],
                "intent": "quick_meal",
                "reasoning": "User needs something quick"
            }
        }
        
        # Default fallback response
        self.default_response = {
            "recommendations": ["comfort food", "soup", "tea"],
            "intent": "general",
            "reasoning": "General food recommendation"
        }
        
        # Validation responses
        self.validation_responses = {
            "positive": ["yes", "true", "correct", "appropriate", "good match"],
            "negative": ["no", "false", "incorrect", "inappropriate", "bad match"],
            "neutral": ["maybe", "partially", "somewhat", "could be"]
        }
    
    def interpret_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> MockLLMResponse:
        """
        Mock query interpretation that returns structured food recommendations.
        
        Args:
            query: User's food query
            context: Optional user context
            
        Returns:
            MockLLMResponse with structured recommendations
        """
        query_lower = query.lower()
        
        # Find the best matching response category
        response = self._find_best_match(query_lower)
        
        # Format as JSON response
        json_response = {
            "recommendations": response["recommendations"],
            "intent": response["intent"],
            "reasoning": response["reasoning"],
            "confidence": 0.85
        }
        
        raw_output = json.dumps(json_response, indent=2)
        
        return MockLLMResponse(
            success=True,
            raw_output=raw_output,
            http_status=200,
            error=None,
            meta={
                "attempts": 1,
                "latency_ms": random.randint(50, 200),
                "mock_mode": True
            }
        )
    
    def validate_prediction(self, ml_prediction: Dict[str, Any], query: str) -> MockLLMResponse:
        """
        Mock prediction validation that returns semantic consistency assessment.
        
        Args:
            ml_prediction: ML model's prediction
            query: Original user query
            
        Returns:
            MockLLMResponse with validation result
        """
        query_lower = query.lower()
        ml_labels = ml_prediction.get("labels", [])
        ml_confidence = ml_prediction.get("confidence", 0.0)
        
        # Simple validation logic based on query content and ML confidence
        if "japanese" in query_lower and any("japanese" in label.lower() for label in ml_labels):
            validation_result = "yes"
        elif "spicy" in query_lower and any("spicy" in label.lower() for label in ml_labels):
            validation_result = "yes"
        elif ml_confidence > 0.7:
            validation_result = "yes"
        elif ml_confidence < 0.3:
            validation_result = "no"
        else:
            validation_result = random.choice(self.validation_responses["neutral"])
        
        # Format response
        json_response = {
            "validation": validation_result,
            "reasoning": f"ML prediction confidence: {ml_confidence}, labels: {ml_labels}",
            "query": query
        }
        
        raw_output = json.dumps(json_response, indent=2)
        
        return MockLLMResponse(
            success=True,
            raw_output=raw_output,
            http_status=200,
            error=None,
            meta={
                "attempts": 1,
                "latency_ms": random.randint(30, 150),
                "mock_mode": True
            }
        )
    
    def generate_recommendations(self, query: str, context: Optional[Dict[str, Any]] = None) -> MockLLMResponse:
        """
        Mock recommendation generation for direct LLM fallback.
        
        Args:
            query: User's food query
            context: Optional user context
            
        Returns:
            MockLLMResponse with food recommendations
        """
        response = self._find_best_match(query.lower())
        
        json_response = {
            "recommendations": response["recommendations"],
            "reasoning": f"Generated recommendations for: {response['intent']}",
            "method": "llm_direct"
        }
        
        raw_output = json.dumps(json_response, indent=2)
        
        return MockLLMResponse(
            success=True,
            raw_output=raw_output,
            http_status=200,
            error=None,
            meta={
                "attempts": 1,
                "latency_ms": random.randint(100, 300),
                "mock_mode": True
            }
        )
    
    def _find_best_match(self, query: str) -> Dict[str, Any]:
        """Find the best matching response category for a query."""
        # Check for exact matches first
        for category, response in self.query_responses.items():
            if category in query:
                return response
        
        # Check for partial matches
        for category, response in self.query_responses.items():
            if any(word in query for word in category.split("_")):
                return response
        
        # Check for keyword matches
        if any(word in query for word in ["hot", "warm", "cozy", "comfort"]):
            return self.query_responses["comfort"]
        elif any(word in query for word in ["fast", "quick", "hurry"]):
            return self.query_responses["quick"]
        elif any(word in query for word in ["good", "healthy", "nutritious"]):
            return self.query_responses["healthy"]
        
        # Default fallback
        return self.default_response
    
    def simulate_rate_limit(self) -> MockLLMResponse:
        """Simulate a rate limit error for testing circuit breaker."""
        return MockLLMResponse(
            success=False,
            raw_output="",
            http_status=429,
            error="Rate limit exceeded",
            meta={
                "attempts": 1,
                "latency_ms": 0,
                "mock_mode": True
            }
        )
    
    def simulate_timeout(self) -> MockLLMResponse:
        """Simulate a timeout error for testing resilience."""
        return MockLLMResponse(
            success=False,
            raw_output="",
            http_status=408,
            error="Request timeout",
            meta={
                "attempts": 1,
                "latency_ms": 10000,
                "mock_mode": True
            }
        )

def get_mock_llm_validator() -> MockLLMValidator:
    """Factory function to get a mock LLM validator instance."""
    return MockLLMValidator()
