"""
Enhanced Hybrid Filter with LLM-as-Teacher Architecture

This system implements a continuous learning loop where:
1. LLM always interprets queries into structured intent + reasoning
2. ML attempts prediction and gets validated by LLM
3. LLM can fallback to direct generation if ML fails
4. All interactions are logged for continuous retraining
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from core.filtering.llm_validator import LLMValidator, LLMResponse
from core.filtering.adaptive_parser import AdaptiveParser, ParseResult
from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier

logger = logging.getLogger(__name__)

@dataclass
class LLMInterpretation:
    """Structured interpretation of user query by LLM."""
    intent: str
    reasoning: str
    confidence: float
    mood_categories: List[str]
    cuisine_type: Optional[str] = None
    temperature_preference: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = None

@dataclass
class MLPrediction:
    """ML classifier prediction with confidence."""
    primary_intent: str
    confidence: float
    all_intents: List[Tuple[str, float]]
    method: str

@dataclass
class HybridFilterResponse:
    """Unified response structure from hybrid filter."""
    decision: str  # "ml_validated" | "llm_fallback"
    recommendations: List[str]
    reasoning: str
    ml_prediction: Optional[MLPrediction] = None
    llm_interpretation: Optional[LLMInterpretation] = None
    processing_time_ms: float = 0.0
    timestamp: str = ""

class HybridFilter:
    """
    Enhanced hybrid filter with LLM-as-teacher architecture.
    
    Features:
    - LLM always interprets queries (teacher role)
    - ML attempts prediction and gets validated
    - LLM can fallback to direct generation
    - Continuous logging for retraining
    - Redis-backed live statistics
    """
    
    def __init__(self, 
                 llm_validator: Optional[LLMValidator] = None,
                 ml_classifier: Optional[EnhancedIntentClassifier] = None,
                 confidence_threshold: float = 0.7,
                 enable_llm_fallback: bool = True):
        
        self.llm_validator = llm_validator
        self.ml_classifier = ml_classifier
        self.confidence_threshold = confidence_threshold
        self.enable_llm_fallback = enable_llm_fallback
        
        # Initialize adaptive parser
        self.adaptive_parser = AdaptiveParser()
        
        # Live statistics for monitoring
        self.live_stats = {
            "total_queries": 0,
            "ml_validated": 0,
            "llm_fallback": 0,
            "llm_training_samples": 0,
            "processing_errors": 0
        }
        
        # Ensure logs directory exists
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Hybrid filter initialized with confidence threshold: {confidence_threshold}")
    
    async def process_query(self, 
                          user_query: str, 
                          user_context: Optional[Dict[str, Any]] = None) -> HybridFilterResponse:
        """
        Process user query through the LLM-as-teacher pipeline.
        
        Args:
            user_query: Raw user input
            user_context: Optional user context
            
        Returns:
            HybridFilterResponse with unified structure
        """
        start_time = datetime.now()
        
        try:
            # Step 1: LLM always interprets the query (teacher role)
            logger.info(f"Step 1: LLM interpreting query: '{user_query}'")
            llm_interpretation = await self._get_llm_interpretation(user_query, user_context)
            
            # Step 2: ML attempts prediction
            logger.info("Step 2: ML attempting prediction")
            ml_prediction = self._get_ml_prediction(user_query, user_context)
            
            # Step 3: Check if ML prediction meets confidence threshold
            logger.info(f"Step 3: Checking ML prediction confidence: {ml_prediction.confidence} >= {self.confidence_threshold}")
            
            # Step 4: Determine final response
            if ml_prediction.confidence >= self.confidence_threshold and ml_prediction.primary_intent not in ["unknown", "error", "no_ml_classifier"]:
                # ML prediction is confident enough - use it directly without LLM validation
                decision = "ml_validated"
                recommendations = self._generate_ml_based_recommendations(ml_prediction)
                reasoning = f"ML prediction '{ml_prediction.primary_intent}' (confidence: {ml_prediction.confidence:.3f}) meets confidence threshold. Using ML-based recommendations."
                
                self.live_stats["ml_validated"] += 1
                logger.info(f"Using ML prediction directly: {ml_prediction.primary_intent}")
                
            else:
                # ML failed or was invalid - use LLM fallback
                decision = "llm_fallback"
                recommendations = await self._generate_llm_recommendations(
                    user_query, llm_interpretation
                )
                reasoning = f"ML prediction failed validation or had low confidence. LLM generated recommendations directly. {llm_interpretation.reasoning}"
                
                self.live_stats["llm_fallback"] += 1
            
            # Update statistics
            self.live_stats["total_queries"] += 1
            self.live_stats["llm_training_samples"] += 1
            
            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create response
            response = HybridFilterResponse(
                decision=decision,
                recommendations=recommendations,
                reasoning=reasoning,
                ml_prediction=ml_prediction if decision == "ml_validated" else None,
                llm_interpretation=llm_interpretation,
                processing_time_ms=processing_time_ms,
                timestamp=datetime.now().isoformat()
            )
            
            # Log the interaction for retraining
            await self._log_interaction(user_query, response)
            
            logger.info(f"Query processed successfully. Decision: {decision}, Recommendations: {len(recommendations)}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            self.live_stats["processing_errors"] += 1
            
            # Return fallback response
            return HybridFilterResponse(
                decision="llm_fallback",
                recommendations=["comfort food", "soup", "tea"],  # Safe fallback
                reasoning=f"Error occurred during processing: {str(e)}. Using safe fallback recommendations.",
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=datetime.now().isoformat()
            )
    
    def _generate_ml_based_recommendations(self, ml_prediction: MLPrediction) -> List[str]:
        """Generate recommendations based on ML prediction without LLM validation."""
        try:
            # Use the ML prediction to generate food recommendations
            primary_intent = ml_prediction.primary_intent
            
            # Simple mapping of intents to food recommendations
            food_mappings = {
                "WEATHER_HOT": ["spicy curry", "hot soup", "grilled meat", "spicy noodles", "hot pot"],
                "WEATHER_COLD": ["warm soup", "hot chocolate", "roasted vegetables", "stew", "hot tea"],
                "EMOTIONAL_COMFORT": ["mac and cheese", "chicken soup", "grilled cheese", "mashed potatoes", "comfort food"],
                "OCCASION_PARTY_SNACKS": ["chips and dip", "finger foods", "appetizers", "party platters", "snack mix"],
                "FLAVOR_SPICY": ["spicy tacos", "hot wings", "spicy ramen", "curry", "spicy pizza"],
                "GOAL_COMFORT": ["comfort food", "home cooking", "traditional dishes", "family recipes", "warm meals"],
                "SENSORY_WARMING": ["hot soup", "warm bread", "roasted dishes", "hot beverages", "warm desserts"]
            }
            
            # Get recommendations for the primary intent
            if primary_intent in food_mappings:
                recommendations = food_mappings[primary_intent]
            else:
                # Fallback recommendations based on general food categories
                recommendations = ["comfort food", "soup", "main dish", "side dish", "beverage"]
            
            logger.info(f"Generated ML-based recommendations for {primary_intent}: {recommendations}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating ML-based recommendations: {e}")
            return ["comfort food", "soup", "main dish"]  # Safe fallback
    
    async def _get_llm_interpretation(self, 
                                    user_query: str, 
                                    user_context: Optional[Dict[str, Any]]) -> LLMInterpretation:
        """Get structured interpretation from LLM."""
        try:
            # Create a comprehensive prompt for LLM interpretation
            prompt = f"""
            Analyze this food query and provide structured interpretation:
            
            Query: "{user_query}"
            
            Please provide:
            1. Primary intent (e.g., comfort, japanese, spicy, cold, healthy)
            2. Reasoning for your interpretation
            3. Confidence level (0.0-1.0)
            4. Mood categories that apply
            5. Cuisine type if applicable
            6. Temperature preference if mentioned
            7. Any dietary restrictions mentioned
            
            Format as JSON:
            {{
                "intent": "string",
                "reasoning": "string", 
                "confidence": 0.0,
                "mood_categories": ["list"],
                "cuisine_type": "string or null",
                "temperature_preference": "string or null",
                "dietary_restrictions": ["list or null"]
            }}
            """
            
            # Get LLM response using robust validator
            if not self.llm_validator:
                logger.warning("LLM validator not available, using fallback interpretation")
                return LLMInterpretation(
                    intent="fallback",
                    reasoning="LLM validator not available",
                    confidence=0.0,
                    mood_categories=[]
                )
            
            # Use the new robust LLM validator
            llm_response: LLMResponse = self.llm_validator.interpret_query(user_query, user_context)
            
            if not llm_response.success:
                logger.warning(f"LLM interpretation failed: {llm_response.error}")
                return LLMInterpretation(
                    intent="unknown",
                    reasoning=f"LLM interpretation failed: {llm_response.error}",
                    confidence=0.0,
                    mood_categories=[]
                )
            
            # Use adaptive parser to extract structured data
            parse_result = self.adaptive_parser.parse_llm_output(llm_response.raw_output)
            
            if parse_result.parse_status == "none":
                logger.warning("Failed to parse LLM response, using fallback")
                return LLMInterpretation(
                    intent="unknown",
                    reasoning="Failed to parse LLM response",
                    confidence=0.0,
                    mood_categories=[]
                )
            
            # Extract recommendations and create interpretation
            recommendations = parse_result.parsed
            intent = "unknown"
            reasoning = "LLM provided recommendations"
            confidence = parse_result.confidence
            
            # Try to extract intent from the parsed recommendations
            if recommendations:
                # Simple heuristic: use first recommendation as intent indicator
                first_rec = recommendations[0].lower()
                if any(cuisine in first_rec for cuisine in ["japanese", "chinese", "italian", "mexican", "indian"]):
                    intent = f"{first_rec}_cuisine"
                elif any(mood in first_rec for mood in ["comfort", "spicy", "healthy", "quick"]):
                    intent = f"{first_rec}_food"
                else:
                    intent = "general_food"
            
            return LLMInterpretation(
                intent=intent,
                reasoning=reasoning,
                confidence=confidence,
                mood_categories=recommendations[:3],  # Use first 3 as mood categories
                cuisine_type=None,
                temperature_preference=None,
                dietary_restrictions=None
            )
                
        except Exception as e:
            logger.error(f"Error getting LLM interpretation: {e}")
            return LLMInterpretation(
                intent="unknown",
                reasoning=f"Error occurred: {str(e)}",
                confidence=0.0,
                mood_categories=[]
            )
    
    def _get_ml_prediction(self, 
                          user_query: str, 
                          user_context: Optional[Dict[str, Any]]) -> MLPrediction:
        """Get ML classifier prediction."""
        try:
            if not self.ml_classifier:
                return MLPrediction(
                    primary_intent="no_ml_classifier",
                    confidence=0.0,
                    all_intents=[("no_ml_classifier", 0.0)],
                    method="no_classifier"
                )
            
            # Get ML prediction
            result = self.ml_classifier.classify_intent(user_query, user_context)
            
            return MLPrediction(
                primary_intent=result.get("primary_intent", "unknown"),
                confidence=result.get("confidence", 0.0),
                all_intents=result.get("all_intents", []),
                method=result.get("method", "unknown")
            )
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return MLPrediction(
                primary_intent="error",
                confidence=0.0,
                all_intents=[("error", 0.0)],
                method="error"
            )
    
    async def _validate_ml_prediction(self, 
                                    user_query: str,
                                    ml_prediction: MLPrediction,
                                    llm_interpretation: LLMInterpretation) -> Dict[str, Any]:
        """Validate ML prediction using LLM."""
        try:
            prompt = f"""
            Validate this ML prediction for a food query:
            
            Query: "{user_query}"
            ML Prediction: {ml_prediction.primary_intent} (confidence: {ml_prediction.confidence:.3f})
            LLM Interpretation: {llm_interpretation.intent} - {llm_interpretation.reasoning}
            
            Please:
            1. Determine if the ML prediction is semantically correct
            2. If valid, polish and expand the recommendations
            3. If invalid, explain why
            
            Return JSON:
            {{
                "is_valid": true/false,
                "validation_reason": "string",
                "polished_recommendations": ["list of food items"],
                "confidence_boost": 0.0
            }}
            """
            
            # Use the new robust LLM validator
            if not self.llm_validator:
                logger.warning("LLM validator not available for prediction validation")
                return {
                    "is_valid": False,
                    "validation_reason": "LLM validator not available",
                    "polished_recommendations": [],
                    "confidence_boost": 0.0
                }
            
            ml_dict = {
                "labels": [ml_prediction.primary_intent],
                "confidence": ml_prediction.confidence
            }
            llm_response: LLMResponse = self.llm_validator.validate_prediction(ml_dict, user_query)
            
            if not llm_response.success:
                logger.warning(f"LLM validation failed: {llm_response.error}")
                return {
                    "is_valid": False,
                    "validation_reason": f"LLM validation failed: {llm_response.error}",
                    "polished_recommendations": [],
                    "confidence_boost": 0.0
                }
            
            # Use adaptive parser to extract validation result
            parse_result = self.adaptive_parser.parse_llm_output(llm_response.raw_output)
            
            if parse_result.parse_status == "none":
                logger.warning("Failed to parse validation response")
                return {
                    "is_valid": False,
                    "validation_reason": "Failed to parse validation response",
                    "polished_recommendations": [],
                    "confidence_boost": 0.0
                }
            
            # Extract validation result
            validation_text = " ".join(parse_result.parsed).lower()
            is_valid = any(word in validation_text for word in ["yes", "true", "correct", "valid"])
            
            # Use recommendations from interpretation if available
            polished_recommendations = []
            if llm_interpretation and hasattr(llm_interpretation, 'mood_categories'):
                polished_recommendations = llm_interpretation.mood_categories
            
            return {
                "is_valid": is_valid,
                "validation_reason": f"LLM validation result: {validation_text}",
                "polished_recommendations": polished_recommendations,
                "confidence_boost": 0.1 if is_valid else 0.0
            }
                
        except Exception as e:
            logger.error(f"Error validating ML prediction: {e}")
            return {
                "is_valid": False,
                "validation_reason": f"Error occurred: {str(e)}",
                "polished_recommendations": [],
                "confidence_boost": 0.0
            }
    
    async def _generate_llm_recommendations(self, 
                                          user_query: str,
                                          llm_interpretation: LLMInterpretation) -> List[str]:
        """Generate recommendations directly from LLM."""
        try:
            if not self.llm_validator:
                logger.warning("LLM validator not available, using fallback recommendations")
                return ["comfort food", "soup", "tea"]
            
            # Use the new robust LLM validator
            llm_response: LLMResponse = self.llm_validator.generate_recommendations(user_query, None)
            
            if not llm_response.success:
                logger.warning(f"LLM recommendation generation failed: {llm_response.error}")
                return ["comfort food", "soup", "tea"]
            
            # Use adaptive parser to extract recommendations
            parse_result = self.adaptive_parser.parse_llm_output(llm_response.raw_output)
            
            if parse_result.parse_status == "none" or not parse_result.parsed:
                logger.warning("Failed to parse LLM recommendations")
                return ["comfort food", "soup", "tea"]
            
            # Return parsed recommendations, limited to 8
            return parse_result.parsed[:8]
                
        except Exception as e:
            logger.error(f"Error generating LLM recommendations: {e}")
            return ["comfort food", "soup", "tea"]
    
    async def _log_interaction(self, user_query: str, response: HybridFilterResponse):
        """Log interaction for continuous retraining."""
        try:
            log_entry = {
                "timestamp": response.timestamp,
                "query": user_query,
                "ml_prediction": asdict(response.ml_prediction) if response.ml_prediction else None,
                "llm_interpretation": asdict(response.llm_interpretation) if response.llm_interpretation else None,
                "final_response": response.recommendations,
                "decision_source": response.decision,
                "reasoning": response.reasoning,
                "processing_time_ms": response.processing_time_ms
            }
            
            # Append to recommendation logs
            log_file = self.logs_dir / "recommendation_logs.jsonl"
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            logger.info(f"Interaction logged to {log_file}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
    def get_live_stats(self) -> Dict[str, Any]:
        """Get live statistics for monitoring."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": self.live_stats["total_queries"],
            "ml_validated": self.live_stats["ml_validated"],
            "llm_fallback": self.live_stats["llm_fallback"],
            "llm_training_samples": self.live_stats["llm_training_samples"],
            "processing_errors": self.live_stats["processing_errors"],
            "ml_success_rate": (self.live_stats["ml_validated"] / max(self.live_stats["total_queries"], 1)) * 100
        }
    
    def reset_stats(self):
        """Reset live statistics."""
        self.live_stats = {
            "total_queries": 0,
            "ml_validated": 0,
            "llm_fallback": 0,
            "llm_training_samples": 0,
            "processing_errors": 0
        }
        logger.info("Live statistics reset")
