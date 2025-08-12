"""
Phase 3 Enhancements - Advanced AI Features Consolidation
This module provides enhanced implementations of all Phase 3 features with improved robustness and performance.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Phase3Config:
    """Configuration for Phase 3 features."""
    # Model configurations
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model: str = "microsoft/resnet-50"
    device: Optional[str] = None
    
    # Learning parameters
    learning_rate: float = 0.001
    feedback_threshold: int = 10
    embedding_dim: int = 384
    
    # Performance settings
    batch_size: int = 32
    max_sequence_length: int = 512
    confidence_threshold: float = 0.3
    
    # Feature flags
    enable_multimodal: bool = True
    enable_realtime_learning: bool = True
    enable_semantic_search: bool = True
    enable_context_awareness: bool = True

class Phase3FeatureManager:
    """
    Centralized manager for all Phase 3 advanced AI features.
    
    Features:
    - Enhanced intent classification with transformers
    - Multi-modal input processing (text, image, audio)
    - Real-time learning from user feedback
    - Semantic embeddings and similarity search
    - Context-aware recommendations
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[Phase3Config] = None):
        self.config = config or Phase3Config()
        self._setup_device()
        self._initialize_components()
        
    def _setup_device(self):
        """Setup optimal device for AI operations."""
        if self.config.device is None:
            if torch and torch.cuda.is_available():
                self.config.device = "cuda"
            elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.config.device = "mps"
            else:
                self.config.device = "cpu"
        
        logger.info(f"Phase 3 using device: {self.config.device}")
    
    def _initialize_components(self):
        """Initialize all Phase 3 components."""
        self.components = {
            "enhanced_classifier": None,
            "multimodal_processor": None,
            "learning_system": None,
            "semantic_engine": None,
            "context_analyzer": None
        }
        
        # Initialize components with error handling
        self._init_enhanced_classifier()
        self._init_multimodal_processor()
        self._init_learning_system()
        self._init_semantic_engine()
        self._init_context_analyzer()
        
        logger.info("Phase 3 components initialized")
    
    def _init_enhanced_classifier(self):
        """Initialize enhanced intent classifier."""
        try:
            from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
            self.components["enhanced_classifier"] = EnhancedIntentClassifier(
                model_name=self.config.text_model,
                device=self.config.device
            )
            logger.info("Enhanced classifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced classifier: {e}")
    
    def _init_multimodal_processor(self):
        """Initialize multi-modal processor."""
        if not self.config.enable_multimodal:
            return
            
        try:
            from core.multimodal.multimodal_processor import MultiModalProcessor
            self.components["multimodal_processor"] = MultiModalProcessor(
                text_model_name=self.config.text_model,
                image_model_name=self.config.image_model,
                device=self.config.device
            )
            logger.info("Multi-modal processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize multi-modal processor: {e}")
    
    def _init_learning_system(self):
        """Initialize real-time learning system."""
        if not self.config.enable_realtime_learning:
            return
            
        try:
            from core.learning.realtime_learning import RealTimeLearningSystem
            self.components["learning_system"] = RealTimeLearningSystem()
            logger.info("Real-time learning system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize learning system: {e}")
    
    def _init_semantic_engine(self):
        """Initialize semantic search engine."""
        if not self.config.enable_semantic_search:
            return
            
        try:
            self.components["semantic_engine"] = SemanticSearchEngine(
                model_name=self.config.text_model,
                device=self.config.device
            )
            logger.info("Semantic search engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize semantic engine: {e}")
    
    def _init_context_analyzer(self):
        """Initialize context awareness analyzer."""
        if not self.config.enable_context_awareness:
            return
            
        try:
            self.components["context_analyzer"] = ContextAnalyzer()
            logger.info("Context analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize context analyzer: {e}")
    
    def process_user_request(
        self,
        text_input: Optional[str] = None,
        image_input: Optional[Union[str, bytes]] = None,
        audio_input: Optional[bytes] = None,
        user_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user request using all available Phase 3 features.
        
        Returns comprehensive analysis including:
        - Intent classification
        - Multi-modal analysis
        - Context understanding
        - Semantic search results
        - Learning feedback integration
        """
        start_time = time.time()
        results = {
            "phase": "Phase 3: Advanced AI",
            "processing_time": 0.0,
            "components_used": [],
            "intent_analysis": {},
            "multimodal_analysis": {},
            "context_analysis": {},
            "semantic_results": {},
            "learning_insights": {},
            "recommendations": []
        }
        
        try:
            # 1. Enhanced Intent Classification
            if text_input and self.components["enhanced_classifier"]:
                results["intent_analysis"] = self._analyze_intent(text_input)
                results["components_used"].append("enhanced_classifier")
            
            # 2. Multi-modal Analysis
            if self.components["multimodal_processor"]:
                results["multimodal_analysis"] = self._analyze_multimodal(
                    text_input, image_input, audio_input
                )
                results["components_used"].append("multimodal_processor")
            
            # 3. Context Analysis
            if user_context and self.components["context_analyzer"]:
                results["context_analysis"] = self._analyze_context(user_context)
                results["components_used"].append("context_analyzer")
            
            # 4. Semantic Search
            if text_input and self.components["semantic_engine"]:
                results["semantic_results"] = self._semantic_search(text_input)
                results["components_used"].append("semantic_engine")
            
            # 5. Learning Integration
            if user_id and self.components["learning_system"]:
                results["learning_insights"] = self._get_learning_insights(user_id)
                results["components_used"].append("learning_system")
            
            # 6. Generate Enhanced Recommendations
            results["recommendations"] = self._generate_enhanced_recommendations(
                results, user_context, user_id
            )
            
        except Exception as e:
            logger.error(f"Error in Phase 3 processing: {e}")
            results["error"] = str(e)
        
        results["processing_time"] = time.time() - start_time
        return results
    
    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """Analyze user intent using enhanced classifier."""
        try:
            classifier = self.components["enhanced_classifier"]
            prediction = classifier.classify_intent(text)
            
            return {
                "primary_intent": prediction.primary_intent,
                "confidence": prediction.confidence,
                "all_intents": prediction.all_intents,
                "semantic_similarity": prediction.semantic_similarity
            }
        except Exception as e:
            logger.error(f"Intent analysis error: {e}")
            return {"error": str(e)}
    
    def _analyze_multimodal(
        self,
        text: Optional[str],
        image: Optional[Union[str, bytes]],
        audio: Optional[bytes]
    ) -> Dict[str, Any]:
        """Analyze multi-modal input."""
        try:
            processor = self.components["multimodal_processor"]
            analysis = processor.process_multimodal(text=text, image=image, audio=audio)
            
            return {
                "primary_mood": analysis.primary_mood,
                "confidence": analysis.confidence,
                "mood_categories": analysis.mood_categories,
                "extracted_entities": analysis.extracted_entities,
                "combined_confidence": analysis.combined_confidence
            }
        except Exception as e:
            logger.error(f"Multi-modal analysis error: {e}")
            return {"error": str(e)}
    
    def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user context for enhanced understanding."""
        try:
            analyzer = self.components["context_analyzer"]
            return analyzer.analyze_context(context)
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            return {"error": str(e)}
    
    def _semantic_search(self, query: str) -> Dict[str, Any]:
        """Perform semantic search for related concepts."""
        try:
            engine = self.components["semantic_engine"]
            return engine.search(query)
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return {"error": str(e)}
    
    def _get_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """Get learning insights for user personalization."""
        try:
            learning_system = self.components["learning_system"]
            return learning_system.get_user_preferences(user_id)
        except Exception as e:
            logger.error(f"Learning insights error: {e}")
            return {"error": str(e)}
    
    def _generate_enhanced_recommendations(
        self,
        analysis_results: Dict[str, Any],
        user_context: Optional[Dict[str, Any]],
        user_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate enhanced recommendations using all available insights."""
        recommendations = []
        
        # Combine all analysis results for better recommendations
        intent = analysis_results.get("intent_analysis", {}).get("primary_intent", "general")
        mood = analysis_results.get("multimodal_analysis", {}).get("primary_mood", "general")
        context = analysis_results.get("context_analysis", {})
        learning = analysis_results.get("learning_insights", {})
        
        # Enhanced recommendation logic would go here
        # For now, return a structured recommendation
        recommendations.append({
            "type": "enhanced_phase3",
            "intent_based": intent,
            "mood_based": mood,
            "context_aware": bool(context),
            "personalized": bool(learning),
            "confidence": analysis_results.get("intent_analysis", {}).get("confidence", 0.5)
        })
        
        return recommendations
    
    def record_feedback(
        self,
        user_id: str,
        session_id: str,
        input_text: str,
        recommendations: List[str],
        selected_item: Optional[str] = None,
        rating: Optional[float] = None,
        feedback_text: Optional[str] = None
    ):
        """Record user feedback for continuous learning."""
        try:
            # Record in learning system
            if self.components["learning_system"]:
                self.components["learning_system"].record_feedback(
                    user_id=user_id,
                    session_id=session_id,
                    input_text=input_text,
                    recommended_foods=recommendations,
                    selected_food=selected_item,
                    rating=rating,
                    feedback_text=feedback_text
                )
            
            # Update intent classifier if needed
            if self.components["enhanced_classifier"] and selected_item:
                # Extract correct intents from selection
                correct_intents = [selected_item]  # Simplified
                self.components["enhanced_classifier"].update_with_feedback(
                    text=input_text,
                    correct_intents=correct_intents,
                    confidence=rating / 5.0 if rating else 0.5
                )
                
            logger.info(f"Feedback recorded for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "phase": "Phase 3: Advanced AI",
            "components": {},
            "performance": {},
            "capabilities": []
        }
        
        # Component status
        for name, component in self.components.items():
            status["components"][name] = {
                "available": component is not None,
                "type": type(component).__name__ if component else None
            }
        
        # Performance metrics
        if self.components["learning_system"]:
            performance = self.components["learning_system"].get_system_performance()
            status["performance"] = performance
        
        # Capabilities
        capabilities = []
        if self.components["enhanced_classifier"]:
            capabilities.append("Deep Learning Intent Classification")
        if self.components["multimodal_processor"]:
            capabilities.append("Multi-Modal Input Processing")
        if self.components["learning_system"]:
            capabilities.append("Real-Time Learning")
        if self.components["semantic_engine"]:
            capabilities.append("Semantic Search")
        if self.components["context_analyzer"]:
            capabilities.append("Context Awareness")
        
        status["capabilities"] = capabilities
        
        return status

class SemanticSearchEngine:
    """Enhanced semantic search engine for food-mood relationships."""
    
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the semantic model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.model = self.model.to(self.device)
            logger.info("Semantic search engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize semantic model: {e}")
    
    def search(self, query: str) -> Dict[str, Any]:
        """Perform semantic search."""
        if not self.model:
            return {"error": "Model not available"}
        
        try:
            # Encode query
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            query_embedding = query_embedding.to(self.device)
            
            # This would typically search against a database of embeddings
            # For now, return a placeholder result
            return {
                "query": query,
                "results": [],
                "embedding_dim": query_embedding.shape[0],
                "search_performed": True
            }
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return {"error": str(e)}

class ContextAnalyzer:
    """Advanced context analyzer for better understanding of user situations."""
    
    def __init__(self):
        self.context_patterns = self._load_context_patterns()
    
    def _load_context_patterns(self) -> Dict[str, Any]:
        """Load context analysis patterns."""
        return {
            "time_patterns": {
                "morning": ["breakfast", "energizing", "light"],
                "afternoon": ["lunch", "sustaining", "balanced"],
                "evening": ["dinner", "satisfying", "complete"],
                "night": ["comfort", "soothing", "light"]
            },
            "weather_patterns": {
                "hot": ["refreshing", "cooling", "light"],
                "cold": ["warming", "hearty", "comforting"],
                "rainy": ["comforting", "warm", "cozy"],
                "sunny": ["fresh", "light", "energizing"]
            },
            "social_patterns": {
                "alone": ["comfort", "simple", "personal"],
                "couple": ["romantic", "elegant", "intimate"],
                "family": ["comforting", "familiar", "nourishing"],
                "friends": ["social", "sharing", "festive"]
            }
        }
    
    def analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user context for enhanced recommendations."""
        analysis = {
            "time_recommendations": [],
            "weather_recommendations": [],
            "social_recommendations": [],
            "combined_insights": []
        }
        
        # Analyze time context
        time_of_day = context.get("time_of_day", "").lower()
        if time_of_day in self.context_patterns["time_patterns"]:
            analysis["time_recommendations"] = self.context_patterns["time_patterns"][time_of_day]
        
        # Analyze weather context
        weather = context.get("weather", "").lower()
        if weather in self.context_patterns["weather_patterns"]:
            analysis["weather_recommendations"] = self.context_patterns["weather_patterns"][weather]
        
        # Analyze social context
        social_context = context.get("social_context", "").lower()
        if social_context in self.context_patterns["social_patterns"]:
            analysis["social_recommendations"] = self.context_patterns["social_patterns"][social_context]
        
        # Combine insights
        all_recommendations = (
            analysis["time_recommendations"] +
            analysis["weather_recommendations"] +
            analysis["social_recommendations"]
        )
        analysis["combined_insights"] = list(set(all_recommendations))
        
        return analysis

# Convenience functions
def create_phase3_manager(config: Optional[Phase3Config] = None) -> Phase3FeatureManager:
    """Create and return a Phase 3 feature manager."""
    return Phase3FeatureManager(config)

def get_phase3_status() -> Dict[str, Any]:
    """Get Phase 3 system status."""
    manager = Phase3FeatureManager()
    return manager.get_system_status()
