"""
Enhanced API Routes for Phase 3: Advanced AI Features
"""

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    # Fallback placeholders if imports fail
    FastAPI = None
    HTTPException = None
    UploadFile = None
    File = None
    CORSMiddleware = None
    BaseModel = None
from typing import List, Dict, Any, Optional, Union
import time
import uuid
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback component implementations
def create_fallback_intent_classifier():
    """Create a fallback intent classifier when the main one fails."""
    class FallbackIntentClassifier:
        def classify_intent(self, text: str, top_k: int = 5):
            # Simple keyword-based fallback
            text_lower = text.lower()
            intents = []
            
            if any(word in text_lower for word in ['warm', 'hot', 'cold', 'weather']):
                intents.append(('WEATHER_BASED', 0.8))
            if any(word in text_lower for word in ['sad', 'happy', 'stressed', 'excited']):
                intents.append(('EMOTIONAL', 0.7))
            if any(word in text_lower for word in ['spicy', 'sweet', 'salty', 'flavor']):
                intents.append(('FLAVOR_BASED', 0.6))
            if any(word in text_lower for word in ['party', 'date', 'family', 'alone']):
                intents.append(('OCCASION_BASED', 0.5))
            
            if not intents:
                intents.append(('GENERAL_FOOD', 0.5))
            
            return type('IntentPrediction', (), {
                'primary_intent': intents[0][0],
                'confidence': intents[0][1],
                'all_intents': intents[:top_k]
            })()
        
        def get_model_info(self):
            return {"status": "fallback", "model": "keyword_based"}
    
    logger.info("Using fallback intent classifier")
    return FallbackIntentClassifier()

def create_fallback_multimodal_processor():
    """Create a fallback multi-modal processor when the main one fails."""
    class FallbackMultiModalProcessor:
        def process_multimodal(self, text=None, image=None, audio=None):
            return type('MultiModalAnalysis', (), {
                'primary_mood': 'general',
                'confidence': 0.5,
                'mood_categories': ['general_food'],
                'extracted_entities': [],
                'combined_confidence': 0.5
            })()
        
        def get_processing_info(self):
            return {"status": "fallback", "processor": "basic_text_only"}
    
    logger.info("Using fallback multi-modal processor")
    return FallbackMultiModalProcessor()

def create_fallback_learning_system():
    """Create a fallback learning system when the main one fails."""
    class FallbackLearningSystem:
        def record_feedback(self, **kwargs):
            logger.info("Fallback learning system: feedback recorded")
            return True
        
        def get_user_preferences(self, user_id: str):
            return {"status": "fallback", "preferences": "default"}
        
        def get_system_performance(self):
            return {"status": "fallback", "performance": "basic"}
    
    logger.info("Using fallback learning system")
    return FallbackLearningSystem()

def create_fallback_recommendation_engine():
    """Create a fallback recommendation engine when the main one fails."""
    class FallbackRecommendationEngine:
        def get_recommendations(self, user_input: str, user_context: dict = None, top_k: int = 5):
            # Basic fallback recommendations
            fallback_foods = [
                {"name": "Comfort Food", "category": "general", "region": "various", "culture": "international"},
                {"name": "Healthy Option", "category": "wellness", "region": "various", "culture": "international"},
                {"name": "Quick Meal", "category": "convenience", "region": "various", "culture": "international"}
            ]
            
            recommendations = []
            for i, food in enumerate(fallback_foods[:top_k]):
                recommendations.append(type('Recommendation', (), {
                    'food_item': type('FoodItem', (), food)(),
                    'score': 0.7 - (i * 0.1),
                    'mood_match': 0.6,
                    'context_match': 0.5,
                    'personalization_score': 0.5,
                    'reasoning': [f"Fallback recommendation {i+1}"],
                    'restaurant': None
                })())
            
            return recommendations
    
    logger.info("Using fallback recommendation engine")
    return FallbackRecommendationEngine()

def create_fallback_phase3_manager():
    """Create a fallback Phase 3 manager when the main one fails."""
    class FallbackPhase3Manager:
        def process_user_request(self, **kwargs):
            return {
                "status": "fallback",
                "components_used": ["fallback_basic"],
                "processing_time": 0.1
            }
        
        def get_system_status(self):
            return {"status": "fallback", "capabilities": "basic"}
    
    logger.info("Using fallback Phase 3 manager")
    return FallbackPhase3Manager()

# Initialize FastAPI app
if FastAPI is None:
    raise ImportError("FastAPI is not installed. Please install it with: pip install fastapi")
app = FastAPI(
    title="AI Mood-Based Food Recommendation System - Phase 3",
    description="Advanced AI system with deep learning, multi-modal input, and real-time learning",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components with robust error handling and fallbacks
enhanced_classifier = None
multimodal_processor = None
learning_system = None
recommendation_engine = None
phase3_manager = None

# Component initialization status
component_status = {
    "enhanced_classifier": False,
    "multimodal_processor": False,
    "learning_system": False,
    "recommendation_engine": False,
    "phase3_manager": False
}

def initialize_component(component_name: str, init_func, fallback_func=None):
    """Initialize a component with fallback and timeout protection."""
    try:
        logger.info(f"Initializing {component_name}...")
        
        # Set a timeout for initialization
        import signal
        import threading
        import time
        
        result = [None]
        exception = [None]
        
        def init_with_timeout():
            try:
                result[0] = init_func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=init_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)  # 30 second timeout
        
        if thread.is_alive():
            logger.error(f"{component_name} initialization timed out after 30 seconds")
            if fallback_func:
                logger.info(f"Using fallback for {component_name}")
                result[0] = fallback_func()
            return None
        elif exception[0]:
            raise exception[0]
        else:
            logger.info(f"{component_name} initialized successfully")
            component_status[component_name] = True
            return result[0]
            
    except Exception as e:
        logger.error(f"Failed to initialize {component_name}: {e}")
        if fallback_func:
            logger.info(f"Using fallback for {component_name}")
            try:
                result = fallback_func()
                component_status[component_name] = True
                return result
            except Exception as fallback_e:
                logger.error(f"Fallback for {component_name} also failed: {fallback_e}")
        return None

# Initialize Enhanced Intent Classifier
try:
    from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
    
    # Resolve taxonomy path relative to current working directory
    import os
    taxonomy_path = os.path.join(os.getcwd(), "data", "taxonomy", "mood_food_taxonomy.json")
    if not os.path.exists(taxonomy_path):
        # Try relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        taxonomy_path = os.path.join(script_dir, "..", "data", "taxonomy", "mood_food_taxonomy.json")
    
    logger.info(f"Using taxonomy path: {taxonomy_path}")
    
    enhanced_classifier = initialize_component(
        "enhanced_classifier",
        lambda: EnhancedIntentClassifier(taxonomy_path=taxonomy_path),
        lambda: create_fallback_intent_classifier()
    )
except Exception as e:
    logger.error(f"Enhanced intent classifier import failed: {e}")
    enhanced_classifier = create_fallback_intent_classifier()

# Initialize Multi-Modal Processor
try:
    from core.multimodal.multimodal_processor import MultiModalProcessor, MultiModalAnalysis
    multimodal_processor = initialize_component(
        "multimodal_processor",
        lambda: MultiModalProcessor(),
        lambda: create_fallback_multimodal_processor()
    )
except Exception as e:
    logger.error(f"Multi-modal processor import failed: {e}")
    multimodal_processor = create_fallback_multimodal_processor()

# Initialize Learning System
try:
    from core.learning.realtime_learning import RealTimeLearningSystem
    learning_system = initialize_component(
        "learning_system",
        lambda: RealTimeLearningSystem(),
        lambda: create_fallback_learning_system()
    )
except Exception as e:
    logger.error(f"Learning system import failed: {e}")
    learning_system = create_fallback_learning_system()

# Initialize Recommendation Engine
try:
    from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine
    recommendation_engine = initialize_component(
        "recommendation_engine",
        lambda: MoodBasedRecommendationEngine(),
        lambda: create_fallback_recommendation_engine()
    )
except Exception as e:
    logger.error(f"Recommendation engine import failed: {e}")
    recommendation_engine = create_fallback_recommendation_engine()

# Initialize Phase 3 Manager
try:
    from core.phase3_enhancements import Phase3FeatureManager, Phase3Config
    phase3_config = Phase3Config(
        enable_multimodal=True,
        enable_realtime_learning=True,
        enable_semantic_search=True,
        enable_context_awareness=True
    )
    phase3_manager = initialize_component(
        "phase3_manager",
        lambda: Phase3FeatureManager(phase3_config),
        lambda: create_fallback_phase3_manager()
    )
except Exception as e:
    logger.error(f"Phase 3 manager import failed: {e}")
    phase3_manager = create_fallback_phase3_manager()

# Log overall system status
operational_components = sum(component_status.values())
total_components = len(component_status)
logger.info(f"System initialization complete: {operational_components}/{total_components} components operational")

# Pydantic models for requests and responses
class UserContext(BaseModel):
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    social_context: Optional[str] = None
    energy_level: Optional[str] = None

class EnhancedRecommendationRequest(BaseModel):
    text_input: Optional[str] = None
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None
    user_context: Optional[UserContext] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = 5

class IntentPredictionResponse(BaseModel):
    primary_intent: str
    confidence: float
    all_intents: List[tuple]

class MultiModalAnalysisResponse(BaseModel):
    primary_mood: str
    confidence: float
    mood_categories: List[str]
    extracted_entities: List[str]
    combined_confidence: float

class EnhancedRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    multimodal_analysis: Optional[MultiModalAnalysisResponse] = None
    intent_prediction: Optional[IntentPredictionResponse] = None
    user_preferences: Optional[Dict[str, Any]] = None
    system_performance: Optional[Dict[str, Any]] = None
    model_version: str
    processing_time: float

class FeedbackRequest(BaseModel):
    user_id: str
    session_id: str
    input_text: str
    recommendations: List[Dict[str, Any]]
    selected_recommendation: Optional[str] = None
    rating: Optional[int] = None
    feedback_text: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class LearningMetricsResponse(BaseModel):
    model_version: str
    total_feedback: int
    recent_performance: Dict[str, float]
    learning_stats: Dict[str, Any]

class MultiModalAnalysisRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None

class ModelInfoResponse(BaseModel):
    enhanced_classifier: Dict[str, Any]
    multimodal_processor: Dict[str, Any]
    learning_system: Dict[str, Any]
    recommendation_engine: Dict[str, Any]

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with Phase 3 system information."""
    return {
        "message": "AI Mood-Based Food Recommendation System - Phase 3",
        "version": "3.0.0",
        "features": [
            "Deep Learning Models",
            "Multi-Modal Input Processing",
            "Real-Time Learning",
            "Semantic Understanding"
        ],
        "status": "operational" if any([enhanced_classifier, multimodal_processor, learning_system, recommendation_engine]) else "degraded"
    }

# Enhanced recommendation endpoint
@app.post("/enhanced-recommend", response_model=EnhancedRecommendationResponse)
async def enhanced_recommend(request: EnhancedRecommendationRequest):
    """Get enhanced food recommendations using multi-modal input and advanced AI."""
    import asyncio  # Import at function level to ensure availability
    start_time = time.time()
    
    try:
        # Validate request
        if not request.text_input and not request.image_base64 and not request.audio_base64:
            raise HTTPException(status_code=400, detail="At least one input type (text, image, or audio) is required")
        
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
        
        logger.info(f"Processing enhanced recommendation request: session={request.session_id}, text_length={len(request.text_input or '')}")
        
        # Process multi-modal input with timeout protection
        multimodal_analysis = None
        intent_prediction = None
        use_multimodal = bool((request.image_base64 and request.image_base64.strip()) or (request.audio_base64 and request.audio_base64.strip()))
        
        if multimodal_processor and use_multimodal:
            try:
                logger.info("Processing multi-modal input...")
                
                # Convert base64 inputs to appropriate formats
                image_data = None
                audio_data = None
                
                if request.image_base64:
                    image_data = base64.b64decode(request.image_base64)
                    logger.info(f"Image data decoded: {len(image_data)} bytes")
                
                if request.audio_base64:
                    audio_data = base64.b64decode(request.audio_base64)
                    logger.info(f"Audio data decoded: {len(audio_data)} bytes")
                
                # Process multi-modal input with timeout
                try:
                    multimodal_analysis = await asyncio.wait_for(
                        asyncio.to_thread(multimodal_processor.process_multimodal, 
                                        text=request.text_input,
                                        image=image_data,
                                        audio=audio_data),
                        timeout=15.0  # 15 second timeout for multi-modal processing
                    )
                    logger.info("Multi-modal processing completed successfully")
                except asyncio.TimeoutError:
                    logger.warning("Multi-modal processing timed out, continuing without it")
                    multimodal_analysis = None
                
                # Use multimodal analysis for intent augmentation
                if multimodal_analysis:
                    request.text_input = request.text_input or ""
                    if hasattr(multimodal_analysis, 'image_analysis') and multimodal_analysis.image_analysis:
                        request.text_input += f" [Image: {multimodal_analysis.image_analysis.get('caption', '')}]"
                    if hasattr(multimodal_analysis, 'audio_analysis') and multimodal_analysis.audio_analysis:
                        request.text_input += f" [Audio: {multimodal_analysis.audio_analysis.get('transcript', '')}]"
                
            except Exception as e:
                logger.error(f"Multi-modal processing error: {e}")
                # Continue without multi-modal analysis
        
        # Enhanced intent classification with timeout protection
        if enhanced_classifier and request.text_input:
            try:
                logger.info("Performing enhanced intent classification...")
                
                # Add timeout for intent classification
                try:
                    intent_prediction = await asyncio.wait_for(
                        asyncio.to_thread(enhanced_classifier.classify_intent, request.text_input),
                        timeout=10.0  # 10 second timeout for intent classification
                    )
                    logger.info(f"Intent classification completed: {intent_prediction.primary_intent if intent_prediction else 'None'}")
                except asyncio.TimeoutError:
                    logger.warning("Intent classification timed out, using fallback")
                    intent_prediction = None
                    
            except Exception as e:
                logger.error(f"Enhanced intent classification error: {e}")
                intent_prediction = None
        
        # Get recommendations with timeout protection
        recommendations_dict = []
        if recommendation_engine:
            try:
                logger.info("Getting recommendations from engine...")
                
                # Add timeout for recommendations
                try:
                    recommendations = await asyncio.wait_for(
                        asyncio.to_thread(recommendation_engine.get_recommendations,
                                        user_input=request.text_input or "general food preference",
                                        user_context=request.user_context.dict() if request.user_context else {},
                                        top_k=request.top_k),
                        timeout=20.0  # 20 second timeout for recommendations
                    )
                    
                    # Convert Recommendation dataclass objects to dictionaries
                    for rec in recommendations:
                        try:
                            rec_dict = {
                                "food_name": getattr(rec.food_item, 'name', 'Unknown Food'),
                                "food_category": getattr(rec.food_item, 'category', 'general'),
                                "food_region": getattr(rec.food_item, 'region', 'various'),
                                "food_culture": getattr(rec.food_item, 'culture', 'international'),
                                "food_tags": getattr(rec.food_item, 'tags', []),
                                "score": getattr(rec, 'score', 0.5),
                                "mood_match": getattr(rec, 'mood_match', 0.5),
                                "context_match": getattr(rec, 'context_match', 0.5),
                                "personalization_score": getattr(rec, 'personalization_score', 0.5),
                                "reasoning": getattr(rec, 'reasoning', ["AI-powered recommendation"]),
                                "restaurant": None
                            }
                            
                            # Safely handle restaurant data
                            if hasattr(rec, 'restaurant') and rec.restaurant:
                                rec_dict["restaurant"] = {
                                    "name": getattr(rec.restaurant, 'name', 'Unknown Restaurant'),
                                    "cuisine_type": getattr(rec.restaurant, 'cuisine_type', 'Various'),
                                    "rating": getattr(rec.restaurant, 'rating', 'N/A'),
                                    "price_range": getattr(rec.restaurant, 'price_range', 'N/A'),
                                    "delivery_available": getattr(rec.restaurant, 'delivery_available', False)
                                }
                            
                            recommendations_dict.append(rec_dict)
                        except Exception as rec_error:
                            logger.error(f"Error processing recommendation: {rec_error}")
                            continue
                    
                    logger.info(f"Generated {len(recommendations_dict)} recommendations")
                    
                except asyncio.TimeoutError:
                    logger.warning("Recommendation generation timed out, using fallback")
                    recommendations_dict = []
                    
            except Exception as e:
                logger.error(f"Recommendation engine error: {e}")
                recommendations_dict = []
        
        # Fallback to basic recommendations if none generated
        if not recommendations_dict:
            logger.info("Using fallback recommendations")
            recommendations_dict = [
                {
                    "food_name": "Comfort Food",
                    "food_category": "general",
                    "food_region": "various",
                    "food_culture": "international",
                    "food_tags": ["comfort", "general"],
                    "score": 0.8,
                    "mood_match": 0.6,
                    "context_match": 0.5,
                    "personalization_score": 0.5,
                    "reasoning": ["Fallback recommendation - system using basic mode"],
                    "restaurant": None
                },
                {
                    "food_name": "Healthy Option",
                    "food_category": "wellness",
                    "food_region": "various",
                    "food_culture": "international",
                    "food_tags": ["healthy", "wellness"],
                    "score": 0.7,
                    "mood_match": 0.5,
                    "context_match": 0.5,
                    "personalization_score": 0.5,
                    "reasoning": ["Fallback recommendation - system using basic mode"],
                    "restaurant": None
                }
            ]
        
        # Get user preferences if user_id provided
        user_preferences = None
        if learning_system and request.user_id:
            try:
                user_preferences = learning_system.get_user_preferences(request.user_id)
            except Exception as e:
                logger.error(f"User preferences error: {e}")
        
        # Get system performance
        system_performance = None
        if learning_system:
            try:
                system_performance = learning_system.get_system_performance()
            except Exception as e:
                logger.error(f"System performance error: {e}")
        
        processing_time = time.time() - start_time
        
        # Convert multimodal analysis to response model
        multimodal_response = None
        if multimodal_analysis:
            multimodal_response = MultiModalAnalysisResponse(
                primary_mood=multimodal_analysis.primary_mood,
                confidence=multimodal_analysis.confidence,
                mood_categories=multimodal_analysis.mood_categories,
                extracted_entities=multimodal_analysis.extracted_entities,
                combined_confidence=multimodal_analysis.combined_confidence
            )
        
        # Convert intent prediction to response model
        intent_response = None
        if intent_prediction:
            intent_response = IntentPredictionResponse(
                primary_intent=intent_prediction.primary_intent,
                confidence=intent_prediction.confidence,
                all_intents=intent_prediction.all_intents
            )
        
        return EnhancedRecommendationResponse(
            recommendations=recommendations_dict,
            multimodal_analysis=multimodal_response,
            intent_prediction=intent_response,
            user_preferences=user_preferences,
            system_performance=system_performance,
            model_version=getattr(enhanced_classifier, 'current_model_version', 'v1.0') if enhanced_classifier else 'v1.0',
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced recommendation error: {str(e)}")

# Multi-modal analysis endpoint
@app.post("/analyze-multimodal", response_model=MultiModalAnalysisResponse)
async def analyze_multimodal(request: MultiModalAnalysisRequest):
    """Analyze multi-modal input (text, image, audio)."""
    if not multimodal_processor:
        raise HTTPException(status_code=503, detail="Multi-modal processor not available")
    
    try:
        # Convert base64 inputs
        image_data = None
        audio_data = None
        
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
        
        if request.audio_base64:
            audio_data = base64.b64decode(request.audio_base64)
        
        # Process multi-modal input
        analysis = multimodal_processor.process_multimodal(
            text=request.text,
            image=image_data,
            audio=audio_data
        )
        
        return MultiModalAnalysisResponse(
            primary_mood=analysis.primary_mood,
            confidence=analysis.confidence,
            mood_categories=analysis.mood_categories,
            extracted_entities=analysis.extracted_entities,
            combined_confidence=analysis.combined_confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-modal analysis error: {str(e)}")

# Feedback endpoint
@app.post("/enhanced-feedback")
async def enhanced_feedback(request: FeedbackRequest):
    """Record user feedback for real-time learning."""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
    
    try:
        learning_system.record_feedback(
            user_id=request.user_id,
            session_id=request.session_id,
            input_text=request.input_text,
            recommended_foods=[rec.get("food_name", "") for rec in request.recommendations],
            selected_food=request.selected_recommendation,
            rating=request.rating,
            feedback_text=request.feedback_text,
            context=request.context
        )
        
        return {"message": "Feedback recorded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback recording error: {str(e)}")

# Learning metrics endpoint
@app.get("/learning-metrics", response_model=LearningMetricsResponse)
async def get_learning_metrics():
    """Get system learning metrics and performance."""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
    
    try:
        performance = learning_system.get_system_performance()
        
        return LearningMetricsResponse(
            model_version=performance.get("current_model_version", "v1.0"),
            total_feedback=performance.get("total_feedback", 0),
            recent_performance=performance.get("recent_performance", {}),
            learning_stats=performance.get("learning_stats", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning metrics error: {str(e)}")

# User preferences endpoint
@app.get("/user-preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """Get learned preferences for a specific user."""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
    
    try:
        preferences = learning_system.get_user_preferences(user_id)
        return preferences
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User preferences error: {str(e)}")

# Model information endpoint
@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about all AI models."""
    try:
        return ModelInfoResponse(
            enhanced_classifier=enhanced_classifier.get_model_info() if enhanced_classifier else {},
            multimodal_processor=multimodal_processor.get_processing_info() if multimodal_processor else {},
            learning_system=learning_system.get_system_performance() if learning_system else {},
            recommendation_engine={"status": "operational"} if recommendation_engine else {"status": "not_available"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

# Phase 3 comprehensive endpoint
@app.post("/phase3-analysis")
async def phase3_comprehensive_analysis(request: EnhancedRecommendationRequest):
    """Comprehensive Phase 3 analysis using all advanced AI features."""
    if not phase3_manager:
        raise HTTPException(status_code=503, detail="Phase 3 manager not available")
    
    try:
        # Convert base64 inputs
        image_data = None
        audio_data = None
        
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
        
        if request.audio_base64:
            audio_data = base64.b64decode(request.audio_base64)
        
        # Process with Phase 3 manager
        results = phase3_manager.process_user_request(
            text_input=request.text_input,
            image_input=image_data,
            audio_input=audio_data,
            user_context=request.user_context.dict() if request.user_context else None,
            user_id=request.user_id
        )
        
        return {
            "phase": "Phase 3: Advanced AI",
            "analysis": results,
            "capabilities_used": results.get("components_used", []),
            "processing_time": results.get("processing_time", 0.0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phase 3 analysis error: {str(e)}")

# Phase 3 status endpoint
@app.get("/phase3-status")
async def get_phase3_status():
    """Get comprehensive Phase 3 system status."""
    if not phase3_manager:
        raise HTTPException(status_code=503, detail="Phase 3 manager not available")
    
    try:
        return phase3_manager.get_system_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phase 3 status error: {str(e)}")

# Image upload endpoint
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload and analyze an image for food recommendations."""
    if not multimodal_processor:
        raise HTTPException(status_code=503, detail="Multi-modal processor not available")
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Process image
        analysis = multimodal_processor.process_image(image_data)
        
        # Get recommendations based on image analysis
        if recommendation_engine:
            recommendations = recommendation_engine.get_recommendations(
                user_input=f"Image analysis: {analysis.get('caption', '')}",
                user_context={},
                top_k=5
            )
        else:
            recommendations = []
        
        return {
            "image_analysis": analysis,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check showing status of all components."""
    return {
        "status": "healthy" if any(component_status.values()) else "degraded",
        "components": {
            "enhanced_classifier": "operational" if enhanced_classifier and component_status["enhanced_classifier"] else "not_available",
            "multimodal_processor": "operational" if multimodal_processor and component_status["multimodal_processor"] else "not_available",
            "learning_system": "operational" if learning_system and component_status["learning_system"] else "not_available",
            "recommendation_engine": "operational" if recommendation_engine and component_status["recommendation_engine"] else "not_available",
            "phase3_manager": "operational" if phase3_manager and component_status["phase3_manager"] else "not_available"
        },
        "component_status": component_status,
        "operational_count": sum(component_status.values()),
        "total_components": len(component_status),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

# Examples endpoint
@app.get("/examples")
async def get_examples():
    """Get example queries for testing."""
    return {
        "text_examples": [
            "I want something warm and comforting",
            "I'm craving spicy food",
            "I need something light and healthy",
            "I want romantic dinner options"
        ],
        "context_examples": {
            "time_of_day": ["morning", "afternoon", "evening", "night"],
            "weather": ["hot", "cold", "rainy", "sunny"],
            "social_context": ["alone", "couple", "family", "friends"],
            "energy_level": ["low", "medium", "high"]
        },
        "multimodal_examples": [
            "Upload a food photo for visual analysis",
            "Record voice describing your mood",
            "Combine text and image for better recommendations"
        ]
    }

# Advanced features endpoint
@app.get("/advanced-features")
async def get_advanced_features():
    """Describe the advanced AI features implemented in Phase 3."""
    return {
        "phase": "Phase 3: Advanced AI Features",
        "features": {
            "deep_learning": {
                "description": "Enhanced intent classification using transformers",
                "models": ["Sentence Transformers", "Semantic Embeddings"],
                "capabilities": ["Intent classification", "Semantic similarity", "Real-time learning"]
            },
            "multimodal_input": {
                "description": "Process text, image, and voice inputs",
                "models": ["ResNet-50", "ViT-GPT2", "Speech Recognition"],
                "capabilities": ["Image analysis", "Voice transcription", "Multi-modal fusion"]
            },
            "real_time_learning": {
                "description": "Continuous improvement from user feedback",
                "features": ["Feedback collection", "Performance tracking", "Model updates"],
                "capabilities": ["User preference learning", "System optimization"]
            },
            "semantic_understanding": {
                "description": "Advanced understanding of food-mood relationships",
                "capabilities": ["Context awareness", "Entity extraction", "Mood mapping"]
            }
        }
    }

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("uvicorn not available. Install with: pip install uvicorn")
