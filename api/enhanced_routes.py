"""
Enhanced API Routes for Phase 3: Advanced AI Features
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import time
import uuid
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
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

# Initialize components with error handling
enhanced_classifier = None
multimodal_processor = None
learning_system = None
recommendation_engine = None

try:
    from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
    enhanced_classifier = EnhancedIntentClassifier()
    logger.info("Enhanced intent classifier initialized")
except Exception as e:
    logger.warning(f"Failed to initialize enhanced intent classifier: {e}")

try:
    from core.multimodal.multimodal_processor import MultiModalProcessor, MultiModalAnalysis
    multimodal_processor = MultiModalProcessor()
    logger.info("Multi-modal processor initialized")
except Exception as e:
    logger.warning(f"Failed to initialize multi-modal processor: {e}")

try:
    from core.learning.realtime_learning import RealTimeLearningSystem
    learning_system = RealTimeLearningSystem()
    logger.info("Real-time learning system initialized")
except Exception as e:
    logger.warning(f"Failed to initialize real-time learning system: {e}")

try:
    from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine
    recommendation_engine = MoodBasedRecommendationEngine()
    logger.info("Recommendation engine initialized")
except Exception as e:
    logger.warning(f"Failed to initialize recommendation engine: {e}")

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
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
        
        # Process multi-modal input (only if image or audio provided to avoid heavy init on text-only)
        multimodal_analysis = None
        intent_prediction = None
        use_multimodal = bool((request.image_base64 and request.image_base64.strip()) or (request.audio_base64 and request.audio_base64.strip()))
        
        if multimodal_processor and use_multimodal:
            try:
                # Convert base64 inputs to appropriate formats
                image_data = None
                audio_data = None
                
                if request.image_base64:
                    image_data = base64.b64decode(request.image_base64)
                
                if request.audio_base64:
                    audio_data = base64.b64decode(request.audio_base64)
                
                # Process multi-modal input
                multimodal_analysis = multimodal_processor.process_multimodal(
                    text=request.text_input,
                    image=image_data,
                    audio=audio_data
                )
                
                # Use multimodal analysis for intent augmentation
                if multimodal_analysis:
                    request.text_input = request.text_input or ""
                    if multimodal_analysis.image_analysis:
                        request.text_input += f" [Image: {multimodal_analysis.image_analysis.get('caption', '')}]"
                    if multimodal_analysis.audio_analysis:
                        request.text_input += f" [Audio: {multimodal_analysis.audio_analysis.get('transcript', '')}]"
                
            except Exception as e:
                logger.error(f"Multi-modal processing error: {e}")
        
        # Enhanced intent classification
        if enhanced_classifier and request.text_input:
            try:
                intent_prediction = enhanced_classifier.classify_intent(request.text_input)
            except Exception as e:
                logger.error(f"Enhanced intent classification error: {e}")
        
        # Get recommendations
        if recommendation_engine:
            recommendations = recommendation_engine.get_recommendations(
                user_input=request.text_input or "general food preference",
                user_context=request.user_context or {},
                top_k=request.top_k
            )
            
            # Convert Recommendation dataclass objects to dictionaries
            recommendations_dict = []
            for rec in recommendations:
                rec_dict = {
                    "food_name": rec.food_item.name,
                    "food_category": rec.food_item.category,
                    "food_region": rec.food_item.region,
                    "food_culture": rec.food_item.culture,
                    "food_tags": rec.food_item.tags,
                    "score": rec.score,
                    "mood_match": rec.mood_match,
                    "context_match": rec.context_match,
                    "personalization_score": rec.personalization_score,
                    "reasoning": rec.reasoning,
                    "restaurant": {
                        "name": rec.restaurant.name,
                        "cuisine_type": rec.restaurant.cuisine_type,
                        "rating": rec.restaurant.rating,
                        "price_range": rec.restaurant.price_range,
                        "delivery_available": rec.restaurant.delivery_available
                    } if rec.restaurant else None
                }
                recommendations_dict.append(rec_dict)
        else:
            # Fallback to basic recommendations
            recommendations_dict = [
                {
                    "food_name": "Sample Food",
                    "food_category": "general",
                    "score": 0.8,
                    "reasoning": ["Fallback recommendation"]
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
            user_input=request.input_text,
            recommendations=request.recommendations,
            selected_recommendation=request.selected_recommendation,
            rating=request.rating,
            text_feedback=request.feedback_text,
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
        "status": "healthy",
        "components": {
            "enhanced_classifier": "operational" if enhanced_classifier else "not_available",
            "multimodal_processor": "operational" if multimodal_processor else "not_available",
            "learning_system": "operational" if learning_system else "not_available",
            "recommendation_engine": "operational" if recommendation_engine else "not_available"
        },
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
