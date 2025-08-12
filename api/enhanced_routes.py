"""
Enhanced API Routes with Phase 3 Advanced AI Features
Integrates:
- Enhanced Intent Classifier with Transformers
- Multi-Modal Input Processing
- Real-Time Learning System
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import json
import base64
import io
import uuid
import time
from datetime import datetime

# Import our enhanced components
from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier, IntentPrediction
from core.multimodal.multimodal_processor import MultiModalProcessor, MultiModalAnalysis
from core.learning.realtime_learning import RealTimeLearningSystem, UserFeedback
from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine

# Initialize FastAPI app
app = FastAPI(
    title="AI Mood Food Recommender - Enhanced Edition",
    description="Advanced AI food recommendation system with deep learning, multi-modal input, and real-time learning",
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

# Initialize enhanced components
try:
    enhanced_classifier = EnhancedIntentClassifier()
    multimodal_processor = MultiModalProcessor()
    learning_system = RealTimeLearningSystem()
    recommendation_engine = MoodBasedRecommendationEngine()
    print("✅ Enhanced AI components initialized successfully!")
except Exception as e:
    print(f"⚠️ Warning: Some enhanced components failed to initialize: {e}")
    print("Falling back to basic components...")
    enhanced_classifier = None
    multimodal_processor = None
    learning_system = None
    recommendation_engine = None

# Pydantic models for enhanced API
class EnhancedRecommendationRequest(BaseModel):
    """Enhanced recommendation request with multi-modal support."""
    text_input: Optional[str] = Field(None, description="Text description of mood/food preference")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio (WAV format)")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User context")
    user_id: Optional[str] = Field(None, description="User identifier for personalization")
    session_id: Optional[str] = Field(None, description="Session identifier")
    top_k: int = Field(10, description="Number of recommendations to return")

class EnhancedRecommendationResponse(BaseModel):
    """Enhanced recommendation response with multi-modal analysis."""
    recommendations: List[Dict[str, Any]]
    multimodal_analysis: Optional[MultiModalAnalysis] = None
    intent_prediction: Optional[IntentPrediction] = None
    user_preferences: Optional[Dict[str, Any]] = None
    system_performance: Optional[Dict[str, Any]] = None
    model_version: str
    processing_time: float

class FeedbackRequest(BaseModel):
    """Enhanced feedback request for learning."""
    user_id: str
    session_id: str
    input_text: str
    recommended_foods: List[str]
    selected_food: Optional[str] = None
    rating: Optional[float] = Field(None, ge=1, le=5, description="Rating 1-5")
    feedback_text: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class LearningMetricsResponse(BaseModel):
    """Learning system metrics response."""
    current_performance: Dict[str, Any]
    learning_history: List[Dict[str, Any]]
    user_insights: Dict[str, Any]
    model_versions: List[Dict[str, Any]]

class MultiModalAnalysisRequest(BaseModel):
    """Multi-modal analysis request."""
    text: Optional[str] = None
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None

class ModelInfoResponse(BaseModel):
    """Model information response."""
    enhanced_classifier: Optional[Dict[str, Any]] = None
    multimodal_processor: Optional[Dict[str, Any]] = None
    learning_system: Optional[Dict[str, Any]] = None
    recommendation_engine: Optional[Dict[str, Any]] = None

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "AI Mood Food Recommender - Enhanced Edition v3.0",
        "phase": "Phase 3: Advanced AI Features",
        "features": [
            "Deep Learning Models with Transformers",
            "Semantic Embeddings for Food-Mood Relationships",
            "Multi-Modal Input (Text, Image, Voice)",
            "Real-Time Learning from User Feedback",
            "Enhanced Intent Classification",
            "Advanced Recommendation Algorithms"
        ],
        "status": "operational",
        "version": "3.0.0"
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
        
        # Process multi-modal input
        multimodal_analysis = None
        intent_prediction = None
        
        if multimodal_processor:
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
                
                # Use multimodal analysis for intent
                if multimodal_analysis:
                    request.text_input = request.text_input or ""
                    if multimodal_analysis.image_analysis:
                        request.text_input += f" [Image: {multimodal_analysis.image_analysis.get('caption', '')}]"
                    if multimodal_analysis.audio_analysis:
                        request.text_input += f" [Audio: {multimodal_analysis.audio_analysis.get('transcript', '')}]"
                
            except Exception as e:
                print(f"Multi-modal processing error: {e}")
        
        # Enhanced intent classification
        if enhanced_classifier and request.text_input:
            try:
                intent_prediction = enhanced_classifier.classify_intent(request.text_input)
            except Exception as e:
                print(f"Enhanced intent classification error: {e}")
        
        # Get recommendations
        if recommendation_engine:
            recommendations = recommendation_engine.get_recommendations(
                user_input=request.text_input or "general food preference",
                user_context=request.user_context or {},
                top_k=request.top_k
            )
        else:
            # Fallback to basic recommendations
            recommendations = [
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
                print(f"User preferences error: {e}")
        
        # Get system performance
        system_performance = None
        if learning_system:
            try:
                system_performance = learning_system.get_system_performance()
            except Exception as e:
                print(f"System performance error: {e}")
        
        processing_time = time.time() - start_time
        
        return EnhancedRecommendationResponse(
            recommendations=recommendations,
            multimodal_analysis=multimodal_analysis,
            intent_prediction=intent_prediction,
            user_preferences=user_preferences,
            system_performance=system_performance,
            model_version=getattr(enhanced_classifier, 'current_model_version', 'v1.0') if enhanced_classifier else 'v1.0',
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced recommendation error: {str(e)}")

# Multi-modal analysis endpoint
@app.post("/analyze-multimodal")
async def analyze_multimodal(request: MultiModalAnalysisRequest):
    """Analyze multi-modal input for mood and intent."""
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
        
        return {
            "analysis": analysis,
            "processing_info": multimodal_processor.get_processing_info()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-modal analysis error: {str(e)}")

# Enhanced feedback endpoint
@app.post("/enhanced-feedback")
async def enhanced_feedback(request: FeedbackRequest):
    """Record enhanced user feedback for learning."""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
    
    try:
        # Record feedback
        learning_system.record_feedback(
            user_id=request.user_id,
            session_id=request.session_id,
            input_text=request.input_text,
            recommended_foods=request.recommended_foods,
            selected_food=request.selected_food,
            rating=request.rating,
            feedback_text=request.feedback_text,
            context=request.context
        )
        
        return {
            "message": "Feedback recorded successfully",
            "user_id": request.user_id,
            "session_id": request.session_id,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback recording error: {str(e)}")

# Learning metrics endpoint
@app.get("/learning-metrics", response_model=LearningMetricsResponse)
async def get_learning_metrics():
    """Get learning system performance metrics."""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
    
    try:
        current_performance = learning_system.get_system_performance()
        
        return LearningMetricsResponse(
            current_performance=current_performance,
            learning_history=learning_system.performance_history[-20:],  # Last 20 records
            user_insights={
                "unique_users": len(learning_system.user_sessions),
                "total_sessions": sum(len(sessions) for sessions in learning_system.user_sessions.values())
            },
            model_versions=learning_system.model_versions[-10:]  # Last 10 versions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning metrics error: {str(e)}")

# User preferences endpoint
@app.get("/user-preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """Get learned user preferences."""
    if not learning_system:
        raise HTTPException(status_code=503, detail="Learning system not available")
    
    try:
        preferences = learning_system.get_user_preferences(user_id)
        return {
            "user_id": user_id,
            "preferences": preferences,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User preferences error: {str(e)}")

# Model information endpoint
@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about all AI models."""
    try:
        info = ModelInfoResponse()
        
        if enhanced_classifier:
            info.enhanced_classifier = enhanced_classifier.get_model_info()
        
        if multimodal_processor:
            info.multimodal_processor = multimodal_processor.get_processing_info()
        
        if learning_system:
            info.learning_system = learning_system.get_system_performance()
        
        if recommendation_engine:
            info.recommendation_engine = {
                "status": "operational",
                "taxonomy_categories": len(recommendation_engine.taxonomy) if hasattr(recommendation_engine, 'taxonomy') else 0
            }
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

# File upload endpoint for images
@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    user_context: str = Form("{}")
):
    """Upload and analyze an image for food recommendations."""
    if not multimodal_processor:
        raise HTTPException(status_code=503, detail="Multi-modal processor not available")
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Process image
        image_analysis = multimodal_processor.process_image(image_data)
        
        # Get recommendations based on image
        if recommendation_engine:
            # Create a description from image analysis
            image_description = image_analysis.get('caption', 'food image')
            recommendations = recommendation_engine.get_recommendations(
                user_input=f"I see {image_description}",
                user_context=json.loads(user_context) if user_context else {},
                top_k=5
            )
        else:
            recommendations = []
        
        return {
            "image_analysis": image_analysis,
            "recommendations": recommendations,
            "filename": file.filename,
            "file_size": len(image_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check with component status."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "components": {
            "enhanced_classifier": "operational" if enhanced_classifier else "unavailable",
            "multimodal_processor": "operational" if multimodal_processor else "unavailable",
            "learning_system": "operational" if learning_system else "unavailable",
            "recommendation_engine": "operational" if recommendation_engine else "unavailable"
        }
    }
    
    # Check if any critical components are missing
    if not any([enhanced_classifier, multimodal_processor, learning_system, recommendation_engine]):
        health_status["status"] = "degraded"
        health_status["message"] = "Some enhanced components are unavailable"
    
    return health_status

# Examples endpoint
@app.get("/examples")
async def get_examples():
    """Get example queries for testing."""
    return {
        "examples": {
            "text_only": [
                "I want something warm and comforting",
                "I'm feeling hot and need something refreshing",
                "It's date night, I want something romantic",
                "I need something quick for lunch break",
                "I'm craving something spicy and exciting"
            ],
            "with_context": [
                {
                    "text": "I want something light",
                    "context": {"time_of_day": "lunch", "weather": "hot", "energy_level": "low"}
                },
                {
                    "text": "Something celebratory",
                    "context": {"social_context": "party", "energy_level": "high"}
                }
            ],
            "multi_modal": [
                "Upload a food photo and describe your mood",
                "Record voice describing what you want to eat",
                "Combine text, image, and context for best results"
            ]
        }
    }

# Advanced features endpoint
@app.get("/advanced-features")
async def get_advanced_features():
    """Get information about advanced AI features."""
    return {
        "phase_3_features": {
            "deep_learning": {
                "description": "Enhanced intent classification with transformers",
                "models": ["Sentence Transformers", "BERT-based models"],
                "capabilities": ["Semantic understanding", "Context awareness", "Multi-label classification"]
            },
            "semantic_embeddings": {
                "description": "Better understanding of food-mood relationships",
                "features": ["Vector similarity", "Semantic matching", "Embedding updates"],
                "benefits": ["More accurate recommendations", "Better mood understanding"]
            },
            "multi_modal": {
                "description": "Support for image, voice, and text input",
                "modalities": ["Text processing", "Image analysis", "Speech recognition"],
                "capabilities": ["Food recognition", "Mood indicators", "Combined analysis"]
            },
            "real_time_learning": {
                "description": "Continuous improvement from user feedback",
                "features": ["Online learning", "Performance tracking", "Adaptive updates"],
                "benefits": ["Self-improving system", "User preference learning"]
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
