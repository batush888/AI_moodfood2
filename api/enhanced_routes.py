"""
Enhanced API Routes for Phase 3 Advanced AI Features
FastAPI implementation with comprehensive error handling and fallbacks
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# Try to import FastAPI components with fallbacks
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
    from pydantic.json import pydantic_encoder
    FASTAPI_AVAILABLE = True
except ImportError as e:
    FASTAPI_AVAILABLE = False
    logging.error(f"FastAPI import failed: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components with error handling
try:
    from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
    from core.multimodal.multimodal_processor import MultiModalProcessor
    from core.learning.realtime_learning import RealTimeLearningSystem
    from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine as RecommendationEngine
    from core.phase3_enhancements import Phase3FeatureManager
    CORE_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Core imports failed: {e}")
    CORE_IMPORTS_AVAILABLE = False

# Pydantic models for API
class UserContext(BaseModel):
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    mood: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = None
    location: Optional[str] = None
    occasion: Optional[str] = None

class EnhancedRecommendationRequest(BaseModel):
    text_input: str = Field(..., description="Text input for mood analysis")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    user_context: Optional[UserContext] = Field(None, description="User context information")

class FeedbackRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    input_text: Optional[str] = None
    recommended_foods: List[str] = Field(default_factory=list)
    selected_food: Optional[str] = None
    rating: Optional[float] = Field(None, ge=0, le=5)
    feedback_text: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class RecentFeedbackWeatherRequest(BaseModel):
    user_id: str
    time_of_day: str

class Restaurant(BaseModel):
    name: str
    cuisine: str
    rating: float
    price_range: str
    location: str

class Reasoning(BaseModel):
    factor: str
    explanation: str
    confidence: float

class Recommendation(BaseModel):
    food_name: str
    food_category: str
    food_region: str
    food_culture: str
    food_tags: List[str]
    score: float
    mood_match: float
    context_match: float
    personalization_score: float
    reasoning: List[str]
    restaurant: Optional[Restaurant] = None

class IntentPrediction(BaseModel):
    primary_intent: str
    confidence: float
    all_intents: List[List[str]]

class EnhancedRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    multimodal_analysis: Optional[Dict[str, Any]] = None
    intent_prediction: Dict[str, Any]
    user_preferences: Optional[Dict[str, Any]] = None
    system_performance: Dict[str, Any] = Field(default_factory=dict)
    model_version: str = "v1.0"
    processing_time: float

# Initialize FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="AI Mood-Based Food Recommendation System",
        description="Phase 3: Advanced AI Features with Multi-modal Input Support",
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
    
    # Mount static files for frontend
    try:
        frontend_path = Path(__file__).parent.parent / "frontend"
        if frontend_path.exists():
            app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
            logger.info(f"Frontend mounted at /static from {frontend_path}")
        else:
            logger.warning(f"Frontend directory not found at {frontend_path}")
    except Exception as e:
        logger.error(f"Failed to mount frontend: {e}")
else:
    app = None
    logger.error("FastAPI not available - API routes cannot be created")

# Global component instances
enhanced_classifier = None
multimodal_processor = None
learning_system = None
recommendation_engine = None
phase3_manager = None

async def initialize_component(component_name: str, init_function, fallback_function, timeout: int = 30):
    """Initialize a component with timeout and fallback protection (async)."""
    logger.info(f"Initializing {component_name}...")
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(init_function),
            timeout=timeout
        )
        logger.info(f"{component_name} initialized successfully")
        return result
    except asyncio.TimeoutError:
        logger.warning(f"{component_name} initialization timed out after {timeout}s")
        return fallback_function()
    except Exception as e:
        logger.error(f"{component_name} initialization failed: {e}")
        return fallback_function()

def create_fallback_enhanced_classifier():
    """Create a fallback enhanced classifier."""
    logger.warning("Using fallback enhanced classifier")
    return None

def create_fallback_multimodal_processor():
    """Create a fallback multi-modal processor."""
    logger.warning("Using fallback multi-modal processor")
    return None

def create_fallback_learning_system():
    """Create a fallback learning system."""
    logger.warning("Using fallback learning system")
    return None

def create_fallback_recommendation_engine():
    """Create a fallback recommendation engine."""
    logger.warning("Using fallback recommendation engine")
    return None

def create_fallback_phase3_manager():
    """Create a fallback phase 3 manager."""
    logger.warning("Using fallback phase 3 manager")
    return None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup."""
    global enhanced_classifier, multimodal_processor, learning_system, recommendation_engine, phase3_manager
    
    logger.info("Starting AI Mood-Based Food Recommendation System...")
    
    # Get taxonomy path
    taxonomy_path = os.path.join(os.getcwd(), "data", "taxonomy", "mood_food_taxonomy.json")
    logger.info(f"Using taxonomy path: {taxonomy_path}")
    
    # Initialize components with timeouts and fallbacks
    enhanced_classifier = await initialize_component(
        "enhanced_classifier",
        lambda: EnhancedIntentClassifier(taxonomy_path) if CORE_IMPORTS_AVAILABLE else None,
        create_fallback_enhanced_classifier,
        timeout=10
    )
    
    multimodal_processor = await initialize_component(
        "multimodal_processor",
        lambda: MultiModalProcessor() if CORE_IMPORTS_AVAILABLE else None,
        create_fallback_multimodal_processor,
        timeout=15
    )
    
    learning_system = await initialize_component(
        "learning_system",
        lambda: RealTimeLearningSystem() if CORE_IMPORTS_AVAILABLE else None,
        create_fallback_learning_system,
        timeout=10
    )
    
    recommendation_engine = await initialize_component(
        "recommendation_engine",
        lambda: RecommendationEngine() if CORE_IMPORTS_AVAILABLE else None,
        create_fallback_recommendation_engine,
        timeout=20
    )
    
    phase3_manager = await initialize_component(
        "phase3_manager",
        lambda: Phase3FeatureManager() if CORE_IMPORTS_AVAILABLE else None,
        create_fallback_phase3_manager,
        timeout=15
    )
    
    # Count operational components
    operational_count = sum([
        enhanced_classifier is not None,
        multimodal_processor is not None,
        learning_system is not None,
        recommendation_engine is not None,
        phase3_manager is not None
    ])
    
    logger.info(f"System initialization complete: {operational_count}/5 components operational")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    operational_count = sum([
        enhanced_classifier is not None,
        multimodal_processor is not None,
        learning_system is not None,
        recommendation_engine is not None,
        phase3_manager is not None
    ])
    
    return {
        "status": "healthy" if operational_count > 0 else "degraded",
        "components": {
            "enhanced_classifier": "operational" if enhanced_classifier else "not_available",
            "multimodal_processor": "operational" if multimodal_processor else "not_available",
            "learning_system": "operational" if learning_system else "not_available",
            "recommendation_engine": "operational" if recommendation_engine else "not_available",
            "phase3_manager": "operational" if phase3_manager else "not_available"
        },
        "component_status": {
            "enhanced_classifier": enhanced_classifier is not None,
            "multimodal_processor": multimodal_processor is not None,
            "learning_system": learning_system is not None,
            "recommendation_engine": recommendation_engine is not None,
            "phase3_manager": phase3_manager is not None
        },
        "operational_count": operational_count,
        "total_components": 5,
        "timestamp": "2025-08-14T14:12:19"
    }

@app.post("/enhanced-recommend", response_model=EnhancedRecommendationResponse)
async def enhanced_recommend(request: EnhancedRecommendationRequest):
    """Enhanced recommendation endpoint with multi-modal support."""
    import asyncio
    
    start_time = asyncio.get_event_loop().time()
    session_id = "2a53f226-5bbc-4149-8571-57e3ddff668f"  # Simplified session handling
    
    # Initialize comprehensive logging
    try:
        from core.logging.query_logger import log_query_sync, log_intent_results_sync, log_recommendations_sync, log_error_sync
        
        # Log incoming query
        query_id = log_query_sync(
            text_input=request.text_input,
            image_input=request.image_base64,
            audio_input=request.audio_base64,
            user_context=request.user_context.model_dump() if request.user_context else None,
            session_id=session_id
        )
        logger.info(f"Logged query {query_id} for session {session_id}")
        
    except ImportError:
        logger.warning("Query logging not available, continuing without logging")
        query_id = None
    
    logger.info(f"Processing enhanced recommendation request: session={session_id}, text_length={len(request.text_input)}")
    
    try:
        # Perf tracing containers
        phase_times_ms: Dict[str, float] = {}
        timeouts: Dict[str, bool] = {"intent": False, "multimodal": False, "engine": False}
        input_sizes: Dict[str, Any] = {
            "text_len": len(request.text_input or ""),
            "image_b64_len": len(request.image_base64) if request.image_base64 else 0,
            "audio_b64_len": len(request.audio_base64) if request.audio_base64 else 0,
        }
        # 1. Enhanced Intent Classification
        logger.info("Performing enhanced intent classification...")
        intent_result = None
        
        if enhanced_classifier:
            try:
                _t0 = asyncio.get_event_loop().time()
                intent_result = await asyncio.wait_for(
                    asyncio.to_thread(enhanced_classifier.classify_intent, request.text_input),
                    timeout=10.0
                )
                phase_times_ms["intent_ms"] = round((asyncio.get_event_loop().time() - _t0) * 1000.0, 2)
                logger.info(f"Intent classification completed: {intent_result.get('primary_intent', 'unknown')}")
                logger.info(f"Full intent result: {intent_result}")
                
                # Log intent classification results
                if query_id and 'log_intent_results_sync' in locals():
                    try:
                        log_intent_results_sync(
                            query_id=query_id,
                            primary_intent=intent_result.get('primary_intent', 'unknown'),
                            confidence=intent_result.get('confidence', 0.0),
                            all_intents=intent_result.get('all_intents', []),
                            method=intent_result.get('method', 'unknown'),
                            processing_time_ms=phase_times_ms["intent_ms"]
                        )
                    except Exception as e:
                        logger.error(f"Failed to log intent results: {e}")
            except asyncio.TimeoutError:
                logger.warning("Intent classification timed out, using fallback")
                timeouts["intent"] = True
                intent_result = {"primary_intent": "fallback", "confidence": 0.5, "all_intents": [["fallback", 0.5]]}
            except Exception as e:
                logger.error(f"Intent classification failed: {e}")
                intent_result = {"primary_intent": "error", "confidence": 0.1, "all_intents": [["error", 0.1]]}
        else:
            # Fallback intent classification
            intent_result = {"primary_intent": "fallback", "confidence": 0.5, "all_intents": [["fallback", 0.5]]}
        
        # 2. Multi-modal Analysis
        multimodal_analysis = None
        if multimodal_processor and (request.image_base64 or request.audio_base64):
            try:
                _t0 = asyncio.get_event_loop().time()
                multimodal_analysis = await asyncio.wait_for(
                    asyncio.to_thread(
                        multimodal_processor.process_multimodal,
                        text=request.text_input,
                        image=request.image_base64,
                        audio=request.audio_base64
                    ),
                    timeout=15.0
                )
                phase_times_ms["multimodal_ms"] = round((asyncio.get_event_loop().time() - _t0) * 1000.0, 2)
            except asyncio.TimeoutError:
                logger.warning("Multi-modal analysis timed out")
                timeouts["multimodal"] = True
            except Exception as e:
                logger.error(f"Multi-modal analysis failed: {e}")
        
        # 3. Get Recommendations
        logger.info("Getting recommendations from engine...")
        recommendations = []
        
        if recommendation_engine:
            try:
                logger.info("Calling recommendation engine...")
                _t0 = asyncio.get_event_loop().time()
                
                # Pass enhanced intent results to recommendation engine
                enhanced_context = request.user_context.model_dump() if request.user_context else {}
                enhanced_context['text_input'] = request.text_input  # Add text input for filtering
                if intent_result and intent_result.get('primary_intent'):
                    enhanced_context['enhanced_intent'] = intent_result.get('primary_intent')
                    enhanced_context['all_intents'] = intent_result.get('all_intents', [])
                    enhanced_context['intent_confidence'] = intent_result.get('confidence', 0.5)
                
                recommendations = await asyncio.wait_for(
                    asyncio.to_thread(
                        recommendation_engine.get_recommendations,
                        request.text_input,
                        enhanced_context,
                        top_k=5
                    ),
                    timeout=20.0
                )
                phase_times_ms["engine_ms"] = round((asyncio.get_event_loop().time() - _t0) * 1000.0, 2)
                
                logger.info(f"Raw recommendations: {len(recommendations)} items")
                
                # Convert recommendations to proper format
                formatted_recommendations = []
                _fmt_t0 = asyncio.get_event_loop().time()
                for i, rec in enumerate(recommendations):
                    try:
                        logger.info(f"Processing recommendation {i+1}: {type(rec)}")
                        
                        # Extract food details from food_item
                        food_item = rec.food_item
                        restaurant = rec.restaurant
                        
                        logger.info(f"Food item: {food_item.name}, Restaurant: {restaurant.name if restaurant else 'None'}")
                        
                        # Format restaurant data
                        restaurant_dict = None
                        if restaurant is not None:
                            try:
                                # Convert location dict to string
                                location_str = f"{restaurant.location.get('lat', 0)}, {restaurant.location.get('lng', 0)}" if isinstance(restaurant.location, dict) else str(restaurant.location)
                                
                                restaurant_dict = {
                                    "name": restaurant.name,
                                    "cuisine": restaurant.cuisine_type,
                                    "rating": restaurant.rating,
                                    "price_range": restaurant.price_range,
                                    "location": location_str
                                }
                            except Exception as restaurant_error:
                                logger.warning(f"Error formatting restaurant: {restaurant_error}")
                                restaurant_dict = None
                        
                        # Create formatted recommendation
                        formatted_rec = {
                            "food_name": food_item.name,
                            "food_category": food_item.category,
                            "food_region": food_item.region,
                            "food_culture": food_item.culture,
                            "food_tags": food_item.tags,
                            "score": rec.score,
                            "mood_match": rec.mood_match,
                            "context_match": rec.context_match,
                            "personalization_score": rec.personalization_score,
                            "reasoning": rec.reasoning,
                            "restaurant": restaurant_dict
                        }
                        
                        formatted_recommendations.append(formatted_rec)
                        logger.info(f"Formatted recommendation {i+1} successfully")
                        
                    except Exception as rec_error:
                        logger.error(f"Error formatting recommendation {i+1}: {rec_error}")
                        continue
                
                recommendations = formatted_recommendations
                phase_times_ms["format_ms"] = round((asyncio.get_event_loop().time() - _fmt_t0) * 1000.0, 2)
                logger.info(f"Generated {len(recommendations)} formatted recommendations")
                
                # Log recommendation results
                if query_id and 'log_recommendations_sync' in locals():
                    try:
                        log_recommendations_sync(
                            query_id=query_id,
                            recommendations=recommendations,
                            engine_time_ms=phase_times_ms.get("engine_ms", 0)
                        )
                    except Exception as e:
                        logger.error(f"Failed to log recommendations: {e}")
                
            except asyncio.TimeoutError:
                logger.warning("Recommendation generation timed out")
                timeouts["engine"] = True
                recommendations = []
            except Exception as e:
                logger.error(f"Recommendation generation failed: {e}")
                import traceback
                traceback.print_exc()
                recommendations = []
        
        # 4. Fallback recommendations if none generated
        if not recommendations:
            logger.warning("No recommendations generated, using fallback")
            recommendations = [
                {
                    "food_name": "comfort food",
                    "food_category": "GENERAL",
                    "food_region": "Global",
                    "food_culture": "Universal",
                    "food_tags": ["comfort", "general"],
                    "score": 0.5,
                    "mood_match": 0.5,
                    "context_match": 0.5,
                    "personalization_score": 0.5,
                    "reasoning": ["Fallback recommendation"],
                    "restaurant": None
                }
            ]
        
        # 5. Calculate processing time
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time

        # Log perf summary
        logger.info(
            "Perf summary | total_ms=%s intent_ms=%s multimodal_ms=%s engine_ms=%s format_ms=%s timeouts=%s sizes=%s",
            round(processing_time * 1000.0, 2),
            phase_times_ms.get("intent_ms"),
            phase_times_ms.get("multimodal_ms"),
            phase_times_ms.get("engine_ms"),
            phase_times_ms.get("format_ms"),
            timeouts,
            input_sizes,
        )
        
        # 6. Create response
        response = EnhancedRecommendationResponse(
            recommendations=recommendations,
            multimodal_analysis=multimodal_analysis.dict() if multimodal_analysis else None,
            intent_prediction={
                "primary_intent": intent_result.get("primary_intent", "unknown"),
                "confidence": intent_result.get("confidence", 0.5),
                "all_intents": intent_result.get("all_intents", [["unknown", 0.5]])
            },
            user_preferences=None,
            system_performance={
                "phase_times_ms": phase_times_ms,
                "timeouts": timeouts,
                "input_sizes": input_sizes,
            },
            model_version="v1.0",
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced recommendation failed: {e}")
        
        # Log error if logging is available
        if query_id and 'log_error_sync' in locals():
            try:
                import traceback
                log_error_sync(
                    query_id=query_id,
                    error_type="recommendation_failure",
                    error_message=str(e),
                    stack_trace=traceback.format_exc()
                )
            except Exception as log_error:
                logger.error(f"Failed to log error: {log_error}")
        
        # Return fallback response
        return EnhancedRecommendationResponse(
            recommendations=[
                {
                    "food_name": "fallback food",
                    "food_category": "FALLBACK",
                    "food_region": "Global",
                    "food_culture": "Universal",
                    "food_tags": ["fallback"],
                    "score": 0.1,
                    "mood_match": 0.1,
                    "context_match": 0.1,
                    "personalization_score": 0.1,
                    "reasoning": [f"System error: {str(e)}"],
                    "restaurant": None
                }
            ],
            multimodal_analysis=None,
            intent_prediction={
                "primary_intent": "error",
                "confidence": 0.1,
                "all_intents": [["error", 0.1]]
            },
            user_preferences=None,
            system_performance={"error": str(e)},
            model_version="v1.0",
            processing_time=asyncio.get_event_loop().time() - start_time
        )

@app.get("/phase3-analysis")
async def phase3_analysis():
    """Get Phase 3 system analysis."""
    return {
        "status": "operational",
        "features": {
            "enhanced_intent_classification": enhanced_classifier is not None,
            "multi_modal_processing": multimodal_processor is not None,
            "real_time_learning": learning_system is not None,
            "advanced_recommendations": recommendation_engine is not None,
            "phase3_manager": phase3_manager is not None
        },
        "capabilities": [
            "Transformer-based intent classification",
            "Multi-modal input processing (text, image, audio)",
            "Semantic embeddings for food-mood relationships",
            "Real-time learning from user feedback",
            "Advanced recommendation algorithms"
        ]
    }

@app.get("/phase3-status")
async def phase3_status():
    """Get detailed Phase 3 component status."""
    return {
        "enhanced_classifier": {
            "status": "operational" if enhanced_classifier else "not_available",
            "capabilities": ["intent_classification", "semantic_embeddings", "fallback_support"] if enhanced_classifier else []
        },
        "multimodal_processor": {
            "status": "operational" if multimodal_processor else "not_available",
            "capabilities": ["text_processing", "image_analysis", "audio_processing", "timeout_protection"] if multimodal_processor else []
        },
        "learning_system": {
            "status": "operational" if learning_system else "not_available",
            "capabilities": ["feedback_processing", "model_updates", "performance_tracking"] if learning_system else []
        },
        "recommendation_engine": {
            "status": "operational" if recommendation_engine else "not_available",
            "capabilities": ["mood_based_recommendations", "context_awareness", "personalization"] if recommendation_engine else []
        },
        "phase3_manager": {
            "status": "operational" if phase3_manager else "not_available",
            "capabilities": ["orchestration", "semantic_search", "context_analysis"] if phase3_manager else []
        }
    }

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    try:
        frontend_path = Path(__file__).parent.parent / "frontend" / "enhanced_ui.html"
        if frontend_path.exists():
            return FileResponse(str(frontend_path))
        else:
            return {"message": "Frontend not found", "path": str(frontend_path)}
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return {"error": "Failed to serve frontend"}

@app.get("/ui")
async def serve_ui():
    """Alternative route to serve the frontend."""
    return await serve_frontend()

@app.get("/examples")
async def get_examples():
    """Get example inputs for the frontend."""
    return {
        "examples": {
            "mood_based": [
                "I'm feeling stressed and need comfort food",
                "I want something energizing for my workout",
                "I'm in a romantic mood, suggest something elegant",
                "I need something healthy and light for lunch"
            ],
            "cuisine_preferences": [
                "I want spicy Indian food",
                "I'm craving Italian pasta",
                "Show me some Japanese sushi options",
                "I want authentic Mexican food"
            ],
            "dietary_restrictions": [
                "I'm vegetarian, suggest healthy options",
                "I need gluten-free food recommendations",
                "I'm on a low-carb diet",
                "I want vegan comfort food"
            ],
            "occasions": [
                "I need food for a business lunch",
                "I want something special for date night",
                "I need quick food for a busy day",
                "I want something festive for a celebration"
            ]
        }
    }

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for real-time learning."""
    try:
        if learning_system is None:
            return {"status": "learning_system_unavailable"}

        # Build context and record
        session_id = feedback.session_id or "web-ui"
        context = feedback.context or {}

        await asyncio.to_thread(
            learning_system.record_feedback,
            feedback.user_id,
            session_id,
            feedback.input_text or "",
            feedback.recommended_foods,
            feedback.selected_food,
            feedback.rating,
            feedback.feedback_text,
            context,
        )

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/recent-feedback-weather")
async def get_recent_feedback_weather(request: RecentFeedbackWeatherRequest):
    """Get the most recent weather from feedback history for a given user and time of day."""
    try:
        if learning_system is None:
            return {"weather": "unknown", "status": "learning_system_unavailable"}

        # Get recent feedback with weather context
        recent_feedback = await asyncio.to_thread(
            learning_system.get_recent_feedback_with_weather,
            request.user_id,
            request.time_of_day
        )

        if recent_feedback and recent_feedback.get('weather'):
            return {
                "weather": recent_feedback['weather'],
                "status": "ok",
                "timestamp": recent_feedback.get('timestamp'),
                "source": "feedback_history"
            }
        else:
            return {
                "weather": "unknown",
                "status": "no_weather_data",
                "message": "No recent weather data found in feedback history"
            }

    except Exception as e:
        logger.error(f"Recent feedback weather fetch failed: {e}")
        return {
            "weather": "unknown",
            "status": "error",
            "message": str(e)
        }

@app.get("/logging/stats")
async def get_logging_statistics():
    """Get comprehensive logging statistics for monitoring and analysis."""
    try:
        from core.logging.query_logger import query_logger
        
        stats = query_logger.get_statistics()
        return {
            "status": "ok",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        return {
            "status": "logging_unavailable",
            "message": "Query logging system not available"
        }
    except Exception as e:
        logger.error(f"Failed to get logging statistics: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/logging/export-training")
async def export_training_dataset(output_file: str = "data/training_dataset.jsonl"):
    """Export high-quality logged queries for training dataset."""
    try:
        from core.logging.query_logger import query_logger
        
        count = query_logger.export_for_training(output_file)
        return {
            "status": "ok",
            "exported_count": count,
            "output_file": output_file,
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        return {
            "status": "logging_unavailable",
            "message": "Query logging system not available"
        }
    except Exception as e:
        logger.error(f"Failed to export training dataset: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/logging/query/{query_id}")
async def get_query_log(query_id: str):
    """Get detailed log entry for a specific query."""
    try:
        from core.logging.query_logger import query_logger
        
        # Read the log file to find the specific query
        import json
        from pathlib import Path
        
        log_file = Path("data/logs/query_logs.jsonl")
        if not log_file.exists():
            return {"status": "not_found", "message": "Log file not found"}
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get('query_id') == query_id:
                        return {
                            "status": "ok",
                            "query": entry,
                            "timestamp": datetime.now().isoformat()
                        }
        
        return {"status": "not_found", "message": f"Query ID {query_id} not found"}
        
    except ImportError:
        return {
            "status": "logging_unavailable",
            "message": "Query logging system not available"
        }
    except Exception as e:
        logger.error(f"Failed to get query log: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
