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
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
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
    # Import Redis-backed global filter stats and hybrid filter
    from core.filtering.global_filter import get_global_filter_live_stats, update_global_filter_stats, reset_global_filter_stats
    from core.filtering.hybrid_filter import HybridFilter, HybridFilterResponse
    from core.filtering.llm_validator import LLMValidator
    CORE_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Core imports failed: {e}")
    CORE_IMPORTS_AVAILABLE = False

# Import proxy routes for secure API key handling
try:
    from api.proxy_routes import router as proxy_router
    PROXY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Proxy routes import failed: {e}")
    PROXY_AVAILABLE = False

# Add these imports at the top of the file
from core.utils.tracing import start_trace, add_trace_event, end_trace, get_trace_summary
from core.monitoring.drift_monitor import get_drift_summary, start_drift_monitoring
from core.monitoring.feedback_system import get_feedback_system, record_explicit_feedback, record_implicit_feedback
from core.monitoring.metrics import get_metrics_manager

# Pydantic models for API
class UserContext(BaseModel):
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    mood: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = None
    location: Optional[str] = None
    occasion: Optional[str] = None

class EnhancedRecommendationRequest(BaseModel):
    # Keep original field for backward compatibility
    text_input: Optional[str] = Field(None, description="Text input for mood analysis")
    # Accept alternative key used by the frontend
    query: Optional[str] = Field(None, description="Alternative text input field")
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
    
    # Include proxy routes for secure API key handling
    if PROXY_AVAILABLE:
        app.include_router(proxy_router)
        logger.info("Proxy routes included for secure API key handling")
    else:
        logger.warning("Proxy routes not available - API keys may be exposed")
else:
    app = None
    logger.error("FastAPI not available - API routes cannot be created")

# Global component instances
enhanced_classifier = None
multimodal_processor = None
learning_system = None
recommendation_engine = None
phase3_manager = None
hybrid_filter = None

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
    global enhanced_classifier, multimodal_processor, learning_system, recommendation_engine, phase3_manager, hybrid_filter
    
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
    
    # Initialize hybrid filter
    hybrid_filter = await initialize_component(
        "hybrid_filter",
        lambda: HybridFilter(
            llm_validator=LLMValidator(),
            ml_classifier=enhanced_classifier,
            confidence_threshold=0.5  # Lowered from 0.7 to 0.5 to allow ML model usage
        ) if CORE_IMPORTS_AVAILABLE else None,
        lambda: None,
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
        phase3_manager is not None,
        hybrid_filter is not None
    ])
    
    logger.info(f"System initialization complete: {operational_count}/6 components operational")
    
    # Start metrics server
    try:
        from core.monitoring.metrics import start_metrics_server
        metrics_started = start_metrics_server(port=9189)
        if metrics_started:
            logger.info("✅ Prometheus metrics server started on port 9189")
        else:
            logger.warning("⚠️ Prometheus metrics server failed to start")
    except Exception as e:
        logger.warning(f"⚠️ Could not start metrics server: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    operational_count = sum([
        enhanced_classifier is not None,
        multimodal_processor is not None,
        learning_system is not None,
        recommendation_engine is not None,
        phase3_manager is not None,
        hybrid_filter is not None
    ])
    
    return {
        "status": "healthy" if operational_count > 0 else "degraded",
        "components": {
            "enhanced_classifier": "operational" if enhanced_classifier else "not_available",
            "multimodal_processor": "operational" if multimodal_processor else "not_available",
            "learning_system": "operational" if learning_system else "not_available",
            "recommendation_engine": "operational" if recommendation_engine else "not_available",
            "phase3_manager": "operational" if phase3_manager else "not_available",
            "hybrid_filter": "operational" if hybrid_filter else "not_available"
        },
        "component_status": {
            "enhanced_classifier": enhanced_classifier is not None,
            "multimodal_processor": multimodal_processor is not None,
            "learning_system": learning_system is not None,
            "recommendation_engine": recommendation_engine is not None,
            "phase3_manager": phase3_manager is not None,
            "hybrid_filter": hybrid_filter is not None
        },
        "operational_count": operational_count,
        "total_components": 6,
        "timestamp": "2025-08-14T14:12:19"
    }

@app.post("/enhanced-recommend", response_model=EnhancedRecommendationResponse)
async def enhanced_recommend(request: EnhancedRecommendationRequest):
    """Enhanced recommendation endpoint with multi-modal support."""
    import asyncio
    
    # Normalize text from either field; prefer text_input if both present
    raw_text_input = request.text_input if request.text_input else None
    raw_query = request.query if request.query else None
    text = (raw_text_input or raw_query or "").strip()
    
    if not text:
        # Fail fast with a helpful message before touching the pipeline
        raise HTTPException(
            status_code=422,
            detail="Either 'text_input' or 'query' is required."
        )
    
    start_time = asyncio.get_event_loop().time()
    session_id = "2a53f226-5bbc-4149-8571-57e3ddff668f"  # Simplified session handling
    
    # Initialize comprehensive logging
    try:
        from core.logging.query_logger import log_query_sync, log_intent_results_sync, log_recommendations_sync, log_error_sync
        
        # Log incoming query
        query_id = log_query_sync(
            text_input=text,
            image_input=request.image_base64,
            audio_input=request.audio_base64,
            user_context=request.user_context.model_dump() if request.user_context else None,
            session_id=session_id
        )
        logger.info(f"Logged query {query_id} for session {session_id}")
        
    except ImportError:
        logger.warning("Query logging not available, continuing without logging")
        query_id = None
    
    logger.info(f"Processing enhanced recommendation request: session={session_id}, text_length={len(text)}")
    
    try:
        # Perf tracing containers
        phase_times_ms: Dict[str, float] = {}
        timeouts: Dict[str, bool] = {"intent": False, "multimodal": False, "engine": False}
        input_sizes: Dict[str, Any] = {
            "text_len": len(text),
            "image_b64_len": len(request.image_base64) if request.image_base64 else 0,
            "audio_b64_len": len(request.audio_base64) if request.audio_base64 else 0,
        }
        # 1. Hybrid Filter Processing (LLM-as-Teacher)
        logger.info("Processing query through hybrid filter (LLM-as-Teacher)...")
        hybrid_result = None
        
        if hybrid_filter:
            try:
                _t0 = asyncio.get_event_loop().time()
                
                # Process query through hybrid filter
                hybrid_result = await hybrid_filter.process_query(
                    user_query=text,
                    user_context=request.user_context.model_dump() if request.user_context else None
                )
                
                phase_times_ms["intent_ms"] = round((asyncio.get_event_loop().time() - _t0) * 1000.0, 2)
                
                logger.info(f"Hybrid filter processing completed: {hybrid_result.decision}")
                logger.info(f"Recommendations: {len(hybrid_result.recommendations)} items")
                logger.info(f"Reasoning: {hybrid_result.reasoning}")
                
                # Extract intent information for compatibility
                if hybrid_result.decision == "ml_validated" and hybrid_result.ml_prediction:
                    # Use ML prediction when available
                    intent_result = {
                        "primary_intent": hybrid_result.ml_prediction.primary_intent,
                        "confidence": hybrid_result.ml_prediction.confidence,
                        "all_intents": hybrid_result.ml_prediction.all_intents,
                        "method": f"hybrid_{hybrid_result.decision}",
                        "hybrid_result": hybrid_result
                    }
                    logger.info(f"Using ML prediction: {hybrid_result.ml_prediction.primary_intent} (confidence: {hybrid_result.ml_prediction.confidence})")
                elif hybrid_result.llm_interpretation:
                    # Fallback to LLM interpretation
                    intent_result = {
                        "primary_intent": hybrid_result.llm_interpretation.intent,
                        "confidence": hybrid_result.llm_interpretation.confidence,
                        "all_intents": [[hybrid_result.llm_interpretation.intent, hybrid_result.llm_interpretation.confidence]],
                        "method": f"hybrid_{hybrid_result.decision}",
                        "hybrid_result": hybrid_result
                    }
                    logger.info(f"Using LLM interpretation: {hybrid_result.llm_interpretation.intent}")
                else:
                    # Final fallback
                    intent_result = {
                        "primary_intent": "fallback",
                        "confidence": 0.5,
                        "all_intents": [["fallback", 0.5]],
                        "method": "hybrid_fallback",
                        "hybrid_result": hybrid_result
                    }
                    logger.info("Using fallback intent classification")
                
                # Log hybrid filter results
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
                logger.warning("Hybrid filter processing timed out, using fallback")
                timeouts["intent"] = True
                intent_result = {"primary_intent": "fallback", "confidence": 0.5, "all_intents": [["fallback", 0.5]]}
            except Exception as e:
                logger.error(f"Hybrid filter processing failed: {e}")
                intent_result = {"primary_intent": "error", "confidence": 0.1, "all_intents": [["error", 0.1]]}
        else:
            # Fallback to enhanced classifier if hybrid filter not available
            logger.info("Hybrid filter not available, using enhanced classifier fallback...")
            if enhanced_classifier:
                try:
                    _t0 = asyncio.get_event_loop().time()
                    intent_result = await asyncio.wait_for(
                        asyncio.to_thread(enhanced_classifier.classify_intent, text),
                        timeout=10.0
                    )
                    phase_times_ms["intent_ms"] = round((asyncio.get_event_loop().time() - _t0) * 1000.0, 2)
                    logger.info(f"Fallback intent classification completed: {intent_result.get('primary_intent', 'unknown')}")
                except Exception as e:
                    logger.error(f"Fallback intent classification failed: {e}")
                    intent_result = {"primary_intent": "fallback", "confidence": 0.5, "all_intents": [["fallback", 0.5]]}
            else:
                # Final fallback
                intent_result = {"primary_intent": "fallback", "confidence": 0.5, "all_intents": [["fallback", 0.5]]}
        
        # 2. Multi-modal Analysis
        multimodal_analysis = None
        if multimodal_processor and (request.image_base64 or request.audio_base64):
            try:
                _t0 = asyncio.get_event_loop().time()
                multimodal_analysis = await asyncio.wait_for(
                    asyncio.to_thread(
                        multimodal_processor.process_multimodal,
                        text=text,
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
        logger.info("Getting recommendations...")
        recommendations = []
        
        # Check if we have hybrid filter results
        if hybrid_result and hybrid_result.recommendations:
            logger.info("Using recommendations from hybrid filter")
            # Convert hybrid filter recommendations to the expected format
            formatted_recommendations = []
            for i, food_name in enumerate(hybrid_result.recommendations):
                formatted_rec = {
                    "food_name": food_name,
                    "food_category": "hybrid_generated",
                    "food_region": "unknown",
                    "food_culture": "unknown",
                    "food_tags": [],
                    "score": 1.0 - (i * 0.1),  # Decreasing score for ranking
                    "mood_match": 1.0,
                    "context_match": 1.0,
                    "personalization_score": 0.8,
                    "reasoning": hybrid_result.reasoning,
                    "restaurant": None
                }
                formatted_recommendations.append(formatted_rec)
            
            recommendations = formatted_recommendations
            phase_times_ms["engine_ms"] = 0  # No engine processing time
            logger.info(f"Generated {len(recommendations)} recommendations from hybrid filter")
            
        elif recommendation_engine:
            try:
                logger.info("Calling recommendation engine...")
                _t0 = asyncio.get_event_loop().time()
                
                # Pass enhanced intent results to recommendation engine
                enhanced_context = request.user_context.model_dump() if request.user_context else {}
                enhanced_context['text_input'] = text  # Add text input for filtering
                if intent_result and intent_result.get('primary_intent'):
                    enhanced_context['enhanced_intent'] = intent_result.get('primary_intent')
                    enhanced_context['all_intents'] = intent_result.get('all_intents', [])
                    enhanced_context['intent_confidence'] = intent_result.get('confidence', 0.5)
                
                recommendations = await asyncio.wait_for(
                    asyncio.to_thread(
                        recommendation_engine.get_recommendations,
                        text,
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
        frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
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

@app.get("/monitoring")
async def serve_monitoring():
    """Serve the monitoring dashboard."""
    try:
        monitoring_path = Path(__file__).parent.parent / "frontend" / "monitoring.html"
        if monitoring_path.exists():
            return FileResponse(str(monitoring_path))
        else:
            return {"message": "Monitoring dashboard not found", "path": str(monitoring_path)}
    except Exception as e:
        logger.error(f"Error serving monitoring dashboard: {e}")
        return {"error": "Failed to serve monitoring dashboard"}

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

@app.get("/logging/filter-stats")
async def get_filter_statistics():
    """Get hybrid filter statistics for monitoring data quality using Redis-backed global filter stats."""
    try:
        # Primary source: Redis-backed global filter stats
        try:
            # Get live stats from the Redis-backed global hybrid filter
            live_stats = get_global_filter_live_stats()
            
            # Always return live stats if available, even if 0
            # This includes source information (redis_global_filter or local_fallback)
            return live_stats
        except Exception as e:
            logger.warning(f"Redis-backed global filter stats not available: {e}")
        
        # Secondary source: Try to get stats from the hybrid filter instance
        try:
            if hybrid_filter:
                hybrid_stats = hybrid_filter.get_live_stats()
                if hybrid_stats and hybrid_stats.get('total_queries', 0) > 0:
                    return {
                        "timestamp": hybrid_stats.get('timestamp', datetime.now().isoformat()),
                        "total_samples": hybrid_stats.get('total_queries', 0),
                        "ml_confident": hybrid_stats.get('ml_validated', 0),
                        "llm_fallback": hybrid_stats.get('llm_fallback', 0),
                        "rejected": hybrid_stats.get('processing_errors', 0),
                        "llm_training_samples": hybrid_stats.get('llm_training_samples', 0),
                        "ml_success_rate": hybrid_stats.get('ml_success_rate', 0.0),
                        "source": "hybrid_filter_live"
                    }
        except Exception as e:
            logger.warning(f"Hybrid filter stats not available: {e}")
        
        # Fallback: Try to get stats from the most recent retrain report
        retrain_history_file = Path("data/logs/retrain_history.jsonl")
        
        if retrain_history_file.exists():
            # Read the most recent retrain entry
            with open(retrain_history_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    # Get the last line (most recent)
                    last_line = lines[-1].strip()
                    if last_line:
                        try:
                            latest_report = json.loads(last_line)
                            filter_stats = latest_report.get('filter_stats', {})
                            
                            if filter_stats:
                                # Extract the required fields
                                total_samples = filter_stats.get('original_count', 0)
                                ml_confident = filter_stats.get('ml_confident', 0)
                                llm_fallback = filter_stats.get('llm_validated', 0)
                                rejected = filter_stats.get('rejected', 0)
                                
                                # If we have the specific fields we need
                                if 'ml_confident' in filter_stats and 'llm_validated' in filter_stats:
                                    return {
                                        "timestamp": latest_report.get('timestamp', datetime.now().isoformat()),
                                        "total_samples": total_samples,
                                        "ml_confident": ml_confident,
                                        "llm_fallback": llm_fallback,
                                        "rejected": rejected,
                                        "source": "latest_retrain"
                                    }
                        except json.JSONDecodeError:
                            pass
        
        # Final fallback: Try to get stats from the retrainer's hybrid filter
        try:
            from scripts.retrain_classifier import AutomatedRetrainer
            retrainer = AutomatedRetrainer()
            if hasattr(retrainer, 'hybrid_filter') and retrainer.hybrid_filter:
                filter_stats = retrainer.hybrid_filter.get_filter_summary()
                if filter_stats:
                    return {
                        "timestamp": datetime.now().isoformat(),
                        "total_samples": filter_stats.get('total_samples', 0),
                        "ml_confident": filter_stats.get('ml_confident', 0),
                        "llm_fallback": filter_stats.get('llm_fallback', 0),
                        "rejected": filter_stats.get('rejected', 0),
                        "source": "retrainer_filter"
                    }
        except Exception:
            pass
        
        # No stats available
        return {
            "timestamp": datetime.now().isoformat(),
            "total_samples": 0,
            "ml_confident": 0,
            "llm_fallback": 0,
            "rejected": 0,
            "llm_training_samples": 0,
            "ml_success_rate": 0.0,
            "source": "no_stats_available",
            "note": "no stats yet"
        }
        
    except Exception as e:
        logger.error(f"Failed to get filter statistics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "total_samples": 0,
            "ml_confident": 0,
            "llm_fallback": 0,
            "rejected": 0,
            "llm_training_samples": 0,
            "ml_success_rate": 0.0,
            "source": "error",
            "note": "error occurred",
            "error": str(e)
        }

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the ML classifier and hot reload it."""
    try:
        logger.info("🚀 Starting model retraining...")
        
        # Import retraining components
        try:
            from scripts.retrain_classifier import AutomatedRetrainer
        except ImportError as e:
            logger.error(f"Failed to import retraining components: {e}")
            return {
                "status": "error",
                "message": "Retraining components not available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Run retraining in background
        async def run_retraining():
            try:
                retrainer = AutomatedRetrainer()
                success = retrainer.retrain(force=True)
                
                if success:
                    # Check if model was actually deployed (performance comparison passed)
                    if hasattr(retrainer, 'performance_comparison') and retrainer.performance_comparison.get('should_deploy', False):
                        # Hot reload the ML classifier
                        if enhanced_classifier:
                            reload_success = enhanced_classifier.reload_ml_classifier()
                            if reload_success:
                                logger.info("✅ Model retrained and hot reloaded successfully")
                                logger.info(f"Performance: {retrainer.performance_comparison.get('reason', 'Unknown')}")
                            else:
                                logger.error("❌ Model retrained but hot reload failed")
                        else:
                            logger.warning("Enhanced classifier not available for hot reload")
                    else:
                        logger.info("✅ Retraining completed but model not deployed due to performance degradation")
                        logger.info(f"Reason: {retrainer.performance_comparison.get('reason', 'Unknown')}")
                else:
                    logger.error("❌ Model retraining failed")
                    
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
        
        # Add retraining task to background
        background_tasks.add_task(run_retraining)
        
        return {
            "status": "started",
            "message": "Model retraining started in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start retraining: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/retrain/status")
async def get_retrain_status():
    """Get retraining status and recommendations."""
    try:
        from scripts.retrain_classifier import AutomatedRetrainer
        
        retrainer = AutomatedRetrainer()
        status = retrainer.get_retrain_status()
        
        # Add current model status
        model_status = {
            "ml_classifier_loaded": enhanced_classifier.ml_classifier is not None if enhanced_classifier else False,
            "ml_labels_count": len(enhanced_classifier.ml_labels) if enhanced_classifier and enhanced_classifier.ml_labels else 0,
            "transformer_loaded": enhanced_classifier.model is not None if enhanced_classifier else False
        }
        
        status["current_model_status"] = model_status
        
        return {
            "status": "ok",
            "retrain_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get retrain status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/retrain/metrics")
async def get_model_metrics():
    """Get current model performance metrics."""
    try:
        import json
        from pathlib import Path
        
        metrics_file = Path("models/intent_classifier/metrics.json")
        if not metrics_file.exists():
            return {
                "status": "not_found",
                "message": "No metrics file found",
                "timestamp": datetime.now().isoformat()
            }
        
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        return {
            "status": "ok",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/model/status")
async def get_model_status():
    """Get comprehensive model status including version, metrics, and retraining info."""
    try:
        import json
        from pathlib import Path
        
        # Get metrics
        metrics_file = Path("models/intent_classifier/metrics.json")
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        
        # Get retraining status
        from scripts.retrain_classifier import AutomatedRetrainer
        retrainer = AutomatedRetrainer()
        retrain_status = retrainer.get_retrain_status()
        
        # Get dataset info
        dataset_file = Path("data/logs/training_dataset.jsonl")
        dataset_size = 0
        if dataset_file.exists():
            with open(dataset_file, 'r') as f:
                dataset_size = sum(1 for line in f)
        
        # Get model file info
        model_file = Path("models/intent_classifier/ml_classifier.pkl")
        model_info = {
            "exists": model_file.exists(),
            "size_mb": model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0,
            "last_modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat() if model_file.exists() else None
        }
        
        # Get enhanced classifier status
        classifier_status = {
            "ml_classifier_loaded": enhanced_classifier.ml_classifier is not None if enhanced_classifier else False,
            "ml_labels_count": len(enhanced_classifier.ml_labels) if enhanced_classifier and enhanced_classifier.ml_labels else 0,
            "transformer_loaded": enhanced_classifier.model is not None if enhanced_classifier else False,
            "fallback_mode": enhanced_classifier.fallback_mode if enhanced_classifier else True
        }
        
        # Get scheduler status
        try:
            from scripts.automated_scheduler import get_scheduler_status
            scheduler_status = get_scheduler_status()
        except:
            scheduler_status = {"status": "not_available"}
        
        return {
            "status": "ok",
            "model_info": {
                "version": metrics.get('training_date', 'unknown'),
                "accuracy": metrics.get('accuracy', 0),
                "f1_macro": metrics.get('f1_macro', 0),
                "f1_weighted": metrics.get('f1_weighted', 0),
                "n_classes": metrics.get('n_classes', 0),
                "n_features": metrics.get('n_features', 0),
                "n_samples": metrics.get('n_samples', 0)
            },
            "retraining_info": {
                "last_retrain": retrain_status.get('last_retrain', 'Never'),
                "next_retrain_recommended": retrain_status.get('next_retrain_recommended', 'Unknown'),
                "retrain_count": retrain_status.get('retrain_count', 0)
            },
            "dataset_info": {
                "total_samples": dataset_size,
                "dataset_file": str(dataset_file),
                "dataset_exists": dataset_file.exists()
            },
            "model_files": model_info,
            "classifier_status": classifier_status,
            "scheduler_status": scheduler_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/retrain")
async def retrain_model():
    """Trigger manual retraining of the ML model."""
    try:
        from core.retraining.retrain_manager import RetrainManager
        from core.retraining.trainer import FlexibleTrainer
        from core.monitoring.retrain_monitor import RetrainMonitor
        
        # Initialize components
        retrain_manager = RetrainManager(buffer_size=100)
        trainer = FlexibleTrainer()
        monitor = RetrainMonitor()
        
        # Update buffer with latest samples
        buffer_stats = retrain_manager.update_buffer()
        
        # Check if retraining should be triggered
        if retrain_manager.should_trigger_retrain():
            logger.info("Retraining threshold met, starting training pipeline...")
            
            # Run training pipeline
            metrics = trainer.run_training_pipeline()
            
            if metrics.model_version != "failed":
                # Log successful retrain
                monitor.log_retrain_event(
                    model_version=metrics.model_version,
                    total_samples=metrics.total_samples,
                    gold_samples=metrics.gold_samples,
                    silver_samples=metrics.silver_samples,
                    gold_ratio=metrics.gold_ratio,
                    training_accuracy=metrics.training_accuracy,
                    validation_accuracy=metrics.validation_accuracy,
                    training_loss=metrics.training_loss,
                    validation_loss=metrics.validation_loss,
                    training_time_seconds=metrics.training_time_seconds,
                    buffer_size=buffer_stats.buffer_size,
                    retrain_trigger="manual",
                    status="success",
                    notes="Manual retrain triggered via API"
                )
                
                return {
                    "status": "success",
                    "message": "Model retrained and reloaded successfully",
                    "model_version": metrics.model_version,
                    "metrics": {
                        "total_samples": metrics.total_samples,
                        "gold_samples": metrics.gold_samples,
                        "silver_samples": metrics.silver_samples,
                        "gold_ratio": metrics.gold_ratio,
                        "training_accuracy": metrics.training_accuracy,
                        "validation_accuracy": metrics.validation_accuracy,
                        "training_time_seconds": metrics.training_time_seconds
                    },
                    "buffer_stats": {
                        "total_samples": buffer_stats.total_samples,
                        "gold_samples": buffer_stats.gold_samples,
                        "silver_samples": buffer_stats.silver_samples,
                        "gold_ratio": buffer_stats.gold_ratio
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Log failed retrain
                monitor.log_retrain_event(
                    model_version="failed",
                    total_samples=0,
                    gold_samples=0,
                    silver_samples=0,
                    gold_ratio=0.0,
                    training_accuracy=0.0,
                    validation_accuracy=0.0,
                    training_loss=float('inf'),
                    validation_loss=float('inf'),
                    training_time_seconds=metrics.training_time_seconds,
                    buffer_size=buffer_stats.buffer_size,
                    retrain_trigger="manual",
                    status="failed",
                    notes="Training pipeline failed"
                )
                
                return {
                    "status": "error",
                    "message": "Training pipeline failed",
                    "timestamp": datetime.now().isoformat()
                }
        else:
            # Log skipped retrain
            monitor.log_retrain_event(
                model_version="skipped",
                total_samples=0,
                gold_samples=0,
                silver_samples=0,
                gold_ratio=0.0,
                training_accuracy=0.0,
                validation_accuracy=0.0,
                training_loss=0.0,
                validation_loss=0.0,
                training_time_seconds=0.0,
                buffer_size=buffer_stats.buffer_size,
                retrain_trigger="manual",
                status="skipped",
                notes=f"Buffer not full: {buffer_stats.total_samples}/{buffer_stats.buffer_size} samples"
            )
            
            return {
                "status": "skipped",
                "message": f"Retraining skipped: buffer not full ({buffer_stats.total_samples}/{buffer_stats.buffer_size} samples)",
                "buffer_stats": {
                    "total_samples": buffer_stats.total_samples,
                    "gold_samples": buffer_stats.gold_samples,
                    "silver_samples": buffer_stats.silver_samples,
                    "gold_ratio": buffer_stats.gold_ratio,
                    "buffer_size": buffer_stats.buffer_size
                },
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        return {
            "status": "error",
            "message": f"Retraining failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Model deployment and versioning endpoints
class DeployRequest(BaseModel):
    version_id: str
    mode: str = "full"  # or "canary"

class RollbackRequest(BaseModel):
    version_id: str

@app.post("/retrain/deploy")
async def deploy_model(req: DeployRequest):
    """Deploy a specific model version."""
    try:
        from scripts.retrain_classifier import AutomatedRetrainer
        
        retrainer = AutomatedRetrainer()
        success = retrainer.deploy_version(req.version_id, mode=req.mode)
        
        if not success:
            raise HTTPException(status_code=500, detail="Deploy failed")
        
        # Record metrics
        try:
            from core.monitoring.metrics import record_deploy_success
            record_deploy_success(req.mode)
        except Exception:
            pass  # Metrics are optional
        
        return {
            "status": "deployed", 
            "version": req.version_id, 
            "mode": req.mode,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deploy failed: {str(e)}")

@app.post("/retrain/rollback")
async def rollback_model(req: RollbackRequest):
    """Rollback to a previous model version."""
    try:
        from scripts.retrain_classifier import AutomatedRetrainer
        
        retrainer = AutomatedRetrainer()
        success = retrainer.rollback_to_version(req.version_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Rollback failed")
        
        # Record metrics
        try:
            from core.monitoring.metrics import record_deploy_success
            record_deploy_success("rollback")
        except Exception:
            pass  # Metrics are optional
        
        return {
            "status": "rolled_back", 
            "version": req.version_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model rollback failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")

@app.get("/retrain/versions")
async def list_versions():
    """List all available model versions."""
    try:
        from scripts.retrain_classifier import AutomatedRetrainer
        
        retrainer = AutomatedRetrainer()
        versions = retrainer.list_versions()
        
        return {
            "status": "ok",
            "versions": versions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list versions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list versions: {str(e)}")

@app.post("/retrain/abtest/start")
async def start_abtest(version_id: str, fraction: float = 0.05):
    """Start an A/B test with a canary version."""
    try:
        from scripts.retrain_classifier import AutomatedRetrainer
        
        retrainer = AutomatedRetrainer()
        success = retrainer.start_abtest(version_id, fraction)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start A/B test")
        
        # Update metrics
        try:
            from core.monitoring.metrics import set_abtest_status
            set_abtest_status(version_id, True)
        except Exception:
            pass  # Metrics are optional
        
        return {
            "status": "ab_started", 
            "version_id": version_id,
            "fraction": fraction,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start A/B test: {str(e)}")

@app.post("/retrain/abtest/stop")
async def stop_abtest():
    """Stop the currently running A/B test."""
    try:
        from scripts.retrain_classifier import AutomatedRetrainer
        
        retrainer = AutomatedRetrainer()
        success = retrainer.stop_abtest()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to stop A/B test")
        
        # Update metrics
        try:
            from core.monitoring.metrics import set_abtest_status
            # Clear all A/B test statuses
            from core.monitoring.metrics import get_metrics_manager
            manager = get_metrics_manager()
            # This is a simplified approach - in production you'd want to track specific versions
            manager.set_abtest_status("", False)
        except Exception:
            pass  # Metrics are optional
        
        return {
            "status": "ab_stopped",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop A/B test: {str(e)}")

@app.get("/retrain/abtest/status")
async def get_abtest_status():
    """Get current A/B test status."""
    try:
        from scripts.retrain_classifier import AutomatedRetrainer
        
        retrainer = AutomatedRetrainer()
        status = retrainer.get_abtest_status()
        
        return {
            "status": "ok",
            "abtest": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get A/B test status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get A/B test status: {str(e)}")

@app.get("/retrain/status")
async def get_retrain_status():
    """Get retraining pipeline status and buffer information."""
    try:
        from core.retraining.retrain_manager import RetrainManager
        from core.retraining.trainer import FlexibleTrainer
        from core.monitoring.retrain_monitor import RetrainMonitor
        
        # Initialize components
        retrain_manager = RetrainManager()
        trainer = FlexibleTrainer()
        monitor = RetrainMonitor()
        
        # Get buffer stats
        buffer_stats = retrain_manager.get_buffer_stats()
        
        # Get training summary
        training_summary = trainer.get_training_summary()
        
        # Get metrics summary
        metrics_summary = monitor.get_metrics_summary()
        
        # Get performance trends
        performance_trends = monitor.get_performance_trends(window_days=30)
        
        return {
            "status": "ok",
            "buffer_status": {
                "total_samples": buffer_stats.total_samples,
                "gold_samples": buffer_stats.gold_samples,
                "silver_samples": buffer_stats.silver_samples,
                "gold_ratio": buffer_stats.gold_ratio,
                "buffer_size": buffer_stats.buffer_size,
                "last_updated": buffer_stats.last_updated,
                "should_retrain": retrain_manager.should_trigger_retrain()
            },
            "training_status": training_summary,
            "metrics_summary": metrics_summary,
            "performance_trends": performance_trends,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get retrain status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get automated scheduler status."""
    try:
        from scripts.automated_scheduler import get_scheduler_status
        status = get_scheduler_status()
        
        return {
            "status": "ok",
            "scheduler": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get scheduler status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/scheduler/start")
async def start_scheduler():
    """Start the automated retraining scheduler."""
    try:
        from scripts.automated_scheduler import start_automated_scheduler
        
        success = start_automated_scheduler()
        
        return {
            "status": "ok" if success else "error",
            "message": "Scheduler started successfully" if success else "Failed to start scheduler",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the automated retraining scheduler."""
    try:
        from scripts.automated_scheduler import stop_automated_scheduler
        
        stop_automated_scheduler()
        
        return {
            "status": "ok",
            "message": "Scheduler stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop scheduler: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/trace/{trace_id}")
async def get_trace_details(trace_id: str):
    """Get detailed trace information for a request"""
    try:
        trace_summary = get_trace_summary(trace_id)
        if trace_summary:
            return {"status": "success", "trace": trace_summary}
        else:
            raise HTTPException(status_code=404, detail="Trace not found")
    except Exception as e:
        logger.error(f"Failed to get trace {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trace")

@app.get("/health/full")
async def get_full_health():
    """Comprehensive health check including all system components"""
    try:
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check LLM availability
        try:
            from core.filtering.llm_validator import get_llm_validator
            llm_validator = get_llm_validator()
            llm_status = "healthy" if llm_validator and llm_validator.is_enabled() else "disabled"
            health_status["components"]["llm_validator"] = {
                "status": llm_status,
                "enabled": llm_validator.is_enabled() if llm_validator else False
            }
        except Exception as e:
            health_status["components"]["llm_validator"] = {"status": "error", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        # Check ML model loading
        try:
            from core.nlu.model_loader import get_current_version, is_model_loaded
            model_version = get_current_version()
            model_loaded = is_model_loaded()
            health_status["components"]["ml_model"] = {
                "status": "healthy" if model_loaded else "error",
                "version": model_version,
                "loaded": model_loaded
            }
            if not model_loaded:
                health_status["overall_status"] = "degraded"
        except Exception as e:
            health_status["components"]["ml_model"] = {"status": "error", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        # Check Redis/logging
        try:
            from core.filtering.global_filter import get_global_filter_live_stats
            filter_stats = get_global_filter_live_stats()
            health_status["components"]["redis_filter_stats"] = {
                "status": "healthy",
                "source": filter_stats.get("source", "unknown"),
                "total_samples": filter_stats.get("total_samples", 0)
            }
        except Exception as e:
            health_status["components"]["redis_filter_stats"] = {"status": "error", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        # Check drift monitor status
        try:
            drift_summary = get_drift_summary()
            health_status["components"]["drift_monitor"] = {
                "status": "healthy",
                "current_drift_score": drift_summary.get("current_drift_score", 0),
                "active_alerts": drift_summary.get("active_alerts", 0)
            }
        except Exception as e:
            health_status["components"]["drift_monitor"] = {"status": "error", "error": str(e)}
        
        # Check metrics server
        try:
            metrics_manager = get_metrics_manager()
            health_status["components"]["metrics_server"] = {
                "status": "healthy" if metrics_manager.is_server_running() else "stopped",
                "running": metrics_manager.is_server_running()
            }
        except Exception as e:
            health_status["components"]["metrics_server"] = {"status": "error", "error": str(e)}
        
        # Check file system
        try:
            import os
            critical_paths = ["logs/", "data/", "models/"]
            file_status = {}
            for path in critical_paths:
                if os.path.exists(path):
                    stat = os.stat(path)
                    file_status[path] = {
                        "exists": True,
                        "writable": os.access(path, os.W_OK),
                        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                else:
                    file_status[path] = {"exists": False, "writable": False}
            
            health_status["components"]["file_system"] = {
                "status": "healthy",
                "paths": file_status
            }
        except Exception as e:
            health_status["components"]["file_system"] = {"status": "error", "error": str(e)}
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/feedback/explicit")
async def submit_explicit_feedback(
    request: dict = Body(...)
):
    """Submit explicit user feedback (ratings, comments)"""
    try:
        trace_id = request.get("trace_id")
        model_version = request.get("model_version")
        query = request.get("query")
        recommendations = request.get("recommendations", [])
        user_rating = request.get("user_rating")
        feedback_text = request.get("feedback_text")
        session_id = request.get("session_id")
        user_id = request.get("user_id")
        
        if not all([trace_id, model_version, query, recommendations, user_rating]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        if not (1 <= user_rating <= 5):
            raise HTTPException(status_code=400, detail="User rating must be between 1 and 5")
        
        feedback_id = record_explicit_feedback(
            trace_id=trace_id,
            model_version=model_version,
            query=query,
            recommendations=recommendations,
            user_rating=user_rating,
            feedback_text=feedback_text,
            session_id=session_id,
            user_id=user_id
        )
        
        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Explicit feedback recorded successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to record explicit feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.post("/feedback/implicit")
async def submit_implicit_feedback(
    request: dict = Body(...)
):
    """Submit implicit user feedback (behavioral signals)"""
    try:
        trace_id = request.get("trace_id")
        model_version = request.get("model_version")
        query = request.get("query")
        recommendations = request.get("recommendations", [])
        implicit_signal = request.get("implicit_signal")
        session_id = request.get("session_id")
        user_id = request.get("user_id")
        
        if not all([trace_id, model_version, query, recommendations, implicit_signal]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        valid_signals = ["re_query", "skip", "accept", "modify", "click", "share"]
        if implicit_signal not in valid_signals:
            raise HTTPException(status_code=400, detail=f"Invalid implicit signal. Must be one of: {valid_signals}")
        
        feedback_id = record_implicit_feedback(
            trace_id=trace_id,
            model_version=model_version,
            query=query,
            recommendations=recommendations,
            implicit_signal=implicit_signal,
            session_id=session_id,
            user_id=user_id
        )
        
        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Implicit feedback recorded successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to record implicit feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/feedback/summary")
async def get_feedback_summary(model_version: Optional[str] = None):
    """Get feedback summary for a specific model version or overall"""
    try:
        from core.monitoring.feedback_system import get_feedback_summary as get_feedback_summary_func
        summary = get_feedback_summary_func(model_version)
        return summary
    except Exception as e:
        logger.error(f"Failed to get feedback summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback summary")

@app.get("/drift/summary")
async def get_drift_summary():
    """Get drift detection summary"""
    try:
        summary = get_drift_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get drift summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve drift summary")

@app.get("/metrics/raw")
async def get_raw_metrics():
    """Get raw Prometheus metrics"""
    try:
        metrics_manager = get_metrics_manager()
        raw_metrics = metrics_manager.get_metrics()
        return Response(content=raw_metrics, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to get raw metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
