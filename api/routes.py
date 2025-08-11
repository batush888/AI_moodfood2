from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Import our core components
from core.mood_mapper import MoodMapper
from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine
from core.nlu.inference import classify_intent

app = FastAPI(
    title="AI Mood-Based Food Recommendation System",
    description="An intelligent system that recommends food based on mood, context, and situational needs",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
mood_mapper = MoodMapper()
recommendation_engine = MoodBasedRecommendationEngine()

# Pydantic models for API requests/responses
class RecommendationRequest(BaseModel):
    user_input: str
    user_context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    top_k: Optional[int] = 10

class RecommendationResponse(BaseModel):
    user_input: str
    recommendations: List[Dict[str, Any]]
    extracted_mood: List[str]
    context_analysis: Dict[str, Any]
    timestamp: str

class FeedbackRequest(BaseModel):
    user_id: str
    food_item_id: str
    rating: float
    feedback: str

class UserPreferencesRequest(BaseModel):
    user_id: str
    preferences: Dict[str, Any]

class MoodAnalysisRequest(BaseModel):
    user_input: str
    context: Optional[Dict[str, Any]] = None

class MoodAnalysisResponse(BaseModel):
    user_input: str
    primary_mood: str
    mood_categories: List[str]
    extracted_entities: List[str]
    confidence_score: float
    context_factors: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "AI Mood-Based Food Recommendation System",
        "version": "2.0.0",
        "description": "Get food recommendations based on your mood and context",
        "endpoints": {
            "/recommend": "Get food recommendations",
            "/analyze-mood": "Analyze user mood and intent",
            "/feedback": "Submit feedback for recommendations",
            "/preferences": "Update user preferences",
            "/taxonomy": "View available mood categories",
            "/health": "System health check"
        }
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized food recommendations based on user input and context.
    
    This endpoint analyzes the user's mood, considers contextual factors,
    and returns ranked food recommendations with restaurant suggestions.
    """
    try:
        # Get recommendations from the engine
        recommendations = recommendation_engine.get_recommendations(
            user_input=request.user_input,
            user_context=request.user_context or {},
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        # Convert recommendations to serializable format
        serialized_recommendations = []
        for rec in recommendations:
            serialized_rec = {
                "food_name": rec.food_item.name,
                "food_category": rec.food_item.category,
                "food_region": rec.food_item.region,
                "food_culture": rec.food_item.culture,
                "food_tags": rec.food_item.tags,
                "restaurant_name": rec.restaurant.name if rec.restaurant else None,
                "restaurant_rating": rec.restaurant.rating if rec.restaurant else None,
                "restaurant_cuisine": rec.restaurant.cuisine_type if rec.restaurant else None,
                "delivery_available": rec.restaurant.delivery_available if rec.restaurant else None,
                "score": round(rec.score, 3),
                "mood_match": round(rec.mood_match, 3),
                "context_match": round(rec.context_match, 3),
                "personalization_score": round(rec.personalization_score, 3),
                "reasoning": rec.reasoning
            }
            serialized_recommendations.append(serialized_rec)
        
        # Analyze the user input to extract mood categories
        mood_categories = recommendation_engine._analyze_user_intent(request.user_input)
        
        # Extract context factors
        context_factors = recommendation_engine._extract_context_factors(
            request.user_context or {}
        )
        
        return RecommendationResponse(
            user_input=request.user_input,
            recommendations=serialized_recommendations,
            extracted_mood=mood_categories,
            context_analysis=context_factors,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.post("/analyze-mood", response_model=MoodAnalysisResponse)
async def analyze_mood(request: MoodAnalysisRequest):
    """
    Analyze user input to understand mood, intent, and extract relevant entities.
    
    This endpoint provides detailed analysis of what the user is looking for
    without generating full recommendations.
    """
    try:
        # Use mood mapper to analyze intent
        mood_analysis = mood_mapper.get_recommendations(
            user_input=request.user_input,
            context=request.context
        )
        
        # Extract mood categories using the recommendation engine
        mood_categories = recommendation_engine._analyze_user_intent(request.user_input)
        
        # Extract entities
        entities = mood_mapper._extract_entities(request.user_input)
        
        # Calculate confidence score (simplified)
        confidence_score = min(1.0, len(mood_categories) * 0.3 + len(entities) * 0.2)
        
        # Get context factors
        context_factors = recommendation_engine._extract_context_factors(
            request.context or {}
        )
        
        return MoodAnalysisResponse(
            user_input=request.user_input,
            primary_mood=mood_categories[0] if mood_categories else "unknown",
            mood_categories=mood_categories,
            extracted_entities=entities,
            confidence_score=round(confidence_score, 3),
            context_factors=context_factors
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing mood: {str(e)}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for food recommendations to improve personalization.
    
    This helps the system learn from user preferences and improve
    future recommendations.
    """
    try:
        recommendation_engine.record_feedback(
            user_id=request.user_id,
            food_item_id=request.food_item_id,
            rating=request.rating,
            feedback=request.feedback
        )
        
        return {
            "message": "Feedback recorded successfully",
            "user_id": request.user_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")

@app.post("/preferences")
async def update_preferences(request: UserPreferencesRequest):
    """
    Update user preferences for personalized recommendations.
    
    Users can specify preferred categories, tags, cultures, and other
    preferences to get more tailored suggestions.
    """
    try:
        recommendation_engine.update_user_preferences(
            user_id=request.user_id,
            preferences=request.preferences
        )
        
        return {
            "message": "Preferences updated successfully",
            "user_id": request.user_id,
            "updated_preferences": request.preferences,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating preferences: {str(e)}")

@app.get("/taxonomy")
async def get_taxonomy():
    """
    Get the complete mood-food taxonomy.
    
    This endpoint provides information about all available mood categories,
    their descriptors, and associated food types.
    """
    try:
        # Load and return the taxonomy
        with open("data/taxonomy/mood_food_taxonomy.json", "r") as f:
            taxonomy = json.load(f)
        
        # Create a summary view
        taxonomy_summary = {}
        for category, data in taxonomy.items():
            taxonomy_summary[category] = {
                "descriptors": data.get("descriptors", []),
                "labels": data.get("labels", []),
                "food_count": len(data.get("foods", [])),
                "example_foods": [food["name"] for food in data.get("foods", [])[:3]]
            }
        
        return {
            "total_categories": len(taxonomy),
            "categories": taxonomy_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading taxonomy: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify system status."""
    try:
        # Basic health checks
        taxonomy_loaded = len(mood_mapper.taxonomy) > 0
        food_items_loaded = len(recommendation_engine.food_items) > 0
        restaurants_loaded = len(recommendation_engine.restaurants) > 0
        
        return {
            "status": "healthy",
            "components": {
                "mood_mapper": "operational" if taxonomy_loaded else "error",
                "recommendation_engine": "operational" if food_items_loaded else "error",
                "restaurant_data": "operational" if restaurants_loaded else "error"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/examples")
async def get_example_queries():
    """
    Get example queries to demonstrate the system's capabilities.
    
    These examples show different types of mood-based food requests
    that users can make.
    """
    examples = {
        "weather_based": [
            "I'm feeling hot and need something refreshing",
            "It's cold outside, give me something warm and comforting",
            "Summer heat calls for something cool and light"
        ],
        "emotional": [
            "I'm feeling sad and need comfort food",
            "I want something romantic for date night",
            "I'm stressed and need something soothing"
        ],
        "energy_based": [
            "I want something light and easy to digest",
            "I need something heavy and filling",
            "Give me something greasy and indulgent"
        ],
        "flavor_based": [
            "I'm craving something sweet",
            "I want something spicy and exciting",
            "I need something salty and satisfying"
        ],
        "occasion_based": [
            "Family dinner, something traditional",
            "Quick lunch break meal",
            "Party snacks for a celebration"
        ],
        "health_based": [
            "I'm sick and need something gentle",
            "I want something healthy and detoxifying",
            "I'm recovering and need nourishment"
        ]
    }
    
    return {
        "examples": examples,
        "total_examples": sum(len(examples) for examples in examples.values()),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 