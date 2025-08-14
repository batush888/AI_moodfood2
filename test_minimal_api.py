#!/usr/bin/env python3
"""
Minimal test API to isolate issues
"""

import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import the recommendation engine
from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine

app = FastAPI(title="Test API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TestRequest(BaseModel):
    text_input: str
    user_context: Dict[str, Any]

class TestResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint."""
    return {"message": "Test API is working"}

@app.post("/test-recommendations")
async def test_recommendations(request: TestRequest):
    """Test recommendation engine directly."""
    try:
        print(f"Testing with input: {request.text_input}")
        print(f"Context: {request.user_context}")
        
        # Initialize recommendation engine
        engine = MoodBasedRecommendationEngine()
        print("✅ Engine initialized")
        
        # Get recommendations
        recommendations = engine.get_recommendations(
            request.text_input,
            request.user_context,
            top_k=3
        )
        print(f"✅ Got {len(recommendations)} recommendations")
        
        # Try to format the first recommendation
        if recommendations:
            rec = recommendations[0]
            print(f"✅ First recommendation: {rec.food_item.name}")
            
            # Try to access restaurant
            if rec.restaurant:
                print(f"✅ Restaurant: {rec.restaurant.name}")
            else:
                print("✅ No restaurant (this is normal)")
            
            # Try to create a simple response
            response_data = {
                "food_name": rec.food_item.name,
                "food_category": rec.food_item.category,
                "restaurant": rec.restaurant.name if rec.restaurant else None
            }
            
            return TestResponse(
                success=True,
                message=f"Successfully processed {len(recommendations)} recommendations",
                data=response_data
            )
        else:
            return TestResponse(
                success=True,
                message="No recommendations generated",
                data={"count": 0}
            )
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        return TestResponse(
            success=False,
            message=f"Error: {str(e)}",
            data={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
