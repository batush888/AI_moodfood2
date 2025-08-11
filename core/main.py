from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mood-Based Food Recommendation API")

class FoodRequest(BaseModel):
    user_input: str
    user_id: Optional[str] = None
    context: Optional[Dict] = None

class FoodRecommendation(BaseModel):
    food_name: str
    cuisine_type: str
    mood_match_score: float
    explanation: str
    preparation_time: Optional[int]
    difficulty_level: str
    ingredients: List[str]
    nutritional_info: Optional[Dict]

@app.post("/recommend", response_model=List[FoodRecommendation])
async def get_food_recommendations(request: FoodRequest):
    try:
        # Process user input through NLU
        nlu_result = nlu_engine.process(request.user_input)
        
        # Extract mood and context
        mood = mood_mapper.extract_mood(nlu_result)
        context = context_processor.process(request.context)
        
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(
            mood, context, request.user_id
        )
        
        # Generate response
        response = response_generator.generate_response(
            recommendations, context
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
