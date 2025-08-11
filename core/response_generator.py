class ResponseGenerator:
    def __init__(self):
        self.templates = self._load_response_templates()
        self.personality_settings = self._load_personality_config()
        
    def generate_response(self, recommendations, user_context, conversation_history):
        response = {
            "primary_recommendations": [],
            "explanations": [],
            "alternatives": [],
            "conversation_flow": ""
        }
        
        # Generate primary recommendations with explanations
        for rec in recommendations[:3]:  # Top 3 recommendations
            explanation = self.generate_explanation(rec, user_context)
            response["primary_recommendations"].append({
                "food_item": rec,
                "explanation": explanation,
                "confidence": rec.confidence_score
            })
        
        # Generate conversational response
        response["conversation_flow"] = self.generate_conversational_text(
            recommendations, user_context, conversation_history
        )
        
        return response
    
    def generate_explanation(self, recommendation, context):
        explanation_factors = []
        
        if recommendation.mood_match_score > 0.8:
            explanation_factors.append(f"This matches your {context.mood} mood perfectly")
        
        if recommendation.cultural_alignment > 0.7:
            explanation_factors.append(f"It's a great {context.user_culture} comfort food")
        
        if recommendation.seasonal_relevance > 0.7:
            explanation_factors.append(f"Perfect for this {context.season} weather")
        
        return ". ".join(explanation_factors)