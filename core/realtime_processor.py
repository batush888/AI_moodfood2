class RealtimeProcessor:
    def __init__(self):
        self.nlu_engine = NLUEngine()
        self.mood_mapper = MoodMapper()
        self.recommendation_engine = RecommendationEngine()
        self.personalization_layer = PersonalizationLayer()
        
    async def process_request(self, user_input, user_context):
        # Parallel processing for speed
        tasks = [
            self.nlu_engine.process_async(user_input),
            self.context_processor.process_async(user_context),
            self.user_profile_loader.load_async(user_context.user_id)
        ]
        
        nlu_result, context, user_profile = await asyncio.gather(*tasks)
        
        # Sequential processing for accuracy
        mood = self.mood_mapper.extract_mood(nlu_result, context)
        base_recommendations = self.recommendation_engine.get_base_recommendations(mood)
        personalized_recs = self.personalization_layer.personalize(
            base_recommendations, user_profile
        )
        
        return personalized_recs