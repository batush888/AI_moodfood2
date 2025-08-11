class AdaptiveLearning:
    def __init__(self):
        self.user_models = {}
        self.global_patterns = {}
        
    def update_user_model(self, user_id, interaction):
        if user_id not in self.user_models:
            self.user_models[user_id] = UserModel()
        
        model = self.user_models[user_id]
        
        # Update based on explicit feedback
        if interaction.rating:
            model.update_preference_weights(interaction.food_item, interaction.rating)
        
        # Update based on implicit feedback
        if interaction.clicked:
            model.increase_preference(interaction.food_item, 0.1)
        
        if interaction.ordered:
            model.increase_preference(interaction.food_item, 0.3)
        
        # Learn mood patterns
        model.update_mood_food_association(interaction.mood, interaction.food_item)
        
    def get_recommendations(self, user_id, current_mood, context):
        user_model = self.user_models.get(user_id, DefaultUserModel())
        
        # Get base recommendations
        base_recs = self.mood_mapper.get_recommendations(current_mood, context)
        
        # Personalize based on user model
        personalized_recs = user_model.personalize_recommendations(base_recs)
        
        return personalized_recs