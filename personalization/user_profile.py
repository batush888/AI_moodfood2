class UserProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        self.cultural_background = []
        self.dietary_restrictions = []
        self.taste_preferences = {}
        self.mood_patterns = {}
        self.historical_choices = []
        self.health_goals = []
        self.social_preferences = {}
        self.location_data = {}
        
    def update_from_interaction(self, interaction_data):
        # Update preferences based on user feedback
        self.update_taste_preferences(interaction_data.food_ratings)
        self.update_mood_patterns(interaction_data.mood_food_pairs)
        self.update_cultural_preferences(interaction_data.cultural_choices)
        
    def get_personalization_vector(self):
        # Create vector representation for personalization
        return {
            "cultural_weights": self.cultural_background,
            "dietary_constraints": self.dietary_restrictions,
            "taste_profile": self.taste_preferences,
            "mood_history": self.mood_patterns,
            "social_context": self.social_preferences
        }
    

class UserProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # Core Demographics
        self.age_range = None
        self.location = None
        self.cultural_background = []
        
        # Dietary Information
        self.dietary_restrictions = []
        self.allergies = []
        self.health_goals = []
        self.calorie_preferences = None
        
        # Taste Profile (learned over time)
        self.flavor_preferences = {
            "sweet": 0.5,      # Scale 0-1
            "salty": 0.5,
            "spicy": 0.5,
            "umami": 0.5,
            "sour": 0.5,
            "bitter": 0.5
        }
        
        self.texture_preferences = {
            "crunchy": 0.5,
            "creamy": 0.5,
            "chewy": 0.5,
            "smooth": 0.5,
            "crispy": 0.5
        }
        
        # Mood-Food Associations (learned)
        self.mood_food_patterns = {
            "stressed": {},
            "happy": {},
            "tired": {},
            "sad": {},
            "excited": {},
            "comfort": {},
            "energetic": {}
        }
        
        # Temporal Patterns
        self.time_preferences = {
            "breakfast": {"preferred_foods": [], "typical_time": None},
            "lunch": {"preferred_foods": [], "typical_time": None},
            "dinner": {"preferred_foods": [], "typical_time": None},
            "snacks": {"preferred_foods": [], "typical_times": []}
        }
        
        # Social Context Preferences
        self.social_preferences = {
            "alone": {"preferred_foods": [], "portion_preferences": "individual"},
            "couple": {"preferred_foods": [], "sharing_preference": True},
            "family": {"preferred_foods": [], "kid_friendly": False},
            "friends": {"preferred_foods": [], "party_foods": True}
        }
        
        # Historical Data
        self.interaction_history = []
        self.rating_history = []
        self.order_history = []
        self.search_history = []
        
        # Learning Metrics
        self.confidence_scores = {}
        self.prediction_accuracy = 0.0
        self.total_interactions = 0

class ProfileUpdater:
    def __init__(self):
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        
    def update_from_interaction(self, user_profile, interaction):
        """Update user profile based on interaction data"""
        
        # Update taste preferences based on ratings
        if interaction.rating:
            self._update_taste_preferences(user_profile, interaction)
        
        # Update mood-food associations
        if interaction.mood and interaction.food_item:
            self._update_mood_patterns(user_profile, interaction)
        
        # Update temporal patterns
        self._update_temporal_patterns(user_profile, interaction)
        
        # Update social context preferences
        if interaction.social_context:
            self._update_social_preferences(user_profile, interaction)
        
        # Update confidence scores
        self._update_confidence_scores(user_profile, interaction)
        
        user_profile.last_updated = datetime.now()
        user_profile.total_interactions += 1
    
    def _update_taste_preferences(self, profile, interaction):
        """Update flavor and texture preferences based on ratings"""
        food_item = interaction.food_item
        rating = interaction.rating  # Scale 1-5
        
        # Convert rating to preference adjustment (-0.2 to +0.2)
        adjustment = (rating - 3) * 0.1
        
        # Update flavor preferences
        for flavor in food_item.flavor_profile:
            current_pref = profile.flavor_preferences.get(flavor, 0.5)
            new_pref = max(0, min(1, current_pref + adjustment))
            profile.flavor_preferences[flavor] = new_pref
        
        # Update texture preferences
        for texture in food_item.texture_profile:
            current_pref = profile.texture_preferences.get(texture, 0.5)
            new_pref = max(0, min(1, current_pref + adjustment))
            profile.texture_preferences[texture] = new_pref
    
    def _update_mood_patterns(self, profile, interaction):
        """Learn mood-food associations"""
        mood = interaction.mood
        food_item = interaction.food_item
        satisfaction = interaction.satisfaction_rating or 3
        
        if mood not in profile.mood_food_patterns:
            profile.mood_food_patterns[mood] = {}
        
        food_key = f"{food_item.cuisine}_{food_item.category}"
        
        if food_key in profile.mood_food_patterns[mood]:
            # Update existing association
            current_score = profile.mood_food_patterns[mood][food_key]
            new_score = current_score + (satisfaction - 3) * 0.1
            profile.mood_food_patterns[mood][food_key] = max(0, min(5, new_score))
        else:
            # Create new association
            profile.mood_food_patterns[mood][food_key] = satisfaction