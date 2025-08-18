import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    MOOD_BASED = "mood_based"
    CONTEXTUAL = "contextual"
    PERSONALIZED = "personalized"
    TRENDING = "trending"

@dataclass
class FoodItem:
    name: str
    category: str
    region: str
    culture: str
    tags: List[str]
    labels: List[str]
    descriptors: List[str]
    nutrition_score: Optional[float] = None
    popularity_score: Optional[float] = None

@dataclass
class Restaurant:
    id: str
    name: str
    cuisine_type: str
    rating: float
    review_count: int
    price_range: str
    location: Dict[str, float]  # lat, lng
    delivery_available: bool
    tags: List[str]
    mood_categories: List[str]

@dataclass
class Recommendation:
    food_item: FoodItem
    restaurant: Optional[Restaurant]
    score: float
    reasoning: List[str]
    mood_match: float
    context_match: float
    personalization_score: float

class MoodBasedRecommendationEngine:
    def __init__(self, taxonomy_path: str = "data/taxonomy/mood_food_taxonomy.json"):
        self.taxonomy = self._load_taxonomy(taxonomy_path)
        self.food_items = self._create_food_items()
        self.restaurants = self._create_sample_restaurants()
        self.user_preferences = {}
        self.feedback_history = {}
        
    def _load_taxonomy(self, path: str) -> Dict:
        """Load the mood-food taxonomy."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _create_food_items(self) -> List[FoodItem]:
        """Create FoodItem objects from taxonomy."""
        food_items = []
        
        for category, data in self.taxonomy.items():
            if 'foods' in data:
                for food in data['foods']:
                    food_item = FoodItem(
                        name=food['name'],
                        category=category,
                        region=food.get('region', 'Unknown'),
                        culture=food.get('culture', 'Unknown'),
                        tags=food.get('tags', []),
                        labels=data.get('labels', []),
                        descriptors=data.get('descriptors', []),
                        nutrition_score=self._calculate_nutrition_score(food.get('tags', [])),
                        popularity_score=self._calculate_popularity_score(food.get('tags', []))
                    )
                    food_items.append(food_item)
        
        return food_items
    
    def _create_sample_restaurants(self) -> List[Restaurant]:
        """Create sample restaurant data for demonstration."""
        return [
            Restaurant(
                id="rest_001",
                name="Comfort Kitchen",
                cuisine_type="American Comfort",
                rating=4.5,
                review_count=1200,
                price_range="$$",
                location={"lat": 40.7128, "lng": -74.0060},
                delivery_available=True,
                tags=["comfort food", "family-friendly", "casual"],
                mood_categories=["EMOTIONAL_COMFORT", "OCCASION_FAMILY_DINNER"]
            ),
            Restaurant(
                id="rest_002",
                name="Spice Garden",
                cuisine_type="Indian",
                rating=4.3,
                review_count=800,
                price_range="$$",
                location={"lat": 40.7589, "lng": -73.9851},
                delivery_available=True,
                tags=["spicy", "aromatic", "authentic"],
                mood_categories=["FLAVOR_SPICY", "EMOTIONAL_ADVENTUROUS"]
            ),
            Restaurant(
                id="rest_003",
                name="Fresh & Light",
                cuisine_type="Health Food",
                rating=4.7,
                review_count=600,
                price_range="$$$",
                location={"lat": 40.7505, "lng": -73.9934},
                delivery_available=False,
                tags=["healthy", "organic", "light"],
                mood_categories=["HEALTH_DETOX", "ENERGY_LIGHT"]
            ),
            Restaurant(
                id="rest_004",
                name="Cozy Corner",
                cuisine_type="Italian",
                rating=4.4,
                review_count=950,
                price_range="$$",
                location={"lat": 40.7614, "lng": -73.9776},
                delivery_available=True,
                tags=["romantic", "intimate", "traditional"],
                mood_categories=["EMOTIONAL_ROMANTIC", "OCCASION_FAMILY_DINNER"]
            ),
            Restaurant(
                id="rest_005",
                name="Quick Bites",
                cuisine_type="Fast Casual",
                rating=4.1,
                review_count=1500,
                price_range="$",
                location={"lat": 40.7484, "lng": -73.9857},
                delivery_available=True,
                tags=["quick", "efficient", "casual"],
                mood_categories=["OCCASION_LUNCH_BREAK", "goal_quick"]
            )
        ]
    
    def _calculate_nutrition_score(self, tags: List[str]) -> float:
        """Calculate nutrition score based on food tags."""
        score = 0.5  # Base score
        
        # Positive nutrition indicators
        positive_tags = ["healthy", "nutritious", "organic", "fresh", "light"]
        for tag in positive_tags:
            if tag in tags:
                score += 0.1
        
        # Negative nutrition indicators
        negative_tags = ["greasy", "fried", "indulgent", "sugary"]
        for tag in negative_tags:
            if tag in tags:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_popularity_score(self, tags: List[str]) -> float:
        """Calculate popularity score based on food tags."""
        score = 0.5  # Base score
        
        # Popular food characteristics
        popular_tags = ["comforting", "satisfying", "familiar", "traditional"]
        for tag in popular_tags:
            if tag in tags:
                score += 0.1
        
        # Niche food characteristics
        niche_tags = ["exotic", "experimental", "sophisticated", "luxurious"]
        for tag in niche_tags:
            if tag in tags:
                score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def get_recommendations(
        self,
        user_input: str,
        user_context: Dict[str, Any],
        user_id: Optional[str] = None,
        top_k: int = 10
    ) -> List[Recommendation]:
        """Get personalized food recommendations based on mood and context."""
        
        # 1. Use enhanced intent classification if available, otherwise fall back to basic analysis
        if user_context.get('enhanced_intent'):
            logger.info(f"Using enhanced intent: {user_context['enhanced_intent']}")
            mood_categories = self._analyze_enhanced_intent(user_context)
        else:
            logger.info("Using basic intent analysis")
            mood_categories = self._analyze_user_intent(user_input)
        
        # 2. Get contextual factors
        context_factors = self._extract_context_factors(user_context)
        
        # 3. Find matching food items
        food_matches = self._find_mood_food_matches(mood_categories, context_factors)
        
        # 4. Find matching restaurants
        restaurant_matches = self._find_mood_restaurant_matches(mood_categories, context_factors)
        
        # 5. Combine and rank recommendations
        recommendations = self._combine_and_rank_recommendations(
            food_matches, restaurant_matches, user_context, user_id
        )
        
        return recommendations[:top_k]
    
    def _analyze_user_intent(self, user_input: str) -> List[str]:
        """Analyze user input to determine mood categories."""
        user_input_lower = user_input.lower()
        matched_categories = []
        
        for category, data in self.taxonomy.items():
            if 'descriptors' in data:
                for descriptor in data['descriptors']:
                    if descriptor.lower() in user_input_lower:
                        matched_categories.append(category)
                        break
        
        # If no direct matches, use semantic similarity
        if not matched_categories:
            matched_categories = self._semantic_intent_matching(user_input)
        
        return matched_categories
    
    def _analyze_enhanced_intent(self, user_context: Dict[str, Any]) -> List[str]:
        """Analyze enhanced intent classification results to determine mood categories."""
        enhanced_intent = user_context.get('enhanced_intent', '').lower()
        all_intents = user_context.get('all_intents', [])
        confidence = user_context.get('intent_confidence', 0.5)
        
        logger.info(f"Enhanced intent: {enhanced_intent}")
        logger.info(f"All intents: {all_intents}")
        
        # Extract intent labels from all_intents
        intent_labels = []
        if isinstance(all_intents, list):
            for intent_item in all_intents:
                if isinstance(intent_item, list) and len(intent_item) >= 1:
                    intent_labels.append(intent_item[0].lower())
                elif isinstance(intent_item, str):
                    intent_labels.append(intent_item.lower())
        
        # Add primary intent if not already in list
        if enhanced_intent and enhanced_intent not in intent_labels:
            intent_labels.insert(0, enhanced_intent)
        
        logger.info(f"Processed intent labels: {intent_labels}")
        
        # Map enhanced intent to mood categories
        intent_to_mood_mapping = {
            # Health and wellness
            'health_illness': ['HEALTH_ILLNESS', 'health_recovery', 'sensory_gentle', 'goal_soothing'],
            'health_recovery': ['health_recovery', 'HEALTH_ILLNESS', 'sensory_gentle'],
            'health_detox': ['HEALTH_DETOX', 'goal_light', 'sensory_refreshing'],
            'goal_health': ['goal_healthy', 'HEALTH_DETOX', 'goal_light'],
            'goal_healthy': ['goal_healthy', 'HEALTH_DETOX', 'goal_light'],
            
            # Comfort and soothing
            'comfort': ['EMOTIONAL_COMFORT', 'goal_comfort', 'sensory_comforting', 'SEASON_WINTER', 'SENSORY_WARMING'],
            'goal_comfort': ['EMOTIONAL_COMFORT', 'goal_comfort', 'sensory_comforting', 'SEASON_WINTER', 'SENSORY_WARMING'],
            'goal_soothing': ['goal_soothing', 'sensory_gentle', 'HEALTH_ILLNESS'],
            'sensory_soothing': ['sensory_soothing', 'goal_soothing', 'HEALTH_ILLNESS'],
            'sensory_warming': ['sensory_warming', 'WEATHER_COLD', 'goal_comfort', 'SEASON_WINTER', 'SENSORY_WARMING'],
            'temperature_warm': ['temperature_warm', 'sensory_warming', 'goal_comfort', 'SEASON_WINTER', 'SENSORY_WARMING'],
            
            # Hydration and refreshment
            'goal_hydration': ['goal_hydration', 'sensory_refreshing'],
            'sensory_refreshing': ['sensory_refreshing', 'goal_hydration'],
            'sensory_cooling': ['sensory_cooling', 'goal_hydration'],
            'temperature_cold': ['temperature_cold', 'sensory_refreshing'],
            
            # Light and healthy
            'goal_light': ['goal_light', 'sensory_light', 'HEALTH_DETOX'],
            'sensory_light': ['sensory_light', 'goal_light'],
            'nutritious': ['goal_healthy', 'HEALTH_DETOX', 'goal_light', 'SEASON_WINTER'],
            'warming': ['SENSORY_WARMING', 'SEASON_WINTER', 'WEATHER_COLD', 'goal_comfort'],
            
            # Energy and activity
            'goal_energy': ['goal_energy', 'goal_healthy'],
            'activity_gym': ['activity_gym', 'goal_energy', 'goal_healthy'],
            
            # Quick and simple
            'goal_quick': ['goal_quick', 'OCCASION_LUNCH_BREAK'],
            'goal_simple': ['goal_simple', 'sensory_simple'],
            
            # Social contexts
            'social_group': ['social_group', 'occasion_family_dinner'],
            'social_alone': ['social_alone', 'EMOTIONAL_COMFORT'],
            'occasion_romantic': ['occasion_romantic', 'EMOTIONAL_ROMANTIC'],
            'occasion_home': ['occasion_home', 'EMOTIONAL_COMFORT'],
            
            # Weather and seasons
            'weather_cold': ['WEATHER_COLD', 'sensory_warming', 'goal_comfort'],
            'season_winter': ['season_winter', 'WEATHER_COLD', 'sensory_warming'],
            'season_summer': ['season_summer', 'sensory_refreshing', 'goal_hydration'],
            
            # Indulgence and rich foods
            'goal_indulgence': ['goal_indulgence', 'sensory_rich', 'EMOTIONAL_COMFORT'],
            'sensory_rich': ['sensory_rich', 'goal_indulgence'],
            
            # Textures
            'texture_creamy': ['texture_creamy', 'sensory_comforting'],
            'texture_crispy': ['texture_crispy', 'sensory_crunchy'],
            'texture_greasy': ['texture_greasy', 'goal_indulgence'],
            
            # Emotions
            'emotion_nostalgic': ['emotion_nostalgic', 'goal_comfort'],
            'emotion_sad': ['emotion_sad', 'goal_comfort'],
            'emotion_stressed': ['emotion_stressed', 'goal_comfort'],
            'emotion_anxious': ['emotion_anxious', 'goal_comfort'],
            'emotion_adventurous': ['emotion_adventurous', 'goal_exploration'],
            
            # Meals
            'meal_dinner': ['meal_dinner', 'OCCASION_DINNER'],
            'meal_lunch': ['meal_lunch', 'OCCASION_LUNCH_BREAK'],
            'meal_breakfast': ['meal_breakfast', 'OCCASION_BREAKFAST'],
            'meal_snack': ['meal_snack', 'occasion_party_snacks'],
            
            # Specific foods
            'food_smoothie': ['food_smoothie', 'goal_hydration', 'goal_healthy'],
            
            # Additional granular labels from the model
            'sensory_warming': ['SENSORY_WARMING', 'SEASON_WINTER', 'WEATHER_COLD', 'goal_comfort', 'EMOTIONAL_COMFORT'],
            'emotional_comfort': ['EMOTIONAL_COMFORT', 'goal_comfort', 'sensory_comforting'],
            'social_couple': ['EMOTIONAL_ROMANTIC', 'occasion_romantic', 'sensory_elegant'],
            'occasion_party_snacks': ['occasion_party_snacks', 'sensory_festive', 'goal_indulgence'],
            'goal_satisfaction': ['goal_satisfaction', 'sensory_satisfying', 'goal_comfort'],
            'goal_healthy': ['goal_healthy', 'HEALTH_DETOX', 'goal_light'],
            'emotional_romantic': ['EMOTIONAL_ROMANTIC', 'occasion_romantic', 'sensory_elegant'],
            'health': ['HEALTH_ILLNESS', 'health_recovery', 'sensory_gentle', 'goal_soothing']
        }
        
        # Get primary mood categories from enhanced intent
        primary_mood_categories = intent_to_mood_mapping.get(enhanced_intent, [])
        logger.info(f"Enhanced intent: {enhanced_intent}")
        logger.info(f"Primary mood categories: {primary_mood_categories}")
        
        # Add additional categories from processed intent labels
        additional_categories = []
        for intent_label in intent_labels:
            if intent_label in intent_to_mood_mapping:
                additional_categories.extend(intent_to_mood_mapping[intent_label])
        
        logger.info(f"Additional categories from intent labels: {additional_categories}")
        
        # Combine and remove duplicates
        all_categories = list(set(primary_mood_categories + additional_categories))
        logger.info(f"All mood categories: {all_categories}")
        
        # Filter out sweet foods if user explicitly doesn't want sweet
        text_input = user_context.get('text_input', '').lower()
        if ('sweet' in text_input and ('not' in text_input or "don't" in text_input or "do not" in text_input)) or \
           ('sweet' in text_input and any(word in text_input for word in ['avoid', 'hate', 'dislike', 'no'])):
            all_categories = [cat for cat in all_categories if 'sweet' not in cat.lower()]
            logger.info("Filtered out sweet categories due to user preference")
        
        # Special handling for illness + no-sweet combination
        if enhanced_intent == 'goal_indulgence' and 'ill' in text_input and 'sweet' in text_input:
            # Override indulgence with health-focused categories
            all_categories = ['HEALTH_ILLNESS', 'health_recovery', 'sensory_gentle', 'goal_soothing', 'goal_light']
            logger.info("Overriding indulgence with health-focused categories for ill user")
        
        # Additional special case: if text contains "ill" and "sweet" but intent is wrong
        if 'ill' in text_input and ('sweet' in text_input or 'don\'t want' in text_input or 'do not want' in text_input):
            if enhanced_intent not in ['health', 'comfort']:
                all_categories = ['HEALTH_ILLNESS', 'health_recovery', 'sensory_gentle', 'goal_soothing', 'goal_light']
                logger.info("Forcing health-focused categories for ill user who doesn't want sweet")
        
        logger.info(f"Enhanced intent analysis: {enhanced_intent} -> {all_categories}")
        return all_categories
    
    def _semantic_intent_matching(self, user_input: str) -> List[str]:
        """Use semantic similarity to match user intent to mood categories."""
        # Enhanced semantic mapping with more comprehensive coverage
        semantic_mapping = {
            # Energy and texture preferences
            "light": ["ENERGY_LIGHT", "HEALTH_DETOX", "sensory_light"],
            "heavy": ["ENERGY_HEAVY", "goal_filling", "sensory_hearty"],
            "greasy": ["ENERGY_GREASY", "goal_indulgence", "sensory_rich"],
            "crunchy": ["sensory_crunchy", "texture_crispy"],
            "smooth": ["sensory_smooth", "texture_creamy"],
            
            # Temperature and weather preferences
            "warm": ["WEATHER_COLD", "EMOTIONAL_COMFORT", "sensory_warm"],
            "cold": ["WEATHER_HOT", "sensory_refreshing", "sensory_cool"],
            "hot": ["WEATHER_COLD", "sensory_warming", "sensory_spicy"],
            "cool": ["WEATHER_HOT", "sensory_refreshing"],
            
            # Emotional states
            "comfort": ["EMOTIONAL_COMFORT", "goal_soothing", "sensory_comforting"],
            "romantic": ["EMOTIONAL_ROMANTIC", "occasion_romantic", "sensory_elegant"],
            "excited": ["EMOTIONAL_CELEBRATORY", "sensory_exciting", "goal_adventure"],
            "sad": ["EMOTIONAL_COMFORT", "goal_soothing", "sensory_comforting"],
            "stressed": ["EMOTIONAL_COMFORT", "goal_relaxing", "sensory_soothing"],
            
            # Flavor profiles
            "sweet": ["FLAVOR_SWEET", "goal_indulgence", "sensory_sweet"],
            "spicy": ["FLAVOR_SPICY", "goal_excitement", "sensory_spicy"],
            "salty": ["FLAVOR_SALTY", "goal_satisfaction", "sensory_savory"],
            "savory": ["FLAVOR_SAVORY", "sensory_umami", "goal_satisfaction"],
            "bitter": ["FLAVOR_BITTER", "sensory_sophisticated"],
            "sour": ["FLAVOR_SOUR", "sensory_refreshing"],
            
            # Occasions and social contexts
            "quick": ["OCCASION_LUNCH_BREAK", "goal_quick", "sensory_efficient"],
            "family": ["OCCASION_FAMILY_DINNER", "social_family", "sensory_comforting"],
            "party": ["EMOTIONAL_CELEBRATORY", "occasion_party", "sensory_festive"],
            "date": ["EMOTIONAL_ROMANTIC", "occasion_romantic", "sensory_elegant"],
            "alone": ["EMOTIONAL_COMFORT", "sensory_simple", "goal_soothing"],
            
            # Health and wellness
            "sick": ["HEALTH_ILLNESS", "health_recovery", "sensory_gentle"],
            "healthy": ["HEALTH_DETOX", "goal_healthy", "sensory_fresh"],
            "fresh": ["HEALTH_DETOX", "sensory_fresh", "goal_healthy"],
            "organic": ["HEALTH_DETOX", "goal_healthy", "sensory_natural"],
            
            # Time-based preferences
            "breakfast": ["OCCASION_BREAKFAST", "sensory_light", "goal_energizing"],
            "lunch": ["OCCASION_LUNCH_BREAK", "sensory_balanced", "goal_sustaining"],
            "dinner": ["OCCASION_DINNER", "sensory_satisfying", "goal_completing"],
            "snack": ["OCCASION_SNACK", "sensory_light", "goal_quick"],
            
            # Cultural and regional preferences
            "asian": ["CULTURE_ASIAN", "sensory_exotic", "goal_adventure"],
            "italian": ["CULTURE_ITALIAN", "sensory_romantic", "goal_comfort"],
            "mexican": ["CULTURE_MEXICAN", "sensory_spicy", "goal_excitement"],
            "indian": ["CULTURE_INDIAN", "sensory_spicy", "goal_adventure"],
            "american": ["CULTURE_AMERICAN", "sensory_comforting", "goal_familiar"]
        }
        
        matched_categories = []
        user_input_lower = user_input.lower()
        
        # Direct keyword matching
        for keyword, categories in semantic_mapping.items():
            if keyword in user_input_lower:
                matched_categories.extend(categories)
        
        # Enhanced phrase matching for better context understanding
        phrase_mappings = {
            "comfort food": ["EMOTIONAL_COMFORT", "sensory_comforting", "goal_soothing"],
            "comforting": ["EMOTIONAL_COMFORT", "sensory_comforting"],
            "warm and cozy": ["EMOTIONAL_COMFORT", "sensory_warm", "goal_soothing"],
            "refreshing": ["sensory_refreshing", "WEATHER_HOT"],
            "hearty meal": ["sensory_hearty", "goal_filling", "ENERGY_HEAVY"],
            "light and fresh": ["sensory_light", "sensory_fresh", "HEALTH_DETOX"],
            "spicy and exciting": ["sensory_spicy", "goal_excitement", "FLAVOR_SPICY"],
            "romantic dinner": ["EMOTIONAL_ROMANTIC", "occasion_romantic", "sensory_elegant"],
            "family meal": ["OCCASION_FAMILY_DINNER", "social_family", "sensory_comforting"],
            "quick bite": ["goal_quick", "OCCASION_SNACK", "sensory_efficient"],
            "healthy choice": ["goal_healthy", "HEALTH_DETOX", "sensory_fresh"],
            "indulgent treat": ["goal_indulgence", "sensory_rich", "FLAVOR_SWEET"]
        }
        
        for phrase, categories in phrase_mappings.items():
            if phrase in user_input_lower:
                matched_categories.extend(categories)
        
        return list(set(matched_categories))
    
    def _extract_context_factors(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context factors for recommendation."""
        context_factors = {
            "time_of_day": user_context.get("time_of_day", "unknown"),
            "weather": user_context.get("weather", "unknown"),
            "location": user_context.get("location", "unknown"),
            "social_context": user_context.get("social_context", "unknown"),
            "occasion": user_context.get("occasion", "casual"),
            "energy_level": user_context.get("energy_level", "medium"),
            "hunger_level": user_context.get("hunger_level", "moderate"),
            "health_status": user_context.get("health_status", "normal")
        }
        return context_factors
    
    def _find_mood_food_matches(
        self, 
        mood_categories: List[str], 
        context_factors: Dict[str, Any]
    ) -> List[Tuple[FoodItem, float]]:
        """Find food items that match the mood categories."""
        food_matches = []
        logger.info(f"Looking for foods matching categories: {mood_categories}")
        logger.info(f"Total food items to check: {len(self.food_items)}")
        
        for food_item in self.food_items:
            match_score = 0.0
            
            # Direct category match
            if food_item.category in mood_categories:
                match_score += 0.4
                logger.info(f"Direct category match for {food_item.name}: {food_item.category}")
            
            # Label matching
            for label in food_item.labels:
                if label in mood_categories:
                    match_score += 0.2
                    logger.info(f"Label match for {food_item.name}: {label}")
            
            # Tag matching
            for tag in food_item.tags:
                if tag in mood_categories:
                    match_score += 0.1
                    logger.info(f"Tag match for {food_item.name}: {tag}")
            
            # Context matching
            context_score = self._calculate_context_match_score(food_item, context_factors)
            match_score += context_score * 0.3
            
            if match_score > 0.1:  # Threshold for relevance
                food_matches.append((food_item, match_score))
                logger.info(f"Added {food_item.name} with score {match_score}")
        
        # Sort by match score
        food_matches.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Found {len(food_matches)} food matches")
        return food_matches
    
    def _find_mood_restaurant_matches(
        self, 
        mood_categories: List[str], 
        context_factors: Dict[str, Any]
    ) -> List[Tuple[Restaurant, float]]:
        """Find restaurants that match the mood categories."""
        restaurant_matches = []
        
        for restaurant in self.restaurants:
            match_score = 0.0
            
            # Mood category matching
            for mood_category in mood_categories:
                if mood_category in restaurant.mood_categories:
                    match_score += 0.3
            
            # Tag matching
            for tag in restaurant.tags:
                if tag in mood_categories:
                    match_score += 0.1
            
            # Context matching
            context_score = self._calculate_restaurant_context_match(restaurant, context_factors)
            match_score += context_score * 0.2
            
            # Rating and popularity
            rating_score = (restaurant.rating - 3.0) / 2.0  # Normalize to 0-1
            match_score += rating_score * 0.2
            
            if match_score > 0.1:  # Threshold for relevance
                restaurant_matches.append((restaurant, match_score))
        
        # Sort by match score
        restaurant_matches.sort(key=lambda x: x[1], reverse=True)
        return restaurant_matches
    
    def _calculate_context_match_score(self, food_item: FoodItem, context_factors: Dict[str, Any]) -> float:
        """Calculate how well a food item matches the current context."""
        context_score = 0.0
        
        # Time of day matching
        time_of_day = context_factors.get("time_of_day", "unknown")
        if time_of_day == "morning" and "breakfast" in food_item.labels:
            context_score += 0.3
        elif time_of_day == "lunch" and "lunch" in food_item.labels:
            context_score += 0.3
        elif time_of_day == "dinner" and "dinner" in food_item.labels:
            context_score += 0.3
        
        # Weather matching
        weather = context_factors.get("weather", "unknown")
        if weather == "hot" and any(tag in ["cold", "refreshing", "cooling"] for tag in food_item.tags):
            context_score += 0.2
        elif weather == "cold" and any(tag in ["warm", "warming", "hearty"] for tag in food_item.tags):
            context_score += 0.2
        
        # Social context matching
        social_context = context_factors.get("social_context", "unknown")
        if social_context == "family" and "family" in food_item.tags:
            context_score += 0.2
        elif social_context == "romantic" and "romantic" in food_item.tags:
            context_score += 0.2
        
        return min(1.0, context_score)
    
    def _calculate_restaurant_context_match(self, restaurant: Restaurant, context_factors: Dict[str, Any]) -> float:
        """Calculate how well a restaurant matches the current context."""
        context_score = 0.0
        
        # Price range matching (simplified)
        price_range = context_factors.get("price_range", "$$")
        if price_range == restaurant.price_range:
            context_score += 0.2
        
        # Delivery preference
        delivery_preferred = context_factors.get("delivery_preferred", False)
        if delivery_preferred and restaurant.delivery_available:
            context_score += 0.2
        
        # Location matching (simplified)
        user_location = context_factors.get("user_location", None)
        if user_location and restaurant.location:
            # Calculate distance and score (simplified)
            context_score += 0.1
        
        return min(1.0, context_score)
    
    def _combine_and_rank_recommendations(
        self,
        food_matches: List[Tuple[FoodItem, float]],
        restaurant_matches: List[Tuple[Restaurant, float]],
        user_context: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[Recommendation]:
        """Combine food and restaurant matches into final recommendations."""
        recommendations = []
        
        # Create recommendations by combining food items with restaurants
        for food_item, food_score in food_matches[:20]:  # Top 20 food matches
            best_restaurant = None
            best_restaurant_score = 0.0
            
            # Find best matching restaurant for this food
            for restaurant, restaurant_score in restaurant_matches[:10]:
                if self._is_food_restaurant_compatible(food_item, restaurant):
                    combined_score = food_score * 0.6 + restaurant_score * 0.4
                    if combined_score > best_restaurant_score:
                        best_restaurant_score = combined_score
                        best_restaurant = restaurant
            
            # Calculate final recommendation score
            final_score = food_score * 0.6 + (best_restaurant_score if best_restaurant else 0.0) * 0.4
            
            # Add personalization score if user_id is provided
            personalization_score = 0.0
            if user_id and user_id in self.user_preferences:
                personalization_score = self._calculate_personalization_score(food_item, user_id)
                final_score += personalization_score * 0.2
            
            # Create reasoning
            reasoning = self._generate_recommendation_reasoning(food_item, best_restaurant, user_context)
            
            recommendation = Recommendation(
                food_item=food_item,
                restaurant=best_restaurant,
                score=final_score,
                reasoning=reasoning,
                mood_match=food_score,
                context_match=best_restaurant_score if best_restaurant else 0.0,
                personalization_score=personalization_score
            )
            
            recommendations.append(recommendation)
        
        # Sort by final score
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations
    
    def _is_food_restaurant_compatible(self, food_item: FoodItem, restaurant: Restaurant) -> bool:
        """Check if a food item is compatible with a restaurant."""
        # Simple compatibility check based on cuisine type and mood categories
        cuisine_mapping = {
            "American Comfort": ["EMOTIONAL_COMFORT", "OCCASION_FAMILY_DINNER"],
            "Indian": ["FLAVOR_SPICY", "EMOTIONAL_ADVENTUROUS"],
            "Health Food": ["HEALTH_DETOX", "ENERGY_LIGHT"],
            "Italian": ["EMOTIONAL_ROMANTIC", "OCCASION_FAMILY_DINNER"],
            "Fast Casual": ["OCCASION_LUNCH_BREAK", "goal_quick"]
        }
        
        restaurant_cuisine = restaurant.cuisine_type
        if restaurant_cuisine in cuisine_mapping:
            compatible_moods = cuisine_mapping[restaurant_cuisine]
            return any(mood in food_item.labels for mood in compatible_moods)
        
        return True  # Default to compatible if no specific mapping
    
    def _calculate_personalization_score(self, food_item: FoodItem, user_id: str) -> float:
        """Calculate personalization score based on user preferences and history."""
        if user_id not in self.user_preferences:
            return 0.0
        
        user_prefs = self.user_preferences[user_id]
        score = 0.0
        
        # Preference matching
        preferred_categories = user_prefs.get("preferred_categories", [])
        if food_item.category in preferred_categories:
            score += 0.3
        
        # Tag preference matching
        preferred_tags = user_prefs.get("preferred_tags", [])
        for tag in food_item.tags:
            if tag in preferred_tags:
                score += 0.1
        
        # Cultural preference matching
        preferred_cultures = user_prefs.get("preferred_cultures", [])
        if food_item.culture in preferred_cultures:
            score += 0.2
        
        return min(1.0, score)
    
    def _generate_recommendation_reasoning(
        self, 
        food_item: FoodItem, 
        restaurant: Optional[Restaurant], 
        user_context: Dict[str, Any]
    ) -> List[str]:
        """Generate human-readable reasoning for the recommendation."""
        reasoning = []
        
        # Food-based reasoning
        if food_item.category:
            reasoning.append(f"Matches your mood: {food_item.category.replace('_', ' ').title()}")
        
        if food_item.tags:
            tag_descriptions = [tag.replace('_', ' ').title() for tag in food_item.tags[:3]]
            reasoning.append(f"Characteristics: {', '.join(tag_descriptions)}")
        
        # Restaurant-based reasoning
        if restaurant:
            reasoning.append(f"Available at {restaurant.name} (Rating: {restaurant.rating}/5)")
            if restaurant.delivery_available:
                reasoning.append("Delivery available")
        
        # Context-based reasoning
        time_of_day = user_context.get("time_of_day", "")
        if time_of_day and time_of_day in ["morning", "lunch", "dinner"]:
            reasoning.append(f"Perfect for {time_of_day}")
        
        weather = user_context.get("weather", "")
        if weather == "hot" and any(tag in ["cold", "refreshing"] for tag in food_item.tags):
            reasoning.append("Great for hot weather")
        elif weather == "cold" and any(tag in ["warm", "warming"] for tag in food_item.tags):
            reasoning.append("Perfect for cold weather")
        
        return reasoning
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences for personalization."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id].update(preferences)
    
    def record_feedback(self, user_id: str, food_item_id: str, rating: float, feedback: str):
        """Record user feedback for continuous learning."""
        if user_id not in self.feedback_history:
            self.feedback_history[user_id] = []
        
        self.feedback_history[user_id].append({
            "food_item_id": food_item_id,
            "rating": rating,
            "feedback": feedback,
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        })