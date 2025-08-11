import numpy as np
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path

class MoodMapper:
    def __init__(self, taxonomy_path: str = "data/taxonomy/mood_food_taxonomy.json"):
        self.taxonomy = self._load_taxonomy(taxonomy_path)
        self.mood_vectors = self._create_mood_vectors()
        self.food_vectors = self._create_food_vectors()
        
    def _load_taxonomy(self, path: str) -> Dict:
        """Load the mood-food taxonomy from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _create_mood_vectors(self) -> Dict[str, np.ndarray]:
        """Create mood vectors for each category based on labels."""
        mood_vectors = {}
        all_labels = set()
        
        # Collect all unique labels
        for category, data in self.taxonomy.items():
            if 'labels' in data:
                all_labels.update(data['labels'])
        
        all_labels = sorted(list(all_labels))
        
        # Create vectors for each category
        for category, data in self.taxonomy.items():
            if 'labels' in data:
                vector = np.zeros(len(all_labels))
                for label in data['labels']:
                    if label in all_labels:
                        vector[all_labels.index(label)] = 1.0
                mood_vectors[category] = vector
        
        self.label_mapping = {label: idx for idx, label in enumerate(all_labels)}
        return mood_vectors
    
    def _create_food_vectors(self) -> List[Dict[str, Any]]:
        """Create food vectors with metadata for matching."""
        food_vectors = []
        
        for category, data in self.taxonomy.items():
            if 'foods' in data:
                for food in data['foods']:
                    food_vector = {
                        'name': food['name'],
                        'category': category,
                        'region': food.get('region', 'Unknown'),
                        'culture': food.get('culture', 'Unknown'),
                        'tags': food.get('tags', []),
                        'labels': data.get('labels', []),
                        'descriptors': data.get('descriptors', [])
                    }
                    food_vectors.append(food_vector)
        
        return food_vectors
    
    def map_mood_to_foods(self, intent: str, entities: List[str] = None, context: Dict = None) -> List[Dict[str, Any]]:
        """Map user intent to relevant food recommendations."""
        # Primary mood vector
        primary_mood = self._get_mood_vector(intent)
        
        # Secondary mood influences from entities
        secondary_moods = self._extract_secondary_moods(entities or [])
        
        # Context modifiers
        context_vector = self._get_context_vector(context or {})
        
        # Combine all mood vectors
        combined_mood = self._combine_vectors(primary_mood, secondary_moods, context_vector)
        
        # Find matching foods
        food_matches = self._find_food_matches(combined_mood)
        
        return food_matches
    
    def _get_mood_vector(self, intent: str) -> np.ndarray:
        """Get mood vector for a given intent."""
        # Find the best matching category
        best_match = None
        best_score = 0
        
        for category, data in self.taxonomy.items():
            if 'descriptors' in data:
                for descriptor in data['descriptors']:
                    # Simple keyword matching - could be enhanced with embeddings
                    if descriptor.lower() in intent.lower():
                        score = len(descriptor.split()) / len(intent.split())
                        if score > best_score:
                            best_score = score
                            best_match = category
        
        if best_match and best_match in self.mood_vectors:
            return self.mood_vectors[best_match]
        
        # Return zero vector if no match
        return np.zeros(len(self.label_mapping))
    
    def _extract_secondary_moods(self, entities: List[str]) -> List[np.ndarray]:
        """Extract secondary mood influences from entities."""
        secondary_moods = []
        
        for entity in entities:
            for category, data in self.taxonomy.items():
                if 'descriptors' in data:
                    for descriptor in data['descriptors']:
                        if descriptor.lower() in entity.lower():
                            if category in self.mood_vectors:
                                secondary_moods.append(self.mood_vectors[category])
        
        return secondary_moods
    
    def _get_context_vector(self, context: Dict) -> np.ndarray:
        """Get context vector based on environmental and situational factors."""
        context_vector = np.zeros(len(self.label_mapping))
        
        # Time-based context
        if 'time_of_day' in context:
            time = context['time_of_day']
            if time in ['morning', 'breakfast']:
                context_vector[self.label_mapping.get('time_morning', 0)] = 1.0
                context_vector[self.label_mapping.get('meal_breakfast', 0)] = 1.0
            elif time in ['lunch', 'midday']:
                context_vector[self.label_mapping.get('time_midday', 0)] = 1.0
                context_vector[self.label_mapping.get('meal_lunch', 0)] = 1.0
            elif time in ['dinner', 'evening']:
                context_vector[self.label_mapping.get('meal_dinner', 0)] = 1.0
            elif time in ['late_night', 'midnight']:
                context_vector[self.label_mapping.get('time_late_night', 0)] = 1.0
        
        # Weather context
        if 'weather' in context:
            weather = context['weather']
            if weather in ['hot', 'sunny', 'summer']:
                context_vector[self.label_mapping.get('weather_hot', 0)] = 1.0
                context_vector[self.label_mapping.get('season_summer', 0)] = 1.0
            elif weather in ['cold', 'snowy', 'winter']:
                context_vector[self.label_mapping.get('weather_cold', 0)] = 1.0
                context_vector[self.label_mapping.get('season_winter', 0)] = 1.0
        
        # Social context
        if 'social_context' in context:
            social = context['social_context']
            if social in ['alone', 'solo']:
                context_vector[self.label_mapping.get('social_alone', 0)] = 1.0
            elif social in ['couple', 'romantic']:
                context_vector[self.label_mapping.get('social_couple', 0)] = 1.0
            elif social in ['family', 'home']:
                context_vector[self.label_mapping.get('social_family', 0)] = 1.0
            elif social in ['friends', 'group', 'party']:
                context_vector[self.label_mapping.get('social_group', 0)] = 1.0
        
        return context_vector
    
    def _combine_vectors(self, primary: np.ndarray, secondary: List[np.ndarray], context: np.ndarray) -> np.ndarray:
        """Combine primary, secondary, and context vectors."""
        combined = primary.copy()
        
        # Add secondary moods with reduced weight
        for sec in secondary:
            combined += 0.5 * sec
        
        # Add context with moderate weight
        combined += 0.7 * context
        
        # Normalize
        if np.sum(combined) > 0:
            combined = combined / np.sum(combined)
        
        return combined
    
    def _find_food_matches(self, mood_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find top food matches based on mood vector."""
        food_scores = []
        
        for food in self.food_vectors:
            # Calculate similarity score
            food_labels = food['labels']
            food_vector = np.zeros(len(self.label_mapping))
            
            for label in food_labels:
                if label in self.label_mapping:
                    food_vector[self.label_mapping[label]] = 1.0
            
            # Cosine similarity
            similarity = np.dot(mood_vector, food_vector) / (np.linalg.norm(mood_vector) * np.linalg.norm(food_vector) + 1e-8)
            
            food_scores.append({
                'food': food,
                'score': similarity,
                'category': food['category']
            })
        
        # Sort by score and return top matches
        food_scores.sort(key=lambda x: x['score'], reverse=True)
        return food_scores[:top_k]
    
    def get_recommendations(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Get comprehensive food recommendations based on user input."""
        # Extract entities (simple approach - could be enhanced with NER)
        entities = self._extract_entities(user_input)
        
        # Get mood mapping
        food_matches = self.map_mood_to_foods(user_input, entities, context)
        
        # Group by category
        recommendations = {
            'user_input': user_input,
            'extracted_entities': entities,
            'context': context,
            'recommendations': food_matches,
            'categories': {}
        }
        
        # Group foods by category
        for match in food_matches:
            category = match['food']['category']
            if category not in recommendations['categories']:
                recommendations['categories'][category] = []
            recommendations['categories'][category].append(match['food'])
        
        return recommendations
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simple keyword extraction)."""
        entities = []
        
        # Extract weather-related entities
        weather_keywords = ['hot', 'cold', 'warm', 'cool', 'sunny', 'rainy', 'snowy', 'summer', 'winter']
        for keyword in weather_keywords:
            if keyword in text.lower():
                entities.append(keyword)
        
        # Extract emotional entities
        emotion_keywords = ['sad', 'happy', 'stressed', 'anxious', 'excited', 'tired', 'energetic']
        for keyword in emotion_keywords:
            if keyword in text.lower():
                entities.append(keyword)
        
        # Extract time entities
        time_keywords = ['morning', 'afternoon', 'evening', 'night', 'breakfast', 'lunch', 'dinner']
        for keyword in time_keywords:
            if keyword in text.lower():
                entities.append(keyword)
        
        # Extract social entities
        social_keywords = ['alone', 'couple', 'family', 'friends', 'party', 'date']
        for keyword in social_keywords:
            if keyword in text.lower():
                entities.append(keyword)
        
        return entities

    CONTEXT_FACTORS = {
        "temporal": {
            "time_of_day": ["morning", "afternoon", "evening", "late_night"],
            "day_of_week": ["weekday", "weekend"],
            "season": ["spring", "summer", "fall", "winter"]
        },
        "environmental": {
            "weather": ["sunny", "rainy", "cold", "hot", "humid"],
            "location": ["home", "office", "restaurant", "outdoor"]
        },
        "social": {
            "dining_context": ["alone", "couple", "family", "friends", "group"],
            "occasion": ["casual", "special", "celebration", "work_meal"]
        },
        "physical": {
            "energy_level": ["low", "medium", "high"],
            "hunger_level": ["light", "moderate", "very_hungry"],
            "health_status": ["normal", "sick", "recovering", "stressed"]
        }
    }