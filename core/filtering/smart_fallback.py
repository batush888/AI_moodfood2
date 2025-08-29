"""
Smart Hybrid Filter Fallback System
===================================

This module provides intelligent fallback mechanisms when the hybrid filter
fails, ensuring users always receive personalized recommendations.
"""

import json
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class FallbackRecommendation:
    """Represents a fallback recommendation with metadata."""
    food_name: str
    food_category: str
    food_region: str
    food_culture: str
    food_tags: List[str]
    score: float
    mood_match: float
    reasoning: List[str]
    fallback_type: str
    confidence: float

class SmartFallbackSystem:
    """Provides intelligent fallback recommendations when hybrid filter fails."""
    
    def __init__(self):
        self.fallback_strategies = [
            self._mood_based_fallback,
            self._context_based_fallback,
            self._popularity_based_fallback
        ]
        
    def generate_smart_fallback(self, 
                               user_query: str,
                               user_context: Optional[Dict[str, Any]] = None,
                               session_memory: Optional[Dict[str, Any]] = None,
                               num_recommendations: int = 3) -> List[FallbackRecommendation]:
        """Generate intelligent fallback recommendations using multiple strategies."""
        logger.info("🔄 Generating smart fallback recommendations")
        
        all_recommendations = []
        
        # Try each fallback strategy
        for strategy in self.fallback_strategies:
            try:
                recommendations = strategy(user_query, user_context, session_memory, num_recommendations)
                all_recommendations.extend(recommendations)
            except Exception as e:
                logger.warning(f"Fallback strategy failed: {e}")
                continue
        
        # If no strategies worked, use emergency fallback
        if not all_recommendations:
            all_recommendations = self._emergency_fallback(num_recommendations)
        
        # Score and rank recommendations
        scored_recommendations = self._score_recommendations(
            all_recommendations, user_query, user_context, session_memory
        )
        
        # Return top recommendations
        top_recommendations = scored_recommendations[:num_recommendations]
        
        logger.info(f"✅ Generated {len(top_recommendations)} smart fallback recommendations")
        return top_recommendations
    
    def _mood_based_fallback(self, user_query: str, user_context: Optional[Dict[str, Any]] = None, session_memory: Optional[Dict[str, Any]] = None, num_recommendations: int = 3) -> List[FallbackRecommendation]:
        """Generate recommendations based on detected mood and context."""
        mood = self._extract_mood(user_query, user_context, session_memory)
        
        # Mood-based food mappings
        mood_foods = {
            'comfort': [
                {"name": "Chicken Soup", "category": "SOUP", "region": "Global", "culture": "Universal", "tags": ["warm", "healing", "simple"]},
                {"name": "Mac and Cheese", "category": "PASTA", "region": "American", "culture": "Western", "tags": ["creamy", "familiar", "satisfying"]}
            ],
            'energy': [
                {"name": "Smoothie Bowl", "category": "BREAKFAST", "region": "Global", "culture": "Modern", "tags": ["fresh", "vibrant", "nutritious"]},
                {"name": "Quinoa Salad", "category": "SALAD", "region": "Global", "culture": "Health-conscious", "tags": ["protein", "fresh", "energizing"]}
            ],
            'celebration': [
                {"name": "Chocolate Cake", "category": "DESSERT", "region": "Global", "culture": "Universal", "tags": ["sweet", "indulgent", "celebratory"]},
                {"name": "Sushi Platter", "category": "JAPANESE", "region": "Japanese", "culture": "Asian", "tags": ["elegant", "fresh", "special"]}
            ]
        }
        
        foods = mood_foods.get(mood, mood_foods['comfort'])
        recommendations = []
        
        for food in foods[:num_recommendations]:
            recommendation = FallbackRecommendation(
                food_name=food['name'],
                food_category=food['category'],
                food_region=food['region'],
                food_culture=food['culture'],
                food_tags=food['tags'],
                score=0.7,
                mood_match=0.8 if mood in food['tags'] else 0.6,
                reasoning=[f"Selected based on your current mood: {mood}"],
                fallback_type="mood_based",
                confidence=0.6
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _context_based_fallback(self, user_query: str, user_context: Optional[Dict[str, Any]] = None, session_memory: Optional[Dict[str, Any]] = None, num_recommendations: int = 3) -> List[FallbackRecommendation]:
        """Generate recommendations based on time, weather, and social context."""
        if not user_context:
            return []
        
        time_of_day = user_context.get('time_of_day', 'afternoon')
        weather = user_context.get('weather', 'moderate')
        social_context = user_context.get('social_context', 'alone')
        
        # Context-based recommendations
        context_foods = [
            {"name": "Oatmeal with Berries", "category": "BREAKFAST", "region": "Global", "culture": "Universal", "tags": ["morning", "healthy", "energizing"]},
            {"name": "Grilled Salmon", "category": "SEAFOOD", "region": "Global", "culture": "Universal", "tags": ["evening", "healthy", "elegant"]},
            {"name": "Pizza Margherita", "category": "PIZZA", "region": "Italian", "culture": "Western", "tags": ["social", "versatile", "satisfying"]}
        ]
        
        recommendations = []
        for food in context_foods[:num_recommendations]:
            recommendation = FallbackRecommendation(
                food_name=food['name'],
                food_category=food['category'],
                food_region=food['region'],
                food_culture=food['culture'],
                food_tags=food['tags'],
                score=0.65,
                mood_match=0.7,
                reasoning=[
                    f"Perfect for {time_of_day}",
                    f"Great for {weather} weather",
                    f"Ideal for {social_context} dining"
                ],
                fallback_type="context_based",
                confidence=0.7
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _popularity_based_fallback(self, user_query: str, user_context: Optional[Dict[str, Any]] = None, session_memory: Optional[Dict[str, Any]] = None, num_recommendations: int = 3) -> List[FallbackRecommendation]:
        """Generate recommendations based on popular and well-liked foods."""
        popular_foods = [
            {"name": "Pizza Margherita", "category": "PIZZA", "region": "Italian", "culture": "Western", "tags": ["popular", "versatile", "satisfying"]},
            {"name": "Chicken Teriyaki", "category": "ASIAN", "region": "Japanese", "culture": "Asian", "tags": ["popular", "healthy", "flavorful"]},
            {"name": "Caesar Salad", "category": "SALAD", "region": "American", "culture": "Western", "tags": ["popular", "fresh", "light"]}
        ]
        
        recommendations = []
        for food in popular_foods[:num_recommendations]:
            recommendation = FallbackRecommendation(
                food_name=food['name'],
                food_category=food['category'],
                food_region=food['region'],
                food_culture=food['culture'],
                food_tags=food['tags'],
                score=0.6,
                mood_match=0.5,
                reasoning=[f"Popular choice with broad appeal"],
                fallback_type="popularity_based",
                confidence=0.8
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _emergency_fallback(self, num_recommendations: int = 3) -> List[FallbackRecommendation]:
        """Emergency fallback when all other strategies fail."""
        emergency_foods = [
            {"name": "Grilled Chicken Breast", "category": "PROTEIN", "region": "Global", "culture": "Universal", "tags": ["healthy", "versatile", "reliable"]},
            {"name": "Mixed Green Salad", "category": "SALAD", "region": "Global", "culture": "Universal", "tags": ["healthy", "light", "refreshing"]},
            {"name": "Fruit Smoothie", "category": "BEVERAGE", "region": "Global", "culture": "Modern", "tags": ["healthy", "refreshing", "nutritious"]}
        ]
        
        recommendations = []
        for food in emergency_foods[:num_recommendations]:
            recommendation = FallbackRecommendation(
                food_name=food['name'],
                food_category=food['category'],
                food_region=food['region'],
                food_culture=food['culture'],
                food_tags=food['tags'],
                score=0.5,
                mood_match=0.3,
                reasoning=["Reliable fallback option"],
                fallback_type="emergency",
                confidence=0.5
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _extract_mood(self, user_query: str, user_context: Optional[Dict[str, Any]] = None, session_memory: Optional[Dict[str, Any]] = None) -> str:
        """Extract mood from user query, context, and session memory."""
        if user_context and user_context.get('mood'):
            return user_context['mood']
        
        query_lower = user_query.lower()
        mood_keywords = {
            'comfort': ['comfort', 'cozy', 'warm', 'healing', 'soothing'],
            'energy': ['energizing', 'boost', 'vitality', 'fresh', 'invigorating'],
            'celebration': ['celebrate', 'special', 'treat', 'indulge', 'party']
        }
        
        for mood, keywords in mood_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return mood
        
        return 'comfort'
    
    def _score_recommendations(self, recommendations: List[FallbackRecommendation], user_query: str, user_context: Optional[Dict[str, Any]] = None, session_memory: Optional[Dict[str, Any]] = None) -> List[FallbackRecommendation]:
        """Score and rank fallback recommendations."""
        for rec in recommendations:
            score = rec.score
            
            # Boost score based on context relevance
            if user_context:
                if user_context.get('time_of_day') == 'morning' and 'breakfast' in rec.food_category.lower():
                    score += 0.1
                if user_context.get('weather') == 'cold' and any(tag in ['warm', 'heating'] for tag in rec.food_tags):
                    score += 0.1
            
            rec.score = min(1.0, max(0.0, score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations

# Global fallback system instance
_fallback_system = None

def get_fallback_system() -> SmartFallbackSystem:
    """Get or create global fallback system instance."""
    global _fallback_system
    if _fallback_system is None:
        _fallback_system = SmartFallbackSystem()
    return _fallback_system
