#!/usr/bin/env python3
"""
Test script to debug intent classification issues.
This will help identify why the system always returns the same recommendations.
"""

import sys
import os
sys.path.append('.')

def test_intent_classification():
    """Test intent classification with different queries."""
    print("🔍 Testing Intent Classification")
    print("=" * 50)
    
    try:
        from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
        
        # Initialize the classifier
        taxonomy_path = "data/taxonomy/mood_food_taxonomy.json"
        classifier = EnhancedIntentClassifier(taxonomy_path)
        
        # Test queries
        test_queries = [
            "I want some Japanese food",
            "I want something cold", 
            "I want some south eastern food",
            "I want comfort food",
            "I want something spicy",
            "I want a healthy meal"
        ]
        
        print("🧪 Testing different queries:")
        print()
        
        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: '{query}'")
            
            try:
                # Classify intent
                result = classifier.classify_intent(query)
                
                print(f"  Primary Intent: {result.get('primary_intent', 'unknown')}")
                print(f"  Confidence: {result.get('confidence', 0.0):.3f}")
                print(f"  Method: {result.get('method', 'unknown')}")
                print(f"  All Intents: {result.get('all_intents', [])}")
                
                if result.get('fallback'):
                    print(f"  ⚠️ Using fallback method")
                
                print()
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                print()
        
        # Test the fallback keywords directly
        print("🔍 Testing Fallback Keywords:")
        print("-" * 30)
        
        if hasattr(classifier, 'fallback_keywords'):
            for intent, keywords in classifier.fallback_keywords.items():
                print(f"{intent}: {keywords}")
        else:
            print("No fallback keywords found")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_recommendation_engine():
    """Test the recommendation engine with different intents."""
    print("\n🍽️ Testing Recommendation Engine")
    print("=" * 50)
    
    try:
        from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine
        
        # Initialize the engine
        engine = MoodBasedRecommendationEngine()
        
        # Test different contexts
        test_contexts = [
            {"enhanced_intent": "comfort", "all_intents": [["comfort", 0.8]]},
            {"enhanced_intent": "japanese", "all_intents": [["japanese", 0.8]]},
            {"enhanced_intent": "cold", "all_intents": [["cold", 0.8]]},
            {"enhanced_intent": "spicy", "all_intents": [["spicy", 0.8]]}
        ]
        
        for i, context in enumerate(test_contexts, 1):
            print(f"Context {i}: {context['enhanced_intent']}")
            
            try:
                # Get recommendations
                recommendations = engine.get_recommendations(
                    user_input=f"I want {context['enhanced_intent']} food",
                    user_context=context,
                    top_k=3
                )
                
                print(f"  Recommendations: {len(recommendations)} items")
                for j, rec in enumerate(recommendations[:3]):
                    print(f"    {j+1}. {rec.food_item.name} (score: {rec.score:.3f})")
                print()
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                print()
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all tests."""
    print("🚀 Intent Classification Debug Test")
    print("=" * 60)
    
    test_intent_classification()
    test_recommendation_engine()
    
    print("✅ Debug test complete!")

if __name__ == "__main__":
    main()
