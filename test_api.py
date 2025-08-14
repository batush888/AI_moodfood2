#!/usr/bin/env python3
"""
Simple test script to isolate API issues
"""

import sys
import os
sys.path.append(os.getcwd())

def test_imports():
    """Test all imports."""
    print("Testing imports...")
    
    try:
        from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
        print("✅ EnhancedIntentClassifier imported")
    except Exception as e:
        print(f"❌ EnhancedIntentClassifier failed: {e}")
        return False
    
    try:
        from core.multimodal.multimodal_processor import MultiModalProcessor
        print("✅ MultiModalProcessor imported")
    except Exception as e:
        print(f"❌ MultiModalProcessor failed: {e}")
        return False
    
    try:
        from core.learning.realtime_learning import RealTimeLearningSystem
        print("✅ RealTimeLearningSystem imported")
    except Exception as e:
        print(f"❌ RealTimeLearningSystem failed: {e}")
        return False
    
    try:
        from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine
        print("✅ MoodBasedRecommendationEngine imported")
    except Exception as e:
        print(f"❌ MoodBasedRecommendationEngine failed: {e}")
        return False
    
    try:
        from core.phase3_enhancements import Phase3FeatureManager
        print("✅ Phase3FeatureManager imported")
    except Exception as e:
        print(f"❌ Phase3FeatureManager failed: {e}")
        return False
    
    return True

def test_enhanced_classifier():
    """Test enhanced classifier initialization."""
    print("\nTesting EnhancedIntentClassifier...")
    
    try:
        from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
        
        taxonomy_path = os.path.join(os.getcwd(), "data", "taxonomy", "mood_food_taxonomy.json")
        print(f"Taxonomy path: {taxonomy_path}")
        
        classifier = EnhancedIntentClassifier(taxonomy_path)
        print("✅ EnhancedIntentClassifier initialized")
        
        # Test classification
        result = classifier.classify_intent("I want comfort food")
        print(f"✅ Classification result: {result}")
        
        return True
    except Exception as e:
        print(f"❌ EnhancedIntentClassifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_recommendation_engine():
    """Test recommendation engine."""
    print("\nTesting MoodBasedRecommendationEngine...")
    
    try:
        from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine
        
        engine = MoodBasedRecommendationEngine()
        print("✅ MoodBasedRecommendationEngine initialized")
        
        # Test recommendations
        recommendations = engine.get_recommendations(
            "I want comfort food",
            {"time_of_day": "evening"},
            top_k=3
        )
        print(f"✅ Got {len(recommendations)} recommendations")
        
        return True
    except Exception as e:
        print(f"❌ MoodBasedRecommendationEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== API Component Test Suite ===\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return
    
    # Test enhanced classifier
    if not test_enhanced_classifier():
        print("\n❌ Enhanced classifier test failed")
        return
    
    # Test recommendation engine
    if not test_recommendation_engine():
        print("\n❌ Recommendation engine test failed")
        return
    
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    main()
