#!/usr/bin/env python3
"""
Simple test script to verify the core components work correctly.
This doesn't require the API server to be running.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mood_mapper():
    """Test the MoodMapper class."""
    print("🧠 Testing MoodMapper...")
    try:
        from core.mood_mapper import MoodMapper
        
        mapper = MoodMapper()
        print(f"✅ MoodMapper initialized with {len(mapper.taxonomy)} categories")
        print(f"✅ Created {len(mapper.mood_vectors)} mood vectors")
        print(f"✅ Created {len(mapper.food_vectors)} food vectors")
        
        # Test mood analysis
        test_input = "I want something warm and comforting"
        entities = mapper._extract_entities(test_input)
        print(f"✅ Extracted entities: {entities}")
        
        # Test mood mapping
        food_matches = mapper.map_mood_to_foods(test_input, entities, {})
        print(f"✅ Found {len(food_matches)} food matches")
        
        return True
        
    except Exception as e:
        print(f"❌ MoodMapper test failed: {e}")
        return False

def test_recommendation_engine():
    """Test the MoodBasedRecommendationEngine class."""
    print("\n🍽️  Testing Recommendation Engine...")
    try:
        from core.recommendation.recommendation_algorithm import MoodBasedRecommendationEngine
        
        engine = MoodBasedRecommendationEngine()
        print(f"✅ Engine initialized with {len(engine.food_items)} food items")
        print(f"✅ Engine initialized with {len(engine.restaurants)} restaurants")
        
        # Test intent analysis
        test_input = "I'm feeling hot and need something refreshing"
        mood_categories = engine._analyze_user_intent(test_input)
        print(f"✅ Extracted mood categories: {mood_categories}")
        
        # Test context extraction
        context = {"weather": "hot", "time_of_day": "afternoon"}
        context_factors = engine._extract_context_factors(context)
        print(f"✅ Extracted context factors: {context_factors}")
        
        return True
        
    except Exception as e:
        print(f"❌ Recommendation Engine test failed: {e}")
        return False

def test_taxonomy_loading():
    """Test that the taxonomy can be loaded correctly."""
    print("\n📚 Testing Taxonomy Loading...")
    try:
        import json
        
        with open("data/taxonomy/mood_food_taxonomy.json", "r") as f:
            taxonomy = json.load(f)
        
        print(f"✅ Taxonomy loaded with {len(taxonomy)} categories")
        
        # Check a few categories
        expected_categories = [
            "WEATHER_HOT", "WEATHER_COLD", "EMOTIONAL_COMFORT", 
            "FLAVOR_SPICY", "OCCASION_FAMILY_DINNER"
        ]
        
        for category in expected_categories:
            if category in taxonomy:
                print(f"✅ Found category: {category}")
                foods = taxonomy[category].get("foods", [])
                print(f"   Contains {len(foods)} food items")
            else:
                print(f"❌ Missing category: {category}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Taxonomy loading test failed: {e}")
        return False

def test_training_data():
    """Test that the training data can be loaded."""
    print("\n📊 Testing Training Data...")
    try:
        import json
        
        with open("data/intent_dataset.json", "r") as f:
            training_data = json.load(f)
        
        print(f"✅ Training data loaded with {len(training_data)} examples")
        
        # Check a few examples
        for i, example in enumerate(training_data[:3]):
            print(f"   Example {i+1}: '{example['text']}'")
            print(f"     Labels: {example['labels']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training data test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Running System Tests...")
    print("=" * 40)
    
    tests = [
        ("Taxonomy Loading", test_taxonomy_loading),
        ("Training Data", test_training_data),
        ("Mood Mapper", test_mood_mapper),
        ("Recommendation Engine", test_recommendation_engine)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Start the API server: python run.py")
        print("   2. Run the demo: python demo_system.py")
        print("   3. Test the API endpoints")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 