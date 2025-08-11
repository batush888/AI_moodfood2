#!/usr/bin/env python3
"""
AI Mood-Based Food Recommendation System - Demonstration Script

This script demonstrates the complete capabilities of the system:
1. Mood analysis and intent classification
2. Context-aware food recommendations
3. Restaurant matching and ranking
4. Personalization and feedback
5. API endpoint testing
"""

import json
import requests
import time
from typing import Dict, Any, List

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ System Status: {health_data['status']}")
            for component, status in health_data['components'].items():
                print(f"   {component}: {status}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_taxonomy():
    """Test the taxonomy endpoint."""
    print("\n📚 Testing Taxonomy Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/taxonomy")
        if response.status_code == 200:
            taxonomy_data = response.json()
            print(f"✅ Total Categories: {taxonomy_data['total_categories']}")
            print("📋 Sample Categories:")
            for i, (category, data) in enumerate(list(taxonomy_data['categories'].items())[:5]):
                print(f"   {i+1}. {category}")
                print(f"      Descriptors: {', '.join(data['descriptors'][:3])}")
                print(f"      Example Foods: {', '.join(data['example_foods'])}")
        else:
            print(f"❌ Taxonomy request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Taxonomy error: {e}")

def test_examples():
    """Test the examples endpoint."""
    print("\n💡 Testing Examples Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/examples")
        if response.status_code == 200:
            examples_data = response.json()
            print(f"✅ Total Examples: {examples_data['total_examples']}")
            print("📝 Sample Queries by Category:")
            for category, queries in examples_data['examples'].items():
                print(f"   {category.replace('_', ' ').title()}:")
                for query in queries[:2]:  # Show first 2 examples
                    print(f"     • {query}")
        else:
            print(f"❌ Examples request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Examples error: {e}")

def test_mood_analysis(user_input: str, context: Dict[str, Any] = None):
    """Test the mood analysis endpoint."""
    print(f"\n🧠 Testing Mood Analysis for: '{user_input}'")
    try:
        payload = {
            "user_input": user_input,
            "context": context or {}
        }
        response = requests.post(f"{BASE_URL}/analyze-mood", json=payload)
        if response.status_code == 200:
            analysis_data = response.json()
            print(f"✅ Primary Mood: {analysis_data['primary_mood']}")
            print(f"✅ Mood Categories: {', '.join(analysis_data['mood_categories'])}")
            print(f"✅ Extracted Entities: {', '.join(analysis_data['extracted_entities'])}")
            print(f"✅ Confidence Score: {analysis_data['confidence_score']}")
            print(f"✅ Context Factors: {analysis_data['context_factors']}")
        else:
            print(f"❌ Mood analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Mood analysis error: {e}")

def test_recommendations(user_input: str, context: Dict[str, Any] = None, user_id: str = None):
    """Test the recommendations endpoint."""
    print(f"\n🍽️  Testing Recommendations for: '{user_input}'")
    try:
        payload = {
            "user_input": user_input,
            "user_context": context or {},
            "user_id": user_id,
            "top_k": 5
        }
        response = requests.post(f"{BASE_URL}/recommend", json=payload)
        if response.status_code == 200:
            rec_data = response.json()
            print(f"✅ Extracted Mood: {', '.join(rec_data['extracted_mood'])}")
            print(f"✅ Recommendations Found: {len(rec_data['recommendations'])}")
            print("\n🍴 Top Recommendations:")
            
            for i, rec in enumerate(rec_data['recommendations'][:3], 1):
                print(f"   {i}. {rec['food_name']} ({rec['food_category']})")
                print(f"      Score: {rec['score']} | Mood Match: {rec['mood_match']}")
                if rec['restaurant_name']:
                    print(f"      Restaurant: {rec['restaurant_name']} ({rec['restaurant_cuisine']})")
                    print(f"      Rating: {rec['restaurant_rating']}/5 | Delivery: {rec['delivery_available']}")
                print(f"      Reasoning: {'; '.join(rec['reasoning'][:2])}")
        else:
            print(f"❌ Recommendations failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Recommendations error: {e}")

def test_user_preferences(user_id: str, preferences: Dict[str, Any]):
    """Test the user preferences endpoint."""
    print(f"\n👤 Testing User Preferences Update for User: {user_id}")
    try:
        payload = {
            "user_id": user_id,
            "preferences": preferences
        }
        response = requests.post(f"{BASE_URL}/preferences", json=payload)
        if response.status_code == 200:
            pref_data = response.json()
            print(f"✅ Preferences Updated Successfully")
            print(f"✅ Updated Preferences: {pref_data['updated_preferences']}")
        else:
            print(f"❌ Preferences update failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Preferences error: {e}")

def test_feedback(user_id: str, food_item_id: str, rating: float, feedback: str):
    """Test the feedback endpoint."""
    print(f"\n💬 Testing Feedback Submission...")
    try:
        payload = {
            "user_id": user_id,
            "food_item_id": food_item_id,
            "rating": rating,
            "feedback": feedback
        }
        response = requests.post(f"{BASE_URL}/feedback", json=payload)
        if response.status_code == 200:
            feedback_data = response.json()
            print(f"✅ Feedback Recorded Successfully")
            print(f"✅ User ID: {feedback_data['user_id']}")
        else:
            print(f"❌ Feedback submission failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Feedback error: {e}")

def run_comprehensive_demo():
    """Run a comprehensive demonstration of all system capabilities."""
    print("🚀 AI Mood-Based Food Recommendation System - Comprehensive Demo")
    print("=" * 70)
    
    # Wait for system to be ready
    print("⏳ Waiting for system to be ready...")
    time.sleep(2)
    
    # 1. Health Check
    test_health_check()
    
    # 2. Taxonomy Overview
    test_taxonomy()
    
    # 3. Examples
    test_examples()
    
    # 4. Test various mood-based queries
    test_cases = [
        {
            "input": "I'm feeling hot and need something refreshing",
            "context": {"weather": "hot", "time_of_day": "afternoon"}
        },
        {
            "input": "I want something comforting because I'm stressed",
            "context": {"social_context": "alone", "time_of_day": "evening"}
        },
        {
            "input": "It's date night, I want something romantic",
            "context": {"social_context": "couple", "occasion": "special"}
        },
        {
            "input": "I need something quick for lunch break",
            "context": {"time_of_day": "lunch", "occasion": "work_meal"}
        },
        {
            "input": "I'm craving something spicy and exciting",
            "context": {"energy_level": "high", "social_context": "friends"}
        }
    ]
    
    print("\n🎯 Testing Various Mood-Based Queries...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        test_mood_analysis(test_case["input"], test_case["context"])
        test_recommendations(test_case["input"], test_case["context"])
    
    # 5. Test personalization
    print("\n🎭 Testing Personalization Features...")
    user_id = "demo_user_001"
    
    # Update user preferences
    preferences = {
        "preferred_categories": ["EMOTIONAL_COMFORT", "FLAVOR_SPICY"],
        "preferred_tags": ["comforting", "spicy", "warm"],
        "preferred_cultures": ["Italian", "Indian"],
        "price_range": "$$",
        "delivery_preferred": True
    }
    test_user_preferences(user_id, preferences)
    
    # Test personalized recommendations
    test_recommendations(
        "I want something comforting and spicy",
        {"time_of_day": "dinner", "social_context": "alone"},
        user_id
    )
    
    # 6. Test feedback system
    print("\n📝 Testing Feedback System...")
    test_feedback(
        user_id=user_id,
        food_item_id="curry_001",
        rating=4.5,
        feedback="Excellent recommendation! The spicy curry was perfect for my mood."
    )
    
    print("\n🎉 Demo Complete! The system is working as expected.")
    print("\n💡 Try these additional queries:")
    print("   • 'I'm sick and need something gentle'")
    print("   • 'Family dinner, something traditional'")
    print("   • 'I want something greasy and indulgent'")
    print("   • 'Party snacks for a celebration'")

def run_quick_test():
    """Run a quick test to verify basic functionality."""
    print("⚡ Quick System Test")
    print("=" * 30)
    
    try:
        # Health check
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ System is running")
            
            # Quick recommendation test
            payload = {
                "user_input": "I want something warm and comforting",
                "user_context": {"weather": "cold", "time_of_day": "evening"}
            }
            response = requests.post(f"{BASE_URL}/recommend", json=payload)
            if response.status_code == 200:
                print("✅ Recommendations working")
                rec_data = response.json()
                print(f"   Found {len(rec_data['recommendations'])} recommendations")
            else:
                print("❌ Recommendations not working")
        else:
            print("❌ System not responding")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        run_comprehensive_demo() 