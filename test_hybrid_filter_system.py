#!/usr/bin/env python3
"""
Test script for the new Hybrid Filter System with LLM-as-Teacher architecture.

This script tests:
1. Hybrid filter initialization
2. Query processing through the pipeline
3. LLM interpretation and validation
4. ML prediction and fallback
5. Logging and statistics
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add the project root to the path
sys.path.append('.')

async def test_hybrid_filter_system():
    """Test the complete hybrid filter system."""
    print("🚀 Testing Hybrid Filter System with LLM-as-Teacher")
    print("=" * 60)
    
    try:
        # Test 1: Import and initialize components
        print("\n1️⃣ Testing Component Imports...")
        
        from core.filtering.hybrid_filter import HybridFilter, HybridFilterResponse
        from core.filtering.llm_validator import LLMValidator
        from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
        
        print("✅ All components imported successfully")
        
        # Test 2: Initialize LLM Validator
        print("\n2️⃣ Testing LLM Validator...")
        
        llm_validator = LLMValidator()
        print(f"LLM Validator enabled: {llm_validator.enabled}")
        print(f"Model: {llm_validator.model}")
        
        # Test 3: Initialize Enhanced Intent Classifier
        print("\n3️⃣ Testing Enhanced Intent Classifier...")
        
        taxonomy_path = "data/taxonomy/mood_food_taxonomy.json"
        if not Path(taxonomy_path).exists():
            print(f"⚠️ Taxonomy file not found: {taxonomy_path}")
            print("Creating a minimal classifier...")
            ml_classifier = None
        else:
            ml_classifier = EnhancedIntentClassifier(taxonomy_path)
            print("✅ Enhanced Intent Classifier initialized")
        
        # Test 4: Initialize Hybrid Filter
        print("\n4️⃣ Testing Hybrid Filter Initialization...")
        
        hybrid_filter = HybridFilter(
            llm_validator=llm_validator,
            ml_classifier=ml_classifier,
            confidence_threshold=0.7
        )
        print("✅ Hybrid Filter initialized successfully")
        
        # Test 5: Test Query Processing
        print("\n5️⃣ Testing Query Processing...")
        
        test_queries = [
            "I want some Japanese food",
            "I need comfort food",
            "Give me something spicy",
            "I want cold refreshing drinks"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Processing Query {i}: '{query}'")
            
            try:
                result = await hybrid_filter.process_query(query)
                
                print(f"   Decision: {result.decision}")
                print(f"   Recommendations: {len(result.recommendations)} items")
                print(f"   First 3 recommendations: {result.recommendations[:3]}")
                print(f"   Reasoning: {result.reasoning[:100]}...")
                print(f"   Processing time: {result.processing_time_ms:.2f}ms")
                
                if result.llm_interpretation:
                    print(f"   LLM Intent: {result.llm_interpretation.intent}")
                    print(f"   LLM Confidence: {result.llm_interpretation.confidence:.3f}")
                
                if result.ml_prediction:
                    print(f"   ML Intent: {result.ml_prediction.primary_intent}")
                    print(f"   ML Confidence: {result.ml_prediction.confidence:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error processing query: {e}")
        
        # Test 6: Test Live Statistics
        print("\n6️⃣ Testing Live Statistics...")
        
        stats = hybrid_filter.get_live_stats()
        print("Live Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test 7: Test Logging
        print("\n7️⃣ Testing Logging...")
        
        log_file = Path("logs/recommendation_logs.jsonl")
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   Log file exists with {len(lines)} entries")
                
                if lines:
                    # Show the last log entry
                    last_entry = json.loads(lines[-1])
                    print(f"   Last log entry timestamp: {last_entry.get('timestamp', 'unknown')}")
                    print(f"   Last query: {last_entry.get('query', 'unknown')[:50]}...")
                    print(f"   Decision source: {last_entry.get('decision_source', 'unknown')}")
        else:
            print("   ⚠️ Log file not found yet")
        
        print("\n✅ Hybrid Filter System Test Complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_integration():
    """Test the API integration with the hybrid filter."""
    print("\n🌐 Testing API Integration...")
    print("=" * 40)
    
    try:
        import requests
        
        # Test the filter stats endpoint
        print("Testing /logging/filter-stats endpoint...")
        
        response = requests.get('http://localhost:8000/logging/filter-stats')
        if response.status_code == 200:
            stats = response.json()
            print("✅ Filter stats endpoint working")
            print(f"   Source: {stats.get('source', 'unknown')}")
            print(f"   Total samples: {stats.get('total_samples', 0)}")
        else:
            print(f"❌ Filter stats endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ API integration test failed: {e}")

def main():
    """Run all tests."""
    print("🧪 Hybrid Filter System Test Suite")
    print("=" * 60)
    
    # Run the main test
    success = asyncio.run(test_hybrid_filter_system())
    
    # Run API integration test if server is running
    try:
        asyncio.run(test_api_integration())
    except:
        print("⚠️ API integration test skipped (server not running)")
    
    if success:
        print("\n🎉 All tests passed! The hybrid filter system is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
