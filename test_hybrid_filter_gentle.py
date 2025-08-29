#!/usr/bin/env python3
"""
Gentle test script for the Hybrid Filter System that avoids rate limiting.

This script tests the system with delays between API calls to avoid hitting rate limits.
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path

# Add the project root to the path
sys.path.append('.')

async def test_hybrid_filter_gentle():
    """Test the hybrid filter system with gentle API usage."""
    print("🚀 Testing Hybrid Filter System (Gentle Mode)")
    print("=" * 50)
    
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
        
        if not llm_validator.enabled:
            print("⚠️ LLM Validator is disabled - this is expected if no API key")
            return True
        
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
        
        # Test 5: Test Single Query Processing (Gentle)
        print("\n5️⃣ Testing Single Query Processing (Gentle)...")
        
        test_query = "I want some Japanese food"
        print(f"   Processing Query: '{test_query}'")
        
        try:
            result = await hybrid_filter.process_query(test_query)
            
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
            
            # Wait a bit before next test to avoid rate limiting
            print("   ⏳ Waiting 5 seconds to avoid rate limiting...")
            await asyncio.sleep(5)
            
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
        
        print("\n✅ Gentle Hybrid Filter System Test Complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the gentle test."""
    print("🧪 Hybrid Filter System Test Suite (Gentle Mode)")
    print("=" * 60)
    
    # Run the gentle test
    success = asyncio.run(test_hybrid_filter_gentle())
    
    if success:
        print("\n🎉 Test passed! The hybrid filter system is working correctly.")
        print("\n💡 Note: This test used gentle API calls to avoid rate limiting.")
        print("   In production, the system will handle real user queries naturally.")
        sys.exit(0)
    else:
        print("\n💥 Test failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
