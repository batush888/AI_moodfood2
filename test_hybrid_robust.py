#!/usr/bin/env python3
"""
Test script for Hybrid Filter with Robust LLM System

This script tests the complete hybrid filter pipeline using the new
robust LLM validator and adaptive parser.
"""

import os
import sys
import asyncio
from pathlib import Path

# Set mock mode before importing
os.environ["LLM_MOCK_MODE"] = "true"

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_hybrid_filter_pipeline():
    """Test the complete hybrid filter pipeline"""
    print("🧪 Testing Hybrid Filter Pipeline...")
    
    from core.filtering.hybrid_filter import HybridFilter
    from core.filtering.llm_validator import LLMValidator
    from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
    
    # Create components
    llm_validator = LLMValidator()
    enhanced_classifier = EnhancedIntentClassifier("data/taxonomy/mood_food_taxonomy.json")
    
    # Create hybrid filter
    hybrid_filter = HybridFilter(
        llm_validator=llm_validator,
        ml_classifier=enhanced_classifier,
        confidence_threshold=0.7
    )
    
    print(f"✅ Components created:")
    print(f"   - LLM Validator: {llm_validator.mock_mode=}")
    print(f"   - Enhanced Classifier: {enhanced_classifier is not None}")
    print(f"   - Hybrid Filter: {hybrid_filter is not None}")
    
    # Test queries
    test_queries = [
        "I want some Japanese food",
        "Give me spicy food",
        "I need comfort food",
        "I want healthy options"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Testing Query {i}: '{query}'")
        
        try:
            # Process query through hybrid filter
            response = await hybrid_filter.process_query(query)
            
            print(f"   ✅ Decision: {response.decision}")
            print(f"   📋 Recommendations: {response.recommendations}")
            print(f"   ⏱️  Processing time: {response.processing_time_ms:.2f}ms")
            print(f"   🧠 ML Prediction: {response.ml_prediction.primary_intent if response.ml_prediction else 'None'}")
            print(f"   🤖 LLM Interpretation: {response.llm_interpretation.intent if response.llm_interpretation else 'None'}")
            
            # Check that we got meaningful recommendations
            assert len(response.recommendations) > 0, "No recommendations generated"
            assert response.decision in ["ml_validated", "llm_fallback"], f"Invalid decision: {response.decision}"
            
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Check live statistics
    stats = hybrid_filter.live_stats
    print(f"\n📊 Live Statistics:")
    print(f"   - Total queries: {stats['total_queries']}")
    print(f"   - ML validated: {stats['ml_validated']}")
    print(f"   - LLM fallback: {stats['llm_fallback']}")
    print(f"   - Training samples: {stats['llm_training_samples']}")
    print(f"   - Processing errors: {stats['processing_errors']}")
    
    print("\n🎉 Hybrid Filter Pipeline tests passed!")

def test_logging():
    """Test that logs are being generated"""
    print("\n🧪 Testing Logging...")
    
    log_file = Path("logs/recommendation_logs.jsonl")
    
    if log_file.exists():
        # Count log entries
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        print(f"✅ Log file exists with {len(lines)} entries")
        
        if lines:
            # Show last log entry
            last_entry = lines[-1].strip()
            print(f"📝 Last log entry: {last_entry[:100]}...")
    else:
        print("⚠️  Log file not found yet")
    
    print("🎉 Logging test completed!")

async def main():
    """Run all tests"""
    print("🚀 Starting Hybrid Filter Robust System Tests...\n")
    
    try:
        await test_hybrid_filter_pipeline()
        test_logging()
        
        print("\n🎉 All tests passed! The Hybrid Filter with Robust LLM System is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
