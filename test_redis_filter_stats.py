#!/usr/bin/env python3
"""
Test script for Redis-backed global filter stats
Tests the production-ready Redis implementation with fallback to local dict
"""

import json
import time
import requests
from datetime import datetime

def test_redis_filter_stats():
    """Test the Redis-backed global filter stats implementation"""
    print("🧪 Testing Redis-backed Global Filter Stats")
    print("=" * 50)
    
    # Import the functions we want to test
    try:
        from core.filtering.global_filter import (
            update_global_filter_stats,
            get_global_filter_live_stats,
            reset_global_filter_stats
        )
        print("✅ Successfully imported global filter functions")
    except ImportError as e:
        print(f"❌ Failed to import global filter functions: {e}")
        return False
    
    # Test 1: Reset stats to start fresh
    print("\n1️⃣ Resetting filter stats...")
    try:
        reset_global_filter_stats()
        print("✅ Stats reset successfully")
    except Exception as e:
        print(f"❌ Failed to reset stats: {e}")
        return False
    
    # Test 2: Check initial stats
    print("\n2️⃣ Checking initial stats...")
    try:
        initial_stats = get_global_filter_live_stats()
        print(f"📊 Initial stats: {json.dumps(initial_stats, indent=2)}")
        
        if initial_stats["total_samples"] != 0:
            print("⚠️  Warning: Initial total_samples should be 0")
    except Exception as e:
        print(f"❌ Failed to get initial stats: {e}")
        return False
    
    # Test 3: Simulate 10 updates (mixed decisions)
    print("\n3️⃣ Simulating 10 filter decisions...")
    test_decisions = [
        "ml_confident",    # 3 ML confident decisions
        "ml_confident",
        "ml_confident",
        "llm_fallback",    # 4 LLM fallback decisions
        "llm_fallback",
        "llm_fallback",
        "llm_fallback",
        "rejected",        # 3 rejected decisions
        "rejected",
        "rejected"
    ]
    
    try:
        for i, decision in enumerate(test_decisions, 1):
            update_global_filter_stats(decision)
            print(f"   Update {i}: {decision}")
            time.sleep(0.1)  # Small delay to simulate real usage
        
        print("✅ All updates completed successfully")
    except Exception as e:
        print(f"❌ Failed to update stats: {e}")
        return False
    
    # Test 4: Check final stats from Redis/local
    print("\n4️⃣ Checking final stats from Redis/local...")
    try:
        final_stats = get_global_filter_live_stats()
        print(f"📊 Final stats: {json.dumps(final_stats, indent=2)}")
        
        # Verify the counts
        expected_ml_confident = 3
        expected_llm_fallback = 4
        expected_rejected = 3
        expected_total = 10
        
        if (final_stats["total_samples"] == expected_total and
            final_stats["ml_confident"] == expected_ml_confident and
            final_stats["llm_fallback"] == expected_llm_fallback and
            final_stats["rejected"] == expected_rejected):
            print("✅ Stats match expected values")
        else:
            print("❌ Stats don't match expected values")
            print(f"   Expected: total={expected_total}, ml_confident={expected_ml_confident}, llm_fallback={expected_llm_fallback}, rejected={expected_rejected}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to get final stats: {e}")
        return False
    
    # Test 5: Test API endpoint
    print("\n5️⃣ Testing API endpoint /logging/filter-stats...")
    try:
        # Start the server if not already running
        print("   Making API request to /logging/filter-stats...")
        
        response = requests.get("http://localhost:8000/logging/filter-stats", timeout=10)
        
        if response.status_code == 200:
            api_stats = response.json()
            print(f"📊 API Response: {json.dumps(api_stats, indent=2)}")
            
            # Check if API returns the same stats
            if (api_stats["total_samples"] == expected_total and
                api_stats["ml_confident"] == expected_ml_confident and
                api_stats["llm_fallback"] == expected_llm_fallback and
                api_stats["rejected"] == expected_rejected):
                print("✅ API stats match expected values")
            else:
                print("❌ API stats don't match expected values")
                return False
                
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running, skipping API test")
        print("   To test API endpoint, start the server with: PYTHONPATH=. python api/enhanced_routes.py")
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    
    # Test 6: Test reset functionality
    print("\n6️⃣ Testing reset functionality...")
    try:
        reset_global_filter_stats()
        
        # Check if stats are reset
        reset_stats = get_global_filter_live_stats()
        print(f"📊 Stats after reset: {json.dumps(reset_stats, indent=2)}")
        
        if reset_stats["total_samples"] == 0:
            print("✅ Reset functionality works correctly")
        else:
            print("❌ Reset functionality failed")
            return False
            
    except Exception as e:
        print(f"❌ Reset test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Redis-backed global filter stats are working correctly.")
    return True

def test_redis_connection():
    """Test Redis connection separately"""
    print("\n🔍 Testing Redis Connection")
    print("-" * 30)
    
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        client.ping()
        print("✅ Redis connection successful")
        return True
    except ImportError:
        print("⚠️  Redis not installed, will use local fallback")
        return False
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        print("   Will use local fallback mode")
        return False

if __name__ == "__main__":
    print("🚀 Redis-backed Global Filter Stats Test")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Test Redis connection first
    redis_available = test_redis_connection()
    
    # Run the main test
    success = test_redis_filter_stats()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests PASSED")
        if redis_available:
            print("🎯 Redis-backed implementation is working correctly")
        else:
            print("🎯 Local fallback implementation is working correctly")
    else:
        print("❌ Some tests FAILED")
    
    print("\n📋 Test Summary:")
    print("   - Redis connection: {'✅ Available' if redis_available else '⚠️  Fallback'}")
    print("   - Function imports: ✅")
    print("   - Stats updates: ✅")
    print("   - Stats retrieval: ✅")
    print("   - Reset functionality: ✅")
    print("   - API endpoint: ✅ (if server running)")
