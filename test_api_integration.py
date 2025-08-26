#!/usr/bin/env python3
"""
Test API integration with Redis-backed global filter stats
"""

import requests
import json
from datetime import datetime

def test_api_integration():
    """Test the API endpoint integration"""
    print("🧪 Testing API Integration with Redis-backed Global Filter")
    print("=" * 60)
    
    try:
        # Test 1: Get initial stats from API
        print("\n1️⃣ Getting initial stats from API...")
        response = requests.get("http://localhost:8000/logging/filter-stats", timeout=10)
        
        if response.status_code == 200:
            initial_stats = response.json()
            print(f"📊 Initial API stats: {json.dumps(initial_stats, indent=2)}")
        else:
            print(f"❌ API request failed with status {response.status_code}")
            return False
            
        # Test 2: Test that the API returns the expected format
        print("\n2️⃣ Verifying API response format...")
        required_fields = ["timestamp", "total_samples", "ml_confident", "llm_fallback", "rejected", "source"]
        
        for field in required_fields:
            if field in initial_stats:
                print(f"   ✅ {field}: {initial_stats[field]}")
            else:
                print(f"   ❌ Missing field: {field}")
                return False
        
        print("✅ API response format is correct")
        
        # Test 3: Verify source information
        print("\n3️⃣ Checking source information...")
        source = initial_stats.get("source", "")
        
        if source in ["redis_global_filter", "local_fallback", "live_hybrid_filter"]:
            print(f"✅ Source is valid: {source}")
        else:
            print(f"⚠️  Unexpected source: {source}")
        
        print("\n🎉 API integration test completed successfully!")
        print(f"📋 Summary:")
        print(f"   - API endpoint: ✅ Working")
        print(f"   - Response format: ✅ Correct")
        print(f"   - Source tracking: ✅ {source}")
        print(f"   - Timestamp: ✅ {initial_stats['timestamp']}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running, skipping API test")
        print("   To test API endpoint, start the server with: PYTHONPATH=. python api/enhanced_routes.py")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 API Integration Test")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    success = test_api_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ API integration test PASSED")
    else:
        print("❌ API integration test FAILED")
