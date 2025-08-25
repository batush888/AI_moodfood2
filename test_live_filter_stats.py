#!/usr/bin/env python3
"""
Test script for Live Hybrid Filter Stats System
"""

import time
import requests
from core.filtering.global_filter import (
    update_global_filter_stats, 
    get_global_filter_live_stats,
    reset_global_filter_stats
)

def test_live_stats():
    """Test the live stats system"""
    
    print("🧪 Testing Live Hybrid Filter Stats System")
    print("=" * 50)
    
    # Reset stats
    reset_global_filter_stats()
    print("✅ Reset global filter stats")
    
    # Simulate some inference decisions
    decisions = [
        "ml_confident",    # ML model confident
        "llm_fallback",    # Used LLM fallback
        "rejected",        # Sample rejected
        "ml_confident",    # ML model confident
        "llm_fallback",    # Used LLM fallback
        "ml_confident",    # ML model confident
        "rejected",        # Sample rejected
        "ml_confident",    # ML model confident
    ]
    
    print(f"\n📊 Simulating {len(decisions)} inference decisions...")
    
    for i, decision in enumerate(decisions, 1):
        update_global_filter_stats(decision)
        print(f"  Decision {i}: {decision}")
        
        # Get live stats after each update
        stats = get_global_filter_live_stats()
        print(f"    Live stats: ML={stats['ml_confident']}, LLM={stats['llm_fallback']}, Rejected={stats['rejected']}")
        
        time.sleep(0.5)  # Small delay to see progression
    
    # Show final stats
    print(f"\n🎯 Final Live Stats:")
    final_stats = get_global_filter_live_stats()
    print(f"  Total samples: {final_stats['total_samples']}")
    print(f"  ML confident: {final_stats['ml_confident']}")
    print(f"  LLM fallback: {final_stats['llm_fallback']}")
    print(f"  Rejected: {final_stats['rejected']}")
    print(f"  Timestamp: {final_stats['timestamp']}")
    
    # Test the API endpoint
    print(f"\n🌐 Testing API endpoint...")
    try:
        response = requests.get("http://localhost:8000/logging/filter-stats", timeout=5)
        if response.status_code == 200:
            api_stats = response.json()
            print(f"✅ API endpoint working!")
            print(f"  Source: {api_stats.get('source', 'unknown')}")
            print(f"  Total: {api_stats.get('total_samples', 0)}")
            print(f"  ML: {api_stats.get('ml_confident', 0)}")
            print(f"  LLM: {api_stats.get('llm_fallback', 0)}")
            print(f"  Rejected: {api_stats.get('rejected', 0)}")
        else:
            print(f"❌ API endpoint returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ API server not running (start with: PYTHONPATH=. python api/enhanced_routes.py)")
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    test_live_stats()
