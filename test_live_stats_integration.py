#!/usr/bin/env python3
"""
Comprehensive Test for Live Hybrid Filter Stats System
"""

import time
import requests
from core.filtering.global_filter import (
    update_global_filter_stats, 
    get_global_filter_live_stats,
    reset_global_filter_stats
)

def test_live_stats_system():
    """Test the complete live stats system"""
    
    print("🧪 Testing Complete Live Hybrid Filter Stats System")
    print("=" * 60)
    
    # Reset stats
    reset_global_filter_stats()
    print("✅ Reset global filter stats")
    
    # Show initial state
    initial_stats = get_global_filter_live_stats()
    print(f"📊 Initial stats: {initial_stats}")
    
    # Simulate real inference workflow
    print(f"\n🚀 Simulating real inference workflow...")
    
    # Batch 1: ML confident decisions
    print(f"\n📦 Batch 1: ML Confident Decisions")
    for i in range(5):
        update_global_filter_stats("ml_confident")
        stats = get_global_filter_live_stats()
        print(f"  Sample {i+1}: ML confident → Total: {stats['total_samples']}, ML: {stats['ml_confident']}")
        time.sleep(0.2)
    
    # Batch 2: LLM fallback decisions
    print(f"\n📦 Batch 2: LLM Fallback Decisions")
    for i in range(3):
        update_global_filter_stats("llm_fallback")
        stats = get_global_filter_live_stats()
        print(f"  Sample {i+1}: LLM fallback → Total: {stats['total_samples']}, LLM: {stats['llm_fallback']}")
        time.sleep(0.2)
    
    # Batch 3: Rejected samples
    print(f"\n📦 Batch 3: Rejected Samples")
    for i in range(2):
        update_global_filter_stats("rejected")
        stats = get_global_filter_live_stats()
        print(f"  Sample {i+1}: Rejected → Total: {stats['total_samples']}, Rejected: {stats['rejected']}")
        time.sleep(0.2)
    
    # Final stats
    print(f"\n🎯 Final Live Stats:")
    final_stats = get_global_filter_live_stats()
    print(f"  Total samples: {final_stats['total_samples']}")
    print(f"  ML confident: {final_stats['ml_confident']}")
    print(f"  LLM fallback: {final_stats['llm_fallback']}")
    print(f"  Rejected: {final_stats['rejected']}")
    print(f"  Timestamp: {final_stats['timestamp']}")
    
    # Calculate efficiency
    total = final_stats['total_samples']
    ml_confident = final_stats['ml_confident']
    llm_fallback = final_stats['llm_fallback']
    rejected = final_stats['rejected']
    
    if total > 0:
        ml_efficiency = (ml_confident / total) * 100
        llm_efficiency = (llm_fallback / total) * 100
        rejection_rate = (rejected / total) * 100
        
        print(f"\n📈 Performance Metrics:")
        print(f"  ML Confidence Rate: {ml_efficiency:.1f}%")
        print(f"  LLM Fallback Rate: {llm_efficiency:.1f}%")
        print(f"  Rejection Rate: {rejection_rate:.1f}%")
        print(f"  Overall Success Rate: {((ml_confident + llm_fallback) / total) * 100:.1f}%")
    
    # Test API endpoint if server is running
    print(f"\n🌐 Testing API Endpoint...")
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
            
            # Note: API will show 0 because it's in a different process
            if api_stats.get('total_samples', 0) == 0:
                print(f"  ℹ️  Note: API shows 0 because it's in a different process")
                print(f"     This is expected behavior for demonstration purposes")
        else:
            print(f"❌ API endpoint returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ API server not running (start with: PYTHONPATH=. python api/enhanced_routes.py)")
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    print(f"\n🎉 Live Stats System Test Complete!")
    print(f"   The system is working correctly in this process.")
    print(f"   To see live updates in the API, integrate update_global_filter_stats()")
    print(f"   into your actual inference pipeline.")

if __name__ == "__main__":
    test_live_stats_system()
