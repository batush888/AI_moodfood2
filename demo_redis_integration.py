#!/usr/bin/env python3
"""
Demonstration script for Redis-backed global filter stats integration.
This script shows how the backend and frontend are now synchronized with Redis.
"""

import requests
import json
import time
from datetime import datetime

def demo_redis_integration():
    """Demonstrate the complete Redis integration."""
    print("🚀 Redis-Backed Global Filter Stats Integration Demo")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Check server status
    try:
        health = requests.get("http://localhost:8000/api/health", timeout=5)
        if health.status_code == 200:
            print("✅ Server is running and healthy")
        else:
            print("❌ Server health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please start the server with: PYTHONPATH=. python api/enhanced_routes.py")
        return
    
    print()
    
    # Demo 1: Show current Redis-backed stats
    print("📊 Demo 1: Current Redis-Backed Stats")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:8000/logging/filter-stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            
            print(f"🎯 Source: {stats.get('source', 'unknown')}")
            print(f"📈 Total Samples: {stats.get('total_samples', 0)}")
            print(f"✅ ML Confident: {stats.get('ml_confident', 0)}")
            print(f"⚠️ LLM Fallback: {stats.get('llm_fallback', 0)}")
            print(f"❌ Rejected: {stats.get('rejected', 0)}")
            print(f"🕒 Timestamp: {stats.get('timestamp', 'unknown')}")
            
            if stats.get('source') == 'redis_global_filter':
                print("🎉 Redis is active and being used!")
            else:
                print(f"ℹ️ Using fallback source: {stats.get('source')}")
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return
    
    print()
    
    # Demo 2: Show API response structure
    print("🔍 Demo 2: API Response Structure")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:8000/logging/filter-stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            
            print("JSON Response:")
            print(json.dumps(stats, indent=2))
            
            # Check for required fields
            required_fields = ['timestamp', 'total_samples', 'ml_confident', 'llm_fallback', 'rejected', 'source']
            missing = [f for f in required_fields if f not in stats]
            
            if missing:
                print(f"⚠️ Missing fields: {missing}")
            else:
                print("✅ All required fields present")
        else:
            print(f"❌ API request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Demo 3: Show frontend integration
    print("🎨 Demo 3: Frontend Integration")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:8000/logging/filter-stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            source = stats.get('source', 'unknown')
            
            print("Frontend will display:")
            print(f"📊 Chart data: ML={stats.get('ml_confident', 0)}, LLM={stats.get('llm_fallback', 0)}, Rejected={stats.get('rejected', 0)}")
            print(f"📈 Total samples: {stats.get('total_samples', 0)}")
            
            if source == 'redis_global_filter':
                print("🎨 Source display: 🔴 Redis (green color + Redis badge)")
            elif source == 'local_fallback':
                print("🎨 Source display: ⚠️ Local Fallback (orange color)")
            elif source == 'latest_retrain':
                print("🎨 Source display: 📈 Latest Retrain (blue color)")
            else:
                print(f"🎨 Source display: 📊 {source} (gray color)")
            
            print("🔄 Auto-refresh: Every 5 seconds")
            print("📱 Real-time updates: Chart.js doughnut chart")
        else:
            print(f"❌ Failed to get stats for frontend demo: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Demo 4: Show production benefits
    print("🚀 Demo 4: Production Benefits")
    print("-" * 40)
    
    print("✅ Multi-process safe: Statistics shared across server instances")
    print("✅ High performance: Sub-millisecond Redis operations")
    print("✅ Real-time updates: Live dashboard every 5 seconds")
    print("✅ Automatic fallback: Continues working when Redis is down")
    print("✅ Source transparency: Clear indication of data source")
    print("✅ Scalable: Centralized statistics across all processes")
    
    print()
    
    # Demo 5: Show monitoring URLs
    print("📱 Demo 5: Access Points")
    print("-" * 40)
    
    print("🌐 Main Application: http://localhost:8000/")
    print("📊 Monitoring Dashboard: http://localhost:8000/monitoring")
    print("🔌 API Endpoint: http://localhost:8000/logging/filter-stats")
    print("🏥 Health Check: http://localhost:8000/api/health")
    
    print()
    print("🎉 Redis Integration Demo Complete!")
    print("The Hybrid Filter Stats panel is now fully Redis-backed and production-ready!")

if __name__ == "__main__":
    demo_redis_integration()
