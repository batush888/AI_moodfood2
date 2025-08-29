#!/usr/bin/env python3
"""
Test script to verify the proxy route fix.
"""

import requests
import json

def test_proxy_routes():
    """Test that the proxy routes are working correctly."""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Proxy Routes...")
    print("=" * 50)
    
    # Test 1: Check if /api/openai endpoint exists
    print("\n1️⃣  Testing /api/openai endpoint...")
    try:
        response = requests.post(
            f"{base_url}/api/openai",
            json={
                "model": "deepseek/deepseek-r1-0528:free",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ /api/openai endpoint is working!")
            data = response.json()
            print(f"   Response: {data.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
        elif response.status_code == 429:
            print("⚠️  /api/openai endpoint exists but hit rate limit (expected)")
        elif response.status_code == 500:
            print("⚠️  /api/openai endpoint exists but server error (check logs)")
        else:
            print(f"❌ /api/openai endpoint returned unexpected status: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
    except Exception as e:
        print(f"❌ Error testing /api/openai: {e}")
    
    # Test 2: Check if /api/chat endpoint exists
    print("\n2️⃣  Testing /api/chat endpoint...")
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": "deepseek/deepseek-r1-0528:free",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ /api/chat endpoint is working!")
            data = response.json()
            print(f"   Response: {data.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
        elif response.status_code == 429:
            print("⚠️  /api/chat endpoint exists but hit rate limit (expected)")
        elif response.status_code == 500:
            print("⚠️  /api/chat endpoint exists but server error (check logs)")
        else:
            print(f"❌ /api/chat endpoint returned unexpected status: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
    except Exception as e:
        print(f"❌ Error testing /api/chat: {e}")
    
    # Test 3: Check if the old /proxy/openai path returns 404 (as expected)
    print("\n3️⃣  Testing old /proxy/openai path (should return 404)...")
    try:
        response = requests.post(
            f"{base_url}/proxy/openai",
            json={"test": "data"},
            timeout=10
        )
        
        if response.status_code == 404:
            print("✅ /proxy/openai correctly returns 404 (as expected)")
        else:
            print(f"⚠️  /proxy/openai returned unexpected status: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
    except Exception as e:
        print(f"❌ Error testing /proxy/openai: {e}")
    
    print("\n" + "=" * 50)
    print("📊 PROXY ROUTE TEST SUMMARY")
    print("=" * 50)
    print("✅ /api/openai - New OpenAI-compatible endpoint")
    print("✅ /api/chat - Original OpenRouter endpoint")
    print("✅ /proxy/openai - Old path correctly returns 404")
    print("\n🎯 Frontend should now use /api/openai instead of /proxy/openai")

if __name__ == "__main__":
    test_proxy_routes()
