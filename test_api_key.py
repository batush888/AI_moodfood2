#!/usr/bin/env python3
import os
import requests
import json

def test_openrouter_api():
    """Test OpenRouter API with the current API key"""
    
    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        return False
    
    print(f"🔑 Testing API key: {api_key[:20]}...")
    
    # Test with a simple request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Mood Food Recommender"
    }
    
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [
            {"role": "user", "content": "Hello, this is a test."}
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API key is valid!")
            print(f"📊 Model: {data.get('model', 'Unknown')}")
            print(f"🔢 Tokens used: {data.get('usage', {}).get('total_tokens', 'Unknown')}")
            return True
        else:
            error_data = response.json() if response.content else {}
            print(f"❌ API error: {error_data}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing OpenRouter API Key...")
    success = test_openrouter_api()
    
    if success:
        print("\n🎉 API key is working correctly!")
    else:
        print("\n💡 Troubleshooting tips:")
        print("1. Check if the API key is correct")
        print("2. Verify the API key hasn't expired")
        print("3. Check OpenRouter account status")
        print("4. Ensure you have credits/quota available")
