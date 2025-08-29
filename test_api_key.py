#!/usr/bin/env python3
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_api_key(api_key: str | None, key_name: str) -> bool:
    """Test an API key with OpenRouter API"""
    
    if not api_key:
        print(f"❌ {key_name} environment variable not set")
        return False
    
    print(f"🔑 Testing {key_name}: {api_key[:20]}...")
    
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
            print(f"✅ {key_name} is valid!")
            print(f"📊 Model: {data.get('model', 'Unknown')}")
            print(f"🔢 Tokens used: {data.get('usage', {}).get('total_tokens', 'Unknown')}")
            return True
        else:
            error_data = response.json() if response.content else {}
            print(f"❌ {key_name} API error: {error_data}")
            return False
            
    except Exception as e:
        print(f"❌ {key_name} request failed: {e}")
        return False

def test_dual_api_keys():
    """Test both API keys for redundancy"""
    
    print("🧪 Testing Dual API Key System...")
    print("=" * 50)
    
    # Test OpenRouter API key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openrouter_success = test_api_key(openrouter_key, "OPENROUTER_API_KEY")
    
    print()
    
    # Test DeepSeek API key
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    deepseek_success = test_api_key(deepseek_key, "DEEPSEEK_API_KEY")
    
    print("\n" + "=" * 50)
    print("📊 DUAL API KEY TEST RESULTS")
    print("=" * 50)
    
    if openrouter_success and deepseek_success:
        print("🎉 Both API keys are working correctly!")
        print("✅ Redundancy achieved - system will automatically fallback if one fails")
    elif openrouter_success or deepseek_success:
        working_key = "OpenRouter" if openrouter_success else "DeepSeek"
        print(f"⚠️  Only {working_key} API key is working")
        print("💡 Consider fixing the non-working key for redundancy")
    else:
        print("❌ No API keys are working")
        print("💡 Check your API key configuration")
    
    # Summary
    print(f"\n🔑 API Key Status:")
    print(f"   • OpenRouter: {'✅ Working' if openrouter_success else '❌ Failed'}")
    print(f"   • DeepSeek: {'✅ Working' if deepseek_success else '❌ Failed'}")
    
    return openrouter_success or deepseek_success

if __name__ == "__main__":
    success = test_dual_api_keys()
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Check if the API keys are correct")
        print("2. Verify the API keys haven't expired")
        print("3. Check OpenRouter account status")
        print("4. Ensure you have credits/quota available")
        print("5. Verify environment variables are set correctly")
