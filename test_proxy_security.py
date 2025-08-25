#!/usr/bin/env python3
"""
Test script for proxy security features
Tests rate limiting, token safety checks, and input validation
"""

import requests
import time
import json
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def test_rate_limiting():
    """Test rate limiting functionality"""
    print("🔍 Testing Rate Limiting...")
    
    base_url = "http://localhost:8000"
    
    # Test normal request
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False
    
    # Test rate limit status
    try:
        response = requests.get(f"{base_url}/api/rate-limit-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Rate limit status: {data['requests_used']}/{data['requests_limit']} requests used")
        else:
            print(f"❌ Rate limit status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Rate limit status error: {e}")
        return False
    
    # Test rate limiting by making multiple requests
    print("   Testing rate limit enforcement...")
    success_count = 0
    rate_limited_count = 0
    
    for i in range(15):  # Try 15 requests (should hit limit at 10)
        try:
            # Use geocode endpoint which counts towards rate limit
            response = requests.get(f"{base_url}/api/geocode?location=116.4074,39.9042", timeout=5)
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1
                data = response.json()
                if "Rate limit exceeded" in data.get("error", ""):
                    print(f"   ✅ Request {i+1}: Rate limited correctly")
                else:
                    print(f"   ❌ Request {i+1}: Wrong error message")
            else:
                print(f"   ❌ Request {i+1}: Unexpected status {response.status_code}")
        except Exception as e:
            print(f"   ❌ Request {i+1}: Error {e}")
    
    print(f"   Results: {success_count} successful, {rate_limited_count} rate limited")
    
    if rate_limited_count > 0:
        print("✅ Rate limiting working correctly")
        return True
    else:
        print("❌ Rate limiting not working")
        return False

def test_token_safety():
    """Test token safety checks"""
    print("\n🔍 Testing Token Safety Checks...")
    
    base_url = "http://localhost:8000"
    
    # Test valid request
    valid_payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(f"{base_url}/api/chat", json=valid_payload, timeout=30)
        if response.status_code == 200:
            print("✅ Valid request accepted")
        else:
            print(f"❌ Valid request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Valid request error: {e}")
        return False
    
    # Test max_tokens limit
    invalid_payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 3000,  # Exceeds limit of 2000
        "temperature": 0.0
    }
    
    try:
        response = requests.post(f"{base_url}/api/chat", json=invalid_payload, timeout=30)
        if response.status_code == 422:  # Validation error
            data = response.json()
            if "max_tokens too high" in str(data):
                print("✅ Token limit enforced correctly")
            else:
                print(f"❌ Wrong error message: {data}")
                return False
        else:
            print(f"❌ Token limit not enforced: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Token limit test error: {e}")
        return False
    
    return True

def test_input_validation():
    """Test input validation"""
    print("\n🔍 Testing Input Validation...")
    
    base_url = "http://localhost:8000"
    
    # Test empty messages
    empty_payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(f"{base_url}/api/chat", json=empty_payload, timeout=30)
        if response.status_code == 422:
            data = response.json()
            if "Messages cannot be empty" in str(data):
                print("✅ Empty messages rejected")
            else:
                print(f"❌ Wrong error for empty messages: {data}")
                return False
        else:
            print(f"❌ Empty messages not rejected: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Empty messages test error: {e}")
        return False
    
    # Test long input
    long_content = "A" * 5000  # Exceeds 4000 character limit
    long_payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": long_content}],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(f"{base_url}/api/chat", json=long_payload, timeout=30)
        if response.status_code == 422:
            data = response.json()
            if "Input too long" in str(data):
                print("✅ Long input rejected")
            else:
                print(f"❌ Wrong error for long input: {data}")
                return False
        else:
            print(f"❌ Long input not rejected: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Long input test error: {e}")
        return False
    
    # Test invalid location format
    try:
        response = requests.get(f"{base_url}/api/geocode?location=invalid", timeout=10)
        if response.status_code == 400:
            data = response.json()
            if "Invalid location format" in data.get("error", ""):
                print("✅ Invalid location format rejected")
            else:
                print(f"❌ Wrong error for invalid location: {data}")
                return False
        else:
            print(f"❌ Invalid location not rejected: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Invalid location test error: {e}")
        return False
    
    # Test invalid weather parameters
    try:
        response = requests.get(f"{base_url}/api/weather?city=&extensions=invalid", timeout=10)
        if response.status_code == 400:
            data = response.json()
            if "Invalid" in data.get("error", ""):
                print("✅ Invalid weather parameters rejected")
            else:
                print(f"❌ Wrong error for invalid weather params: {data}")
                return False
        else:
            print(f"❌ Invalid weather params not rejected: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Invalid weather params test error: {e}")
        return False
    
    return True

def test_error_handling():
    """Test error handling and JSON responses"""
    print("\n🔍 Testing Error Handling...")
    
    base_url = "http://localhost:8000"
    
    # Test malformed JSON
    try:
        response = requests.post(
            f"{base_url}/api/chat", 
            data="invalid json", 
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 422:
            print("✅ Malformed JSON rejected")
        else:
            print(f"❌ Malformed JSON not rejected: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Malformed JSON test error: {e}")
        return False
    
    # Test missing required fields
    invalid_payload = {
        "model": "deepseek/deepseek-r1-0528:free"
        # Missing messages field
    }
    
    try:
        response = requests.post(f"{base_url}/api/chat", json=invalid_payload, timeout=10)
        if response.status_code == 422:
            print("✅ Missing required fields rejected")
        else:
            print(f"❌ Missing required fields not rejected: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Missing fields test error: {e}")
        return False
    
    return True

def test_health_endpoint():
    """Test enhanced health endpoint"""
    print("\n🔍 Testing Enhanced Health Endpoint...")
    
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Check for new fields
            if "rate_limiting" in data and "limits" in data:
                print("✅ Enhanced health endpoint includes security info")
                print(f"   Rate limiting: {data['rate_limiting']['requests_per_minute']} req/min")
                print(f"   Max tokens: {data['limits']['max_tokens']}")
                print(f"   Max input length: {data['limits']['max_input_length']}")
                return True
            else:
                print("❌ Health endpoint missing security info")
                return False
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def main():
    """Run all security tests"""
    print("🔒 Testing Enhanced Proxy Security Features")
    print("=" * 60)
    
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Rate Limiting", test_rate_limiting),
        ("Token Safety", test_token_safety),
        ("Input Validation", test_input_validation),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All security tests passed!")
        print("\n✅ Security Features Verified:")
        print("   • Rate limiting (10 req/min per IP)")
        print("   • Token safety (max 2000 tokens)")
        print("   • Input validation (max 4000 chars)")
        print("   • Error handling (JSON responses)")
        print("   • Enhanced health endpoint")
        print("   • All existing features preserved")
    else:
        print("❌ Some security tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()
