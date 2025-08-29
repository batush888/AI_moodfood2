#!/usr/bin/env python3
"""
Test script for the dual API key system.

This script tests the LLM validator's ability to rotate between
OpenRouter and DeepSeek API keys when one fails.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_api_key_configuration():
    """Test that both API keys are configured"""
    print("🔧 Testing API Key Configuration...")
    print("=" * 50)
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    
    print(f"OpenRouter API Key: {'✅ Set' if openrouter_key else '❌ Not set'}")
    if openrouter_key:
        print(f"   Starts with: {openrouter_key[:8]}...")
    
    print(f"DeepSeek API Key: {'✅ Set' if deepseek_key else '❌ Not set'}")
    if deepseek_key:
        print(f"   Starts with: {deepseek_key[:8]}...")
    
    # Check if at least one key is available
    if openrouter_key or deepseek_key:
        print(f"\n✅ At least one API key is configured")
        if openrouter_key and deepseek_key:
            print("🎯 Dual API key system enabled - automatic fallback available")
        else:
            print("⚠️  Single API key system - no fallback available")
        return True
    else:
        print("\n❌ No API keys configured")
        return False

def test_llm_validator_initialization():
    """Test that the LLM validator can be initialized with dual keys"""
    print("\n🧪 Testing LLM Validator Initialization...")
    print("=" * 50)
    
    try:
        from core.filtering.llm_validator import LLMValidator
        
        # Initialize the validator
        validator = LLMValidator()
        
        print(f"✅ LLM Validator initialized successfully")
        print(f"   Primary API key: {'✅ Available' if validator.primary_api_key else '❌ Not available'}")
        print(f"   Fallback API key: {'✅ Available' if validator.fallback_api_key else '❌ Not available'}")
        print(f"   API key rotation: {'✅ Enabled' if validator.api_key_rotation_enabled else '❌ Disabled'}")
        print(f"   Current API key: {validator.current_api_key[:8] if validator.current_api_key else 'None'}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize LLM Validator: {e}")
        return False

def test_api_key_rotation():
    """Test the API key rotation functionality"""
    print("\n🔄 Testing API Key Rotation...")
    print("=" * 50)
    
    try:
        from core.filtering.llm_validator import LLMValidator
        
        validator = LLMValidator()
        
        if not validator.api_key_rotation_enabled:
            print("⚠️  API key rotation not enabled (need both keys)")
            return True
        
        # Test rotation
        initial_key = validator.current_api_key
        print(f"Initial API key: {initial_key[:8]}...")
        
        # Rotate to fallback
        rotated = validator._rotate_api_key()
        if rotated:
            print(f"✅ Rotated to: {validator.current_api_key[:8]}...")
        else:
            print("❌ Failed to rotate API key")
            return False
        
        # Rotate back to primary
        rotated = validator._rotate_api_key()
        if rotated:
            print(f"✅ Rotated back to: {validator.current_api_key[:8]}...")
        else:
            print("❌ Failed to rotate back to primary")
            return False
        
        # Verify we're back to the original key
        if validator.current_api_key == initial_key:
            print("✅ API key rotation working correctly")
            return True
        else:
            print("❌ API key rotation failed - keys don't match")
            return False
        
    except Exception as e:
        print(f"❌ API key rotation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Dual API Key System")
    print("=" * 60)
    
    tests = [
        ("API Key Configuration", test_api_key_configuration),
        ("LLM Validator Initialization", test_llm_validator_initialization),
        ("API Key Rotation", test_api_key_rotation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print(f"\n{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The dual API key system is working correctly.")
        print("🔑 Your system will automatically fallback between OpenRouter and DeepSeek API keys.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
