#!/usr/bin/env python3
"""
Test script for the new /logging/filter-stats endpoint
"""

import requests
import json
from datetime import datetime

def test_filter_stats_endpoint():
    """Test the new filter stats endpoint"""
    
    base_url = "http://localhost:8000"
    endpoint = "/logging/filter-stats"
    
    print(f"🧪 Testing {base_url}{endpoint}")
    print("=" * 50)
    
    try:
        # Test the endpoint
        response = requests.get(f"{base_url}{endpoint}", timeout=10)
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"📡 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Response Data:")
            print(json.dumps(data, indent=2, default=str))
            
            # Validate response structure
            required_fields = ['timestamp', 'total_samples', 'ml_confident', 'llm_fallback', 'rejected']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
            else:
                print(f"✅ All required fields present")
                
            # Check if we have actual data or fallback
            if data.get('note') == 'no stats yet':
                print(f"ℹ️  Endpoint working but no filter stats available yet")
            elif data.get('source'):
                print(f"✅ Endpoint working with data from: {data['source']}")
            else:
                print(f"ℹ️  Endpoint working with fallback data")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed - is the server running on {base_url}?")
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_filter_stats_endpoint()
