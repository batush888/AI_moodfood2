#!/usr/bin/env python3
"""
Generate Secure Frontend Config
----------------------------------
This script generates frontend/config.js with secure configuration
that uses proxy endpoints instead of exposing API keys.
"""

import os
import re
from pathlib import Path

def generate_secure_frontend_config():
    """Generate secure frontend config that uses proxy endpoints."""
    
    config_content = '''// Frontend Configuration
// SECURE CONFIGURATION - No API keys exposed to frontend
// All API calls go through secure backend proxy endpoints
// Auto-generated - DO NOT EDIT MANUALLY
// Run: python scripts/generate_frontend_config.py

const config = {
    // API Configuration
    API_BASE: 'http://localhost:8000',
    
    // Secure Proxy Endpoints
    PROXY: {
        // OpenRouter proxy endpoint
        CHAT: '/api/chat',
        // Gaode geocoding proxy endpoint
        GEOCODE: '/api/geocode',
        // Gaode weather proxy endpoint
        WEATHER: '/api/weather',
        // Health check endpoint
        HEALTH: '/api/health'
    },
    
    // Weather Configuration (no API keys in frontend)
    WEATHER: {
        // Enable fallback weather detection
        ENABLE_FALLBACK: true,
        // Fallback timeout in seconds
        FALLBACK_TIMEOUT: 5,
        // Default coordinates (Beijing)
        DEFAULT_LAT: 39.9042,
        DEFAULT_LNG: 116.4074
    },
    
    // LLM Configuration (no API keys in frontend)
    LLM: {
        MODEL: 'deepseek/deepseek-r1-0528:free',
        MAX_TOKENS: 256,
        TEMPERATURE: 0.0
    },
    
    // UI Configuration
    UI: {
        // Animation duration in milliseconds
        ANIMATION_DURATION: 300,
        // Auto-refresh interval in seconds
        AUTO_REFRESH_INTERVAL: 30,
        // Loading timeout in seconds
        LOADING_TIMEOUT: 10
    }
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = config;
} else {
    // Browser environment
    window.appConfig = config;
}
'''
    
    return config_content

def main():
    """Main function to generate secure frontend config."""
    
    # Paths
    config_path = Path('frontend/config.js')
    
    print("🔧 Generating secure frontend config...")
    print("✅ No API keys will be exposed to frontend")
    print("✅ All API calls will go through secure proxy endpoints")
    
    # Generate secure frontend config
    config_content = generate_secure_frontend_config()
    
    # Write to file
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Secure frontend config generated: {config_path}")
        print("🔒 API keys are kept secure on the backend")
        print("🌐 Frontend uses proxy endpoints for all external API calls")
        
    except Exception as e:
        print(f"❌ Error generating config: {e}")
        return
    
    print("\n🎉 Secure frontend configuration complete!")

if __name__ == "__main__":
    main()
