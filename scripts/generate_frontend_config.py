#!/usr/bin/env python3
"""
Generate Frontend Config from .env
----------------------------------
This script reads the .env file and generates frontend/config.js
with the actual API keys for browser use.
"""

import os
import re
from pathlib import Path

def parse_env_file(env_path):
    """Parse .env file and extract key-value pairs."""
    config = {}
    if not os.path.exists(env_path):
        print(f"Warning: {env_path} not found")
        return config
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key] = value
    
    return config

def generate_frontend_config(env_config):
    """Generate frontend/config.js content from .env values."""
    
    # Get API keys from .env
    gaode_api_key = env_config.get('GAODE_API_KEY', '')
    openrouter_api_key = env_config.get('OPENROUTER_API_KEY', '')
    
    config_content = f'''// Frontend Configuration
// Auto-generated from .env file - DO NOT EDIT MANUALLY
// Run: python scripts/generate_frontend_config.py

const config = {{
    // API Configuration
    API_BASE: 'http://localhost:8000',
    
    // Weather API Configuration
    WEATHER: {{
        // Gaode/Amap API Key (from .env file)
        GAODE_API_KEY: '{gaode_api_key}',
        
        // Fallback weather settings
        ENABLE_FALLBACK: true,
        FALLBACK_TIMEOUT: 5000,
        
        // Default coordinates (San Francisco) if geolocation fails
        DEFAULT_LAT: 37.7749,
        DEFAULT_LNG: -122.4194
    }},
    
    // LLM API Configuration
    LLM: {{
        // OpenRouter API Key (from .env file)
        OPENROUTER_API_KEY: '{openrouter_api_key}',
        MODEL: 'deepseek/deepseek-r1-0528:free'
    }},
    
    // UI Configuration
    UI: {{
        REFRESH_INTERVAL: 300000, // 5 minutes
        LOADING_TIMEOUT: 10000,   // 10 seconds
        ANIMATION_DURATION: 300   // 300ms
    }}
}};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = config;
}} else {{
    // Browser environment
    window.appConfig = config;
}}
'''
    
    return config_content

def main():
    """Main function to generate frontend config."""
    
    # Paths
    env_path = Path('.env')
    config_path = Path('frontend/config.js')
    
    print("🔧 Generating frontend config from .env file...")
    
    # Parse .env file
    env_config = parse_env_file(env_path)
    
    if not env_config:
        print("❌ No configuration found in .env file")
        return
    
    # Generate frontend config
    config_content = generate_frontend_config(env_config)
    
    # Write to frontend/config.js
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Generated {config_path}")
    print(f"📋 Found keys: {', '.join(env_config.keys())}")
    
    # Show what was generated
    if env_config.get('GAODE_API_KEY'):
        print(f"🌤️  Gaode API Key: {env_config['GAODE_API_KEY'][:8]}...")
    if env_config.get('OPENROUTER_API_KEY'):
        print(f"🤖 OpenRouter API Key: {env_config['OPENROUTER_API_KEY'][:8]}...")

if __name__ == "__main__":
    main()
