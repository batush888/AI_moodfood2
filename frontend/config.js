// Frontend Configuration
// Auto-generated from .env file - DO NOT EDIT MANUALLY
// Run: python scripts/generate_frontend_config.py

const config = {
    // API Configuration
    API_BASE: 'http://localhost:8000',
    
    // Weather API Configuration
    WEATHER: {
        // Gaode/Amap API Key (from .env file)
        GAODE_API_KEY: 'f2f2807dd694c0e3e46d99bc30e9139f',
        
        // Fallback weather settings
        ENABLE_FALLBACK: true,
        FALLBACK_TIMEOUT: 5000,
        
        // Default coordinates (San Francisco) if geolocation fails
        DEFAULT_LAT: 37.7749,
        DEFAULT_LNG: -122.4194
    },
    
    // LLM API Configuration
    LLM: {
        // OpenRouter API Key (from .env file)
        OPENROUTER_API_KEY: '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',
        MODEL: 'deepseek/deepseek-r1-0528:free'
    },
    
    // UI Configuration
    UI: {
        REFRESH_INTERVAL: 300000, // 5 minutes
        LOADING_TIMEOUT: 10000,   // 10 seconds
        ANIMATION_DURATION: 300   // 300ms
    }
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = config;
} else {
    // Browser environment
    window.appConfig = config;
}
