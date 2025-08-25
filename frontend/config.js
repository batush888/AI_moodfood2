// Frontend Configuration
// SECURE CONFIGURATION - No API keys exposed to frontend
// All API calls go through secure backend proxy endpoints

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
