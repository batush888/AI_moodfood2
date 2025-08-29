// Frontend Configuration
// This file contains configuration settings for the frontend

const CONFIG = {
    // API Configuration
    API_BASE: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
        ? 'http://localhost:8000' 
        : window.location.origin,
    
    // Feature Flags
    ENABLE_DEBUG: true,
    ENABLE_ANALYTICS: false,
    ENABLE_FEEDBACK: true,
    
    // UI Configuration
    MAX_RECOMMENDATIONS: 10,
    ANIMATION_DURATION: 300,
    LOADING_TIMEOUT: 30000,
    
    // Default Values
    DEFAULT_QUERY_PLACEHOLDER: "What are you craving? Describe your mood, preferences, or any specific food you'd like...",
    DEFAULT_ERROR_MESSAGE: "Something went wrong. Please try again.",
    
    // Social Context Detection
    SOCIAL_CONTEXT_KEYWORDS: {
        'family': ['family', 'kids', 'children', 'parents', 'grandparents'],
        'romantic': ['romantic', 'date', 'anniversary', 'valentine', 'couple'],
        'party': ['party', 'celebration', 'birthday', 'anniversary', 'festival'],
        'business': ['business', 'meeting', 'lunch', 'dinner', 'client'],
        'casual': ['casual', 'quick', 'simple', 'easy', 'fast']
    }
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
} else {
    window.CONFIG = CONFIG;
}
