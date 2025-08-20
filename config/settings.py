#!/usr/bin/env python3
"""
Configuration settings for the AI Mood-Based Food Recommendation System
"""

import os
from typing import Tuple

# ============================================================================
# Filtering Configuration
# ============================================================================

# ML-based filtering thresholds
MIN_CONFIDENCE_THRESHOLD = 0.5
LLM_FALLBACK_RANGE = (0.45, 0.55)  # Range for borderline cases

# Data quality thresholds
MIN_SAMPLE_LENGTH = 3  # Minimum characters for valid query
MAX_SAMPLE_LENGTH = 500  # Maximum characters for valid query

# ============================================================================
# LLM API Configuration
# ============================================================================

# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = "deepseek/deepseek-r1-0528:free"
LLM_MAX_TOKENS = 50
LLM_TEMPERATURE = 0.0

# LLM validation settings
LLM_VALIDATION_TIMEOUT = 10  # seconds
LLM_MAX_RETRIES = 3

# ============================================================================
# Logging Configuration
# ============================================================================

# Log file paths
FILTER_STATS_FILE = "data/logs/filter_stats.jsonl"
RETRAIN_HISTORY_FILE = "data/logs/retrain_history.json"
QUERY_LOGS_FILE = "data/logs/query_logs.jsonl"
ERROR_LOGS_FILE = "data/logs/error_logs.jsonl"

# ============================================================================
# Model Configuration
# ============================================================================

# Model directories
MODEL_DIR = "models/intent_classifier"
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")

# Training configuration
MIN_TRAINING_SAMPLES = 10
PERFORMANCE_TOLERANCE = 0.01  # 1% tolerance for performance degradation

# ============================================================================
# Scheduler Configuration
# ============================================================================

# Retraining schedules
WEEKLY_RETRAIN = {
    'enabled': True,
    'day_of_week': 'sun',  # Sunday
    'hour': 3,             # 3 AM
    'minute': 0
}

MONTHLY_RETRAIN = {
    'enabled': True,
    'day': 1,              # 1st of month
    'hour': 2,             # 2 AM
    'minute': 0
}

ADAPTIVE_RETRAIN = {
    'enabled': True,
    'min_samples': 100,    # Retrain when 100+ new samples
    'min_days': 7,         # Or at least 7 days
    'check_interval_hours': 6  # Check every 6 hours
}

# ============================================================================
# API Configuration
# ============================================================================

# Server settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# CORS settings
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://localhost:8002",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8002"
]

# ============================================================================
# Validation Functions
# ============================================================================

def validate_config() -> bool:
    """Validate that all required configuration is present"""
    errors = []
    
    # Check API key
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY not set in environment")
    
    # Check directories
    required_dirs = [
        "data/logs",
        "models/intent_classifier",
        "config"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {e}")
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True

def get_llm_fallback_range() -> Tuple[float, float]:
    """Get the LLM fallback range for borderline cases"""
    return LLM_FALLBACK_RANGE

def is_borderline_case(confidence: float) -> bool:
    """Check if a sample is in the borderline range for LLM validation"""
    min_conf, max_conf = LLM_FALLBACK_RANGE
    return min_conf <= confidence <= max_conf

def is_valid_sample(query: str, confidence: float) -> bool:
    """Basic validation for a training sample"""
    if not query or not isinstance(query, str):
        return False
    
    if len(query.strip()) < MIN_SAMPLE_LENGTH:
        return False
    
    if len(query) > MAX_SAMPLE_LENGTH:
        return False
    
    if not isinstance(confidence, (int, float)):
        return False
    
    if confidence < 0 or confidence > 1:
        return False
    
    return True