#!/usr/bin/env python3
"""
Global Hybrid Filter Instance for System-wide Access with Redis Backend
"""

import logging
import threading
from datetime import datetime
from typing import Dict, Optional

# Redis availability flag (will be set when needed)
REDIS_AVAILABLE = False
logger = logging.getLogger(__name__)

from core.filtering.hybrid_filter import HybridFilter

# Global instance that can be accessed from anywhere in the system
global_hybrid_filter = HybridFilter()

# Redis connection (will be initialized on first use)
_redis_client = None
_redis_lock = threading.Lock()

# Fallback local dict for when Redis is unavailable
_fallback_stats = {
    "total": 0,
    "ml_confident": 0,
    "llm_fallback": 0,
    "rejected": 0
}
_fallback_lock = threading.Lock()

# Redis key prefix for filter stats
REDIS_KEY_PREFIX = "filter_stats:"

def _get_redis_client():
    """
    Get Redis client with lazy initialization and error handling
    
    Returns:
        Redis client or None if Redis is unavailable
    """
    global _redis_client, REDIS_AVAILABLE
    
    # Try to import Redis if not already attempted
    if not REDIS_AVAILABLE:
        try:
            import redis
            REDIS_AVAILABLE = True
            logger.info("Redis available for global filter stats")
        except ImportError:
            logger.warning("Redis not available, falling back to local dict for global filter stats")
            return None
    
    if not REDIS_AVAILABLE:
        return None
    
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                try:
                    # Initialize Redis client with default localhost:6379
                    _redis_client = redis.Redis(
                        host='localhost',
                        port=6379,
                        db=0,
                        decode_responses=True,  # Return strings instead of bytes
                        socket_connect_timeout=5,
                        socket_timeout=5
                    )
                    # Test connection
                    _redis_client.ping()
                    logger.info("Redis connection established for global filter stats")
                except Exception as e:
                    logger.warning(f"Failed to connect to Redis: {e}")
                    _redis_client = None
    
    return _redis_client

def get_global_hybrid_filter() -> HybridFilter:
    """Get the global hybrid filter instance"""
    return global_hybrid_filter

def update_global_filter_stats(decision: str) -> None:
    """
    Update global filter statistics during inference
    
    Args:
        decision: One of 'ml_confident', 'llm_fallback', or 'rejected'
    """
    redis_client = _get_redis_client()
    
    if redis_client is not None:
        # Use Redis for production scalability
        try:
            # Increment total samples counter
            redis_client.incr(f"{REDIS_KEY_PREFIX}total_samples")
            
            # Increment the specific decision counter
            if decision in ["ml_confident", "llm_fallback", "rejected"]:
                redis_client.incr(f"{REDIS_KEY_PREFIX}{decision}")
            else:
                logger.warning(f"Unknown decision type: {decision}")
                
        except Exception as e:
            logger.error(f"Failed to update Redis filter stats: {e}")
            # Fallback to local dict
            _update_fallback_stats(decision)
    else:
        # Fallback to local dict when Redis is unavailable
        _update_fallback_stats(decision)

def _update_fallback_stats(decision: str) -> None:
    """Update fallback local stats when Redis is unavailable"""
    with _fallback_lock:
        _fallback_stats["total"] += 1
        if decision == "ml_confident":
            _fallback_stats["ml_confident"] += 1
        elif decision == "llm_fallback":
            _fallback_stats["llm_fallback"] += 1
        elif decision == "rejected":
            _fallback_stats["rejected"] += 1
        else:
            logger.warning(f"Unknown decision type: {decision}")

def get_global_filter_live_stats() -> Dict:
    """
    Get live stats from global hybrid filter
    
    Returns:
        Dict containing live stats with timestamp and source information
    """
    redis_client = _get_redis_client()
    
    if redis_client is not None:
        # Try to get stats from Redis
        try:
            # Get all counters from Redis
            total_samples = int(redis_client.get(f"{REDIS_KEY_PREFIX}total_samples") or 0)
            ml_confident = int(redis_client.get(f"{REDIS_KEY_PREFIX}ml_confident") or 0)
            llm_fallback = int(redis_client.get(f"{REDIS_KEY_PREFIX}llm_fallback") or 0)
            rejected = int(redis_client.get(f"{REDIS_KEY_PREFIX}rejected") or 0)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_samples": total_samples,
                "ml_confident": ml_confident,
                "llm_fallback": llm_fallback,
                "rejected": rejected,
                "source": "redis_global_filter"
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis filter stats: {e}")
            # Fallback to local dict
            return _get_fallback_stats()
    else:
        # Use fallback local dict when Redis is unavailable
        return _get_fallback_stats()

def _get_fallback_stats() -> Dict:
    """Get fallback local stats when Redis is unavailable"""
    with _fallback_lock:
        return {
            "timestamp": datetime.now().isoformat(),
            "total_samples": _fallback_stats["total"],
            "ml_confident": _fallback_stats["ml_confident"],
            "llm_fallback": _fallback_stats["llm_fallback"],
            "rejected": _fallback_stats["rejected"],
            "source": "local_fallback"
        }

def reset_global_filter_stats() -> None:
    """Reset global filter statistics (set all counters to 0)"""
    redis_client = _get_redis_client()
    
    if redis_client is not None:
        # Reset Redis counters
        try:
            # Delete all filter stats keys
            keys_to_delete = [
                f"{REDIS_KEY_PREFIX}total_samples",
                f"{REDIS_KEY_PREFIX}ml_confident",
                f"{REDIS_KEY_PREFIX}llm_fallback",
                f"{REDIS_KEY_PREFIX}rejected"
            ]
            
            for key in keys_to_delete:
                redis_client.delete(key)
            
            logger.info("Redis filter stats reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset Redis filter stats: {e}")
            # Fallback to local dict reset
            _reset_fallback_stats()
    else:
        # Reset fallback local dict when Redis is unavailable
        _reset_fallback_stats()

def _reset_fallback_stats() -> None:
    """Reset fallback local stats when Redis is unavailable"""
    with _fallback_lock:
        _fallback_stats.update({
            "total": 0,
            "ml_confident": 0,
            "llm_fallback": 0,
            "rejected": 0
        })
        logger.info("Local fallback filter stats reset successfully")
