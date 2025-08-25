#!/usr/bin/env python3
"""
Global Hybrid Filter Instance for System-wide Access
"""

from core.filtering.hybrid_filter import HybridFilter

# Global instance that can be accessed from anywhere in the system
global_hybrid_filter = HybridFilter()

def get_global_hybrid_filter() -> HybridFilter:
    """Get the global hybrid filter instance"""
    return global_hybrid_filter

def update_global_filter_stats(decision: str) -> None:
    """Update global filter statistics during inference"""
    global_hybrid_filter.update_live_stats(decision)

def get_global_filter_live_stats():
    """Get live stats from global hybrid filter"""
    return global_hybrid_filter.get_live_stats()

def reset_global_filter_stats() -> None:
    """Reset global filter statistics"""
    global_hybrid_filter.reset_live_stats()
