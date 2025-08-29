#!/usr/bin/env python3
"""
Advanced Metrics System for Production Observability

This module provides comprehensive Prometheus metrics for monitoring
system health, performance, and reliability.
"""

import logging
import time
from typing import Dict, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fallback to dummy implementation
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, 
        start_http_server, generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available - using dummy metrics")

# Dummy metric classes for when Prometheus is unavailable
class DummyMetric:
    """Dummy metric that does nothing when Prometheus is unavailable"""
    def __init__(self, name: str, documentation: str = ""):
        self.name = name
        self.documentation = documentation
    
    def inc(self, amount: float = 1.0, **kwargs):
        pass
    
    def set(self, value: float, **kwargs):
        pass
    
    def observe(self, value: float, **kwargs):
        pass
    
    def time(self, **kwargs):
        return DummyContextManager()
    
    def count_exceptions(self, **kwargs):
        return DummyContextManager()

class DummyContextManager:
    """Dummy context manager for timing metrics"""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Define metrics with fallback to dummy implementations
if PROMETHEUS_AVAILABLE:
    # Retraining metrics
    RETRAIN_COUNTER = Counter(
        "moodfood_retrains_total", 
        "Total retraining attempts",
        ["status", "trigger", "model_version"]
    )
    RETRAIN_DURATION = Histogram(
        "moodfood_retrain_duration_seconds",
        "Retraining duration in seconds",
        ["status", "model_version"]
    )
    
    # Deployment metrics
    DEPLOY_SUCCESS = Counter(
        "moodfood_model_deploys_total", 
        "Total successful model deployments",
        ["mode", "model_version"]
    )
    DEPLOY_DURATION = Histogram(
        "moodfood_deploy_duration_seconds",
        "Model deployment duration in seconds",
        ["mode", "model_version"]
    )
    
    # Model versioning metrics
    CURRENT_MODEL_VERSION = Gauge(
        "moodfood_current_model_version", 
        "Current model version hash"
    )
    MODEL_ACCURACY = Gauge(
        "moodfood_model_accuracy", 
        "Current model accuracy",
        ["model_version"]
    )
    
    # A/B testing metrics
    ABTEST_REQUESTS = Counter(
        "moodfood_abtest_requests_total", 
        "Total A/B test requests",
        ["version", "decision"]
    )
    ABTEST_ACTIVE = Gauge(
        "moodfood_abtest_active", 
        "Whether A/B test is currently active"
    )
    
    # System health metrics
    SYSTEM_HEALTH = Gauge(
        "moodfood_system_health", 
        "Overall system health score (0-100)"
    )
    
    # Query processing metrics
    QUERY_PROCESSING_TIME = Histogram(
        "moodfood_query_processing_time_ms",
        "Query processing time in milliseconds",
        ["stage", "model_version", "decision"]
    )
    QUERY_COUNTER = Counter(
        "moodfood_queries_total",
        "Total queries processed",
        ["status", "model_version", "decision"]
    )
    
    # LLM API metrics
    LLM_API_CALLS = Counter(
        "moodfood_llm_api_calls_total",
        "Total LLM API calls",
        ["model", "status", "reason"]
    )
    LLM_API_LATENCY = Histogram(
        "moodfood_llm_api_latency_ms",
        "LLM API call latency in milliseconds",
        ["model", "status"]
    )
    
    # Training buffer metrics
    TRAINING_BUFFER_SIZE = Gauge(
        "moodfood_training_buffer_size",
        "Current training buffer size"
    )
    TRAINING_BUFFER_GOLD_RATIO = Gauge(
        "moodfood_training_buffer_gold_ratio",
        "Ratio of gold quality samples in training buffer"
    )
    
    # Drift detection metrics
    DRIFT_ALERTS = Counter(
        "moodfood_drift_alerts_total",
        "Total drift alerts triggered",
        ["type", "severity", "model_version"]
    )
    DRIFT_SCORE = Gauge(
        "moodfood_drift_score",
        "Current drift score (0-1, higher = more drift)",
        ["type", "model_version"]
    )
    
    # Feedback metrics
    USER_FEEDBACK = Counter(
        "moodfood_user_feedback_total",
        "Total user feedback received",
        ["type", "sentiment", "model_version"]
    )
    FEEDBACK_SCORE = Gauge(
        "moodfood_feedback_score",
        "Current user feedback score (-1 to 1)",
        ["model_version"]
    )
    
    # Tracing metrics
    TRACE_EVENTS = Counter(
        "moodfood_trace_events_total",
        "Total trace events",
        ["stage", "status", "model_version"]
    )
    TRACE_DURATION = Histogram(
        "moodfood_trace_duration_ms",
        "Trace duration in milliseconds",
        ["status", "model_version"]
    )
    
else:
    # Create dummy metrics when Prometheus is unavailable
    RETRAIN_COUNTER = DummyMetric("moodfood_retrains_total")
    RETRAIN_DURATION = DummyMetric("moodfood_retrain_duration_seconds")
    DEPLOY_SUCCESS = DummyMetric("moodfood_model_deploys_total")
    DEPLOY_DURATION = DummyMetric("moodfood_deploy_duration_seconds")
    CURRENT_MODEL_VERSION = DummyMetric("moodfood_current_model_version")
    MODEL_ACCURACY = DummyMetric("moodfood_model_accuracy")
    ABTEST_REQUESTS = DummyMetric("moodfood_abtest_requests_total")
    ABTEST_ACTIVE = DummyMetric("moodfood_abtest_active")
    SYSTEM_HEALTH = DummyMetric("moodfood_system_health")
    QUERY_PROCESSING_TIME = DummyMetric("moodfood_query_processing_time_ms")
    QUERY_COUNTER = DummyMetric("moodfood_queries_total")
    LLM_API_CALLS = DummyMetric("moodfood_llm_api_calls_total")
    LLM_API_LATENCY = DummyMetric("moodfood_llm_api_latency_ms")
    TRAINING_BUFFER_SIZE = DummyMetric("moodfood_training_buffer_size")
    TRAINING_BUFFER_GOLD_RATIO = DummyMetric("moodfood_training_buffer_gold_ratio")
    DRIFT_ALERTS = DummyMetric("moodfood_drift_alerts_total")
    DRIFT_SCORE = DummyMetric("moodfood_drift_score")
    USER_FEEDBACK = DummyMetric("moodfood_user_feedback_total")
    FEEDBACK_SCORE = DummyMetric("moodfood_feedback_score")
    TRACE_EVENTS = DummyMetric("moodfood_trace_events_total")
    TRACE_DURATION = DummyMetric("moodfood_trace_duration_ms")

class MetricsManager:
    """Manages Prometheus metrics and provides convenience methods"""
    
    def __init__(self, port: int = 9189):
        self.port = port
        self._server_started = False
        
        logger.info(f"MetricsManager initialized on port {port}")
    
    def start_metrics_server(self) -> bool:
        """Start the Prometheus metrics HTTP server"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available - cannot start metrics server")
            return False
        
        try:
            start_http_server(self.port)
            self._server_started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False
    
    def is_server_running(self) -> bool:
        """Check if metrics server is running"""
        return self._server_started
    
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus metrics not available\n"
        
        try:
            return generate_latest().decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return f"# Error generating metrics: {e}\n"
    
    # Retraining metrics
    def record_retrain(self, status: str, trigger: str, model_version: str, duration_seconds: float):
        """Record retraining metrics"""
        RETRAIN_COUNTER.labels(status=status, trigger=trigger, model_version=model_version).inc()
        RETRAIN_DURATION.labels(status=status, model_version=model_version).observe(duration_seconds)
    
    def record_deploy(self, mode: str, model_version: str, duration_seconds: float):
        """Record deployment metrics"""
        DEPLOY_SUCCESS.labels(mode=mode, model_version=model_version).inc()
        DEPLOY_DURATION.labels(mode=mode, model_version=model_version).observe(duration_seconds)
    
    def update_model_version(self, version_hash: str):
        """Update current model version"""
        CURRENT_MODEL_VERSION.set(hash(version_hash) % 1000000)  # Use hash for numeric value
    
    def update_model_accuracy(self, accuracy: float, model_version: str):
        """Update model accuracy metric"""
        MODEL_ACCURACY.labels(model_version=model_version).set(accuracy)
    
    # A/B testing metrics
    def record_abtest_request(self, version: str, decision: str):
        """Record A/B test request"""
        ABTEST_REQUESTS.labels(version=version, decision=decision).inc()
    
    def set_abtest_status(self, active: bool):
        """Set A/B test active status"""
        ABTEST_ACTIVE.set(1.0 if active else 0.0)
    
    # System health metrics
    def update_system_health(self, health_score: float):
        """Update system health score (0-100)"""
        SYSTEM_HEALTH.set(max(0.0, min(100.0, health_score)))
    
    # Query processing metrics
    def record_query_processing(self, stage: str, duration_ms: float, model_version: str, decision: str):
        """Record query processing metrics"""
        QUERY_PROCESSING_TIME.labels(
            stage=stage, 
            model_version=model_version, 
            decision=decision
        ).observe(duration_ms)
        
        QUERY_COUNTER.labels(
            status="success", 
            model_version=model_version, 
            decision=decision
        ).inc()
    
    def record_query_error(self, model_version: str, decision: str):
        """Record query error"""
        QUERY_COUNTER.labels(
            status="error", 
            model_version=model_version, 
            decision=decision
        ).inc()
    
    # LLM API metrics
    def record_llm_api_call(self, model: str, status: str, reason: str = ""):
        """Record LLM API call"""
        LLM_API_CALLS.labels(model=model, status=status, reason=reason).inc()
    
    def record_llm_latency(self, model: str, status: str, latency_ms: float):
        """Record LLM API latency"""
        LLM_API_LATENCY.labels(model=model, status=status).observe(latency_ms)
    
    # Training buffer metrics
    def update_training_buffer(self, size: int, gold_ratio: float):
        """Update training buffer metrics"""
        TRAINING_BUFFER_SIZE.set(size)
        TRAINING_BUFFER_GOLD_RATIO.set(gold_ratio)
    
    # Drift detection metrics
    def record_drift_alert(self, alert_type: str, severity: str, model_version: str):
        """Record drift alert"""
        DRIFT_ALERTS.labels(
            type=alert_type, 
            severity=severity, 
            model_version=model_version
        ).inc()
    
    def update_drift_score(self, drift_type: str, score: float, model_version: str):
        """Update drift score"""
        DRIFT_SCORE.labels(type=drift_type, model_version=model_version).set(score)
    
    # Feedback metrics
    def record_user_feedback(self, feedback_type: str, sentiment: str, model_version: str):
        """Record user feedback"""
        USER_FEEDBACK.labels(
            type=feedback_type, 
            sentiment=sentiment, 
            model_version=model_version
        ).inc()
    
    def update_feedback_score(self, score: float, model_version: str):
        """Update feedback score (-1 to 1)"""
        FEEDBACK_SCORE.labels(model_version=model_version).set(max(-1.0, min(1.0, score)))
    
    # Tracing metrics
    def record_trace_event(self, stage: str, status: str, model_version: str):
        """Record trace event"""
        TRACE_EVENTS.labels(
            stage=stage, 
            status=status, 
            model_version=model_version
        ).inc()
    
    def record_trace_duration(self, status: str, duration_ms: float, model_version: str):
        """Record trace duration"""
        TRACE_DURATION.labels(status=status, model_version=model_version).observe(duration_ms)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics for monitoring"""
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "server_running": self._server_started,
            "port": self.port,
            "timestamp": datetime.utcnow().isoformat()
        }

# Global metrics manager instance
_metrics_manager = MetricsManager()

def get_metrics_manager() -> MetricsManager:
    """Get the global metrics manager instance"""
    return _metrics_manager

def start_metrics_server(port: int = 9189) -> bool:
    """Start the Prometheus metrics server"""
    global _metrics_manager
    _metrics_manager = MetricsManager(port)
    return _metrics_manager.start_metrics_server()

# Convenience functions for common metric operations
def record_retrain_success(trigger: str, model_version: str, duration_seconds: float):
    """Record successful retraining"""
    _metrics_manager.record_retrain("success", trigger, model_version, duration_seconds)

def record_retrain_failure(trigger: str, model_version: str, duration_seconds: float):
    """Record failed retraining"""
    _metrics_manager.record_retrain("failure", trigger, model_version, duration_seconds)

def record_deploy_success(mode: str, model_version: str, duration_seconds: float):
    """Record successful deployment"""
    _metrics_manager.record_deploy(mode, model_version, duration_seconds)

def record_query_success(stage: str, duration_ms: float, model_version: str, decision: str):
    """Record successful query processing"""
    _metrics_manager.record_query_processing(stage, duration_ms, model_version, decision)

def record_llm_success(model: str, latency_ms: float):
    """Record successful LLM API call"""
    _metrics_manager.record_llm_api_call(model, "success")
    _metrics_manager.record_llm_latency(model, "success", latency_ms)

def record_llm_failure(model: str, reason: str):
    """Record failed LLM API call"""
    _metrics_manager.record_llm_api_call(model, "failure", reason)

def record_drift_detected(alert_type: str, severity: str, model_version: str):
    """Record drift detection"""
    _metrics_manager.record_drift_alert(alert_type, severity, model_version)

def record_user_feedback_positive(feedback_type: str, model_version: str):
    """Record positive user feedback"""
    _metrics_manager.record_user_feedback(feedback_type, "positive", model_version)

def record_user_feedback_negative(feedback_type: str, model_version: str):
    """Record negative user feedback"""
    _metrics_manager.record_user_feedback(feedback_type, "negative", model_version)
