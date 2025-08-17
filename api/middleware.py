# AI_moodfood2/api/middleware.py
# Comprehensive monitoring and logging middleware

import time
import json
import logging
import uuid
from typing import Callable, Dict, Any
from datetime import datetime
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------
# Metrics Collection
# ------------------
class MetricsCollector:
    """Collects and stores API metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.endpoint_stats = {}
        self.error_logs = []
        
    def record_request(self, endpoint: str, method: str, response_time: float, status_code: int):
        """Record a request metric."""
        self.request_count += 1
        
        if status_code >= 400:
            self.error_count += 1
            
        self.response_times.append(response_time)
        
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                'count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'errors': 0,
                'methods': set()
            }
        
        stats = self.endpoint_stats[endpoint]
        stats['count'] += 1
        stats['total_time'] += response_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['methods'].add(method)
        
        if status_code >= 400:
            stats['errors'] += 1
    
    def record_error(self, error: Exception, context: Dict[str, Any]):
        """Record an error with context."""
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }
        self.error_logs.append(error_log)
        
        # Keep only last 100 errors
        if len(self.error_logs) > 100:
            self.error_logs = self.error_logs[-100:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.response_times:
            return {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'avg_response_time': 0.0,
                'endpoint_stats': self.endpoint_stats
            }
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'avg_response_time': sum(self.response_times) / len(self.response_times),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'endpoint_stats': self.endpoint_stats,
            'error_rate': self.error_count / self.request_count if self.request_count > 0 else 0.0
        }

# Global metrics instance
metrics = MetricsCollector()

# ------------------
# Request Logging Middleware
# ------------------
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request logging and monitoring."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.app = app
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Record start time
        start_time = time.time()
        
        # Extract request details
        method = request.method
        url = str(request.url)
        endpoint = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request start
        logger.info(f"🚀 [{request_id}] {method} {endpoint} | IP: {client_ip} | UA: {user_agent[:50]}...")
        
        # Add request ID to headers for tracing
        request.headers.__dict__["_list"].append((b"x-request-id", request_id.encode()))
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Record metrics
            metrics.record_request(endpoint, method, response_time, response.status_code)
            
            # Log successful response
            logger.info(f"✅ [{request_id}] {method} {endpoint} | Status: {response.status_code} | Time: {response_time:.3f}s")
            
            # Add response headers for monitoring
            response.headers["x-request-id"] = request_id
            response.headers["x-response-time"] = f"{response_time:.3f}"
            
            return response
            
        except Exception as e:
            # Calculate response time
            response_time = time.time() - start_time
            
            # Record error
            error_context = {
                'request_id': request_id,
                'method': method,
                'endpoint': endpoint,
                'client_ip': client_ip,
                'user_agent': user_agent,
                'response_time': response_time
            }
            metrics.record_error(e, error_context)
            
            # Log error
            logger.error(f"💥 [{request_id}] {method} {endpoint} | Error: {type(e).__name__}: {str(e)} | Time: {response_time:.3f}s")
            
            # Re-raise the exception
            raise

# ------------------
# Performance Monitoring Middleware
# ------------------
class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed performance monitoring."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.app = app
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Record detailed timing
        timing_data = {
            'start': time.time(),
            'phases': {}
        }
        
        # Phase 1: Request parsing
        phase_start = time.time()
        response = await call_next(request)
        timing_data['phases']['request_processing'] = time.time() - phase_start
        
        # Calculate total time
        timing_data['total'] = time.time() - timing_data['start']
        
        # Log performance data for slow requests
        if timing_data['total'] > 1.0:  # Log requests taking more than 1 second
            logger.warning(f"🐌 Slow request detected: {request.method} {request.url.path} | "
                          f"Total: {timing_data['total']:.3f}s | "
                          f"Phases: {json.dumps(timing_data['phases'], indent=2)}")
        
        # Add performance headers
        response.headers["x-performance-total"] = f"{timing_data['total']:.3f}"
        response.headers["x-performance-phases"] = json.dumps(timing_data['phases'])
        
        return response

# ------------------
# Error Handling Middleware
# ------------------
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for graceful error handling and logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.app = app
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Log detailed error information
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'method': request.method,
                'url': str(request.url),
                'client_ip': request.client.host if request.client else "unknown",
                'error_type': type(e).__name__,
                'error_message': str(e),
                'headers': dict(request.headers),
                'query_params': dict(request.query_params)
            }
            
            logger.error(f"🚨 Unhandled error: {json.dumps(error_info, indent=2)}")
            
            # Record in metrics
            metrics.record_error(e, error_info)
            
            # Re-raise for proper error handling
            raise

# ------------------
# CORS and Security Middleware
# ------------------
class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and CORS."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.app = app
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["x-frame-options"] = "DENY"
        response.headers["x-xss-protection"] = "1; mode=block"
        response.headers["referrer-policy"] = "strict-origin-when-cross-origin"
        
        return response

# ------------------
# Utility Functions
# ------------------
def get_metrics_summary() -> Dict[str, Any]:
    """Get current metrics summary."""
    return metrics.get_summary()

def reset_metrics():
    """Reset all metrics (useful for testing)."""
    global metrics
    metrics = MetricsCollector()

def log_system_health():
    """Log system health information."""
    summary = metrics.get_summary()
    
    logger.info("🏥 System Health Check:")
    logger.info(f"  📊 Total Requests: {summary['request_count']}")
    logger.info(f"  ❌ Total Errors: {summary['error_count']}")
    logger.info(f"  📈 Error Rate: {summary['error_rate']:.2%}")
    logger.info(f"  ⏱️  Avg Response Time: {summary['avg_response_time']:.3f}s")
    
    if summary['endpoint_stats']:
        logger.info("  🎯 Endpoint Statistics:")
        for endpoint, stats in summary['endpoint_stats'].items():
            logger.info(f"    {endpoint}: {stats['count']} requests, "
                       f"{stats['errors']} errors, "
                       f"avg {stats['avg_time']:.3f}s")

# ------------------
# Middleware Stack Configuration
# ------------------
def create_middleware_stack(app: ASGIApp) -> ASGIApp:
    """Create a middleware stack with all monitoring components."""
    
    # Apply middleware in order (last applied = first executed)
    app = SecurityMiddleware(app)
    app = ErrorHandlingMiddleware(app)
    app = PerformanceMonitoringMiddleware(app)
    app = RequestLoggingMiddleware(app)
    
    logger.info("🔧 Monitoring middleware stack configured")
    return app

# ------------------
# Health Check Endpoint Helper
# ------------------
def create_health_response() -> Dict[str, Any]:
    """Create a comprehensive health check response."""
    summary = metrics.get_summary()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running",  # Could be enhanced with actual uptime tracking
        "metrics": summary,
        "version": "1.0.0"
    }
