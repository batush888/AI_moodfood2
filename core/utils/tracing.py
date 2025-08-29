#!/usr/bin/env python3
"""
Request Tracing System for End-to-End Visibility

This module provides comprehensive tracing capabilities to track requests
through the entire pipeline: frontend → LLM → ML → logging → retraining.
"""

import uuid
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)

@dataclass
class TraceEvent:
    """Individual event within a trace"""
    event_id: str
    trace_id: str
    stage: str
    timestamp: str
    duration_ms: float
    metadata: Dict[str, Any]
    status: str  # "success", "error", "warning"
    error_message: Optional[str] = None

@dataclass
class TraceContext:
    """Complete trace context for a request"""
    trace_id: str
    request_id: str
    start_time: str
    events: List[TraceEvent]
    total_duration_ms: float
    status: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    model_version: Optional[str] = None
    final_decision: Optional[str] = None

class TraceManager:
    """Manages request tracing and trace storage"""
    
    def __init__(self, trace_file: str = "logs/traces.jsonl"):
        self.trace_file = trace_file
        self._lock = threading.RLock()
        self._active_traces: Dict[str, TraceContext] = {}
        
        # Ensure trace log directory exists
        from pathlib import Path
        Path(trace_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TraceManager initialized: {trace_file}")
    
    def start_trace(self, 
                   request_id: str,
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   model_version: Optional[str] = None) -> str:
        """
        Start a new trace for a request.
        
        Args:
            request_id: Unique identifier for the request
            user_id: Optional user identifier
            session_id: Optional session identifier
            model_version: Current model version being used
            
        Returns:
            str: Generated trace_id
        """
        trace_id = f"trace_{uuid.uuid4().hex[:16]}"
        start_time = datetime.utcnow().isoformat()
        
        trace_context = TraceContext(
            trace_id=trace_id,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            start_time=start_time,
            events=[],
            total_duration_ms=0.0,
            status="active",
            model_version=model_version
        )
        
        with self._lock:
            self._active_traces[trace_id] = trace_context
        
        logger.debug(f"Started trace {trace_id} for request {request_id}")
        return trace_id
    
    def add_event(self, 
                  trace_id: str,
                  stage: str,
                  metadata: Dict[str, Any],
                  status: str = "success",
                  error_message: Optional[str] = None,
                  duration_ms: Optional[float] = None) -> bool:
        """
        Add an event to an active trace.
        
        Args:
            trace_id: Trace identifier
            stage: Pipeline stage (e.g., "llm", "ml", "logging")
            metadata: Additional event data
            status: Event status
            error_message: Error details if status is "error"
            duration_ms: Duration of this stage in milliseconds
            
        Returns:
            bool: True if event added successfully
        """
        with self._lock:
            if trace_id not in self._active_traces:
                logger.warning(f"Trace {trace_id} not found")
                return False
            
            trace_context = self._active_traces[trace_id]
            
            event = TraceEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                trace_id=trace_id,
                stage=stage,
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=duration_ms or 0.0,
                metadata=metadata,
                status=status,
                error_message=error_message
            )
            
            trace_context.events.append(event)
            
            # Update trace status if error occurred
            if status == "error":
                trace_context.status = "error"
            
            logger.debug(f"Added event to trace {trace_id}: {stage} - {status}")
            return True
    
    def end_trace(self, 
                  trace_id: str,
                  final_decision: Optional[str] = None,
                  status: Optional[str] = None) -> bool:
        """
        End a trace and save it to storage.
        
        Args:
            trace_id: Trace identifier
            final_decision: Final decision made by the pipeline
            status: Final trace status (overrides auto-detected status)
            
        Returns:
            bool: True if trace ended successfully
        """
        with self._lock:
            if trace_id not in self._active_traces:
                logger.warning(f"Trace {trace_id} not found")
                return False
            
            trace_context = self._active_traces[trace_id]
            
            # Calculate total duration
            start_time = datetime.fromisoformat(trace_context.start_time)
            end_time = datetime.utcnow()
            total_duration = (end_time - start_time).total_seconds() * 1000
            
            trace_context.total_duration_ms = total_duration
            trace_context.final_decision = final_decision
            
            if status:
                trace_context.status = status
            
            # Save trace to storage
            self._save_trace(trace_context)
            
            # Remove from active traces
            del self._active_traces[trace_id]
            
            logger.info(f"Ended trace {trace_id}: {trace_context.status}, {total_duration:.2f}ms")
            return True
    
    def _save_trace(self, trace_context: TraceContext) -> None:
        """Save trace to persistent storage"""
        try:
            trace_data = asdict(trace_context)
            
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to save trace {trace_context.trace_id}: {e}")
    
    def get_trace(self, trace_id: str) -> Optional[TraceContext]:
        """Retrieve a trace by ID"""
        with self._lock:
            return self._active_traces.get(trace_id)
    
    def get_active_traces(self) -> List[str]:
        """Get list of active trace IDs"""
        with self._lock:
            return list(self._active_traces.keys())
    
    def cleanup_expired_traces(self, max_age_hours: int = 24) -> int:
        """Clean up traces older than specified age"""
        current_time = datetime.utcnow()
        expired_count = 0
        
        with self._lock:
            expired_traces = []
            
            for trace_id, trace_context in self._active_traces.items():
                start_time = datetime.fromisoformat(trace_context.start_time)
                age_hours = (current_time - start_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    expired_traces.append(trace_id)
                    expired_count += 1
            
            # Remove expired traces
            for trace_id in expired_traces:
                del self._active_traces[trace_id]
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired traces")
        
        return expired_count

# Global trace manager instance
_trace_manager = TraceManager()

def get_trace_manager() -> TraceManager:
    """Get the global trace manager instance"""
    return _trace_manager

def generate_trace_id() -> str:
    """Generate a new trace ID"""
    return f"trace_{uuid.uuid4().hex[:16]}"

def start_trace(request_id: str, **kwargs) -> str:
    """Start a new trace"""
    return _trace_manager.start_trace(request_id, **kwargs)

def add_trace_event(trace_id: str, stage: str, **kwargs) -> bool:
    """Add an event to a trace"""
    return _trace_manager.add_event(trace_id, stage, **kwargs)

def end_trace(trace_id: str, **kwargs) -> bool:
    """End a trace"""
    return _trace_manager.end_trace(trace_id, **kwargs)

@contextmanager
def trace_stage(trace_id: str, stage: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for tracing a pipeline stage with automatic timing.
    
    Usage:
        with trace_stage(trace_id, "llm_call", {"model": "deepseek"}):
            # LLM call code here
            pass
    """
    start_time = time.time()
    stage_metadata = metadata or {}
    
    try:
        # Add start event
        add_trace_event(trace_id, f"{stage}_start", metadata=stage_metadata, status="start")
        
        yield
        
        # Add success event
        duration_ms = (time.time() - start_time) * 1000
        add_trace_event(trace_id, f"{stage}_end", 
                       metadata={**stage_metadata, "duration_ms": duration_ms}, 
                       status="success", duration_ms=duration_ms)
        
    except Exception as e:
        # Add error event
        duration_ms = (time.time() - start_time) * 1000
        add_trace_event(trace_id, f"{stage}_error", 
                       metadata={**stage_metadata, "duration_ms": duration_ms}, 
                       status="error", error_message=str(e), duration_ms=duration_ms)
        raise

def get_trace_summary(trace_id: str) -> Optional[Dict[str, Any]]:
    """Get a summary of a trace for API responses"""
    trace_context = _trace_manager.get_trace(trace_id)
    
    if not trace_context:
        return None
    
    return {
        "trace_id": trace_context.trace_id,
        "status": trace_context.status,
        "total_duration_ms": trace_context.total_duration_ms,
        "stages": [event.stage for event in trace_context.events],
        "model_version": trace_context.model_version,
        "final_decision": trace_context.final_decision
    }
