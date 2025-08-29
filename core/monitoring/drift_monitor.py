#!/usr/bin/env python3
"""
Drift Detection System for ML Model Monitoring
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    timestamp: str
    alert_type: str
    severity: str
    model_version: str
    drift_score: float
    threshold: float
    details: Dict[str, Any]
    status: str = "active"

@dataclass
class DriftMetrics:
    """Drift detection metrics"""
    timestamp: str
    model_version: str
    overall_drift_score: float
    alerts_triggered: int
    status: str

class DriftMonitor:
    """Monitors data drift between training and inference distributions"""
    
    def __init__(self, 
                 query_logs_file: str = "logs/query_logs.jsonl",
                 drift_alerts_file: str = "logs/drift_alerts.jsonl",
                 check_interval_minutes: int = 60,
                 drift_threshold: float = 0.1):
        
        self.query_logs_file = Path(query_logs_file)
        self.drift_alerts_file = Path(drift_alerts_file)
        self.check_interval_minutes = check_interval_minutes
        self.drift_threshold = drift_threshold
        
        # Ensure log directories exist
        self.drift_alerts_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Background monitoring thread
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        # Drift history and alerts
        self._drift_history: List[DriftMetrics] = []
        self._active_alerts: Dict[str, DriftAlert] = {}
        
        logger.info(f"DriftMonitor initialized: threshold={drift_threshold}")
    
    def start_monitoring(self) -> bool:
        """Start background drift monitoring"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return False
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Drift monitoring started")
        return True
    
    def stop_monitoring(self) -> bool:
        """Stop background drift monitoring"""
        if not self._monitor_thread or not self._monitor_thread.is_alive():
            return False
        
        self._stop_monitoring.set()
        self._monitor_thread.join(timeout=5.0)
        logger.info("Drift monitoring stopped")
        return True
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                self.check_drift()
                time.sleep(self.check_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in drift monitoring loop: {e}")
                time.sleep(60)
    
    def check_drift(self) -> Optional[DriftMetrics]:
        """Perform drift detection check"""
        try:
            current_model_version = self._get_current_model_version()
            recent_queries = self._load_recent_queries()
            
            if not recent_queries:
                return None
            
            # Calculate simple drift score based on confidence distribution
            drift_score = self._calculate_simple_drift(recent_queries)
            
            drift_metrics = DriftMetrics(
                timestamp=datetime.utcnow().isoformat(),
                model_version=current_model_version,
                overall_drift_score=drift_score,
                alerts_triggered=0,
                status="normal"
            )
            
            # Check for drift alerts
            alerts_triggered = self._check_drift_alerts(drift_metrics)
            drift_metrics.alerts_triggered = len(alerts_triggered)
            
            if alerts_triggered:
                drift_metrics.status = "alert"
                logger.warning(f"Drift detected: score={drift_score:.3f}")
            
            # Store metrics
            self._store_drift_metrics(drift_metrics)
            self._drift_history.append(drift_metrics)
            
            if len(self._drift_history) > 100:
                self._drift_history = self._drift_history[-100:]
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Failed to check drift: {e}")
            return None
    
    def _load_recent_queries(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Load recent queries from logs"""
        try:
            if not self.query_logs_file.exists():
                return []
            
            recent_queries = []
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            with open(self.query_logs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        query_time = datetime.fromisoformat(entry.get('timestamp', ''))
                        if query_time >= cutoff_time:
                            recent_queries.append(entry)
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            return recent_queries
            
        except Exception as e:
            logger.error(f"Failed to load recent queries: {e}")
            return []
    
    def _get_current_model_version(self) -> str:
        """Get current model version"""
        try:
            from core.nlu.model_loader import get_current_version
            return get_current_version() or "unknown"
        except ImportError:
            return "unknown"
    
    def _calculate_simple_drift(self, queries: List[Dict[str, Any]]) -> float:
        """Calculate simple drift score based on confidence distribution"""
        try:
            confidences = []
            for query in queries:
                confidence = query.get('confidence')
                if confidence is not None:
                    confidences.append(float(confidence))
            
            if not confidences:
                return 0.0
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)
            
            # Expected confidence range (0.5-0.9 is normal)
            expected_confidence = 0.7
            
            # Calculate drift as normalized difference
            drift_score = abs(avg_confidence - expected_confidence) / expected_confidence
            
            return min(1.0, drift_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate drift: {e}")
            return 0.0
    
    def _check_drift_alerts(self, drift_metrics: DriftMetrics) -> List[DriftAlert]:
        """Check if drift metrics trigger alerts"""
        alerts = []
        
        try:
            if drift_metrics.overall_drift_score >= self.drift_threshold:
                alert = self._create_drift_alert(
                    "confidence_drift",
                    "high",
                    drift_metrics.model_version,
                    drift_metrics.overall_drift_score,
                    self.drift_threshold,
                    {"drift_metrics": asdict(drift_metrics)}
                )
                alerts.append(alert)
            
            # Store alerts
            for alert in alerts:
                self._store_drift_alert(alert)
                self._active_alerts[alert.alert_id] = alert
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to check drift alerts: {e}")
            return []
    
    def _create_drift_alert(self, 
                           alert_type: str,
                           severity: str,
                           model_version: str,
                           drift_score: float,
                           threshold: float,
                           details: Dict[str, Any]) -> DriftAlert:
        """Create a new drift alert"""
        alert_id = f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(alert_type) % 10000}"
        
        return DriftAlert(
            alert_id=alert_id,
            timestamp=datetime.utcnow().isoformat(),
            alert_type=alert_type,
            severity=severity,
            model_version=model_version,
            drift_score=drift_score,
            threshold=threshold,
            details=details
        )
    
    def _store_drift_metrics(self, metrics: DriftMetrics) -> None:
        """Store drift metrics to file"""
        try:
            with open(self.drift_alerts_file.parent / "drift_metrics.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to store drift metrics: {e}")
    
    def _store_drift_alert(self, alert: DriftAlert) -> None:
        """Store drift alert to file"""
        try:
            with open(self.drift_alerts_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(alert), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to store drift alert: {e}")
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of current drift status"""
        with self._lock:
            if not self._drift_history:
                return {"status": "no_data", "message": "No drift data available"}
            
            latest_metrics = self._drift_history[-1]
            
            return {
                "status": "monitoring",
                "current_drift_score": latest_metrics.overall_drift_score,
                "drift_threshold": self.drift_threshold,
                "active_alerts": len(self._active_alerts),
                "last_check": latest_metrics.timestamp,
                "model_version": latest_metrics.model_version
            }
    
    def get_active_alerts(self) -> List[DriftAlert]:
        """Get list of active drift alerts"""
        with self._lock:
            return list(self._active_alerts.values())

# Global drift monitor instance
_drift_monitor = None

def get_drift_monitor() -> DriftMonitor:
    """Get the global drift monitor instance"""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor

def start_drift_monitoring() -> bool:
    """Start drift monitoring"""
    return get_drift_monitor().start_monitoring()

def get_drift_summary() -> Dict[str, Any]:
    """Get drift summary"""
    return get_drift_monitor().get_drift_summary()
