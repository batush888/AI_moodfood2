#!/usr/bin/env python3
"""
Retraining Monitor for Self-Learning Loop

This module tracks retraining metrics and pushes them to the monitoring dashboard
for real-time visibility into the ML evolution process.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class RetrainMetrics:
    """Retraining performance metrics"""
    timestamp: str
    model_version: str
    total_samples: int
    gold_samples: int
    silver_samples: int
    gold_ratio: float
    training_accuracy: float
    validation_accuracy: float
    training_loss: float
    validation_loss: float
    training_time_seconds: float
    buffer_size: int
    retrain_trigger: str  # "manual", "automatic", "scheduled"
    status: str  # "success", "failed", "skipped"
    notes: Optional[str] = None

class RetrainMonitor:
    """
    Monitors and logs retraining metrics for the self-learning loop.
    """
    
    def __init__(self, 
                 logfile: str = "logs/retrain_metrics.jsonl",
                 metrics_dir: str = "data/metrics"):
        
        self.logfile = Path(logfile)
        self.metrics_dir = Path(metrics_dir)
        
        # Ensure directories exist
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RetrainMonitor initialized: logfile={logfile}, metrics_dir={metrics_dir}")
    
    def log_metrics(self, metrics: RetrainMetrics) -> bool:
        """
        Log retraining metrics to the metrics file.
        
        Args:
            metrics: RetrainMetrics object to log
            
        Returns:
            bool: True if logging successful, False otherwise
        """
        try:
            # Convert to dict for JSON serialization
            metrics_dict = asdict(metrics)
            
            # Write to log file
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_dict, ensure_ascii=False) + "\n")
            
            # Also save individual metric file for easy access
            metric_file = self.metrics_dir / f"{metrics.model_version}_metrics.json"
            with open(metric_file, "w", encoding="utf-8") as f:
                json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Retrain metrics logged: {metrics.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log retrain metrics: {e}")
            return False
    
    def log_retrain_event(self, 
                          model_version: str,
                          total_samples: int,
                          gold_samples: int,
                          silver_samples: int,
                          gold_ratio: float,
                          training_accuracy: float,
                          validation_accuracy: float,
                          training_loss: float,
                          validation_loss: float,
                          training_time_seconds: float,
                          buffer_size: int,
                          retrain_trigger: str = "manual",
                          status: str = "success",
                          notes: Optional[str] = None) -> bool:
        """
        Convenience method to log a retrain event with individual parameters.
        
        Args:
            model_version: Version identifier for the trained model
            total_samples: Total number of training samples
            gold_samples: Number of high-quality samples
            silver_samples: Number of lower-quality samples
            gold_ratio: Ratio of gold to total samples
            training_accuracy: Training accuracy
            validation_accuracy: Validation accuracy
            training_loss: Training loss
            validation_loss: Validation loss
            training_time_seconds: Time taken for training
            buffer_size: Size of training buffer
            retrain_trigger: What triggered the retrain
            status: Status of the retrain operation
            notes: Optional additional notes
            
        Returns:
            bool: True if logging successful, False otherwise
        """
        metrics = RetrainMetrics(
            timestamp=datetime.now().isoformat(),
            model_version=model_version,
            total_samples=total_samples,
            gold_samples=gold_samples,
            silver_samples=silver_samples,
            gold_ratio=gold_ratio,
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            training_loss=training_loss,
            validation_loss=validation_loss,
            training_time_seconds=training_time_seconds,
            buffer_size=buffer_size,
            retrain_trigger=retrain_trigger,
            status=status,
            notes=notes
        )
        
        return self.log_metrics(metrics)
    
    def get_latest_metrics(self) -> Optional[RetrainMetrics]:
        """
        Get the most recent retraining metrics.
        
        Returns:
            Optional[RetrainMetrics]: Latest metrics or None if none available
        """
        if not self.logfile.exists():
            return None
        
        try:
            with open(self.logfile, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    return None
                
                # Get the last line (most recent)
                latest_line = lines[-1]
                metrics_dict = json.loads(latest_line)
                
                return RetrainMetrics(**metrics_dict)
                
        except Exception as e:
            logger.error(f"Failed to get latest metrics: {e}")
            return None
    
    def get_metrics_history(self, limit: int = 10) -> list[RetrainMetrics]:
        """
        Get recent retraining metrics history.
        
        Args:
            limit: Maximum number of metrics to return
            
        Returns:
            list[RetrainMetrics]: List of recent metrics
        """
        if not self.logfile.exists():
            return []
        
        try:
            with open(self.logfile, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                
                # Get the last N lines
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                
                metrics_list = []
                for line in recent_lines:
                    try:
                        metrics_dict = json.loads(line)
                        metrics_list.append(RetrainMetrics(**metrics_dict))
                    except Exception as e:
                        logger.warning(f"Failed to parse metrics line: {e}")
                        continue
                
                return metrics_list
                
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of retraining metrics for the monitoring dashboard.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not self.logfile.exists():
            return {
                "total_retrains": 0,
                "last_retrain": None,
                "avg_training_time": 0.0,
                "avg_accuracy": 0.0,
                "gold_ratio_trend": 0.0,
                "status": "no_metrics"
            }
        
        try:
            metrics_list = self.get_metrics_history(limit=50)  # Last 50 retrains
            
            if not metrics_list:
                return {"total_retrains": 0, "status": "no_metrics"}
            
            # Calculate summary statistics
            total_retrains = len(metrics_list)
            successful_retrains = [m for m in metrics_list if m.status == "success"]
            
            if successful_retrains:
                avg_training_time = sum(m.training_time_seconds for m in successful_retrains) / len(successful_retrains)
                avg_accuracy = sum(m.validation_accuracy for m in successful_retrains) / len(successful_retrains)
                avg_gold_ratio = sum(m.gold_ratio for m in successful_retrains) / len(successful_retrains)
            else:
                avg_training_time = 0.0
                avg_accuracy = 0.0
                avg_gold_ratio = 0.0
            
            # Calculate gold ratio trend (comparing recent vs older)
            if len(metrics_list) >= 2:
                recent_gold_ratio = metrics_list[-1].gold_ratio
                older_gold_ratio = metrics_list[0].gold_ratio
                gold_ratio_trend = recent_gold_ratio - older_gold_ratio
            else:
                gold_ratio_trend = 0.0
            
            # Get last retrain info
            latest = metrics_list[-1]
            last_retrain = {
                "timestamp": latest.timestamp,
                "model_version": latest.model_version,
                "status": latest.status,
                "samples": latest.total_samples,
                "accuracy": latest.validation_accuracy
            }
            
            return {
                "total_retrains": total_retrains,
                "successful_retrains": len(successful_retrains),
                "last_retrain": last_retrain,
                "avg_training_time": round(avg_training_time, 2),
                "avg_accuracy": round(avg_accuracy, 3),
                "avg_gold_ratio": round(avg_gold_ratio, 3),
                "gold_ratio_trend": round(gold_ratio_trend, 3),
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"total_retrains": 0, "status": "error", "error": str(e)}
    
    def get_performance_trends(self, window_days: int = 30) -> Dict[str, Any]:
        """
        Get performance trends over a time window.
        
        Args:
            window_days: Number of days to look back
            
        Returns:
            Dict[str, Any]: Performance trends
        """
        if not self.logfile.exists():
            return {"trends": "no_data"}
        
        try:
            cutoff_date = datetime.now().timestamp() - (window_days * 24 * 3600)
            
            with open(self.logfile, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            recent_metrics = []
            for line in lines:
                try:
                    metrics_dict = json.loads(line)
                    timestamp = datetime.fromisoformat(metrics_dict["timestamp"]).timestamp()
                    if timestamp >= cutoff_date:
                        recent_metrics.append(RetrainMetrics(**metrics_dict))
                except Exception:
                    continue
            
            if not recent_metrics:
                return {"trends": "no_recent_data"}
            
            # Calculate trends
            accuracy_trend = []
            gold_ratio_trend = []
            training_time_trend = []
            
            for metrics in recent_metrics:
                if metrics.status == "success":
                    accuracy_trend.append(metrics.validation_accuracy)
                    gold_ratio_trend.append(metrics.gold_ratio)
                    training_time_trend.append(metrics.training_time_seconds)
            
            if accuracy_trend:
                accuracy_change = accuracy_trend[-1] - accuracy_trend[0] if len(accuracy_trend) > 1 else 0.0
                gold_ratio_change = gold_ratio_trend[-1] - gold_ratio_trend[0] if len(gold_ratio_trend) > 1 else 0.0
                training_time_change = training_time_trend[-1] - training_time_trend[0] if len(training_time_trend) > 1 else 0.0
            else:
                accuracy_change = gold_ratio_change = training_time_change = 0.0
            
            return {
                "trends": "active",
                "window_days": window_days,
                "samples_in_window": len(recent_metrics),
                "accuracy_trend": {
                    "change": round(accuracy_change, 3),
                    "direction": "improving" if accuracy_change > 0 else "declining" if accuracy_change < 0 else "stable"
                },
                "gold_ratio_trend": {
                    "change": round(gold_ratio_change, 3),
                    "direction": "improving" if gold_ratio_change > 0 else "declining" if gold_ratio_change < 0 else "stable"
                },
                "training_time_trend": {
                    "change": round(training_time_change, 2),
                    "direction": "faster" if training_time_change < 0 else "slower" if training_time_change > 0 else "stable"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            return {"trends": "error", "error": str(e)}
    
    def clear_old_metrics(self, keep_days: int = 90) -> int:
        """
        Clear old metrics to prevent log file from growing too large.
        
        Args:
            keep_days: Number of days of metrics to keep
            
        Returns:
            int: Number of metrics removed
        """
        if not self.logfile.exists():
            return 0
        
        try:
            cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 3600)
            
            with open(self.logfile, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Filter recent metrics
            recent_lines = []
            removed_count = 0
            
            for line in lines:
                try:
                    metrics_dict = json.loads(line)
                    timestamp = datetime.fromisoformat(metrics_dict["timestamp"]).timestamp()
                    if timestamp >= cutoff_date:
                        recent_lines.append(line)
                    else:
                        removed_count += 1
                except Exception:
                    # Keep lines we can't parse
                    recent_lines.append(line)
            
            # Rewrite file with only recent metrics
            with open(self.logfile, "w", encoding="utf-8") as f:
                for line in recent_lines:
                    f.write(line + "\n")
            
            logger.info(f"Cleared {removed_count} old metrics (kept last {keep_days} days)")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to clear old metrics: {e}")
            return 0
