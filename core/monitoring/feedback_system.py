#!/usr/bin/env python3
"""
Feedback Integration System for User Feedback Tracking

This module tracks user feedback (explicit and implicit) to inform
retraining decisions and model improvement.
"""

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class UserFeedback:
    """User feedback entry"""
    feedback_id: str
    timestamp: str
    trace_id: str
    model_version: str
    feedback_type: str  # "explicit", "implicit"
    sentiment: str      # "positive", "negative", "neutral"
    query: str
    recommendations: List[str]
    user_rating: Optional[int] = None  # 1-5 scale for explicit feedback
    feedback_text: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class FeedbackMetrics:
    """Feedback metrics summary"""
    timestamp: str
    model_version: str
    total_feedback: int
    positive_feedback: int
    negative_feedback: int
    neutral_feedback: int
    positive_ratio: float
    average_rating: float
    feedback_trend: str  # "improving", "declining", "stable"

class FeedbackSystem:
    """
    Manages user feedback collection and analysis for model improvement.
    """
    
    def __init__(self, 
                 feedback_file: str = "logs/user_feedback.jsonl",
                 feedback_metrics_file: str = "logs/feedback_metrics.jsonl"):
        
        self.feedback_file = Path(feedback_file)
        self.feedback_metrics_file = Path(feedback_metrics_file)
        
        # Ensure log directories exist
        for log_file in [self.feedback_file, self.feedback_metrics_file]:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for safe concurrent access
        self._lock = threading.RLock()
        
        # Feedback history and metrics
        self._feedback_history: List[UserFeedback] = []
        self._feedback_metrics: Dict[str, FeedbackMetrics] = {}
        
        # Load existing feedback
        self._load_existing_feedback()
        
        logger.info(f"FeedbackSystem initialized: {self.feedback_file}")
    
    def record_explicit_feedback(self,
                                trace_id: str,
                                model_version: str,
                                query: str,
                                recommendations: List[str],
                                user_rating: int,
                                feedback_text: Optional[str] = None,
                                session_id: Optional[str] = None,
                                user_id: Optional[str] = None) -> str:
        """
        Record explicit user feedback (thumbs up/down, ratings).
        
        Args:
            trace_id: Associated trace ID
            model_version: Model version that generated recommendations
            query: User's original query
            recommendations: Recommendations provided
            user_rating: User rating (1-5 scale)
            feedback_text: Optional feedback text
            session_id: Optional session identifier
            user_id: Optional user identifier
            
        Returns:
            str: Generated feedback ID
        """
        feedback_id = f"feedback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(trace_id) % 10000}"
        
        # Determine sentiment from rating
        if user_rating >= 4:
            sentiment = "positive"
        elif user_rating <= 2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            timestamp=datetime.utcnow().isoformat(),
            trace_id=trace_id,
            model_version=model_version,
            feedback_type="explicit",
            sentiment=sentiment,
            query=query,
            recommendations=recommendations,
            user_rating=user_rating,
            feedback_text=feedback_text,
            session_id=session_id,
            user_id=user_id
        )
        
        self._store_feedback(feedback)
        self._feedback_history.append(feedback)
        
        # Update metrics
        self._update_feedback_metrics(model_version)
        
        logger.info(f"Recorded explicit feedback: {feedback_id} - {sentiment} (rating: {user_rating})")
        return feedback_id
    
    def record_implicit_feedback(self,
                                trace_id: str,
                                model_version: str,
                                query: str,
                                recommendations: List[str],
                                implicit_signal: str,  # "re_query", "skip", "accept", "modify"
                                session_id: Optional[str] = None,
                                user_id: Optional[str] = None) -> str:
        """
        Record implicit user feedback based on behavior.
        
        Args:
            trace_id: Associated trace ID
            model_version: Model version that generated recommendations
            query: User's original query
            recommendations: Recommendations provided
            implicit_signal: Implicit feedback signal
            session_id: Optional session identifier
            user_id: Optional user identifier
            
        Returns:
            str: Generated feedback ID
        """
        feedback_id = f"feedback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(trace_id) % 10000}"
        
        # Map implicit signals to sentiment
        sentiment_mapping = {
            "re_query": "negative",      # User re-querying suggests dissatisfaction
            "skip": "negative",          # Skipping recommendations suggests disinterest
            "accept": "positive",        # Accepting recommendations suggests satisfaction
            "modify": "neutral",         # Modifying suggests partial satisfaction
            "click": "positive",         # Clicking on recommendations suggests interest
            "share": "positive"          # Sharing suggests high satisfaction
        }
        
        sentiment = sentiment_mapping.get(implicit_signal, "neutral")
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            timestamp=datetime.utcnow().isoformat(),
            trace_id=trace_id,
            model_version=model_version,
            feedback_type="implicit",
            sentiment=sentiment,
            query=query,
            recommendations=recommendations,
            user_rating=None,
            feedback_text=f"Implicit signal: {implicit_signal}",
            session_id=session_id,
            user_id=user_id
        )
        
        self._store_feedback(feedback)
        self._feedback_history.append(feedback)
        
        # Update metrics
        self._update_feedback_metrics(model_version)
        
        logger.info(f"Recorded implicit feedback: {feedback_id} - {sentiment} (signal: {implicit_signal})")
        return feedback_id
    
    def _store_feedback(self, feedback: UserFeedback) -> None:
        """Store feedback to persistent storage"""
        try:
            with open(self.feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(feedback), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
    
    def _load_existing_feedback(self) -> None:
        """Load existing feedback from storage"""
        try:
            if not self.feedback_file.exists():
                return
            
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        feedback_data = json.loads(line.strip())
                        feedback = UserFeedback(**feedback_data)
                        self._feedback_history.append(feedback)
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            logger.info(f"Loaded {len(self._feedback_history)} existing feedback entries")
            
            # Initialize metrics for all model versions
            model_versions = set(f.model_version for f in self._feedback_history)
            for version in model_versions:
                self._update_feedback_metrics(version)
                
        except Exception as e:
            logger.error(f"Failed to load existing feedback: {e}")
    
    def _update_feedback_metrics(self, model_version: str) -> None:
        """Update feedback metrics for a specific model version"""
        try:
            # Filter feedback for this model version
            version_feedback = [f for f in self._feedback_history if f.model_version == model_version]
            
            if not version_feedback:
                return
            
            # Calculate metrics
            total_feedback = len(version_feedback)
            positive_feedback = len([f for f in version_feedback if f.sentiment == "positive"])
            negative_feedback = len([f for f in version_feedback if f.sentiment == "negative"])
            neutral_feedback = len([f for f in version_feedback if f.sentiment == "neutral"])
            
            positive_ratio = positive_feedback / total_feedback if total_feedback > 0 else 0.0
            
            # Calculate average rating (only for explicit feedback)
            explicit_feedback = [f for f in version_feedback if f.feedback_type == "explicit" and f.user_rating]
            average_rating = sum(f.user_rating for f in explicit_feedback) / len(explicit_feedback) if explicit_feedback else 0.0
            
            # Determine trend (simplified - compare recent vs older feedback)
            if len(version_feedback) >= 10:
                recent_feedback = version_feedback[-10:]
                older_feedback = version_feedback[-20:-10] if len(version_feedback) >= 20 else version_feedback[:-10]
                
                recent_positive = len([f for f in recent_feedback if f.sentiment == "positive"])
                older_positive = len([f for f in older_feedback if f.sentiment == "positive"])
                
                if recent_positive > older_positive:
                    feedback_trend = "improving"
                elif recent_positive < older_positive:
                    feedback_trend = "declining"
                else:
                    feedback_trend = "stable"
            else:
                feedback_trend = "insufficient_data"
            
            # Create metrics
            metrics = FeedbackMetrics(
                timestamp=datetime.utcnow().isoformat(),
                model_version=model_version,
                total_feedback=total_feedback,
                positive_feedback=positive_feedback,
                negative_feedback=negative_feedback,
                neutral_feedback=neutral_feedback,
                positive_ratio=positive_ratio,
                average_rating=average_rating,
                feedback_trend=feedback_trend
            )
            
            self._feedback_metrics[model_version] = metrics
            
            # Store metrics
            self._store_feedback_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Failed to update feedback metrics: {e}")
    
    def _store_feedback_metrics(self, metrics: FeedbackMetrics) -> None:
        """Store feedback metrics to file"""
        try:
            with open(self.feedback_metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to store feedback metrics: {e}")
    
    def get_feedback_summary(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        """Get feedback summary for a specific model version or overall"""
        with self._lock:
            if model_version:
                if model_version not in self._feedback_metrics:
                    return {"status": "no_data", "message": f"No feedback data for {model_version}"}
                
                metrics = self._feedback_metrics[model_version]
                return {
                    "status": "success",
                    "model_version": model_version,
                    "total_feedback": metrics.total_feedback,
                    "positive_ratio": metrics.positive_ratio,
                    "average_rating": metrics.average_rating,
                    "feedback_trend": metrics.feedback_trend,
                    "last_updated": metrics.timestamp
                }
            else:
                # Overall summary across all models
                if not self._feedback_metrics:
                    return {"status": "no_data", "message": "No feedback data available"}
                
                total_feedback = sum(m.total_feedback for m in self._feedback_metrics.values())
                overall_positive_ratio = sum(m.positive_ratio * m.total_feedback for m in self._feedback_metrics.values()) / total_feedback if total_feedback > 0 else 0.0
                
                return {
                    "status": "success",
                    "total_feedback": total_feedback,
                    "models_tracked": len(self._feedback_metrics),
                    "overall_positive_ratio": overall_positive_ratio,
                    "model_versions": list(self._feedback_metrics.keys())
                }
    
    def get_feedback_for_retraining(self, model_version: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent feedback for retraining decisions.
        
        Args:
            model_version: Model version to get feedback for
            hours: Hours of feedback to include
            
        Returns:
            List of feedback entries suitable for retraining
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            recent_feedback = []
            for feedback in self._feedback_history:
                if (feedback.model_version == model_version and 
                    datetime.fromisoformat(feedback.timestamp) >= cutoff_time):
                    
                    # Convert to retraining-friendly format
                    retraining_entry = {
                        "query": feedback.query,
                        "recommendations": feedback.recommendations,
                        "sentiment": feedback.sentiment,
                        "feedback_type": feedback.feedback_type,
                        "timestamp": feedback.timestamp,
                        "trace_id": feedback.trace_id
                    }
                    
                    recent_feedback.append(retraining_entry)
            
            return recent_feedback
            
        except Exception as e:
            logger.error(f"Failed to get feedback for retraining: {e}")
            return []
    
    def get_feedback_trends(self, model_version: str, days: int = 7) -> Dict[str, Any]:
        """Get feedback trends over time for a model version"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            # Group feedback by day
            daily_feedback = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
            
            for feedback in self._feedback_history:
                if feedback.model_version == model_version:
                    feedback_time = datetime.fromisoformat(feedback.timestamp)
                    if feedback_time >= cutoff_time:
                        day_key = feedback_time.strftime("%Y-%m-%d")
                        daily_feedback[day_key][feedback.sentiment] += 1
            
            # Convert to sorted list
            trends = []
            for day in sorted(daily_feedback.keys()):
                day_data = daily_feedback[day]
                total = sum(day_data.values())
                if total > 0:
                    trends.append({
                        "date": day,
                        "total": total,
                        "positive": day_data["positive"],
                        "negative": day_data["negative"],
                        "neutral": day_data["neutral"],
                        "positive_ratio": day_data["positive"] / total
                    })
            
            return {
                "model_version": model_version,
                "period_days": days,
                "trends": trends
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback trends: {e}")
            return {}

# Global feedback system instance
_feedback_system = None

def get_feedback_system() -> FeedbackSystem:
    """Get the global feedback system instance"""
    global _feedback_system
    if _feedback_system is None:
        _feedback_system = FeedbackSystem()
    return _feedback_system

def record_explicit_feedback(trace_id: str, model_version: str, **kwargs) -> str:
    """Record explicit user feedback"""
    return get_feedback_system().record_explicit_feedback(trace_id, model_version, **kwargs)

def record_implicit_feedback(trace_id: str, model_version: str, **kwargs) -> str:
    """Record implicit user feedback"""
    return get_feedback_system().record_implicit_feedback(trace_id, model_version, **kwargs)

def get_feedback_summary(model_version: Optional[str] = None) -> Dict[str, Any]:
    """Get feedback summary"""
    return get_feedback_system().get_feedback_summary(model_version)
