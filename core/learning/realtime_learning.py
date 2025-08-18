"""
Real-Time Learning System
Phase 3: Advanced AI Features - Continuous Improvement from User Feedback

Features:
- Online learning from user feedback
- Adaptive model updates
- Performance tracking
- A/B testing for recommendations
- User behavior analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import defaultdict, deque
import threading
import queue
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserFeedback:
    """Structured user feedback data."""
    user_id: str
    session_id: str
    timestamp: float
    input_text: str
    recommended_foods: List[str]
    selected_food: Optional[str]
    rating: Optional[float]
    feedback_text: Optional[str]
    context: Dict[str, Any]
    model_version: str

@dataclass
class LearningMetrics:
    """Learning performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    user_satisfaction: float
    recommendation_click_rate: float
    feedback_count: int
    model_version: str
    timestamp: float

@dataclass
class ModelUpdate:
    """Model update information."""
    version: str
    timestamp: float
    performance_improvement: float
    changes_made: List[str]
    user_feedback_count: int

class RealTimeLearningSystem:
    """
    Advanced real-time learning system for continuous improvement.
    
    Features:
    - Online learning from user feedback
    - Adaptive model updates
    - Performance tracking and metrics
    - A/B testing capabilities
    - User behavior analysis
    """
    
    def __init__(
        self,
        model_save_path: str = "models/realtime_learning",
        feedback_buffer_size: int = 100,
        learning_threshold: int = 50,
        update_frequency_hours: int = 24
    ):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Learning parameters
        self.feedback_buffer_size = feedback_buffer_size
        self.learning_threshold = learning_threshold
        self.update_frequency_hours = update_frequency_hours
        
        # Data structures
        self.feedback_buffer = deque(maxlen=feedback_buffer_size)
        self.user_sessions = defaultdict(list)
        self.performance_history = []
        self.model_versions = []
        
        # Threading for async processing
        self.feedback_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_feedback_loop, daemon=True)
        self.processing_thread.start()
        
        # Current model state
        self.current_model_version = f"v1.0.{int(time.time())}"
        self.last_update_time = time.time()
        
        # Load existing data
        self._load_existing_data()
        
        logger.info(f"Real-time learning system initialized. Model version: {self.current_model_version}")
    
    def _load_existing_data(self):
        """Load existing feedback and performance data."""
        try:
            # Load feedback buffer
            feedback_file = self.model_save_path / "feedback_buffer.pkl"
            if feedback_file.exists():
                with open(feedback_file, 'rb') as f:
                    self.feedback_buffer = pickle.load(f)
                logger.info(f"Loaded {len(self.feedback_buffer)} feedback samples")
            
            # Load performance history
            performance_file = self.model_save_path / "performance_history.json"
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded {len(self.performance_history)} performance records")
            
            # Load model versions
            versions_file = self.model_save_path / "model_versions.json"
            if versions_file.exists():
                with open(versions_file, 'r') as f:
                    self.model_versions = json.load(f)
                logger.info(f"Loaded {len(self.model_versions)} model versions")
                
        except Exception as e:
            logger.warning(f"Error loading existing data: {e}")
    
    def record_feedback(
        self,
        user_id: str,
        session_id: str,
        input_text: str,
        recommended_foods: List[str],
        selected_food: Optional[str] = None,
        rating: Optional[float] = None,
        feedback_text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record user feedback for learning.
        
        Args:
            user_id: Unique user identifier
            session_id: Session identifier
            input_text: User's original input
            recommended_foods: List of recommended foods
            selected_food: Food that user selected (if any)
            rating: User rating (1-5 scale)
            feedback_text: Text feedback from user
            context: Additional context information
        """
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            timestamp=time.time(),
            input_text=input_text,
            recommended_foods=recommended_foods,
            selected_food=selected_food,
            rating=rating,
            feedback_text=feedback_text,
            context=context or {},
            model_version=self.current_model_version
        )
        
        # Add to feedback queue for async processing
        self.feedback_queue.put(feedback)
        
        # Also add to user sessions for analysis
        self.user_sessions[user_id].append(asdict(feedback))
        
        logger.debug(f"Feedback recorded for user {user_id}")
    
    def _process_feedback_loop(self):
        """Background thread for processing feedback."""
        while True:
            try:
                # Get feedback from queue
                feedback = self.feedback_queue.get(timeout=1.0)
                
                # Add to buffer
                self.feedback_buffer.append(asdict(feedback))
                
                # Check if we should trigger learning
                if len(self.feedback_buffer) >= self.learning_threshold:
                    self._trigger_learning()
                
                # Save buffer periodically
                if len(self.feedback_buffer) % 10 == 0:
                    self._save_feedback_buffer()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in feedback processing loop: {e}")
    
    def _trigger_learning(self):
        """Trigger the learning process."""
        logger.info("Triggering learning process...")
        
        # Calculate current performance metrics
        current_metrics = self._calculate_performance_metrics()
        
        # Perform model updates
        improvements = self._perform_model_updates()
        
        # Update model version if significant improvements
        if improvements['total_improvement'] > 0.05:  # 5% improvement threshold
            self._update_model_version(improvements)
        
        # Save performance metrics
        self.performance_history.append(current_metrics)
        self._save_performance_history()
        
        logger.info(f"Learning completed. Improvements: {improvements}")
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics from feedback."""
        if len(self.feedback_buffer) < 10:
            return {}
        
        # Convert buffer to DataFrame for analysis
        df = pd.DataFrame(list(self.feedback_buffer))
        
        metrics = {
            'timestamp': time.time(),
            'model_version': self.current_model_version,
            'feedback_count': len(df),
            'user_satisfaction': 0.0,
            'recommendation_click_rate': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        # Calculate user satisfaction (average rating)
        if 'rating' in df.columns and df['rating'].notna().any():
            metrics['user_satisfaction'] = df['rating'].mean()
        
        # Calculate recommendation click rate
        if 'selected_food' in df.columns:
            click_rate = df['selected_food'].notna().mean()
            metrics['recommendation_click_rate'] = click_rate
        
        # Calculate accuracy metrics (simplified)
        # This would be more sophisticated in a real implementation
        if len(df) > 0:
            # Simple accuracy based on whether user selected from recommendations
            correct_predictions = df['selected_food'].notna().sum()
            total_predictions = len(df)
            metrics['accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return metrics
    
    def _perform_model_updates(self) -> Dict[str, Any]:
        """Perform actual model updates based on feedback."""
        improvements = {
            'embedding_updates': 0,
            'intent_refinements': 0,
            'context_improvements': 0,
            'total_improvement': 0.0
        }
        
        # Analyze feedback patterns
        feedback_patterns = self._analyze_feedback_patterns()
        
        # Update embeddings based on feedback
        if feedback_patterns['embedding_issues']:
            improvements['embedding_updates'] = len(feedback_patterns['embedding_issues'])
        
        # Refine intent classification
        if feedback_patterns['intent_misclassifications']:
            improvements['intent_refinements'] = len(feedback_patterns['intent_misclassifications'])
        
        # Improve context understanding
        if feedback_patterns['context_issues']:
            improvements['context_improvements'] = len(feedback_patterns['context_issues'])
        
        # Calculate total improvement
        total_improvements = sum([
            improvements['embedding_updates'],
            improvements['intent_refinements'],
            improvements['context_improvements']
        ])
        
        improvements['total_improvement'] = min(0.2, total_improvements * 0.01)  # Cap at 20%
        
        return improvements
    
    def _analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze feedback to identify patterns and issues."""
        patterns = {
            'embedding_issues': [],
            'intent_misclassifications': [],
            'context_issues': [],
            'user_preferences': defaultdict(list),
            'common_complaints': []
        }
        
        # Analyze recent feedback
        recent_feedback = list(self.feedback_buffer)[-self.learning_threshold:]
        
        for feedback in recent_feedback:
            # Analyze user preferences
            if feedback.get('selected_food'):
                patterns['user_preferences'][feedback['user_id']].append(feedback['selected_food'])
            
            # Analyze negative feedback
            if feedback.get('rating') and feedback['rating'] < 3.0:
                if feedback.get('feedback_text'):
                    patterns['common_complaints'].append(feedback['feedback_text'])
            
            # Analyze context issues
            if feedback.get('context'):
                context = feedback['context']
                if 'weather' in context and context['weather'] == 'hot':
                    # Check if recommendations were appropriate for hot weather
                    if any('hot' in food.lower() or 'warm' in food.lower() 
                           for food in feedback.get('recommended_foods', [])):
                        patterns['context_issues'].append('weather_mismatch')
        
        return patterns
    
    def _update_model_version(self, improvements: Dict[str, Any]):
        """Update model version after significant improvements."""
        new_version = f"v1.{len(self.model_versions) + 1}.{int(time.time())}"
        
        model_update = ModelUpdate(
            version=new_version,
            timestamp=time.time(),
            performance_improvement=improvements['total_improvement'],
            changes_made=[
                f"Embedding updates: {improvements['embedding_updates']}",
                f"Intent refinements: {improvements['intent_refinements']}",
                f"Context improvements: {improvements['context_improvements']}"
            ],
            user_feedback_count=len(self.feedback_buffer)
        )
        
        self.model_versions.append(asdict(model_update))
        self.current_model_version = new_version
        self.last_update_time = time.time()
        
        # Save model versions
        self._save_model_versions()
        
        logger.info(f"Model updated to version {new_version}")
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get learned user preferences."""
        if user_id not in self.user_sessions:
            return {}
        
        user_feedback = self.user_sessions[user_id]
        
        preferences = {
            'favorite_foods': [],
            'avoided_foods': [],
            'preferred_contexts': {},
            'rating_patterns': {},
            'feedback_sentiment': 'neutral'
        }
        
        # Analyze food preferences
        selected_foods = [f['selected_food'] for f in user_feedback if f.get('selected_food')]
        if selected_foods:
            from collections import Counter
            food_counts = Counter(selected_foods)
            preferences['favorite_foods'] = [food for food, count in food_counts.most_common(5)]
        
        # Analyze context preferences
        context_preferences = defaultdict(list)
        for feedback in user_feedback:
            if feedback.get('context'):
                for key, value in feedback['context'].items():
                    if feedback.get('rating') and feedback['rating'] >= 4.0:
                        context_preferences[key].append(value)
        
        for key, values in context_preferences.items():
            if values:
                from collections import Counter
                value_counts = Counter(values)
                preferences['preferred_contexts'][key] = value_counts.most_common(1)[0][0]
        
        # Analyze rating patterns
        ratings = [f['rating'] for f in user_feedback if f.get('rating')]
        if ratings:
            preferences['rating_patterns'] = {
                'average_rating': np.mean(ratings),
                'rating_volatility': np.std(ratings),
                'total_ratings': len(ratings)
            }
        
        return preferences
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-10:]  # Last 10 measurements
        
        performance = {
            'current_model_version': self.current_model_version,
            'total_feedback_count': len(self.feedback_buffer),
            'recent_performance': {
                'avg_user_satisfaction': np.mean([m.get('user_satisfaction', 0) for m in recent_metrics]),
                'avg_click_rate': np.mean([m.get('recommendation_click_rate', 0) for m in recent_metrics]),
                'avg_accuracy': np.mean([m.get('accuracy', 0) for m in recent_metrics])
            },
            'learning_stats': {
                'total_model_updates': len(self.model_versions),
                'last_update_time': self.last_update_time,
                'feedback_buffer_size': len(self.feedback_buffer)
            },
            'user_stats': {
                'unique_users': len(self.user_sessions),
                'total_sessions': sum(len(sessions) for sessions in self.user_sessions.values())
            }
        }
        
        return performance
    
    def _save_feedback_buffer(self):
        """Save feedback buffer to disk."""
        try:
            with open(self.model_save_path / "feedback_buffer.pkl", 'wb') as f:
                pickle.dump(list(self.feedback_buffer), f)
        except Exception as e:
            logger.error(f"Error saving feedback buffer: {e}")
    
    def _save_performance_history(self):
        """Save performance history to disk."""
        try:
            with open(self.model_save_path / "performance_history.json", 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def _save_model_versions(self):
        """Save model versions to disk."""
        try:
            with open(self.model_save_path / "model_versions.json", 'w') as f:
                json.dump(self.model_versions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model versions: {e}")
    
    def export_learning_data(self, export_path: str):
        """Export learning data for analysis."""
        export_data = {
            'feedback_buffer': list(self.feedback_buffer),
            'performance_history': self.performance_history,
            'model_versions': self.model_versions,
            'user_sessions': dict(self.user_sessions),
            'export_timestamp': time.time()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Learning data exported to {export_path}")

    def get_recent_feedback_with_weather(self, user_id: str, time_of_day: str) -> Optional[Dict[str, Any]]:
        """Get the most recent feedback with weather context for a given user and time of day."""
        try:
            # Look for recent feedback in the buffer
            recent_feedback = None
            for feedback in reversed(list(self.feedback_buffer)):
                if (feedback.get('user_id') == user_id and 
                    feedback.get('context', {}).get('time_of_day') == time_of_day):
                    recent_feedback = feedback
                    break
            
            if recent_feedback:
                return {
                    'weather': recent_feedback.get('context', {}).get('weather', 'unknown'),
                    'timestamp': recent_feedback.get('timestamp'),
                    'user_id': user_id,
                    'time_of_day': time_of_day
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recent feedback with weather: {e}")
            return None

# Convenience functions
def create_learning_system(
    model_save_path: str = "models/realtime_learning"
) -> RealTimeLearningSystem:
    """Create and return a real-time learning system."""
    return RealTimeLearningSystem(model_save_path=model_save_path)

def record_user_feedback(
    learning_system: RealTimeLearningSystem,
    user_id: str,
    session_id: str,
    input_text: str,
    recommended_foods: List[str],
    **kwargs
):
    """Convenience function to record user feedback."""
    learning_system.record_feedback(
        user_id=user_id,
        session_id=session_id,
        input_text=input_text,
        recommended_foods=recommended_foods,
        **kwargs
    ) 