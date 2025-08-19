#!/usr/bin/env python3
"""
Comprehensive Query Logging System for Automatic Dataset Growth
Logs all user queries, API calls, responses, and metadata for future training
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)

@dataclass
class QueryLogEntry:
    """Structured log entry for all queries and responses"""
    # Query information
    query_id: str
    timestamp: str
    text_input: str
    image_input: Optional[str] = None
    audio_input: Optional[str] = None
    
    # User context
    user_context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Intent classification results
    primary_intent: Optional[str] = None
    confidence: Optional[float] = None
    all_intents: Optional[List[List[str]]] = None
    classification_method: Optional[str] = None
    
    # LLM results (if using hybrid)
    llm_labels: Optional[List[str]] = None
    validated_labels: Optional[List[str]] = None
    ml_labels: Optional[List[str]] = None
    comparison_score: Optional[float] = None
    
    # Recommendation results
    recommendations: Optional[List[Dict[str, Any]]] = None
    recommendation_count: Optional[int] = None
    
    # Performance metrics
    processing_time_ms: Optional[float] = None
    intent_time_ms: Optional[float] = None
    engine_time_ms: Optional[float] = None
    
    # System information
    api_version: str = "v1.0"
    system_status: str = "operational"
    
    # Metadata for dataset growth
    tags: Optional[List[str]] = None
    quality_score: Optional[float] = None
    notes: Optional[str] = None

class QueryLogger:
    """Main query logging system for automatic dataset growth"""
    
    def __init__(self, log_dir: str = "data/logs", auto_labeled_file: str = "data/auto_labeled.jsonl"):
        self.log_dir = Path(log_dir)
        self.auto_labeled_file = Path(auto_labeled_file)
        self.log_file = self.log_dir / "query_logs.jsonl"
        self.error_log_file = self.log_dir / "error_logs.jsonl"
        
        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for safe concurrent logging
        self._lock = threading.Lock()
        
        # Initialize logging
        self._setup_logging()
        
        logger.info(f"QueryLogger initialized: {self.log_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create log files if they don't exist
            if not self.log_file.exists():
                self.log_file.touch()
            if not self.error_log_file.exists():
                self.error_log_file.touch()
                
        except Exception as e:
            logger.error(f"Failed to setup logging files: {e}")
    
    def _generate_query_id(self) -> str:
        """Generate unique query ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"query_{timestamp}"
    
    def log_query(self, 
                  text_input: str,
                  image_input: Optional[str] = None,
                  audio_input: Optional[str] = None,
                  user_context: Optional[Dict[str, Any]] = None,
                  session_id: Optional[str] = None,
                  user_id: Optional[str] = None) -> str:
        """Log incoming query and return query ID"""
        
        query_id = self._generate_query_id()
        
        entry = QueryLogEntry(
            query_id=query_id,
            timestamp=datetime.now().isoformat(),
            text_input=text_input,
            image_input=image_input,
            audio_input=audio_input,
            user_context=user_context,
            session_id=session_id,
            user_id=user_id
        )
        
        try:
            with self._lock:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')
                    
            logger.info(f"Logged query {query_id}: {text_input[:50]}...")
            return query_id
            
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            return query_id
    
    def log_intent_results(self, 
                          query_id: str,
                          primary_intent: str,
                          confidence: float,
                          all_intents: List[List[str]],
                          method: str,
                          processing_time_ms: float):
        """Log intent classification results"""
        
        try:
            # Read existing log entry
            updated_entry = self._update_log_entry(query_id, {
                'primary_intent': primary_intent,
                'confidence': confidence,
                'all_intents': all_intents,
                'classification_method': method,
                'processing_time_ms': processing_time_ms,
                'intent_time_ms': processing_time_ms
            })
            
            if updated_entry:
                logger.info(f"Updated intent results for {query_id}: {primary_intent} ({confidence:.3f})")
                
        except Exception as e:
            logger.error(f"Failed to log intent results: {e}")
    
    def log_llm_results(self, 
                       query_id: str,
                       llm_labels: List[str],
                       validated_labels: List[str],
                       ml_labels: Optional[List[str]] = None,
                       comparison_score: Optional[float] = None):
        """Log LLM classification results"""
        
        try:
            updated_entry = self._update_log_entry(query_id, {
                'llm_labels': llm_labels,
                'validated_labels': validated_labels,
                'ml_labels': ml_labels,
                'comparison_score': comparison_score
            })
            
            if updated_entry:
                logger.info(f"Updated LLM results for {query_id}: {len(validated_labels)} labels")
                
        except Exception as e:
            logger.error(f"Failed to log LLM results: {e}")
    
    def log_recommendations(self, 
                           query_id: str,
                           recommendations: List[Dict[str, Any]],
                           engine_time_ms: float):
        """Log recommendation results"""
        
        try:
            updated_entry = self._update_log_entry(query_id, {
                'recommendations': recommendations,
                'recommendation_count': len(recommendations),
                'engine_time_ms': engine_time_ms
            })
            
            if updated_entry:
                logger.info(f"Updated recommendations for {query_id}: {len(recommendations)} items")
                
        except Exception as e:
            logger.error(f"Failed to log recommendations: {e}")
    
    def log_error(self, 
                  query_id: str,
                  error_type: str,
                  error_message: str,
                  stack_trace: Optional[str] = None):
        """Log errors for debugging and quality improvement"""
        
        error_entry = {
            'query_id': query_id,
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'stack_trace': stack_trace
        }
        
        try:
            with self._lock:
                with open(self.error_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(error_entry, ensure_ascii=False) + '\n')
                    
            logger.error(f"Logged error for {query_id}: {error_type}")
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def _update_log_entry(self, query_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing log entry with new information"""
        
        try:
            # Read all entries
            entries = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
            
            # Find and update the target entry
            for entry in entries:
                if entry.get('query_id') == query_id:
                    entry.update(updates)
                    break
            else:
                logger.warning(f"Query ID {query_id} not found for update")
                return False
            
            # Write back all entries
            with open(self.log_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update log entry: {e}")
            return False
    
    def export_for_training(self, output_file: str = "data/training_dataset.jsonl") -> int:
        """Export high-quality entries for training dataset"""
        
        try:
            training_entries = []
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        
                        # Filter for high-quality entries
                        if self._is_high_quality(entry):
                            training_entry = self._format_for_training(entry)
                            training_entries.append(training_entry)
            
            # Write training dataset
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in training_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Exported {len(training_entries)} entries to {output_file}")
            return len(training_entries)
            
        except Exception as e:
            logger.error(f"Failed to export training dataset: {e}")
            return 0
    
    def _is_high_quality(self, entry: Dict[str, Any]) -> bool:
        """Determine if an entry is high quality for training"""
        
        # Must have text input
        if not entry.get('text_input'):
            return False
        
        # Must have intent classification results
        if not entry.get('primary_intent'):
            return False
        
        # Must have reasonable confidence
        confidence = entry.get('confidence', 0)
        if confidence < 0.3:  # Low confidence entries are filtered out
            return False
        
        # Must have recommendations (successful queries)
        if not entry.get('recommendations'):
            return False
        
        return True
    
    def _format_for_training(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Format entry for training dataset"""
        
        return {
            'text': entry['text_input'],
            'labels': entry.get('validated_labels', [entry.get('primary_intent', '')]),
            'timestamp': entry['timestamp'],
            'method': entry.get('classification_method', 'unknown'),
            'llm_labels': entry.get('llm_labels', []),
            'ml_labels': entry.get('ml_labels', []),
            'confidence': entry.get('confidence', 0.0),
            'recommendations': entry.get('recommendations', []),
            'user_context': entry.get('user_context', {}),
            'quality_score': entry.get('quality_score', 1.0)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        
        try:
            total_queries = 0
            successful_queries = 0
            error_count = 0
            
            # Count total queries
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        total_queries += 1
                        entry = json.loads(line)
                        if entry.get('recommendations'):
                            successful_queries += 1
            
            # Count errors
            with open(self.error_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        error_count += 1
            
            return {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'error_count': error_count,
                'success_rate': (successful_queries / total_queries * 100) if total_queries > 0 else 0,
                'log_file_size': self.log_file.stat().st_size if self.log_file.exists() else 0,
                'error_log_size': self.error_log_file.stat().st_size if self.error_log_file.exists() else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

# Global logger instance
query_logger = QueryLogger()

def log_query_sync(*args, **kwargs) -> str:
    """Synchronous wrapper for query logging"""
    return query_logger.log_query(*args, **kwargs)

def log_intent_results_sync(*args, **kwargs):
    """Synchronous wrapper for intent logging"""
    return query_logger.log_intent_results(*args, **kwargs)

def log_llm_results_sync(*args, **kwargs):
    """Synchronous wrapper for LLM logging"""
    return query_logger.log_llm_results(*args, **kwargs)

def log_recommendations_sync(*args, **kwargs):
    """Synchronous wrapper for recommendation logging"""
    return query_logger.log_recommendations(*args, **kwargs)

def log_error_sync(*args, **kwargs):
    """Synchronous wrapper for error logging"""
    return query_logger.log_error(*args, **kwargs)
