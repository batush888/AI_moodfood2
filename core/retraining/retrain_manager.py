#!/usr/bin/env python3
"""
Retraining Orchestrator for Flexible Self-Learning Loop

This module manages the continuous learning process by:
1. Reading recommendation logs
2. Classifying samples as gold/silver based on parse quality
3. Managing training buffer
4. Triggering retraining when thresholds are met
"""

import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrainingSample:
    """Structured training sample for ML retraining"""
    query: str
    labels: List[str]
    ml_pred: Dict[str, Any]
    source: str
    parse_status: str
    quality_score: float
    timestamp: str
    session_id: str

@dataclass
class BufferStats:
    """Statistics about the training buffer"""
    total_samples: int
    gold_samples: int
    silver_samples: int
    gold_ratio: float
    last_updated: str
    buffer_size: int

class RetrainManager:
    """
    Manages the self-learning loop by collecting and organizing training samples
    from hybrid filter interactions.
    """
    
    def __init__(self, 
                 log_file: str = "logs/recommendation_logs.jsonl",
                 buffer_file: str = "data/training_buffer.jsonl",
                 buffer_size: int = 100,
                 gold_threshold: float = 0.8):
        
        self.log_file = Path(log_file)
        self.buffer_file = Path(buffer_file)
        self.buffer_size = buffer_size
        self.gold_threshold = gold_threshold
        
        # Ensure data directory exists
        self.buffer_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RetrainManager initialized: buffer_size={buffer_size}, gold_threshold={gold_threshold}")
    
    def collect_samples(self) -> List[TrainingSample]:
        """
        Collect training samples from recommendation logs.
        
        Returns:
            List[TrainingSample]: Collected training samples
        """
        samples = []
        
        if not self.log_file.exists():
            logger.warning(f"Log file not found: {self.log_file}")
            return samples
        
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        
                        # Extract required fields
                        query = entry.get("query", "")
                        llm_interpret_parsed = entry.get("llm_interpret_parsed", [])
                        ml_prediction = entry.get("ml_prediction", {})
                        source = entry.get("source", "unknown")
                        parse_status = entry.get("llm_interpret_parse_status", "none")
                        quality_score = entry.get("quality_score", 0.0)
                        timestamp = entry.get("timestamp", "")
                        session_id = entry.get("session_id", "")
                        
                        # Only include samples with parsed LLM outputs
                        if query and llm_interpret_parsed and parse_status != "none":
                            sample = TrainingSample(
                                query=query,
                                labels=llm_interpret_parsed,
                                ml_pred=ml_prediction,
                                source=source,
                                parse_status=parse_status,
                                quality_score=quality_score,
                                timestamp=timestamp,
                                session_id=session_id
                            )
                            samples.append(sample)
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse log line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing log line {line_num}: {e}")
                        continue
            
            logger.info(f"Collected {len(samples)} training samples from logs")
            
        except Exception as e:
            logger.error(f"Failed to collect samples: {e}")
        
        return samples
    
    def classify_sample_quality(self, sample: TrainingSample) -> str:
        """
        Classify sample as gold or silver based on parse quality.
        
        Args:
            sample: Training sample to classify
            
        Returns:
            str: "gold" or "silver"
        """
        # Gold samples: high quality parses with good confidence
        if (sample.quality_score >= self.gold_threshold and 
            sample.parse_status in ["json", "code_block"]):
            return "gold"
        
        # Silver samples: lower quality but still usable
        elif sample.quality_score > 0.0:
            return "silver"
        
        # Reject samples with no quality
        else:
            return "reject"
    
    def filter_and_classify_samples(self, samples: List[TrainingSample]) -> Tuple[List[TrainingSample], List[TrainingSample]]:
        """
        Filter and classify samples into gold and silver sets.
        
        Args:
            samples: Raw training samples
            
        Returns:
            Tuple[List[TrainingSample], List[TrainingSample]]: Gold and silver samples
        """
        gold_samples = []
        silver_samples = []
        
        for sample in samples:
            quality_class = self.classify_sample_quality(sample)
            
            if quality_class == "gold":
                gold_samples.append(sample)
            elif quality_class == "silver":
                silver_samples.append(sample)
            # Reject samples are filtered out
        
        logger.info(f"Sample classification: {len(gold_samples)} gold, {len(silver_samples)} silver")
        
        return gold_samples, silver_samples
    
    def update_buffer(self) -> BufferStats:
        """
        Update the training buffer with new samples.
        
        Returns:
            BufferStats: Current buffer statistics
        """
        # Collect all samples
        all_samples = self.collect_samples()
        
        # Classify samples
        gold_samples, silver_samples = self.filter_and_classify_samples(all_samples)
        
        # Combine and sort by timestamp (newest first)
        combined_samples = gold_samples + silver_samples
        combined_samples.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Keep only the most recent samples up to buffer size
        buffer_samples = combined_samples[:self.buffer_size]
        
        # Write to buffer file
        try:
            with open(self.buffer_file, "w", encoding="utf-8") as f:
                for sample in buffer_samples:
                    # Convert to dict for JSON serialization
                    sample_dict = {
                        "query": sample.query,
                        "labels": sample.labels,
                        "ml_pred": sample.ml_pred,
                        "source": sample.source,
                        "parse_status": sample.parse_status,
                        "quality_score": sample.quality_score,
                        "timestamp": sample.timestamp,
                        "session_id": sample.session_id
                    }
                    f.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
            
            logger.info(f"Buffer updated: {len(buffer_samples)} samples")
            
        except Exception as e:
            logger.error(f"Failed to update buffer: {e}")
            return BufferStats(0, 0, 0, 0.0, "", self.buffer_size)
        
        # Calculate and return statistics
        gold_count = len([s for s in buffer_samples if self.classify_sample_quality(s) == "gold"])
        silver_count = len([s for s in buffer_samples if self.classify_sample_quality(s) == "silver"])
        gold_ratio = gold_count / len(buffer_samples) if buffer_samples else 0.0
        
        stats = BufferStats(
            total_samples=len(buffer_samples),
            gold_samples=gold_count,
            silver_samples=silver_count,
            gold_ratio=round(gold_ratio, 3),
            last_updated=datetime.now().isoformat(),
            buffer_size=self.buffer_size
        )
        
        return stats
    
    def should_trigger_retrain(self) -> bool:
        """
        Check if retraining should be triggered.
        
        Returns:
            bool: True if retraining should be triggered
        """
        if not self.buffer_file.exists():
            return False
        
        try:
            with open(self.buffer_file, "r") as f:
                sample_count = sum(1 for _ in f)
            
            should_retrain = sample_count >= self.buffer_size
            
            if should_retrain:
                logger.info(f"Retraining threshold met: {sample_count}/{self.buffer_size} samples")
            
            return should_retrain
            
        except Exception as e:
            logger.error(f"Failed to check retrain threshold: {e}")
            return False
    
    def get_buffer_stats(self) -> BufferStats:
        """
        Get current buffer statistics.
        
        Returns:
            BufferStats: Current buffer statistics
        """
        if not self.buffer_file.exists():
            return BufferStats(0, 0, 0, 0.0, "", self.buffer_size)
        
        try:
            with open(self.buffer_file, "r") as f:
                samples = [json.loads(line.strip()) for line in f if line.strip()]
            
            gold_count = len([s for s in samples if s.get("quality_score", 0) >= self.gold_threshold])
            silver_count = len([s for s in samples if 0 < s.get("quality_score", 0) < self.gold_threshold])
            gold_ratio = gold_count / len(samples) if samples else 0.0
            
            return BufferStats(
                total_samples=len(samples),
                gold_samples=gold_count,
                silver_samples=silver_count,
                gold_ratio=round(gold_ratio, 3),
                last_updated=datetime.now().isoformat(),
                buffer_size=self.buffer_size
            )
            
        except Exception as e:
            logger.error(f"Failed to get buffer stats: {e}")
            return BufferStats(0, 0, 0, 0.0, "", self.buffer_size)
    
    def clear_buffer(self):
        """Clear the training buffer."""
        try:
            if self.buffer_file.exists():
                self.buffer_file.unlink()
                logger.info("Training buffer cleared")
        except Exception as e:
            logger.error(f"Failed to clear buffer: {e}")
    
    def get_sample_distribution(self) -> Dict[str, Any]:
        """
        Get detailed sample distribution for monitoring.
        
        Returns:
            Dict[str, Any]: Sample distribution statistics
        """
        stats = self.get_buffer_stats()
        
        # Parse status distribution
        parse_status_counts = {}
        source_counts = {}
        
        if self.buffer_file.exists():
            try:
                with open(self.buffer_file, "r") as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line.strip())
                            parse_status = sample.get("parse_status", "unknown")
                            source = sample.get("source", "unknown")
                            
                            parse_status_counts[parse_status] = parse_status_counts.get(parse_status, 0) + 1
                            source_counts[source] = source_counts.get(source, 0) + 1
                            
            except Exception as e:
                logger.error(f"Failed to analyze sample distribution: {e}")
        
        return {
            "buffer_stats": {
                "total_samples": stats.total_samples,
                "gold_samples": stats.gold_samples,
                "silver_samples": stats.silver_samples,
                "gold_ratio": stats.gold_ratio,
                "last_updated": stats.last_updated
            },
            "parse_status_distribution": parse_status_counts,
            "source_distribution": source_counts,
            "buffer_size": self.buffer_size,
            "gold_threshold": self.gold_threshold
        }
