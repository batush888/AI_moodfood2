#!/usr/bin/env python3
"""
Unit tests for RetrainManager
"""

import json
import tempfile
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from core.retraining.retrain_manager import RetrainManager, TrainingSample, BufferStats


class TestRetrainManager:
    """Test cases for RetrainManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test_recommendation_logs.jsonl"
        self.buffer_file = Path(self.temp_dir) / "test_training_buffer.jsonl"
        
        # Create sample log data
        self.sample_logs = [
            {
                "query": "I want Japanese food",
                "llm_interpret_parsed": ["sushi", "ramen", "tempura"],
                "ml_prediction": {"labels": ["goal_comfort"], "confidence": 0.12},
                "source": "llm_fallback",
                "llm_interpret_parse_status": "json",
                "quality_score": 0.9,
                "timestamp": "2025-01-27T10:00:00Z",
                "session_id": "session_123"
            },
            {
                "query": "I need something warm",
                "llm_interpret_parsed": ["soup", "hot chocolate"],
                "ml_prediction": {"labels": ["goal_comfort"], "confidence": 0.85},
                "source": "ml_validated",
                "llm_interpret_parse_status": "code_block",
                "quality_score": 0.8,
                "timestamp": "2025-01-27T10:01:00Z",
                "session_id": "session_124"
            },
            {
                "query": "Give me spicy food",
                "llm_interpret_parsed": ["curry", "hot wings"],
                "ml_prediction": {"labels": ["goal_energy"], "confidence": 0.45},
                "source": "llm_fallback",
                "llm_interpret_parse_status": "heuristic",
                "quality_score": 0.6,
                "timestamp": "2025-01-27T10:02:00Z",
                "session_id": "session_125"
            }
        ]
        
        # Write sample logs
        with open(self.log_file, "w") as f:
            for log in self.sample_logs:
                f.write(json.dumps(log) + "\n")
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test RetrainManager initialization"""
        manager = RetrainManager(
            log_file=str(self.log_file),
            buffer_file=str(self.buffer_file),
            buffer_size=50,
            gold_threshold=0.8
        )
        
        assert manager.log_file == self.log_file
        assert manager.buffer_file == self.buffer_file
        assert manager.buffer_size == 50
        assert manager.gold_threshold == 0.8
    
    def test_collect_samples(self):
        """Test sample collection from logs"""
        manager = RetrainManager(
            log_file=str(self.log_file),
            buffer_file=str(self.buffer_file)
        )
        
        samples = manager.collect_samples()
        
        assert len(samples) == 3
        assert all(isinstance(s, TrainingSample) for s in samples)
        
        # Check first sample
        first_sample = samples[0]
        assert first_sample.query == "I want Japanese food"
        assert first_sample.labels == ["sushi", "ramen", "tempura"]
        assert first_sample.parse_status == "json"
        assert first_sample.quality_score == 0.9
    
    def test_collect_samples_empty_log(self):
        """Test sample collection with empty log file"""
        empty_log = Path(self.temp_dir) / "empty_log.jsonl"
        empty_log.touch()
        
        manager = RetrainManager(log_file=str(empty_log))
        samples = manager.collect_samples()
        
        assert len(samples) == 0
    
    def test_collect_samples_missing_log(self):
        """Test sample collection with missing log file"""
        missing_log = Path(self.temp_dir) / "missing_log.jsonl"
        
        manager = RetrainManager(log_file=str(missing_log))
        samples = manager.collect_samples()
        
        assert len(samples) == 0
    
    def test_classify_sample_quality_gold(self):
        """Test gold sample classification"""
        manager = RetrainManager(gold_threshold=0.8)
        
        # High quality JSON sample
        sample = TrainingSample(
            query="test",
            labels=["label1"],
            ml_pred={},
            source="test",
            parse_status="json",
            quality_score=0.9,
            timestamp="2025-01-27T10:00:00Z",
            session_id="test"
        )
        
        quality = manager.classify_sample_quality(sample)
        assert quality == "gold"
    
    def test_classify_sample_quality_silver(self):
        """Test silver sample classification"""
        manager = RetrainManager(gold_threshold=0.8)
        
        # Medium quality heuristic sample
        sample = TrainingSample(
            query="test",
            labels=["label1"],
            ml_pred={},
            source="test",
            parse_status="heuristic",
            quality_score=0.6,
            timestamp="2025-01-27T10:00:00Z",
            session_id="test"
        )
        
        quality = manager.classify_sample_quality(sample)
        assert quality == "silver"
    
    def test_classify_sample_quality_reject(self):
        """Test sample rejection"""
        manager = RetrainManager(gold_threshold=0.8)
        
        # Low quality sample
        sample = TrainingSample(
            query="test",
            labels=["label1"],
            ml_pred={},
            source="test",
            parse_status="none",
            quality_score=0.0,
            timestamp="2025-01-27T10:00:00Z",
            session_id="test"
        )
        
        quality = manager.classify_sample_quality(sample)
        assert quality == "reject"
    
    def test_filter_and_classify_samples(self):
        """Test sample filtering and classification"""
        manager = RetrainManager(gold_threshold=0.8)
        
        # Create test samples
        samples = [
            TrainingSample(
                query="test1",
                labels=["label1"],
                ml_pred={},
                source="test",
                parse_status="json",
                quality_score=0.9,
                timestamp="2025-01-27T10:00:00Z",
                session_id="test1"
            ),
            TrainingSample(
                query="test2",
                labels=["label2"],
                ml_pred={},
                source="test",
                parse_status="heuristic",
                quality_score=0.6,
                timestamp="2025-01-27T10:01:00Z",
                session_id="test2"
            ),
            TrainingSample(
                query="test3",
                labels=["label3"],
                ml_pred={},
                source="test",
                parse_status="none",
                quality_score=0.0,
                timestamp="2025-01-27T10:02:00Z",
                session_id="test3"
            )
        ]
        
        gold_samples, silver_samples = manager.filter_and_classify_samples(samples)
        
        assert len(gold_samples) == 1
        assert len(silver_samples) == 1
        assert gold_samples[0].query == "test1"
        assert silver_samples[0].query == "test2"
    
    def test_update_buffer(self):
        """Test buffer update"""
        manager = RetrainManager(
            log_file=str(self.log_file),
            buffer_file=str(self.buffer_file),
            buffer_size=2  # Small buffer for testing
        )
        
        stats = manager.update_buffer()
        
        assert stats.total_samples == 2  # Limited by buffer size
        assert stats.buffer_size == 2
        assert stats.gold_samples >= 0
        assert stats.silver_samples >= 0
        assert 0.0 <= stats.gold_ratio <= 1.0
        
        # Check that buffer file was created
        assert self.buffer_file.exists()
        
        # Check buffer contents
        with open(self.buffer_file, "r") as f:
            buffer_lines = [line.strip() for line in f if line.strip()]
        
        assert len(buffer_lines) == 2
    
    def test_should_trigger_retrain(self):
        """Test retrain trigger logic"""
        manager = RetrainManager(
            log_file=str(self.log_file),
            buffer_file=str(self.buffer_file),
            buffer_size=2
        )
        
        # Buffer not full yet
        assert not manager.should_trigger_retrain()
        
        # Update buffer to fill it
        manager.update_buffer()
        
        # Now should trigger retrain
        assert manager.should_trigger_retrain()
    
    def test_get_buffer_stats(self):
        """Test buffer statistics retrieval"""
        manager = RetrainManager(
            log_file=str(self.log_file),
            buffer_file=str(self.buffer_file)
        )
        
        # Update buffer first
        manager.update_buffer()
        
        # Get stats
        stats = manager.get_buffer_stats()
        
        assert isinstance(stats, BufferStats)
        assert stats.total_samples > 0
        assert stats.buffer_size == manager.buffer_size
        assert 0.0 <= stats.gold_ratio <= 1.0
    
    def test_get_buffer_stats_empty(self):
        """Test buffer stats with empty buffer"""
        manager = RetrainManager(
            log_file=str(self.log_file),
            buffer_file=str(self.buffer_file)
        )
        
        stats = manager.get_buffer_stats()
        
        assert stats.total_samples == 0
        assert stats.gold_samples == 0
        assert stats.silver_samples == 0
        assert stats.gold_ratio == 0.0
    
    def test_clear_buffer(self):
        """Test buffer clearing"""
        manager = RetrainManager(
            log_file=str(self.log_file),
            buffer_file=str(self.buffer_file)
        )
        
        # Update buffer first
        manager.update_buffer()
        assert self.buffer_file.exists()
        
        # Clear buffer
        manager.clear_buffer()
        assert not self.buffer_file.exists()
    
    def test_get_sample_distribution(self):
        """Test sample distribution analysis"""
        manager = RetrainManager(
            log_file=str(self.log_file),
            buffer_file=str(self.buffer_file)
        )
        
        # Update buffer first
        manager.update_buffer()
        
        # Get distribution
        distribution = manager.get_sample_distribution()
        
        assert "buffer_stats" in distribution
        assert "parse_status_distribution" in distribution
        assert "source_distribution" in distribution
        assert distribution["buffer_size"] == manager.buffer_size
        assert distribution["gold_threshold"] == manager.gold_threshold
    
    def test_get_sample_distribution_empty(self):
        """Test sample distribution with empty buffer"""
        manager = RetrainManager(
            log_file=str(self.log_file),
            buffer_file=str(self.buffer_file)
        )
        
        distribution = manager.get_sample_distribution()
        
        assert distribution["buffer_stats"]["total_samples"] == 0
        assert distribution["parse_status_distribution"] == {}
        assert distribution["source_distribution"] == {}


class TestTrainingSample:
    """Test cases for TrainingSample dataclass"""
    
    def test_training_sample_creation(self):
        """Test TrainingSample creation"""
        sample = TrainingSample(
            query="test query",
            labels=["label1", "label2"],
            ml_pred={"confidence": 0.8},
            source="test_source",
            parse_status="json",
            quality_score=0.9,
            timestamp="2025-01-27T10:00:00Z",
            session_id="test_session"
        )
        
        assert sample.query == "test query"
        assert sample.labels == ["label1", "label2"]
        assert sample.ml_pred == {"confidence": 0.8}
        assert sample.source == "test_source"
        assert sample.parse_status == "json"
        assert sample.quality_score == 0.9
        assert sample.timestamp == "2025-01-27T10:00:00Z"
        assert sample.session_id == "test_session"


class TestBufferStats:
    """Test cases for BufferStats dataclass"""
    
    def test_buffer_stats_creation(self):
        """Test BufferStats creation"""
        stats = BufferStats(
            total_samples=100,
            gold_samples=80,
            silver_samples=20,
            gold_ratio=0.8,
            last_updated="2025-01-27T10:00:00Z",
            buffer_size=100
        )
        
        assert stats.total_samples == 100
        assert stats.gold_samples == 80
        assert stats.silver_samples == 20
        assert stats.gold_ratio == 0.8
        assert stats.last_updated == "2025-01-27T10:00:00Z"
        assert stats.buffer_size == 100


if __name__ == "__main__":
    pytest.main([__file__])
