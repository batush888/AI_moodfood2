#!/usr/bin/env python3
"""
Unit tests for FlexibleTrainer
"""

import json
import tempfile
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from core.retraining.trainer import FlexibleTrainer, TrainingMetrics, build_training_dataset


class TestFlexibleTrainer:
    """Test cases for FlexibleTrainer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.buffer_file = Path(self.temp_dir) / "test_training_buffer.jsonl"
        self.model_output_dir = Path(self.temp_dir) / "models"
        
        # Create sample buffer data
        self.sample_buffer = [
            {
                "query": "I want Japanese food",
                "labels": ["sushi", "ramen", "tempura"],
                "ml_pred": {"confidence": 0.12},
                "source": "llm_fallback",
                "parse_status": "json",
                "quality_score": 0.9,
                "timestamp": "2025-01-27T10:00:00Z",
                "session_id": "session_123"
            },
            {
                "query": "I need something warm",
                "labels": ["soup", "hot chocolate"],
                "ml_pred": {"confidence": 0.85},
                "source": "ml_validated",
                "parse_status": "code_block",
                "quality_score": 0.8,
                "timestamp": "2025-01-27T10:01:00Z",
                "session_id": "session_124"
            },
            {
                "query": "Give me spicy food",
                "labels": ["curry", "hot wings"],
                "ml_pred": {"confidence": 0.45},
                "source": "llm_fallback",
                "parse_status": "heuristic",
                "quality_score": 0.6,
                "timestamp": "2025-01-27T10:02:00Z",
                "session_id": "session_125"
            }
        ]
        
        # Write sample buffer
        with open(self.buffer_file, "w") as f:
            for sample in self.sample_buffer:
                f.write(json.dumps(sample) + "\n")
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test FlexibleTrainer initialization"""
        trainer = FlexibleTrainer(
            buffer_file=str(self.buffer_file),
            model_output_dir=str(self.model_output_dir),
            gold_weight=1.0,
            silver_weight=0.5
        )
        
        assert trainer.buffer_file == self.buffer_file
        assert trainer.model_output_dir == self.model_output_dir
        assert trainer.gold_weight == 1.0
        assert trainer.silver_weight == 0.5
        
        # Check that output directory was created
        assert self.model_output_dir.exists()
    
    def test_build_training_dataset(self):
        """Test training dataset building"""
        trainer = FlexibleTrainer(
            buffer_file=str(self.buffer_file),
            gold_weight=1.0,
            silver_weight=0.5
        )
        
        data = trainer.build_training_dataset()
        
        assert len(data) == 3
        assert all(len(item) == 3 for item in data)  # (query, labels, weight)
        
        # Check first sample
        first_item = data[0]
        assert first_item[0] == "I want Japanese food"  # query
        assert first_item[1] == ["sushi", "ramen", "tempura"]  # labels
        assert first_item[2] == 1.0  # weight (gold)
        
        # Check second sample (silver)
        second_item = data[1]
        assert second_item[2] == 0.5  # weight (silver)
    
    def test_build_training_dataset_empty_buffer(self):
        """Test dataset building with empty buffer"""
        empty_buffer = Path(self.temp_dir) / "empty_buffer.jsonl"
        empty_buffer.touch()
        
        trainer = FlexibleTrainer(buffer_file=str(empty_buffer))
        data = trainer.build_training_dataset()
        
        assert len(data) == 0
    
    def test_build_training_dataset_missing_buffer(self):
        """Test dataset building with missing buffer"""
        missing_buffer = Path(self.temp_dir) / "missing_buffer.jsonl"
        
        trainer = FlexibleTrainer(buffer_file=str(missing_buffer))
        data = trainer.build_training_dataset()
        
        assert len(data) == 0
    
    def test_build_training_dataset_custom_file(self):
        """Test dataset building with custom buffer file"""
        trainer = FlexibleTrainer()
        
        data = trainer.build_training_dataset(str(self.buffer_file))
        
        assert len(data) == 3
        assert all(len(item) == 3 for item in data)
    
    def test_prepare_training_data(self):
        """Test training data preparation"""
        trainer = FlexibleTrainer()
        
        # Create sample data
        raw_data = [
            ("query1", ["label1"], 1.0),
            ("query2", ["label2"], 0.5),
            ("query3", ["label3"], 1.0)
        ]
        
        training_data = trainer.prepare_training_data(raw_data)
        
        assert "queries" in training_data
        assert "labels" in training_data
        assert "sample_weights" in training_data
        assert "statistics" in training_data
        
        assert len(training_data["queries"]) == 3
        assert len(training_data["labels"]) == 3
        assert len(training_data["sample_weights"]) == 3
        
        # Check statistics
        stats = training_data["statistics"]
        assert stats["total_samples"] == 3
        assert stats["gold_samples"] == 2
        assert stats["silver_samples"] == 1
        assert stats["gold_ratio"] == 2/3
        assert stats["avg_weight"] == (2*1.0 + 1*0.5) / 3
    
    def test_prepare_training_data_empty(self):
        """Test training data preparation with empty data"""
        trainer = FlexibleTrainer()
        
        training_data = trainer.prepare_training_data([])
        
        assert training_data == {}
    
    def test_train_model_success(self):
        """Test successful model training"""
        trainer = FlexibleTrainer()
        
        # Create sample training data
        training_data = {
            "queries": ["query1", "query2"],
            "labels": [["label1"], ["label2"]],
            "sample_weights": [1.0, 0.5],
            "statistics": {
                "total_samples": 2,
                "gold_samples": 1,
                "silver_samples": 1,
                "gold_ratio": 0.5
            }
        }
        
        with patch.object(trainer, '_save_model') as mock_save:
            metrics = trainer.train_model(training_data)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.model_version != "failed"
        assert metrics.total_samples == 2
        assert metrics.gold_samples == 1
        assert metrics.silver_samples == 1
        assert metrics.gold_ratio == 0.5
        assert metrics.training_accuracy > 0
        assert metrics.validation_accuracy > 0
        assert metrics.training_time_seconds > 0
        
        # Check that save was called
        mock_save.assert_called_once()
    
    def test_train_model_failure(self):
        """Test model training failure"""
        trainer = FlexibleTrainer()
        
        # Pass invalid training data to trigger failure
        training_data = None
        
        metrics = trainer.train_model(training_data)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.model_version == "failed"
        assert metrics.total_samples == 0
        assert metrics.training_accuracy == 0.0
        assert metrics.validation_accuracy == 0.0
        assert metrics.training_loss == float('inf')
        assert metrics.validation_loss == float('inf')
    
    def test_train_model_exception(self):
        """Test model training with exception"""
        trainer = FlexibleTrainer()
        
        # Create sample training data
        training_data = {
            "queries": ["query1"],
            "labels": [["label1"]],
            "sample_weights": [1.0],
            "statistics": {"total_samples": 1, "gold_samples": 1, "silver_samples": 0, "gold_ratio": 1.0}
        }
        
        # Mock _save_model to raise exception
        with patch.object(trainer, '_save_model', side_effect=Exception("Save failed")):
            metrics = trainer.train_model(training_data)
        
        assert metrics.model_version == "failed"
    
    def test_save_model(self):
        """Test model saving"""
        trainer = FlexibleTrainer(model_output_dir=str(self.model_output_dir))
        
        training_data = {
            "statistics": {
                "total_samples": 10,
                "gold_samples": 8,
                "silver_samples": 2,
                "gold_ratio": 0.8
            }
        }
        
        model_version = "v2.0.20250127_120000"
        
        trainer._save_model(model_version, training_data)
        
        # Check that metadata file was created
        metadata_file = self.model_output_dir / f"{model_version}_metadata.json"
        assert metadata_file.exists()
        
        # Check metadata contents
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        assert metadata["model_version"] == model_version
        assert metadata["training_samples"] == 10
        assert metadata["gold_ratio"] == 0.8
        assert "weights_config" in metadata
    
    def test_run_training_pipeline_success(self):
        """Test successful training pipeline"""
        trainer = FlexibleTrainer(buffer_file=str(self.buffer_file))
        
        with patch.object(trainer, 'train_model') as mock_train:
            # Mock successful training
            mock_metrics = TrainingMetrics(
                total_samples=3,
                gold_samples=2,
                silver_samples=1,
                gold_ratio=2/3,
                training_accuracy=0.85,
                validation_accuracy=0.82,
                training_loss=0.15,
                validation_loss=0.18,
                training_time_seconds=5.0,
                model_version="v2.0.20250127_120000",
                timestamp="2025-01-27T12:00:00"
            )
            mock_train.return_value = mock_metrics
            
            metrics = trainer.run_training_pipeline()
        
        assert metrics == mock_metrics
        mock_train.assert_called_once()
    
    def test_run_training_pipeline_no_data(self):
        """Test training pipeline with no data"""
        empty_buffer = Path(self.temp_dir) / "empty_buffer.jsonl"
        empty_buffer.touch()
        
        trainer = FlexibleTrainer(buffer_file=str(empty_buffer))
        
        metrics = trainer.run_training_pipeline()
        
        assert metrics.model_version == "no_data"
    
    def test_get_training_summary(self):
        """Test training summary retrieval"""
        trainer = FlexibleTrainer(
            buffer_file=str(self.buffer_file),
            model_output_dir=str(self.model_output_dir)
        )
        
        summary = trainer.get_training_summary()
        
        assert "buffer_status" in summary
        assert "model_status" in summary
        assert "training_config" in summary
        
        # Check buffer status
        buffer_status = summary["buffer_status"]
        assert buffer_status["buffer_exists"] == True
        assert buffer_status["sample_count"] == 3
        
        # Check training config
        training_config = summary["training_config"]
        assert training_config["gold_weight"] == 1.0
        assert training_config["silver_weight"] == 0.5
        assert training_config["buffer_file"] == str(self.buffer_file)
    
    def test_get_training_summary_no_buffer(self):
        """Test training summary with no buffer"""
        missing_buffer = Path(self.temp_dir) / "missing_buffer.jsonl"
        
        trainer = FlexibleTrainer(buffer_file=str(missing_buffer))
        
        summary = trainer.get_training_summary()
        
        buffer_status = summary["buffer_status"]
        assert buffer_status["buffer_exists"] == False


class TestTrainingMetrics:
    """Test cases for TrainingMetrics dataclass"""
    
    def test_training_metrics_creation(self):
        """Test TrainingMetrics creation"""
        metrics = TrainingMetrics(
            total_samples=100,
            gold_samples=80,
            silver_samples=20,
            gold_ratio=0.8,
            training_accuracy=0.85,
            validation_accuracy=0.82,
            training_loss=0.15,
            validation_loss=0.18,
            training_time_seconds=30.5,
            model_version="v2.0.20250127_120000",
            timestamp="2025-01-27T12:00:00"
        )
        
        assert metrics.total_samples == 100
        assert metrics.gold_samples == 80
        assert metrics.silver_samples == 20
        assert metrics.gold_ratio == 0.8
        assert metrics.training_accuracy == 0.85
        assert metrics.validation_accuracy == 0.82
        assert metrics.training_loss == 0.15
        assert metrics.validation_loss == 0.18
        assert metrics.training_time_seconds == 30.5
        assert metrics.model_version == "v2.0.20250127_120000"
        assert metrics.timestamp == "2025-01-27T12:00:00"


class TestBuildTrainingDataset:
    """Test cases for build_training_dataset convenience function"""
    
    def test_build_training_dataset_convenience(self):
        """Test build_training_dataset convenience function"""
        # Create temporary buffer file
        temp_dir = tempfile.mkdtemp()
        buffer_file = Path(temp_dir) / "test_buffer.jsonl"
        
        try:
            # Create sample buffer
            sample_data = [
                {
                    "query": "test query",
                    "labels": ["label1"],
                    "parse_status": "json",
                    "quality_score": 0.9
                }
            ]
            
            with open(buffer_file, "w") as f:
                for sample in sample_data:
                    f.write(json.dumps(sample) + "\n")
            
            # Test convenience function
            data = build_training_dataset(str(buffer_file))
            
            assert len(data) == 1
            assert data[0][0] == "test query"  # query
            assert data[0][1] == ["label1"]    # labels
            assert data[0][2] == 1.0           # weight (gold)
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
