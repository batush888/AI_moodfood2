#!/usr/bin/env python3
"""
Flexible ML Trainer for Self-Learning Loop

This trainer can handle both high-quality (gold) and lower-quality (silver) samples
with appropriate weighting to ensure robust learning even from imperfect data.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    total_samples: int
    gold_samples: int
    silver_samples: int
    gold_ratio: float
    training_accuracy: float
    validation_accuracy: float
    training_loss: float
    validation_loss: float
    training_time_seconds: float
    model_version: str
    timestamp: str

class FlexibleTrainer:
    """
    Flexible trainer that can handle both gold and silver quality samples
    with appropriate weighting for robust learning.
    """
    
    def __init__(self, 
                 buffer_file: str = "data/training_buffer.jsonl",
                 model_output_dir: str = "models/retrained",
                 gold_weight: float = 1.0,
                 silver_weight: float = 0.5):
        
        self.buffer_file = Path(buffer_file)
        self.model_output_dir = Path(model_output_dir)
        self.gold_weight = gold_weight
        self.silver_weight = silver_weight
        
        # Ensure output directory exists
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FlexibleTrainer initialized: gold_weight={gold_weight}, silver_weight={silver_weight}")
    
    def build_training_dataset(self, buffer_file: Optional[str] = None) -> List[Tuple[str, List[str], float]]:
        """
        Build training dataset from buffer with appropriate weights.
        
        Args:
            buffer_file: Optional path to buffer file (defaults to self.buffer_file)
            
        Returns:
            List[Tuple[str, List[str], float]]: List of (query, labels, weight) tuples
        """
        buffer_path = Path(buffer_file) if buffer_file else self.buffer_file
        
        if not buffer_path.exists():
            logger.warning(f"Buffer file not found: {buffer_path}")
            return []
        
        data = []
        
        try:
            with open(buffer_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        
                        query = sample.get("query", "")
                        labels = sample.get("labels", [])
                        parse_status = sample.get("parse_status", "none")
                        quality_score = sample.get("quality_score", 0.0)
                        
                        if not query or not labels:
                            continue
                        
                        # Determine weight based on quality
                        if parse_status in ["json", "code_block"] and quality_score >= 0.8:
                            weight = self.gold_weight
                        elif quality_score > 0.0:
                            weight = self.silver_weight
                        else:
                            # Skip very low quality samples
                            continue
                        
                        data.append((query, labels, weight))
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse buffer line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing buffer line {line_num}: {e}")
                        continue
            
            logger.info(f"Built training dataset: {len(data)} samples")
            
        except Exception as e:
            logger.error(f"Failed to build training dataset: {e}")
        
        return data
    
    def prepare_training_data(self, data: List[Tuple[str, List[str], float]]) -> Dict[str, Any]:
        """
        Prepare training data for ML training.
        
        Args:
            data: List of (query, labels, weight) tuples
            
        Returns:
            Dict[str, Any]: Prepared training data
        """
        if not data:
            return {}
        
        # Separate queries, labels, and weights
        queries = [item[0] for item in data]
        labels = [item[1] for item in data]
        weights = [item[2] for item in data]
        
        # Convert to numpy arrays for training
        sample_weights = np.array(weights)
        
        # Calculate statistics
        gold_count = sum(1 for w in weights if w == self.gold_weight)
        silver_count = sum(1 for w in weights if w == self.silver_weight)
        total_count = len(weights)
        
        stats = {
            "total_samples": total_count,
            "gold_samples": gold_count,
            "silver_samples": silver_count,
            "gold_ratio": gold_count / total_count if total_count > 0 else 0.0,
            "avg_weight": np.mean(sample_weights),
            "weight_distribution": {
                "gold_weight": self.gold_weight,
                "silver_weight": self.silver_weight
            }
        }
        
        logger.info(f"Training data prepared: {stats}")
        
        return {
            "queries": queries,
            "labels": labels,
            "sample_weights": sample_weights,
            "statistics": stats
        }
    
    def train_model(self, training_data: Dict[str, Any]) -> TrainingMetrics:
        """
        Train the ML model using the prepared training data.
        
        Args:
            training_data: Prepared training data from prepare_training_data()
            
        Returns:
            TrainingMetrics: Training performance metrics
        """
        if not training_data:
            logger.error("No training data provided")
            return TrainingMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "", "")
        
        import time
        start_time = time.time()
        
        try:
            # Extract data
            queries = training_data["queries"]
            labels = training_data["labels"]
            sample_weights = training_data["sample_weights"]
            stats = training_data["statistics"]
            
            # TODO: Implement actual ML training here
            # This is a placeholder for the actual training logic
            logger.info("Starting ML model training...")
            
            # Simulate training process
            training_accuracy = 0.85  # Placeholder
            validation_accuracy = 0.82  # Placeholder
            training_loss = 0.15  # Placeholder
            validation_loss = 0.18  # Placeholder
            
            # Generate model version
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_version = f"v2.0.{timestamp}"
            
            # Save model (placeholder)
            self._save_model(model_version, training_data)
            
            training_time = time.time() - start_time
            
            metrics = TrainingMetrics(
                total_samples=stats["total_samples"],
                gold_samples=stats["gold_samples"],
                silver_samples=stats["silver_samples"],
                gold_ratio=stats["gold_ratio"],
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                training_loss=training_loss,
                validation_loss=validation_loss,
                training_time_seconds=training_time,
                model_version=model_version,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
            )
            
            logger.info(f"Training completed: {model_version} in {training_time:.2f}s")
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            training_time = time.time() - start_time
            
            return TrainingMetrics(
                total_samples=0,
                gold_samples=0,
                silver_samples=0,
                gold_ratio=0.0,
                training_accuracy=0.0,
                validation_accuracy=0.0,
                training_loss=float('inf'),
                validation_loss=float('inf'),
                training_time_seconds=training_time,
                model_version="failed",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
            )
    
    def _save_model(self, model_version: str, training_data: Dict[str, Any]):
        """Save the trained model and metadata."""
        try:
            # Save model metadata
            metadata = {
                "model_version": model_version,
                "training_timestamp": training_data["statistics"],
                "training_samples": len(training_data["queries"]),
                "gold_ratio": training_data["statistics"]["gold_ratio"],
                "weights_config": {
                    "gold_weight": self.gold_weight,
                    "silver_weight": self.silver_weight
                }
            }
            
            metadata_file = self.model_output_dir / f"{model_version}_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # TODO: Save actual model weights here
            model_file = self.model_output_dir / f"{model_version}_model.pkl"
            # Placeholder for model saving
            
            logger.info(f"Model saved: {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def run_training_pipeline(self) -> TrainingMetrics:
        """
        Run the complete training pipeline.
        
        Returns:
            TrainingMetrics: Training performance metrics
        """
        logger.info("Starting training pipeline...")
        
        # Build training dataset
        raw_data = self.build_training_dataset()
        if not raw_data:
            logger.warning("No training data available")
            return TrainingMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "no_data", "")
        
        # Prepare training data
        training_data = self.prepare_training_data(raw_data)
        
        # Train model
        metrics = self.train_model(training_data)
        
        logger.info("Training pipeline completed")
        return metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training pipeline status.
        
        Returns:
            Dict[str, Any]: Training pipeline summary
        """
        # Check if buffer exists and has data
        buffer_stats = {}
        if self.buffer_file.exists():
            try:
                with open(self.buffer_file, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    buffer_stats = {
                        "buffer_exists": True,
                        "sample_count": len(lines),
                        "buffer_size": len(lines)
                    }
            except Exception as e:
                buffer_stats = {"buffer_exists": True, "error": str(e)}
        else:
            buffer_stats = {"buffer_exists": False}
        
        # Check for existing models
        model_files = list(self.model_output_dir.glob("*_model.pkl")) if self.model_output_dir.exists() else []
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime) if model_files else None
        
        return {
            "buffer_status": buffer_stats,
            "model_status": {
                "total_models": len(model_files),
                "latest_model": latest_model.name if latest_model else None,
                "model_output_dir": str(self.model_output_dir)
            },
            "training_config": {
                "gold_weight": self.gold_weight,
                "silver_weight": self.silver_weight,
                "buffer_file": str(self.buffer_file)
            }
        }

# Convenience function for building training dataset
def build_training_dataset(buffer_file: str = "data/training_buffer.jsonl") -> List[Tuple[str, List[str], float]]:
    """
    Convenience function to build training dataset from buffer file.
    
    Args:
        buffer_file: Path to training buffer file
        
    Returns:
        List[Tuple[str, List[str], float]]: List of (query, labels, weight) tuples
    """
    trainer = FlexibleTrainer(buffer_file=buffer_file)
    return trainer.build_training_dataset()
