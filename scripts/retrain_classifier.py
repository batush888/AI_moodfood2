#!/usr/bin/env python3
"""
Monthly Retraining Script for Intent Classifier
-----------------------------------------------
This script automates:
1. Loading new training data from logs
2. Merging with existing dataset
3. Retraining the ML classifier
4. Saving updated model + label mappings
5. Performance evaluation and comparison
"""

import os
import json
import logging
import sys
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
try:
    from core.logging.query_logger import query_logger
    from utils.label_utils import load_taxonomy, load_dataset_labels
    from core.filtering.hybrid_filter import HybridFilter
    LOGGING_AVAILABLE = True
    FILTERING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    FILTERING_AVAILABLE = False
    logging.warning("Core logging/filtering components not available")

# Paths
LOGGED_DATA = "data/logs/training_dataset.jsonl"
BASE_DATASET = "data/intent_dataset.jsonl"
MODEL_DIR = "models/intent_classifier"
BACKUP_DIR = "models/backups"
MODEL_FILE = os.path.join(MODEL_DIR, "ml_classifier.pkl")
MAPPING_FILE = os.path.join(MODEL_DIR, "label_mappings.json")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
RETRAIN_LOG = "data/logs/retrain_history.jsonl"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RetrainingPipeline")

class AutomatedRetrainer:
    """Automated retraining pipeline for continuous model improvement"""
    
    def __init__(self):
        self.retrain_history = self._load_retrain_history()
        self.training_metrics = {}
        self.performance_comparison = {}
        self.filter_stats = {}
        
        # Initialize hybrid filter if available
        if FILTERING_AVAILABLE:
            self.hybrid_filter = HybridFilter()
            logger.info("Hybrid filtering system initialized")
        else:
            self.hybrid_filter = None
            logger.warning("Hybrid filtering not available - using basic filtering")
        
    def _load_retrain_history(self) -> List[Dict[str, Any]]:
        """Load retraining history for tracking improvements"""
        try:
            if os.path.exists(RETRAIN_LOG):
                with open(RETRAIN_LOG, 'r', encoding='utf-8') as f:
                    return [json.loads(line) for line in f if line.strip()]
            return []
        except Exception as e:
            logger.error(f"Failed to load retrain history: {e}")
            return []
    
    def _save_retrain_history(self, entry: Dict[str, Any]):
        """Save retraining history entry"""
        try:
            os.makedirs(os.path.dirname(RETRAIN_LOG), exist_ok=True)
            with open(RETRAIN_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to save retrain history: {e}")

    def log_retraining_event(self, trigger: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Log retraining event to the query logger"""
        try:
            if LOGGING_AVAILABLE:
                query_logger.log_retraining_event(trigger, status, details)
            else:
                logger.warning("Query logger not available for retraining event logging")
        except Exception as e:
            logger.error(f"Failed to log retraining event: {e}")
    
    def _backup_current_model(self) -> str:
        """Create backup of current model before retraining"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(BACKUP_DIR, f"model_backup_{timestamp}")
            
            if os.path.exists(MODEL_DIR):
                shutil.copytree(MODEL_DIR, backup_path)
                logger.info(f"✅ Current model backed up to: {backup_path}")
                return backup_path
            else:
                logger.warning("No existing model to backup")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to backup current model: {e}")
            return ""
    
    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load JSONL file with error handling"""
        try:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                return []
            
            with open(path, 'r', encoding='utf-8') as f:
                data = []
                for i, line in enumerate(f):
                    try:
                        if line.strip():
                            data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {i+1}: {e}")
                        continue
                
                logger.info(f"Loaded {len(data)} entries from {path}")
                return data
                
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return []
    
    def _save_jsonl(self, path: str, data: List[Dict[str, Any]]):
        """Save data to JSONL file"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(data)} entries to {path}")
        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")
    
    def _merge_datasets(self, base_data: List[Dict], logged_data: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Merge datasets with deduplication and quality analysis"""
        
        # Create text-to-entry mapping for deduplication
        text_map = {}
        duplicates = 0
        
        # Process base dataset first (preserve original)
        for entry in base_data:
            text = entry.get('text', '').strip().lower()
            if text and text not in text_map:
                text_map[text] = entry
        
        # Process logged data (newer entries override older ones)
        for entry in logged_data:
            text = entry.get('text', '').strip().lower()
            if text:
                if text in text_map:
                    duplicates += 1
                text_map[text] = entry
        
        merged_data = list(text_map.values())
        
        # Analyze dataset quality
        quality_metrics = {
            'total_entries': len(merged_data),
            'base_entries': len(base_data),
            'logged_entries': len(logged_data),
            'duplicates_removed': duplicates,
            'unique_entries': len(merged_data),
            'avg_labels_per_entry': sum(len(entry.get('labels', [])) for entry in merged_data) / len(merged_data) if merged_data else 0,
            'label_distribution': {}
        }
        
        # Count label distribution
        for entry in merged_data:
            for label in entry.get('labels', []):
                quality_metrics['label_distribution'][label] = quality_metrics['label_distribution'].get(label, 0) + 1
        
        logger.info(f"Dataset merge complete:")
        logger.info(f"  - Base entries: {len(base_data)}")
        logger.info(f"  - Logged entries: {len(logged_data)}")
        logger.info(f"  - Duplicates removed: {duplicates}")
        logger.info(f"  - Final dataset size: {len(merged_data)}")
        
        return merged_data, quality_metrics
    
    def _filter_training_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter training data using hybrid approach (ML + LLM fallback)"""
        
        logger.info(f"Filtering {len(data)} training samples...")
        
        if self.hybrid_filter:
            # Use hybrid filtering system
            logger.info("Using hybrid filtering system (ML + LLM fallback)")
            filtered_data, filter_stats = self.hybrid_filter.filter_training_data(data)
            
            # Store filter stats for logging
            self.filter_stats = filter_stats
            
            # Log detailed summary
            logger.info(self.hybrid_filter.get_filter_summary())
            
            return filtered_data
        else:
            # Fallback to basic filtering
            logger.info("Using basic filtering (hybrid filter not available)")
            
            # Track filtering stats
            filter_stats = {
                'original_count': len(data),
                'duplicates_removed': 0,
                'low_confidence_removed': 0,
                'malformed_removed': 0,
                'final_count': 0
            }
            
            # Remove duplicates (based on text content)
            seen_texts = set()
            unique_data = []
            
            for entry in data:
                text = entry.get('text', '').strip().lower()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    unique_data.append(entry)
                else:
                    filter_stats['duplicates_removed'] += 1
            
            # Remove low confidence samples
            high_confidence_data = []
            for entry in unique_data:
                confidence = entry.get('confidence', 1.0)
                if confidence >= 0.5:  # Minimum confidence threshold
                    high_confidence_data.append(entry)
                else:
                    filter_stats['low_confidence_removed'] += 1
            
            # Remove malformed entries
            valid_data = []
            for entry in high_confidence_data:
                text = entry.get('text', '').strip()
                labels = entry.get('labels', [])
                
                if text and len(text) >= 3 and labels and len(labels) > 0:
                    valid_data.append(entry)
                else:
                    filter_stats['malformed_removed'] += 1
            
            filter_stats['final_count'] = len(valid_data)
            
            logger.info(f"Basic data filtering complete:")
            logger.info(f"  Original: {filter_stats['original_count']}")
            logger.info(f"  Duplicates removed: {filter_stats['duplicates_removed']}")
            logger.info(f"  Low confidence removed: {filter_stats['low_confidence_removed']}")
            logger.info(f"  Malformed removed: {filter_stats['malformed_removed']}")
            logger.info(f"  Final: {filter_stats['final_count']}")
            
            # Store filter stats for logging
            self.filter_stats = filter_stats
            
            return valid_data

    def _prepare_training_data(self, data: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
        """Prepare training data for ML classifier"""
        
        # Apply quality filters
        filtered_data = self._filter_training_data(data)
        
        texts = []
        labels = []
        
        for entry in filtered_data:
            text = entry.get('text', '').strip()
            entry_labels = entry.get('labels', [])
            
            if text and entry_labels:
                texts.append(text)
                labels.append(entry_labels)
        
        logger.info(f"Prepared {len(texts)} training samples with labels")
        return texts, labels
    
    def _train_ml_classifier(self, texts: List[str], labels: List[List[str]]) -> Tuple[Any, Any, List[str]]:
        """Train ML classifier on prepared data"""
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score
            from sklearn.preprocessing import MultiLabelBinarizer
            import joblib
            
            # Convert labels to binary matrix
            mlb = MultiLabelBinarizer()
            Y = mlb.fit_transform(labels)
            
            # Vectorize text
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            X = vectorizer.fit_transform(texts)
            
            # Train/test split
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42, stratify=None
            )
            
            # Train classifier
            clf = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            
            logger.info("Training ML classifier...")
            clf.fit(X_train, Y_train)
            
            # Evaluate
            Y_pred = clf.predict(X_test)
            accuracy = accuracy_score(Y_test, Y_pred)
            
            # Generate detailed report
            report = classification_report(Y_test, Y_pred, target_names=mlb.classes_, output_dict=True)
            
            # Calculate F1 scores
            f1_macro = report.get('macro avg', {}).get('f1-score', 0)
            f1_weighted = report.get('weighted avg', {}).get('f1-score', 0)
            
            # Store metrics
            self.training_metrics = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'classification_report': report,
                'n_classes': len(mlb.classes_),
                'n_features': X.shape[1],
                'n_samples': len(texts),
                'training_date': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Training complete - Accuracy: {accuracy:.4f}")
            logger.info(f"   Classes: {len(mlb.classes_)}")
            logger.info(f"   Features: {X.shape[1]}")
            
            return clf, vectorizer, mlb.classes_.tolist()
            
        except ImportError as e:
            logger.error(f"Required ML libraries not available: {e}")
            logger.error("Please install: pip install scikit-learn joblib")
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            traceback.print_exc()
            raise
    
    def _load_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Load current model metrics for comparison"""
        try:
            if os.path.exists(METRICS_FILE):
                with open(METRICS_FILE, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                logger.info(f"Loaded current metrics: accuracy={metrics.get('accuracy', 0):.4f}")
                return metrics
            else:
                logger.info("No current metrics file found")
                return None
        except Exception as e:
            logger.error(f"Failed to load current metrics: {e}")
            return None

    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save model metrics to file"""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(METRICS_FILE, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Metrics saved to: {METRICS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def _compare_models(self, new_metrics: Dict[str, Any], old_metrics: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """Compare new model performance with old model"""
        
        comparison = {
            'new_accuracy': new_metrics.get('accuracy', 0),
            'new_f1_macro': new_metrics.get('f1_macro', 0),
            'old_accuracy': old_metrics.get('accuracy', 0) if old_metrics else 0,
            'old_f1_macro': old_metrics.get('f1_macro', 0) if old_metrics else 0,
            'accuracy_improvement': 0,
            'f1_improvement': 0,
            'should_deploy': False,
            'reason': ''
        }
        
        if old_metrics is None:
            # No previous model, deploy new one
            comparison['should_deploy'] = True
            comparison['reason'] = 'No previous model to compare against'
            logger.info("No previous model found, deploying new model")
            return True, comparison
        
        new_accuracy = new_metrics.get('accuracy', 0)
        new_f1 = new_metrics.get('f1_macro', 0)
        old_accuracy = old_metrics.get('accuracy', 0)
        old_f1 = old_metrics.get('f1_macro', 0)
        
        accuracy_improvement = new_accuracy - old_accuracy
        f1_improvement = new_f1 - old_f1
        
        comparison['accuracy_improvement'] = accuracy_improvement
        comparison['f1_improvement'] = f1_improvement
        
        # Deploy if accuracy improves or stays the same (with tolerance)
        accuracy_threshold = -0.01  # Allow 1% degradation tolerance
        f1_threshold = -0.01
        
        if accuracy_improvement >= accuracy_threshold and f1_improvement >= f1_threshold:
            comparison['should_deploy'] = True
            if accuracy_improvement > 0 or f1_improvement > 0:
                comparison['reason'] = f'Performance improved: accuracy +{accuracy_improvement:.4f}, F1 +{f1_improvement:.4f}'
            else:
                comparison['reason'] = 'Performance maintained within tolerance'
        else:
            comparison['should_deploy'] = False
            comparison['reason'] = f'Performance degraded: accuracy {accuracy_improvement:.4f}, F1 {f1_improvement:.4f}'
        
        logger.info(f"Model comparison: accuracy {accuracy_improvement:+.4f}, F1 {f1_improvement:+.4f}")
        logger.info(f"Deploy decision: {comparison['should_deploy']} - {comparison['reason']}")
        
        return comparison['should_deploy'], comparison

    def _save_model(self, classifier: Any, vectorizer: Any, labels: List[str]):
        """Save trained model and components"""
        
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Save model components
            import joblib
            model_data = {
                'classifier': classifier,
                'vectorizer': vectorizer,
                'labels': labels,
                'training_date': datetime.now().isoformat(),
                'metrics': self.training_metrics
            }
            
            joblib.dump(model_data, MODEL_FILE)
            
            # Save label mappings
            label_mappings = {
                'labels': labels,
                'n_labels': len(labels),
                'last_updated': datetime.now().isoformat(),
                'training_metrics': self.training_metrics
            }
            
            with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
                json.dump(label_mappings, f, indent=2, ensure_ascii=False)
            
            # Save metrics separately for easy comparison
            self._save_metrics(self.training_metrics)
            
            logger.info(f"✅ Model saved to: {MODEL_FILE}")
            logger.info(f"✅ Label mappings saved to: {MAPPING_FILE}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def _validate_model(self) -> bool:
        """Validate the newly trained model"""
        
        try:
            import joblib
            
            if not os.path.exists(MODEL_FILE):
                logger.error("Model file not found")
                return False
            
            # Load and test model
            model_data = joblib.load(MODEL_FILE)
            
            # Basic validation
            required_keys = ['classifier', 'vectorizer', 'labels']
            for key in required_keys:
                if key not in model_data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            # Test prediction
            test_text = "I want comfort food"
            vectorizer = model_data['vectorizer']
            classifier = model_data['classifier']
            
            X_test = vectorizer.transform([test_text])
            prediction = classifier.predict(X_test)
            
            logger.info(f"✅ Model validation successful")
            logger.info(f"   Test prediction shape: {prediction.shape}")
            logger.info(f"   Available labels: {len(model_data['labels'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _generate_retrain_report(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive retraining report"""
        
        report = {
            'retrain_id': f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'dataset_quality': quality_metrics,
            'training_metrics': self.training_metrics,
            'model_files': {
                'model_file': MODEL_FILE,
                'mapping_file': MAPPING_FILE
            },
            'system_info': {
                'python_version': sys.version,
                'working_directory': os.getcwd(),
                'available_memory': self._get_system_info()
            }
        }
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for monitoring"""
        
        try:
            import psutil
            return {
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'cpu_count': psutil.cpu_count(),
                'disk_usage': psutil.disk_usage('.').percent
            }
        except ImportError:
            return {'note': 'psutil not available'}
    
    def retrain(self, force: bool = False) -> bool:
        """Main retraining pipeline"""
        
        start_time = datetime.now()
        logger.info("🚀 Starting Automated Retraining Pipeline")
        logger.info("=" * 60)
        
        try:
            # Check if retraining is needed
            if not force and not self._should_retrain():
                logger.info("⏭️  Retraining not needed at this time")
                return True
            
            # Backup current model
            backup_path = self._backup_current_model()
            
            # Load datasets
            base_data = self._load_jsonl(BASE_DATASET)
            logged_data = self._load_jsonl(LOGGED_DATA)
            
            if not logged_data:
                logger.warning("No new logged training data found. Skipping retraining.")
                return False
            
            # Merge datasets
            merged_data, quality_metrics = self._merge_datasets(base_data, logged_data)
            
            # Prepare training data
            texts, labels = self._prepare_training_data(merged_data)
            
            if len(texts) < 10:
                logger.warning("Insufficient training data. Need at least 10 samples.")
                return False
            
            # Load current metrics for comparison
            old_metrics = self._load_current_metrics()
            
            # Train classifier
            classifier, vectorizer, label_list = self._train_ml_classifier(texts, labels)
            
            # Compare performance
            should_deploy, comparison = self._compare_models(self.training_metrics, old_metrics)
            
            if should_deploy:
                # Save new model
                self._save_model(classifier, vectorizer, label_list)
                
                # Validate model
                if not self._validate_model():
                    logger.error("❌ Model validation failed. Restoring backup...")
                    if backup_path and os.path.exists(backup_path):
                        shutil.rmtree(MODEL_DIR)
                        shutil.copytree(backup_path, MODEL_DIR)
                        logger.info("✅ Backup restored")
                    return False
            else:
                # Performance degraded, don't deploy
                logger.warning(f"❌ Retraining skipped: {comparison['reason']}")
                logger.info("Keeping existing model due to performance degradation")
                
                # Restore backup if we made one
                if backup_path and os.path.exists(backup_path):
                    shutil.rmtree(MODEL_DIR)
                    shutil.copytree(backup_path, MODEL_DIR)
                    logger.info("✅ Previous model restored")
                
                # Update training metrics to reflect the decision
                self.training_metrics = {
                    'accuracy': comparison['old_accuracy'],
                    'f1_macro': comparison['old_f1_macro'],
                    'deployment_status': 'rejected',
                    'reason': comparison['reason'],
                    'comparison': comparison,
                    'training_date': datetime.now().isoformat()
                }
            
            # Store comparison for logging
            self.performance_comparison = comparison
            
            # Save filter statistics
            if self.hybrid_filter:
                retrain_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.hybrid_filter.save_filter_stats(retrain_id)
            
            # Generate and save report
            report = self._generate_retrain_report(quality_metrics)
            self._save_retrain_history(report)
            
            # Calculate duration
            duration = datetime.now() - start_time
            
            # Log retraining event with performance details
            deployment_status = "deployed" if self.performance_comparison.get('should_deploy', False) else "rejected"
            self.log_retraining_event(
                trigger="api_triggered",
                status=deployment_status,
                details={
                    "duration_seconds": duration.total_seconds(),
                    "accuracy": self.training_metrics.get('accuracy', 0),
                    "f1_macro": self.training_metrics.get('f1_macro', 0),
                    "dataset_size": len(merged_data),
                    "new_samples": len(logged_data),
                    "filter_stats": self.filter_stats,
                    "comparison": self.performance_comparison,
                    "deployment_reason": self.performance_comparison.get('reason', 'Unknown')
                }
            )
            
            # Log success
            logger.info("🎉 Retraining Pipeline Completed Successfully!")
            logger.info(f"⏱️  Duration: {duration}")
            logger.info(f"📊 Final accuracy: {self.training_metrics.get('accuracy', 0):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Retraining pipeline failed: {e}")
            traceback.print_exc()
            
            # Log failure
            failure_report = {
                'retrain_id': f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self._save_retrain_history(failure_report)
            
            return False
    
    def _should_retrain(self) -> bool:
        """Determine if retraining is needed"""
        
        try:
            # Check if logged data exists and is recent
            if not os.path.exists(LOGGED_DATA):
                return False
            
            # Check file modification time
            file_time = datetime.fromtimestamp(os.path.getmtime(LOGGED_DATA))
            days_old = (datetime.now() - file_time).days
            
            # Retrain if data is older than 30 days or if forced
            if days_old >= 30:
                logger.info(f"Logged data is {days_old} days old. Retraining recommended.")
                return True
            
            # Check if we have significant new data
            logged_data = self._load_jsonl(LOGGED_DATA)
            if len(logged_data) >= 50:  # Retrain if we have 50+ new samples
                logger.info(f"Found {len(logged_data)} new training samples. Retraining recommended.")
                return True
            
            logger.info(f"Logged data is {days_old} days old with {len(logged_data)} samples. Retraining not needed yet.")
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
            return True  # Retrain on error to be safe
    
    def get_retrain_status(self) -> Dict[str, Any]:
        """Get current retraining status and recommendations"""
        
        try:
            status = {
                'last_retrain': None,
                'next_retrain_recommended': None,
                'logged_data_status': {},
                'model_status': {},
                'recommendations': []
            }
            
            # Check last retrain
            if self.retrain_history:
                last_retrain = self.retrain_history[-1]
                if last_retrain.get('status') == 'success':
                    status['last_retrain'] = last_retrain.get('timestamp')
            
            # Check logged data
            if os.path.exists(LOGGED_DATA):
                file_time = datetime.fromtimestamp(os.path.getmtime(LOGGED_DATA))
                days_old = (datetime.now() - file_time).days
                logged_data = self._load_jsonl(LOGGED_DATA)
                
                status['logged_data_status'] = {
                    'last_updated': file_time.isoformat(),
                    'days_old': days_old,
                    'sample_count': len(logged_data)
                }
                
                if days_old >= 30:
                    status['next_retrain_recommended'] = 'Overdue - data is 30+ days old'
                    status['recommendations'].append('Run retraining pipeline immediately')
                elif len(logged_data) >= 50:
                    status['next_retrain_recommended'] = 'Recommended - 50+ new samples available'
                    status['recommendations'].append('Consider running retraining pipeline')
                else:
                    status['next_retrain_recommended'] = f'Not yet - need {50 - len(logged_data)} more samples or {30 - days_old} more days'
            
            # Check model status
            if os.path.exists(MODEL_FILE):
                model_time = datetime.fromtimestamp(os.path.getmtime(MODEL_FILE))
                status['model_status'] = {
                    'last_updated': model_time.isoformat(),
                    'days_old': (datetime.now() - model_time).days,
                    'exists': True
                }
            else:
                status['model_status'] = {'exists': False}
                status['recommendations'].append('No trained model found - run retraining pipeline')
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting retrain status: {e}")
            return {'error': str(e)}

def main():
    """Main function for command-line usage"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Retraining Pipeline for Intent Classifier')
    parser.add_argument('--force', action='store_true', help='Force retraining even if not needed')
    parser.add_argument('--status', action='store_true', help='Show retraining status and recommendations')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually retraining')
    
    args = parser.parse_args()
    
    retrainer = AutomatedRetrainer()
    
    if args.status:
        status = retrainer.get_retrain_status()
        print("\n📊 Retraining Pipeline Status")
        print("=" * 40)
        print(f"Last retrain: {status.get('last_retrain', 'Never')}")
        print(f"Next retrain: {status.get('next_retrain_recommended', 'Unknown')}")
        print(f"Logged data: {status.get('logged_data_status', {}).get('sample_count', 0)} samples")
        print(f"Model exists: {status.get('model_status', {}).get('exists', False)}")
        
        if status.get('recommendations'):
            print("\n💡 Recommendations:")
            for rec in status['recommendations']:
                print(f"  - {rec}")
        
        return
    
    if args.dry_run:
        print("🔍 Dry Run Mode - No actual retraining will occur")
        status = retrainer.get_retrain_status()
        if retrainer._should_retrain():
            print("✅ Retraining would be triggered")
        else:
            print("⏭️  Retraining would be skipped")
        return
    
    # Run retraining
    success = retrainer.retrain(force=args.force)
    
    if success:
        print("🎉 Retraining completed successfully!")
        sys.exit(0)
    else:
        print("❌ Retraining failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
