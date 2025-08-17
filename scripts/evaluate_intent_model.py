#!/usr/bin/env python3
"""
Evaluate Intent Model - Per-label metrics and analysis
Provides detailed per-label precision/recall/F1 to identify weak labels
"""

import json
import os
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_artifacts(model_dir):
    """Load model artifacts for evaluation."""
    model_dir = Path(model_dir)
    
    # Load label mappings
    with open(model_dir / "unified_label_mappings.json", "r") as f:
        mappings = json.load(f)
    
    # Load thresholds if available
    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
        logger.info(f"✅ Loaded {len(thresholds)} tuned thresholds")
    else:
        thresholds = None
        logger.warning("📁 No thresholds.json found, will use default 0.5")
    
    return mappings, thresholds

def load_validation_data(logs_dir):
    """Load validation predictions and true labels."""
    logs_dir = Path(logs_dir)
    
    # Look for validation data files
    val_logits_path = logs_dir / "val_logits.npy"
    val_true_path = logs_dir / "val_true.npy"
    
    if val_logits_path.exists() and val_true_path.exists():
        val_logits = np.load(val_logits_path)
        val_true = np.load(val_true_path)
        logger.info(f"✅ Loaded validation data: {val_logits.shape}")
        return val_logits, val_true
    else:
        logger.error("❌ Validation data not found. Run training first to generate val_logits.npy and val_true.npy")
        return None, None

def compute_per_label_metrics(val_logits, val_true, thresholds=None):
    """Compute per-label precision, recall, F1."""
    probs = 1.0 / (1.0 + np.exp(-val_logits))
    
    if thresholds is not None:
        # Use tuned thresholds
        y_pred = (probs >= np.array(thresholds)).astype(int)
        logger.info("🎯 Using tuned thresholds for prediction")
    else:
        # Use default threshold
        y_pred = (probs >= 0.5).astype(int)
        logger.info("📊 Using default 0.5 threshold for prediction")
    
    # Compute per-label metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        val_true, y_pred, average=None, zero_division=0
    )
    
    return y_pred, precision, recall, f1, support

def analyze_label_performance(precision, recall, f1, support, label_names):
    """Analyze and rank label performance."""
    # Create performance summary
    performance = []
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        if i < len(label_names):
            label_name = label_names[i]
        else:
            label_name = f"label_{i}"
        
        performance.append({
            'label': label_name,
            'precision': p,
            'recall': r,
            'f1': f,
            'support': s
        })
    
    # Sort by F1 score (worst first)
    performance.sort(key=lambda x: x['f1'])
    
    return performance

def print_performance_report(performance):
    """Print detailed performance report."""
    print("\n" + "="*80)
    print("🎯 INTENT MODEL EVALUATION REPORT")
    print("="*80)
    
    # Overall summary
    avg_f1 = np.mean([p['f1'] for p in performance])
    avg_precision = np.mean([p['precision'] for p in performance])
    avg_recall = np.mean([p['recall'] for p in performance])
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Average F1:      {avg_f1:.3f}")
    print(f"   Average Precision: {avg_precision:.3f}")
    print(f"   Average Recall:    {avg_recall:.3f}")
    print(f"   Total Labels:      {len(performance)}")
    
    # Top performers
    print(f"\n🏆 TOP PERFORMERS (F1 > 0.7):")
    top_performers = [p for p in performance if p['f1'] > 0.7]
    for p in top_performers:
        print(f"   {p['label']:<25} F1: {p['f1']:.3f} | P: {p['precision']:.3f} | R: {p['recall']:.3f}")
    
    # Weak labels (need attention)
    print(f"\n⚠️  WEAK LABELS (F1 < 0.3):")
    weak_labels = [p for p in performance if p['f1'] < 0.3]
    for p in weak_labels:
        print(f"   {p['label']:<25} F1: {p['f1']:.3f} | P: {p['precision']:.3f} | R: {p['recall']:.3f}")
    
    # Detailed breakdown
    print(f"\n📋 DETAILED BREAKDOWN:")
    print(f"{'Label':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
    print("-" * 70)
    
    for p in performance:
        print(f"{p['label']:<25} {p['f1']:<8.3f} {p['precision']:<10.3f} {p['recall']:<8.3f} {p['support']:<8.0f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if weak_labels:
        print(f"   • Add more training examples for weak labels: {', '.join([p['label'] for p in weak_labels[:5]])}")
        print(f"   • Consider boosting loss for labels with F1 < 0.3")
    
    if avg_f1 < 0.5:
        print(f"   • Overall performance is low. Consider:")
        print(f"     - Adding more training data")
        print(f"     - Adjusting class weights")
        print(f"     - Fine-tuning hyperparameters")
    
    print(f"   • Re-run threshold tuning if you've added new data")
    print(f"   • Monitor support counts - labels with < 5 examples may be unreliable")

def main():
    """Main evaluation function."""
    # Configuration
    MODEL_DIR = "models/intent_classifier"
    LOGS_DIR = "logs"
    
    logger.info("🔍 Starting intent model evaluation...")
    
    # Check if model exists
    if not os.path.exists(MODEL_DIR):
        logger.error(f"❌ Model directory not found: {MODEL_DIR}")
        logger.error("   Run training first: python scripts/train_intent_model.py")
        return
    
    # Load model artifacts
    try:
        mappings, thresholds = load_model_artifacts(MODEL_DIR)
        label_names = mappings.get('unified_id_to_label', [])
        logger.info(f"✅ Loaded {len(label_names)} label names")
    except Exception as e:
        logger.error(f"❌ Failed to load model artifacts: {e}")
        return
    
    # Load validation data
    val_logits, val_true = load_validation_data(LOGS_DIR)
    if val_logits is None:
        return
    
    # Compute metrics
    logger.info("📊 Computing per-label metrics...")
    y_pred, precision, recall, f1, support = compute_per_label_metrics(
        val_logits, val_true, thresholds
    )
    
    # Analyze performance
    performance = analyze_label_performance(precision, recall, f1, support, label_names)
    
    # Print report
    print_performance_report(performance)
    
    # Save detailed results
    results_path = Path(LOGS_DIR) / "evaluation_results.json"
    results = {
        'timestamp': str(np.datetime64('now')),
        'model_dir': MODEL_DIR,
        'overall_metrics': {
            'avg_f1': float(np.mean(f1)),
            'avg_precision': float(np.mean(precision)),
            'avg_recall': float(np.mean(recall))
        },
        'per_label_metrics': performance,
        'thresholds_used': thresholds is not None
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Detailed results saved to: {results_path}")
    logger.info("✅ Evaluation complete!")

if __name__ == "__main__":
    main()
