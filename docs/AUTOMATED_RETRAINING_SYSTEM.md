# 🤖 Automated Retraining System - Complete Implementation

## Overview

The AI Mood-Based Food Recommendation System now features a comprehensive automated retraining pipeline that continuously improves the model's performance while maintaining safety and transparency. This system includes performance safeguards, data quality filtering, automated scheduling, and real-time monitoring.

## 🎯 Key Features

### 1. Performance Safeguard
- **Automatic Model Comparison**: New models are only deployed if they improve or maintain performance
- **Tolerance System**: Allows 1% degradation tolerance to account for minor fluctuations
- **Metrics Tracking**: Monitors accuracy, F1-macro, and F1-weighted scores
- **Rollback Protection**: Automatically restores previous model if new one underperforms

### 2. Data Quality Filtering
- **Duplicate Removal**: Eliminates identical training samples
- **Confidence Filtering**: Removes low-confidence predictions (< 0.5)
- **Malformed Data Detection**: Filters out invalid or incomplete entries
- **Quality Metrics**: Tracks filtering statistics for transparency

### 3. Automated Scheduling
- **Weekly Retraining**: Every Sunday at 3 AM
- **Monthly Retraining**: 1st of each month at 2 AM
- **Adaptive Retraining**: Triggers when 100+ new samples or 7+ days have passed
- **Configurable Intervals**: Easy to adjust scheduling parameters

### 4. Real-Time Monitoring
- **Comprehensive Dashboard**: Web-based monitoring interface
- **API Endpoints**: RESTful endpoints for status and control
- **Performance Metrics**: Real-time accuracy and F1 score tracking
- **System Health**: Model loading status and fallback mode detection

### 5. Enhanced Logging
- **Structured Logging**: JSONL format for easy analysis
- **Retraining Events**: Detailed logs of all retraining activities
- **Performance Tracking**: Before/after metrics comparison
- **Filter Statistics**: Data quality improvement tracking

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Queries  │───▶│  Query Logger   │───▶│ Training Dataset│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Scheduler      │───▶│ Retrain Pipeline│───▶│ Model Validator │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Performance     │◀───│ Model Comparison│◀───│ New Model       │
│ Safeguard       │    └─────────────────┘    └─────────────────┘
└─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Hot Reload      │───▶│ Enhanced API    │───▶│ Monitoring      │
│ (if approved)   │    └─────────────────┘    │ Dashboard       │
└─────────────────┘                          └─────────────────┘
```

## 📁 File Structure

```
AI_moodfood2/
├── scripts/
│   ├── retrain_classifier.py          # Main retraining pipeline
│   ├── automated_scheduler.py         # Automated scheduling
│   └── test_performance_safeguard.py  # Integration tests
├── api/
│   └── enhanced_routes.py             # API endpoints
├── core/
│   └── logging/
│       └── query_logger.py            # Enhanced logging
├── frontend/
│   └── monitoring.html                # Monitoring dashboard
├── data/
│   ├── logs/
│   │   ├── query_logs.jsonl           # Query logs
│   │   ├── training_dataset.jsonl     # Auto-labeled data
│   │   └── retrain_history.json       # Retraining history
│   └── auto_labeled.jsonl             # LLM-labeled data
└── models/
    └── intent_classifier/
        ├── ml_classifier.pkl          # ML model
        ├── metrics.json               # Performance metrics
        └── label_mappings.json        # Label mappings
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install apscheduler joblib scikit-learn
```

### 2. Start the API Server

```bash
PYTHONPATH=. python api/enhanced_routes.py
```

### 3. Access the Monitoring Dashboard

Open `http://localhost:8000/static/monitoring.html` in your browser.

### 4. Start Automated Scheduler

```bash
curl -X POST http://localhost:8000/scheduler/start
```

## 📊 API Endpoints

### Model Management
- `GET /model/status` - Comprehensive model status
- `GET /retrain/metrics` - Current performance metrics
- `GET /retrain/status` - Retraining history and status
- `POST /retrain` - Trigger manual retraining

### Scheduler Control
- `GET /scheduler/status` - Scheduler status and configuration
- `POST /scheduler/start` - Start automated scheduler
- `POST /scheduler/stop` - Stop automated scheduler

### Logging and Monitoring
- `GET /logging/stats` - System usage statistics
- `GET /logging/export-training` - Export training dataset
- `GET /logging/query/{query_id}` - Get specific query details

## 🔧 Configuration

### Scheduler Configuration

```python
schedule_config = {
    'weekly_retrain': {
        'enabled': True,
        'day_of_week': 'sun',  # Sunday
        'hour': 3,             # 3 AM
        'minute': 0
    },
    'monthly_retrain': {
        'enabled': True,
        'day': 1,              # 1st of month
        'hour': 2,             # 2 AM
        'minute': 0
    },
    'adaptive_retrain': {
        'enabled': True,
        'min_samples': 100,    # Retrain when 100+ new samples
        'min_days': 7,         # Or at least 7 days
        'check_interval_hours': 6  # Check every 6 hours
    }
}
```

### Performance Safeguard Settings

```python
# Tolerance for performance degradation (1%)
PERFORMANCE_TOLERANCE = 0.01

# Minimum confidence threshold for data filtering
MIN_CONFIDENCE_THRESHOLD = 0.5

# Minimum training samples required
MIN_TRAINING_SAMPLES = 10
```

## 📈 Monitoring Dashboard

The monitoring dashboard provides real-time insights into:

### Model Performance
- Current accuracy and F1 scores
- Model version and training date
- Number of classes and features

### Retraining Status
- Last retraining date
- Retraining count
- Next recommended retraining

### Dataset Information
- Total training samples
- Dataset file status
- Data quality metrics

### System Health
- Model loading status
- Fallback mode detection
- Overall health score

### Automated Scheduler
- Scheduler status (running/stopped)
- Active jobs count
- Next scheduled runs

### Recent Activity
- Total queries processed
- Today's query count
- Retraining events
- Error count

## 🔍 Testing

### Run Integration Tests

```bash
python scripts/test_performance_safeguard.py
```

### Test API Endpoints

```bash
# Check model status
curl http://localhost:8000/model/status

# Trigger retraining
curl -X POST http://localhost:8000/retrain

# Start scheduler
curl -X POST http://localhost:8000/scheduler/start
```

### Test Performance Safeguard

```bash
# Simulate good retrain (improved performance)
python -c "
from scripts.retrain_classifier import AutomatedRetrainer
retrainer = AutomatedRetrainer()
old_metrics = {'accuracy': 0.75, 'f1_macro': 0.72}
new_metrics = {'accuracy': 0.82, 'f1_macro': 0.79}
should_deploy, comparison = retrainer._compare_models(new_metrics, old_metrics)
print(f'Should deploy: {should_deploy}')
print(f'Reason: {comparison[\"reason\"]}')
"
```

## 📝 Logging Format

### Retraining Event Log

```json
{
  "event_type": "retraining",
  "trigger": "api_triggered",
  "status": "deployed",
  "timestamp": "2025-08-20T14:30:00Z",
  "details": {
    "duration_seconds": 45.2,
    "accuracy": 0.85,
    "f1_macro": 0.82,
    "dataset_size": 1250,
    "new_samples": 150,
    "filter_stats": {
      "original_count": 1300,
      "duplicates_removed": 25,
      "low_confidence_removed": 15,
      "malformed_removed": 10,
      "final_count": 1250
    },
    "comparison": {
      "old_accuracy": 0.80,
      "new_accuracy": 0.85,
      "accuracy_improvement": 0.05,
      "should_deploy": true,
      "reason": "Performance improved: accuracy +0.0500"
    }
  },
  "performance_summary": {
    "old_accuracy": 0.80,
    "new_accuracy": 0.85,
    "accuracy_improvement": 0.05,
    "deployment_reason": "Performance improved: accuracy +0.0500"
  }
}
```

## 🛡️ Safety Features

### 1. Performance Safeguard
- **Automatic Comparison**: Every new model is compared against the current one
- **Tolerance System**: Allows minor performance fluctuations
- **Rollback Protection**: Automatically restores previous model if needed
- **Detailed Logging**: Records all decisions and reasons

### 2. Data Quality Protection
- **Duplicate Detection**: Prevents training on identical samples
- **Confidence Filtering**: Removes low-quality predictions
- **Validation Checks**: Ensures data integrity
- **Quality Metrics**: Tracks filtering effectiveness

### 3. System Stability
- **Backup Creation**: Creates backups before model updates
- **Validation Testing**: Tests new models before deployment
- **Error Handling**: Graceful failure recovery
- **Monitoring**: Real-time system health tracking

## 🎯 Benefits

### For Users
- **Consistent Performance**: Models only improve, never degrade
- **Transparent Operations**: Full visibility into system status
- **Reliable Service**: Automatic error recovery and fallbacks

### For Developers
- **Automated Maintenance**: No manual intervention required
- **Comprehensive Monitoring**: Real-time insights into system health
- **Easy Debugging**: Detailed logging and error tracking
- **Scalable Architecture**: Easy to extend and modify

### For System Performance
- **Continuous Improvement**: Models get better over time
- **Data-Driven Decisions**: Performance metrics guide improvements
- **Quality Assurance**: Multiple validation layers ensure reliability
- **Efficient Resource Usage**: Only retrains when beneficial

## 🔮 Future Enhancements

### Planned Features
1. **A/B Testing**: Compare multiple model versions
2. **Advanced Metrics**: Precision, recall, and confusion matrices
3. **Custom Thresholds**: Per-label performance requirements
4. **Model Ensembles**: Combine multiple model predictions
5. **Distributed Training**: Scale across multiple machines

### Potential Integrations
1. **MLflow**: Model versioning and experiment tracking
2. **Prometheus**: Advanced metrics and alerting
3. **Grafana**: Custom dashboards and visualizations
4. **Kubernetes**: Container orchestration for scaling

## 📚 Additional Resources

- [Performance Safeguard Documentation](PERFORMANCE_SAFEGUARD.md)
- [API Integration Guide](API_RETRAINING_INTEGRATION.md)
- [Frontend README](../frontend/README.md)
- [Hybrid LLM System Documentation](../frontend/README.md)

---

**🎉 The automated retraining system is now fully operational and ready to continuously improve your AI model while maintaining safety and transparency!**
