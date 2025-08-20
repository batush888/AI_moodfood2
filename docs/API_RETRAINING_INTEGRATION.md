# 🔄 API Retraining Integration

## Overview

The MoodFood AI system now includes **seamless API integration** with the automated retraining pipeline. This enables:

- **Hot reloading** of ML models without server restarts
- **Background retraining** via API endpoints
- **Real-time model updates** for continuous improvement
- **Comprehensive monitoring** and status tracking

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Request   │    │   Background     │    │   Hot Reload    │
│   POST /retrain │───▶│   Retraining     │───▶│   ML Classifier │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Immediate     │    │   Model          │    │   Updated       │
│   Response      │    │   Training       │    │   Predictions   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 New API Endpoints

### 1. **POST /retrain**
Triggers model retraining in the background.

**Request:**
```bash
curl -X POST http://localhost:8000/retrain
```

**Response:**
```json
{
  "status": "started",
  "message": "Model retraining started in background",
  "timestamp": "2025-08-20T14:30:00"
}
```

**Features:**
- **Background execution** - doesn't block API requests
- **Automatic hot reload** - new model loaded when ready
- **Comprehensive logging** - all events tracked
- **Error handling** - graceful failure management

### 2. **GET /retrain/status**
Get detailed retraining status and recommendations.

**Request:**
```bash
curl http://localhost:8000/retrain/status
```

**Response:**
```json
{
  "status": "ok",
  "retrain_status": {
    "last_retrain": "2025-08-20T03:00:00",
    "next_retrain_recommended": "Not yet - need 25 more samples or 15 more days",
    "logged_data_status": {
      "last_updated": "2025-08-20T14:25:00",
      "days_old": 15,
      "sample_count": 25
    },
    "model_status": {
      "last_updated": "2025-08-20T03:00:00",
      "days_old": 0,
      "exists": true
    },
    "recommendations": [
      "Consider running retraining pipeline"
    ]
  },
  "current_model_status": {
    "ml_classifier_loaded": true,
    "ml_labels_count": 138,
    "transformer_loaded": false
  },
  "timestamp": "2025-08-20T14:30:00"
}
```

## 🔧 Enhanced Intent Classifier

### **ML Classifier Integration**

The `EnhancedIntentClassifier` now includes:

```python
class EnhancedIntentClassifier:
    def __init__(self, taxonomy_path: str, model_dir: str = "models/intent_classifier"):
        # ML Classifier components
        self.ml_classifier = None
        self.ml_vectorizer = None
        self.ml_labels = []
        self.ml_classifier_path = self.model_dir / "ml_classifier.pkl"
        self.ml_mapping_path = self.model_dir / "label_mappings.json"
        
        # Load components
        self._load_ml_classifier()
    
    def reload_ml_classifier(self) -> bool:
        """Hot reload the ML classifier from disk."""
        # Implementation for hot reloading
    
    def _ml_classification(self, text: str) -> Dict[str, Any]:
        """Classify intent using ML classifier."""
        # Implementation for ML-based classification
```

### **Classification Priority**

The system now uses this priority order:

1. **ML Classifier** (fastest, most reliable)
2. **Transformer Model** (fallback)
3. **Keyword Fallback** (safety net)

## 📊 Monitoring & Logging

### **Retraining Events**

All retraining events are logged to the query logger:

```python
# Log retraining event
query_logger.log_retraining_event(
    trigger="api_triggered",
    status="success",
    details={
        "duration_seconds": 45.2,
        "accuracy": 0.85,
        "dataset_size": 1250,
        "new_samples": 150
    }
)
```

### **Event Types**

| Event | Description | Status Values |
|-------|-------------|---------------|
| `retraining` | Model retraining event | `started`, `success`, `failed` |
| `trigger` | What triggered retraining | `api_triggered`, `cron_triggered`, `manual` |

### **Health Check Updates**

The `/health` endpoint now includes ML classifier status:

```json
{
  "enhanced_classifier": {
    "ml_classifier_loaded": true,
    "ml_labels_count": 138,
    "transformer_loaded": false,
    "ml_classifier_available": true
  }
}
```

## 🔄 Hot Reloading Process

### **How It Works**

1. **API Request** → `POST /retrain`
2. **Background Task** → Retraining starts
3. **Model Training** → New ML classifier created
4. **File Save** → Model saved to disk
5. **Hot Reload** → `reload_ml_classifier()` called
6. **Memory Update** → New model loaded in memory
7. **Immediate Use** → Next requests use new model

### **Safety Features**

- **Backup Creation** → Old model backed up before retraining
- **Validation** → New model validated before deployment
- **Rollback** → Automatic rollback if validation fails
- **Thread Safety** → Safe concurrent access during reload

## 🧪 Testing

### **Test Script**

Use the provided test script:

```bash
python scripts/test_api_integration.py
```

This will test:
- ✅ Retraining status endpoint
- ✅ Model status endpoint  
- ✅ Recommendation endpoint
- ✅ Retraining endpoint

### **Manual Testing**

```bash
# 1. Check current status
curl http://localhost:8000/retrain/status

# 2. Trigger retraining
curl -X POST http://localhost:8000/retrain

# 3. Check status again
curl http://localhost:8000/retrain/status

# 4. Test recommendation (should use new model)
curl -X POST http://localhost:8000/enhanced-recommend \
  -H "Content-Type: application/json" \
  -d '{"text_input": "I want comfort food"}'
```

## 📈 Performance Impact

### **Before Integration**
- **Manual retraining** required server restart
- **Downtime** during model updates
- **No hot reloading** capability
- **Limited monitoring** of retraining

### **After Integration**
- **Zero downtime** retraining
- **Immediate model updates** via hot reload
- **Background processing** doesn't block API
- **Comprehensive monitoring** and logging

### **Expected Improvements**

| Metric | Before | After |
|--------|--------|-------|
| **Retraining Downtime** | 5-10 minutes | 0 seconds |
| **Model Update Time** | Server restart | Immediate |
| **API Availability** | Interrupted | Continuous |
| **Monitoring** | Manual | Automated |

## 🎯 Usage Examples

### **Production Deployment**

```python
# 1. Setup automated retraining
python scripts/setup_cron_retraining.py --setup

# 2. Monitor via API
curl http://localhost:8000/retrain/status

# 3. Manual retraining when needed
curl -X POST http://localhost:8000/retrain
```

### **Development Workflow**

```python
# 1. Make changes to training data
# 2. Test retraining
curl -X POST http://localhost:8000/retrain

# 3. Verify new model
curl http://localhost:8000/retrain/status

# 4. Test recommendations
curl -X POST http://localhost:8000/enhanced-recommend \
  -d '{"text_input": "test query"}'
```

### **Monitoring Dashboard**

```python
# Get comprehensive status
status_response = requests.get("http://localhost:8000/retrain/status")
health_response = requests.get("http://localhost:8000/health")

# Combine for dashboard
dashboard_data = {
    "retraining": status_response.json(),
    "system_health": health_response.json()
}
```

## 🔍 Troubleshooting

### **Common Issues**

#### **Retraining Fails**
```bash
# Check logs
tail -f data/logs/retrain.log

# Check status
curl http://localhost:8000/retrain/status

# Force retraining
curl -X POST http://localhost:8000/retrain
```

#### **Hot Reload Fails**
```bash
# Check model files
ls -la models/intent_classifier/

# Check API health
curl http://localhost:8000/health

# Restart API if needed
pkill -f "python.*enhanced_routes.py"
PYTHONPATH=. python api/enhanced_routes.py
```

#### **ML Classifier Not Loading**
```bash
# Check dependencies
pip install scikit-learn joblib

# Check model files
ls -la models/intent_classifier/ml_classifier.pkl

# Verify model format
python -c "import joblib; print(joblib.load('models/intent_classifier/ml_classifier.pkl').keys())"
```

### **Debug Mode**

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test ML classifier directly
from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
classifier = EnhancedIntentClassifier("data/taxonomy/mood_food_taxonomy.json")
print(f"ML Classifier loaded: {classifier.ml_classifier is not None}")
```

## 🚀 Best Practices

### **1. Production Deployment**
- **Monitor retraining** regularly via API
- **Set up alerts** for failed retraining
- **Backup models** before major updates
- **Test hot reloading** in staging

### **2. Development**
- **Use test script** before deployment
- **Monitor logs** during retraining
- **Validate models** after retraining
- **Test recommendations** with new models

### **3. Monitoring**
- **Check status** daily via API
- **Review logs** weekly
- **Monitor accuracy** improvements
- **Track retraining frequency**

## 🎉 Benefits

### **Immediate Benefits**
- **Zero downtime** model updates
- **Real-time improvements** from user data
- **Automated monitoring** and logging
- **Seamless integration** with existing API

### **Long-term Benefits**
- **Continuous learning** from user interactions
- **Improved accuracy** over time
- **Reduced maintenance** overhead
- **Better user experience** with updated models

## 📞 Support

### **Getting Help**
```bash
# Check all endpoints
curl http://localhost:8000/health
curl http://localhost:8000/retrain/status

# View logs
tail -f data/logs/retrain.log
tail -f data/logs/query_logs.jsonl

# Test integration
python scripts/test_api_integration.py
```

### **Common Commands**
```bash
# Trigger retraining
curl -X POST http://localhost:8000/retrain

# Check status
curl http://localhost:8000/retrain/status

# Test recommendation
curl -X POST http://localhost:8000/enhanced-recommend \
  -H "Content-Type: application/json" \
  -d '{"text_input": "I want comfort food"}'
```

**🎯 Your API is now fully integrated with automated retraining and hot reloading!**
