# 🚀 Comprehensive Query Logging System

## Overview

The MoodFood AI system now includes a comprehensive query logging system that automatically captures **every user interaction** to grow your training dataset continuously. This system provides:

- **Automatic logging** of all queries, responses, and errors
- **Real-time monitoring** of system performance and quality
- **Dataset growth** through passive collection of user interactions
- **Training data export** for model improvement
- **Comprehensive analytics** for system optimization

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   Logging       │
│   (Browser)     │───▶│   (FastAPI)      │───▶│   System        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Local Storage   │    │ Query Logger     │    │ Training       │
│ (localStorage)  │    │ (Python)         │    │ Dataset Export │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 File Structure

```
core/logging/
├── query_logger.py          # Main logging system
├── __init__.py             # Package initialization

data/
├── logs/                   # Comprehensive query logs
│   ├── query_logs.jsonl    # All queries and responses
│   └── error_logs.jsonl    # Error tracking
├── auto_labeled.jsonl      # Legacy auto-labeled data
└── training_dataset.jsonl  # High-quality training export

scripts/
├── demo_logging.py         # Demonstration script
└── hybrid_infer.py         # Hybrid classification with logging

api/
└── enhanced_routes.py      # API with integrated logging
```

## 🔧 How It Works

### 1. **Automatic Query Capture**
Every user query is automatically logged with:
- **Query text** and context
- **User preferences** (time, weather, mood)
- **Session information** for tracking
- **Timestamp** for temporal analysis

### 2. **Comprehensive Result Logging**
The system logs:
- **Intent classification** results and confidence
- **LLM vs ML** comparison scores
- **Recommendation** generation and quality
- **Performance metrics** and timing
- **Error conditions** and debugging info

### 3. **Quality Assessment**
Entries are automatically scored for:
- **Classification confidence** (threshold: 0.3)
- **Successful recommendations** (must have results)
- **Complete data** (all required fields present)
- **Performance metrics** (response time, success rate)

### 4. **Training Dataset Export**
High-quality entries are automatically formatted for:
- **Model retraining** with new examples
- **Dataset expansion** with real user data
- **Quality improvement** through pattern analysis
- **Performance optimization** based on usage

## 🚀 Getting Started

### 1. **Start the API Server**
```bash
cd /path/to/moodfood
PYTHONPATH=. python api/enhanced_routes.py
```

### 2. **Use the Web Interface**
- Open `http://localhost:8000` in your browser
- Make food recommendation queries
- Watch as queries are automatically logged

### 3. **Monitor Logging Statistics**
```bash
# View real-time statistics
curl http://localhost:8000/logging/stats

# Export training dataset
curl -X POST http://localhost:8000/logging/export-training

# Get specific query details
curl http://localhost:8000/logging/query/{query_id}
```

### 4. **Run the Demo Script**
```bash
python scripts/demo_logging.py
```

## 📊 API Endpoints

### **GET** `/logging/stats`
Returns comprehensive logging statistics:
```json
{
  "status": "ok",
  "statistics": {
    "total_queries": 150,
    "successful_queries": 142,
    "error_count": 8,
    "success_rate": 94.67,
    "log_file_size": 245760,
    "error_log_size": 8192
  },
  "timestamp": "2025-08-20T02:45:00"
}
```

### **POST** `/logging/export-training`
Exports high-quality data for training:
```json
{
  "status": "ok",
  "exported_count": 89,
  "output_file": "data/training_dataset.jsonl",
  "timestamp": "2025-08-20T02:45:00"
}
```

### **GET** `/logging/query/{query_id}`
Returns detailed log for specific query:
```json
{
  "status": "ok",
  "query": {
    "query_id": "query_20250820_024500_123456",
    "timestamp": "2025-08-20T02:45:00",
    "text_input": "I want comfort food",
    "primary_intent": "goal_comfort",
    "confidence": 0.85,
    "recommendations": [...],
    "processing_time_ms": 150.0
  }
}
```

## 📈 Frontend Logging

The web interface automatically logs:
- **User queries** in real-time
- **API responses** and performance
- **Error conditions** and debugging
- **Local statistics** and monitoring

### **Export Local Logs**
- Click **📊 Export Logs** button in header
- Downloads JSON file with all local query data
- Useful for offline analysis and debugging

### **View Local Statistics**
- Click **📈 Log Stats** button in header
- Shows real-time statistics for current session
- Includes success rates and error counts

## 🔍 Data Analysis

### **Query Patterns**
Analyze common user requests:
```python
from core.logging.query_logger import query_logger

# Get all queries
stats = query_logger.get_statistics()
print(f"Total queries: {stats['total_queries']}")

# Export for analysis
query_logger.export_for_training("data/analysis_dataset.jsonl")
```

### **Performance Metrics**
Monitor system health:
- **Response times** by query type
- **Success rates** by intent category
- **Error patterns** and debugging info
- **User satisfaction** through feedback

### **Quality Assessment**
Identify improvement opportunities:
- **Low-confidence** classifications
- **Failed recommendations**
- **User feedback** patterns
- **System bottlenecks**

## 🎯 Dataset Growth Strategy

### **Passive Collection**
- **Every interaction** is automatically logged
- **No user intervention** required
- **Continuous improvement** through usage
- **Real-world data** for training

### **Quality Filtering**
- **Confidence thresholds** (0.3 minimum)
- **Success validation** (must have recommendations)
- **Complete data** requirements
- **Performance metrics** tracking

### **Training Export**
- **High-quality entries** automatically identified
- **Formatted for training** pipelines
- **Regular exports** for model updates
- **Version control** for dataset evolution

## 🛠️ Customization

### **Logging Configuration**
```python
# Custom log directory
logger = QueryLogger(
    log_dir="custom/logs",
    auto_labeled_file="custom/auto_labeled.jsonl"
)

# Custom quality thresholds
def custom_quality_check(entry):
    return entry.get('confidence', 0) > 0.5  # Higher threshold
```

### **Export Formats**
```python
# Custom export format
def custom_export_format(entry):
    return {
        'text': entry['text_input'],
        'intent': entry['primary_intent'],
        'quality_score': entry.get('quality_score', 1.0)
    }
```

## 📚 Best Practices

### **1. Regular Monitoring**
- Check statistics weekly: `GET /logging/stats`
- Monitor error rates and patterns
- Track performance metrics over time

### **2. Dataset Maintenance**
- Export training data monthly
- Archive old logs for historical analysis
- Clean up low-quality entries periodically

### **3. Quality Improvement**
- Analyze failed queries for patterns
- Use feedback data to improve recommendations
- Monitor confidence scores for model tuning

### **4. Performance Optimization**
- Track response times by query type
- Identify bottlenecks in processing pipeline
- Optimize based on usage patterns

## 🚨 Troubleshooting

### **Common Issues**

#### **Logging Not Working**
```bash
# Check if logging module is available
python -c "from core.logging.query_logger import query_logger; print('OK')"

# Verify directory permissions
ls -la data/logs/
```

#### **Empty Log Files**
```bash
# Check API server logs
tail -f api/enhanced_routes.py.log

# Verify logging integration
grep -n "log_query" api/enhanced_routes.py
```

#### **Performance Issues**
```bash
# Check log file sizes
du -sh data/logs/*

# Monitor API response times
curl -w "@curl-format.txt" http://localhost:8000/logging/stats
```

### **Debug Mode**
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Check individual logging calls
query_logger.log_query("test query")
```

## 🔮 Future Enhancements

### **Planned Features**
- **Real-time dashboard** for monitoring
- **Automated alerts** for system issues
- **Advanced analytics** and visualization
- **Machine learning** for quality prediction
- **Integration** with external monitoring tools

### **API Extensions**
- **WebSocket** for real-time updates
- **GraphQL** for flexible data querying
- **Bulk operations** for data management
- **Authentication** for secure access

## 📖 Examples

### **Complete Log Entry**
```json
{
  "query_id": "query_20250820_024500_123456",
  "timestamp": "2025-08-20T02:45:00.123456",
  "text_input": "I want something warm and comforting for dinner",
  "user_context": {
    "time_of_day": "evening",
    "weather": "cold",
    "social_context": "alone"
  },
  "primary_intent": "goal_comfort",
  "confidence": 0.85,
  "all_intents": [["goal_comfort", 0.85], ["sensory_warming", 0.78]],
  "classification_method": "hybrid_llm",
  "llm_labels": ["goal_comfort", "sensory_warming"],
  "validated_labels": ["goal_comfort", "sensory_warming"],
  "recommendations": [
    {
      "food_name": "chicken soup",
      "score": 0.9,
      "mood_match": 0.85
    }
  ],
  "processing_time_ms": 150.0,
  "quality_score": 1.0
}
```

### **Training Export Format**
```json
{
  "text": "I want something warm and comforting for dinner",
  "labels": ["goal_comfort", "sensory_warming"],
  "timestamp": "2025-08-20T02:45:00",
  "method": "hybrid_llm",
  "confidence": 0.85,
  "quality_score": 1.0
}
```

## 🎉 Conclusion

The Comprehensive Query Logging System transforms MoodFood AI from a static recommendation engine into a **continuously learning, self-improving system**. By automatically capturing every user interaction, it provides:

- **Unlimited dataset growth** through real usage
- **Continuous quality improvement** through pattern analysis
- **Performance optimization** through metrics tracking
- **Debugging capabilities** through comprehensive logging
- **Training data generation** for model evolution

Start using the system today and watch your AI model grow smarter with every query! 🚀
