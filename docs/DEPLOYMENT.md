# AI Mood-Based Food Recommendation System - Deployment Guide

## Overview

This guide covers the deployment and operational management of the AI Mood-Based Food Recommendation System with production-grade features including model versioning, A/B testing, and monitoring.

## Prerequisites

### System Requirements
- Python 3.11+
- Redis server (for distributed stats and caching)
- 8GB+ RAM (for ML models and LLM processing)
- 2+ CPU cores
- 10GB+ disk space (for model versions and logs)

### Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- FastAPI
- Redis
- Prometheus Client
- MLflow (optional)
- Scikit-learn
- Sentence Transformers

## Environment Configuration

### 1. Environment Variables (.env)
```bash
# LLM API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL=deepseek/deepseek-r1-0528:free
LLM_MOCK_MODE=false

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=optional_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Monitoring
PROMETHEUS_PORT=9189
METRICS_ENABLED=true

# Model Configuration
MODEL_VERSION_RETENTION=10
AUTO_RETRAIN_ENABLED=true
RETRAIN_SCHEDULE=0 2 * * *  # Daily at 2 AM
```

### 2. Redis Setup
```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping
# Should return: PONG
```

## Application Startup

### 1. Development Mode
```bash
# Set environment variables
export PYTHONPATH=.
export LLM_MOCK_MODE=true  # For development

# Start the API server
uvicorn api.enhanced_routes:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Production Mode
```bash
# Set environment variables
export PYTHONPATH=.
export LLM_MOCK_MODE=false
export DEBUG=false

# Start with multiple workers
uvicorn api.enhanced_routes:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Docker Deployment
```bash
# Build the image
docker build -t moodfood-ai .

# Run the container
docker run -d \
  --name moodfood-api \
  -p 8000:8000 \
  -p 9189:9189 \
  -e OPENROUTER_API_KEY=your_key \
  -e REDIS_HOST=redis \
  --link redis:redis \
  moodfood-ai
```

## Monitoring Setup

### 1. Prometheus Metrics
The application exposes Prometheus metrics on port 9189:

```bash
# Start metrics server (automatic in production)
curl http://localhost:9189/metrics

# Key metrics:
# - moodfood_retrains_total
# - moodfood_model_deploys_total  
# - moodfood_current_model_version
# - moodfood_query_processing_time
# - moodfood_llm_api_calls_total
```

### 2. Health Check
```bash
# Application health
curl http://localhost:8000/health

# Component status
curl http://localhost:8000/logging/filter-stats
```

### 3. Dashboard Access
Access the monitoring dashboard at:
```
http://localhost:8000/static/monitoring.html
```

## Model Management

### 1. Manual Model Retraining
```bash
# Trigger retraining via API
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Or run script directly
python scripts/retrain_classifier.py
```

### 2. Model Deployment
```bash
# List available model versions
curl http://localhost:8000/retrain/versions

# Deploy specific version
curl -X POST http://localhost:8000/retrain/deploy \
  -H "Content-Type: application/json" \
  -d '{"version_id": "20231201_120000_abcd1234", "mode": "full"}'

# Canary deployment (5% traffic)
curl -X POST http://localhost:8000/retrain/deploy \
  -H "Content-Type: application/json" \
  -d '{"version_id": "20231201_130000_efgh5678", "mode": "canary"}'
```

### 3. Model Rollback
```bash
# Rollback to previous version
curl -X POST http://localhost:8000/retrain/rollback \
  -H "Content-Type: application/json" \
  -d '{"version_id": "20231201_100000_prev1234"}'
```

## A/B Testing

### 1. Start A/B Test
```bash
# Start with 5% traffic to canary
curl -X POST 'http://localhost:8000/retrain/abtest/start?version_id=canary_version&fraction=0.05'

# Gradually increase traffic
curl -X POST 'http://localhost:8000/retrain/abtest/start?version_id=canary_version&fraction=0.10'
```

### 2. Monitor A/B Test
```bash
# Check A/B test status
curl http://localhost:8000/retrain/abtest/status

# Monitor metrics in dashboard
# - Canary performance vs production
# - Error rates
# - Response times
```

### 3. Stop A/B Test
```bash
# Stop A/B test
curl -X POST http://localhost:8000/retrain/abtest/stop

# Promote canary to production (if successful)
curl -X POST http://localhost:8000/retrain/deploy \
  -H "Content-Type: application/json" \
  -d '{"version_id": "canary_version", "mode": "full"}'
```

## Backup and Recovery

### 1. Model Backup
```bash
# Models are automatically versioned in:
# models/intent_classifier/versions/

# Backup all versions
tar -czf model_backup_$(date +%Y%m%d).tar.gz models/

# Upload to cloud storage (example with AWS S3)
aws s3 cp model_backup_$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
```

### 2. Data Backup
```bash
# Backup logs and training data
tar -czf data_backup_$(date +%Y%m%d).tar.gz logs/ data/

# Backup Redis data
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb redis_backup_$(date +%Y%m%d).rdb
```

### 3. Recovery Procedure
```bash
# Restore models
tar -xzf model_backup_YYYYMMDD.tar.gz

# Restore Redis data
sudo systemctl stop redis-server
sudo cp redis_backup_YYYYMMDD.rdb /var/lib/redis/dump.rdb
sudo chown redis:redis /var/lib/redis/dump.rdb
sudo systemctl start redis-server

# Restart application
sudo systemctl restart moodfood-api
```

## Performance Tuning

### 1. Application Settings
```bash
# Increase worker processes
uvicorn api.enhanced_routes:app --workers 4

# Adjust memory limits for ML models
export SKLEARN_N_JOBS=2
export OMP_NUM_THREADS=2
```

### 2. Redis Optimization
```bash
# Redis configuration (/etc/redis/redis.conf)
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
```

### 3. LLM Rate Limiting
```bash
# Environment variables for rate limiting
export LLM_RETRIES=5
export LLM_BACKOFF_BASE=1.0
export LLM_TIMEOUT=30
export LLM_CIRCUIT_BREAKER_THRESHOLD=10
```

## Security

### 1. API Key Management
```bash
# Use environment variables or secret management
export OPENROUTER_API_KEY=$(cat /secrets/openrouter_key)

# For Kubernetes, use secrets:
kubectl create secret generic api-keys \
  --from-literal=openrouter-key=your_key_here
```

### 2. Network Security
```bash
# Firewall rules
sudo ufw allow 8000/tcp  # API
sudo ufw allow 9189/tcp  # Metrics (internal only)
sudo ufw deny 6379/tcp   # Redis (internal only)
```

### 3. HTTPS Configuration
```bash
# Use reverse proxy with SSL (nginx example)
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### 1. Common Issues

**Application won't start:**
```bash
# Check dependencies
pip list | grep -E "(fastapi|redis|sklearn)"

# Check environment variables
env | grep -E "(OPENROUTER|REDIS|LLM)"

# Check logs
tail -f logs/error_logs.jsonl
```

**Redis connection issues:**
```bash
# Test Redis connection
redis-cli ping

# Check Redis logs
sudo journalctl -u redis-server -f
```

**Model loading failures:**
```bash
# Check model files
ls -la models/intent_classifier/current/

# Verify symlink
readlink models/intent_classifier/current

# Check permissions
sudo chown -R app:app models/
```

### 2. Performance Issues

**High memory usage:**
```bash
# Monitor memory
free -h
ps aux | grep python

# Reduce model complexity or use fewer workers
export SKLEARN_N_JOBS=1
```

**Slow response times:**
```bash
# Check LLM API latency
curl -w "@curl-format.txt" -X POST http://localhost:8000/enhanced-recommend

# Monitor Redis performance
redis-cli --latency -h localhost
```

### 3. Model Issues

**Poor recommendation quality:**
```bash
# Check training data quality
python -c "from core.retraining.retrain_manager import RetrainManager; rm = RetrainManager(); print(rm.get_buffer_stats())"

# Review recent retraining logs
tail -n 50 logs/retrain_history.json

# Force retrain with clean data
curl -X POST http://localhost:8000/retrain -d '{"clean_data": true}'
```

## Maintenance

### 1. Regular Tasks

**Daily:**
- Check application health
- Monitor error rates
- Review recommendation quality

**Weekly:**
- Analyze A/B test results
- Review model performance trends
- Clean old log files

**Monthly:**
- Update dependencies
- Backup model versions
- Performance optimization review

### 2. Log Rotation
```bash
# Add to crontab
0 2 * * * find /app/logs -name "*.jsonl" -mtime +30 -delete
0 3 * * * find /app/models/intent_classifier/versions -maxdepth 1 -type d -mtime +90 -exec rm -rf {} \;
```

### 3. Monitoring Alerts
Set up alerts for:
- High error rates (>5%)
- Slow response times (>2s)
- Model accuracy degradation (<80%)
- Redis connection failures
- LLM API failures

## Support

### 1. Logs Location
```bash
# Application logs
logs/query_logs.jsonl       # User queries
logs/recommendation_logs.jsonl  # ML predictions
logs/error_logs.jsonl       # Errors
logs/retrain_history.json   # Retraining history

# System logs
/var/log/nginx/access.log   # Web server
/var/log/redis/redis-server.log  # Redis
```

### 2. Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Restart with debug output
uvicorn api.enhanced_routes:app --log-level debug
```

### 3. Contact Information
- **System Admin**: AI Moodfood
- **ML Team**: Team Moodfood
- **On-call**: Coming soon

---

*Last updated: $(date +%Y-%m-%d)*
