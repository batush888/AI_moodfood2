# monitoring/metrics.py
from prometheus_client import Counter, Summary

REQUEST_COUNT = Counter("requests_total", "Total API Requests", ["endpoint"])
RECOMMENDATION_LATENCY = Summary("recommendation_latency_seconds", "Recommendation latency")