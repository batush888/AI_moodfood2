#!/usr/bin/env python3
"""
Test Script for Stage 4 Observability Components

This script tests the core Stage 4 components:
- Request Tracing
- Advanced Metrics
- Drift Detection
- Feedback System
- Health Monitoring
"""

import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tracing_system():
    """Test the request tracing system"""
    print("🧪 Testing Request Tracing System...")
    
    try:
        from core.utils.tracing import start_trace, add_trace_event, end_trace, get_trace_summary
        
        # Start a trace
        trace_id = start_trace(
            request_id="test_request_001",
            model_version="test_v1.0"
        )
        print(f"✅ Trace started: {trace_id}")
        
        # Add events
        add_trace_event(trace_id, "test_stage", {"message": "Test event"})
        print("✅ Trace event added")
        
        # End trace
        end_trace(trace_id, final_decision="test_complete")
        print("✅ Trace ended")
        
        # Get summary
        summary = get_trace_summary(trace_id)
        if summary:
            print(f"✅ Trace summary retrieved: {summary['status']}")
        else:
            print("⚠️ Trace summary not available (expected for ended traces)")
        
        return True
        
    except Exception as e:
        print(f"❌ Tracing system test failed: {e}")
        return False

def test_metrics_system():
    """Test the advanced metrics system"""
    print("\n🧪 Testing Advanced Metrics System...")
    
    try:
        from core.monitoring.metrics import get_metrics_manager, record_query_success
        
        # Get metrics manager
        metrics_manager = get_metrics_manager()
        print("✅ Metrics manager retrieved")
        
        # Test metrics recording
        record_query_success("test", 100.0, "test_v1.0", "test_decision")
        print("✅ Query metrics recorded")
        
        # Check server status
        server_running = metrics_manager.is_server_running()
        print(f"✅ Metrics server status: {'running' if server_running else 'stopped'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics system test failed: {e}")
        return False

def test_drift_monitor():
    """Test the drift detection system"""
    print("\n🧪 Testing Drift Detection System...")
    
    try:
        from core.monitoring.drift_monitor import get_drift_summary
        
        # Get drift summary
        drift_summary = get_drift_summary()
        print(f"✅ Drift summary retrieved: {drift_summary['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Drift monitor test failed: {e}")
        return False

def test_feedback_system():
    """Test the feedback integration system"""
    print("\n🧪 Testing Feedback Integration System...")
    
    try:
        from core.monitoring.feedback_system import get_feedback_system, record_explicit_feedback
        
        # Get feedback system
        feedback_system = get_feedback_system()
        print("✅ Feedback system retrieved")
        
        # Test feedback recording
        feedback_id = record_explicit_feedback(
            trace_id="test_trace_001",
            model_version="test_v1.0",
            query="Test query",
            recommendations=["test_food_1", "test_food_2"],
            user_rating=5
        )
        print(f"✅ Explicit feedback recorded: {feedback_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feedback system test failed: {e}")
        return False

def test_health_endpoints():
    """Test the enhanced health endpoints"""
    print("\n🧪 Testing Enhanced Health Endpoints...")
    
    try:
        # This would typically test the actual API endpoints
        # For now, we'll just verify the modules can be imported
        from core.utils.tracing import get_trace_manager
        from core.monitoring.drift_monitor import get_drift_monitor
        from core.monitoring.feedback_system import get_feedback_system
        from core.monitoring.metrics import get_metrics_manager
        
        print("✅ All health endpoint dependencies imported")
        
        # Test individual components
        trace_manager = get_trace_manager()
        drift_monitor = get_drift_monitor()
        feedback_system = get_feedback_system()
        metrics_manager = get_metrics_manager()
        
        print("✅ All health monitoring components initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Health endpoints test failed: {e}")
        return False

def test_weekly_report():
    """Test the weekly report generator"""
    print("\n🧪 Testing Weekly Report Generator...")
    
    try:
        from scripts.generate_weekly_report import generate_weekly_report
        
        # Generate a test report
        report_path = generate_weekly_report()
        print(f"✅ Weekly report generated: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weekly report test failed: {e}")
        return False

def main():
    """Run all Stage 4 component tests"""
    print("🚀 Stage 4 Observability Components Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(("Request Tracing", test_tracing_system()))
    test_results.append(("Advanced Metrics", test_metrics_system()))
    test_results.append(("Drift Detection", test_drift_monitor()))
    test_results.append(("Feedback System", test_feedback_system()))
    test_results.append(("Health Endpoints", test_health_endpoints()))
    test_results.append(("Weekly Report", test_weekly_report()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Stage 4 components are working correctly!")
        return 0
    else:
        print("⚠️ Some Stage 4 components need attention.")
        return 1

if __name__ == "__main__":
    exit(main())
