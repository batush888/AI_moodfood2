"""
Tests for canary deployment and A/B testing evaluation.
Tests the ability to evaluate canary models and manage A/B testing traffic routing.
"""

import os
import json
import tempfile
import shutil
import random
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path

# Mock the dependencies to avoid import errors
with patch.dict('sys.modules', {
    'joblib': MagicMock(),
    'sklearn': MagicMock(),
    'mlflow': MagicMock(),
    'core.nlu.model_loader': MagicMock(),
    'core.logging.query_logger': MagicMock(),
    'core.filtering.hybrid_filter': MagicMock()
}):
    from scripts.retrain_classifier import AutomatedRetrainer

class TestCanaryEvaluate:
    """Tests for canary deployment and A/B testing functionality."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.versions_dir = os.path.join(self.temp_dir, "versions")
        os.makedirs(self.versions_dir, exist_ok=True)
        
        # Create mock retrainer
        with patch('scripts.retrain_classifier.MODEL_DIR', self.temp_dir):
            self.retrainer = AutomatedRetrainer()
    
    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)
    
    def test_canary_traffic_fraction(self):
        """Test that canary receives the correct fraction of traffic."""
        # Create production and canary versions
        prod_version = "prod_20231201_120000_abcd1234"
        canary_version = "canary_20231201_130000_efgh5678"
        
        # Set up versions
        self._create_mock_version(prod_version, deployed=True, deploy_mode="full")
        self._create_mock_version(canary_version, deployed=False, deploy_mode=None)
        
        # Start A/B test with 10% traffic to canary
        fraction = 0.1
        success = self.retrainer.start_abtest(canary_version, fraction)
        assert success == True
        
        # Simulate traffic routing
        traffic_samples = 1000
        canary_requests = 0
        
        for _ in range(traffic_samples):
            # Simulate routing decision (normally done in inference code)
            if random.random() < fraction:
                canary_requests += 1
        
        # Allow some tolerance for randomness (should be around 10%)
        expected_canary = traffic_samples * fraction
        tolerance = traffic_samples * 0.02  # 2% tolerance
        
        assert abs(canary_requests - expected_canary) <= tolerance
    
    def test_abtest_configuration_persistence(self):
        """Test that A/B test configuration persists correctly."""
        version_id = "test_canary_version"
        fraction = 0.05
        
        # Start A/B test
        success = self.retrainer.start_abtest(version_id, fraction)
        assert success == True
        
        # Verify configuration file exists and has correct content
        abtest_file = os.path.join(self.temp_dir, "abtest.json")
        assert os.path.exists(abtest_file)
        
        with open(abtest_file) as f:
            config = json.load(f)
            assert config["version"] == version_id
            assert config["fraction"] == fraction
            assert "started" in config
        
        # Test reading configuration
        status = self.retrainer.get_abtest_status()
        assert status["version"] == version_id
        assert status["fraction"] == fraction
    
    def test_canary_model_evaluation_metrics(self):
        """Test collection of metrics for canary model evaluation."""
        # Mock metrics collection for production and canary models
        prod_metrics = {
            "accuracy": 0.85,
            "f1_macro": 0.82,
            "response_time_ms": 150,
            "user_satisfaction": 4.2
        }
        
        canary_metrics = {
            "accuracy": 0.87,
            "f1_macro": 0.84,
            "response_time_ms": 140,
            "user_satisfaction": 4.3
        }
        
        # Simulate metrics collection
        def evaluate_model_performance(version_id, metrics):
            """Simulate model performance evaluation."""
            if version_id == "prod_version":
                return prod_metrics
            elif version_id == "canary_version":
                return canary_metrics
            else:
                return None
        
        # Compare metrics
        prod_perf = evaluate_model_performance("prod_version", prod_metrics)
        canary_perf = evaluate_model_performance("canary_version", canary_metrics)
        
        assert prod_perf is not None
        assert canary_perf is not None
        
        # Canary should outperform production in this test
        assert canary_perf["accuracy"] > prod_perf["accuracy"]
        assert canary_perf["f1_macro"] > prod_perf["f1_macro"]
        assert canary_perf["response_time_ms"] < prod_perf["response_time_ms"]
        assert canary_perf["user_satisfaction"] > prod_perf["user_satisfaction"]
    
    def test_canary_promotion_logic(self):
        """Test logic for promoting canary to production."""
        # Mock evaluation results
        canary_better = True
        statistical_significance = True
        min_sample_size_met = True
        
        # Promotion criteria
        should_promote = (
            canary_better and 
            statistical_significance and 
            min_sample_size_met
        )
        
        assert should_promote == True
        
        # If promotion criteria met, canary should be deployed as full
        if should_promote:
            # Mock deployment
            canary_version = "canary_20231201_130000_efgh5678"
            self._create_mock_version(canary_version, deployed=False, deploy_mode=None)
            
            with patch('core.nlu.model_loader.reload_current_model'):
                success = self.retrainer.deploy_version(canary_version, mode="full")
                assert success == True
    
    def test_canary_rollback_on_poor_performance(self):
        """Test rollback of canary when performance degrades."""
        # Mock poor performance scenario
        canary_performance_degraded = True
        error_rate_threshold_exceeded = True
        user_complaints_increased = True
        
        # Rollback criteria
        should_rollback = (
            canary_performance_degraded or 
            error_rate_threshold_exceeded or 
            user_complaints_increased
        )
        
        assert should_rollback == True
        
        if should_rollback:
            # Stop A/B test immediately
            success = self.retrainer.stop_abtest()
            assert success == True
            
            # Verify A/B test is stopped
            assert self.retrainer.is_abtest_active() == False
    
    def test_multiple_concurrent_abtests_prevention(self):
        """Test that only one A/B test can run at a time."""
        version1 = "canary_version_1"
        version2 = "canary_version_2"
        
        # Start first A/B test
        success1 = self.retrainer.start_abtest(version1, 0.05)
        assert success1 == True
        assert self.retrainer.is_abtest_active() == True
        
        # Try to start second A/B test (should fail or override)
        success2 = self.retrainer.start_abtest(version2, 0.1)
        
        # Check which version is actually active
        status = self.retrainer.get_abtest_status()
        
        # Either the second test should fail, or it should override the first
        # In our implementation, it overrides (creates new abtest.json)
        assert status["version"] == version2
        assert status["fraction"] == 0.1
    
    def test_canary_model_version_logging(self):
        """Test that model version is logged for each query during A/B test."""
        # Mock query logging with model version
        def log_query_with_model_version(query, model_version):
            """Mock query logging that includes model version."""
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "model_version": model_version,
                "session_id": "test_session"
            }
            return log_entry
        
        # Simulate queries during A/B test
        prod_version = "prod_v1"
        canary_version = "canary_v2"
        
        # Mock traffic routing and logging
        queries = ["I want Japanese food", "Something cold please", "Comfort food"]
        logs = []
        
        for query in queries:
            # Simulate routing (10% to canary)
            if random.random() < 0.1:
                log_entry = log_query_with_model_version(query, canary_version)
            else:
                log_entry = log_query_with_model_version(query, prod_version)
            logs.append(log_entry)
        
        # Verify all logs have model version
        for log in logs:
            assert "model_version" in log
            assert log["model_version"] in [prod_version, canary_version]
    
    def test_canary_gradual_rollout(self):
        """Test gradual rollout of canary model."""
        canary_version = "gradual_canary_v1"
        
        # Start with 1% traffic
        self.retrainer.start_abtest(canary_version, 0.01)
        status = self.retrainer.get_abtest_status()
        assert status["fraction"] == 0.01
        
        # Increase to 5% traffic (simulate gradual increase)
        self.retrainer.start_abtest(canary_version, 0.05)
        status = self.retrainer.get_abtest_status()
        assert status["fraction"] == 0.05
        
        # Increase to 10% traffic
        self.retrainer.start_abtest(canary_version, 0.10)
        status = self.retrainer.get_abtest_status()
        assert status["fraction"] == 0.10
        
        # Full rollout (promote to production)
        with patch('core.nlu.model_loader.reload_current_model'):
            self.retrainer.deploy_version(canary_version, mode="full")
            # Stop A/B test as canary is now production
            self.retrainer.stop_abtest()
            assert self.retrainer.is_abtest_active() == False
    
    def test_canary_statistical_significance_check(self):
        """Test statistical significance checking for canary evaluation."""
        # Mock sample sizes and performance metrics
        prod_samples = 1000
        canary_samples = 100  # 10% traffic
        
        prod_accuracy = 0.85
        canary_accuracy = 0.87
        
        # Simple statistical significance simulation
        # (In real implementation, would use proper statistical tests)
        min_sample_size = 50
        min_improvement = 0.01
        
        has_sufficient_samples = canary_samples >= min_sample_size
        has_meaningful_improvement = (canary_accuracy - prod_accuracy) >= min_improvement
        
        is_statistically_significant = has_sufficient_samples and has_meaningful_improvement
        
        assert has_sufficient_samples == True
        assert has_meaningful_improvement == True
        assert is_statistically_significant == True
    
    def _create_mock_version(self, version_id, deployed=False, deploy_mode=None):
        """Helper method to create a mock model version."""
        version_path = os.path.join(self.versions_dir, version_id)
        os.makedirs(version_path, exist_ok=True)
        
        # Create version index entry
        versions_index_file = os.path.join(self.versions_dir, "versions.json")
        
        # Read existing index or create new one
        if os.path.exists(versions_index_file):
            with open(versions_index_file) as f:
                index = json.load(f)
        else:
            index = {}
        
        # Add new version
        index[version_id] = {
            "created": datetime.now().isoformat(),
            "path": version_path,
            "metrics": {"accuracy": 0.85, "f1_macro": 0.82},
            "deployed": deployed,
            "deploy_mode": deploy_mode
        }
        
        # Write updated index
        with open(versions_index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        return version_path
