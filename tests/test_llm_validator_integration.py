"""
Integration test for the LLM Validator with Mock Mode

This test verifies that the robust LLM validator works correctly
in mock mode for testing and development.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.filtering.llm_validator import LLMValidator
from core.filtering.llm_mock import get_mock_llm_validator

class TestLLMValidatorIntegration:
    """Integration tests for LLM Validator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Test with mock mode enabled
        os.environ["LLM_MOCK_MODE"] = "true"
        self.validator = LLMValidator()
    
    def teardown_method(self):
        """Clean up after tests"""
        if "LLM_MOCK_MODE" in os.environ:
            del os.environ["LLM_MOCK_MODE"]
    
    def test_mock_mode_initialization(self):
        """Test that mock mode initializes correctly"""
        assert self.validator.mock_mode == True
        assert self.validator.enabled == True
        assert hasattr(self.validator, 'mock_validator')
    
    def test_mock_query_interpretation(self):
        """Test mock query interpretation"""
        query = "I want some Japanese food"
        response = self.validator.interpret_query(query)
        
        assert response.success == True
        assert response.http_status == 200
        assert response.error is None
        assert "mock_mode" in response.meta
        assert response.meta["mock_mode"] == True
        
        # Check that raw output contains recommendations
        assert len(response.raw_output) > 0
    
    def test_mock_prediction_validation(self):
        """Test mock prediction validation"""
        ml_prediction = {
            "labels": ["japanese_cuisine"],
            "confidence": 0.8
        }
        query = "I want some Japanese food"
        
        response = self.validator.validate_prediction(ml_prediction, query)
        
        assert response.success == True
        assert response.http_status == 200
        assert response.error is None
        assert "mock_mode" in response.meta
    
    def test_mock_recommendation_generation(self):
        """Test mock recommendation generation"""
        query = "I want some spicy food"
        response = self.validator.generate_recommendations(query)
        
        assert response.success == True
        assert response.http_status == 200
        assert response.error is None
        assert "mock_mode" in response.meta
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality"""
        # Simulate rate limit failures
        for _ in range(5):  # Below threshold
            self.validator.circuit_breaker.record_failure()
        
        assert not self.validator.circuit_breaker.is_open()
        
        # Exceed threshold
        for _ in range(10):
            self.validator.circuit_breaker.record_failure()
        
        assert self.validator.circuit_breaker.is_open()
        
        # Reset circuit breaker
        self.validator.reset_circuit_breaker()
        assert not self.validator.circuit_breaker.is_open()
    
    def test_circuit_breaker_ttl(self):
        """Test circuit breaker TTL expiration"""
        # Open circuit
        for _ in range(10):
            self.validator.circuit_breaker.record_failure()
        
        assert self.validator.circuit_breaker.is_open()
        
        # Simulate TTL expiration by manually setting time
        import time
        self.validator.circuit_breaker.last_failure_time = time.time() - 1000  # 1000 seconds ago
        
        # Should automatically close
        assert not self.validator.circuit_breaker.is_open()
    
    def test_stats_retrieval(self):
        """Test statistics retrieval"""
        stats = self.validator.get_stats()
        
        assert "enabled" in stats
        assert "mock_mode" in stats
        assert "api_provider" in stats
        assert "model" in stats
        assert "circuit_breaker" in stats
        
        circuit_stats = stats["circuit_breaker"]
        assert "circuit_open" in circuit_stats
        assert "failure_count" in circuit_stats
        assert "threshold" in circuit_stats
        assert "ttl" in circuit_stats
    
    def test_backward_compatibility(self):
        """Test backward compatibility methods"""
        # Test validate_sample method
        is_valid = self.validator.validate_sample(
            query="I want Japanese food",
            label="japanese_cuisine",
            confidence=0.8
        )
        
        # Should work (though result may vary based on mock responses)
        assert isinstance(is_valid, bool)
        
        # Test validate_batch method
        samples = [
            {"query": "I want sushi", "label": "japanese", "confidence": 0.9},
            {"query": "I want pizza", "label": "italian", "confidence": 0.7}
        ]
        
        results = self.validator.validate_batch(samples)
        assert len(results) == 2
        assert all(isinstance(r, bool) for r in results)

def test_mock_validator_direct():
    """Test mock validator directly"""
    mock_validator = get_mock_llm_validator()
    
    # Test interpretation
    response = mock_validator.interpret_query("I want Japanese food")
    assert response.success == True
    assert response.http_status == 200
    
    # Test validation
    response = mock_validator.validate_prediction(
        {"labels": ["japanese"], "confidence": 0.8},
        "I want Japanese food"
    )
    assert response.success == True
    
    # Test recommendation generation
    response = mock_validator.generate_recommendations("I want spicy food")
    assert response.success == True
    
    # Test rate limit simulation
    rate_limit_response = mock_validator.simulate_rate_limit()
    assert rate_limit_response.success == False
    assert rate_limit_response.http_status == 429
    assert "Rate limit exceeded" in rate_limit_response.error

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
