#!/usr/bin/env python3
"""
Robust LLM-based validation for borderline training samples

This module provides a production-ready LLM validator with:
- Retry/backoff mechanisms
- Circuit breaker for rate limits
- Mock mode for testing
- Structured response objects
- Comprehensive error handling
"""

import json
import logging
import os
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import requests

from config.settings import (
    LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    LLM_VALIDATION_TIMEOUT, LLM_MAX_RETRIES,
    LLM_API_PROVIDER, LLM_API_KEY, LLM_RETRIES, LLM_BACKOFF_BASE,
    LLM_TIMEOUT, LLM_MOCK_MODE, LLM_CIRCUIT_BREAKER_THRESHOLD,
    LLM_CIRCUIT_BREAKER_TTL
)

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structured LLM API response"""
    success: bool
    raw_output: str
    http_status: int
    error: Optional[str]
    meta: Dict[str, Any]

class CircuitBreaker:
    """Simple circuit breaker for rate limit handling"""
    
    def __init__(self, threshold: int = 10, ttl: int = 900):
        self.threshold = threshold
        self.ttl = ttl
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
    
    def record_failure(self):
        """Record a failure and potentially open the circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.threshold:
            self.circuit_open = True
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def record_success(self):
        """Record a success and potentially close the circuit"""
        self.failure_count = 0
        if self.circuit_open:
            self.circuit_open = False
            logger.info("Circuit breaker closed after successful request")
    
    def is_open(self) -> bool:
        """Check if circuit is open"""
        if not self.circuit_open:
            return False
        
        # Check if TTL has expired
        if time.time() - self.last_failure_time > self.ttl:
            self.circuit_open = False
            self.failure_count = 0
            logger.info("Circuit breaker TTL expired, closing circuit")
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "circuit_open": self.circuit_open,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "threshold": self.threshold,
            "ttl": self.ttl
        }

class LLMValidator:
    """Robust LLM-based validator for borderline training samples"""
    
    def __init__(self):
        # Initialize dual API key system with fallback
        self.primary_api_key = LLM_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
        self.fallback_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.current_api_key = self.primary_api_key
        self.api_key_rotation_enabled = bool(self.primary_api_key and self.fallback_api_key)
        
        self.model = LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.timeout = LLM_TIMEOUT
        self.max_retries = LLM_RETRIES
        self.backoff_base = LLM_BACKOFF_BASE
        self.mock_mode = LLM_MOCK_MODE
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            threshold=LLM_CIRCUIT_BREAKER_THRESHOLD,
            ttl=LLM_CIRCUIT_BREAKER_TTL
        )
        
        # Mock mode setup
        if self.mock_mode:
            try:
                from core.filtering.llm_mock import get_mock_llm_validator
                self.mock_validator = get_mock_llm_validator()
                logger.info("LLM validator initialized in MOCK MODE")
            except ImportError:
                logger.warning("Mock mode requested but llm_mock module not available")
                self.mock_mode = False
    
    def _rotate_api_key(self):
        """Rotate to the fallback API key when the primary one fails."""
        if not self.api_key_rotation_enabled:
            return False
        
        if self.current_api_key == self.primary_api_key:
            self.current_api_key = self.fallback_api_key
            logger.info("Switched to fallback API key (DeepSeek)")
            return True
        else:
            # Already using fallback, switch back to primary
            self.current_api_key = self.primary_api_key
            logger.info("Switched back to primary API key (OpenRouter)")
            return True
        
        if not self.mock_mode:
            if not self.primary_api_key and not self.fallback_api_key:
                logger.warning("No LLM API keys found. LLM validation will be disabled.")
                self.enabled = False
            else:
                self.enabled = True
                if self.api_key_rotation_enabled:
                    logger.info(f"LLM validator initialized with dual API keys (primary: OpenRouter, fallback: DeepSeek) for model: {self.model}")
                else:
                    logger.info(f"LLM validator initialized with single API key for model: {self.model}")
        else:
            self.enabled = True
    
    def interpret_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Interpret a user query to extract intent and recommendations.
        
        Args:
            query: User's food query
            context: Optional user context
            
        Returns:
            LLMResponse with structured interpretation
        """
        if self.mock_mode:
            return self.mock_validator.interpret_query(query, context)
        
        if not self.enabled:
            return LLMResponse(
                success=False,
                raw_output="",
                http_status=0,
                error="LLM validation disabled",
                meta={"attempts": 0, "latency_ms": 0}
            )
        
        if self.circuit_breaker.is_open():
            return LLMResponse(
                success=False,
                raw_output="",
                http_status=429,
                error="Circuit breaker open - rate limit protection active",
                meta={"attempts": 0, "latency_ms": 0, "circuit_open": True}
            )
        
        try:
            start_time = time.time()
            
            # Create interpretation prompt
            prompt = self._create_interpretation_prompt(query, context)
            
            # Call LLM API with retry logic
            response = self._call_llm_api_with_retry(prompt)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if response.success:
                self.circuit_breaker.record_success()
            else:
                if response.http_status == 429:
                    self.circuit_breaker.record_failure()
            
            return response
            
        except Exception as e:
            logger.error(f"Query interpretation failed: {e}")
            return LLMResponse(
                success=False,
                raw_output="",
                http_status=0,
                error=str(e),
                meta={"attempts": 0, "latency_ms": 0}
            )
    
    def validate_prediction(self, ml_prediction: Dict[str, Any], query: str) -> LLMResponse:
        """
        Validate ML prediction using LLM semantic consistency check.
        
        Args:
            ml_prediction: ML model's prediction
            query: Original user query
            
        Returns:
            LLMResponse with validation result
        """
        if self.mock_mode:
            return self.mock_validator.validate_prediction(ml_prediction, query)
        
        if not self.enabled:
            return LLMResponse(
                success=False,
                raw_output="",
                http_status=0,
                error="LLM validation disabled",
                meta={"attempts": 0, "latency_ms": 0}
            )
        
        if self.circuit_breaker.is_open():
            return LLMResponse(
                success=False,
                raw_output="",
                http_status=429,
                error="Circuit breaker open - rate limit protection active",
                meta={"attempts": 0, "latency_ms": 0, "circuit_open": True}
            )
        
        try:
            start_time = time.time()
            
            # Create validation prompt
            prompt = self._create_validation_prompt(query, ml_prediction)
            
            # Call LLM API with retry logic
            response = self._call_llm_api_with_retry(prompt)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if response.success:
                self.circuit_breaker.record_success()
            else:
                if response.http_status == 429:
                    self.circuit_breaker.record_failure()
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction validation failed: {e}")
            return LLMResponse(
                success=False,
                raw_output="",
                http_status=0,
                error=str(e),
                meta={"attempts": 0, "latency_ms": 0}
            )
    
    def generate_recommendations(self, query: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Generate food recommendations directly from LLM.
        
        Args:
            query: User's food query
            context: Optional user context
            
        Returns:
            LLMResponse with food recommendations
        """
        if self.mock_mode:
            return self.mock_validator.generate_recommendations(query, context)
        
        if not self.enabled:
            return LLMResponse(
                success=False,
                raw_output="",
                http_status=0,
                error="LLM validation disabled",
                meta={"attempts": 0, "latency_ms": 0}
            )
        
        if self.circuit_breaker.is_open():
            return LLMResponse(
                success=False,
                raw_output="",
                http_status=429,
                error="Circuit breaker open - rate limit protection active",
                meta={"attempts": 0, "latency_ms": 0, "circuit_open": True}
            )
        
        try:
            start_time = time.time()
            
            # Create recommendation prompt
            prompt = self._create_recommendation_prompt(query, context)
            
            # Call LLM API with retry logic
            response = self._call_llm_api_with_retry(prompt)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if response.success:
                self.circuit_breaker.record_success()
            else:
                if response.http_status == 429:
                    self.circuit_breaker.record_failure()
            
            return response
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return LLMResponse(
                success=False,
                raw_output="",
                http_status=0,
                error=str(e),
                meta={"attempts": 0, "latency_ms": 0}
            )
    
    # Backward compatibility methods
    def validate_sample(self, query: str, label: str, confidence: float) -> bool:
        """Backward compatibility method for sample validation"""
        ml_prediction = {"labels": [label], "confidence": confidence}
        response = self.validate_prediction(ml_prediction, query)
        
        if response.success:
            # Parse the response to determine validation result
            return self._parse_validation_response(response.raw_output)
        else:
            logger.warning(f"LLM validation failed: {response.error}")
            return False
    
    def validate_batch(self, samples: List[Dict]) -> List[bool]:
        """Backward compatibility method for batch validation"""
        if not self.enabled and not self.mock_mode:
            logger.warning("LLM validation disabled - accepting all samples")
            return [True] * len(samples)
        
        results = []
        for i, sample in enumerate(samples):
            try:
                is_valid = self.validate_sample(
                    sample['query'], 
                    sample['label'], 
                    sample['confidence']
                )
                results.append(is_valid)
                
                # Add small delay to avoid rate limiting
                if i < len(samples) - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to validate sample {i}: {e}")
                results.append(False)
        
        return results
    
    def _create_interpretation_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Create prompt for query interpretation"""
        system_prompt = """You are a food recommendation AI assistant. Analyze the user's query and provide structured food recommendations.

Return a JSON response with:
- recommendations: list of 3-5 specific food items
- intent: the user's food intent (e.g., "japanese_cuisine", "comfort_food", "spicy_food")
- reasoning: brief explanation of your recommendations
- confidence: confidence score (0.0 to 1.0)"""

        user_prompt = f"""Query: "{query}"\n\nContext: {context or 'None'}\n\nProvide food recommendations in JSON format."""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _create_validation_prompt(self, query: str, ml_prediction: Dict[str, Any]) -> Dict[str, str]:
        """Create prompt for ML prediction validation"""
        system_prompt = """You are a semantic validator for a food mood AI system. Determine if the ML prediction is semantically consistent with the user query.

Return a JSON response with:
- validation: "yes", "no", or "maybe"
- reasoning: explanation of your decision
- confidence: confidence score (0.0 to 1.0)"""

        user_prompt = f"""Query: "{query}"
ML Prediction: {json.dumps(ml_prediction, indent=2)}

Is this ML prediction semantically correct for the query? Provide your assessment in JSON format."""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _create_recommendation_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Create prompt for direct recommendation generation"""
        system_prompt = """You are a food recommendation AI. Generate specific food recommendations based on the user's query.

Return a JSON response with:
- recommendations: list of 3-5 specific food items
- reasoning: brief explanation of your recommendations
- method: "llm_direct" """

        user_prompt = f"""Query: "{query}"\n\nContext: {context or 'None'}\n\nGenerate food recommendations in JSON format."""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _call_llm_api_with_retry(self, prompt: Dict[str, str]) -> LLMResponse:
        """Call LLM API with retry logic, exponential backoff, and API key rotation"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Track if we've tried both API keys
        tried_both_keys = False
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                headers = {
                    "Authorization": f"Bearer {self.current_api_key}",
                    "Content-Type": "application/json",
                }
                
                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
                
                response = requests.post(
                    url=url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                if response.status_code == 429:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}) with API key: {self.current_api_key[:8]}...")
                    
                    # Try API key rotation if we haven't tried both keys yet
                    if self.api_key_rotation_enabled and not tried_both_keys:
                        if self._rotate_api_key():
                            tried_both_keys = True
                            logger.info("Retrying with rotated API key...")
                            continue
                    
                    if attempt < self.max_retries - 1:
                        backoff_time = self.backoff_base * (2 ** attempt)
                        time.sleep(backoff_time)
                        continue
                    else:
                        return LLMResponse(
                            success=False,
                            raw_output="",
                            http_status=429,
                            error="Rate limit exceeded after all retries",
                            meta={"attempts": attempt + 1, "latency_ms": latency_ms, "api_key_rotated": tried_both_keys}
                        )
                
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                return LLMResponse(
                    success=True,
                    raw_output=content,
                    http_status=response.status_code,
                    error=None,
                    meta={"attempts": attempt + 1, "latency_ms": latency_ms, "api_key_used": self.current_api_key[:8] + "..."}
                )
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{self.max_retries}) with API key {self.current_api_key[:8]}...: {e}")
                
                # Try API key rotation if we haven't tried both keys yet
                if self.api_key_rotation_enabled and not tried_both_keys:
                    if self._rotate_api_key():
                        tried_both_keys = True
                        logger.info("Retrying with rotated API key...")
                        continue
                
                if attempt < self.max_retries - 1:
                    backoff_time = self.backoff_base * (2 ** attempt)
                    time.sleep(backoff_time)
                else:
                    return LLMResponse(
                        success=False,
                        raw_output="",
                        http_status=0,
                        error=str(e),
                        meta={"attempts": attempt + 1, "latency_ms": 0, "api_key_rotated": tried_both_keys}
                    )
        
        return LLMResponse(
            success=False,
            raw_output="",
            http_status=0,
            error="All retry attempts failed",
            meta={"attempts": self.max_retries, "latency_ms": 0, "api_key_rotated": tried_both_keys}
        )
    
    def _parse_validation_response(self, response: str) -> bool:
        """Parse validation response to determine result (backward compatibility)"""
        response = response.strip().lower()
        
        # Check for positive responses
        if response.startswith(('yes', 'y', 'true', 'correct', 'valid')):
            return True
        
        # Check for negative responses
        if response.startswith(('no', 'n', 'false', 'incorrect', 'invalid')):
            return False
        
        # If response is ambiguous, be conservative and reject
        logger.warning(f"Ambiguous LLM response: '{response}' - rejecting sample")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics including circuit breaker status"""
        stats = {
            "enabled": self.enabled,
            "mock_mode": self.mock_mode,
            "api_provider": LLM_API_PROVIDER,
            "model": self.model,
            "circuit_breaker": self.circuit_breaker.get_status()
        }
        
        if hasattr(self, 'mock_validator'):
            stats["mock_validator_available"] = True
        
        return stats
    
    def is_enabled(self) -> bool:
        """Check if the LLM validator is enabled and functional"""
        return (
            self.enabled and 
            not self.mock_mode and 
            not self.circuit_breaker.is_open()
        )
    
    def reset_circuit_breaker(self):
        """Reset the circuit breaker (useful for testing)"""
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.circuit_open = False
        self.circuit_breaker.last_failure_time = 0
        logger.info("Circuit breaker reset")

# Backward compatibility
def call_llm_api(prompt: str) -> str:
    """Backward compatibility function for simple string-based LLM calls"""
    validator = LLMValidator()
    if not validator.enabled:
        return ""
    
    # Convert string prompt to structured format
    structured_prompt = {
        "system": "You are a helpful AI assistant.",
        "user": prompt
    }
    
    response = validator._call_llm_api_with_retry(structured_prompt)
    return response.raw_output if response.success else ""
