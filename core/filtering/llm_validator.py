#!/usr/bin/env python3
"""
LLM-based validation for borderline training samples
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import requests

from config.settings import (
    LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    LLM_VALIDATION_TIMEOUT, LLM_MAX_RETRIES,
    OPENROUTER_API_KEY
)

logger = logging.getLogger(__name__)

class LLMValidator:
    """LLM-based validator for borderline training samples"""
    
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.model = LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.timeout = LLM_VALIDATION_TIMEOUT
        self.max_retries = LLM_MAX_RETRIES
        
        if not self.api_key:
            logger.warning("No OpenRouter API key found. LLM validation will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"LLM validator initialized with model: {self.model}")
    
    def validate_sample(self, query: str, label: str, confidence: float) -> bool:
        """
        Validate a borderline sample using LLM
        
        Args:
            query: User query text
            label: Predicted label
            confidence: Prediction confidence
            
        Returns:
            bool: True if LLM validates the sample, False otherwise
        """
        if not self.enabled:
            logger.warning("LLM validation disabled - accepting sample")
            return True
        
        try:
            # Prepare the prompt
            prompt = self._create_validation_prompt(query, label)
            
            # Call LLM API
            response = self._call_llm_api(prompt)
            
            # Parse response
            is_valid = self._parse_llm_response(response)
            
            logger.debug(f"LLM validation: query='{query[:50]}...' label='{label}' confidence={confidence:.3f} -> {is_valid}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"LLM validation failed for query '{query[:50]}...': {e}")
            # On error, be conservative and reject the sample
            return False
    
    def validate_batch(self, samples: List[Dict]) -> List[bool]:
        """
        Validate a batch of borderline samples
        
        Args:
            samples: List of sample dictionaries with 'query', 'label', 'confidence' keys
            
        Returns:
            List[bool]: Validation results for each sample
        """
        if not self.enabled:
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
    
    def _create_validation_prompt(self, query: str, label: str) -> str:
        """Create the validation prompt for the LLM"""
        system_prompt = """You are a semantic validator for a food mood AI system. Your job is to determine if a user query and its predicted label are semantically consistent.

Rules:
- Focus on semantic meaning, not exact word matches
- Consider context and intent
- Be conservative - if unsure, say no
- Answer only 'yes' or 'no'"""

        user_prompt = f"""Query: "{query}"
Label: "{label}"

Is this label semantically correct for the query? Answer only yes or no."""

        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _call_llm_api(self, prompt: Dict[str, str]) -> str:
        """Call the OpenRouter API"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url=url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip().lower()
                
                return content
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    raise
    
    def _parse_llm_response(self, response: str) -> bool:
        """Parse the LLM response to determine validation result"""
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
    
    def get_stats(self) -> Dict[str, any]:
        """Get validation statistics"""
        return {
            "enabled": self.enabled,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
