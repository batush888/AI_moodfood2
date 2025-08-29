import json
import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any, Union
import yaml
import os
import re

# Setup logging first
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ .env file loaded successfully in LLM Parser")
except ImportError:
    logger.warning("⚠️ python-dotenv not installed, using system environment variables")
except Exception as e:
    logger.warning(f"⚠️ Error loading .env file: {e}")

class RobustLLMParser:
    """Enhanced LLM parser with robust JSON parsing and fallback mechanisms."""
    
    def __init__(self):
        self.parsing_attempts = 0
        self.max_parsing_attempts = 3
        
    def parse_llm_response(self, response_text: str, expected_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Robustly parse LLM response with multiple fallback strategies.
        
        Args:
            response_text: Raw text response from LLM
            expected_schema: Expected JSON schema for validation
            
        Returns:
            Parsed and validated response data
        """
        self.parsing_attempts = 0
        
        # Strategy 1: Direct JSON parsing
        try:
            parsed = json.loads(response_text.strip())
            if self._validate_schema(parsed, expected_schema):
                logger.info("✅ Direct JSON parsing successful")
                return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
        
        # Strategy 2: Extract JSON from markdown or code blocks
        extracted_json = self._extract_json_from_text(response_text)
        if extracted_json:
            try:
                parsed = json.loads(extracted_json)
                if self._validate_schema(parsed, expected_schema):
                    logger.info("✅ JSON extraction from text successful")
                    return parsed
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Extracted JSON parsing failed: {e}")
        
        # Strategy 3: Structured text extraction
        extracted_data = self._extract_structured_data(response_text, expected_schema)
        if extracted_data:
            logger.info("✅ Structured text extraction successful")
            return extracted_data
        
        # Strategy 4: Smart fallback parsing
        fallback_data = self._smart_fallback_parsing(response_text, expected_schema)
        if fallback_data:
            logger.info("✅ Smart fallback parsing successful")
            return fallback_data
        
        # Strategy 5: Generate structured fallback
        logger.warning("⚠️ All parsing strategies failed, generating structured fallback")
        return self._generate_structured_fallback(response_text, expected_schema)
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from markdown code blocks or quoted text."""
        # Look for JSON in code blocks
        code_block_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'`([^`]+)`',
            r'"([^"]*)"'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Clean up the extracted text
                cleaned = match.strip()
                if cleaned.startswith('{') and cleaned.endswith('}'):
                    return cleaned
                elif cleaned.startswith('[') and cleaned.endswith(']'):
                    return cleaned
        
        # Look for JSON-like structures
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text)
        if matches:
            return matches[0]
        
        return None
    
    def _extract_structured_data(self, text: str, expected_schema: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Extract structured data using regex patterns and NLP techniques."""
        if not expected_schema:
            return None
            
        extracted_data = {}
        
        # Extract intent patterns
        intent_patterns = [
            r'intent["\s:]*([^"\n,}]+)',
            r'goal["\s:]*([^"\n,}]+)',
            r'purpose["\s:]*([^"\n,}]+)'
        ]
        
        for pattern in intent_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted_data['intent'] = match.group(1).strip()
                break
        
        # Extract food recommendations
        food_patterns = [
            r'food["\s:]*\[([^\]]+)\]',
            r'recommendations["\s:]*\[([^\]]+)\]',
            r'suggestions["\s:]*\[([^\]]+)\]'
        ]
        
        for pattern in food_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                food_list = [f.strip() for f in match.group(1).split(',')]
                extracted_data['foods'] = food_list
                break
        
        # Extract scores and confidence
        score_patterns = [
            r'confidence["\s:]*([0-9.]+)',
            r'score["\s:]*([0-9.]+)',
            r'probability["\s:]*([0-9.]+)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted_data['confidence'] = float(match.group(1))
                break
        
        return extracted_data if extracted_data else None
    
    def _smart_fallback_parsing(self, text: str, expected_schema: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Smart fallback parsing using context clues and pattern matching."""
        fallback_data = {}
        
        # Extract any key-value pairs
        kv_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*([^,\n}]+)'
        matches = re.findall(kv_pattern, text)
        
        for key, value in matches:
            key = key.strip().lower()
            value = value.strip().strip('"\'')
            
            # Map common keys to expected schema
            if 'intent' in key or 'goal' in key:
                fallback_data['intent'] = value
            elif 'food' in key or 'dish' in key or 'meal' in key:
                if 'foods' not in fallback_data:
                    fallback_data['foods'] = []
                fallback_data['foods'].append(value)
            elif 'confidence' in key or 'score' in key:
                try:
                    fallback_data['confidence'] = float(value)
                except ValueError:
                    pass
        
        return fallback_data if fallback_data else None
    
    def _generate_structured_fallback(self, text: str, expected_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a structured fallback response when all parsing fails."""
        fallback = {
            'intent': 'unknown',
            'foods': [],
            'confidence': 0.0,
            'parsing_status': 'fallback_generated',
            'original_text': text[:200] + '...' if len(text) > 200 else text
        }
        
        # Try to extract any food-related terms
        food_keywords = ['pizza', 'sushi', 'ramen', 'burger', 'pasta', 'salad', 'soup', 'steak']
        found_foods = []
        
        for keyword in food_keywords:
            if keyword.lower() in text.lower():
                found_foods.append(keyword)
        
        if found_foods:
            fallback['foods'] = found_foods[:3]  # Limit to 3 items
            fallback['confidence'] = 0.3  # Low confidence for fallback
        
        return fallback
    
    def _validate_schema(self, data: Dict[str, Any], expected_schema: Dict[str, Any] = None) -> bool:
        """Validate parsed data against expected schema."""
        if not expected_schema:
            return True  # No validation required
        
        # Basic validation - check required fields
        required_fields = expected_schema.get('required', [])
        for field in required_fields:
            if field not in data:
                return False
        
        return True
    
    def self_correct_prompt(self, failed_response: str, expected_format: str) -> str:
        """Generate a self-correction prompt for the LLM."""
        return f"""
Your previous response was not in the correct format. Please respond ONLY in this exact JSON format:

{expected_format}

Do not include any explanations, markdown, or additional text. Only the JSON response.
"""

class LLMParser:
    """Main LLM parser class with enhanced robustness."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.robust_parser = RobustLLMParser()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Default configuration
        return {
            'api_key': os.getenv('OPENROUTER_API_KEY', ''),
            'model': 'deepseek/deepseek-r1-0528:free',
            'max_tokens': 256,
            'temperature': 0.0,
            'timeout': 30
        }
    
    def parse_response(self, response_text: str, expected_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse LLM response using robust parsing strategies."""
        return self.robust_parser.parse_llm_response(response_text, expected_schema)
    
    def get_self_correction_prompt(self, failed_response: str, expected_format: str) -> str:
        """Get a prompt to ask the LLM to self-correct its response."""
        return self.robust_parser.self_correct_prompt(failed_response, expected_format)
