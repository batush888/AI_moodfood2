"""
Adaptive Parser for LLM Responses

This module provides robust parsing utilities to extract food recommendations
from various LLM response formats, including JSON, markdown, lists, and plain text.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParseResult:
    """Result of parsing LLM output"""
    parsed: List[str]
    parse_status: str  # "json", "code_block", "list_regex", "heuristic", "none"
    raw_output: str
    confidence: float  # 0.0 to 1.0 based on parse quality

class AdaptiveParser:
    """Robust parser that can extract food recommendations from various LLM response formats."""
    
    def __init__(self, strict_mode: bool = True, min_token_length: int = 2, max_recommendations: int = 10):
        self.strict_mode = strict_mode
        self.min_token_length = min_token_length
        self.max_recommendations = max_recommendations
        
        # Common food-related words to filter out in strict mode
        self.generic_words = {
            'food', 'meal', 'dish', 'cuisine', 'restaurant', 'place', 'option',
            'choice', 'thing', 'item', 'stuff', 'good', 'nice', 'delicious'
        }
        
        # List patterns for regex matching
        self.list_patterns = [
            r'[-*•]\s*([^,\n]+)',  # Bullet points
            r'\d+\.\s*([^,\n]+)',  # Numbered lists
            r'([^,\n]+)(?:,|and)\s*([^,\n]+)',  # Comma/and separated
        ]
    
    def parse_llm_output(self, raw_text: str) -> ParseResult:
        """Main parsing orchestrator that tries multiple parsing strategies."""
        if not raw_text or not raw_text.strip():
            return ParseResult(parsed=[], parse_status="none", raw_output=raw_text or "", confidence=0.0)
        
        # Try JSON parsing first (highest confidence)
        json_result = self._parse_json_safe(raw_text)
        if json_result and json_result.parsed:
            return json_result
        
        # Try code block extraction
        code_result = self._extract_code_block(raw_text)
        if code_result and code_result.parsed:
            return code_result
        
        # Try list regex matching
        list_result = self._extract_list_items(raw_text)
        if list_result and list_result.parsed:
            return list_result
        
        # Try heuristic text analysis
        heuristic_result = self._heuristic_from_text(raw_text)
        if heuristic_result and heuristic_result.parsed:
            return heuristic_result
        
        # Fallback to none
        return ParseResult(parsed=[], parse_status="none", raw_output=raw_text, confidence=0.0)
    
    def _parse_json_safe(self, raw_text: str) -> Optional[ParseResult]:
        """Try to parse JSON response safely"""
        try:
            cleaned_text = raw_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            parsed = json.loads(cleaned_text)
            recommendations = self._extract_from_json(parsed)
            
            if recommendations:
                normalized = self._normalize_recommendations(recommendations)
                return ParseResult(
                    parsed=normalized,
                    parse_status="json",
                    raw_output=raw_text,
                    confidence=0.9
                )
        
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.debug(f"JSON parsing failed: {e}")
        
        return None
    
    def _extract_from_json(self, parsed_json: Any) -> List[str]:
        """Extract recommendations from parsed JSON structure"""
        recommendations = []
        
        if isinstance(parsed_json, dict):
            for key in ['recommendations', 'foods', 'dishes', 'items', 'suggestions']:
                if key in parsed_json and isinstance(parsed_json[key], list):
                    recommendations.extend(parsed_json[key])
            
            for value in parsed_json.values():
                if isinstance(value, str) and self._is_valid_food_item(value):
                    recommendations.append(value)
        
        elif isinstance(parsed_json, list):
            recommendations = parsed_json
        
        return recommendations
    
    def _extract_code_block(self, raw_text: str) -> Optional[ParseResult]:
        """Extract and parse content from markdown code blocks"""
        try:
            code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            matches = re.findall(code_block_pattern, raw_text, re.DOTALL | re.IGNORECASE)
            
            if matches:
                code_content = matches[0].strip()
                
                try:
                    parsed = json.loads(code_content)
                    recommendations = self._extract_from_json(parsed)
                    if recommendations:
                        normalized = self._normalize_recommendations(recommendations)
                        return ParseResult(
                            parsed=normalized,
                            parse_status="code_block",
                            raw_output=raw_text,
                            confidence=0.8
                        )
                except json.JSONDecodeError:
                    lines = [line.strip() for line in code_content.split('\n') if line.strip()]
                    if lines:
                        normalized = self._normalize_recommendations(lines)
                        if normalized:
                            return ParseResult(
                                parsed=normalized,
                                parse_status="code_block",
                                raw_output=raw_text,
                                confidence=0.7
                            )
        
        except Exception as e:
            logger.debug(f"Code block extraction failed: {e}")
        
        return None
    
    def _extract_list_items(self, raw_text: str) -> Optional[ParseResult]:
        """Extract recommendations using regex patterns for lists"""
        try:
            all_items = []
            
            for pattern in self.list_patterns:
                matches = re.findall(pattern, raw_text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            all_items.extend([item.strip() for item in match if item.strip()])
                        else:
                            all_items.append(match.strip())
            
            if all_items:
                normalized = self._normalize_recommendations(all_items)
                if normalized:
                    return ParseResult(
                        parsed=normalized,
                        parse_status="list_regex",
                        raw_output=raw_text,
                        confidence=0.6
                    )
        
        except Exception as e:
            logger.debug(f"List regex extraction failed: {e}")
        
        return None
    
    def _heuristic_from_text(self, raw_text: str) -> Optional[ParseResult]:
        """Fallback heuristic parsing for plain text"""
        try:
            delimiters = [',', ';', 'and', 'or', '\n', '.']
            text = raw_text.lower()
            
            for delimiter in delimiters:
                if delimiter in text:
                    parts = text.split(delimiter)
                    break
            else:
                parts = [text]
            
            food_items = []
            for part in parts:
                part = part.strip()
                if part and self._is_valid_food_item(part):
                    food_items.append(part)
            
            if food_items:
                normalized = self._normalize_recommendations(food_items)
                if normalized:
                    return ParseResult(
                        parsed=normalized,
                        parse_status="heuristic",
                        raw_output=raw_text,
                        confidence=0.4
                    )
        
        except Exception as e:
            logger.debug(f"Heuristic parsing failed: {e}")
        
        return None
    
    def _is_valid_food_item(self, item: str) -> bool:
        """Check if a string looks like a valid food item"""
        if not item or len(item.strip()) < self.min_token_length:
            return False
        
        if self.strict_mode:
            item_lower = item.lower().strip()
            if item_lower in self.generic_words:
                return False
        
        words = item.split()
        if len(words) < 1 or len(words) > 4:
            return False
        
        return True
    
    def _normalize_recommendations(self, recommendations: List[str]) -> List[str]:
        """Normalize and filter recommendations"""
        if not recommendations:
            return []
        
        normalized = []
        seen = set()
        
        for item in recommendations:
            if not item:
                continue
            
            cleaned = item.strip()
            if not cleaned:
                continue
            
            if self.strict_mode and not self._is_valid_food_item(cleaned):
                continue
            
            normalized_item = cleaned.lower()
            if normalized_item not in seen:
                seen.add(normalized_item)
                normalized.append(cleaned)
        
        if len(normalized) > self.max_recommendations:
            normalized = normalized[:self.max_recommendations]
        
        return normalized

def parse_llm_output(raw_text: str, strict_mode: bool = True) -> ParseResult:
    """Convenience function for parsing LLM output."""
    parser = AdaptiveParser(strict_mode=strict_mode)
    return parser.parse_llm_output(raw_text)
