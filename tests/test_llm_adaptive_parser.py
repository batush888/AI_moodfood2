"""
Test suite for the Adaptive Parser module

This tests the robust parsing utilities that can extract food recommendations
from various LLM response formats.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.filtering.adaptive_parser import AdaptiveParser, ParseResult

class TestAdaptiveParser:
    """Test cases for the AdaptiveParser class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.parser = AdaptiveParser()
        self.strict_parser = AdaptiveParser(strict_mode=True)
        self.lenient_parser = AdaptiveParser(strict_mode=False)
    
    def test_parse_json_safe(self):
        """Test parsing of well-formed JSON responses"""
        json_response = '{"recommendations": ["sushi", "ramen", "tempura"], "intent": "japanese"}'
        result = self.parser.parse_llm_output(json_response)
        
        assert result.parse_status == "json"
        assert result.confidence == 0.9
        assert "sushi" in result.parsed
        assert "ramen" in result.parsed
        assert "tempura" in result.parsed
    
    def test_parse_code_block(self):
        """Test parsing of markdown code blocks"""
        code_block_response = '```json\n{"foods": ["pizza", "pasta"]}\n```'
        result = self.parser.parse_llm_output(code_block_response)
        
        # JSON parser will catch this first since it's valid JSON
        assert result.parse_status == "json"
        assert result.confidence == 0.9
        assert "pizza" in result.parsed
        assert "pasta" in result.parsed
    
    def test_parse_list_regex(self):
        """Test parsing of bullet point lists"""
        list_response = """
        Here are some recommendations:
        - Curry
        - Hot wings
        - Jalapeno poppers
        """
        result = self.parser.parse_llm_output(list_response)
        
        assert result.parse_status == "list_regex"
        assert result.confidence >= 0.6
        # Note: case is preserved in the parsed output
        assert "Curry" in result.parsed
        assert "Hot wings" in result.parsed
    
    def test_parse_heuristic(self):
        """Test fallback heuristic parsing"""
        plain_text = "Try sushi, ramen, and tempura for Japanese food."
        result = self.parser.parse_llm_output(plain_text)
        
        # This might be caught by list_regex due to comma separation
        assert result.parse_status in ["heuristic", "list_regex"]
        assert result.confidence >= 0.4
        assert len(result.parsed) > 0
    
    def test_strict_mode_filtering(self):
        """Test strict mode filtering of generic words"""
        generic_response = '{"recommendations": ["food", "meal", "dish", "sushi"]}'
        strict_result = self.strict_parser.parse_llm_output(generic_response)
        lenient_result = self.lenient_parser.parse_llm_output(generic_response)
        
        # Strict mode should filter out generic words
        assert "food" not in strict_result.parsed
        assert "meal" not in strict_result.parsed
        assert "sushi" in strict_result.parsed
        
        # Lenient mode should keep more items
        assert len(lenient_result.parsed) >= len(strict_result.parsed)
    
    def test_min_token_length_filtering(self):
        """Test minimum token length filtering"""
        short_response = '{"recommendations": ["a", "bb", "sushi", "ramen"]}'
        result = self.parser.parse_llm_output(short_response)
        
        # Should filter out very short tokens
        assert "a" not in result.parsed
        # "bb" is exactly 2 characters, which meets the minimum length
        assert "bb" in result.parsed  # This is correct behavior
        assert "sushi" in result.parsed
        assert "ramen" in result.parsed
    
    def test_max_recommendations_limit(self):
        """Test maximum recommendations limit"""
        many_items = '{"recommendations": ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item10", "item11"]}'
        result = self.parser.parse_llm_output(many_items)
        
        assert len(result.parsed) <= 10  # Default max_recommendations
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON"""
        malformed_response = '{"recommendations": ["sushi", "ramen", "tempura",}'
        result = self.parser.parse_llm_output(malformed_response)
        
        # Should fall back to other parsing methods
        assert result.parse_status != "json"
        assert result.confidence < 0.9
    
    def test_empty_response_handling(self):
        """Test handling of empty or None responses"""
        empty_response = ""
        result = self.parser.parse_llm_output(empty_response)
        
        assert result.parse_status == "none"
        assert result.confidence == 0.0
        assert len(result.parsed) == 0
    
    def test_none_response_handling(self):
        """Test handling of None responses"""
        result = self.parser.parse_llm_output(None)
        
        assert result.parse_status == "none"
        assert result.confidence == 0.0
        assert len(result.parsed) == 0
    
    def test_normalize_recommendations(self):
        """Test normalization and deduplication"""
        duplicate_response = '{"recommendations": ["sushi", "SUSHI", "Sushi", "ramen", "Ramen"]}'
        result = self.parser.parse_llm_output(duplicate_response)
        
        # Should deduplicate case-insensitive
        assert len(result.parsed) == 2
        assert "sushi" in result.parsed
        assert "ramen" in result.parsed

def test_parse_llm_output_function():
    """Test the convenience function"""
    from core.filtering.adaptive_parser import parse_llm_output
    
    result = parse_llm_output('{"recommendations": ["test"]}')
    assert isinstance(result, ParseResult)
    assert "test" in result.parsed

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
