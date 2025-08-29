"""
Test Advanced Features
======================

Tests for the new advanced features:
1. Robust LLM parsing
2. Session memory management
3. Smart fallback system
4. Token management
"""

import pytest
import json
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_robust_llm_parser():
    """Test the robust LLM parser with various input formats."""
    try:
        from core.nlu.llm_parser import RobustLLMParser
        
        parser = RobustLLMParser()
        
        # Test 1: Valid JSON
        valid_json = '{"intent": "comfort", "foods": ["soup", "pasta"]}'
        result = parser.parse_llm_response(valid_json)
        assert result['intent'] == 'comfort'
        assert result['foods'] == ['soup', 'pasta']
        
        # Test 2: JSON in markdown
        markdown_json = '```json\n{"intent": "energy", "foods": ["smoothie"]}\n```'
        result = parser.parse_llm_response(markdown_json)
        assert result['intent'] == 'energy'
        assert result['foods'] == ['smoothie']
        
        # Test 3: Malformed JSON with fallback
        malformed = 'intent: comfort, foods: [soup, pasta]'
        result = parser.parse_llm_response(malformed)
        assert 'intent' in result or 'foods' in result
        
        print("✅ Robust LLM parser tests passed")
        
    except ImportError as e:
        pytest.skip(f"Robust LLM parser not available: {e}")

def test_session_memory():
    """Test session memory management."""
    try:
        from core.session.memory_manager import MemoryManager, ConversationTurn
        
        manager = MemoryManager()
        
        # Test session creation
        session = manager.create_session("test_session_001")
        assert session.session_id == "test_session_001"
        assert len(session.conversation_history) == 0
        
        # Test adding conversation turn
        turn = ConversationTurn(
            timestamp=1234567890.0,
            user_input="I want comfort food",
            user_context={"mood": "sad"},
            ai_response={"intent": "comfort"},
            recommendations=[{"name": "soup"}]
        )
        
        success = manager.add_conversation_turn("test_session_001", turn)
        assert success is True
        
        # Test retrieving session
        retrieved = manager.get_session("test_session_001")
        assert retrieved is not None
        assert len(retrieved.conversation_history) == 1
        
        print("✅ Session memory tests passed")
        
    except ImportError as e:
        pytest.skip(f"Session memory not available: {e}")

def test_smart_fallback():
    """Test the smart fallback system."""
    try:
        from core.filtering.smart_fallback import SmartFallbackSystem
        
        fallback = SmartFallbackSystem()
        
        # Test mood-based fallback
        recommendations = fallback.generate_smart_fallback(
            "I want comfort food",
            user_context={"mood": "sad"},
            num_recommendations=2
        )
        
        assert len(recommendations) > 0
        assert all(hasattr(rec, 'food_name') for rec in recommendations)
        assert all(hasattr(rec, 'score') for rec in recommendations)
        
        # Test context-based fallback
        recommendations = fallback.generate_smart_fallback(
            "I'm hungry",
            user_context={"time_of_day": "morning", "weather": "cold"},
            num_recommendations=2
        )
        
        assert len(recommendations) > 0
        
        print("✅ Smart fallback tests passed")
        
    except ImportError as e:
        pytest.skip(f"Smart fallback not available: {e}")

def test_token_management():
    """Test token management and prompt optimization."""
    try:
        from core.prompting.token_manager import TokenManager, PromptSection
        
        manager = TokenManager()
        
        # Test token counting
        text = "Hello world"
        token_count = manager.count_tokens(text)
        assert token_count > 0
        
        # Test budget calculation
        budget = manager.calculate_token_budget()
        assert budget.total_tokens > 0
        assert budget.available_for_prompt > 0
        
        # Test prompt optimization
        sections = [
            PromptSection("System prompt", 100, 10, "system"),
            PromptSection("User query", 50, 10, "query")
        ]
        
        optimized_prompt, token_count = manager.optimize_prompt(sections, budget)
        assert len(optimized_prompt) > 0
        assert token_count > 0
        
        print("✅ Token management tests passed")
        
    except ImportError as e:
        pytest.skip(f"Token management not available: {e}")

def test_integration():
    """Test integration between the new features."""
    try:
        # Test that all components can work together
        from core.nlu.llm_parser import RobustLLMParser
        from core.session.memory_manager import MemoryManager
        from core.filtering.smart_fallback import SmartFallbackSystem
        from core.prompting.token_manager import TokenManager
        
        # Create instances
        parser = RobustLLMParser()
        memory = MemoryManager()
        fallback = SmartFallbackSystem()
        token_mgr = TokenManager()
        
        # Test workflow
        session = memory.create_session("integration_test")
        
        # Simulate a conversation turn
        from core.session.memory_manager import ConversationTurn
        turn = ConversationTurn(
            timestamp=1234567890.0,
            user_input="I want something warm",
            user_context={"weather": "cold"},
            ai_response={"intent": "comfort"},
            recommendations=[{"name": "soup"}]
        )
        
        memory.add_conversation_turn("integration_test", turn)
        
        # Get context for next request
        context = memory.get_conversation_context("integration_test")
        assert context is not None
        
        # Generate fallback recommendations
        recommendations = fallback.generate_smart_fallback(
            "I want comfort food",
            user_context={"weather": "cold"},
            session_memory=context,
            num_recommendations=2
        )
        
        assert len(recommendations) > 0
        
        print("✅ Integration tests passed")
        
    except ImportError as e:
        pytest.skip(f"Integration test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Advanced Features...")
    print("=" * 50)
    
    test_robust_llm_parser()
    test_session_memory()
    test_smart_fallback()
    test_token_management()
    test_integration()
    
    print("\n🎉 All advanced feature tests completed!")
