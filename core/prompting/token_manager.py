"""
Adaptive Token & Prompt Management
==================================

This module provides intelligent token management and prompt optimization
to prevent truncation and API rate limits while maintaining context quality.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class TokenBudget:
    """Represents available token budget for a request."""
    total_tokens: int
    reserved_for_response: int
    available_for_prompt: int
    model_context_limit: int

@dataclass
class PromptSection:
    """Represents a section of the prompt with token count."""
    content: str
    token_count: int
    priority: int  # Higher priority = more important to keep
    section_type: str

class TokenManager:
    """Manages token usage and prompt optimization."""
    
    def __init__(self, model_name: str = "deepseek/deepseek-r1-0528:free"):
        self.model_name = model_name
        self.encoding = self._get_encoding()
        self.model_limits = self._get_model_limits()
        
    def _get_encoding(self) -> Any:
        """Get the appropriate tokenizer for the model."""
        try:
            # Try to get encoding for the specific model
            if "deepseek" in self.model_name.lower():
                return tiktoken.get_encoding("cl100k_base")  # GPT-4 style encoding
            elif "gpt" in self.model_name.lower():
                return tiktoken.get_encoding("cl100k_base")
            else:
                return tiktoken.get_encoding("cl100k_base")  # Default
        except Exception as e:
            logger.warning(f"Failed to get encoding: {e}, using fallback")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return None
    
    def _get_model_limits(self) -> Dict[str, int]:
        """Get token limits for the model."""
        limits = {
            "deepseek/deepseek-r1-0528:free": 8192,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 4096,
            "claude-3": 200000,
            "default": 4096
        }
        
        for model, limit in limits.items():
            if model in self.model_name.lower():
                return {"context_limit": limit, "max_tokens": limit - 1000}
        
        return {"context_limit": limits["default"], "max_tokens": limits["default"] - 1000}
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}")
        
        # Fallback: rough estimation
        return len(text) // 4
    
    def calculate_token_budget(self, max_response_tokens: int = 512) -> TokenBudget:
        """Calculate available token budget for a request."""
        context_limit = self.model_limits["context_limit"]
        reserved_for_response = max_response_tokens
        available_for_prompt = context_limit - reserved_for_response - 100  # Safety margin
        
        return TokenBudget(
            total_tokens=context_limit,
            reserved_for_response=reserved_for_response,
            available_for_prompt=available_for_prompt,
            model_context_limit=context_limit
        )
    
    def optimize_prompt(self, sections: List[PromptSection], budget: TokenBudget) -> Tuple[str, int]:
        """Optimize prompt to fit within token budget."""
        # Sort sections by priority (highest first)
        sorted_sections = sorted(sections, key=lambda x: x.priority, reverse=True)
        
        optimized_prompt = ""
        total_tokens = 0
        included_sections = []
        
        # Include high-priority sections first
        for section in sorted_sections:
            if total_tokens + section.token_count <= budget.available_for_prompt:
                optimized_prompt += section.content + "\n\n"
                total_tokens += section.token_count
                included_sections.append(section.section_type)
            else:
                logger.debug(f"Skipping section {section.section_type} due to token limit")
        
        logger.info(f"✅ Optimized prompt: {total_tokens}/{budget.available_for_prompt} tokens used")
        logger.info(f"📋 Included sections: {', '.join(included_sections)}")
        
        return optimized_prompt.strip(), total_tokens
    
    def create_structured_prompt(self, 
                               system_prompt: str,
                               user_query: str,
                               conversation_context: Optional[Dict[str, Any]] = None,
                               user_preferences: Optional[Dict[str, Any]] = None,
                               max_response_tokens: int = 512) -> str:
        """Create a structured prompt with optimal token usage."""
        budget = self.calculate_token_budget(max_response_tokens)
        
        # Build prompt sections with priorities
        sections = []
        
        # System prompt (highest priority)
        system_section = PromptSection(
            content=f"System: {system_prompt}",
            token_count=self.count_tokens(system_prompt),
            priority=10,
            section_type="system"
        )
        sections.append(system_section)
        
        # User preferences (high priority)
        if user_preferences:
            prefs_text = self._format_preferences(user_preferences)
            prefs_section = PromptSection(
                content=f"User Preferences: {prefs_text}",
                token_count=self.count_tokens(prefs_text),
                priority=8,
                section_type="preferences"
            )
            sections.append(prefs_section)
        
        # Conversation context (medium priority)
        if conversation_context:
            context_text = self._format_conversation_context(conversation_context)
            context_section = PromptSection(
                content=f"Recent Conversation: {context_text}",
                token_count=self.count_tokens(context_text),
                priority=6,
                section_type="conversation"
            )
            sections.append(context_section)
        
        # User query (highest priority)
        query_section = PromptSection(
            content=f"User Query: {user_query}",
            token_count=self.count_tokens(user_query),
            priority=10,
            section_type="query"
        )
        sections.append(query_section)
        
        # Response format (high priority)
        format_section = PromptSection(
            content="Respond ONLY in this exact JSON format: {\"intent\": \"...\", \"foods\": [...], \"reasoning\": \"...\"}",
            token_count=self.count_tokens("Respond ONLY in this exact JSON format: {\"intent\": \"...\", \"foods\": [...], \"reasoning\": \"...\"}"),
            priority=9,
            section_type="format"
        )
        sections.append(format_section)
        
        # Optimize and return
        optimized_prompt, token_count = self.optimize_prompt(sections, budget)
        
        return optimized_prompt
    
    def _format_preferences(self, preferences: Dict[str, Any]) -> str:
        """Format user preferences for prompt inclusion."""
        if not preferences:
            return "None specified"
        
        formatted = []
        for key, value in preferences.items():
            if isinstance(value, (int, float)):
                formatted.append(f"{key}: {value}")
            elif isinstance(value, list):
                formatted.append(f"{key}: {', '.join(map(str, value))}")
            else:
                formatted.append(f"{key}: {value}")
        
        return "; ".join(formatted)
    
    def _format_conversation_context(self, context: Dict[str, Any]) -> str:
        """Format conversation context for prompt inclusion."""
        if not context:
            return "None"
        
        # Extract recent conversation
        recent = context.get('recent_conversation', [])
        if not recent:
            return "No recent conversation"
        
        # Format last 2-3 turns
        formatted_turns = []
        for turn in recent[-3:]:
            user_input = turn.get('user_input', '')[:100]  # Truncate long inputs
            ai_response = turn.get('ai_response', {})
            
            # Extract key info from AI response
            if isinstance(ai_response, dict):
                intent = ai_response.get('intent', 'unknown')
                foods = ai_response.get('foods', [])
                food_summary = ', '.join(foods[:3]) if foods else 'none'
            else:
                intent = 'unknown'
                food_summary = 'unknown'
            
            turn_summary = f"User: '{user_input}' → AI: intent={intent}, foods=[{food_summary}]"
            formatted_turns.append(turn_summary)
        
        return " | ".join(formatted_turns)
    
    def estimate_response_tokens(self, prompt_tokens: int, model_context_limit: int) -> int:
        """Estimate how many tokens the response will use."""
        # Conservative estimate: response is typically 1/3 to 1/2 of prompt
        estimated_response = prompt_tokens // 3
        
        # Ensure we don't exceed model limits
        max_possible = model_context_limit - prompt_tokens - 100  # Safety margin
        
        return min(estimated_response, max_possible)
    
    def should_retry_with_shorter_prompt(self, 
                                       current_tokens: int, 
                                       budget: TokenBudget,
                                       retry_count: int) -> bool:
        """Determine if we should retry with a shorter prompt."""
        if retry_count >= 2:  # Max 2 retries
            return False
        
        # Retry if we're using more than 80% of available tokens
        usage_ratio = current_tokens / budget.available_for_prompt
        return usage_ratio > 0.8
    
    def create_fallback_prompt(self, user_query: str, max_tokens: int = 1000) -> str:
        """Create a minimal fallback prompt when token budget is exceeded."""
        fallback_system = "You are a food recommendation AI. Respond in JSON format."
        
        # Ensure we stay within limits
        available_tokens = max_tokens - self.count_tokens(fallback_system) - 100
        
        # Truncate query if necessary
        if self.count_tokens(user_query) > available_tokens:
            # Rough truncation (not perfect but functional)
            max_chars = available_tokens * 4
            user_query = user_query[:max_chars] + "..." if len(user_query) > max_chars else user_query
        
        prompt = f"""System: {fallback_system}

User Query: {user_query}

Respond ONLY in JSON: {{"intent": "...", "foods": [...], "reasoning": "..."}}"""
        
        return prompt
    
    def get_prompt_analytics(self, prompt: str) -> Dict[str, Any]:
        """Get analytics about prompt composition."""
        token_count = self.count_tokens(prompt)
        char_count = len(prompt)
        word_count = len(prompt.split())
        
        # Analyze prompt structure
        sections = {
            'system': len(re.findall(r'System:', prompt)),
            'user': len(re.findall(r'User', prompt)),
            'conversation': len(re.findall(r'Conversation', prompt)),
            'preferences': len(re.findall(r'Preferences', prompt))
        }
        
        return {
            'token_count': token_count,
            'char_count': char_count,
            'word_count': word_count,
            'sections': sections,
            'token_efficiency': char_count / token_count if token_count > 0 else 0
        }

# Global token manager instance
_token_manager = None

def get_token_manager(model_name: str = "deepseek/deepseek-r1-0528:free") -> TokenManager:
    """Get or create global token manager instance."""
    global _token_manager
    if _token_manager is None or _token_manager.model_name != model_name:
        _token_manager = TokenManager(model_name)
    return _token_manager
