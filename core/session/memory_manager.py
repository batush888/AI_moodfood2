"""
Session Memory Manager for Continuous Conversation
================================================

This module provides session-based memory management to enable continuous
conversation and context awareness in the AI food recommendation system.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import redis
import os

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: float
    user_input: str
    user_context: Dict[str, Any]
    ai_response: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    feedback: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SessionMemory:
    """Represents the complete memory for a user session."""
    session_id: str
    user_id: Optional[str]
    created_at: float
    last_updated: float
    conversation_history: List[ConversationTurn]
    user_preferences: Dict[str, Any]
    context_summary: Dict[str, Any]
    total_tokens: int = 0
    max_tokens: int = 4000  # Conservative token limit

class MemoryManager:
    """Manages session memory with intelligent pruning and summarization."""
    
    def __init__(self, redis_url: str = None, max_sessions: int = 1000):
        self.max_sessions = max_sessions
        self.redis_client = None
        
        # Initialize Redis if available
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("✅ Redis connection established for session memory")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}, using in-memory storage")
                self.redis_client = None
        
        # In-memory fallback storage
        self.sessions: Dict[str, SessionMemory] = {}
        
    def create_session(self, session_id: str, user_id: Optional[str] = None) -> SessionMemory:
        """Create a new session with initial memory."""
        session = SessionMemory(
            session_id=session_id,
            user_id=user_id,
            created_at=time.time(),
            last_updated=time.time(),
            conversation_history=[],
            user_preferences={},
            context_summary={}
        )
        
        if self.redis_client:
            self._save_to_redis(session)
        else:
            self.sessions[session_id] = session
            
        logger.info(f"🆕 Created new session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Retrieve a session from memory."""
        if self.redis_client:
            return self._load_from_redis(session_id)
        else:
            return self.sessions.get(session_id)
    
    def add_conversation_turn(self, session_id: str, turn: ConversationTurn) -> bool:
        """Add a new conversation turn to the session."""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"⚠️ Session not found: {session_id}")
            return False
        
        # Add the turn
        session.conversation_history.append(turn)
        session.last_updated = time.time()
        
        # Update context summary
        self._update_context_summary(session)
        
        # Check if we need to prune memory
        if self._should_prune_memory(session):
            self._prune_memory(session)
        
        # Save updated session
        if self.redis_client:
            self._save_to_redis(session)
        else:
            self.sessions[session_id] = session
            
        logger.debug(f"💬 Added conversation turn to session {session_id}")
        return True
    
    def get_conversation_context(self, session_id: str, max_turns: int = 5) -> Dict[str, Any]:
        """Get recent conversation context for LLM prompts."""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Get recent turns
        recent_turns = session.conversation_history[-max_turns:] if session.conversation_history else []
        
        # Build context
        context = {
            'session_id': session_id,
            'user_preferences': session.user_preferences,
            'context_summary': session.context_summary,
            'recent_conversation': []
        }
        
        for turn in recent_turns:
            context['recent_conversation'].append({
                'user_input': turn.user_input,
                'ai_response': turn.ai_response,
                'recommendations': turn.recommendations,
                'timestamp': turn.timestamp
            })
        
        return context
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences based on conversation and feedback."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Merge new preferences with existing ones
        for key, value in preferences.items():
            if key in session.user_preferences:
                # For numeric values, take average or latest
                if isinstance(value, (int, float)) and isinstance(session.user_preferences[key], (int, float)):
                    session.user_preferences[key] = (session.user_preferences[key] + value) / 2
                else:
                    session.user_preferences[key] = value
            else:
                session.user_preferences[key] = value
        
        session.last_updated = time.time()
        
        # Save updated session
        if self.redis_client:
            self._save_to_redis(session)
        else:
            self.sessions[session_id] = session
            
        logger.info(f"🔄 Updated user preferences for session {session_id}")
        return True
    
    def add_feedback(self, session_id: str, turn_index: int, feedback: Dict[str, Any]) -> bool:
        """Add user feedback to a specific conversation turn."""
        session = self.get_session(session_id)
        if not session or turn_index >= len(session.conversation_history):
            return False
        
        # Add feedback to the turn
        session.conversation_history[turn_index].feedback = feedback
        session.last_updated = time.time()
        
        # Update preferences based on feedback
        self._update_preferences_from_feedback(session, feedback)
        
        # Save updated session
        if self.redis_client:
            self._save_to_redis(session)
        else:
            self.sessions[session_id] = session
            
        logger.info(f"📝 Added feedback to session {session_id}, turn {turn_index}")
        return True
    
    def _update_context_summary(self, session: SessionMemory) -> None:
        """Update the context summary based on recent conversation."""
        if not session.conversation_history:
            return
        
        # Analyze recent turns for context
        recent_turns = session.conversation_history[-3:]  # Last 3 turns
        
        # Extract common themes
        themes = {}
        for turn in recent_turns:
            # Extract food preferences
            if turn.recommendations:
                for rec in turn.recommendations:
                    category = rec.get('food_category', 'unknown')
                    themes[category] = themes.get(category, 0) + 1
            
            # Extract mood patterns
            if turn.user_context:
                mood = turn.user_context.get('mood', 'unknown')
                themes[mood] = themes.get(mood, 0) + 1
        
        # Update context summary
        session.context_summary = {
            'recent_themes': themes,
            'conversation_length': len(session.conversation_history),
            'last_update': time.time()
        }
    
    def _update_preferences_from_feedback(self, session: SessionMemory, feedback: Dict[str, Any]) -> None:
        """Update user preferences based on feedback."""
        if not feedback:
            return
        
        # Extract preference signals from feedback
        if feedback.get('accepted_recommendations'):
            for rec in feedback['accepted_recommendations']:
                category = rec.get('food_category', 'unknown')
                session.user_preferences[f'preferred_{category}'] = session.user_preferences.get(f'preferred_{category}', 0) + 1
        
        if feedback.get('rejected_recommendations'):
            for rec in feedback['rejected_recommendations']:
                category = rec.get('food_category', 'unknown')
                session.user_preferences[f'avoided_{category}'] = session.user_preferences.get(f'avoided_{category}', 0) + 1
    
    def _should_prune_memory(self, session: SessionMemory) -> bool:
        """Check if memory should be pruned."""
        # Prune if too many turns or too old
        if len(session.conversation_history) > 20:  # Max 20 turns
            return True
        
        # Prune if session is very old (24 hours)
        if time.time() - session.created_at > 86400:
            return True
        
        return False
    
    def _prune_memory(self, session: SessionMemory) -> None:
        """Prune old conversation history while preserving important context."""
        if len(session.conversation_history) <= 10:
            return
        
        # Keep first 2 turns (establishment) and last 8 turns (recent)
        turns_to_keep = session.conversation_history[:2] + session.conversation_history[-8:]
        
        # Summarize middle turns
        middle_turns = session.conversation_history[2:-8]
        if middle_turns:
            summary = self._summarize_turns(middle_turns)
            # Add summary as a special turn
            summary_turn = ConversationTurn(
                timestamp=time.time(),
                user_input="[Conversation Summary]",
                user_context={},
                ai_response={"summary": summary},
                recommendations=[],
                metadata={"type": "summary", "original_turns": len(middle_turns)}
            )
            turns_to_keep.insert(2, summary_turn)
        
        session.conversation_history = turns_to_keep
        logger.info(f"🧹 Pruned memory for session {session.session_id}")
    
    def _summarize_turns(self, turns: List[ConversationTurn]) -> str:
        """Create a summary of multiple conversation turns."""
        if not turns:
            return ""
        
        # Extract key information
        food_preferences = set()
        mood_patterns = set()
        
        for turn in turns:
            if turn.recommendations:
                for rec in turn.recommendations:
                    category = rec.get('food_category', 'unknown')
                    food_preferences.add(category)
            
            if turn.user_context:
                mood = turn.user_context.get('mood', 'unknown')
                mood_patterns.add(mood)
        
        summary_parts = []
        if food_preferences:
            summary_parts.append(f"Discussed foods: {', '.join(food_preferences)}")
        if mood_patterns:
            summary_parts.append(f"Mood patterns: {', '.join(mood_patterns)}")
        
        return "; ".join(summary_parts) if summary_parts else "General food discussion"
    
    def _save_to_redis(self, session: SessionMemory) -> None:
        """Save session to Redis."""
        if not self.redis_client:
            return
        
        try:
            # Convert to JSON-serializable format
            session_data = asdict(session)
            session_data['conversation_history'] = [
                asdict(turn) for turn in session.conversation_history
            ]
            
            # Save to Redis with expiration (24 hours)
            key = f"session:{session.session_id}"
            self.redis_client.setex(
                key,
                86400,  # 24 hours
                json.dumps(session_data, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to save session to Redis: {e}")
    
    def _load_from_redis(self, session_id: str) -> Optional[SessionMemory]:
        """Load session from Redis."""
        if not self.redis_client:
            return None
        
        try:
            key = f"session:{session_id}"
            data = self.redis_client.get(key)
            
            if not data:
                return None
            
            session_data = json.loads(data)
            
            # Reconstruct ConversationTurn objects
            conversation_history = []
            for turn_data in session_data.get('conversation_history', []):
                turn = ConversationTurn(**turn_data)
                conversation_history.append(turn)
            
            # Create SessionMemory object
            session = SessionMemory(
                session_id=session_data['session_id'],
                user_id=session_data.get('user_id'),
                created_at=session_data['created_at'],
                last_updated=session_data['last_updated'],
                conversation_history=conversation_history,
                user_preferences=session_data.get('user_preferences', {}),
                context_summary=session_data.get('context_summary', {}),
                total_tokens=session_data.get('total_tokens', 0),
                max_tokens=session_data.get('max_tokens', 4000)
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session from Redis: {e}")
            return None
    
    def cleanup_old_sessions(self) -> int:
        """Clean up old sessions to free memory."""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            # Remove sessions older than 24 hours
            if current_time - session.last_updated > 86400:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            logger.info(f"🧹 Cleaned up {len(sessions_to_remove)} old sessions")
        
        return len(sessions_to_remove)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        total_sessions = len(self.sessions)
        total_conversations = sum(len(s.conversation_history) for s in self.sessions.values())
        
        return {
            'total_sessions': total_sessions,
            'total_conversations': total_conversations,
            'avg_conversations_per_session': total_conversations / total_sessions if total_sessions > 0 else 0,
            'memory_usage_mb': total_conversations * 0.1  # Rough estimate
        }

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        redis_url = os.getenv('REDIS_URL')
        _memory_manager = MemoryManager(redis_url=redis_url)
    return _memory_manager
