from enum import Enum
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import asyncio

class FeedbackType(Enum):
    EXPLICIT_RATING = "explicit_rating"
    THUMBS_FEEDBACK = "thumbs_feedback"
    TEXT_FEEDBACK = "text_feedback"
    IMPLICIT_CLICK = "implicit_click"
    IMPLICIT_TIME = "implicit_time"
    IMPLICIT_ORDER = "implicit_order"
    IMPLICIT_SHARE = "implicit_share"
    IMPLICIT_DISMISS = "implicit_dismiss"
    CONTEXTUAL_FEEDBACK = "contextual_feedback"

class FeedbackCollector:
    def __init__(self):
        self.feedback_processors = {
            FeedbackType.EXPLICIT_RATING: self._process_rating_feedback,
            FeedbackType.THUMBS_FEEDBACK: self._process_thumbs_feedback,
            FeedbackType.TEXT_FEEDBACK: self._process_text_feedback,
            FeedbackType.IMPLICIT_CLICK: self._process_click_feedback,
            FeedbackType.IMPLICIT_TIME: self._process_time_feedback,
            FeedbackType.IMPLICIT_ORDER: self._process_order_feedback,
            FeedbackType.IMPLICIT_SHARE: self._process_share_feedback,
            FeedbackType.IMPLICIT_DISMISS: self._process_dismiss_feedback,
            FeedbackType.CONTEXTUAL_FEEDBACK: self._process_contextual_feedback
        }
        
        self.feedback_weights = {
            FeedbackType.EXPLICIT_RATING: 1.0,
            FeedbackType.THUMBS_FEEDBACK: 0.8,
            FeedbackType.TEXT_FEEDBACK: 0.9,
            FeedbackType.IMPLICIT_CLICK: 0.3,
            FeedbackType.IMPLICIT_TIME: 0.2,
            FeedbackType.IMPLICIT_ORDER: 0.9,
            FeedbackType.IMPLICIT_SHARE: 0.7,
            FeedbackType.IMPLICIT_DISMISS: 0.6,
            FeedbackType.CONTEXTUAL_FEEDBACK: 0.5
        }
    
    async def collect_feedback(self, feedback_data: Dict) -> Dict:
        """Collect and process feedback from multiple sources"""
        
        processed_feedback = {
            "feedback_id": feedback_data.get("feedback_id"),
            "user_id": feedback_data.get("user_id"),
            "recommendation_id": feedback_data.get("recommendation_id"),
            "food_id": feedback_data.get("food_id"),
            "timestamp": datetime.now(),
            "feedback_components": [],
            "overall_score": 0.0,
            "confidence": 0.0
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        # Process each type of feedback present
        for feedback_type, processor in self.feedback_processors.items():
            if feedback_type.value in feedback_data:
                component_result = await processor(
                    feedback_data[feedback_type.value],
                    feedback_data.get("context", {})
                )
                
                if component_result:
                    weight = self.feedback_weights[feedback_type]
                    processed_feedback["feedback_components"].append({
                        "type": feedback_type.value,
                        "score": component_result["score"],
                        "confidence": component_result["confidence"],
                        "weight": weight
                    })
                    
                    weighted_score += component_result["score"] * weight
                    total_weight += weight
        
        # Calculate overall feedback score
        if total_weight > 0:
            processed_feedback["overall_score"] = weighted_score / total_weight
            processed_feedback["confidence"] = min(total_weight / 3.0, 1.0)  # Normalize confidence
        
        return processed_feedback
    
    async def _process_rating_feedback(self, rating: int, context: Dict) -> Dict:
        """Process explicit 1-5 star ratings"""
        # Convert 1-5 rating to -1 to +1 scale
        normalized_score = (rating - 3) / 2.0
        
        return {
            "score": normalized_score,
            "confidence": 0.9,  # High confidence for explicit ratings
            "raw_value": rating
        }
    
    async def _process_thumbs_feedback(self, thumbs_up: bool, context: Dict) -> Dict:
        """Process thumbs up/down feedback"""
        score = 1.0 if thumbs_up else -1.0
        
        return {
            "score": score,
            "confidence": 0.8,
            "raw_value": thumbs_up
        }
    
    async def _process_text_feedback(self, text: str, context: Dict) -> Dict:
        """Process text feedback using sentiment analysis"""
        # Use sentiment analysis to extract score from text
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_result = await sentiment_analyzer.analyze(text)
        
        return {
            "score": sentiment_result["score"],  # -1 to +1
            "confidence": sentiment_result["confidence"],
            "raw_value": text,
            "sentiment_details": sentiment_result
        }
    
    async def _process_click_feedback(self, clicked: bool, context: Dict) -> Dict:
        """Process click behavior"""
        if not clicked:
            return None
        
        # Clicking indicates some level of interest
        score = 0.3  # Mild positive signal
        
        # Adjust based on context
        if context.get("position", 0) > 5:  # Clicked on lower-ranked item
            score += 0.2  # Higher interest signal
        
        return {
            "score": score,
            "confidence": 0.3,
            "raw_value": clicked
        }
    
    async def _process_time_feedback(self, time_spent: float, context: Dict) -> Dict:
        """Process time spent viewing recommendation"""
        if time_spent < 5:  # Less than 5 seconds
            return None
        
        # Normalize time spent (cap at 2 minutes)
        normalized_time = min(time_spent / 120.0, 1.0)
        
        # Convert to score (more time = more interest)
        score = (normalized_time - 0.1) * 0.5  # Scale to -0.05 to +0.45
        
        return {
            "score": score,
            "confidence": 0.2,
            "raw_value": time_spent
        }
    
    async def _process_order_feedback(self, ordered: bool, context: Dict) -> Dict:
        """Process order/purchase behavior"""
        if not ordered:
            return None
        
        # Ordering is a strong positive signal
        score = 1.0
        
        # Adjust based on context
        if context.get("delivery_time") and context["delivery_time"] > 60:
            score += 0.2  # Willing to wait longer = stronger preference
        
        return {
            "score": score,
            "confidence": 0.9,
            "raw_value": ordered
        }
    
    async def _process_share_feedback(self, shared: bool, context: Dict) -> Dict:
        """Process sharing behavior"""
        if not shared:
            return None
        
        # Sharing indicates strong positive sentiment
        score = 0.8
        
        return {
            "score": score,
            "confidence": 0.7,
            "raw_value": shared
        }
    
    async def _process_dismiss_feedback(self, dismissed: bool, context: Dict) -> Dict:
        """Process dismissal behavior"""
        if not dismissed:
            return None
        
        # Dismissing indicates negative sentiment
        score = -0.5
        
        # Adjust based on how quickly it was dismissed
        if context.get("time_to_dismiss", 0) < 2:  # Dismissed very quickly
            score = -0.8  # Stronger negative signal
        
        return {
            "score": score,
            "confidence": 0.6,
            "raw_value": dismissed
        }
    
    async def _process_contextual_feedback(self, contextual_data: Dict, context: Dict) -> Optional[Dict]:
        """Process contextual feedback (repeat behavior, etc.)"""
        score = 0.0
        confidence = 0.0
        
        # Repeat ordering behavior
        if contextual_data.get("repeat_order"):
            score += 0.6
            confidence += 0.3
        
        # Time of day consistency
        if contextual_data.get("consistent_timing"):
            score += 0.2
            confidence += 0.2
        
        # Social sharing patterns
        if contextual_data.get("social_validation"):
            score += 0.3
            confidence += 0.2
        
        return {
            "score": min(score, 1.0),
            "confidence": min(confidence, 1.0),
            "raw_value": contextual_data
        } if confidence > 0 else None
    
