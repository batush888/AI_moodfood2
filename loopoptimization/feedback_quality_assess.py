class FeedbackQualityAssessor:
    """Assesses the quality and reliability of user feedback"""
    
    def __init__(self):
        self.user_reliability_scores = {}
        self.feedback_consistency_threshold = 0.7
        
    async def assess_feedback_quality(self, feedback_data: Dict) -> Dict:
        """Assess the quality of feedback"""
        
        user_id = feedback_data.get("user_id")
        
        # Calculate reliability score for user
        user_reliability = await self._calculate_user_reliability(user_id)
        
        # Assess feedback consistency
        consistency_score = await self._assess_feedback_consistency(feedback_data)
        
        # Assess feedback completeness
        completeness_score = self._assess_feedback_completeness(feedback_data)
        
        # Calculate overall quality score
        quality_score = (
            user_reliability * 0.4 +
            consistency_score * 0.4 +
            completeness_score * 0.2
        )
        
        return {
            "quality_score": quality_score,
            "user_reliability": user_reliability,
            "consistency_score": consistency_score,
            "completeness_score": completeness_score,
            "should_use_feedback": quality_score > 0.5
        }
    
    async def _calculate_user_reliability(self, user_id: str) -> float:
        """Calculate user's historical feedback reliability"""
        
        if user_id not in self.user_reliability_scores:
            # New user - start with neutral reliability
            self.user_reliability_scores[user_id] = 0.7
        
        # In production, this would analyze:
        # - Consistency of ratings over time
        # - Correlation with other users' ratings
        # - Feedback vs. actual behavior (e.g., rating vs. ordering)
        
        return self.user_reliability_scores[user_id]
    
    async def _assess_feedback_consistency(self, feedback_data: Dict) -> float:
        """Assess internal consistency of feedback"""
        
        consistency_score = 1.0
        
        # Check for contradictory signals
        explicit_rating = feedback_data.get("explicit_rating")
        ordered = feedback_data.get("ordered", False)
        dismissed = feedback_data.get("dismissed", False)
        
        if explicit_rating:
            # High rating but dismissed - inconsistent
            if explicit_rating >= 4 and dismissed:
                consistency_score -= 0.3
            
            # Low rating but ordered - inconsistent
            if explicit_rating <= 2 and ordered:
                consistency_score -= 0.3
        
        # Check time spent vs. other signals
        time_spent = feedback_data.get("time_spent_viewing", 0)
        if time_spent < 5 and ordered:  # Ordered very quickly
            consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _assess_feedback_completeness(self, feedback_data: Dict) -> float:
        """Assess completeness of feedback"""
        
        completeness_factors = [
            "explicit_rating" in feedback_data,
            "clicked" in feedback_data,
            "time_spent_viewing" in feedback_data,
            "context" in feedback_data,
            feedback_data.get("confidence", 0) > 0.5
        ]
        
        return sum(completeness_factors) / len(completeness_factors)