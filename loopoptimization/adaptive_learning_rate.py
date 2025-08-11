class AdaptiveLearningRate:
    """Dynamically adjusts learning rates based on performance"""
    
    def __init__(self):
        self.base_learning_rate = 0.1
        self.performance_history = []
        self.adjustment_factor = 0.1
        
    def calculate_learning_rate(self, current_performance: float, user_confidence: float) -> float:
        """Calculate adaptive learning rate"""
        
        # Base learning rate
        learning_rate = self.base_learning_rate
        
        # Adjust based on performance trend
        if len(self.performance_history) >= 3:
            recent_trend = self._calculate_performance_trend()
            
            if recent_trend > 0:  # Performance improving
                learning_rate *= (1 + self.adjustment_factor)
            elif recent_trend < -0.1:  # Performance declining
                learning_rate *= (1 - self.adjustment_factor)
        
        # Adjust based on user confidence
        confidence_multiplier = 0.5 + (user_confidence * 0.5)  # 0.5 to 1.0
        learning_rate *= confidence_multiplier
        
        # Clamp learning rate
        learning_rate = max(0.01, min(0.5, learning_rate))
        
        # Record performance
        self.performance_history.append(current_performance)
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        return learning_rate
    
    def _calculate_performance_trend(self) -> float:
        """Calculate recent performance trend"""
        if len(self.performance_history) < 3:
            return 0.0
        
        recent = self.performance_history[-3:]
        return (recent[-1] - recent[0]) / len(recent)
    