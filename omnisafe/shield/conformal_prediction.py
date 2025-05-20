import numpy as np
from typing import List
from collections import deque
from .util import derivative_of


class ConformalPrediction:
    def __init__(self, window_size: int = 10, significance_level: float = 0.02, learning_rate: float = 0.01, min_samples: int = 10):
        self.window_size = window_size
        # significance level for conformal prediction, which is the adaptive failure probability
        self.significance_level = significance_level
        self.robot_failure_prob = significance_level
        self.learning_rate = learning_rate
        self.min_samples = min_samples
                
        # Initialize constraints (C_t^τ)
        self.robot_constraint = float('inf')

        # Track violations
        self.robot_violations = deque(maxlen=window_size)

        self.robot_nonconformity_scores = deque(maxlen=self.window_size)
        
        self.robot_failure_prob = self.significance_level   
        self.robot_constraint = float('inf')

    def _update_failure_probability(self, current_prob: float, violation: bool) -> float:
        """
        Update failure probability using the recursive formula:
        λ_t^τ = λ_{t-1}^τ + α(δ - I_{C_{t-1}^τ < β_{t-1}^τ})
        """
        return current_prob + self.learning_rate * (self.significance_level - float(violation))

    def _update_constraint(self, scores: list, failure_prob: float) -> float:
        """
        Update constraint (C_t^τ) based on quantile of historical scores
        """
        if len(scores) == 0:
            return float('inf')
            
        N = len(scores)
        q = int(np.ceil((N + 1) * (1 - failure_prob)))
        
        if q > N:
            return float('inf')
        elif q < 1:
            return 0.0
        else:
            return np.sort(scores)[q - 1]
        
    def _calculate_nonconformity_score(self, true_value: float, predicted_value: float) -> float:
        """Calculate nonconformity score as absolute error"""
        nonconformity_score = np.linalg.norm(true_value - predicted_value)
        return nonconformity_score

    def _set_conformal_thresholds(self):
        self.robot_conformal_threshold = self._get_conformal_threshold(self.robot_nonconformity_scores)
    
    def _get_conformal_threshold(self, scores: List[float]) -> float:
        """Calculate conformal threshold based on recent scores"""
        if len(scores) < self.min_samples:
            return self.static_threshold  # Return default threshold if not enough samples
        
        scores_array = np.array(scores)
        rank = int(np.ceil((1 - self.significance_level) * (len(scores) + 1)))
        sorted_scores = np.sort(scores_array)
        threshold = sorted_scores[min(rank - 1, len(scores) - 1)]
        return threshold

    def initialize_calibration_scores(self, robot_train_scores: List[float]):
        self.robot_nonconformity_scores.extend(robot_train_scores[:self.window_size])
        # Optionally update constraints immediately
        if len(self.robot_nonconformity_scores) >= self.min_samples:
            self.robot_constraint = self._update_constraint(
                list(self.robot_nonconformity_scores), self.robot_failure_prob
            )

    def update_conformality_scores(self):
        """Update nonconformity scores and adaptive parameters"""
        if len(self.robot_predictions_history) > 0 and len(self.robot_actual_history) > 0:
            robot_score = self._calculate_nonconformity_score(
                self.robot_actual_history[-1],  # Current actual position
                self.robot_predictions_history[-1]  # Previous prediction for current position
            )
            self.robot_nonconformity_scores.append(robot_score)
            
            # Check for violation and update failure probability
            robot_violation = robot_score > self.robot_constraint
            self.robot_violations.append(robot_violation)
            self.robot_failure_prob = self._update_failure_probability(self.robot_failure_prob, robot_violation)

        # Update constraints
        if len(self.robot_nonconformity_scores) >= self.min_samples:
            self.robot_constraint = self._update_constraint(list(self.robot_nonconformity_scores), self.robot_failure_prob)
