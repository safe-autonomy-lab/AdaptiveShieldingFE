import math
import logging
import os
import numpy as np
from typing import List
from collections import deque, defaultdict
import torch
from .util import derivative_of


class ConformalPrediction:
    def __init__(self, window_size: int = 10, significance_level: float = 0.02, learning_rate: float = 0.01, min_samples: int = 10):
        self._logger = logging.getLogger("shield")
        self._setup_file_logger()
        self.window_size = window_size
        # significance level for conformal prediction, which is the adaptive failure probability
        self.significance_level = significance_level
        self.robot_failure_prob = significance_level
        self.learning_rate = learning_rate
        self.min_samples = min_samples
                
        # Initialize constraints (C_t^τ)
        self.robot_constraint = float('inf')
        self.mo_constraint = float('inf')

        # Track violations
        self.robot_violations = deque(maxlen=window_size)
        self.mo_violations = deque(maxlen=window_size)

        self.robot_nonconformity_scores = deque(maxlen=self.window_size)
        self.mo_nonconformity_scores = deque(maxlen=self.window_size)
        
        self.robot_failure_prob = self.significance_level   
        self.robot_constraint = float('inf')

        self.mo_failure_prob = self.significance_level
        self.mo_constraint = float('inf')

        self.robot_conformal_threshold = float('inf')
        self.mo_conformal_threshold = float('inf')

    def _setup_file_logger(self):
        log_path = os.getenv("SHIELD_LOG_PATH")
        if not log_path:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "shield_conformal.log")
        else:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_path):
                return
        self._logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        
    def _update_failure_probability(self, current_prob: float, violation: bool) -> float:
        """
        Update failure probability using the recursive formula:
        λ_t^τ = λ_{t-1}^τ + α(δ - I_{C_{t-1}^τ < β_{t-1}^τ})
        """
        return current_prob + self.learning_rate * (self.significance_level - float(violation))

    def _update_constraint(self, scores: list[torch.tensor], failure_prob: float) -> float:
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
            return torch.sort(torch.stack(scores))[0][q - 1]
        
    def _calculate_nonconformity_score(self, true_value: torch.tensor, predicted_value: torch.tensor) -> float:
        """Calculate nonconformity score as absolute error"""
        nonconformity_score = torch.linalg.norm(true_value - predicted_value)
        return nonconformity_score

    def _set_conformal_thresholds(self):
        self.robot_conformal_threshold = self._get_conformal_threshold(self.robot_nonconformity_scores)
        self.mo_conformal_threshold = self._get_conformal_threshold(self.mo_nonconformity_scores)
        self._logger.info(
            "Conformal thresholds: robot=%s mo=%s (n_robot=%s n_mo=%s)",
            self.robot_conformal_threshold,
            self.mo_conformal_threshold,
            len(self.robot_nonconformity_scores),
            len(self.mo_nonconformity_scores),
        )
    
    def _get_conformal_threshold(self, scores: List[float]) -> float:
        """Calculate conformal threshold based on recent scores"""
        N = len(scores)
        if N < self.min_samples:
            return self.static_threshold  # Return default threshold if not enough samples
        scores = list(scores)
        rank = math.ceil((1 - self.significance_level) * (N + 1))
        sorted_scores = torch.sort(torch.stack(scores)).values
        threshold = sorted_scores[min(rank - 1, N - 1)]
        return threshold.item() # return float

    def initialize_calibration_scores(self, robot_train_scores: List[float], mo_train_scores: List[float]):
        self.robot_nonconformity_scores.extend(robot_train_scores[:self.window_size])
        self.mo_nonconformity_scores.extend(mo_train_scores[:self.window_size])
        # Optionally update constraints immediately
        if len(self.robot_nonconformity_scores) >= self.min_samples:
            self.robot_constraint = self._update_constraint(
                list(self.robot_nonconformity_scores), self.robot_failure_prob
            )
        if len(self.mo_nonconformity_scores) >= self.min_samples:
            self.mo_constraint = self._update_constraint(
                list(self.mo_nonconformity_scores), self.mo_failure_prob
            )

    def add_extra_information(self, moving_obstacles_history):
        vx = derivative_of(moving_obstacles_history[..., 0], dt=0.02)
        vy = derivative_of(moving_obstacles_history[..., 1], dt=0.02)
        moving_obstacles_history = np.stack([
                moving_obstacles_history[..., 0], 
                moving_obstacles_history[..., 1],
                vx, vy
        ], axis=-1)  # [num_ped, history_len, 4]
        return moving_obstacles_history

    # TODO: I should correct this function to update the scores and constraints for both robot and mo
    # TODO: because shielding mechanism is not triggered every time step, there is a gap between the scores and constraints
    def update_conformality_scores(self):
        """Update nonconformity scores and adaptive parameters"""
        if len(self.robot_predictions_history) > 0 and len(self.robot_actual_history) > 0:
            robot_score = self._calculate_nonconformity_score(
                self.robot_actual_history[-1],  # actual position
                self.robot_predictions_history[-1]  # prediction
            )
            self.robot_nonconformity_scores.append(robot_score)
            
            # Check for violation and update failure probability
            robot_violation = robot_score > self.robot_constraint
            self.robot_violations.append(robot_violation)
            self.robot_failure_prob = self._update_failure_probability(self.robot_failure_prob, robot_violation)
            self._logger.info(
                "Robot conformal score=%s violation=%s failure_prob=%s constraint=%s",
                robot_score,
                robot_violation,
                self.robot_failure_prob,
                self.robot_constraint,
            )

        if len(self.mo_predictions_history) > 0 and len(self.mo_actual_history) > 0:
            mo_score = self._calculate_nonconformity_score(self.mo_actual_history[-1], self.mo_predictions_history[-1])
            self.mo_nonconformity_scores.append(mo_score)
            
            mo_violation = mo_score > self.mo_constraint
            self.mo_violations.append(mo_violation)
            self.mo_failure_prob = self._update_failure_probability(self.mo_failure_prob, mo_violation)
            self._logger.info(
                "MO conformal score=%s violation=%s failure_prob=%s constraint=%s",
                mo_score,
                mo_violation,
                self.mo_failure_prob,
                self.mo_constraint,
            )

        # Update constraints
        if len(self.robot_nonconformity_scores) >= self.min_samples:
            self.robot_constraint = self._update_constraint(list(self.robot_nonconformity_scores), self.robot_failure_prob)
            self._logger.info("Robot constraint updated: %s", self.robot_constraint)

        if len(self.mo_nonconformity_scores) >= self.min_samples:
            self.mo_constraint = self._update_constraint(list(self.mo_nonconformity_scores), self.mo_failure_prob)
            self._logger.info("MO constraint updated: %s", self.mo_constraint)
