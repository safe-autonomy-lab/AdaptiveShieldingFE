from typing import Dict, Tuple
import logging
from collections import deque
import torch
import numpy as np
from shield.base_shield import BaseShield
from shield.util import compute_min_distance
from omnisafe.envs.wrapper import Normalizer
from FunctionEncoder import FunctionEncoder


class VectorizedShield(BaseShield):
    """Shield structure for RL algorithms that produces safer policies.
    
    Takes a policy and observations to generate safer actions by predicting future states
    and checking for potential collisions.
    
    Attributes:
        scene: Scene information for efficient use of obstacle history
        dynamic_predictor: Predicts agent's future states
        moving_obstacles_predictor: Predicts moving obstacles' future states  
        prediction_horizon: Number of prediction steps
    """

    def __init__(
        self,
        dynamic_predictor: FunctionEncoder,
        mo_predictor: FunctionEncoder,
        sampling_nbr: int,
        prediction_horizon: int,
        safety_bonus: float = 1.0,
        window_size: int = 100,
        significance_level: float = 0.1,
        static_threshold: float = 0.225,
        mo_threshold: float = 0.225,
        example_nbr: int = 100,
        warm_up_epochs: int = 0,
        idle_condition: int = 4,
        dynamics_model_type: str = 'fe',
        scale: float = 0.05,
        penalty_type: str = 'reward',
        **kwargs,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self.is_object_checked = False
        self.is_swimmer = False
        self.is_cheetah = False
        self.static_threshold = static_threshold
        self.mo_threshold = mo_threshold
        self.example_nbr = example_nbr
        self.safety_bonus = safety_bonus
        self.shield_triggered = False
        self.warm_up_epochs = warm_up_epochs
        self.dynamics_model_type = dynamics_model_type
        self.scale = scale
        self.penalty_type = penalty_type
        if self.penalty_type == "shield":
            print("Safe bonus override: no SRO since we use shield only mode, from self.penalty_type == 'shield'")
            self.safety_bonus = 0.0
        """Initialize the shield.

        Args:
            env_id: Environment identifier
            dynamic_predictor_cfgs: Configuration for dynamic state predictor
            moving_obstacles_predictor_cfgs: Configuration for obstacle predictor
            sampling_nbr: Number of samples for prediction
            prediction_horizon: Number of steps to predict ahead
            threshold: Safety threshold distance
            discount_factor: Discount factor for future predictions
            use_hidden_param: Whether to use hidden parameters
            use_online_update: Whether to update online
            window_size: Window size for predictions
            significance_level: Statistical significance level
            safety_bonus: Safety bonus
            gradient_scale: Gradient scale
            warm_up_epochs: Number of warm-up epochs
        """
        super().__init__(
            dynamic_predictor=dynamic_predictor,
            mo_predictor=mo_predictor,
            sampling_nbr=sampling_nbr,
            prediction_horizon=prediction_horizon,
            window_size=window_size,
            significance_level=significance_level,
            idle_condition=idle_condition,
            **kwargs,
        )
        self.xs_history = []
        self.ys_history = []
        self.robot_slices = slice(None)
        self._last_norm = None
        self._x_queue = deque(maxlen=self.prediction_horizon + 1)
        self._pos_queue = deque(maxlen=self.prediction_horizon + 1)
        self.normalizer = Normalizer(shape=(self.n_basis,), clip=1000.).to(self.device)
        
    def update_weights(self, step):
        shield_trigger = False
        example_x = torch.cat([self.prev_dp_input, self.prev_action], axis=-1).detach()
        example_y = self.dp_y
        
        if self.prediction_horizon > 1:
            self._x_queue.append(example_x)
            self._pos_queue.append(example_y)
            if len(self._pos_queue) >= self.prediction_horizon + 1:
                deltas = [
                    self._pos_queue[i + 1] - self._pos_queue[i]
                    for i in range(self.prediction_horizon)
                ]
                target_y = torch.cat(deltas, dim=1)
                target_x = self._x_queue[0]
                self._x_queue.popleft()
                self._pos_queue.popleft()
                self.xs_history.append(target_x.unsqueeze(1))
                self.ys_history.append(target_y.unsqueeze(1))
        else:
            self.xs_history.append(example_x.unsqueeze(1))
            self.ys_history.append(example_y.unsqueeze(1))

        if len(self.xs_history) > self.example_nbr:
            self.xs_history.pop(0)
            self.ys_history.pop(0)
        
        if len(self.xs_history) == 0 or len(self.ys_history) == 0:
            return False
        example_xs = torch.cat(self.xs_history, dim=1)
        example_ys = torch.cat(self.ys_history, dim=1)

        if len(self.xs_history) >= self.example_nbr:
            weights = self._compute_coefs(example_xs, example_ys)
            if self.normalizer.mean.device != weights.device:
                self.normalizer.to(weights.device)
            self.normalizer.update(weights)
            self.normalized_coeffs_for_dynamics_prediction = self.normalizer.normalize(weights)
            shield_trigger = True
            
        return shield_trigger
        
    def _process_agent_information(self, info: Dict):
        """Process and normalize agent position from environment info."""
        agent_pos = np.stack(info['agent_pos'])
        agent_mat = np.stack(info['agent_mat'])
        if len(agent_pos.shape) == 1:
            agent_pos = agent_pos.reshape(1, -1)
        if len(agent_mat.shape) == 1:
            agent_mat = agent_mat.reshape(1, -1)
        
        return agent_pos, agent_mat

    def _check_presafety_condition(self, info: Dict, enhanced_safety: float = 0.0):
        """Check if the current state satisfies safety conditions."""
        if self.is_circle:
            agent_pos = np.array([pos[:2] for pos in info["agent_pos"]])
            range_limit_check = np.abs(agent_pos) > self.range_limit - 0.125
            unsafe_condition = np.any(range_limit_check, axis=1, keepdims=False)
        else:
            if 'min_distance' in info:
                unsafe_condition = info['min_distance'] < self.static_threshold + enhanced_safety
            else:
                if self.is_swimmer:
                    unsafe_condition = np.any(np.abs(info['x_velocity'].reshape(-1, 1)) > 0.095, axis=1, keepdims=False)
                elif self.is_cheetah:
                    unsafe_condition = np.any(np.abs(info['x_velocity'].reshape(-1, 1)) > 1.9, axis=1, keepdims=False)
                else:
                    raise ValueError(f'Unknown environment: {self.env_info}')
            
        return unsafe_condition

    @torch.no_grad()
    def _compute_coefs(self, example_xs, example_ys):
        if self.dynamics_model_type in ['fe', 'transformer']:
            coefs, _ = self.dynamic_predictor.compute_representation(example_xs, example_ys, method='least_squares')
            self.coeffs_for_dynamics_prediction = coefs
        else:
            coefs = 0
        return coefs
        
    def process_info(self, info: Dict, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process environment info to extract relevant elements.

        Args:
            info: Dictionary containing environment information

        Returns:
            Tuple containing:
            - Agent position and orientation matrix
            - Goal position
            - Button positions
            - Static obstacle positions
            - Moving obstacle positions
            - Moving obstacle z-coordinates
            - Hidden parameters
        """    
        
        agent_pos = (
            np.array([pos for pos in info["agent_pos"]]) 
            if "agent_pos" in info else np.zeros((batch_size, 3))
        )

        agent_mat = (
            np.array([mat for mat in info["agent_mat"]])
            if "agent_mat" in info else np.zeros((batch_size, 9))
        )

        buttons = (
            np.array([button for button in info["buttons"]])
            if "buttons" in info else np.array([])
        )

        goal_pos = (
            np.array([pos for pos in info["goal_pos"]])
            if "goal_pos" in info else np.array([])
        )

        hazards = (
            np.array([hazard for hazard in info["hazards"]])
            if "hazards" in info else np.array([])
        )
        vases = (
            np.array([vase for vase in info["vases"]])
            if "vases" in info else np.array([])
        )
        pillars = (
            np.array([pillar for pillar in info["pillars"]])
            if "pillars" in info else np.array([])
        )
        push_boxes = (
            np.array([push_box for push_box in info["push_box"]])
            if "push_box" in info else np.array([])
        )
        gremlins = (
            np.array([gremlin for gremlin in info["gremlins"]])
            if "gremlins" in info else np.array([])
        )
        circle = (
            np.array([circle for circle in info["circle"]])
            if "circle" in info else np.array([])
        )
        
        return agent_pos, agent_mat, buttons, goal_pos, hazards, vases, pillars, push_boxes, gremlins, circle

    def sample_safe_actions(
        self,
        dp_input: np.ndarray,
        agent_pos: np.ndarray,
        buttons: np.ndarray,
        goal_pos: np.ndarray,
        hazards: np.ndarray,
        vases: np.ndarray,
        pillars: np.ndarray,
        push_boxes: np.ndarray,
        gremlins: np.ndarray,
        circle: np.ndarray,
        first_action: np.ndarray,
        device: str = 'cpu',
        selection_method: str = 'top-k',
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # check if the object is present in the environment
        if not self.is_object_checked and not self.is_velocity:
            self.is_gremlins = gremlins.shape[0] > 0
            self.is_push_boxes = push_boxes.shape[0] > 0
            self.is_vases = vases.shape[0] > 0
            self.is_hazards = hazards.shape[0] > 0
            self.is_pillars = pillars.shape[0] > 0
            self.is_buttons = buttons.shape[0] > 0
            self.is_goal = goal_pos.shape[0] > 0
            self.is_circle = circle.shape[0] > 0
            self.is_object_checked = True            

        # Repeat dp_input to match first_action's first dimension
        # dp_input shape: (4, 12) -> (sampling_nbr, vector_num_env, robot_dim + action_dim)
        self.shield_triggered = True
        dp_input_repeated = dp_input.unsqueeze(0).repeat(self.sampling_nbr, 1, 1)

        obs_dp_input = (
            torch.cat([dp_input_repeated, first_action], dim=-1)  # → (sampling_nbr, num_env, robot+action)
            .transpose(0, 1)                                # → (num_env, sampling_nbr, robot+action)
            .contiguous()                                   # (optional) for safe downstream reshaping
        )
        
        with torch.no_grad():
            robot_delta_predictions = self.dynamic_predictor.predict(obs_dp_input, self.coeffs_for_dynamics_prediction).transpose(0, 1) # (sampling_nbr, num_env, output_size)

        sampling_nbr, num_envs, output_size = robot_delta_predictions.shape
        per_step_dim = 1 if self.is_velocity else 2
        has_multi_step = (
            self.prediction_horizon > 1
            and output_size == per_step_dim * self.prediction_horizon
        )
        if self.prediction_horizon > 1 and not has_multi_step and not self.is_velocity:
            raise ValueError(
                f"Multi-step shield expects output_size={per_step_dim * self.prediction_horizon}, got {output_size}"
            )
        robot_xy_predictions_steps = None

        if self.is_velocity:
            robot_xy_predictions = robot_delta_predictions
            if not has_multi_step:
                robot_xy_predictions_steps = robot_xy_predictions.unsqueeze(2)
        elif has_multi_step and not self.is_velocity:
            deltas = robot_delta_predictions.view(
                sampling_nbr,
                num_envs,
                self.prediction_horizon,
                per_step_dim,
            )
            base_pos = torch.from_numpy(agent_pos[:, :2]).float().to(device)
            robot_xy_predictions_steps = base_pos.unsqueeze(0).unsqueeze(2) + torch.cumsum(deltas, dim=2)
            robot_xy_predictions = robot_xy_predictions_steps[:, :, 0, :]
        else:
            robot_xy_predictions = robot_delta_predictions + torch.from_numpy(agent_pos[np.newaxis, :, :2]).float().to(device)
            if not self.is_velocity:
                robot_xy_predictions_steps = robot_xy_predictions.unsqueeze(2)
        
        mean_robot_xy_predictions = torch.mean(robot_xy_predictions, axis=0)
        self.robot_predictions_history.append(mean_robot_xy_predictions.detach())
        
        if not self.is_velocity and self.is_gremlins and self.mo_last_obs is not None:
            predicted_gremlins, _ = self.get_mo_predictions()
            for i in range(self.prediction_horizon):
                self.mo_predictions_history.append(predicted_gremlins[:, :, i, :])

            vectorized_gremlins = predicted_gremlins.reshape(num_envs, -1, self.prediction_horizon, 2).unsqueeze(0).repeat(self.sampling_nbr, 1, 1, 1, 1)        
        elif not self.is_velocity and self.is_gremlins:
            gremlins_xy = gremlins[:, np.newaxis, :2]
            gremlins_xy = np.tile(gremlins_xy[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1))
            vectorized_gremlins = torch.from_numpy(gremlins_xy).float().to(device)
            vectorized_gremlins = vectorized_gremlins.unsqueeze(3).repeat(1, 1, 1, self.prediction_horizon, 1)

        conformal_threshold = self.robot_conformal_threshold
        static_adjusted_threshold = self.static_threshold + conformal_threshold
        mo_adjusted_threshold = self.mo_threshold + conformal_threshold
        range_limit_adjusted_threshold = 0.1 + conformal_threshold
        
        # weighted_distance_min will be safety measure for each prediction step
        weighted_distance_min = 0.0
        if self.is_circle:
            abs_robot_pos = torch.abs(robot_xy_predictions)
            distance2bounds = torch.max(abs_robot_pos, dim=-1).values
            min_indices = torch.argmin(distance2bounds, dim=0)
            distance2bound = distance2bounds[min_indices, torch.arange(len(min_indices))]
            safe_mask = distance2bound < self.range_limit - range_limit_adjusted_threshold        
            safe_mask = safe_mask.detach().cpu().numpy().astype(bool)
            circle = circle[:, np.newaxis, :]
            
        elif self.is_velocity:
            abs_robot_vel = torch.abs(robot_delta_predictions)
            if has_multi_step:
                abs_robot_vel = abs_robot_vel.view(
                    sampling_nbr,
                    num_envs,
                    self.prediction_horizon,
                ).max(dim=2).values
            else:
                abs_robot_vel = abs_robot_vel.squeeze(-1)
            distance2bounds = abs_robot_vel
            min_indices = torch.argmax(distance2bounds, dim=0)
            distance2bound = distance2bounds[min_indices, torch.arange(len(min_indices))]
            # for velocity case, we need to ensure tighter threshold, so substract
            if self.is_swimmer:
                safe_threshold = 0.09
            elif self.is_cheetah:
                safe_threshold = 1.9
            safe_mask = distance2bounds < safe_threshold - conformal_threshold
            safe_mask = safe_mask.detach().cpu().numpy().astype(bool)
            weighted_distance_min = distance2bounds.detach().cpu().numpy()
        else:
            vectorized_hazards = torch.from_numpy(np.tile(hazards[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1))).float().to(device) if self.is_hazards else torch.inf
            vectorized_pillars = torch.from_numpy(np.tile(pillars[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1))).float().to(device) if self.is_pillars else torch.inf
            safe_mask = np.ones((self.sampling_nbr, num_envs)).astype(bool)

        # Vectorized safety check across all future steps
        if self.is_circle:
            abs_robot_pos = torch.abs(robot_xy_predictions_steps)
            distance2bound = torch.max(abs_robot_pos, dim=-1).values  # (sampling, env, horizon)
            unsafe_mask = distance2bound > self.range_limit - range_limit_adjusted_threshold
            unsafe_any = unsafe_mask.any(dim=2)
            safe_mask = safe_mask & ~unsafe_any.detach().cpu().numpy()
            distance_min = distance2bound

        elif not self.is_velocity:
            inf_tensor = torch.full(
                (sampling_nbr, num_envs, self.prediction_horizon),
                float("inf"),
                device=device,
            )
            if self.is_gremlins:
                gremlins_steps = vectorized_gremlins.permute(0, 1, 3, 2, 4)
                distance2gremlins = torch.norm(
                    gremlins_steps - robot_xy_predictions_steps.unsqueeze(3),
                    dim=-1,
                ).min(dim=-1).values
            else:
                distance2gremlins = inf_tensor

            if self.is_hazards:
                hazards_steps = vectorized_hazards[:, :, :, :2].unsqueeze(2)
                distance2hazards = torch.norm(
                    hazards_steps - robot_xy_predictions_steps.unsqueeze(3),
                    dim=-1,
                ).min(dim=-1).values
            else:
                distance2hazards = inf_tensor

            if self.is_pillars:
                pillars_steps = vectorized_pillars[:, :, :, :2].unsqueeze(2)
                distance2pillars = torch.norm(
                    pillars_steps - robot_xy_predictions_steps.unsqueeze(3),
                    dim=-1,
                ).min(dim=-1).values
            else:
                distance2pillars = inf_tensor

            distance2static = torch.minimum(distance2pillars, distance2hazards)
            distance_min = torch.minimum(distance2gremlins, distance2static)

            unsafe_mask = torch.logical_or(
                distance2gremlins <= mo_adjusted_threshold,
                distance2static <= static_adjusted_threshold,
            )
            unsafe_any = unsafe_mask.any(dim=2)
            safe_mask = safe_mask & ~unsafe_any.detach().cpu().numpy()

        if not self.is_velocity:
            discount = (0.9 ** torch.arange(self.prediction_horizon, device=device)).view(1, 1, -1)
            weighted_distance_min = (distance_min * discount).sum(dim=2).detach().cpu().numpy()

        final_indices = np.zeros(num_envs, dtype=int)

        for env_idx in range(num_envs):
            wdm_env = weighted_distance_min[:, env_idx]
            safe_mask_env = safe_mask[:, env_idx]
            safe_action_indices = np.where(safe_mask_env)[0]

            if len(safe_action_indices) > 0:
                # At least one safe action exists for this environment
                safe_wdm = wdm_env[safe_action_indices]

                if selection_method == 'greedy':
                    best_safe_idx_in_filtered = np.argmax(safe_wdm)
                    selected_idx = safe_action_indices[best_safe_idx_in_filtered]
                elif selection_method == 'top-k':
                    # Ensure k is not larger than the number of safe actions
                    actual_k = min(k, len(safe_action_indices))
                    if actual_k <= 0:
                         # Fallback to greedy if k is 0 or less (should not happen with len > 0 check, but defensive)
                         best_safe_idx_in_filtered = np.argmax(safe_wdm)
                         selected_idx = safe_action_indices[best_safe_idx_in_filtered]
                    else:
                        # Find the indices of the top k distances among safe actions
                        top_k_indices_in_filtered = np.argsort(safe_wdm)[-actual_k:]
                        # Randomly choose one index from the top k
                        chosen_top_k_idx = np.random.choice(top_k_indices_in_filtered)
                        # Map back to the original action index
                        selected_idx = safe_action_indices[chosen_top_k_idx]
                else:
                    # Fallback to greedy as a safety measure
                    best_safe_idx_in_filtered = np.argmax(safe_wdm)
                    selected_idx = safe_action_indices[best_safe_idx_in_filtered]

            else:
                # No safe actions found, choose the action with the highest score (least unsafe)
                selected_idx = np.argmax(wdm_env)

            final_indices[env_idx] = selected_idx

        # Assign the computed indices to max_indices for the return statement
        max_indices = final_indices

        return torch.tensor(safe_mask).to(device), torch.from_numpy(max_indices).to(device), weighted_distance_min
