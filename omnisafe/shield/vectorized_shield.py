from typing import Dict, Tuple
import torch
import numpy as np
from stable_baselines3.common.logger import configure
from .base_shield import BaseShield
from .util import compute_min_distance_batch_numpy
from configuration import DynamicPredictorConfig, EnvironmentConfig
from omnisafe.envs.wrapper import Normalizer

# Configure logging
LOGGER = configure("shields_debug", [])
t2numpy = lambda x: x.cpu().detach().numpy()


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
        env_info: str,
        dynamic_predictor_cfgs: DynamicPredictorConfig,
        env_config: EnvironmentConfig,
        sampling_nbr: int,
        prediction_horizon: int,
        safety_measure_discount: float = 0.9,
        static_threshold: float = 0.3,
        use_hidden_param: bool = True,
        use_online_update: bool = False,
        window_size: int = 10,
        significance_level: float = 0.1,
        vector_env_nums: int = 1,
        safety_bonus: float = 1.0,
        gradient_scale: float = 2.0,
        warm_up_epochs: int = 10,
        use_acp: bool = True,
        **kwargs,
    ) -> None:
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
            vector_env_nums: Number of vectorized environments
            safety_bonus: Safety bonus
            gradient_scale: Gradient scale
            warm_up_epochs: Number of warm-up epochs
        """
        super().__init__(
            env_info=env_info,
            dynamic_predictor_cfgs=dynamic_predictor_cfgs,
            env_config=env_config,
            sampling_nbr=sampling_nbr,
            prediction_horizon=prediction_horizon,
            safety_measure_discount=safety_measure_discount,
            static_threshold=static_threshold,
            use_hidden_param=use_hidden_param,
            use_online_update=use_online_update,
            window_size=window_size,
            significance_level=significance_level,
            vector_env_nums=vector_env_nums,
            safety_bonus=safety_bonus,
            **kwargs,
        )
        self.gradient_scale = gradient_scale
        self.warm_up_epochs = warm_up_epochs
        self.use_acp = use_acp

    def _setup_environment_params(self) -> None:
        """Set up environment-specific parameters based on info dict."""
        self.range_limit = self.env_cfgs.RANGE_LIMIT
        self.radius = self.env_cfgs.RADIUS
        self.example_nbr = self.env_cfgs.EXAMPLE_NBR
        self.circle_env = self.env_cfgs.CIRCLE_ENV
        self.speed_limit = self.env_cfgs.SPEED_LIMIT
        self.general_env = self.env_cfgs.GENERAL_ENV
        # Assert that exactly one option is True
        assert sum([self.circle_env, self.speed_limit, self.general_env]) == 1, (
            "Exactly one of circle_env, speed_limit, or general_env must be True. "
            f"Got: circle_env={self.circle_env}, speed_limit={self.speed_limit}, "
            f"general_env={self.general_env}"
        )

    def _process_agent_information(self, info: Dict):
        """Process and normalize agent position from environment info."""
        agent_pos = np.stack(info['agent_pos'])
        agent_mat = np.stack(info['agent_mat'])
        if len(agent_pos.shape) == 1:
            agent_pos = agent_pos.reshape(1, -1)
        if len(agent_mat.shape) == 1:
            agent_mat = agent_mat.reshape(1, -1)
        
        return agent_pos[:, :2], agent_mat

    def _check_presafety_condition(self, info: Dict, enhanced_safety: float = 0.0):
        """Check if the current state satisfies safety conditions."""
        # General environment
        if self.general_env:
            unsafe_condition = info['min_distance'] < self.static_threshold + enhanced_safety
        # Circle environment
        elif self.circle_env:
            agent_pos = (
                np.array([pos[:2] for pos in info["agent_pos"]]) 
                if "agent_pos" in info else np.zeros((self.vector_env_nums, 2))
            )
            range_limit_check = np.abs(agent_pos) > self.range_limit - 0.125
            unsafe_condition = np.any(range_limit_check, axis=1, keepdims=False)
        return unsafe_condition

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
            - Hidden parameters
        """    
        
        agent_pos = (
            np.array([pos[:2] for pos in info["agent_pos"]]) 
            if "agent_pos" in info else np.zeros((batch_size, 2))
        )

        agent_mat = (
            np.array([mat for mat in info["agent_mat"]])
            if "agent_mat" in info else np.zeros((batch_size, 2))
        )

        hazards = (
            np.array([hazard for hazard in info["hazards_pos"]])
            if "hazards_pos" in info else np.array([])
        )
        pillars = (
            np.array([pillar for pillar in info["pillars_pos"]])
            if "pillars_pos" in info else np.array([])
        )
        gremlins = (
            np.array([gremlin for gremlin in info["gremlins_pos"]])
            if "gremlins_pos" in info else np.array([])
        )
        goal_pos = (
            np.array([goal for goal in info["goal_pos"]])
            if "goal_pos" in info else np.array([])
        )

        return agent_pos, agent_mat, goal_pos, hazards, pillars, gremlins

    def sample_safe_actions(
        self,
        dp_input: np.ndarray,
        agent_pos: np.ndarray,
        agent_mat: np.ndarray,
        goal_pos: np.ndarray,
        hazards: np.ndarray,
        pillars: np.ndarray,
        gremlins: np.ndarray,
        first_action: np.ndarray,
        policy, 
        dp_acp_region: float,
        device: str = 'cpu',
        selection_method: str = 'greedy',
        k: int = 1,
        normalizer: Normalizer = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Repeat dp_input to match first_action's first dimension
        # dp_input shape: (4, 12) -> (sampling_nbr, vector_num_env, robot_dim + action_dim)
        self.shield_triggered = True
        dp_input_repeated = np.tile(dp_input[np.newaxis, :, :], (self.sampling_nbr, 1, 1))
        # After transposing, the shape of obs_dp_input is (vector_num_env, sampling_nbr, robot_dim + action_dim)
        obs_dp_input = np.concatenate([dp_input_repeated, first_action], axis=-1).transpose(1, 0, 2)
        robot_predictions = self.dp_model.predict(self.ws, obs_dp_input)
        robot_predictions = robot_predictions.reshape(self.sampling_nbr, self.vector_env_nums, self.dp_output_size)
        robot_pos_predictions = robot_predictions[:, :, :2] + agent_pos[np.newaxis, :, :]
        mean_robot_pos_predictions = np.mean(robot_pos_predictions, axis=0)
        self.robot_predictions_history.append(mean_robot_pos_predictions)

        if self.use_acp:
            static_adjusted_threshold = self.static_threshold + dp_acp_region
            range_limit_adjusted_threshold = 0.1 + dp_acp_region
            
        else:
            static_adjusted_threshold = self.static_threshold
            range_limit_adjusted_threshold = 0.1
        
        # Circle Environment
        if self.circle_env:
            # for distance in circle, we want to minimize the distance to the boundary to keep safety
            abs_robot_pos = np.abs(robot_pos_predictions)
            distance2bounds = np.max(abs_robot_pos, axis=-1)
            min_indices = np.argmin(distance2bounds, axis=0)
            distance2bound = distance2bounds[min_indices, np.arange(len(min_indices))]
            safe_mask = distance2bound < self.range_limit - range_limit_adjusted_threshold
            return torch.tensor(safe_mask).to(device), torch.from_numpy(min_indices).to(device), distance2bound
        
        # weighted_distance_min will be safety measure for each prediction step
        weighted_distance_min = 0.0
        if not self.is_object_checked:
            self.is_gremlins = gremlins.shape[0] > 0
            self.is_hazards = hazards.shape[0] > 0
            self.is_pillars = pillars.shape[0] > 0
            self.is_object_checked = True

        for i in range(self.prediction_horizon):
            if i > 0:
                input_for_policy = robot_predictions.copy()
                if self.is_hazards:
                    vector2hazards = (vectorized_hazards - robot_pos_predictions[:, :, np.newaxis, :]).reshape(self.sampling_nbr, self.vector_env_nums, -1)
                    input_for_policy = np.concatenate([input_for_policy, vector2hazards], axis=-1)
            
                if self.is_pillars:
                    vector2pillars = (vectorized_pillars - robot_pos_predictions[:, :, np.newaxis, :]).reshape(self.sampling_nbr, self.vector_env_nums, -1)
                    input_for_policy = np.concatenate([input_for_policy, vector2pillars], axis=-1)
                
                input_for_policy = input_for_policy.reshape(self.sampling_nbr * self.vector_env_nums, -1)

                # 12 + 16 + 16 + 2 = 46
                hidden_infer = np.tile(np.array(self.ws_representation)[np.newaxis, :, :], (self.sampling_nbr, 1, 1)).reshape(self.sampling_nbr * self.vector_env_nums, -1)
                input_for_policy = np.concatenate([input_for_policy, hidden_infer], axis=-1)
                # We cut off the first 11 dimensions because they are the robot positions and matrix
                input_for_policy = torch.from_numpy(input_for_policy).to(device)[:, 11:]
                input_for_policy = normalizer.normalize(input_for_policy.float())
                # later_actions, _, _, _, _ = policy(input_for_policy)      
                policy_output = policy(input_for_policy)
                if isinstance(policy_output, tuple):
                    later_actions = policy_output[0]
                else:
                    later_actions = policy_output

                later_actions = later_actions.reshape(self.sampling_nbr, self.vector_env_nums, -1).cpu().detach().numpy()
                
                obs_dp_input= np.concatenate([robot_predictions, later_actions], axis=-1).transpose(1, 0, 2)
                robot_predictions = self.dp_model.predict(self.ws, obs_dp_input).transpose(1, 0, 2)
                robot_pos_predictions = robot_predictions[:, :, :2] + robot_pos_predictions
            
            vectorized_hazards = np.tile(hazards[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1)) if self.is_hazards else np.inf
            vectorized_pillars = np.tile(pillars[np.newaxis, :, :, :], (self.sampling_nbr, 1, 1, 1)) if self.is_pillars else np.inf
            distance2hazards = compute_min_distance_batch_numpy(vectorized_hazards, robot_pos_predictions) if self.is_hazards else np.inf
            distance2pillars = compute_min_distance_batch_numpy(vectorized_pillars, robot_pos_predictions) if self.is_pillars else np.inf
            distance2static = np.minimum(distance2pillars, distance2hazards)
            weighted_distance_min += self.safety_measure_discount ** i * distance2static
        
        unsafe_mask = distance2static <= static_adjusted_threshold
        safe_mask = ~unsafe_mask

        num_samples, num_envs = weighted_distance_min.shape
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