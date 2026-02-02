import torch
import numpy as np
from collections import defaultdict, deque
from .conformal_prediction import ConformalPrediction
from FunctionEncoder import FunctionEncoder
from .sensor import LidarSensorProcessor


class BaseShield(LidarSensorProcessor, ConformalPrediction):
    """
    Shield structure for RL algorithms. It takes a policy and observations to produce a safer policy.

    Attributes:
        scene: Scene information to efficiently use moving obstacles' history
        moving_obstacles_predictor: Predicts moving obstacles' future states
        dynamic_predictor: Predicts the dynamic state of the system
        sampling_nbr: Number of samples to use in prediction
        prediction_horizon: Number of steps to predict into future
        node_type: Type of node in the network
        threshold: Threshold for determining safe actions
        use_density: Whether to use density information
        default_safe_action: Whether to use default safe action
        load_safe_action: Whether to load a safe action
        selection_type: Type of selection method for safe actions
        action_clip: Maximum allowed action magnitude
    """

    def __init__(
        self,
        dynamic_predictor: FunctionEncoder,
        mo_predictor: FunctionEncoder,
        sampling_nbr: int,  
        prediction_horizon: int,
        window_size: int = 100,
        significance_level: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        idle_condition: int = 0,
        **kwargs
    ) -> None:  
        # Initialize both parent classes
        LidarSensorProcessor.__init__(self)
        ConformalPrediction.__init__(self, window_size, significance_level)
        self.device = device
        self.sampling_nbr = sampling_nbr
        self.prediction_horizon = prediction_horizon

        self.dynamic_predictor = dynamic_predictor
        self.mo_predictor = mo_predictor
        self.history_length = self.mo_predictor.history_length if self.mo_predictor is not None else 0
        self.idle_condition = idle_condition
        self.step_last_triggered = 0

        self.n_basis = dynamic_predictor.n_basis

        # This is used to store the robot actual history for conformal prediction
        self.robot_actual_history = deque(maxlen=self.window_size)
        self.mo_actual_history = deque(maxlen=self.window_size)
        self.robot_predictions_history = deque(maxlen=self.window_size)
        self.mo_predictions_history = deque(maxlen=self.window_size)

        self.dynamics_input_examples = []
        self.dynamics_target_examples = []

        self.mo_input_examples = []
        self.mo_target_examples = []

        # This is used to store the mo obs history for the future prediction
        self.mo_obs_history = deque(maxlen=self.history_length)
        self.coeffs_for_dynamics_prediction = None
        self.normalized_coeffs_for_dynamics_prediction = None
        self.coeffs_for_mo_prediction = None
        self.normalized_coeffs_for_mo_prediction = None

        self.reset()

    def prepare_dp_input(self, agent_obs, agent_pos, agent_mat, device):
        self.prev_dp_input = self.dp_input
        # th_agent_pos = torch.from_numpy(agent_pos).float().to(device)
        # th_agent_mat = torch.from_numpy(agent_mat).float().to(device)
        # self.dp_input = torch.cat([agent_obs, th_agent_pos, th_agent_mat], axis=-1).detach().to(device)
        self.dp_input = agent_obs
        self.dp_y = torch.from_numpy(agent_pos[:, :2]).float().to(device)
        
    def add_coefficients_to_obs(self, obs: torch.tensor):
        obs[:, -self.n_basis:] = self.normalized_coeffs_for_dynamics_prediction
        return obs.float()
    
    def update_mo_obs_history(self, mo_obs: torch.tensor, update_coeffs: bool = False):
        mo_last_obs = mo_obs.reshape(-1, 2)[:, np.newaxis, np.newaxis, :] # -> extend history dimension, so the dimenision will be (n_env * number of moving obstacles, 1, history_length, position)
        self.mo_last_obs = torch.from_numpy(mo_last_obs).float().to(self.device)
        if update_coeffs and len(self.mo_obs_history) == self.history_length: 
            self.mo_example = np.concatenate(list(self.mo_obs_history), axis=-2)
            mo_x = torch.from_numpy(self.mo_example).float().to(self.device)
            self.coeffs_for_mo_prediction, _ = self.mo_predictor.compute_representation(mo_x, self.mo_last_obs[:, :, 0, :])

        self.mo_obs_history.append(mo_last_obs)
        self.last_mo_x = torch.from_numpy(np.concatenate(list(self.mo_obs_history), axis=-2)).float().to(self.device)
    
    def update_mo_actual_history(self, mo_obs: np.ndarray):
        self.mo_actual_history.append(mo_obs)        
    
    def update_robot_actual_history(self, robot_obs: np.ndarray):
        self.robot_actual_history.append(robot_obs)

    def update_coeffs_for_dynamics_prediction(self):
        assert len(self.dynamics_input_examples) > 0
        example_x = torch.stack(self.dynamics_input_examples, axis=0).transpose(0, 1) # (num_env, nbr_of_examples, input_size)
        example_y = torch.stack(self.dynamics_target_examples, axis=0).transpose(0, 1) # (num_env, nbr_of_examples, output_size)
        
        self.coeffs_for_dynamics_prediction, _ = self.dynamic_predictor.compute_representation(example_x, example_y)

    def get_mo_predictions(self):
        if len(self.mo_obs_history) < self.history_length:
            return None
        else:
            with torch.no_grad():
                prediction = self.mo_predictor.predict(x=self.last_mo_x, coeffs=self.coeffs_for_mo_prediction, prediction_horizon=self.prediction_horizon)
        return prediction

    def reset(self) -> None:
        """Reset shield's internal state"""
        self.dynamics_input_examples = []
        self.dynamics_target_examples = []

        self.mo_input_examples = []
        self.mo_target_examples = []

        self.predicted_states = []
        # We need at least MAX_HISTORY of mo obs to predict future
        
        self.shield_triggered = False
        self.last_predicted_robot_obs = None
        self.last_predicted_mo_obs = None
        
        self.prev_agent_pos = None
        self.agent_pos = None        
        self.prev_dp_input = None
        self.dp_input = None        
        self.prev_action = None
        self.action = None

        self.mo_example = None
        self.mo_last_obs = None

    