import torch
import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict, deque
from .conformal_prediction import ConformalPrediction
from .run_utils.train_util import create_train_state
from .dynamic_predictor.function_encoder import FunctionEncoder
from .run_utils.load_utils import call_transition_dataset
from configuration import DynamicPredictorConfig, EnvironmentConfig
from .util import dict_to_dataclass


class BaseShield(ConformalPrediction):
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
        env_info: str,
        dynamic_predictor_cfgs: DynamicPredictorConfig,
        env_config: EnvironmentConfig,
        sampling_nbr: int,  
        prediction_horizon: int,
        static_threshold: float = 0.3,
        use_hidden_param: bool = True,
        use_online_update: bool = False,   
        window_size: int = 100,
        significance_level: float = 0.1,
        vector_env_nums: int = 1,
        safety_bonus: float = 1.0,
        safety_measure_discount: float = 0.9,
        **kwargs
    ) -> None:  
        # Initialize both parent classes
        robot_type = 'Point' if 'Point' in env_info else 'Car'
        ConformalPrediction.__init__(self, window_size, significance_level)
        
        # Representation of underlying contexts
        
        self.safety_bonus = torch.tensor(safety_bonus).float()
        self.env_info = env_info
        self.prediction_horizon = prediction_horizon
        self.safety_measure_discount = safety_measure_discount
        if isinstance(dynamic_predictor_cfgs, dict):
            self.dp_cfgs, self.env_cfgs = dict_to_dataclass(dynamic_predictor_cfgs, DynamicPredictorConfig), dict_to_dataclass(env_config, EnvironmentConfig)
        else:
            self.dp_cfgs, self.env_cfgs = dynamic_predictor_cfgs, env_config

        self.use_fe_representation = self.env_cfgs.USE_FE_REPRESENTATION

        dp_dataset = call_transition_dataset(env_info)
        self.dynamic_predictor = self.create_function_encoder(self.dp_cfgs, dp_dataset)

        self.dp_model = self.load_dynamic_predictor()
        self.max_history = self.dp_cfgs.MAX_HISTORY

        self.dp_state = self.dp_model.get_state()

        self.vector_env_nums = vector_env_nums
        self.sampling_nbr = sampling_nbr
        
        self.static_threshold = static_threshold
        self.use_hidden_param = use_hidden_param
        self._set_conformal_thresholds()

        self.dp_input_size = self.dp_model.input_size
        self.dp_output_size = self.dp_model.output_size
        self.action_dim = 2

        self.is_object_checked = False
    
        w = list(self.dp_model.dataset.experiences.keys())[0]
        w = np.array(w).reshape(1, -1)
        
        self.hidden_param_dim = w.shape[1]
        self.ws = np.empty((self.vector_env_nums, self.hidden_param_dim))
        
        for i in range(self.vector_env_nums):
            self.set_hyperparameter(w, i)
        
        self.scale = np.linalg.norm(self.ws, axis=1)[0] if self.use_hidden_param else 0.
        self.ws_representation = self.ws.copy() * self.scale
        self.default_ws = self.ws.copy()

        self.N = 1
        self.use_online_update = use_online_update
        
        self.robot_actual_history = deque(maxlen=self.window_size)
        self.robot_predictions_history = deque(maxlen=self.window_size)

        self.reset()

    def update_ws(self, new_ws):
        self.ws = new_ws

    def online_update_ws(self, new_ws):
        self.ws = (self.N / (self.N + 1)) * self.ws + (1 / (self.N + 1)) * new_ws

    def load_dynamic_predictor(self):
        state = self.dynamic_predictor.get_state()
        save_file_path = "./saved_files/trained_dynamic_predictor"
        save_file_name = f"{self.env_info}_{self.dp_cfgs.EPOCH}.pkl"
        self.dynamic_predictor.load_train_state(save_file_name, default_path=save_file_path, state_template=state)
        self.dynamic_predictor.compute_init_coefficients()
        print("Dynamic predictor loaded successfully")
        return self.dynamic_predictor
    
    def create_function_encoder(self, model_config, dataset):
        rng = jax.random.PRNGKey(0)
        LEARNING_RATE = model_config.LEARNING_RATE
        HIDDEN_SIZE = model_config.HIDDEN_SIZE
        N_BASIS = model_config.N_BASIS
        AVERAGE_FUNCTION = model_config.AVERAGE_FUNCTION
        LEAST_SQUARES = model_config.LEAST_SQUARES
        USE_ATTENTION = model_config.USE_ATTENTION
        MAX_HISTORY = model_config.MAX_HISTORY
        INPUT_DIM = dataset.input_size[0]
        OUTPUT_DIM = dataset.output_size[0]
        LEARNING_DOMAIN = model_config.LEARNING_DOMAIN
        function_encoder = FunctionEncoder(INPUT_DIM, OUTPUT_DIM, activation='relu', n_basis=N_BASIS, hidden_size=HIDDEN_SIZE, least_squares=LEAST_SQUARES, average_function=AVERAGE_FUNCTION, use_attention=USE_ATTENTION, history_length=MAX_HISTORY)
        state = create_train_state(rng, function_encoder, learning_rate=LEARNING_RATE, input_size=INPUT_DIM, learning_domain=LEARNING_DOMAIN, output_size=OUTPUT_DIM)
        function_encoder.set_state(state)
        function_encoder.set_dataset(dataset)
        function_encoder.set_train_step()
        return function_encoder

    def save_state(self, state) -> None:
        """Save current actor state"""
        self.actor_state = state

    def set_hyperparameter(self, w, i):
        self.ws[i] = w

    def learn_dynamics(self, epochs: int = 60, logger=None) -> None:
        """Train the dynamic predictor"""
        self.dp_model.train_model(epochs, logger=logger)

    def extend_dataset(self, w):
        self.dp_model.extend_dataset(w)
    
    def prepare_dp_input(self, obs, agent_pos, agent_mat, device):
        self.prev_dp_input = self.dp_input
        self.dp_input = obs[:, :self.dp_input_size - self.action_dim]

    def add_coefficients_to_obs(self, obs):
        obs[:, -self.hidden_param_dim:] = torch.from_numpy(np.array(self.ws_representation))
        return obs.float()
    
    def update_function_encoder_after_episode(self, w):
        self.dynamic_predictor.online_dataset_update(w)
    
    def compute_coefficients_fn(self):
        return self.dynamic_predictor.get_compute_coefficients_fn()

    def update_coefficients(self):
        self.dynamic_predictor.recalculate_coefficients()
    
    def add_episode_data_memory(self):
        data = jnp.concatenate([self.prev_dp_input, self.prev_action, self.dp_input], axis=-1)
        self.dynamic_predictor.add_episode_data_memory(data)
        example_x = data[:, :self.dp_input_size]
        example_y = data[:, self.dp_input_size:]

        if len(self.dynamic_predictor.episode_data_memory) == 1:
            w = self.dynamic_predictor.get_compute_coefficients_fn()(
                self.dynamic_predictor.get_state(), 
                example_x[None], 
                example_y[None]
            )
            self.set_hyperparameter(w)
            self.w_for_representation = w
        else:
            N = len(self.dynamic_predictor.episode_data_memory)
            online_w = self.dynamic_predictor.get_compute_coefficients_fn()(
                self.dynamic_predictor.get_state(),
                example_x[None],
                example_y[None]
            )
            w = (N - 1) / N * self.w + 1 / N * online_w
            self.set_hyperparameter(w)

    def update_robot_actual_history(self, robot_obs: np.ndarray):
        self.robot_actual_history.append(robot_obs)

    def reset(self) -> None:
        """Reset shield's internal state"""
        self.train_dynamics_in = defaultdict(list)
        self.train_dynamics_targs = defaultdict(list)
        self.predicted_states = []

        self.dynamic_predictor.set_episode_data_memory()
        # We need at least MAX_HISTORY of mo obs to predict future
        
        self.shield_triggered = False
        self.last_predicted_robot_obs = None
        
        self.prev_agent_pos = None
        self.agent_pos = None        
        self.prev_dp_input = None
        self.dp_input = None        
        self.prev_w = None
        self.prev_action = None
        self.action = None
        self.ws = self.default_ws
        self.ws_representation = self.default_ws
        self.N = 1

    