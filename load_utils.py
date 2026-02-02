"""Script to load and evaluate a saved TRPO model from OmniSafe."""
import json
from typing import Dict, Tuple

import numpy as np
import torch
from gymnasium.spaces import Box
from omnisafe.models import ActorCritic
from omnisafe.models.actor_critic.constraint_actor_q_and_v_critic import ConstraintActorQAndVCritic
from omnisafe.utils.config import Config, ModelConfig
from omnisafe.envs.wrapper import Normalizer, ObsNormalize



def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def create_model_config(config_dict: Dict) -> ModelConfig:
    """Create a ModelConfig object from a configuration dictionary.
    
    Args:
        config_dict: The configuration dictionary
        
    Returns:
        The model configuration object
    """
    model_cfg = ModelConfig()
    model_cfg.actor = Config()
    model_cfg.critic = Config()
    
    # Set actor configuration
    model_cfg.actor.hidden_sizes = config_dict["model_cfgs"]["actor"]["hidden_sizes"]
    model_cfg.actor.activation = config_dict["model_cfgs"]["actor"]["activation"]
    model_cfg.actor.lr = config_dict["model_cfgs"]["actor"]["lr"]
    
    # Set critic configuration
    model_cfg.critic.hidden_sizes = config_dict["model_cfgs"]["critic"]["hidden_sizes"]
    model_cfg.critic.activation = config_dict["model_cfgs"]["critic"]["activation"]
    model_cfg.critic.lr = config_dict["model_cfgs"]["critic"]["lr"]
    
    # Set other model configurations
    model_cfg.weight_initialization_mode = config_dict["model_cfgs"]["weight_initialization_mode"]
    model_cfg.actor_type = config_dict["model_cfgs"]["actor_type"]
    model_cfg.linear_lr_decay = config_dict["model_cfgs"]["linear_lr_decay"]
    model_cfg.exploration_noise_anneal = config_dict["model_cfgs"]["exploration_noise_anneal"]
    model_cfg.std_range = config_dict["model_cfgs"]["std_range"]
    return model_cfg


def load_model(
    model_path: str,
    config: Dict,
    env,
    training_steps: int = 2_000_000,
    device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
    extra_obs_dim: int = 0,
) -> Tuple[ActorCritic, Dict]:
    """Load the saved TRPO model and its configuration.
    
    Args:
        model_path: Path to the saved model weights
        config: The model configuration dictionary
        env: The environment instance
        
    Returns:
        Tuple containing the loaded model and its configuration
    """
    device = torch.device(device)
    # Load the saved model weights with CPU
    checkpoint = torch.load(model_path, map_location=device)
    # for key, we ahve 'pi' and 'obs normalizer'
    policy_state_dict = checkpoint["pi"]
    
    # Create model configuration
    model_cfg = create_model_config(config)
    action_space = env.action_space
    if 'Saute' in config['algo'] or 'Simmer' in config['algo']:
        safety_budget = (
            config['algo_cfgs']['safety_budget']
            * (1 - config['algo_cfgs']['saute_gamma']**config['algo_cfgs']['max_ep_len'])
            / (1 - config['algo_cfgs']['saute_gamma'])
            / config['algo_cfgs']['max_ep_len']
            * torch.ones(1)
        )
        config['safety_budget'] = safety_budget

    observation_space = env.observation_space
    if extra_obs_dim > 0:
        observation_space = Box(
            low=np.hstack((env.observation_space.low, [-np.inf] * extra_obs_dim)),
            high=np.hstack((env.observation_space.high, [np.inf] * extra_obs_dim)),
            shape=(env.observation_space.shape[0] + extra_obs_dim,),
        )
    if 'Saute' in config['algo'] or 'Simmer' in config['algo']:
        observation_space = Box(
            low=np.hstack((env.observation_space.low, -np.inf)),
            high=np.hstack((env.observation_space.high, np.inf)),
            shape=(env.observation_space.shape[0] + 1,),
        )

    # If checkpoint expects a different obs dim, override to match the saved model.
    obs_dim_from_ckpt = None
    for key, value in policy_state_dict.items():
        if value.ndim == 2:
            obs_dim_from_ckpt = value.shape[1]
            break
    if obs_dim_from_ckpt is not None and observation_space.shape[0] != obs_dim_from_ckpt:
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim_from_ckpt,),
            dtype=np.float32,
        )
    
    # Create model with the same architecture
    model = ConstraintActorQAndVCritic(
        obs_space=observation_space,
        act_space=action_space,
        model_cfgs=model_cfg,
        epochs=config["train_cfgs"]["epochs"],
    ).to(device)  # Explicitly move to CPU
    
    # Load the weights into the model
    model.actor.load_state_dict(policy_state_dict)    
    # Set up observation normalization if available
    if "obs_normalizer" in checkpoint:
        state = checkpoint["obs_normalizer"]
        if "mean" in state and "var" in state:
            expected = env.observation_space.shape[0]
            if state["mean"].shape[0] != expected:
                raise RuntimeError(
                    f"obs_normalizer dim mismatch: checkpoint={state['mean'].shape[0]} env={expected}. "
                    "Ensure eval obs space matches training (including FE coefficients)."
                )
        normalizer = Normalizer(env.observation_space.shape).to(device)
        normalizer.count = torch.tensor(training_steps).to(device)
        normalizer.load_state_dict(state)
        env = ObsNormalize(env, norm=normalizer, device=device)

    model.eval()
    return model, env, config

def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
