import sys
import os
import pickle
import logging

import torch
from stable_baselines3 import PPO
import numpy as np
from envs.safety_gymnasium.configuration import EnvironmentConfig
from shield.dataset import collect_safety_gym_transitions, collect_safety_velocity_transitions
from envs import make

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def _transitions_filename(env_info, data_purpose, prediction_horizon, position_only_prediction):
    suffix = "_transitions"
    if position_only_prediction:
        suffix += "_position_only"
    suffix += f"_h{prediction_horizon}"
    return f"{env_info}_{data_purpose}{suffix}.pkl"

def save_data(
    transitions,
    env_info,
    data_purpose="train",
    transition_save=False,
    position_only_prediction=False,
    prediction_horizon=1,
):
    if transition_save:
        os.makedirs('saved_files/env_transitions', exist_ok=True)
        filename = _transitions_filename(
            env_info,
            data_purpose,
            prediction_horizon,
            position_only_prediction,
        )
        with open(f'saved_files/env_transitions/{filename}', 'wb') as f:
            pickle.dump(transitions, f)

def load_data(env_name, data_purpose, position_only_prediction=False, prediction_horizon=1):
    filename = _transitions_filename(
        env_name,
        data_purpose,
        prediction_horizon,
        position_only_prediction,
    )
    with open(f'saved_files/env_transitions/{filename}', 'rb') as f:
        env_transitions = pickle.load(f)

    return env_transitions

def collect_transition(env_id: str, nbr_of_episodes: int, use_trained_policy: bool, prediction_horizon: int = 1):
    env_config = EnvironmentConfig()
    env_info = env_id.split('-')[0]
    env_config.FIX_HIDDEN_PARAMETERS = False
    
    # If we want to use the trained policy to collect transition dynamics, we need to load the trained policy first
    if use_trained_policy:
        log_dir = f"./trained_policies_for_collection/{env_id}/"
        model = PPO.load(os.path.join(log_dir, "best_model"))
    # If we don't want to use the trained policy, we can use a random policy
    else:
        model = None

    # We collect transition dynamics for both training and evaluation
    for data_purpose in ['train', 'eval']:
        episodes = nbr_of_episodes
        if data_purpose == 'eval':
            # For evaluation, we only use 20 perecent of the episodes during training
            episodes = nbr_of_episodes // 5 
            env_config.IS_OUT_OF_DISTRIBUTION = True
        
        if 'Velocity' in env_info:
            env_config.MIN_MULT, env_config.MAX_MULT = (0.7, 1.3)
            env = make(env_info + '-v0', env_config=env_config, render_mode='rgb_array')
            transitions = collect_safety_velocity_transitions(env, policy=model, num_episodes=episodes, prediction_horizon=prediction_horizon)
        else:
            env_config.MIN_MULT, env_config.MAX_MULT = (0.3, 1.7)
            env = make(env_info + '-v0', env_config=env_config, render_mode='rgb_array')
            transitions = collect_safety_gym_transitions(env, policy=model, num_episodes=episodes, prediction_horizon=prediction_horizon)
        save_data(
            transitions,
            env_info,
            data_purpose=data_purpose,
            transition_save=True,
            prediction_horizon=prediction_horizon,
        )

if __name__ == "__main__":
    # Example: python 1.collect_transition.py SafetyPointGoal1-v0 1000 0 1
    # Example: python 1.collect_transition.py SafetyHalfCheetahVelocity-v0 1000 0 2
    env_id = sys.argv[1]
    nbr_of_episodes = int(sys.argv[2])
    use_trained_policy = int(sys.argv[3])
    prediction_horizon = int(sys.argv[4])
    collect_transition(env_id=env_id, nbr_of_episodes=nbr_of_episodes, use_trained_policy=use_trained_policy, prediction_horizon=prediction_horizon)
