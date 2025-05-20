
import os
import pickle
import envs
from omnisafe.shield.dataset.collect_trainsition_dataset import collect_safety_gym_transitions
from envs import make
import sys
import torch
from stable_baselines3 import PPO
import numpy as np
from configuration import EnvironmentConfig

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def save_data(train_transitions, trajectories, mo_transitions, env_info, data_purpose: str = 'train', trainsition_save: bool = False, trajectory_save: bool = False, mo_transitions_save: bool = False):
    if trainsition_save:
        os.makedirs('saved_files/env_transitions', exist_ok=True)
        with open(f'saved_files/env_transitions/{env_info}_{data_purpose}_transitions.pkl', 'wb') as f:
            pickle.dump(train_transitions, f)

    if trajectory_save:
        os.makedirs('saved_files/env_trajectories', exist_ok=True)
        with open(f'saved_files/env_trajectories/{env_info}_{data_purpose}_trajectories.pkl', 'wb') as f:
            pickle.dump(trajectories, f)

    if mo_transitions_save:
        os.makedirs('saved_files/env_mo_transitions', exist_ok=True)
        with open(f'saved_files/env_mo_transitions/{env_info}_{data_purpose}_mo_transitions.pkl', 'wb') as f:
            pickle.dump(mo_transitions, f)

def load_data(env_name, data_purpose):
    with open(f'saved_files/env_transitions/{env_name}_{data_purpose}_transitions.pkl', 'rb') as f:
        env_transitions = pickle.load(f)

    with open(f'saved_files/env_trajectories/{env_name}_{data_purpose}_trajectories.pkl', 'rb') as f:
        env_trajectories = pickle.load(f)

    return env_transitions, env_trajectories

env_id = sys.argv[1]
env_info = env_id.split('-')[0]
nbr_of_episodes = int(sys.argv[2])
use_trained_policy = int(sys.argv[3])
use_mo_transitions = False
trajectory_save = False
# Since we only use one step prediction, we don't have to predict the whole state, position would be enough for our purpose
position_only_prediction = True
env_config = EnvironmentConfig()

# If we want to use the trained policy to collect transition dynamics, we need to load the trained policy
if use_trained_policy:
    log_dir = f"./trained_policies_for_collection/{env_id}/"
    model = PPO.load(os.path.join(log_dir, "ppo_policy"))
# If we don't want to use the trained policy, we can use a random policy
else:
    model = None

# We collect transition dynamics for both training and evaluation
for data_purpose in ['train', 'eval']:
    env = make(env_id[:-1] + str(0), env_config=env_config)
    if data_purpose == 'eval':
        # For evaluation, we only use 20 perecent of the episodes during training
        nbr_of_episodes = nbr_of_episodes // 5 
    env.set_seed(seed)
    transitions, trajectories, mo_transitions = collect_safety_gym_transitions(env, policy=model, num_episodes=nbr_of_episodes, position_only_prediction=position_only_prediction, use_mo_transitions=use_mo_transitions)
    if len(trajectories) > 0:
        trajectory_save = True
        
    save_data(transitions, trajectories, mo_transitions, env_info, data_purpose=data_purpose, trainsition_save=True, trajectory_save=trajectory_save, mo_transitions_save=use_mo_transitions)