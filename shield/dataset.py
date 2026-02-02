import os
import logging
from collections import defaultdict
import pickle
import numpy as np
from tqdm import trange
from stable_baselines3.common.running_mean_std import RunningMeanStd
from copy import deepcopy

logger = logging.getLogger(__name__)

def collect_safety_gym_transitions(
    env,
    policy=None,
    num_episodes=100,
    prediction_horizon=1,
):
    """Collect transition data from safety gym environment.

    Args:
        env: The gymnasium environment instance
        num_episodes: Number of episodes to collect data from

    Returns:
        Dictionary mapping hidden parameters to lists of (state-action, stacked position_deltas) tuples.
        When prediction_horizon > 1, each y is a concatenation of per-step deltas in order.

    Raises:
        ValueError: If environment doesn't provide required information
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    
    obs, info = env.reset()
    slices = env.unwrapped.get_slices()
    transitions_X = defaultdict(list)
    transitions_Y = defaultdict(list)
    transitions = dict()

    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    def normalize_obs(obs):
        obs_ = deepcopy(obs)
        obs_ = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -1000, 1000)
        return obs_

    for episode in trange(num_episodes, desc="Collecting safety gym transitions"):
        obs, info = env.reset()
        hidden_parameter = tuple(info["hidden_parameters_features"])
        robot_obs = obs[slices['robot']]
        robot_pos = info["agent_pos"][:2]
        pending = []
        done = False
        while not done:
            prev_robot_obs = robot_obs.copy()
            prev_robot_pos = robot_pos.copy()

            if policy is None:
                action = env.action_space.sample()
            else:
                # Order matters
                obs_rms.update(obs)
                obs = normalize_obs(obs)
                action, _ = policy.predict(obs)
                
            obs, _, _, terminated, truncated, info = env.step(action)
                            
            done = truncated | terminated
            robot_obs = obs[slices['robot']]
            robot_pos = info["agent_pos"][:2]
            x = np.concatenate([prev_robot_obs, action])
            step_delta = robot_pos - prev_robot_pos

            if pending:
                for sample in pending:
                    sample["deltas"].append(step_delta)
            pending.append({"x": x, "deltas": [step_delta]})

            if prediction_horizon == 1:
                transitions_X[hidden_parameter].append(x)
                transitions_Y[hidden_parameter].append(step_delta)
                pending.clear()
            else:
                ready = []
                keep = []
                for sample in pending:
                    if len(sample["deltas"]) >= prediction_horizon:
                        ready.append(sample)
                    else:
                        keep.append(sample)
                pending = keep
                for sample in ready:
                    transitions_X[hidden_parameter].append(sample["x"])
                    transitions_Y[hidden_parameter].append(
                        np.concatenate(sample["deltas"][:prediction_horizon])
                    )

    transitions = {
        'X': transitions_X,
        'Y': transitions_Y
    }
    env.close()
    return transitions

def collect_safety_velocity_transitions(
    env,
    policy=None,
    num_episodes=100,
    prediction_horizon=1,
):
    """Collect transition data from safety gym environment.

    Args:
        env: The gymnasium environment instance
        num_episodes: Number of episodes to collect data from

    Returns:
        Dictionary mapping hidden parameters to lists of (state-action, stacked position_deltas) tuples.
        When prediction_horizon > 1, each y is a concatenation of per-step deltas in order.

    Raises:
        ValueError: If environment doesn't provide required information
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    
    obs, info = env.reset()
    robot_slices = info['robot']
    hidden_param_slices = info['hidden_param_slices']
    transitions_X = defaultdict(list)
    transitions_Y = defaultdict(list)
    transitions = dict()

    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    def normalize_obs(obs):
        obs_ = deepcopy(obs)
        obs_ = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -1000, 1000)
        return obs_
    
    
    for episode in trange(num_episodes, desc="Collecting safety velocity transitions"):
        obs, info = env.reset()
        hidden_parameter = tuple(info["hidden_parameters_features"])
        
        done = False
        velocity = None
        robot_obs = obs[robot_slices]
        pending = []
        while not done:
            prev_robot_obs = robot_obs.copy()
            
            if policy is None:
                action = env.action_space.sample()
            else:
                # Order matters
                obs_rms.update(obs)
                obs = normalize_obs(obs)
                action, _ = policy.predict(obs)
                
            obs, _, _, terminated, truncated, info = env.step(action)
            # last episode velocity.
            velocity = info.get('x_velocity', velocity)
                            
            done = truncated | terminated
            robot_obs = obs[robot_slices]
            # to simplify, we directly predict the velocity
            x = np.concatenate([prev_robot_obs, action])
            step_velocity = np.array([velocity])

            if pending:
                for sample in pending:
                    sample["vals"].append(step_velocity)
            pending.append({"x": x, "vals": [step_velocity]})

            if prediction_horizon == 1:
                transitions_X[hidden_parameter].append(x)
                transitions_Y[hidden_parameter].append(step_velocity)
                pending.clear()
            else:
                ready = []
                keep = []
                for sample in pending:
                    if len(sample["vals"]) >= prediction_horizon:
                        ready.append(sample)
                    else:
                        keep.append(sample)
                pending = keep
                for sample in ready:
                    transitions_X[hidden_parameter].append(sample["x"])
                    transitions_Y[hidden_parameter].append(
                        np.concatenate(sample["vals"][:prediction_horizon])
                    )

    transitions = {
        'X': transitions_X,
        'Y': transitions_Y
    }
    env.close()
    return transitions

def save_transitions(
    train_transitions,
    eval_transitions,
    env_id,
    default_path="."
):
    """Save collected transitions to pickle files.

    Args:
        train_transitions: Training data transitions
        eval_transitions: Evaluation data transitions
        env_id: Environment identifier string
        default_path: Directory path to save files

    Raises:
        ValueError: If env_id is invalid
        OSError: If directory creation or file writing fails
    """
    if not env_id or not isinstance(env_id, str):
        raise ValueError("Invalid env_id provided")

    train_filename = f"{env_id.split('-')[0][:-1]}_train_transitions.pkl"
    eval_filename = f"{env_id.split('-')[0][:-1]}_eval_transitions.pkl"
    os.makedirs(default_path, exist_ok=True)

    train_path = os.path.join(default_path, train_filename)
    eval_path = os.path.join(default_path, eval_filename)

    with open(train_path, "wb") as f:
        pickle.dump(train_transitions, f)
    print(f"Training transitions saved to {train_filename}")

    with open(eval_path, "wb") as f:
        pickle.dump(eval_transitions, f)
    print(f"Evaluation transitions saved to {eval_filename}")
