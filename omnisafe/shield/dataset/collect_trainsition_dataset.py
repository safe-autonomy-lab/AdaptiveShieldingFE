from typing import Dict, List, Tuple, Any, DefaultDict, Optional
import os
from collections import defaultdict
import pickle
import numpy as np
from numpy.typing import NDArray
from tqdm import trange
from gymnasium import Env
from stable_baselines3 import PPO


def collect_safety_gym_transitions(
    env: Env,
    policy: Optional[PPO] = None,
    num_episodes: int = 100,
    position_only_prediction: bool = False,
    use_mo_transitions: bool = False
) -> DefaultDict[Tuple[float, ...], List[Tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    """Collect transition data from safety gym environment.

    Args:
        env: The gymnasium environment instance
        num_episodes: Number of episodes to collect data from

    Returns:
        Dictionary mapping hidden parameters to lists of (state-action, position_delta) tuples

    Raises:
        ValueError: If environment doesn't provide required information
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    transitions: DefaultDict[Tuple[float, ...], List[Tuple[NDArray[np.float64], NDArray[np.float64]]]] = defaultdict(list)
    trajectories: DefaultDict[Tuple[float, ...], List[Tuple[NDArray[np.float64], NDArray[np.float64]]]] = defaultdict(list)
    mo_transitions: DefaultDict[Tuple[float, ...], List[Tuple[NDArray[np.float64], NDArray[np.float64]]]] = defaultdict(list)
    
    for episode in trange(num_episodes, desc="Collecting safety gym transitions"):
        obs, info = env.reset()
        
        if "agent_pos" not in info or "hidden_parameters" not in info:
            raise ValueError("Environment info missing required fields")

        obs_dims = env.get_obs_dims()
        robot_dim = obs_dims["robot"]
        # We assume that there is no static obstacles
        if use_mo_transitions:
            goal_dim = obs_dims["goal"]
            mo_dim = obs_dims["gremlins"]
            index_cut = slice(robot_dim + goal_dim, robot_dim + goal_dim + mo_dim)
        
        robot_pos = info["agent_pos"][:2]
        robot_mat = info["agent_mat"]
        hidden_parameter = tuple(info["hidden_parameters"])
        robot_obs = obs[:robot_dim]
        # This will be triggered only if moving obstalces exist
        if not position_only_prediction:
            # This should be 2 + 9 + 12 + 16 = 39 dimensions
            goal_dim = obs_dims["goal"]
            obs_pos_concat = np.concatenate([robot_pos, robot_mat, obs[: robot_dim + goal_dim]])
        else:
            obs_pos_concat = robot_obs

        if "gremlins_pos" in info:
            mo_pos = info["gremlins_pos"]
            trajectories[hidden_parameter].append(mo_pos)
            # We need to predict this information based on the current agent position, rotation matrix and the mo position
            # Dimnesion of x will be 2 + 9 + 2 * number of mo, if number of mo is 8, then it will be 2 + 9 + 16 = 27
            if use_mo_transitions:
                x = np.concatenate([robot_pos, robot_mat, mo_pos.flatten()])
                y = obs[index_cut]
                mo_transitions[hidden_parameter].append((x, y))
        
        done = False
        while not done:
            prev_robot_obs = robot_obs.copy()
            prev_robot_pos = robot_pos.copy()
            prev_robot_mat = robot_mat.copy()
            prev_obs_pos_concat = obs_pos_concat.copy()
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs)
                
            obs, _, _, terminated, truncated, info = env.step(action)
            
            if "agent_pos" not in info:
                raise ValueError("Environment step info missing agent_pos")
                            
            done = truncated | terminated
            robot_obs = obs[:robot_dim]
            robot_pos = info["agent_pos"][:2]
            robot_mat = info["agent_mat"]
             
            if "gremlins_pos" in info:
                mo_pos = info["gremlins_pos"]
                trajectories[hidden_parameter].append(mo_pos)
                if use_mo_transitions:
                    x = np.concatenate([robot_pos, robot_mat, mo_pos.flatten()])
                    y = obs[index_cut]
                    mo_transitions[hidden_parameter].append((x, y))
                
            if position_only_prediction:
                x = np.concatenate([prev_robot_obs, action])
                y = robot_pos - prev_robot_pos
                transitions[hidden_parameter].append((x, y))
            else:
                obs_pos_concat = np.concatenate([robot_pos, robot_mat, obs[: robot_dim + goal_dim]])
                x = np.concatenate([obs_pos_concat, action])
                y = obs_pos_concat - prev_obs_pos_concat
                transitions[hidden_parameter].append((x, y))

    env.close()
    return transitions, trajectories, mo_transitions

def collect_safety_velocity_gym_transitions(
    env: Env,
    num_episodes: int = 100
) -> DefaultDict[Tuple[float, ...], List[Tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    """Collect transition data including velocity from safety gym environment.

    Args:
        env: The gymnasium environment instance
        num_episodes: Number of episodes to collect data from

    Returns:
        Dictionary mapping hidden parameters to lists of (state-action, position_velocity_delta) tuples

    Raises:
        ValueError: If environment doesn't provide required information
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    transitions: DefaultDict[Tuple[float, ...], List[Tuple[NDArray[np.float64], NDArray[np.float64]]]] = defaultdict(list)
    
    for episode in trange(num_episodes, desc="Collecting safety gym transitions"):
        obs, info = env.reset()
        
        if not all(key in info for key in ["agent_pos", "agent_vel", "obs_dims", "hidden_parameters"]):
            raise ValueError("Environment info missing required fields")

        agent_pos = info["agent_pos"]
        robot_pos = np.array([agent_pos[0]]) if len(agent_pos) > 1 else agent_pos
        robot_vel = info["agent_vel"]
        robot_dim = info["obs_dims"]["robot"] if isinstance(info["obs_dims"], dict) else info["obs_dims"]
        hidden_parameter = tuple(info["hidden_parameters"])
        robot_obs = obs[:robot_dim]
        
        done = False
        while not done:
            prev_robot_obs = robot_obs.copy()
            prev_robot_pos = robot_pos.copy()
            prev_robot_vel = robot_vel.copy()
            action = env.action_space.sample()
            obs, _, _, terminated, truncated, info = env.step(action)
            
            if not all(key in info for key in ["agent_pos", "agent_vel"]):
                raise ValueError("Environment step info missing required fields")
                
            robot_obs = obs[:robot_dim]
            done = truncated | terminated
            
            agent_pos = info["agent_pos"]
            robot_pos = np.array([agent_pos[0]]) if len(agent_pos) > 1 else agent_pos
            robot_vel = info["agent_vel"]
            
            x = np.concatenate([prev_robot_obs, action])
            y = robot_pos - prev_robot_pos
            z = robot_vel - prev_robot_vel
            target = np.concatenate([y, [z]])
            transitions[hidden_parameter].append((x, target))

    env.close()
    return transitions

def save_transitions(
    train_transitions: Dict[Any, List[Any]],
    eval_transitions: Dict[Any, List[Any]],
    env_id: str,
    default_path: str = "."
) -> None:
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
