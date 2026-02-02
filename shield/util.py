import logging
import os
import pickle
import yaml
import torch.nn as nn
import numpy as np
import torch
from dataclasses import is_dataclass, fields
from typing import Type, TypeVar
from FunctionEncoder import FunctionEncoder
from FunctionEncoder.Wrapper import TimeSeriesWrapper
from FunctionEncoder.Dataset.TransitionDataset import TransitionDataset
from shield.dynamics_model.transformer import TransformerDynamics
from shield.dynamics_model.pem import PEM
from shield.dynamics_model.oracle import OracleMLP
from packaging import version  # comes with pip, setuptools, etc.
import sys

T = TypeVar('T')
logger = logging.getLogger(__name__)

def derivative_of(x: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """Calculate time derivatives using central difference method for interior points
    and forward/backward differences for endpoints.
    
    Args:
        x: Array of shape (num_pedestrians, history_length) containing position/velocity data
        dt: Time step size in seconds, defaults to 0.02s (50Hz)
        
    Returns:
        Array of same shape as input containing derivatives
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (num_pedestrians, history_length), got shape {x.shape}")
        
    num_peds, history_len = x.shape
    derivatives = np.zeros_like(x)
    
    if history_len < 2:
        return derivatives
    
    # Handle interior points using central difference
    derivatives[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (2 * dt)
    
    # Handle endpoints
    # Forward difference for first point
    derivatives[:, 0] = (x[:, 1] - x[:, 0]) / dt
    
    # Backward difference for last point
    derivatives[:, -1] = (x[:, -1] - x[:, -2]) / dt
    
    return derivatives

def dict_to_dataclass(data: dict, dataclass_type: Type[T]) -> T:
    """Convert a dictionary to a dataclass instance.
    
    Args:
        data: Dictionary containing configuration values
        dataclass_type: Type of dataclass to create
        
    Returns:
        Instance of the specified dataclass type
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type.__name__} is not a dataclass")
    
    # Create a dictionary of field values, handling case-insensitive matching
    field_values = {}
    data_lower = {k.upper(): v for k, v in data.items()}
    
    for field in fields(dataclass_type):
        field_name = field.name
        field_name_lower = field_name.upper()
        
        if field_name_lower in data_lower:
            field_values[field_name] = data_lower[field_name_lower]
            
    return dataclass_type(**field_values)


def compute_min_distance(objects_positions, agent_position):
    """Compute minimum distances between agents and objects using JAX.

    Args:
        objects_positions (jnp.ndarray): Array of object positions with shape 
            (sampling_nbr, num envs, num objects, 2)
        agent_position (jnp.ndarray): Array of agent positions with shape 
            (sampling_nbr, num envs, 2)

    Returns:
        jnp.ndarray: Minimum distances for each agent with shape (num envs, num objects)
    """
    agent_xy = agent_position.unsqueeze(-2)
    objects_xy = objects_positions
    distances = torch.norm(objects_xy - agent_xy, dim=-1)
    return torch.min(distances, dim=-1).values

# Returns the desired activation function by name
def get_activation(activation):
    if activation == "relu":
        return nn.relu
    if activation == "relu6":
        return nn.relu6
    elif activation == "tanh":
        return nn.tanh
    elif activation == "sigmoid":
        return nn.sigmoid
    else:
        raise ValueError(f"Unknown activation: {activation}")

def load_data(env_name, data_purpose, position_only_prediction: bool = False, prediction_horizon: int = 1):
    suffix = "_transitions"
    if position_only_prediction:
        suffix += "_position_only"
    suffix += f"_h{prediction_horizon}"
    filename = f"{env_name}_{data_purpose}{suffix}.pkl"
    path = f"saved_files/env_transitions/{filename}"
    try:
        with open(path, 'rb') as f:
            env_transitions = pickle.load(f)
    except FileNotFoundError:
        fallback_suffix = "_transitions_position_only" if position_only_prediction else "_transitions"
        fallback_filename = f"{env_name}_{data_purpose}{fallback_suffix}.pkl"
        with open(f"saved_files/env_transitions/{fallback_filename}", 'rb') as f:
            env_transitions = pickle.load(f)

    return env_transitions

def save_config(config, path):
    yaml_config = {}
    for key, value in config.items():
        if isinstance(value, tuple):
            yaml_config[key] = list(value)
        else:
            yaml_config[key] = value

    with open(path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, indent=2)

def load_config(path):
    with open(path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    restored_config = {}
    tuple_keys = {'input_size', 'output_size'}
    for key, value in yaml_config.items():
        if key in tuple_keys and isinstance(value, list):
            restored_config[key] = tuple(value)
        else:
            restored_config[key] = value
    return restored_config

def _select_device(device):
    if device is None:
        try:
            if torch.cuda.is_available():
                torch.cuda.device_count()
                return 'cuda'
            return 'cpu'
        except Exception as exc:
            logger.warning("CUDA check failed: %s. Falling back to CPU.", exc)
            return 'cpu'
    if device == 'cuda':
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return 'cpu'
            torch.cuda.device_count()
        except Exception as exc:
            logger.warning("CUDA init failed: %s. Falling back to CPU.", exc)
            return 'cpu'
    return device


def _resolve_model_folder(folder_path, prediction_horizon):
    if prediction_horizon is None:
        return folder_path
    horizon_suffix = f"h{prediction_horizon}"
    if horizon_suffix in folder_path:
        return folder_path
    candidate = os.path.join(folder_path, horizon_suffix)
    if os.path.isdir(candidate):
        return candidate
    return folder_path


def _pick_first_existing(paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None




def load_model(
    folder_path,
    model_type,
    n_basis=3,
    seed=0,
    device=None,
    prediction_horizon=1,
):
    if prediction_horizon is not None and prediction_horizon <= 0:
        prediction_horizon = 1
    device = _select_device(device)
    folder_path = _resolve_model_folder(folder_path, prediction_horizon)

    if model_type in ['dynamics_predictor', 'mo_predictor']:
        config_path = _pick_first_existing([os.path.join(folder_path, "config.yaml")])
        if config_path is None:
            raise FileNotFoundError(f"Missing config.yaml in {folder_path}")
        config = load_config(config_path)
        config['device'] = device
        model = FunctionEncoder(**config).to(device)
        model_path = _pick_first_existing([os.path.join(folder_path, "model.pth")])
        if model_path is None:
            raise FileNotFoundError(f"Missing model.pth in {folder_path}")
        model.load(model_path, device=device)

        if model_type == 'mo_predictor':
            history_length = config['model_kwargs']['encoder_kwargs']['history_length']
            feature_dim = config['model_kwargs']['encoder_kwargs']['input_size']
            model = TimeSeriesWrapper(model, history_length=history_length, feature_dim=feature_dim)
        return model

    if model_type not in ['fe', 'transformer', 'pem', 'oracle']:
        raise ValueError(f"Unknown model type: {model_type}")

    config_candidates = [
        os.path.join(folder_path, f"{model_type}_config.yaml"),
        os.path.join(folder_path, f"{model_type}{n_basis}_config.yaml"),
        os.path.join(folder_path, "config.yaml"),
    ]
    config_path = _pick_first_existing(config_candidates)
    if config_path is None:
        raise FileNotFoundError(f"No config file found in {folder_path}")
    config = load_config(config_path)
    config['device'] = device

    if model_type == 'fe':
        model = FunctionEncoder(**config)
    elif model_type == 'transformer':
        model = TransformerDynamics(**config)
    elif model_type == 'pem':
        model = PEM(**config)
    else:
        model = OracleMLP(**config)

    model = model.to(device)

    model_candidates = [
        os.path.join(folder_path, f"{model_type}_model_seed{seed}.pth"),
        os.path.join(folder_path, f"{model_type}_model.pth"),
        os.path.join(folder_path, f"{model_type}{n_basis}_model_{seed}.pth"),
        os.path.join(folder_path, "model.pth"),
    ]
    model_path = _pick_first_existing(model_candidates)
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {folder_path}")
    model.load(model_path, device=device)
    return model
