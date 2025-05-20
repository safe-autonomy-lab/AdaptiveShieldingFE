import jax.numpy as jnp
import jax

from flax import struct
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from dataclasses import is_dataclass, fields
from typing import Type, TypeVar

T = TypeVar('T')

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

@jax.jit
def compute_min_distance_batch_numpy(objects_positions, agent_position):
    """Compute minimum distances between agents and objects using JAX.

    Args:
        objects_positions (jnp.ndarray): Array of object positions with shape 
            (sampling_nbr, num envs, num objects, 2)
        agent_position (jnp.ndarray): Array of agent positions with shape 
            (sampling_nbr, num envs, 2)

    Returns:
        jnp.ndarray: Minimum distances for each agent with shape (num envs, num objects)
    """
    # Convert inputs to JAX arrays if they aren't already
    agent_xy = jnp.array(agent_position)[:, :, jnp.newaxis, :]
    objects_xy = jnp.array(objects_positions)
    
    # Compute distances using JAX operations
    distances = jnp.linalg.norm(objects_xy - agent_xy, axis=-1)
    
    # Return minimum distance for each agent
    return jnp.min(distances, axis=-1)

class Swish(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x * nn.sigmoid(x)

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
    elif activation == "swish":
        return Swish()
    else:
        raise ValueError(f"Unknown activation: {activation}")

@struct.dataclass
class TrainState(train_state.TrainState):
    inputs_mu: jnp.ndarray
    inputs_sigma: jnp.ndarray
    max_logvar: jnp.ndarray
    min_logvar: jnp.ndarray

def create_train_state(rng, model, learning_rate, input_size):
    """Create training state with He initialization for model parameters.
    
    Args:
        rng: JAX random key
        model: The model to initialize
        learning_rate: Learning rate for optimizer
        input_size: Size of input dimension
        output_size: Size of output dimension
        
    Returns:
        TrainState object with initialized parameters and optimizer
    """
    # Initialize with He initialization
    init_rng, dropout_rng = jax.random.split(rng)
    
    # Create dummy input with correct shape
    dummy_input = jnp.ones((1, 1, input_size))
    
    # Initialize model with He initialization
    # init_fn = jax.nn.initializers.he_normal()
    # init_fn = jax.nn.initializers.glorot_normal()
    init_fn = jax.nn.initializers.kaiming_uniform()
    params = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_input)['params']
    
    # Create optimizer
    tx = optax.adam(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

