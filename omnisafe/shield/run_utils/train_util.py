from functools import partial
import jax
import jax.numpy as jnp
from typing import TypeVar, Union
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax

T = TypeVar('T')
Array = Union[np.ndarray, jnp.ndarray]
REGULARIZATION_LAMBDA = 1e-1
NORM_PENALTY = 1e-1
HUBER_DELTA = 1.  # Adjust this value to control the transition point between L2 and L1 loss

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

def create_train_state(rng, model, learning_rate, input_size, output_size, learning_domain: str = 'ds'):
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
    if learning_domain == 'ts':
        dummy_input = jnp.ones((1, 1, model.history_length, input_size))
    else:
        dummy_input = jnp.ones((1, model.history_length, input_size))
    params = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_input)['params']
    # Create optimizer
    tx = optax.adam(learning_rate)
    
    return train_state.TrainState.create(apply_fn=model.apply,params=params, tx=tx)

def _deterministic_inner_product(features: Array, targets: Array) -> Array:
    """Compute deterministic inner product between features and targets.
    
    Args:
        features: Feature array of shape [..., dim1]
        targets: Target array of shape [..., dim2]
        
    Returns:
        Inner product array of shape [..., dim1, dim2]
    """
    # Handle 3D inputs by expanding last dimension
    unsqueezed_features = len(features.shape) == 3
    unsqueezed_targets = len(targets.shape) == 3
    
    if unsqueezed_features:
        features = jnp.expand_dims(features, axis=-1)
    if unsqueezed_targets:
        targets = jnp.expand_dims(targets, axis=-1)
    
    # Compute inner products
    element_wise_products = jnp.einsum("fdmk,fdml->fdkl", features, targets)
    
    inner_product = jnp.mean(element_wise_products, axis=1)
    
    # Restore original dimensions if needed
    if unsqueezed_features:
        inner_product = jnp.squeeze(inner_product, axis=-2)
    if unsqueezed_targets:
        inner_product = jnp.squeeze(inner_product, axis=-1)
    
    return inner_product

def huber_loss(x, delta=HUBER_DELTA):
    """Compute Huber loss with delta threshold."""
    abs_x = jnp.abs(x)
    return jnp.where(
        abs_x <= delta,
        0.5 * x**2,
        delta * abs_x - 0.5 * delta**2
    )

@partial(jax.jit, static_argnames=["output_size", "input_size", "n_basis", "l2_penalty", "least_squares", "average_function"])
def train_step(state, example_xs, example_ys, xs, ys, 
            input_size: int = 4, output_size: int = 4, n_basis: int = 100, 
            l2_penalty: float = 1e-4, least_squares: bool = False, average_function: bool = False):

    nbr_of_functions = xs.shape[0]
    def loss_fn(params):
        
        # Get outputs from both networks
        bfv_examples, avg_means_examples = state.apply_fn({'params': params}, example_xs)  
        bfv_examples = jnp.reshape(bfv_examples, (nbr_of_functions, -1, output_size, n_basis))
        
        # Instead of modifying example_ys, create a new variable
        example_ys_centered = example_ys
        if average_function:
            avg_means_examples = jnp.reshape(avg_means_examples, example_ys.shape)
            example_ys_centered = example_ys - avg_means_examples

        if not least_squares:
            coefficients = _deterministic_inner_product(bfv_examples, example_ys_centered)
        else:
            gram = _deterministic_inner_product(bfv_examples, bfv_examples)
            # Add stronger regularization to prevent ill-conditioning
            gram_reg = gram + REGULARIZATION_LAMBDA * jnp.eye(n_basis) + 1e-6 * jnp.ones((n_basis, n_basis))
            ip_representation = _deterministic_inner_product(bfv_examples, example_ys_centered)
            coefficients = jnp.einsum("fkl,fl->fk", jnp.linalg.inv(gram_reg), ip_representation)
            
        # Get outputs for xs
        bfv, avg_means = state.apply_fn({'params': params}, xs)    
        bfv = jnp.reshape(bfv, (nbr_of_functions, -1, output_size, n_basis))
        y_hat = jnp.einsum("fdmk,fk->fdm", bfv, coefficients)

        # Center ys if using average function
        ys_centered = ys
        if average_function:
            avg_means = jnp.reshape(avg_means, ys.shape)
            ys_centered = ys - avg_means

        # Calculate losses with Huber loss
        prediction_loss = jnp.mean(huber_loss(y_hat - ys_centered))
        norm_loss = jnp.mean(huber_loss(jnp.diagonal(gram, axis1=1, axis2=2) - 1)) if least_squares else 0
        
        # Add norm losses using Huber loss
        y_hat_norm_loss = jnp.mean(huber_loss(jnp.sqrt(jnp.sum(y_hat**2, axis=-1))))
        avg_means_norm_loss = jnp.mean(huber_loss(jnp.sqrt(jnp.sum(avg_means**2, axis=-1)))) if average_function else 0.0
        
        weight_penalty = sum(jnp.sum(huber_loss(param)) for param in jax.tree_util.tree_leaves(params))
        total_loss = (
            prediction_loss 
            + norm_loss 
            + l2_penalty * weight_penalty 
            + NORM_PENALTY * (y_hat_norm_loss + avg_means_norm_loss)
        )
        
        if average_function:
            avg_loss = jnp.mean(huber_loss(avg_means - ys))
            total_loss += avg_loss

        return total_loss, (prediction_loss, norm_loss, l2_penalty * weight_penalty, 
                          jnp.linalg.norm(coefficients), y_hat_norm_loss, avg_means_norm_loss)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    prediction_loss, norm_loss, weight_penalty, coefficients, y_hat_norm_loss, avg_means_norm_loss = aux
    
    # Clip gradients to prevent explosion
    grads = jax.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grads)
    
    return state.apply_gradients(grads=grads), (loss, prediction_loss, norm_loss, weight_penalty, coefficients, y_hat_norm_loss, avg_means_norm_loss)


@partial(jax.jit, static_argnames=["output_size", "n_basis", "l2_penalty", "least_squares", "average_function"])
def attn_train_step(state, example_xs, example_ys, xs, ys, 
                    output_size: int = 4, n_basis: int = 100, 
                    l2_penalty: float = 1e-4, 
                    least_squares: bool = False, average_function: bool = False):

    nbr_of_functions = xs.shape[0]
    # bfv stands for basis function values
    # ex stands for example
    def loss_fn(params):
            
        # Get outputs from both networks
        ex_bfv_means, ex_bfv_log_vars, ex_avg_means, ex_avg_log_vars = state.apply_fn({'params': params}, example_xs)  
        
        ex_bfv_means = jnp.reshape(ex_bfv_means, (nbr_of_functions, -1, output_size, n_basis))
        ex_bfv_log_vars = jnp.reshape(ex_bfv_log_vars, (nbr_of_functions, -1, output_size, n_basis))
        
        # for average function
        if average_function:
            ex_avg_means = jnp.reshape(ex_avg_means, (nbr_of_functions, -1, output_size))
            ex_avg_log_vars = jnp.reshape(ex_avg_log_vars, (nbr_of_functions, -1, output_size))
            ex_ys_centered = example_ys - ex_avg_means
        else:
            ex_ys_centered = example_ys
        
        if least_squares:
            gram = _deterministic_inner_product(ex_bfv_means, ex_bfv_means)
            gram_reg = gram + REGULARIZATION_LAMBDA * jnp.eye(n_basis)
            ip_representation = _deterministic_inner_product(ex_bfv_means, ex_ys_centered)
            coefficients = jnp.einsum("fkl,fl->fk", jnp.linalg.inv(gram_reg), ip_representation)
        else:
            coefficients = _deterministic_inner_product(ex_bfv_means, ex_ys_centered)
            
        # Get outputs for xs
        bfv_means, bfv_log_vars, avg_means, avg_log_vars = state.apply_fn({'params': params}, xs)    
        
        bfv_means = jnp.reshape(bfv_means, (nbr_of_functions, -1, output_size, n_basis))
        bfv_log_vars = jnp.reshape(bfv_log_vars, (nbr_of_functions, -1, output_size, n_basis))
        
        mu_pred = jnp.einsum("fdmk,fk->fdm", bfv_means, coefficients)
        logvar_pred = jnp.einsum("fdmk,fk->fdm", bfv_log_vars, coefficients)

        if average_function:
            avg_means = jnp.reshape(avg_means, (nbr_of_functions, -1, output_size))
            avg_log_vars = jnp.reshape(avg_log_vars, (nbr_of_functions, -1, output_size))
            ys_centered = ys - avg_means
        else:
            ys_centered = ys

        # Calculate losses
        gaussian_loss = 0.5 * (jnp.exp(-logvar_pred) * (mu_pred - ys_centered)**2 + logvar_pred).mean()
        norm_loss = ((jnp.diagonal(gram, axis1=1, axis2=2) - 1)**2).mean() if least_squares else 0
        weight_penalty = sum(jnp.sum(param ** 2) for param in jax.tree_util.tree_leaves(params))
        total_loss = gaussian_loss + norm_loss + l2_penalty * weight_penalty
        
        if average_function:
            avg_loss = 0.5 * (jnp.exp(-avg_log_vars) * (avg_means - ys) ** 2 + avg_log_vars).mean()
            total_loss += avg_loss

        return total_loss, (gaussian_loss, norm_loss, l2_penalty * weight_penalty, jnp.linalg.norm(coefficients))
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    gaussian_loss, norm_loss, weight_penalty, coefficients = aux
    
    # Clip gradients to prevent explosion
    grads = jax.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grads)
    
    return state.apply_gradients(grads=grads), (loss, gaussian_loss, norm_loss, weight_penalty, coefficients)
