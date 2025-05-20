from typing import Union, Optional, TypeVar, Sequence
import numpy as np
from functools import partial
from omnisafe.shield.dataset.base_dataset import BaseDataset
from tqdm import trange
import jax
import jax.numpy as jnp
from flax import linen as nn
from omnisafe.shield.run_utils.train_util import get_activation
from omnisafe.shield.dynamic_predictor.backbone import Backbone
from tqdm import trange

REGULARIZATION_LAMBDA = 1e-2
REGULARIZATION_STRENGTH = 1e-4

# Type definitions
T = TypeVar('T')
Array = Union[np.ndarray, jnp.ndarray]
LayerSequence = Sequence[nn.Module]

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

@partial(jax.jit, static_argnames=["output_size", "input_size", "n_basis", "l2_penalty", "least_squares", "average_function"])
def train_step(state, example_xs, example_ys, xs, ys, 
            input_size: int = 4, output_size: int = 4, n_basis: int = 100, 
            l2_penalty: float = 1e-4, least_squares: bool = False, average_function: bool = False):

    nbr_of_functions = xs.shape[0]
    example_xs = jnp.reshape(example_xs, (-1, input_size))
    xs = jnp.reshape(xs, (-1, input_size))

    def loss_fn(params):
        # Get outputs from both networks
        basis_function_values_for_example, avg_means_examples = state.apply_fn({'params': params}, example_xs)  
        basis_function_values_for_example = jnp.reshape(basis_function_values_for_example, (nbr_of_functions, -1, output_size, n_basis))
        
        # Instead of modifying example_ys, create a new variable
        example_ys_centered = example_ys
        if average_function:
            avg_means_examples = jnp.reshape(avg_means_examples, example_ys.shape)
            example_ys_centered = example_ys - avg_means_examples

        if not least_squares:
            coefficients = _deterministic_inner_product(basis_function_values_for_example, example_ys_centered)
        else:
            gram = _deterministic_inner_product(basis_function_values_for_example, basis_function_values_for_example)
            gram_reg = gram + REGULARIZATION_LAMBDA * jnp.eye(n_basis)
            ip_representation = _deterministic_inner_product(basis_function_values_for_example, example_ys_centered)
            coefficients = jnp.einsum("fkl,fl->fk", jnp.linalg.inv(gram_reg), ip_representation)
        
        # Get outputs for xs
        basis_function_values, avg_means = state.apply_fn({'params': params}, xs)    
        basis_function_values = jnp.reshape(basis_function_values, (nbr_of_functions, -1, output_size, n_basis))
        y_hat = jnp.einsum("fdmk,fk->fdm", basis_function_values, coefficients)

        # Center ys if using average function
        ys_centered = ys
        if average_function:
            avg_means = jnp.reshape(avg_means, ys.shape)
            ys_centered = ys - avg_means

        # Calculate losses
        prediction_loss = jnp.mean((y_hat - ys_centered)**2)
        norm_loss = ((jnp.diagonal(gram, axis1=1, axis2=2) - 1)**2).mean() if least_squares else 0

        weight_penalty = sum(jnp.sum(param ** 2) for param in jax.tree_util.tree_leaves(params))
        # total_loss = prediction_loss + norm_loss
        total_loss = prediction_loss + norm_loss + l2_penalty * weight_penalty
        # total_loss = prediction_loss + norm_loss * 1e-2 + l2_penalty * weight_penalty
        
        if average_function:
            avg_loss = jnp.mean((avg_means - ys)**2)
            total_loss += avg_loss

        return total_loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss


class FunctionEncoder(Backbone):
    """Neural network for encoding functions with basis functions.
    
    Attributes:
        input_size: Dimension of input space
        output_size: Dimension of output space
        n_basis: Number of basis functions
        l2_penalty: L2 regularization strength
        hidden_size: Size of hidden layers
        n_layers: Number of hidden layers
        activation: Activation function name
        least_squares: Whether to use least squares optimization
        average_function: Whether to learn average function
    """
    
    input_size: int
    output_size: int
    n_basis: int = 100
    l2_penalty: float = 1e-4
    hidden_size: int = 512
    n_layers: int = 4
    activation: str = "relu"
    least_squares: bool = False 
    average_function: bool = False
    
    def setup(self) -> None:
        """Initialize network architectures."""
        self.networks = self._build_networks(self.n_basis)
        if self.average_function:
            self.average_networks = self._build_networks(1)
    
    def _build_networks(self, n_basis: int) -> nn.Sequential:
        """Build neural network layers.
        
        Args:
            n_basis: Number of basis functions for output layer
            
        Returns:
            Sequential neural network
        """
        layers: LayerSequence = []
        # Input layer
        layers.append(nn.Dense(self.hidden_size))
        layers.append(get_activation(self.activation))
        
        # Hidden layers
        for _ in range(self.n_layers - 2):
            layers.append(nn.Dense(self.hidden_size))
            layers.append(get_activation(self.activation))
            
        # Output layer
        layers.append(nn.Dense(self.output_size * n_basis))
        
        return nn.Sequential(layers)

    @nn.compact
    def __call__(self, x):
        means = self.networks(x)
        means = jnp.reshape(means, (-1, self.output_size * self.n_basis))
        avg_means = None
        if self.average_function:
            avg_means = self.average_networks(x)
            avg_means = jnp.reshape(avg_means, (-1, self.output_size))
        return means, avg_means

    def compute_init_coefficients(self, nbr_of_episodes: int = 100):
        """Recalculate coefficients for all datasets."""
        self.N = {}
        # for offline datasets
        example_xs, example_ys, xs, ys, _ = self.dataset.sample(mode='train')
        total_x = jnp.concatenate([example_xs, xs], axis=1)[:nbr_of_episodes]
        total_y = jnp.concatenate([example_ys, ys], axis=1)[:nbr_of_episodes]
        ws = self.get_compute_coefficients_fn()(self.state, total_x, total_y)
        for idx, w in enumerate(ws):
            w_key = tuple(float(val) for val in jax.device_get(w.flatten()))
            self.dataset.add_experiences(total_x[idx], total_y[idx], w_key)
            nbr_of_samples = total_x[idx].shape[0]
            self.N[w_key] = nbr_of_samples

    def recalculate_coefficients(self, batch_size: int = 1000):
        """Recalculate coefficients for all datasets."""
        self.N = {}
        previous_experiences = self.dataset.experiences.copy()
        self.dataset.reset_experiences()
        for w in previous_experiences:
            x, y = previous_experiences[w]
            w = self.get_compute_coefficients_fn()(self.state, x[None], y[None])
            w_key = tuple(float(val) for val in jax.device_get(w.flatten()))
            self.dataset.add_experiences(x[: min(1000, len(x))], y[: min(1000, len(y))], w_key)
            self.N[w_key] = len(x)
        
    def online_update_coefficients(self, inputs, targets, w, weight=True):
        scale_weight = 1 + N / (N + more_sample) if not weight else 1.
        more_sample = inputs.shape[1]
        w_online = self.get_compute_coefficients_fn()(self.state, inputs, scale_weight * targets)
        N = len(self.episode_data_memory)
        new_w = (N / (N + more_sample)) * w + (more_sample / (N + more_sample)) * w_online
        return new_w
    
    def online_update_by_coefficients(self, prev_coefficients, new_coefficients, N):
        return (N / (N + 1)) * prev_coefficients + (1 / (N + 1)) * new_coefficients
    
    def extend_dataset(self, w, epsilon: float = 0.1):
        examples = jnp.array(self.episode_data_memory).reshape((1, -1, self.input_size + self.output_size))
        x = examples[:, :, :self.input_size]
        y = examples[:, :, self.input_size:] - x[:, :, :self.output_size] # delta prediction
        w_key = tuple(float(val) for val in jax.device_get(w.flatten()))
        self.dataset.add_experiences(x, y, w_key)
        return w_key
    
    # online update for coefficients
    def online_dataset_update(self, w, epsilon: float = 0.1):
        w_key = self.extend_dataset(w, epsilon)
        return w_key

    def set_train_step(self):
        self.train_step = partial(train_step, input_size=self.input_size, output_size=self.output_size, n_basis=self.n_basis, l2_penalty=self.l2_penalty, least_squares=self.least_squares, average_function=self.average_function)

    def train_model(self, epochs: int = 1, dataset: Optional[BaseDataset] = None, logger = None, eval_func=None):
        if dataset is None:
            dataset = self.dataset
        
        train_step_fn = self.get_train_step()

        def one_epoch(mode='train'):
            example_xs, example_ys, xs, ys, _ = dataset.sample(mode=mode)
            self.state, loss = train_step_fn(self.state, example_xs, example_ys, xs, ys)
            if logger is not None:
                if mode == 'train':
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
                    logger.record('loss', loss.item())
                        
        for epoch in trange(epochs):
            one_epoch(mode='train')
            if eval_func:
                eval_loss = eval_func(self)
                
                print(f"Evaluation Loss: {eval_loss:.4f}")
                logger.record('eval_loss', eval_loss)

            if logger is not None:
                logger.dump(step=epoch)

    def get_compute_coefficients_fn(self, use_least_squares: bool = True):
        return partial(FunctionEncoder.compute_coefficients, n_basis=self.n_basis, input_size=self.input_size, output_size=self.output_size, least_squares=use_least_squares, average_function=self.average_function)
    
    def predict_from_examples(self, example_xs, example_ys, xs):
        coefficients = self.get_compute_coefficients_fn()(self.state, example_xs, example_ys)
        basis_function_values = self.state.apply_fn({'params': self.state.params}, xs)    
        basis_function_values = jnp.reshape(basis_function_values, (xs.shape[0], -1, self.output_size, self.n_basis))
        y_hat = jnp.einsum("fdmk,fk->fdm", basis_function_values, coefficients)
        return y_hat
    
    def get_predict_next_obs_by_coefficients_fn(self):
        return partial(FunctionEncoder.predict_next_obs_by_coefficients, n_basis=self.n_basis, output_size=self.output_size, input_size=self.input_size, average_function=self.average_function)
    
    def get_predict_next_obs_by_coefficients_timeseries_fn(self):
        return partial(FunctionEncoder.predict_next_obs_by_coefficients_timeseries, n_basis=self.n_basis, output_size=self.output_size, input_size=self.input_size, average_function=self.average_function)
    
    def get_predict_next_delta_by_coefficients_fn(self):
        return partial(FunctionEncoder.predict_next_delta_by_coefficients, n_basis=self.n_basis, output_size=self.output_size, input_size=self.input_size, average_function=self.average_function)
    
    # this will be online
    @staticmethod
    @partial(jax.jit, static_argnames=["n_basis", "output_size", "input_size", "return_delta", "average_function"])
    def predict_next_obs_by_coefficients(state, obs, acs, coefficients, output_size, input_size, n_basis, return_delta: bool = False, average_function: bool = False):
        obs_acs = jnp.concatenate([obs, acs], axis=-1)
        if len(obs_acs.shape) == 2:
            # expand dimension of the first axis, which indicates that the number of functions (hidden parameter) is only one
            obs_acs = obs_acs[None]
        assert obs_acs.shape[-1] == input_size, f'Input size: {obs_acs.shape[-1]} does not match the expected input size: {input_size}'
        basis_function_values, avg_means = state.apply_fn({'params': state.params}, obs_acs)
        basis_function_values = jnp.reshape(basis_function_values, (1, -1, output_size, n_basis))
        y_hat = jnp.einsum("fdmk,fk->fdm", basis_function_values, coefficients)
        if average_function:
            avg_means = jnp.reshape(avg_means, (1, -1, output_size))
            y_hat += avg_means

        if return_delta:
            return y_hat
        return (obs[:, :input_size] + y_hat)[0]

    @staticmethod
    @partial(jax.jit, static_argnames=["n_basis", "output_size", "input_size", "average_function"])
    def predict_next_delta_by_coefficients(state, obs_acs, coefficients, output_size, input_size, n_basis, average_function: bool = False):
        if len(obs_acs.shape) == 2:
            # expand dimension of the first axis, which indicates that the number of functions (hidden parameter) is only one
            obs_acs = obs_acs[None]
        assert obs_acs.shape[-1] == input_size, f'Input size: {obs_acs.shape[-1]} does not match the expected input size: {input_size}'
        basis_function_values, avg_means = state.apply_fn({'params': state.params}, obs_acs)    
        basis_function_values = jnp.reshape(basis_function_values, (1, -1, output_size, n_basis))
        y_hat = jnp.einsum("fdmk,fk->fdm", basis_function_values, coefficients)
        if average_function:
            avg_means = jnp.reshape(avg_means, (1, -1, output_size))
            y_hat += avg_means
        return y_hat
    
    @staticmethod
    @partial(jax.jit, static_argnames=["n_basis", "output_size"])
    def get_basis_function_value(state, x, n_basis, output_size):
        nbr_of_functions = x.shape[0]
        basis_function_values = state.apply_fn({'params': state.params}, x)    
        return jnp.reshape(basis_function_values, (nbr_of_functions, -1, output_size, n_basis))

    @staticmethod    
    @partial(jax.jit, static_argnames=["output_size", "input_size", "n_basis", "least_squares", "average_function"])
    def compute_coefficients(state, xs, ys, n_basis, input_size, output_size, least_squares, average_function):
        nbr_of_functions = xs.shape[0]
        xs = jnp.reshape(xs, (-1, input_size))
        basis_function_values_mean, avg_means = state.apply_fn({'params': state.params}, xs)    
        basis_function_values_mean = jnp.reshape(basis_function_values_mean, (nbr_of_functions, -1, output_size, n_basis))

        if average_function:
            avg_means = jnp.reshape(avg_means, (nbr_of_functions, -1, output_size))
            ys = ys - avg_means

        if not least_squares:
            coefficients = _deterministic_inner_product(basis_function_values_mean, ys)
        else:
            gram = _deterministic_inner_product(basis_function_values_mean, basis_function_values_mean)
            gram_reg = gram + REGULARIZATION_LAMBDA * jnp.eye(n_basis)
            ip_representation = _deterministic_inner_product(basis_function_values_mean, ys)
            coefficients = jnp.einsum("fkl,fl->fk", jnp.linalg.inv(gram_reg), ip_representation)  # this is just batch matrix multiplication
            
        return coefficients