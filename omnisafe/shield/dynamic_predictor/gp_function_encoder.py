from typing import Union, Optional, TypeVar, Sequence
import numpy as np
from functools import partial
from omnisafe.shield.dataset.base_dataset import BaseDataset
from tqdm import trange
import jax
import jax.numpy as jnp
from flax import linen as nn
from ..run_utils.train_util import get_activation, _deterministic_inner_product, REGULARIZATION_LAMBDA, train_step, attn_train_step
from .backbone import Backbone
from tqdm import trange
from .attn_encoder import AttentionHistoryEncoder
import wandb

# Type definitions
T = TypeVar('T')
Array = Union[np.ndarray, jnp.ndarray]
LayerSequence = Sequence[nn.Module]

class GPFunctionEncoder(Backbone):
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
        history_length: Length of history sequence for attention encoder
        use_attn_encoder: Whether to use attention encoder
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
    history_length: int = 5
    use_attention: bool = False  

    def setup(self) -> None:
        """Initialize network architectures."""
        if self.use_attention:
            self.attention_encoder = AttentionHistoryEncoder(input_size=self.input_size, hidden_size=self.hidden_size, num_heads=4, history_length=self.history_length)
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
        layers.append(nn.Dense(self.output_size * n_basis * 2))

        self.split_size = self.output_size * self.n_basis
        
        return nn.Sequential(layers)

    @nn.compact
    def __call__(self, x):
        if self.use_attention:
            x = self.attention_encoder(x)
        y = self.networks(x)
        y = jnp.reshape(y, (-1, self.output_size * self.n_basis * 2))
        means, log_std = jnp.split(y, [self.split_size], axis=-1)
        avg_means = None
        avg_log_std = None
        if self.average_function:
            y = self.average_networks(x)
            y = jnp.reshape(y, (-1, self.output_size * 2))
            avg_means, avg_log_std = jnp.split(y, [self.output_size], axis=-1)
        return means, log_std, avg_means, avg_log_std

    def compute_init_coefficients(self):
        """Recalculate coefficients for all datasets."""
        self.N = {}
        # for offline datasets
        example_xs, example_ys, xs, ys, _ = self.dataset.sample(mode='train')
        total_x = jnp.concatenate([example_xs, xs], axis=1)
        total_y = jnp.concatenate([example_ys, ys], axis=1)
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
    
    def extend_dataset(self, w):
        examples = jnp.array(self.episode_data_memory).reshape((1, -1, self.input_size + self.output_size))
        x = examples[:, :, :self.input_size]
        y = examples[:, :, self.input_size:] - x[:, :, :self.output_size] # delta prediction

        w_key = tuple(float(val) for val in jax.device_get(w.flatten()))
        self.dataset.add_experiences(x, y, w_key)
        return w_key
    
    def set_train_step(self):
        self.train_step = partial(
            attn_train_step, 
            output_size=self.output_size, 
            n_basis=self.n_basis, 
            l2_penalty=self.l2_penalty, 
            least_squares=self.least_squares, 
            average_function=self.average_function
        )

    def train_model(self, epochs: int = 1, dataset: Optional[BaseDataset] = None, logger = None, eval_func=None):        
        dataset = self.dataset if dataset is None else dataset
        train_step_fn = self.get_train_step()

        def one_epoch(mode='train'):
            example_xs, example_ys, xs, ys, _ = dataset.sample(mode=mode)
            self.state, (loss, prediction_loss, norm_loss, weight_penalty, coefficients) = train_step_fn(self.state, example_xs, example_ys, xs, ys)
            if logger is not None:
                if mode == 'train':
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
                    logger.record('loss', loss.item())
                    logger.record('prediction_loss', float(jax.device_get(prediction_loss)))
                    logger.record('norm_loss', float(jax.device_get(norm_loss)))
                    logger.record('weight_penalty', float(jax.device_get(weight_penalty)))
                    logger.record('coefficients', float(jax.device_get(coefficients)))
                        
        for epoch in trange(epochs):
            one_epoch(mode='train')
            if eval_func:
                eval_loss = eval_func(self)
                
                print(f"Evaluation Loss: {eval_loss:.4f}")
                logger.record('eval_loss', eval_loss)

                if use_wandb:
                    wandb.log({'eval_loss': eval_loss})

            if logger is not None:
                logger.dump(step=epoch)

    def predict_from_examples(self, example_xs, example_ys, xs):
        coefficients = self.get_compute_coefficients_fn()(self.state, example_xs, example_ys)
        basis_function_values = self.state.apply_fn({'params': self.state.params}, xs)    
        basis_function_values = jnp.reshape(basis_function_values, (xs.shape[0], -1, self.output_size, self.n_basis))
        y_hat = jnp.einsum("fdmk,fk->fdm", basis_function_values, coefficients)
        return y_hat
    
    def get_bfv_fn(self):
        return partial(GPFunctionEncoder.get_basis_function_value, n_basis=self.n_basis, output_size=self.output_size)
    
    def get_compute_coefficients_fn(self, use_least_squares: bool = True):
        return partial(GPFunctionEncoder.compute_coefficients, n_basis=self.n_basis, output_size=self.output_size, least_squares=use_least_squares, average_function=self.average_function)
    
    def get_predict_next_obs_by_coefficients_fn(self):
        return partial(GPFunctionEncoder.predict_next_obs_by_coefficients, n_basis=self.n_basis, output_size=self.output_size, average_function=self.average_function)
    
    def get_predict_next_obs_by_coefficients_timeseries_fn(self):
        return partial(GPFunctionEncoder.predict_next_obs_by_coefficients_timeseries, n_basis=self.n_basis, output_size=self.output_size, average_function=self.average_function)
    
    def get_predict_next_delta_by_coefficients_fn(self):
        return partial(GPFunctionEncoder.predict_next_delta_by_coefficients, n_basis=self.n_basis, output_size=self.output_size, average_function=self.average_function)
    
    # this will be online
    @staticmethod
    @partial(jax.jit, static_argnames=["n_basis", "output_size", "return_delta", "average_function"])
    def predict_next_obs_by_coefficients(state, obs, acs, coefficients, output_size, n_basis, return_delta: bool = False, average_function: bool = False):
        obs_acs = jnp.concatenate([obs, acs], axis=-1)
        if len(obs_acs.shape) == 2:
            # expand dimension of the first axis, which indicates that the number of functions (hidden parameter) is only one
            obs_acs = obs_acs[None]
        basis_function_values, avg_means = state.apply_fn({'params': state.params}, obs_acs)
        basis_function_values = jnp.reshape(basis_function_values, (1, -1, output_size, n_basis))
        y_hat = jnp.einsum("fdmk,fk->fdm", basis_function_values, coefficients)
        if average_function:
            avg_means = jnp.reshape(avg_means, (1, -1, output_size))
            y_hat += avg_means

        if return_delta:
            return y_hat
        return (obs + y_hat)[0]

    @staticmethod
    @partial(jax.jit, static_argnames=["n_basis", "output_size", "average_function"])
    def predict_next_delta_by_coefficients(state, obs_acs, coefficients, output_size, n_basis, average_function: bool = False):
        if len(obs_acs.shape) == 2:
            # expand dimension of the first axis, which indicates that the number of functions (hidden parameter) is only one
            obs_acs = obs_acs[None]
        bfv_means, bfv_log_vars, avg_means, avg_log_vars = state.apply_fn({'params': state.params}, obs_acs)    
        bfv_means = jnp.reshape(bfv_means, (1, -1, output_size, n_basis))
        bfv_log_vars = jnp.reshape(bfv_log_vars, (1, -1, output_size, n_basis))
        y_hat = jnp.einsum("fdmk,fk->fdm", bfv_means, coefficients)
        if average_function:
            avg_means = jnp.reshape(avg_means, (1, -1, output_size))
            y_hat += avg_means
        return y_hat
    
    @staticmethod
    @partial(jax.jit, static_argnames=["n_basis", "output_size"])
    def get_basis_function_value(state, x, n_basis, output_size):
        nbr_of_functions = x.shape[0]
        bfv, log_vars, avg_means, avg_log_vars = state.apply_fn({'params': state.params}, x)    
        bfv = jnp.reshape(bfv, (nbr_of_functions, -1, output_size, n_basis))
        log_vars = jnp.reshape(log_vars, (nbr_of_functions, -1, output_size, n_basis))
        avg_means = jnp.reshape(avg_means, (nbr_of_functions, -1, output_size))
        avg_log_vars = jnp.reshape(avg_log_vars, (nbr_of_functions, -1, output_size))
        return bfv, log_vars, avg_means, avg_log_vars

    @staticmethod    
    @partial(jax.jit, static_argnames=["output_size", "n_basis", "least_squares", "average_function"])
    def compute_coefficients(state, xs, ys, n_basis, output_size, least_squares, average_function):
        nbr_of_functions = xs.shape[0]
        bfv_means, bfv_log_vars, avg_means, avg_log_vars = state.apply_fn({'params': state.params}, xs)    
        bfv_means = jnp.reshape(bfv_means, (nbr_of_functions, -1, output_size, n_basis))
        bfv_log_vars = jnp.reshape(bfv_log_vars, (nbr_of_functions, -1, output_size, n_basis))

        if average_function:
            avg_means = jnp.reshape(avg_means, (nbr_of_functions, -1, output_size))
            ys = ys - avg_means

        if not least_squares:
            coefficients = _deterministic_inner_product(bfv_means, ys)
        else:
            gram = _deterministic_inner_product(bfv_means, bfv_means)
            gram_reg = gram + REGULARIZATION_LAMBDA * jnp.eye(n_basis)
            ip_representation = _deterministic_inner_product(bfv_means, ys)
            coefficients = jnp.einsum("fkl,fl->fk", jnp.linalg.inv(gram_reg), ip_representation)  # this is just batch matrix multiplication
            
        return coefficients
    
    def predict_from_examples(self, example_xs, example_ys, xs):
        coefficients = self.get_compute_coefficients_fn()(self.state, example_xs, example_ys)
        bfv, _, avg_means, _ = self.get_bfv_fn()(self.state, xs)    
        y_hat = jnp.einsum("fdmk,fk->fdm", bfv, coefficients)
        if self.average_function:
            y_hat += avg_means
        return y_hat

    def predict(self, coefficients, x):
        bfv, _, avg_means, _ = self.get_bfv_fn()(self.state, x)  
        y_hat = jnp.einsum("fdmk,fk->fdm", bfv, coefficients)
        if self.average_function:
            y_hat += avg_means
        return y_hat