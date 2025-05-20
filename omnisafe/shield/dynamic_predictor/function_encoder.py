from typing import Union, Optional, TypeVar, Sequence
import numpy as np
from functools import partial
from omnisafe.shield.dataset.base_dataset import BaseDataset
from tqdm import trange
import jax
import jax.numpy as jnp
from flax import linen as nn
from ..run_utils.train_util import get_activation
from .backbone import Backbone
from tqdm import trange
from ..run_utils.train_util import REGULARIZATION_LAMBDA, _deterministic_inner_product, train_step
from .attn_encoder import AttentionHistoryEncoder
import wandb

# Type definitions
T = TypeVar('T')
Array = Union[np.ndarray, jnp.ndarray]
LayerSequence = Sequence[nn.Module]

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
    history_length: int = 1
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
        layers.append(nn.Dense(self.output_size * n_basis))
        
        return nn.Sequential(layers)

    @nn.compact
    def __call__(self, x):
        if self.use_attention:
            x = self.attention_encoder(x)
        means = self.networks(x)
        means = jnp.reshape(means, (-1, self.output_size * self.n_basis))
        avg_means = None
        if self.average_function:
            avg_means = self.average_networks(x)
            avg_means = jnp.reshape(avg_means, (-1, self.output_size))
        return means, avg_means

    def compute_init_coefficients(self, max_episode_length: int = 100):
        """Recalculate coefficients for all datasets."""
        self.N = {}
        # for offline datasets
        example_xs, example_ys, xs, ys, _ = self.dataset.sample(mode='train')
        total_x = jnp.concatenate([example_xs[:max_episode_length, :, :], xs[:max_episode_length, :, :]], axis=1)
        total_y = jnp.concatenate([example_ys[:max_episode_length, :, :], ys[:max_episode_length, :, :]], axis=1)
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
    
    def extend_dataset(self, w, epsilon: float = 0.1, augment_data: bool = False):
        examples = jnp.array(self.episode_data_memory).reshape((1, -1, self.input_size + self.output_size))
        x = examples[:, :, :self.input_size]
        y = examples[:, :, self.input_size:] - x[:, :, :self.output_size] # delta prediction
        w_similar = None
        if augment_data and self.dataset.get_size() > 0:
            X_similar, y_similar, w_similar = self.dataset.retrieve_similar_experiences(w, epsilon)
            N = len(X_similar)
            if N > 0:
                x = jnp.concatenate([x[0], X_similar], axis=0)
                y = jnp.concatenate([y[0], y_similar], axis=0)

        w_key = tuple(float(val) for val in jax.device_get(w.flatten()))
        self.dataset.add_experiences(x, y, w_key)
        return w_key, w_similar
    
    # online update for coefficients
    def online_dataset_update(self, w, epsilon: float = 0.1, augment_data: bool = False, reduce_data: bool = False):
        w_key, w_similar = self.extend_dataset(w, epsilon, augment_data)
        if reduce_data:
            before_reduce = len(self.dataset.experiences)
            for w in set(w_similar):
                if w in self.dataset.experiences:
                    self.dataset.experiences.pop(w)
                
            after_reduce = len(self.dataset.experiences)
            assert before_reduce - after_reduce == len(set(w_similar)), f"Length of w_similar: {len(set(w_similar))} and length of dataset before and after reduction do not match"

    def set_train_step(self):
        self.train_step = partial(train_step, input_size=self.input_size, output_size=self.output_size, n_basis=self.n_basis, l2_penalty=self.l2_penalty, least_squares=self.least_squares, average_function=self.average_function)

    def train_model(self, epochs: int = 1, batch_size: int = 50, dataset: Optional[BaseDataset] = None, logger = None, eval_func=None, use_wandb: bool = False):
        dataset = self.dataset if dataset is None else dataset
        train_step_fn = self.get_train_step()

        def one_epoch(mode='train'):
            example_xs, example_ys, xs, ys, _ = dataset.sample(mode=mode)
            epoch_loss = 0
            for i in range(0, example_xs.shape[0], batch_size):
                self.state, (loss, prediction_loss, norm_loss, weight_penalty, coefficients, y_hat_norm_loss, avg_means_norm_loss) = train_step_fn(self.state, example_xs[i: i + batch_size], example_ys[i: i + batch_size], xs[i: i + batch_size], ys[i: i + batch_size])
                epoch_loss += loss
                
            if logger is not None:
                if mode == 'train':
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
                    logger.record('loss', epoch_loss)
                    logger.record('prediction_loss', float(jax.device_get(prediction_loss)))
                    logger.record('norm_loss', float(jax.device_get(norm_loss)))
                    logger.record('weight_penalty', float(jax.device_get(weight_penalty)))
                    logger.record('coefficients', float(jax.device_get(coefficients)))
                    logger.record('y_hat_norm_loss', float(jax.device_get(y_hat_norm_loss)))
                    logger.record('avg_means_norm_loss', float(jax.device_get(avg_means_norm_loss)))
                        
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

    def get_compute_coefficients_fn(self, use_least_squares: bool = True):
        return partial(FunctionEncoder.compute_coefficients, n_basis=self.n_basis, input_size=self.input_size, output_size=self.output_size, least_squares=use_least_squares, average_function=self.average_function)
    
    def get_bfv_fn(self):
        return partial(FunctionEncoder.get_basis_function_value, n_basis=self.n_basis, output_size=self.output_size)
    
    def get_predict_next_obs_by_coefficients_fn(self):
        return partial(FunctionEncoder.predict_next_obs_by_coefficients, n_basis=self.n_basis, output_size=self.output_size, input_size=self.input_size, average_function=self.average_function)
    
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
        bfv, avg_means = state.apply_fn({'params': state.params}, obs_acs)    
        bfv = jnp.reshape(bfv, (1, -1, output_size, n_basis))
        y_hat = jnp.einsum("fdmk,fk->fdm", bfv, coefficients)
        if average_function:
            avg_means = jnp.reshape(avg_means, (1, -1, output_size))
            y_hat += avg_means

        if return_delta:
            return y_hat
        return (obs[:, :input_size] + y_hat)[0]

    @staticmethod
    @partial(jax.jit, static_argnames=["n_basis", "output_size"])
    def get_basis_function_value(state, x, n_basis, output_size):
        nbr_of_functions = x.shape[0]
        bvf, mean_values = state.apply_fn({'params': state.params}, x)
        mean_values = jnp.reshape(mean_values, (nbr_of_functions, -1, output_size)) if mean_values is not None else None
        return jnp.reshape(bvf, (nbr_of_functions, -1, output_size, n_basis)), mean_values

    @staticmethod    
    @partial(jax.jit, static_argnames=["output_size", "input_size", "n_basis", "least_squares", "average_function"])
    def compute_coefficients(state, xs, ys, n_basis, input_size, output_size, least_squares, average_function):
        nbr_of_functions = xs.shape[0]
        # Preserve the shape for attention encoder
        bfv, avg_means = state.apply_fn({'params': state.params}, xs)    
        bfv = jnp.reshape(bfv, (nbr_of_functions, -1, output_size, n_basis))
        
        if average_function:
            avg_means = jnp.reshape(avg_means, (nbr_of_functions, -1, output_size))
            ys = ys - avg_means

        if not least_squares:
            coefficients = _deterministic_inner_product(bfv, ys)
        else:
            gram = _deterministic_inner_product(bfv, bfv)
            gram_reg = gram + REGULARIZATION_LAMBDA * jnp.eye(n_basis)
            ip_representation = _deterministic_inner_product(bfv, ys)
            coefficients = jnp.einsum("fkl,fl->fk", jnp.linalg.inv(gram_reg), ip_representation)  # this is just batch matrix multiplication
            
        return coefficients
    
    def predict_from_examples(self, example_xs, example_ys, xs):
        coefficients = self.get_compute_coefficients_fn()(self.state, example_xs, example_ys)
        bfv, mean_values = self.get_bfv_fn()(self.state, xs)    
        y_hat = jnp.einsum("fdmk,fk->fdm", bfv, coefficients)
        if self.average_function:
            y_hat += mean_values
        return y_hat    

    def predict(self, coefficients, x):
        bfv, mean_values = self.get_bfv_fn()(self.state, x)  
        y_hat = jnp.einsum("fdmk,fk->fdm", bfv, coefficients)
        if self.average_function:
            y_hat += mean_values
        return y_hat