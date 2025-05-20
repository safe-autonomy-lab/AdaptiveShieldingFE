from functools import partial
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from tqdm import trange
from flax.training import train_state
from flax import struct
from typing import Optional, Any, Callable
from omnisafe.shield.dataset import BaseDataset
from ..run_utils.train_util import Swish
from .backbone import Backbone
from .attn_encoder import AttentionHistoryEncoder
import wandb


@struct.dataclass
class TrainState(train_state.TrainState):
    ensemble_size: int
    inputs_mu: jnp.ndarray
    inputs_sigma: jnp.ndarray
    max_logvar: jnp.ndarray
    min_logvar: jnp.ndarray

def create_train_state_pem(rng, model, learning_rate, input_size, output_size, ensemble_size, learning_domain: str = 'ds'):
    # Initialize with correct shape: (1, batch_size, history_length, input_size)
    if learning_domain == 'ts':
        params = model.init(rng, jnp.ones((1, 1, model.history_length, input_size)))['params']
        inputs_mu = jnp.zeros((1, 1, model.history_length, input_size))
        inputs_sigma = jnp.ones((1, 1, model.history_length, input_size))
    else:
        params = model.init(rng, jnp.ones((1, model.history_length, input_size)))['params']
        inputs_mu = jnp.zeros((1, model.history_length, input_size))
        inputs_sigma = jnp.ones((1, model.history_length, input_size))
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, 
                             ensemble_size=ensemble_size,
                             inputs_mu=inputs_mu,
                             inputs_sigma=inputs_sigma,
                             max_logvar=jnp.ones((1, output_size)) / 2.0,
                             min_logvar=-jnp.ones((1, output_size)) * 10.0)

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        means, vars = state.apply_fn({'params': params}, x)
        loss = 0.01 * (state.max_logvar.sum() - state.min_logvar.sum())
        weight_penalty = 0.0
        for param in jax.tree_util.tree_leaves(params):
            weight_penalty += jnp.sum(param ** 2)  # L2 regularization
            
        loss += 1e-4 * weight_penalty
        # Clip the variance values to prevent numerical instability
        vars = jnp.clip(vars, 1e-6, 1e6)
        loss += jnp.mean(((means - y)**2 / vars + jnp.log(vars)).sum(-1))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Clip gradients to prevent explosion
    grads = jax.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grads)
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

class ProbabilisticEnsembleModel(Backbone):
    input_size: int
    output_size: int
    ensemble_size: int = 2
    hidden_size: int = 512
    history_length: int = 5
    attn_encoder: bool = True

    def setup(self):
        if self.attn_encoder:
            self.attention_encoder = AttentionHistoryEncoder(input_size=self.input_size, hidden_size=self.hidden_size, num_heads=4, history_length=self.history_length)
        self.networks = [
            nn.Sequential([
                nn.Dense(self.hidden_size),
                Swish(),
                nn.Dense(self.hidden_size),
                Swish(),
                nn.Dense(self.output_size * 2)  # Mean and log_var
            ]) for _ in range(self.ensemble_size)
        ]
        if self.attn_encoder:
            self.inputs_mu = self.param('inputs_mu', lambda _: jnp.zeros((1, 1, self.history_length, self.input_size)))
            self.inputs_sigma = self.param('inputs_sigma', lambda _: jnp.ones((1, 1, self.history_length, self.input_size)))
        else:
            self.inputs_mu = self.param('inputs_mu', lambda _: jnp.zeros((1, self.history_length, self.input_size)))
            self.inputs_sigma = self.param('inputs_sigma', lambda _: jnp.ones((1, self.history_length, self.input_size)))
        self.max_logvar = self.param('max_logvar', lambda _: jnp.ones((1, self.output_size)) / 2.0)
        self.min_logvar = self.param('min_logvar', lambda _: -jnp.ones((1, self.output_size)) * 10.0)
    
    def __call__(self, x):
        # this will be triggered when we use this model for timeseries prediction
        if len(x.shape) == 3:
            x = x[None]
        # Normalize input
        x = (x - self.inputs_mu) / self.inputs_sigma
        if self.attn_encoder:
            x = self.attention_encoder(x)
        outputs = [net(x) for net in self.networks]
        outputs = jnp.stack(outputs, axis=0)
        means, log_vars = jnp.split(outputs, 2, axis=-1)
        
        log_vars = self.max_logvar - nn.softplus(self.max_logvar - log_vars)
        log_vars = self.min_logvar + nn.softplus(log_vars - self.min_logvar)
        vars = jnp.exp(log_vars)
        return means, vars
    
    def set_train_step(self):
        self.train_step = train_step
    
    def train_model(self, epochs: int = 1, batch_size: int = 256, dataset: Optional[BaseDataset] = None, logger = None, eval_func: Callable[[Any], float] = None, use_wandb: bool = False):
        dataset = self.dataset if dataset is None else dataset

        # Warmup with small batch for initial compilation
        warmup_batch_size = 32
        example_xs, example_ys, xs, ys, _ = dataset.sample(add_hidden_params=False)
        
        if len(example_xs) > 0:
            small_example_x = example_xs[0][:warmup_batch_size]
            small_example_y = example_ys[0][:warmup_batch_size]
            _ = train_step(self.state, small_example_x, small_example_y)

        for epoch in trange(epochs, desc="Training"):
            example_xs, example_ys, xs, ys, _ = dataset.sample(add_hidden_params=False)
            epoch_loss = 0
            num_batches = 0

            for example_x, example_y, x, y in zip(example_xs, example_ys, xs, ys):
                num_samples = example_x.shape[0]
                # Process data in batches
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    
                    # Process example data
                    batch_example_x = example_x[i:end_idx]
                    batch_example_y = example_y[i:end_idx]
                    self.state, loss1 = train_step(self.state, batch_example_x, batch_example_y)
                    
                    # Process current data
                    batch_x = x[i:end_idx]
                    batch_y = y[i:end_idx]
                    self.state, loss2 = train_step(self.state, batch_x, batch_y)
                    
                    epoch_loss += (loss1 + loss2)
                    num_batches += 1 

            avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            if eval_func:
                eval_loss = eval_func(self)
                print(f"Evaluation Loss: {eval_loss:.4f}")
                if use_wandb:
                    wandb.log({'eval_loss': eval_loss})

            if logger:
                logger.record("loss", avg_loss)
                if eval_func:
                    logger.record("eval_loss", eval_loss)
                logger.dump()

    @staticmethod
    @partial(jax.jit, static_argnames=["ign_var"])
    def predict_difference(state, inputs, ign_var: bool = False):
        mean, var = state.apply_fn({'params': state.params}, inputs)
        mean = jnp.mean(mean, axis=0)[0]
        var = jnp.mean(var, axis=0)[0]
        
        if ign_var:
            return mean, var
        else:
            return mean + jax.random.normal(jax.random.PRNGKey(0), var.shape) * jnp.sqrt(var), var

    @staticmethod
    @jax.jit
    def predict_next_obs(state, obs, acs, ign_var: bool = False):
        inputs = jnp.concatenate([obs, acs], axis=-1)
        mean, var = state.apply_fn({'params': state.params}, inputs)
        mean = jnp.mean(mean, axis=0)
        var = jnp.mean(var, axis=0)
        
        if ign_var:
            return obs + mean, var
        else:
            return obs + mean + jax.random.normal(jax.random.PRNGKey(0), var.shape) * jnp.sqrt(var), var