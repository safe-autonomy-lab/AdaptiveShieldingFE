import jax
import jax.numpy as jnp

import optax
import flax.linen as nn
from tqdm import trange
from flax.training import train_state
from flax import struct

from .backbone import Backbone
from typing import Optional, Callable, Any
from omnisafe.shield.dataset.base_dataset import BaseDataset
from .attn_encoder import AttentionHistoryEncoder
from flax.training import train_state
import wandb

DEVICE = 'cuda' if jax.default_backend() == 'gpu' else 'cpu'


def oracle_create_train_state(rng, model, learning_rate, input_size, output_size, learning_domain: str = 'ds'):
    if learning_domain == 'ds':
        params = model.init(rng, jnp.ones((1, 1, model.history_length, input_size)))['params']
    else:
        params = model.init(rng, jnp.ones((1, model.history_length, input_size)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        means = state.apply_fn({'params': params}, x)
        weight_penalty = 0.0
        loss = 0.0
        for param in jax.tree_util.tree_leaves(params):
            weight_penalty += jnp.sum(param ** 2)  # L2 regularization
            
        # loss += 1e-4 * weight_penalty
        loss += jnp.mean(((means - y)**2).sum(-1))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Clip gradients to prevent explosion
    grads = jax.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grads)
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

class OrcaleMLP(Backbone):
    input_size: int
    output_size: int
    hidden_size: int = 256
    history_length: int = 5
    attn_encoder: bool = True

    def setup(self):
        if self.attn_encoder:
            self.attention_encoder = AttentionHistoryEncoder(input_size=self.input_size, hidden_size=self.hidden_size, num_heads=4, history_length=self.history_length)
        self.networks = [
            nn.Sequential([
                nn.Dense(self.hidden_size),
                nn.relu,
                nn.Dense(self.hidden_size),
                nn.relu,
                nn.Dense(self.hidden_size),
                nn.relu,
                nn.Dense(self.output_size)
            ])
        ]

    def __call__(self, x):
        if self.attn_encoder:
            x = self.attention_encoder(x)
        outputs = [net(x) for net in self.networks]
        outputs = jnp.stack(outputs, axis=0)
        means = jnp.mean(outputs, axis=0)
        return means
    
    def set_train_step(self):
        self.train_step = train_step
    
    def train_model(self, epochs: int = 1, batch_size: int = 256, dataset: Optional[BaseDataset] = None, logger = None, eval_func: Callable[[Any], float] = None, use_wandb: bool = False):
        dataset = self.dataset if dataset is None else dataset

        # Warmup with small batch for initial compilation
        warmup_batch_size = 32
        example_xs, example_ys, xs, ys, _ = dataset.sample(add_hidden_params=True)

        if len(example_xs) > 0:
            small_example_x = example_xs[0][:warmup_batch_size]
            small_example_y = example_ys[0][:warmup_batch_size]
            _ = train_step(self.state, small_example_x, small_example_y)

        for epoch in trange(epochs, desc="Training"):
            example_xs, example_ys, xs, ys, _ = dataset.sample(add_hidden_params=True)
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
    @jax.jit
    def predict_difference(state, inputs):
        mean = state.apply_fn({'params': state.params}, inputs)
        return mean

    @staticmethod
    @jax.jit
    def predict_next_obs(state, obs, acs):
        inputs = jnp.concatenate([obs, acs], axis=-1)
        mean = state.apply_fn({'params': state.params}, inputs)
        mean = jnp.mean(mean, axis=0)
        return obs + mean
