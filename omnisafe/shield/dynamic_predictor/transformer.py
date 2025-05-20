from typing import Optional, Callable, Any
from tqdm import trange
from omnisafe.shield.dynamic_predictor.backbone import Backbone
from omnisafe.shield.dataset import BaseDataset
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import wandb

DEBUG = False

@jax.jit
def train_step(state, example_data, query, target):
    def loss_fn(params):
        pred_next_states = state.apply_fn({'params': params}, example_data, query, training=True)
        return jnp.mean(jnp.square(pred_next_states - target))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss

def transformer_create_train_state(rng, model, learning_rate, input_size, max_len=100):
    rng, init_rng = jax.random.split(rng)
    dummy_example_data = jnp.ones((1, max_len, input_size))
    dummy_query = jnp.ones((1, max_len, input_size))
    params = model.init(init_rng, dummy_example_data, dummy_query)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

class MultiHeadAttention(nn.Module):
    num_heads: int
    d_model: int
    dropout: float = 0.1

    def setup(self):
        head_dim = self.d_model // self.num_heads
        self.query = nn.Dense(self.d_model, use_bias=False)
        self.key = nn.Dense(self.d_model, use_bias=False)
        self.value = nn.Dense(self.d_model, use_bias=False)
        self.out = nn.Dense(self.d_model)

    @nn.compact
    def __call__(self, q, k, v, mask=None, training=False):
        batch_size = q.shape[0]
        q = self.query(q).reshape(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(0, 2, 1, 3)
        k = self.key(k).reshape(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(0, 2, 1, 3)
        v = self.value(v).reshape(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(0, 2, 1, 3)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.d_model // self.num_heads)
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)
        
        attention = nn.softmax(scores, axis=-1)
        attention = nn.Dropout(rate=self.dropout)(attention, deterministic=not training, rng=jax.random.PRNGKey(0))
        
        out = jnp.matmul(attention, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        return self.out(out)

class TransformerEncoder(nn.Module):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float

    @nn.compact
    def __call__(self, src: jnp.ndarray, src_mask: Optional[jnp.ndarray] = None, training: bool = False) -> jnp.ndarray:
        x = MultiHeadAttention(num_heads=self.nhead, d_model=self.d_model, dropout=self.dropout)(src, src, src, mask=src_mask, training=training)
        x = nn.LayerNorm()(src + x)
        y = nn.Dense(self.dim_feedforward)(x)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=not training, rng=jax.random.PRNGKey(0))
        y = nn.Dense(self.d_model)(y)
        return nn.LayerNorm()(x + y)

class TransformerDecoder(nn.Module):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float

    @nn.compact
    def __call__(self, tgt: jnp.ndarray, memory: jnp.ndarray, tgt_mask: Optional[jnp.ndarray] = None, 
                 memory_mask: Optional[jnp.ndarray] = None, training: bool = False) -> jnp.ndarray:
        x = MultiHeadAttention(num_heads=self.nhead, d_model=self.d_model, dropout=self.dropout)(tgt, tgt, tgt, mask=tgt_mask, training=training)
        x = nn.LayerNorm()(tgt + x)
        x2 = MultiHeadAttention(num_heads=self.nhead, d_model=self.d_model, dropout=self.dropout)(x, memory, memory, mask=memory_mask, training=training)
        x = nn.LayerNorm()(x + x2)
        y = nn.Dense(self.dim_feedforward)(x)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=not training, rng=jax.random.PRNGKey(0))
        y = nn.Dense(self.d_model)(y)
        return nn.LayerNorm()(x + y)

class TransformerDynamicPredictor(Backbone):
    input_size: int
    output_size: int
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    hidden_size: int = 256
    dropout: float = 0.1
    max_len: int = 100  # Maximum number of example data points

    def setup(self):
        self.encoder_embedding = nn.Dense(self.d_model)
        self.decoder_embedding = nn.Dense(self.d_model)
        self.encoder_layers = [TransformerEncoder(self.d_model, self.nhead, self.hidden_size, self.dropout) 
                               for _ in range(self.num_encoder_layers)]
        self.decoder_layers = [TransformerDecoder(self.d_model, self.nhead, self.hidden_size, self.dropout) 
                               for _ in range(self.num_decoder_layers)]
        self.output_layer = nn.Dense(self.output_size)

    def __call__(self, example_data: jnp.ndarray, query: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Reshape inputs
        if len(example_data.shape) == 4:
            first_dim, second_dim, third_dim, fourth_dim = example_data.shape
            example_data = example_data.reshape(first_dim, second_dim, third_dim * fourth_dim)
            query = query.reshape(first_dim, -1, third_dim * fourth_dim)
        # Embed inputs
        example_data = self.encoder_embedding(example_data)
        query = self.decoder_embedding(query)

        # Encoder
        for encoder_layer in self.encoder_layers:
            example_data = encoder_layer(example_data, training=training)

        # Decoder
        for decoder_layer in self.decoder_layers:
            query = decoder_layer(query, example_data, training=training)

        # Output layer
        return self.output_layer(query)

    def predict_difference(self, state, example_data, inputs):
        mean = state.apply_fn({'params': state.params}, example_data, inputs)
        return mean

    def reshape_and_sample(self, data, target_shape, key):
        """
        Reshape the input data and sample to create a new array with the target shape.
        
        :param data: Input array with shape (256, 4)
        :param target_shape: Desired output shape (256, 100, 4)
        :param key: JAX random key for sampling
        :return: Reshaped and sampled array with shape (256, 100, 4)
        """
        # Create indices for sampling
        batch_size, sample_size, _ = target_shape
        indices = jax.random.randint(key, shape=(batch_size, sample_size), minval=0, maxval=batch_size)
        
        # Use advanced indexing to sample and create the output array
        output = jnp.take(data, indices, axis=0)
        
        return output

    def set_train_step(self):
        self.train_step = train_step
    
    def train_model(self, epochs: int = 1, batch_size: int = 256, dataset: Optional[BaseDataset] = None, logger = None, eval_func: Callable[[Any], float] = None, use_wandb: bool = False):
        dataset = self.dataset if dataset is None else dataset

        # Warmup with small batch for initial compilation
        warmup_batch_size = 32
        example_xs, example_ys, xs, ys, _ = dataset.sample()
        if len(example_xs) > 0:
            small_example_x = example_xs[0][:warmup_batch_size][None]
            small_x = xs[0][:warmup_batch_size][None]
            small_y = ys[0][:warmup_batch_size][None]
            _ = train_step(self.state, small_example_x, small_x, small_y)

        for epoch in trange(epochs, desc="Training"):
            example_xs, example_ys, xs, ys, _ = dataset.sample()
            epoch_loss = 0
            num_batches = 0

            for example_x, example_y, x, y in zip(example_xs, example_ys, xs, ys):
                num_samples = example_x.shape[0]
                
                # Process data in batches
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    batch_example_x = example_x[i:end_idx][None]
                    batch_x = x[i:end_idx][None]
                    batch_y = y[i:end_idx][None]
                    
                    self.state, loss = train_step(self.state, batch_example_x, batch_x, batch_y)
                    epoch_loss += loss
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