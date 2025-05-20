import jax.numpy as jnp
import flax.linen as nn

# Define a custom TransformerEncoderLayer to match PyTorch's nn.TransformerEncoderLayer
class TransformerEncoderLayer(nn.Module):
    hidden_size: int
    num_heads: int

    def setup(self):
        # Multi-head self-attention, equivalent to PyTorch's transformer layer attention
        self.self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size
        )
        # Feed-forward network, matching PyTorch's default (ReLU activation)
        self.feed_forward = nn.Sequential([
            nn.Dense(self.hidden_size * 4),
            nn.relu,
            nn.Dense(self.hidden_size)
        ])
        # Layer normalization layers (post-norm, as in PyTorch default)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x, key_padding_mask=None):
        # Convert key_padding_mask (True means padded) to attention mask (True means attend)
        if key_padding_mask is not None:
            attention_mask = ~key_padding_mask[:, None, None, :]
        else:
            attention_mask = None
        
        # Self-attention with residual connection
        attn_output = self.self_attn(inputs_q=x, inputs_kv=x, mask=attention_mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

# Main AttentionHistoryEncoder class
class AttentionHistoryEncoder(nn.Module):
    input_size: int = 4
    hidden_size: int = 128
    num_heads: int = 4
    history_length: int = 5

    def setup(self):
        # Linear projection from input_size to hidden_size
        self.proj = nn.Dense(self.hidden_size)
        # Learnable positional encodings, initialized to zeros
        self.pos_encodings = self.param('pos_encodings', nn.initializers.zeros, (1, self.history_length, self.hidden_size))
        # Transformer encoder layer
        self.encoder_layer = TransformerEncoderLayer(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads
        )

    def __call__(self, x, mask=None):
        # Input x shape: (nbr_of_fn, batch_size, history_length, input_size)
        # or (batch_size, history_length, input_size)
        nbr_of_fn, batch_size = x.shape[0], x.shape[1]
        
        # Reshape input to (nbr_of_fn * batch_size, history_length, input_size)
        x = x.reshape(-1, self.history_length, self.input_size)
        if mask is not None:
            mask = mask.reshape(-1, self.history_length)
        
        # Project input to hidden size
        x_proj = self.proj(x)  # Shape: (nbr_of_fn * batch_size, history_length, hidden_size)
        
        # Add positional encodings (broadcasts over batch dimension)
        x_pos = x_proj + self.pos_encodings
        
        # Pass through transformer encoder layer
        output = self.encoder_layer(x_pos, key_padding_mask=mask)
        
        # Extract encoded representation
        if mask is not None:
            # For masked sequences, find the last valid position
            # mask: True means padded, ~mask: True means valid
            valid_indices = jnp.where(~mask, jnp.arange(self.history_length)[None, :], -1)
            last_valid_idx = jnp.max(valid_indices, axis=1)
            # If all positions are masked (max is -1), use index 0
            last_valid_idx = jnp.where(last_valid_idx == -1, 0, last_valid_idx)
            # Gather the output at the last valid indices
            encoded_repr = output[jnp.arange(len(output)), last_valid_idx, :]
        else:
            # If no mask, take the last position
            encoded_repr = output[:, -1, :]
        
        # Reshape output back to (nbr_of_fn, batch_size, hidden_size)
        encoded_repr = encoded_repr.reshape(nbr_of_fn, batch_size, self.hidden_size)
        
        return encoded_repr