import torch
import torch.nn as nn
import math
from typing import Optional
import os
import torch.nn.functional as F
from tqdm import tqdm
import einops


def build_sliding_windows(
    seq_x: torch.Tensor,  # [T, in_dim]
    seq_y: torch.Tensor,  # [T, out_dim]
    history_len: int
):
    """
    Build sliding windows (sequence-to-one) for a single trajectory.
    Returns:
        X_win: [T - H + 1, H, in_dim]
        y_win: [T - H + 1, out_dim]  (predict y at the last index)
    """
    T, D = seq_x.shape
    H = history_len
    if T < H:
        # pad left if too short (rare for query, but possible for examples)
        pad = H - T
        pad_x = torch.zeros(pad, D, device=seq_x.device, dtype=seq_x.dtype)
        pad_y = torch.zeros(pad, seq_y.shape[-1], device=seq_y.device, dtype=seq_y.dtype)
        seq_x = torch.cat([pad_x, seq_x], dim=0)
        seq_y = torch.cat([pad_y, seq_y], dim=0)
        T = H
    num = T - H + 1
    X = []
    Y = []
    for i in range(num):
        X.append(seq_x[i:i+H])         # [H, in_dim]
        Y.append(seq_y[i+H-1])         # predict at the last position of the window
    X = torch.stack(X, dim=0)
    Y = torch.stack(Y, dim=0)
    return X, Y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerDynamics(nn.Module):
    """
    Sequence-to-one: input a window [B, H, in_dim] -> predict delta [B, 2]
    """
    def __init__(self, input_size: int, output_size: int = 2, d_model: int = 128, nhead: int = 4, num_layers: int = 1, dim_ff: int = 128, representation_dim=8, **kwargs):
        super().__init__()
        self.n_basis = representation_dim
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.project_hidden = nn.Linear(d_model, representation_dim)
        self.project_example = nn.Linear(input_size + output_size, d_model)
        self.project_query = nn.Linear(input_size, d_model)
        self.head = nn.Linear(representation_dim + d_model, output_size)
        
    def forward(self, example_xs, example_ys, query):
        # example_data: [B, H, in_dim]
        # query: [B, H, in_dim]
        representation, _ = self.compute_representation(example_xs, example_ys)
        q = self.project_query(query)
        q = self.encoder(q) # [B, H, d_model]
        B, H, D = q.shape

        q_flat = einops.rearrange(q, 'b h d -> (b h) d')

        # Expand representation to [B*H, repr_dim] in one go:
        rep_flat = einops.repeat(
            representation,      # [B, repr_dim]
            'b d -> (b h) d',
            h=H
        )
        
        # Concatenate and predict [B, 2]
        h = torch.cat([rep_flat, q_flat], dim=-1)  # [B, 2*d_model]
        out = self.head(h)                       # [B, output_size]
        out = einops.rearrange(out, '(b h) d -> b h d', h=H)
        return out


    def compute_representation(self, example_xs, example_ys, **kwargs):
        example_data = torch.cat([example_xs, example_ys], dim=-1)
        ex = self.project_example(example_data)
        ex = self.pos(ex)
        z = self.encoder(ex)
        representation = self.project_hidden(z)[:,-1,:]
        # None is dummy to match the shape of function encoder's representation return
        return representation, None

    @torch.no_grad()
    def predict(self, input_xs, coeffs):
        self.eval()
        env_nbr, sampling_nbr, _ = input_xs.shape
        representation = coeffs.unsqueeze(1).repeat(1, sampling_nbr, 1)
        q = self.project_query(input_xs)
        q = self.encoder(q) # [B, H, d_model]
        h = torch.cat([representation, q], dim=-1)  # [B, 2*d_model]
        out = self.head(h)                       # [B, output_size]
        return out


    def train_model(self, dataset, epochs: int = 100, batch_size: int = 128, progress_bar: bool = True, callback=None, save_folder: str = None):
        self.device = dataset.device
        self.to(self.device).train()
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0)
        losses = []

        bar = tqdm(range(epochs), desc="Training Transformer Dynamics") if progress_bar else range(epochs)
        for ep in bar:
            if hasattr(dataset, 'reset_batch_state'):
                dataset.reset_batch_state()

            original_n_functions = dataset.n_functions
            total_functions = dataset.train_X.shape[0]
            n_batches = max(1, (total_functions + batch_size - 1) // batch_size)
            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                dataset.n_functions = min(batch_size, total_functions - batch_idx * batch_size)

                if dataset.n_functions <= 0:
                    break
                example_xs, example_ys, query_xs, query_ys, info = dataset.sample(phase='train')
                opt.zero_grad()
                pred = self.forward(example_xs=example_xs, example_ys=example_ys, query=query_xs)
                
                batch_loss = F.mse_loss(pred, query_ys)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

                epoch_loss += batch_loss.item()

                del batch_loss

            if callback is not None:
                # Create local variables for callback
                callback_info ={'self': self}
                callback.on_step(callback_info)

        dataset.n_functions = original_n_functions
        avg_epoch_loss = epoch_loss / n_batches
        losses.append(avg_epoch_loss)

        if progress_bar and hasattr(bar, 'set_postfix'):
            bar.set_postfix({'loss': f'{avg_epoch_loss:.6f}', 'batches': n_batches})
        if save_folder and (ep % 500 == 0):
            self.save(f'{save_folder}/model_{ep}.pth')
        
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
        return losses

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None, device: Optional[str] = None):
        self.load_state_dict(torch.load(path, map_location=map_location or device or self.device))