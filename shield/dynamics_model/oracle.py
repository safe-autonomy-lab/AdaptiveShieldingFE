import torch
import torch.nn as nn
from typing import Optional
import os
import gc
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# from .train_batch import train_torch_dynamics_batch


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        blocks = []
        d = input_size
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        blocks += [nn.Linear(d, output_size)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OracleMLP(nn.Module):
    """
    Oracle MLP that optionally concatenates oracle/context features of dimension oracle_dim.
    Input each step: [input_size + oracle_dim] if oracle present, else [input_size].
    Predicts delta position (2).
    """
    def __init__(self, input_size: int, output_size: int = 2, oracle_dim: int = 0, hidden: int = 256, layers: int = 3, **kwargs):
        super().__init__()
        self.oracle_dim = oracle_dim
        # this is dummy value to work with shield!
        self.n_basis = 0
        self.mlp = MLP(input_size + oracle_dim, output_size, hidden=hidden, layers=layers)

    def forward(self, x: torch.Tensor, oracle_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, in_dim]
        if self.oracle_dim > 0 and oracle_feat is not None:
            # oracle_feat can be [B, oracle_dim] or [oracle_dim] (broadcast)
            if oracle_feat.dim() == 1:
                oracle_feat = oracle_feat.unsqueeze(0).expand(x.shape[0], -1)
            x = torch.cat([x, oracle_feat], dim=-1)
        return self.mlp(x)

    def train_model(self, dataset, epochs: int = 100, batch_size: int = 32, progress_bar: bool = True, callback=None, save_folder: str = None):
        """
        Meta-train across contexts on the TRAIN phase transitions:
        We fit a global model using all (context, example) pairs concatenated.
        """
        self.device = dataset.device
        self.to(self.device).train()
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0)
        losses = []

        bar = tqdm(range(epochs), desc="Training Oracle MLP") if progress_bar else range(epochs)
        for ep in bar:
            # reset batch idx for new epoch
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

                example_xs, example_ys, query_xs, query_ys, info = dataset.sample(phase='train', use_oracle_features=True)

                X = torch.cat([example_xs, query_xs], dim=1)
                Y = torch.cat([example_ys, query_ys], dim=1)
                X = X.reshape(-1, X.shape[-1]).to(self.device)
                Y = Y.reshape(-1, Y.shape[-1]).to(self.device)
                
                opt.zero_grad()
                pred = self.forward(X)
                batch_loss = F.mse_loss(pred, Y)
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

    # Optional single-shot predict helper
    @torch.no_grad()
    def predict(self, x: torch.Tensor, dummy_input: torch.Tensor = None) -> torch.Tensor:
        self.eval()
        output = self.forward(x)
        return output
