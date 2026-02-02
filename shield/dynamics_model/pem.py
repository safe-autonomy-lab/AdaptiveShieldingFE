import torch
import torch.nn as nn
import math
from typing import Tuple
from .oracle import MLP
from typing import Optional
import os
import gc
from tqdm import tqdm

class PEMMember(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.mean_head = MLP(input_size, output_size, hidden=hidden, layers=layers)
        self.logvar_head = MLP(input_size, output_size, hidden=hidden, layers=layers)

    def forward(self, x: torch.Tensor):
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        # clamp logvar for stability
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        return mean, logvar


class PEM(nn.Module):
    """
    Probabilistic Ensemble Model: outputs (mean, logvar) per member; aggregates by mean.
    """
    def __init__(self, input_size: int, output_size: int, ens_size: int = 5, hidden: int = 256, layers: int = 3, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([PEMMember(input_size, output_size, hidden, layers) for _ in range(ens_size)])
        self.ens_size = ens_size
        # this is dummy value to work with shield!
        self.n_basis = 0

    def forward(self, x: torch.Tensor):
        means = []
        logvars = []
        for m in self.members:
            mu, lv = m(x)
            means.append(mu)
            logvars.append(lv)
        means = torch.stack(means, dim=0)      # [E, B, out_dim]
        logvars = torch.stack(logvars, dim=0)  # [E, B, out_dim]
        # Aggregate: mean of means; average variance (via mean exp(logvar))
        mean = means.mean(dim=0)               # [B, out_dim]
        var = torch.exp(logvars).mean(dim=0)   # [B, out_dim]
        return mean, var

    @staticmethod
    def nll_loss(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor):
        # Gaussian NLL per dimension, then sum
        # Add small epsilon for numerical stability
        eps = 1e-6
        var = var + eps
        nll = 0.5 * (torch.log(2*math.pi*var) + (target - mean)**2 / var)
        return nll.sum(dim=-1).mean()

    def train_model(self, dataset, epochs: int = 100, batch_size: int = 128, progress_bar: bool = True, callback=None, save_folder: str = None):
        self.device = dataset.device
        self.to(self.device).train()
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0)
        losses = []

        bar = tqdm(range(epochs), desc="Epochs") if progress_bar else range(epochs)
        for epoch in bar:
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

                example_xs, example_ys, query_xs, query_ys, _ = dataset.sample(phase='train', use_oracle_features=True)
                
                X = torch.cat([example_xs, query_xs], dim=1)
                Y = torch.cat([example_ys, query_ys], dim=1)
                X = X.reshape(-1, X.shape[-1]).to(self.device)
                Y = Y.reshape(-1, Y.shape[-1]).to(self.device)

                opt.zero_grad()
                mean, var = self.forward(X)
                batch_loss = self.nll_loss(mean, var, Y)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

                # Force garbage collection every few batches to free memory
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
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

        if epoch % 500 == 0:
            self.save(f'{save_folder}/model_{epoch}.pth')

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)

        return losses

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None, device: Optional[str] = None):
        self.load_state_dict(torch.load(path, map_location=map_location or device or self.device))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, dummy_input: torch.Tensor = None):
        self.eval()
        mu, _ = self.forward(x)
        return mu
