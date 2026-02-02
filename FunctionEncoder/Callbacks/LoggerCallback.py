# shield/Callbacks/LoggerCallback.py
from typing import Union, Optional, Tuple, Dict
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from stable_baselines3.common.logger import configure

# Optional: only needed for isinstance checks and sliding windows
from shield.dynamics_model.oracle import OracleMLP
from shield.dynamics_model.pem import PEM
from shield.dynamics_model.transformer import TransformerDynamics, build_sliding_windows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_errors(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """Return scalar error metrics matching your original logging."""
    diff = y_true - y_pred
    l2 = torch.norm(diff, p=2, dim=-1)        # per-sample
    l1 = torch.norm(diff, p=1, dim=-1)
    mse = torch.mean(diff ** 2, dim=-1)
    rmse = torch.sqrt(mse)

    metrics = {
        "mean_l2":  float(l2.mean().item()),
        "mean_l1":  float(l1.mean().item()),
        "mean_mse": float(mse.mean().item()),
        "mean_rmse":float(rmse.mean().item()),
        "total_l2": float(l2.sum().item()),
        "total_l1": float(l1.sum().item()),
        "total_mse": float(mse.sum().item()),
        "total_rmse":float(rmse.sum().item()),
    }
    return metrics


def _eval_fe(function_encoder: FunctionEncoder,
             example_xs: torch.Tensor, example_ys: torch.Tensor,
             query_xs: torch.Tensor,  query_ys: torch.Tensor) -> Dict[str, float]:
    """Original FE pathway: use predict_from_examples and compute errors."""
    with torch.no_grad():
        y_hat = function_encoder.predict_from_examples(
            example_xs, example_ys, query_xs, method="least_squares"
        )  # [N_ctx, Tq, out_dim]
        y_hat = y_hat.reshape(-1, y_hat.shape[-1])
        y_true = query_ys.reshape(-1, query_ys.shape[-1])
        return _compute_errors(y_true, y_hat)


def _zero_shot_eval_supervised(model: Union[OracleMLP, PEM, TransformerDynamics],
                               example_xs: torch.Tensor, example_ys: torch.Tensor,
                               query_xs: torch.Tensor,  query_ys: torch.Tensor,
                               history_len: int = 5,
                               batch_size: int = 1024) -> Dict[str, float]:
    """
    Fast zero-shot eval for Torch dynamics models (no finetune).
    - OracleMLP/PEM: directly predict on query_xs.
    - Transformer: build windows on query and predict.
    """
    model.eval()

    if isinstance(model, TransformerDynamics):
        # Build windows per context, concatenate, run in batches
        preds, trues = [], []
        for i in range(query_xs.shape[0]):  # over contexts
            Xw_q, Yw_q = build_sliding_windows(query_xs[i], query_ys[i], history_len)
            trues.append(Yw_q)
            # batch
            cur_preds = []
            for j in range(0, Xw_q.shape[0], batch_size):
                xb = Xw_q[j:j + batch_size].to(DEVICE)
                with torch.no_grad():
                    cur_preds.append(model(xb))
            preds.append(torch.cat(cur_preds, dim=0).cpu())
        y_pred = torch.cat(preds, dim=0)
        y_true = torch.cat(trues, dim=0)
        return _compute_errors(y_true, y_pred)

    else:
        # OracleMLP / PEM: direct inference on query_xs
        # Flatten [N_ctx, Tq, in_dim] -> [N_ctx*Tq, in_dim]
        X = query_xs.reshape(-1, query_xs.shape[-1]).to(DEVICE)
        with torch.no_grad():
            if isinstance(model, PEM):
                mean, _ = model(X)
                y_pred = mean
            else:
                y_pred = model(X)
        y_true = query_ys.reshape(-1, query_ys.shape[-1])
        return _compute_errors(y_true, y_pred)


class LoggerCallback(BaseCallback):
    """
    Model-agnostic logger:
    - If model is FunctionEncoder -> uses predict_from_examples (original behavior)
    - If model is OracleMLP / PEM / TransformerDynamics -> zero-shot eval on Query
      (optionally do a tiny few-shot finetune across examples before eval).
    """

    def __init__(self,
                 model: Union[FunctionEncoder, OracleMLP, PEM, TransformerDynamics],
                 testing_dataset: BaseDataset,
                 logdir: Union[str, None] = None,
                 prefix: str = "test",
                 history_len: int = 5,
                 batch_size: int = 256):
        super(LoggerCallback, self).__init__()
        self.logger = configure(logdir)
        self.total_epochs = 0
        self.model = model
        self.testing_dataset = testing_dataset
        self.prefix = prefix
        self.history_len = history_len
        self.batch_size = batch_size

    def _log_phase(self, phase: str):
        example_xs, example_ys, query_xs, query_ys, info = self.testing_dataset.sample(phase=phase)

        if isinstance(self.model, FunctionEncoder):
            metrics = _eval_fe(self.model, example_xs, example_ys, query_xs, query_ys)
        else:
            metrics = _zero_shot_eval_supervised(
                self.model, example_xs, example_ys, query_xs, query_ys,
                history_len=self.history_len,
                batch_size=self.batch_size
            )

        # Record with the same keys you used previously
        self.logger.record(f"{phase}_mse",        metrics["mean_mse"])
        self.logger.record(f"{phase}_l2_error",   metrics["mean_l2"])
        self.logger.record(f"{phase}_l1_error",   metrics["mean_l1"])
        self.logger.record(f"{phase}_rmse_error", metrics["mean_rmse"])
        self.logger.record(f"{phase}_total_l2_error", metrics["total_l2"])
        self.logger.record(f"{phase}_total_l1_error", metrics["total_l1"])
        self.logger.record(f"{phase}_total_mse_error", metrics["total_mse"])
        self.logger.record(f"{phase}_total_rmse_error", metrics["total_rmse"])

    def on_step(self, locals: dict):
        with torch.no_grad():
            for phase in ["train", "eval"]:
                self._log_phase(phase)

        self.total_epochs += 1
        self.logger.dump(self.total_epochs)
