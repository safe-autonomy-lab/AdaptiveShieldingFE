from typing import Dict, Optional, Any
import torch
import torch.nn.functional as F
import copy
from .config import AdaptConfig
from .oracle import OracleMLP
from .pem import PEM
from .transformer import TransformerDynamics
from torch.utils.data import DataLoader


def set_seed(seed: int = 0):
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mse_mae(pred: torch.Tensor, target: torch.Tensor):
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    return mse, mae




def adapt_and_eval_oracle(
    base_model: OracleMLP,
    dataset: Any,
    example_x: torch.Tensor, example_y: torch.Tensor,
    query_x: torch.Tensor, query_y: torch.Tensor,
    oracle_feat: Optional[torch.Tensor],
    cfg: AdaptConfig
):
    """
    Per-context: clone base model -> finetune on example -> eval on query.
    """
    model = copy.deepcopy(base_model).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = dataset

    # Train (example set)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optim.zero_grad()
            pred = model(xb, oracle_feat.to(cfg.device) if oracle_feat is not None else None)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optim.step()

    # Eval (query set)
    model.eval()
    with torch.no_grad():
        pred = model(query_x.to(cfg.device), oracle_feat.to(cfg.device) if oracle_feat is not None else None)
        mse, mae = mse_mae(pred, query_y.to(cfg.device))
    return {"mse": mse, "mae": mae}


def adapt_and_eval_pem(
    base_model: PEM,
    dataset: Any,
    query_x: torch.Tensor, query_y: torch.Tensor,
    cfg: AdaptConfig
):
    model = copy.deepcopy(base_model).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Train (example set) with NLL
    ds = dataset
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optim.zero_grad()
            mean, var = model(xb)
            nll = model.nll_loss(mean, var, yb)
            loss = cfg.nll_weight * nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

    # Eval (query set)
    model.eval()
    with torch.no_grad():
        mean, _ = model(query_x.to(cfg.device))
        mse, mae = mse_mae(mean, query_y.to(cfg.device))
    return {"mse": mse, "mae": mae}


def adapt_and_eval_transformer(
    base_model: TransformerDynamics,
    dataset: Any,
    example_x: torch.Tensor, example_y: torch.Tensor,
    query_x: torch.Tensor,  query_y: torch.Tensor,
    cfg: AdaptConfig
):
    """
    Build sliding windows per context, finetune & evaluate.
    """
    model = copy.deepcopy(base_model).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Windows from examples
    Xw_ex, Yw_ex = build_sliding_windows(example_x, example_y, cfg.history_len)  # [N_ex, H, in], [N_ex, 2]
    ds = dataset
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optim.zero_grad()
            pred = model(xb)                   # [B, 2]
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

    # Windows from query
    Xw_q, Yw_q = build_sliding_windows(query_x, query_y, cfg.history_len)
    model.eval()
    with torch.no_grad():
        pred = []
        bs = cfg.batch_size
        for i in range(0, Xw_q.shape[0], bs):
            xb = Xw_q[i:i+bs].to(cfg.device)
            pred.append(model(xb))
        pred = torch.cat(pred, dim=0)        # [N_q, 2]
        mse, mae = mse_mae(pred, Yw_q.to(cfg.device))
    return {"mse": mse, "mae": mae}
