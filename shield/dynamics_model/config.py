from dataclasses import dataclass

@dataclass
class AdaptConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 128
    history_len: int = 5   # only used for Transformer
    # For PEM
    nll_weight: float = 1.0
    device: str = "cuda"