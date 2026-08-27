"""PyTorch MLP, matching the architecture in spec section 9.2.

Two changes from the naive version, both of which affect results:

* EARLY STOPPING WATCHES VALIDATION LOSS. Watching training loss stops when the
  model stops fitting the training set, which is the opposite of the signal you
  want -- training loss keeps falling long after generalisation has peaked. If
  no validation set is supplied, the fit is capped by `epochs` and says so.
* THE NETWORK OUTPUTS LOGITS, not probabilities, and the loss is
  BCEWithLogitsLoss. Sigmoid-then-BCE loses precision in the tails; the fused
  version is the numerically stable form. `predict_proba` applies the sigmoid.

Every seed used is recorded in the returned history so a run can be reproduced.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_all_seeds(seed: int) -> dict:
    """Seed every RNG that can affect a run. Returns what was set, for the log."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    return {"python": seed, "numpy": seed, "torch": seed}


def build_mlp(n_features: int, hidden1: int = 64, hidden2: int = 32,
              dropout1: float = 0.15, dropout2: float = 0.10,
              seed: int | None = 42) -> nn.Module:
    """64 -> 32 -> 1. Returns LOGITS; pair with BCEWithLogitsLoss.

    `seed` is applied HERE, before the Linear layers draw their initial weights.
    Seeding only inside `train_mlp` is too late -- initialisation has already
    consumed the global RNG, so two runs with the same recorded seed diverge and
    the seed in the experiment log does not reproduce the run it labels.
    """
    if seed is not None:
        set_all_seeds(seed)
    return nn.Sequential(
        nn.Linear(n_features, hidden1), nn.ReLU(), nn.Dropout(dropout1),
        nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout2),
        nn.Linear(hidden2, 1),
    )


@dataclass
class TrainHistory:
    train_loss: list
    val_loss: list
    seed: int
    seeds: dict
    epochs_run: int
    best_epoch: int
    stopped_early: bool
    early_stopping_signal: str

    def __contains__(self, key) -> bool:      # history["seed"] style access
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)


def train_mlp(model: nn.Module, X, y, *, X_val=None, y_val=None,
              epochs: int = 100, lr: float = 3e-4, batch: int = 128,
              patience: int = 10, weight_decay: float = 1e-4,
              seed: int = 42, shuffle: bool = True):
    """Train with AdamW + BCEWithLogitsLoss. Returns (model, TrainHistory).

    `shuffle` shuffles minibatches WITHIN the training window. That is not the
    thing the "never shuffle a time series" rule forbids -- the chronological
    split has already happened upstream and is untouched here.
    """
    seeds = set_all_seeds(seed)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    yt = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1))
    if torch.isnan(Xt).any():
        raise ValueError("NaN in training features -- clean or impute before training")

    gen = torch.Generator().manual_seed(seed)      # reproducible batch order
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch,
                        shuffle=shuffle, generator=gen)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    has_val = X_val is not None and y_val is not None and len(np.asarray(X_val)) > 0
    if has_val:
        Xv = torch.tensor(np.asarray(X_val, dtype=np.float32))
        yv = torch.tensor(np.asarray(y_val, dtype=np.float32).reshape(-1, 1))

    train_loss: list = []
    val_loss: list = []
    best = float("inf")
    best_epoch, bad, stopped = 0, 0, False
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        train_loss.append(float(np.mean(batch_losses)))

        if has_val:
            model.eval()
            with torch.no_grad():
                v = float(loss_fn(model(Xv), yv).item())
            val_loss.append(v)
            monitored = v
        else:
            monitored = train_loss[-1]

        if monitored < best - 1e-6:
            best, best_epoch, bad = monitored, epoch, 0
            best_state = {k: val.clone() for k, val in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                stopped = True
                break

    model.load_state_dict(best_state)      # restore the best epoch, not the last
    return model, TrainHistory(
        train_loss=train_loss, val_loss=val_loss, seed=seed, seeds=seeds,
        epochs_run=len(train_loss), best_epoch=best_epoch, stopped_early=stopped,
        early_stopping_signal="val_loss" if has_val else "train_loss (NO VALIDATION SET)",
    )


def predict_proba(model: nn.Module, X) -> np.ndarray:
    """Sigmoid of the logits, in eval mode so dropout is off."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(X, dtype=np.float32)))
        return torch.sigmoid(logits).numpy().ravel()


class TorchModel:
    """Adapter giving the MLP an sklearn-style `.predict_proba` (n, 2) interface."""

    def __init__(self, model: nn.Module, history=None):
        self.model = model
        self.history = history

    def predict_proba(self, X) -> np.ndarray:
        p = predict_proba(self.model, X)
        return np.column_stack([1.0 - p, p])
