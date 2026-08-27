"""A neural network written out by hand in NumPy.

This exists because the spec asks for it: build forward propagation, binary
cross-entropy and backpropagation yourself before reaching for PyTorch, so the
PyTorch version is a convenience rather than a black box.

Gradient derivation for the output layer, since it is the step people skip:

    L    = -[y*log(p) + (1-y)*log(1-p)]        binary cross-entropy
    p    = sigmoid(z)
    dL/dp = (p - y) / (p*(1-p))
    dp/dz = p*(1-p)
    dL/dz = (p - y)                            the two terms cancel

That cancellation is why sigmoid and BCE are paired, and why `_backward` starts
from `(p - y)` with no sigmoid derivative in sight.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip before exp: at |z| > ~700 exp overflows and the loss becomes nan.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class NumpyMLP:
    """MLP with ReLU hidden layers and a sigmoid output, trained by hand.

    `dims` is the full layer spec: [n_features, hidden..., 1]. So [2, 4, 1] is
    one hidden layer of 4 units.
    """

    def __init__(self, dims, seed: int = 42):
        if len(dims) < 2:
            raise ValueError("dims needs at least an input and an output layer")
        if dims[-1] != 1:
            raise ValueError("this is a binary classifier; final dim must be 1")
        self.dims = list(dims)
        self.seed = seed
        rng = np.random.default_rng(seed)
        # He initialisation: variance 2/fan_in keeps ReLU activations from
        # collapsing toward zero as depth grows.
        self.W = [rng.normal(0.0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                  for i in range(len(dims) - 1)]
        self.b = [np.zeros((1, dims[i + 1])) for i in range(len(dims) - 1)]

    # ---------------------------------------------------------------- forward
    def forward(self, X):
        """Return (probabilities, cache). Cache holds each layer's activation."""
        a = np.asarray(X, dtype=float)
        cache = [a]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            a = np.maximum(0.0, a @ W + b)          # ReLU
            cache.append(a)
        out = _sigmoid(a @ self.W[-1] + self.b[-1])
        cache.append(out)
        return out, cache

    def predict_proba(self, X) -> np.ndarray:
        return self.forward(X)[0]

    # ------------------------------------------------------------------- loss
    @staticmethod
    def bce_loss(pred, y) -> float:
        eps = 1e-12
        p = np.asarray(pred, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    # --------------------------------------------------------------- backward
    def backward(self, cache, y):
        """Gradients of BCE w.r.t. every weight and bias."""
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n = cache[-1].shape[0]
        dz = (cache[-1] - y) / n                    # see module docstring
        gW: list = [None] * len(self.W)
        gb: list = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            a_in = cache[i]
            gW[i] = a_in.T @ dz
            gb[i] = dz.sum(axis=0, keepdims=True)
            if i > 0:
                # a_in is post-ReLU, so (a_in > 0) is exactly ReLU'(z) here.
                dz = (dz @ self.W[i].T) * (a_in > 0)
        return gW, gb

    def update(self, grads, lr: float) -> None:
        gW, gb = grads
        for i in range(len(self.W)):
            self.W[i] -= lr * gW[i]
            self.b[i] -= lr * gb[i]

    # -------------------------------------------------------------------- fit
    def fit(self, X, y, epochs: int = 200, lr: float = 0.1, verbose: bool = False):
        """Full-batch gradient descent. Returns the loss history."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        history = []
        for epoch in range(epochs):
            pred, cache = self.forward(X)
            loss = self.bce_loss(pred, y)
            history.append(loss)
            self.update(self.backward(cache, y), lr)
            if verbose and epoch % 50 == 0:
                print(f"epoch {epoch:4d}  loss {loss:.5f}")
        return history

    def gradient_check(self, X, y, eps: float = 1e-5) -> float:
        """Largest relative error between analytic and numeric gradients.

        Run this after touching `backward`. Below ~1e-6 means the derivation and
        the code agree; anything above means they do not.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        _, cache = self.forward(X)
        gW, _ = self.backward(cache, y)
        worst = 0.0
        for li in range(len(self.W)):
            it = np.nditer(self.W[li], flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = self.W[li][idx]
                self.W[li][idx] = orig + eps
                lp = self.bce_loss(self.forward(X)[0], y)
                self.W[li][idx] = orig - eps
                lm = self.bce_loss(self.forward(X)[0], y)
                self.W[li][idx] = orig
                numeric = (lp - lm) / (2 * eps)
                denom = max(abs(numeric) + abs(gW[li][idx]), 1e-12)
                worst = max(worst, abs(numeric - gW[li][idx]) / denom)
                it.iternext()
        return float(worst)
