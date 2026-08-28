"""Baselines. The neural net has to beat these or it does not get used.

Ordered weakest to strongest, as the spec requires:
  1. no-trade      -- the honest null: zero trades, zero P&L, zero risk
  2. always-trade  -- take every candidate that passes the filters
  3. simple rule   -- one transparent threshold on credit/width
  4. logistic      -- transparent probabilistic baseline
  5. gradient boosting -- the real bar for tabular data
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from features.features import FEATURE_INDEX

DEFAULT_SEED = 42

# Index resolved by NAME, not a hard-coded integer. The old plan's comment said
# "col 20 = credit_to_width" while column 20 was actually `credit`, so the rule
# silently thresholded the wrong feature.
CREDIT_TO_WIDTH = FEATURE_INDEX["credit_to_width"]


def train_logistic(X, y, seed: int = DEFAULT_SEED, max_iter: int = 1000):
    m = LogisticRegression(max_iter=max_iter, random_state=seed)
    m.fit(np.asarray(X, dtype=float), np.asarray(y))
    return m


def train_gradient_boosting(X, y, seed: int = DEFAULT_SEED, n_estimators: int = 100,
                            learning_rate: float = 0.1, max_depth: int = 3):
    m = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=seed,
    )
    m.fit(np.asarray(X, dtype=float), np.asarray(y))
    return m


DEFAULT_CTW_CUTOFF = 0.096      # median credit/width on SPY 2015-2023; see below


def simple_rule_predict(X, cutoff: float = DEFAULT_CTW_CUTOFF,
                        high: float = 0.6, low: float = 0.4):
    """No-ML baseline: a flat probability either side of a credit/width cutoff.

    The cutoff was 0.40, which is unreachable. A $5-wide 20-delta SPY put spread
    collects roughly 10% of its width, so measured credit/width never exceeds
    0.206 and its 99th percentile is 0.156 -- the rule returned `low` for every
    candidate and was a constant, not a baseline.

    0.096 is the median of the observed distribution, which splits the population
    in half. The direction is right: win rate rises monotonically with
    credit/width (bottom decile 56.5%, top decile 67.5%).

    Prefer `train_simple_rule`, which takes the cutoff from the training fold
    instead of this constant and so carries no in-sample knowledge at all.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] <= CREDIT_TO_WIDTH:
        raise ValueError(
            f"simple_rule_predict needs the full feature matrix "
            f"({len(FEATURE_INDEX)} columns); got {X.shape[1]}"
        )
    return np.where(X[:, CREDIT_TO_WIDTH] > cutoff, high, low)


class SimpleRule:
    """A credit/width threshold rule whose cutoff was fit on training data."""

    def __init__(self, cutoff: float, high: float = 0.6, low: float = 0.4):
        self.cutoff, self.high, self.low = cutoff, high, low

    def predict_proba(self, X) -> np.ndarray:
        p = simple_rule_predict(X, self.cutoff, self.high, self.low)
        return np.column_stack([1.0 - p, p])

    def __repr__(self) -> str:
        return f"SimpleRule(cutoff={self.cutoff:.4f})"


def train_simple_rule(X, y=None, quantile: float = 0.5) -> SimpleRule:
    """Fit the cutoff as a quantile of credit/width IN THE TRAINING FOLD.

    Taking the cutoff from the whole dataset would leak the test period's
    distribution into a baseline the model is judged against.
    """
    X = np.asarray(X, dtype=float)
    return SimpleRule(float(np.quantile(X[:, CREDIT_TO_WIDTH], quantile)))


def always_trade_predict(X):
    """Take every candidate. The 'does the model add anything?' control."""
    return np.ones(len(np.asarray(X)), dtype=float)


def no_trade_predict(X):
    """Trade nothing. Zero P&L is the bar a costly strategy must clear."""
    return np.zeros(len(np.asarray(X)), dtype=float)


def predict_proba(model, X) -> np.ndarray:
    """Positive-class probability from an sklearn model or a plain callable."""
    X = np.asarray(X, dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model(X), dtype=float).ravel()
