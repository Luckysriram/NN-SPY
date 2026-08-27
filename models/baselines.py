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


def simple_rule_predict(X, cutoff: float = 0.4, high: float = 0.6, low: float = 0.4):
    """No-ML baseline: a flat probability either side of a credit/width cutoff."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] <= CREDIT_TO_WIDTH:
        raise ValueError(
            f"simple_rule_predict needs the full feature matrix "
            f"({len(FEATURE_INDEX)} columns); got {X.shape[1]}"
        )
    return np.where(X[:, CREDIT_TO_WIDTH] > cutoff, high, low)


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
