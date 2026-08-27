"""Classification, calibration and trading metrics.

Read every classification number against the base rate. On a strategy that wins
70% of the time by construction, 70% accuracy is what you get for predicting
"win" unconditionally -- so accuracy is reported, never celebrated.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def roc_auc(y, proba) -> float:
    y = _arr(y)
    if len(np.unique(y)) < 2:
        return float("nan")          # undefined with one class; do not return 0.5
    return float(roc_auc_score(y, _arr(proba)))


def pr_auc(y, proba) -> float:
    y = _arr(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, _arr(proba)))


def brier_score(y, proba) -> float:
    return float(brier_score_loss(_arr(y), _arr(proba)))


def base_rate(y) -> float:
    y = _arr(y)
    return float(y.mean()) if len(y) else float("nan")


def accuracy(y, proba, threshold: float = 0.5) -> float:
    return float(((_arr(proba) >= threshold).astype(float) == _arr(y)).mean())


def calibration_bins(y, proba, bins: int = 10):
    """[(bin_center, mean_predicted, actual_frequency, n), ...] for occupied bins."""
    y, p = _arr(y), _arr(proba)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        if not m.any():
            continue
        out.append(((lo + hi) / 2.0, float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def expected_calibration_error(y, proba, bins: int = 10) -> float:
    """Weighted mean gap between predicted and realised frequency.

    This is the number that answers the spec's question directly: when the model
    says 70%, do 70% of those trades win?
    """
    rows = calibration_bins(y, proba, bins)
    n = len(_arr(y))
    if not rows or n == 0:
        return float("nan")
    return float(sum(cnt / n * abs(pred - act) for _, pred, act, cnt in rows))


def profit_factor(pnl) -> float:
    pnl = _arr(pnl)
    pnl = pnl[~np.isnan(pnl)]
    gross = pnl[pnl > 0].sum()
    loss = -pnl[pnl < 0].sum()
    if loss <= 0:
        return float("inf") if gross > 0 else float("nan")
    return float(gross / loss)


def win_rate(pnl) -> float:
    pnl = _arr(pnl)
    pnl = pnl[~np.isnan(pnl)]
    return float((pnl > 0).mean()) if len(pnl) else float("nan")


def max_drawdown(pnl) -> float:
    """Worst peak-to-trough drop of the cumulative P&L curve. Returns <= 0."""
    pnl = _arr(pnl)
    pnl = pnl[~np.isnan(pnl)]
    if not len(pnl):
        return float("nan")
    curve = np.cumsum(pnl)
    return float(np.min(curve - np.maximum.accumulate(curve)))


def sharpe(pnl, periods_per_year: int = 252) -> float:
    """Reported cautiously: option-spread P&L is skewed, so this flatters."""
    pnl = _arr(pnl)
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 2 or pnl.std(ddof=1) == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(periods_per_year))


def sortino(pnl, periods_per_year: int = 252) -> float:
    pnl = _arr(pnl)
    pnl = pnl[~np.isnan(pnl)]
    downside = pnl[pnl < 0]
    # len(downside) < 2, not "not len(downside)": std(ddof=1) on a single
    # observation divides by zero and warns before returning nan.
    if len(pnl) < 2 or len(downside) < 2 or downside.std(ddof=1) == 0:
        return float("nan")
    return float(pnl.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def classification_report(y, proba, threshold: float = 0.5) -> dict:
    return {
        "n": int(len(_arr(y))),
        "base_rate": base_rate(y),
        "roc_auc": roc_auc(y, proba),
        "pr_auc": pr_auc(y, proba),
        "brier": brier_score(y, proba),
        "ece": expected_calibration_error(y, proba),
        "accuracy": accuracy(y, proba, threshold),
    }


def trading_report(pnl) -> dict:
    pnl = _arr(pnl)
    clean = pnl[~np.isnan(pnl)]
    return {
        "n_trades": int(len(clean)),
        "total_pnl": float(clean.sum()) if len(clean) else 0.0,
        "avg_pnl": float(clean.mean()) if len(clean) else float("nan"),
        "win_rate": win_rate(clean),
        "profit_factor": profit_factor(clean),
        "max_drawdown": max_drawdown(clean),
        "sharpe": sharpe(clean),
        "sortino": sortino(clean),
    }
