"""The daily 3:45 PM decision. Paper trading only -- nothing here places orders.

Order of gates, strictest last:
  candidate exists -> features complete -> no feature drift -> probability
  clears the frozen threshold -> expected value positive -> risk rules allow it

Every path returns a decision record, and NO_TRADE records carry the reason.
"""
from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np

from data.candidates import accepted_only, generate_candidates
from features.features import FEATURE_COLS, build_feature_vector
from models.select_threshold import expected_value, spread_payoffs
from reports.daily_report import daily_signal_json
from risk.risk import Portfolio, RiskConfig, evaluate_candidate, feature_drift_kill
from schemas import Candidate


def _placeholder(decision_time: datetime) -> Candidate:
    """A stand-in candidate so a no-candidate day still produces a record.

    Uses the decision date rather than `date.today()`: a log line whose expiry
    silently depends on when the report was rendered is not reproducible.
    """
    return Candidate(
        entry_time=decision_time, short_strike=0.0, long_strike=0.0,
        expiry=decision_time.date(), dte=0, entry_credit=0.0, width=0.0,
        underlying_price=0.0, short_delta=0.0, short_iv=0.0,
        short_bid=0.0, short_ask=0.0,
    )


def _no_trade(candidate, reasons, proba=0.0, ev=0.0) -> dict:
    return daily_signal_json(candidate, proba, ev, "NO_TRADE", reasons)


def paper_signal(quotes, bars, vix_bars, decision_time, *, model, scaler,
                 threshold: float, config: dict | None = None,
                 calibrator=None, portfolio: Portfolio | None = None,
                 risk_config: RiskConfig | None = None,
                 calendar=None, train_mean=None, train_std=None) -> dict:
    """Produce today's decision record."""
    cfg = config or {}
    rc = risk_config or RiskConfig()
    pf = portfolio or Portfolio(account_size=rc.account_size)

    candidates = generate_candidates(quotes, decision_time, cfg, calendar=calendar)
    accepted = accepted_only(candidates)
    if not accepted:
        reasons = sorted({c.rejection_reason for c in candidates}) or ["no_candidate"]
        return _no_trade(candidates[0] if candidates else _placeholder(decision_time),
                         reasons)

    # Highest credit-to-width among the accepted candidates.
    cand = max(accepted, key=lambda c: c.entry_credit / c.width if c.width else 0.0)

    vec = build_feature_vector(cand, bars, vix_bars, decision_time, calendar)
    X = np.array([[vec[k] for k in FEATURE_COLS]], dtype=float)

    missing = [FEATURE_COLS[i] for i in np.where(np.isnan(X[0]))[0]]
    if missing:
        return _no_trade(cand, [f"incomplete_features:{','.join(missing[:4])}"])

    if train_mean is not None and train_std is not None:
        drifted, why = feature_drift_kill(train_mean, train_std, X[0])
        if drifted:
            return _no_trade(cand, [why])

    Xs = scaler.transform(X)
    proba = _proba(model, Xs)
    if calibrator is not None:
        proba = float(calibrator.transform([proba])[0])

    profit, loss = spread_payoffs(cand.entry_credit)
    costs = cfg.get("cost_per_share", 0.026)
    ev = expected_value(proba, profit, loss, costs)

    allow, risk_reasons = evaluate_candidate(
        cand, proba, ev, pf, rc, decision_time=decision_time,
        data_is_clean=all(q.is_clean for q in quotes) if cfg.get("require_clean_quotes", True) else True,
    )
    if proba < threshold:
        return _no_trade(cand, ["below_threshold"], proba, ev)
    if not allow:
        return _no_trade(cand, risk_reasons, proba, ev)

    return daily_signal_json(cand, proba, ev, "PAPER_TRADE",
                             ["passes probability threshold", "passes liquidity filter",
                              "passes risk limits"])


def _proba(model, Xs) -> float:
    """Positive-class probability from an sklearn model, a torch adapter or a callable."""
    if hasattr(model, "predict_proba"):
        out = np.asarray(model.predict_proba(Xs), dtype=float)
        return float(out[0, -1] if out.ndim == 2 else out.ravel()[0])
    return float(np.asarray(model(Xs), dtype=float).ravel()[0])
