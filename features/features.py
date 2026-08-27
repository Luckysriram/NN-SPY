"""Timestamped feature engineering.

Two rules hold everywhere in this module:

* NO LOOKAHEAD. Every function takes an `as_of` and uses only bars stamped at or
  before it. `underlying_features` filters before it computes, never after.
* MISSING IS NaN, NEVER ZERO. A 20-day moving average with 5 bars of history is
  not "0.0 distance from the MA" -- it is unknown. Zero-filling it teaches the
  model that early-history rows are all sitting exactly on their moving average.
  Callers drop or impute NaN explicitly; `assert_no_nan` makes that a decision
  rather than an accident.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from data.events import (EVENT_FEATURE_SUFFIX, MAJOR_EVENTS, days_to_event,
                         load_event_calendar)
from schemas import Candidate, UnderlyingBar

NAN = float("nan")

MARKET_COLS = [
    "spy_close", "spy_return_1d", "spy_return_5d", "spy_return_21d",
    "spy_realized_vol_5d", "spy_realized_vol_21d",
    "spy_distance_from_ma20", "spy_distance_from_ma50", "spy_drawdown_63d",
    "vix_close", "vix_change_5d", "vix_percentile_252d",
]
SPREAD_COLS = [
    "dte", "short_put_delta", "short_put_iv", "short_put_bid_ask_pct",
    "short_put_open_interest", "short_put_volume", "long_put_delta",
    "spread_width", "credit", "credit_to_width", "break_even_distance_pct",
    "net_delta", "net_theta", "net_vega", "net_gamma", "spread_bid_ask_pct",
]
CALENDAR_COLS = ["weekday", "month"] + [
    f"days_to_{EVENT_FEATURE_SUFFIX[e]}" for e in MAJOR_EVENTS
]

FEATURE_COLS = MARKET_COLS + SPREAD_COLS + CALENDAR_COLS

# Name -> column position. Every consumer indexes through this, so a column can
# be inserted without silently repointing a downstream rule at its neighbour.
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_COLS)}
N_FEATURES = len(FEATURE_COLS)

MIN_HISTORY_BARS = 63          # below this, long-window features stay NaN


def _past_closes(bars, as_of: datetime):
    past = sorted((b for b in bars if b.timestamp <= as_of), key=lambda b: b.timestamp)
    return past, np.array([b.close for b in past], dtype=float)


def underlying_features(bars, as_of: datetime, vix_bars=None) -> dict:
    """Market features from bars known at or before `as_of`."""
    feat = {c: NAN for c in MARKET_COLS}
    _, closes = _past_closes(bars, as_of)
    if len(closes) == 0:
        return feat

    feat["spy_close"] = float(closes[-1])
    for p in (1, 5, 21):
        if len(closes) > p and closes[-1 - p] > 0:
            feat[f"spy_return_{p}d"] = float((closes[-1] - closes[-1 - p]) / closes[-1 - p])

    if len(closes) >= 2:
        ratio = closes[1:] / closes[:-1]
        logret = np.log(np.where(ratio > 0, ratio, np.nan))
        for w in (5, 21):
            if len(logret) >= w and not np.isnan(logret[-w:]).any():
                feat[f"spy_realized_vol_{w}d"] = float(np.std(logret[-w:], ddof=1) * np.sqrt(252))

    for ma in (20, 50):
        if len(closes) >= ma:
            mavg = float(np.mean(closes[-ma:]))
            if mavg > 0:
                feat[f"spy_distance_from_ma{ma}"] = float((closes[-1] - mavg) / mavg)

    if len(closes) >= 63:
        peak = float(np.max(closes[-63:]))
        if peak > 0:
            feat["spy_drawdown_63d"] = float(closes[-1] / peak - 1.0)

    if vix_bars:
        _, vc = _past_closes(vix_bars, as_of)
        if len(vc):
            feat["vix_close"] = float(vc[-1])
            if len(vc) > 5 and vc[-6] > 0:
                feat["vix_change_5d"] = float((vc[-1] - vc[-6]) / vc[-6])
            if len(vc) >= 252:
                feat["vix_percentile_252d"] = float(np.mean(vc[-252:] <= vc[-1]))
    return feat


def spread_features(c: Candidate) -> dict:
    """Option and spread features, all sourced from the candidate's real legs.

    Position convention: short 1 put, long 1 put. A net greek is therefore
    `-short + long`, which makes net_theta positive (decay helps a credit
    spread) and net_vega negative (rising IV hurts it).
    """
    short_ba = c.short_ask - c.short_bid
    long_ba = c.long_ask - c.long_bid
    spread_mid = c.short_mid - c.long_mid

    return {
        "dte": float(c.dte),
        "short_put_delta": c.short_delta,
        "short_put_iv": c.short_iv,
        # bid-ask as a fraction of the leg's own mid -- a real percentage.
        # (The old version divided a dollar spread by implied vol.)
        "short_put_bid_ask_pct": short_ba / c.short_mid if c.short_mid > 0 else NAN,
        "short_put_open_interest": float(c.short_open_interest),
        "short_put_volume": float(c.short_volume),
        "long_put_delta": c.long_delta,
        "spread_width": c.width,
        "credit": c.entry_credit,
        "credit_to_width": c.entry_credit / c.width if c.width else NAN,
        "break_even_distance_pct": (
            (c.break_even - c.underlying_price) / c.underlying_price
            if c.underlying_price > 0 else NAN
        ),
        "net_delta": -c.short_delta + c.long_delta,
        "net_theta": -c.short_theta + c.long_theta,
        "net_vega": -c.short_vega + c.long_vega,
        "net_gamma": -c.short_gamma + c.long_gamma,
        "spread_bid_ask_pct": (short_ba + long_ba) / spread_mid if spread_mid > 0 else NAN,
    }


def calendar_features(decision_time: datetime, calendar=None) -> dict:
    """Weekday, month, and signed days to each major event.

    Emits exactly the CALENDAR_COLS names -- the mapping lives in
    `data.events.EVENT_FEATURE_SUFFIX` so the two lists cannot drift.
    Outside the calendar's coverage window every event feature is NaN.
    """
    cal = calendar if calendar is not None else load_event_calendar()
    d = decision_time.date()
    feat = {"weekday": float(decision_time.weekday()),
            "month": float(decision_time.month)}
    covered = cal.covers(d)
    for event in MAJOR_EVENTS:
        col = f"days_to_{EVENT_FEATURE_SUFFIX[event]}"
        if not covered:
            feat[col] = NAN
            continue
        n = days_to_event(d, cal.for_event(event))
        feat[col] = NAN if n is None else float(n)
    return feat


def build_feature_vector(candidate: Candidate, bars, vix_bars, as_of: datetime,
                         calendar=None) -> dict:
    """Merge the three feature groups into one vector keyed by FEATURE_COLS."""
    feat: dict = {}
    feat.update(underlying_features(bars, as_of, vix_bars))
    feat.update(spread_features(candidate))
    feat.update(calendar_features(as_of, calendar))
    missing = set(FEATURE_COLS) - set(feat)
    if missing:
        raise KeyError(f"feature builders did not produce: {sorted(missing)}")
    return {k: feat[k] for k in FEATURE_COLS}


def to_matrix(vectors) -> np.ndarray:
    """Stack feature dicts into an (n, N_FEATURES) array in FEATURE_COLS order."""
    return np.array([[v[k] for k in FEATURE_COLS] for v in vectors], dtype=float)


def nan_report(X: np.ndarray) -> dict:
    """Per-column NaN counts -- run this before training, not after."""
    X = np.asarray(X, dtype=float)
    counts = np.isnan(X).sum(axis=0)
    return {FEATURE_COLS[i]: int(n) for i, n in enumerate(counts) if n}


def assert_no_nan(X: np.ndarray) -> None:
    bad = nan_report(X)
    if bad:
        raise ValueError(
            f"NaN features present: {bad}. Drop those rows or impute deliberately "
            "-- do not let a scaler turn them into zeros."
        )


def drop_nan_rows(X: np.ndarray, y=None):
    """Keep only complete rows. Returns (X, y, kept_mask)."""
    X = np.asarray(X, dtype=float)
    keep = ~np.isnan(X).any(axis=1)
    if y is None:
        return X[keep], None, keep
    return X[keep], np.asarray(y)[keep], keep


def assert_no_lookahead(bars, as_of: datetime) -> None:
    """Fail loudly if any bar handed to a feature builder postdates the decision."""
    future = [b.timestamp for b in bars if b.timestamp > as_of]
    if future:
        raise ValueError(
            f"{len(future)} bar(s) after as_of={as_of} reached a feature builder "
            f"(first: {min(future)}). This is lookahead."
        )
