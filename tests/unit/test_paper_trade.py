"""Paper-trade pipeline tests.

The previous version of this test asserted `decision in ("PAPER_TRADE",
"NO_TRADE")` -- which is every possible value -- on a fixture whose candidate was
rejected by the event gate before the model ran. It passed while hiding two
crashes on the path it never reached. These tests reach the model.
"""
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from features.features import N_FEATURES
from features.scalers import FitScaler
from paper_trade.daily_signal import paper_signal
from risk.risk import Portfolio, RiskConfig
from schemas import UnderlyingBar
from tests.conftest import dated_calendar, open_calendar, quote

TS = datetime(2024, 1, 30, 15, 45)
EXPIRY = date(2024, 3, 15)          # 45 DTE
CFG = {"min_oi": 100, "slippage_per_leg": 0.025, "min_credit": 0.20}


def chain(ts=TS):
    return [
        quote(timestamp=ts, expiry=EXPIRY, dte=45, strike=440.0, delta=-0.20,
              bid=2.0, ask=2.4, open_interest=1000, underlying_price=450.0),
        quote(timestamp=ts, expiry=EXPIRY, dte=45, strike=435.0, delta=-0.14,
              bid=0.8, ask=1.2, open_interest=1000, underlying_price=450.0),
    ]


def history(n=300, symbol="SPY", close=450.0):
    return [UnderlyingBar(timestamp=TS - timedelta(days=n - i), symbol=symbol,
                          open=close, high=close, low=close,
                          close=close + (i % 7) * 0.5, adj_close=close,
                          volume=1_000_000)
            for i in range(n)]


class DummyModel:
    """sklearn-shaped: predict_proba returns (n, 2)."""
    def __init__(self, p=0.7):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1.0 - self.p, self.p]] * len(X))


def fitted_scaler():
    rng = np.random.default_rng(0)
    return FitScaler().fit(rng.normal(size=(50, N_FEATURES))).freeze()


def run(model=None, threshold=0.5, **kw):
    return paper_signal(
        chain(), history(), history(symbol="^VIX", close=16.0), TS,
        model=model or DummyModel(), scaler=fitted_scaler(), threshold=threshold,
        config=CFG, calendar=dated_calendar(), **kw)


# ------------------------------------------------------------- the happy path
def test_paper_signal_reaches_the_model_and_can_return_paper_trade():
    sig = run()
    assert sig["decision"] == "PAPER_TRADE", sig["reasons"]
    assert sig["model_probability"] == pytest.approx(0.7)
    assert sig["short_strike"] == 440


def test_scaler_and_feature_vector_agree_on_width():
    """The old fixture fit a 30-column scaler and fed it 32 columns."""
    sig = run()
    assert sig["decision"] == "PAPER_TRADE"


def test_signal_includes_a_timestamp_and_expected_value():
    sig = run()
    assert "timestamp" in sig and "expected_value" in sig


# ------------------------------------------------------------------ the gates
def test_below_threshold_is_no_trade():
    sig = run(model=DummyModel(0.30), threshold=0.60)
    assert sig["decision"] == "NO_TRADE"
    assert sig["reasons"] == ["below_threshold"]


def test_no_candidate_is_recorded_with_the_rejection_reason():
    sig = paper_signal([], history(), history(symbol="^VIX"), TS,
                       model=DummyModel(), scaler=fitted_scaler(), threshold=0.5,
                       config=CFG, calendar=dated_calendar())
    assert sig["decision"] == "NO_TRADE"
    assert sig["reasons"] == ["no_candidate"]


def test_event_gate_reports_the_event_that_blocked_the_trade():
    cal = dated_calendar(FOMC=[date(2024, 2, 20)])
    sig = paper_signal(chain(), history(), history(symbol="^VIX"), TS,
                       model=DummyModel(), scaler=fitted_scaler(), threshold=0.5,
                       config=CFG, calendar=cal)
    assert sig["decision"] == "NO_TRADE"
    assert sig["reasons"] == ["event_FOMC"]


def test_incomplete_features_block_the_trade_rather_than_being_zero_filled():
    sig = paper_signal(chain(), history(n=5), [], TS, model=DummyModel(),
                       scaler=fitted_scaler(), threshold=0.5, config=CFG,
                       calendar=dated_calendar())
    assert sig["decision"] == "NO_TRADE"
    assert sig["reasons"][0].startswith("incomplete_features")


def test_feature_drift_blocks_the_trade():
    sig = run(train_mean=np.full(N_FEATURES, 1e6), train_std=np.ones(N_FEATURES))
    assert sig["decision"] == "NO_TRADE"
    assert "drift" in sig["reasons"][0]


def test_risk_limits_block_the_trade():
    pf = Portfolio(account_size=100_000.0)
    pf.add_open_risk(99_000.0)
    sig = run(portfolio=pf, risk_config=RiskConfig())
    assert sig["decision"] == "NO_TRADE"
    assert "max_open_risk" in sig["reasons"]


def test_no_candidate_record_is_deterministic():
    """A placeholder expiry from date.today() makes the log unreproducible."""
    a = paper_signal([], history(), [], TS, model=DummyModel(),
                     scaler=fitted_scaler(), threshold=0.5, config=CFG,
                     calendar=dated_calendar())
    assert a["expiration"] == TS.date().isoformat()


def test_accepts_a_plain_callable_model():
    sig = run(model=lambda X: np.array([0.9]))
    assert sig["decision"] == "PAPER_TRADE"
