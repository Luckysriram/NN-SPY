"""Full-pipeline integration test on a synthetic dataset.

Runs every module in order -- normalize, validate, candidates, simulate,
features, split, train, calibrate, backtest, report -- and asserts the
properties that must hold across the whole chain rather than inside any one
module. The most important of those is the leakage check.
"""
from datetime import date, datetime

import numpy as np
import pytest

from backtest.engine import run_walk_forward
from data.adapters.feasibility_check import assess
from data.events import EventCalendar
from features.features import N_FEATURES
from features.scalers import FitScaler
from features.splits import assert_split_is_clean, walk_forward_folds
from models.baselines import train_gradient_boosting, train_logistic
from pipeline import build_dataset
from reports.metrics import roc_auc
from reports.monthly_report import monthly_report, render
from schemas import EMBARGO_DAYS
from tests.fixtures.make_fixture import make_fixture, to_bars

CFG = {"min_oi": 100, "min_credit": 0.20, "slippage_per_leg": 0.025}


def wide_calendar() -> EventCalendar:
    years = (2022, 2023, 2024, 2025, 2026)
    return EventCalendar(
        dates={
            "FOMC": frozenset(date(y, m, 15) for y in years for m in (3, 6, 9, 12)),
            "CPI": frozenset(date(y, m, 12) for y in years for m in (2, 8)),
            "JOBS": frozenset(date(y, m, 5) for y in years for m in (4, 10)),
        },
        coverage_start=date(2020, 1, 1), coverage_end=date(2028, 1, 1),
        source="test")


@pytest.fixture(scope="module")
def built():
    u, v, c = make_fixture(n_days=900, seed=7, warmup_days=300)
    ds, report = build_dataset(
        c, to_bars(u), to_bars(v, "^VIX"), config=CFG, calendar=wide_calendar(),
        fees_per_spread=0.026, slippage=0.02)
    return ds, report, (u, v, c)


# --------------------------------------------------------------- the build
def test_feasibility_check_accepts_the_fixture(built):
    _, _, (_, _, chain_rows) = built
    rep = assess(chain_rows[:20000])
    assert rep.has_greeks and rep.has_iv
    assert rep.days_with_full_spread > 0


def test_pipeline_produces_a_usable_dataset(built):
    ds, report, _ = built
    assert len(ds) > 50, report.render()
    assert ds.X.shape[1] == N_FEATURES
    assert set(np.unique(ds.y)) <= {0.0, 1.0}


def test_no_nan_survives_into_the_training_matrix(built):
    ds, _, _ = built
    assert not np.isnan(ds.X).any()


def test_every_rejection_is_accounted_for(built):
    _, report, _ = built
    assert sum(report.rejections.values()) > 0
    assert "_accepted" in report.rejections


def test_no_data_trades_are_excluded_from_the_dataset(built):
    ds, _, _ = built
    assert all(o.exit_reason != "NO_DATA" for o in ds.outcomes)


def test_labels_match_their_outcomes(built):
    ds, _, _ = built
    for outcome, label in zip(ds.outcomes, ds.y):
        assert label == (1.0 if outcome.exit_reason == "PROFIT_TARGET" else 0.0)


def test_both_classes_are_present(built):
    """A single-class dataset makes every classification metric meaningless."""
    ds, _, _ = built
    assert 0.05 < float(ds.y.mean()) < 0.95


# ------------------------------------------------------------- no leakage
def test_walk_forward_folds_are_leak_free(built):
    ds, _, _ = built
    folds = walk_forward_folds(ds.entry_dates, ds.entry_dates[len(ds) // 3],
                               ds.entry_dates[-1])
    assert folds
    for train_idx, test_idx in folds:
        assert_split_is_clean(ds.entry_dates, train_idx, test_idx)


def test_no_training_trade_is_still_open_when_testing_starts(built):
    """The exact leak the embargo exists to prevent, checked with real exit dates."""
    ds, _, _ = built
    folds = walk_forward_folds(ds.entry_dates, ds.entry_dates[len(ds) // 3],
                               ds.entry_dates[-1])
    exits = [o.exit_time for o in ds.outcomes]
    for train_idx, test_idx in folds:
        assert_split_is_clean(ds.entry_dates, train_idx, test_idx, exit_dates=exits)


def test_features_never_use_a_bar_after_the_decision(built):
    ds, _, (u, v, _) = built
    from features.features import assert_no_lookahead
    bars = to_bars(u)
    for entry in ds.entry_dates[:20]:
        assert_no_lookahead([b for b in bars if b.timestamp <= entry], entry)


# --------------------------------------------------------------- modelling
def test_baselines_train_and_score(built):
    ds, _, _ = built
    cut = int(len(ds) * 0.7)
    # Scale first, fit on the training slice only. Logistic regression will not
    # converge on raw features here -- open interest is in the thousands while
    # delta is in [-1, 0] -- which is why the engine always scales before it
    # trains rather than leaving it to the caller.
    scaler = FitScaler().fit(ds.X[:cut]).freeze()
    for train in (train_logistic, train_gradient_boosting):
        model = train(scaler.transform(ds.X[:cut]), ds.y[:cut])
        p = model.predict_proba(scaler.transform(ds.X[cut:]))[:, 1]
        assert p.shape == (len(ds) - cut,)
        assert ((p >= 0) & (p <= 1)).all()


def test_baselines_are_reproducible(built):
    ds, _, _ = built
    a = train_gradient_boosting(ds.X, ds.y, seed=3).predict_proba(ds.X)[:, 1]
    b = train_gradient_boosting(ds.X, ds.y, seed=3).predict_proba(ds.X)[:, 1]
    assert np.allclose(a, b)


def test_full_walk_forward_backtest_runs(built):
    ds, _, _ = built
    folds = walk_forward_folds(ds.entry_dates, ds.entry_dates[len(ds) // 3],
                               ds.entry_dates[-1])
    result = run_walk_forward(ds, folds, commission_per_contract=0.65,
                              min_trades_for_threshold=5)
    assert result.folds
    summary = result.summary()
    assert summary["n_trades"] == len(result.ledger)
    # Every recorded trade paid its costs and has a real exit.
    for row in result.ledger.trades:
        assert row.fees > 0
        assert row.max_risk_dollars > 0
        assert row.gross_pnl - row.fees == pytest.approx(row.net_pnl)


def test_report_renders_from_backtest_output(built):
    ds, _, _ = built
    folds = walk_forward_folds(ds.entry_dates, ds.entry_dates[len(ds) // 3],
                               ds.entry_dates[-1])
    result = run_walk_forward(ds, folds, min_trades_for_threshold=5)
    if len(result.ledger) < 2:
        pytest.skip("threshold selected no trades on this fixture")
    rep = monthly_report(result.ledger.trades, result.probabilities, result.labels,
                         pnl=result.ledger.pnl_series())
    assert rep["trades"] == len(result.ledger)
    assert "by_prob_bucket" in rep
    text = render(rep)
    assert "CALIBRATION AUDIT" in text


def test_a_model_cannot_beat_chance_on_pure_noise():
    """A sanity check on the metric itself: shuffled labels must score ~0.5."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 10))
    y = rng.integers(0, 2, 400).astype(float)
    model = train_logistic(X[:300], y[:300])
    auc = roc_auc(y[300:], model.predict_proba(X[300:])[:, 1])
    assert 0.3 < auc < 0.7


def test_embargo_constant_is_wide_enough_for_the_fixture(built):
    ds, _, _ = built
    held = [(o.exit_time - c.entry_time).days
            for c, o in zip(ds.candidates, ds.outcomes)]
    assert max(held) <= EMBARGO_DAYS
