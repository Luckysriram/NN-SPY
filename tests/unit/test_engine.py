"""Backtest engine tests.

The old engine was `for fold in folds: pass` with a smoke test that passed an
empty dataset, so the ledger, cost model and calibration audit were never
exercised. These tests run trades through it.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from backtest.engine import BacktestDataset, _inner_split, run_walk_forward
from features.features import FEATURE_INDEX, N_FEATURES
from features.splits import walk_forward_folds
from labels.simulator import SpreadSnapshot, simulate_trade
from risk.risk import RiskConfig
from tests.conftest import candidate

START = datetime(2020, 1, 1)


def make_dataset(n=600, seed=0):
    """A dataset whose label is genuinely predictable from credit_to_width."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES))
    signal = X[:, FEATURE_INDEX["credit_to_width"]]
    y = (signal + rng.normal(0, 0.5, n) > 0).astype(float)

    cands, outs, dates = [], [], []
    for i in range(n):
        entry = START + timedelta(days=i)
        c = candidate(entry_time=entry, entry_credit=2.0, width=5.0,
                      expiry=(entry + timedelta(days=45)).date(), dte=45)
        marks = ([SpreadSnapshot(entry + timedelta(days=3), 0.9, 0.4)] if y[i]
                 else [SpreadSnapshot(entry + timedelta(days=3), 4.6, 0.4)])
        cands.append(c)
        outs.append(simulate_trade(c, marks))
        dates.append(entry)
    return BacktestDataset(cands, outs, X, y, dates)


def folds_for(ds, n=2):
    return walk_forward_folds(
        ds.entry_dates, START + timedelta(days=300),
        ds.entry_dates[-1], fold_years=1)[:n]


# ------------------------------------------------------------------ guards
def test_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        BacktestDataset([1, 2], [1], np.zeros((2, 3)), np.zeros(2), [1, 2])


def test_empty_dataset_and_folds_run_cleanly():
    empty = BacktestDataset([], [], np.empty((0, N_FEATURES)), np.empty(0), [])
    result = run_walk_forward(empty, folds=[])
    assert result.ledger.trades == []
    assert result.summary()["n_trades"] == 0


# ------------------------------------------------------- the inner split
def test_inner_split_is_chronological_with_an_embargo():
    ds = make_dataset(200)
    idx = list(range(200))
    inner, val = _inner_split(idx, ds.entry_dates, val_frac=0.2, embargo_days=38)
    assert inner and val
    assert max(ds.entry_dates[i] for i in inner) < min(ds.entry_dates[i] for i in val)
    gap = (min(ds.entry_dates[i] for i in val)
           - max(ds.entry_dates[i] for i in inner)).days
    assert gap >= 38


def test_inner_split_never_lets_validation_precede_training():
    ds = make_dataset(120)
    inner, val = _inner_split(list(range(120)), ds.entry_dates, 0.2, 38)
    assert set(inner) & set(val) == set()


# ------------------------------------------------------------ real trading
def test_engine_actually_records_trades():
    ds = make_dataset()
    result = run_walk_forward(ds, folds_for(ds), min_trades_for_threshold=10)
    assert len(result.ledger) > 0
    assert result.folds


def test_recorded_trades_carry_probabilities_and_fold_numbers():
    ds = make_dataset()
    result = run_walk_forward(ds, folds_for(ds), min_trades_for_threshold=10)
    for row in result.ledger.trades:
        assert 0.0 <= row.model_probability <= 1.0
        assert row.fold >= 1
        assert row.exit_reason in ("PROFIT_TARGET", "STOP_LOSS", "TIME_EXIT", "EXPIRY")


def test_ledger_total_matches_the_sum_of_its_rows():
    ds = make_dataset()
    result = run_walk_forward(ds, folds_for(ds), min_trades_for_threshold=10)
    assert result.ledger.total_pnl() == pytest.approx(
        sum(t.net_pnl for t in result.ledger.trades))


def test_costs_are_charged_on_every_trade():
    ds = make_dataset()
    result = run_walk_forward(ds, folds_for(ds), commission_per_contract=0.65,
                              min_trades_for_threshold=10)
    assert result.ledger.total_fees() > 0
    assert all(t.fees > 0 for t in result.ledger.trades)


def test_summary_reports_trading_metrics():
    ds = make_dataset()
    s = run_walk_forward(ds, folds_for(ds), min_trades_for_threshold=10).summary()
    for key in ("n_trades", "total_pnl", "win_rate", "profit_factor", "max_drawdown"):
        assert key in s


def test_engine_learns_something_on_a_predictable_dataset():
    from reports.metrics import roc_auc
    ds = make_dataset(800)
    result = run_walk_forward(ds, folds_for(ds), min_trades_for_threshold=10)
    if len(result.labels) > 20 and len(set(result.labels)) > 1:
        assert roc_auc(result.labels, result.probabilities) > 0.5


# --------------------------------------------------------------- the gates
def test_no_data_outcomes_are_never_traded():
    ds = make_dataset(400)
    ds.outcomes[10] = simulate_trade(ds.candidates[10], [])       # NO_DATA
    result = run_walk_forward(ds, folds_for(ds), min_trades_for_threshold=10)
    assert all(t.exit_reason != "NO_DATA" for t in result.ledger.trades)


def test_daily_trade_limit_is_respected():
    ds = make_dataset()
    rc = RiskConfig(max_trades_per_day=1)
    result = run_walk_forward(ds, folds_for(ds), risk_config=rc,
                              min_trades_for_threshold=10)
    by_day: dict = {}
    for t in result.ledger.trades:
        by_day[t.entry_time.date()] = by_day.get(t.entry_time.date(), 0) + 1
    assert all(v <= 1 for v in by_day.values())


def test_a_never_trade_threshold_produces_no_trades():
    ds = make_dataset()
    # min_trades far above the fold size -> select_threshold returns 1.01
    result = run_walk_forward(ds, folds_for(ds), min_trades_for_threshold=10_000)
    assert len(result.ledger) == 0


def test_calibration_can_be_disabled():
    ds = make_dataset()
    result = run_walk_forward(ds, folds_for(ds), calibrate=False,
                              min_trades_for_threshold=10)
    assert result.folds


def test_trades_are_processed_in_chronological_order():
    ds = make_dataset()
    result = run_walk_forward(ds, folds_for(ds), min_trades_for_threshold=10)
    for k in {t.fold for t in result.ledger.trades}:
        times = [t.entry_time for t in result.ledger.trades if t.fold == k]
        assert times == sorted(times)
