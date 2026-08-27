import numpy as np
import pytest

from models.select_threshold import (breakeven_win_rate, decide, expected_value,
                                     select_threshold, spread_payoffs,
                                     threshold_table)


def test_spread_payoffs_follow_the_exit_rules():
    """Close at 50% of credit to win; at 2x credit the loss is 1x credit."""
    profit, loss = spread_payoffs(2.00)
    assert profit == pytest.approx(1.00)
    assert loss == pytest.approx(2.00)


def test_breakeven_win_rate_is_not_fifty_percent():
    """The old code hardcoded profit=loss=1.0, implying a 50% break-even."""
    profit, loss = spread_payoffs(2.00)
    assert breakeven_win_rate(profit, loss) == pytest.approx(2 / 3, abs=1e-9)


def test_breakeven_rises_with_costs():
    profit, loss = spread_payoffs(2.00)
    assert breakeven_win_rate(profit, loss, costs=0.10) > breakeven_win_rate(profit, loss)


def test_select_threshold_picks_best_on_validation():
    p = np.array([0.1, 0.2, 0.9, 0.95] * 20)
    y = np.array([0.0, 0.0, 1.0, 1.0] * 20)
    t = select_threshold(p, y, costs=0.1, min_trades=10)
    assert 0.2 < t < 0.9


def test_select_threshold_respects_the_minimum_trade_count():
    """Without a floor the search maximises a statistic over one lucky trade."""
    p = np.array([0.1, 0.2, 0.9, 0.95])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert select_threshold(p, y, min_trades=100) == pytest.approx(1.01)


def test_threshold_of_1_01_means_never_trade():
    p = np.full(200, 0.9)
    y = np.zeros(200)                     # every trade loses
    assert select_threshold(p, y, min_trades=10) == pytest.approx(1.01)


def test_a_55_percent_win_rate_is_rejected_under_real_payoffs():
    """55% clears a 50% break-even but not the true 66.7% one."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 0.95, 400)
    y = (rng.uniform(size=400) < 0.55).astype(float)   # win rate 55%, unrelated to p
    profit, loss = spread_payoffs(2.00)
    assert select_threshold(p, y, expected_profit=profit, expected_loss=loss,
                            min_trades=30) == pytest.approx(1.01)


def test_realized_pnl_objective_is_used_when_supplied():
    p = np.linspace(0.1, 0.95, 200)
    y = (p > 0.6).astype(float)
    pnl = np.where(y > 0, 1.0, -2.0)
    t = select_threshold(p, y, pnl=pnl, min_trades=20)
    assert 0.4 < t < 0.75


def test_threshold_table_reports_trade_counts_and_win_rates():
    p = np.array([0.1, 0.5, 0.9])
    y = np.array([0.0, 1.0, 1.0])
    rows = threshold_table(p, y, grid=[0.05, 0.6])
    assert rows[0]["n_trades"] == 3
    assert rows[1]["n_trades"] == 1
    assert rows[1]["win_rate"] == 1.0


def test_expected_value_formula():
    assert expected_value(0.6, 10.0, 5.0, 1.0) == pytest.approx(0.6 * 10 - 0.4 * 5 - 1.0)


def test_decide_requires_both_probability_and_expected_value():
    ok, reasons = decide(0.8, 0.6, ev=1.0)
    assert ok and reasons == ["passes probability threshold"]

    ok, reasons = decide(0.5, 0.6, ev=1.0)
    assert not ok and "below_threshold" in reasons

    ok, reasons = decide(0.8, 0.6, ev=-1.0)
    assert not ok and "expected_value" in reasons
