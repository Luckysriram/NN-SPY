from datetime import datetime

import numpy as np
import pytest

from risk.risk import (Portfolio, RiskConfig, evaluate_candidate,
                       feature_drift_kill, position_size, should_kill)
from tests.conftest import candidate

DAY = datetime(2024, 1, 3, 15, 45)


def cfg(**kw):
    base = dict(account_size=100_000.0, max_loss_per_trade_pct=0.01,
                max_open_risk_pct=0.05, max_trades_per_day=2,
                kill_daily_loss_pct=0.03, kill_weekly_loss_pct=0.06)
    base.update(kw)
    return RiskConfig(**base)


def test_config_loads_from_the_yaml_shape():
    from config import load_named
    rc = RiskConfig.from_dict(load_named("risk"))
    assert rc.max_open_risk_pct > 0 and rc.paper_trade_only is True


def test_evaluate_candidate_allows_a_good_trade():
    allow, reasons = evaluate_candidate(candidate(), 0.8, 5.0, Portfolio(), cfg(),
                                        decision_time=DAY)
    assert allow, reasons


def test_evaluate_candidate_blocks_when_max_open_risk_exceeded():
    p = Portfolio()
    p.add_open_risk(6_000.0)              # 6% of 100k, cap is 5%
    allow, reasons = evaluate_candidate(candidate(), 0.8, 1.0, p, cfg(),
                                        decision_time=DAY)
    assert not allow and "max_open_risk" in reasons


def test_evaluate_candidate_blocks_low_ev():
    allow, reasons = evaluate_candidate(candidate(), 0.5, -0.5, Portfolio(), cfg(),
                                        decision_time=DAY)
    assert not allow and "expected_value" in reasons


def test_max_trades_per_day_is_actually_enforced():
    """The old config carried this limit and nothing ever read it."""
    p = Portfolio()
    c = cfg(max_trades_per_day=2)
    p.record_entry(DAY, 100.0)
    p.record_entry(DAY, 100.0)
    allow, reasons = evaluate_candidate(candidate(), 0.9, 5.0, p, c, decision_time=DAY)
    assert not allow and "max_trades_per_day" in reasons


def test_projected_open_risk_blocks_a_trade_that_would_breach_the_cap():
    p = Portfolio()
    p.add_open_risk(4_900.0)              # just under 5%
    allow, reasons = evaluate_candidate(candidate(), 0.9, 5.0, p, cfg(),
                                        decision_time=DAY)
    assert not allow and "max_open_risk" in reasons


def test_dirty_data_blocks_the_trade():
    allow, reasons = evaluate_candidate(candidate(), 0.9, 5.0, Portfolio(), cfg(),
                                        decision_time=DAY, data_is_clean=False)
    assert not allow and "data_quality" in reasons


def test_a_rejected_candidate_is_never_allowed():
    c = candidate(rejection_reason="oi_too_low")
    allow, reasons = evaluate_candidate(c, 0.9, 5.0, Portfolio(), cfg(),
                                        decision_time=DAY)
    assert not allow and "candidate_oi_too_low" in reasons


def test_an_inverted_spread_is_flagged_as_undefined_risk():
    c = candidate(short_strike=395.0, long_strike=400.0)
    allow, reasons = evaluate_candidate(c, 0.9, 5.0, Portfolio(), cfg(),
                                        decision_time=DAY)
    assert not allow and "undefined_risk" in reasons


def test_position_size_respects_the_per_trade_loss_cap():
    """$300 risk per spread, $1000 cap -> 3 contracts."""
    assert position_size(candidate(entry_credit=2.0, width=5.0), cfg()) == 3


def test_position_size_is_zero_when_one_spread_exceeds_the_cap():
    assert position_size(candidate(entry_credit=2.0, width=5.0),
                         cfg(max_loss_per_trade_pct=0.001)) == 0


def test_kill_switch_on_daily_loss():
    p = Portfolio()
    p.add_pnl(DAY, -4_000.0)              # -4% in one day, limit is 3%
    kill, reason = should_kill(p, cfg())
    assert kill and "daily" in reason


def test_kill_switch_on_weekly_loss():
    from datetime import timedelta
    p = Portfolio()
    for i in range(4):
        p.add_pnl(DAY + timedelta(days=i), -1_800.0)   # -7.2% over the week
    kill, reason = should_kill(p, cfg(), DAY + timedelta(days=3))
    assert kill and "weekly" in reason


def test_no_kill_on_a_normal_day():
    p = Portfolio()
    p.add_pnl(DAY, -500.0)
    assert should_kill(p, cfg()) == (False, "")


def test_weekly_pnl_window_is_trailing_seven_days():
    from datetime import timedelta
    p = Portfolio()
    p.add_pnl(DAY, -100.0)
    p.add_pnl(DAY - timedelta(days=10), -900.0)        # outside the window
    assert p.weekly_pnl(DAY) == pytest.approx(-100.0)


def test_open_risk_is_released_when_a_trade_closes():
    p = Portfolio()
    p.record_entry(DAY, 300.0)
    assert p.open_risk == 300.0
    p.close_trade(DAY, 100.0, 300.0)
    assert p.open_risk == 0.0
    assert p.daily_pnl(DAY) == 100.0


def test_feature_drift_kill_fires_on_an_out_of_distribution_input():
    mean, std = np.zeros(5), np.ones(5)
    assert feature_drift_kill(mean, std, np.zeros(5))[0] is False
    drifted, why = feature_drift_kill(mean, std, np.array([0, 0, 12.0, 0, 0]))
    assert drifted and "feature_drift" in why


def test_feature_drift_ignores_nan_values():
    mean, std = np.zeros(3), np.ones(3)
    assert feature_drift_kill(mean, std, np.array([0.0, np.nan, 0.0]))[0] is False


def test_a_constant_training_feature_flags_any_different_live_value():
    """No variance means no scale, so any deviation is out-of-distribution."""
    mean, std = np.zeros(3), np.array([0.0, 1.0, 1.0])
    drifted, why = feature_drift_kill(mean, std, np.array([5.0, 0.0, 0.0]))
    assert drifted and "constant_in_training" in why
    assert feature_drift_kill(mean, std, np.array([0.0, 0.0, 0.0]))[0] is False
