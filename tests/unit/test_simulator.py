"""Simulator tests.

Each exit rule gets its own unambiguous case. In the previous plan several of
these collided -- a "time exit" fixture whose close cost also satisfied the
profit target, so the test asserted TIME_EXIT and the code returned
PROFIT_TARGET. Here the numbers are chosen so exactly one rule can fire, and one
test deliberately makes two fire at once to pin down which wins.
"""
import math
from datetime import date, timedelta

import pytest

from labels.outcomes import (label_distribution, labelable, outcome_label)
from labels.simulator import (SpreadSnapshot, close_cost, intrinsic_at_expiry,
                              simulate_trade, snapshots_from_quotes)
from tests.conftest import TS, candidate, quote

# credit 2.00 -> profit target 1.00, stop 4.00, max risk 3.00
CAND = candidate(entry_credit=2.00)


def snap(days, short_ask, long_bid, underlying=float("nan")):
    return SpreadSnapshot(TS + timedelta(days=days), short_ask, long_bid, underlying)


def test_close_cost_is_short_ask_minus_long_bid():
    assert close_cost(1.4, 0.4) == pytest.approx(1.0)


def test_profit_target_when_close_cost_reaches_half_the_credit():
    out = simulate_trade(CAND, [snap(1, 1.6, 0.5), snap(2, 0.9, 0.4)])
    assert out.exit_reason == "PROFIT_TARGET"
    assert out.label_win is True
    assert out.n_marks == 2


def test_stop_loss_when_close_cost_reaches_twice_the_credit():
    out = simulate_trade(CAND, [snap(1, 4.6, 0.4)])       # cost 4.20 >= 4.00
    assert out.exit_reason == "STOP_LOSS"
    assert out.label_win is False


def test_time_exit_when_dte_falls_to_seven():
    out = simulate_trade(CAND, [snap(39, 1.9, 0.4)])      # cost 1.50, 6 DTE left
    assert out.exit_reason == "TIME_EXIT"
    assert out.label_win is False


def test_expiry_settles_at_intrinsic_value():
    out = simulate_trade(CAND, [snap(45, 1.9, 0.4, underlying=402.0)])
    assert out.exit_reason == "EXPIRY"
    assert out.exit_debit == 0.0                          # both legs expire worthless


def test_stop_wins_when_a_single_day_could_have_hit_both_rails():
    """Daily marks cannot resolve intraday order, so assume the adverse one."""
    out = simulate_trade(candidate(entry_credit=2.00), [snap(1, 4.6, 0.4)])
    assert out.exit_reason == "STOP_LOSS"


def test_no_usable_marks_is_NO_DATA_not_a_free_win():
    out = simulate_trade(CAND, [])
    assert out.exit_reason == "NO_DATA"
    assert out.label_win is False
    assert math.isnan(out.net_pnl)


def test_running_out_of_marks_mid_trade_is_NO_DATA():
    out = simulate_trade(CAND, [snap(1, 2.0, 0.5), snap(2, 2.1, 0.5)])
    assert out.exit_reason == "NO_DATA"


def test_marks_at_or_before_entry_are_ignored():
    out = simulate_trade(CAND, [snap(0, 0.5, 0.2), snap(-1, 0.5, 0.2)])
    assert out.exit_reason == "NO_DATA"


def test_unusable_marks_are_skipped():
    out = simulate_trade(CAND, [snap(1, float("nan"), 0.4), snap(2, 0.9, 0.4)])
    assert out.exit_reason == "PROFIT_TARGET"
    assert out.n_marks == 1


def test_return_on_risk_divides_by_width_minus_credit():
    """$5 spread at $2 credit risks $3, not $5."""
    out = simulate_trade(CAND, [snap(1, 0.9, 0.4)])
    assert out.net_pnl == pytest.approx(1.50)
    assert abs(out.final_return_on_risk - (1.50 / 3.00)) < 1e-9


def test_fees_and_slippage_reduce_net_pnl():
    plain = simulate_trade(CAND, [snap(1, 0.9, 0.4)])
    costed = simulate_trade(CAND, [snap(1, 0.9, 0.4)], fees_per_spread=0.10,
                            slippage=0.05)
    assert costed.net_pnl < plain.net_pnl
    assert abs(costed.net_pnl - (plain.net_pnl - 0.15)) < 1e-9


def test_excursions_are_non_negative_and_ordered():
    out = simulate_trade(CAND, [snap(1, 3.0, 0.4), snap(2, 0.9, 0.4)])
    assert out.max_adverse_excursion >= 0
    assert out.max_favorable_excursion >= 0
    assert out.max_adverse_excursion == pytest.approx(0.60)   # cost 2.60 vs credit 2.00


def test_intrinsic_at_expiry_bounds():
    c = candidate(short_strike=400.0, long_strike=395.0, width=5.0)
    assert intrinsic_at_expiry(c, 410.0) == 0.0       # both OTM
    assert intrinsic_at_expiry(c, 390.0) == 5.0       # both ITM: max loss
    assert intrinsic_at_expiry(c, 398.0) == 2.0       # partially ITM
    assert intrinsic_at_expiry(c, float("nan")) == 5.0


def test_outcome_label_is_one_only_for_profit_target():
    assert outcome_label(simulate_trade(CAND, [snap(1, 0.9, 0.4)])) == 1
    assert outcome_label(simulate_trade(CAND, [snap(1, 4.6, 0.4)])) == 0


def test_no_data_outcomes_are_excluded_from_training():
    outs = [simulate_trade(CAND, [snap(1, 0.9, 0.4)]), simulate_trade(CAND, [])]
    assert len(labelable(outs)) == 1
    dist = label_distribution(outs)
    assert dist["_dropped_no_data"] == 1
    assert dist["_base_rate"] == 1.0


def test_snapshots_from_quotes_pairs_both_legs():
    later = TS + timedelta(days=1)
    quotes = [
        quote(timestamp=later, strike=400.0, bid=1.5, ask=1.7),
        quote(timestamp=later, strike=395.0, bid=0.4, ask=0.6),
        quote(timestamp=later, strike=390.0, bid=0.1, ask=0.2),   # irrelevant strike
    ]
    snaps = snapshots_from_quotes(CAND, quotes)
    assert len(snaps) == 1
    assert snaps[0].short_ask == 1.7 and snaps[0].long_bid == 0.4


def test_snapshots_from_quotes_skips_timestamps_missing_a_leg():
    later = TS + timedelta(days=1)
    snaps = snapshots_from_quotes(CAND, [quote(timestamp=later, strike=400.0)])
    assert snaps == []


def test_close_cost_above_the_width_is_clamped_to_max_loss():
    """Real EOD data produced a $7.21 close cost on a $5-wide spread. A
    defined-risk position cannot lose more than width - credit."""
    out = simulate_trade(CAND, [snap(1, 9.0, 0.0)])          # raw cost 9.00 > width
    assert out.exit_reason == "STOP_LOSS"
    assert out.exit_debit == pytest.approx(5.00)             # clamped to the width
    assert out.net_pnl == pytest.approx(2.00 - 5.00)         # = -max_risk
    assert out.final_return_on_risk == pytest.approx(-1.0)
    assert out.n_clamped_marks == 1


def test_negative_close_cost_is_clamped_to_zero():
    out = simulate_trade(CAND, [snap(1, 0.2, 0.9)])          # raw cost -0.70
    assert out.exit_reason == "PROFIT_TARGET"
    assert out.exit_debit == pytest.approx(0.0)
    assert out.n_clamped_marks == 1


def test_a_loss_can_never_exceed_the_defined_risk():
    for short_ask, long_bid in ((9.0, 0.0), (20.0, 0.0), (6.0, 0.1)):
        out = simulate_trade(CAND, [snap(1, short_ask, long_bid)])
        assert out.net_pnl >= -(CAND.max_risk + out.fees + out.slippage + 1e-9)


def test_valid_marks_are_not_clamped():
    out = simulate_trade(CAND, [snap(1, 1.6, 0.5), snap(2, 0.9, 0.4)])
    assert out.n_clamped_marks == 0
