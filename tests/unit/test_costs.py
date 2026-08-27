import pytest

from backtest.costs import (apply_costs, cost_per_spread_per_share,
                            max_risk_dollars, roundtrip_fees)


def test_roundtrip_fees_charge_both_legs_on_both_sides():
    """One spread traded once = 4 contract-legs: 2 legs x (open + close)."""
    assert roundtrip_fees(0.65, contracts=1) == pytest.approx(2.60)


def test_roundtrip_fees_scale_with_contracts():
    assert roundtrip_fees(0.65, contracts=3) == pytest.approx(7.80)


def test_roundtrip_fees_include_exchange_fees():
    assert roundtrip_fees(0.65, 1, exchange_fee_per_contract=0.05) == pytest.approx(2.80)


def test_apply_costs_net_pnl_is_in_dollars():
    """P&L uses the 100x contract multiplier, so the risk engine can size it."""
    fees, net = apply_costs(entry_credit=2.00, exit_debit=0.80,
                            commission_per_contract=0.65, contracts=1)
    assert fees == pytest.approx(2.60)
    assert net == pytest.approx((2.00 - 0.80) * 100 - 2.60)


def test_apply_costs_on_a_losing_trade():
    _, net = apply_costs(2.00, 4.00, 0.65)
    assert net == pytest.approx(-200.0 - 2.60)


def test_apply_costs_scales_with_contracts():
    _, one = apply_costs(2.00, 0.80, 0.0, contracts=1)
    _, five = apply_costs(2.00, 0.80, 0.0, contracts=5)
    assert five == pytest.approx(one * 5)


def test_cost_per_share_is_comparable_to_a_credit():
    assert cost_per_spread_per_share(0.65) == pytest.approx(0.026)


def test_max_risk_dollars_uses_width_minus_credit():
    assert max_risk_dollars(5.0, 2.0, contracts=1) == pytest.approx(300.0)
    assert max_risk_dollars(5.0, 2.0, contracts=4) == pytest.approx(1200.0)


def test_max_risk_never_negative():
    assert max_risk_dollars(5.0, 9.0) == 0.0
