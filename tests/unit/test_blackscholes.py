"""Black-Scholes tests.

The Greeks are checked against finite differences of the pricer rather than
against hand-copied reference numbers. If the analytic derivative and the
numerical derivative of the same function agree to 1e-8, the calculus is right;
a table of constants only proves someone typed them correctly.
"""
import math

import pytest

from models.blackscholes import (Greeks, bs_greeks, bs_price,
                                 early_exercise_risk, greeks_from_price,
                                 implied_vol, implied_vol_detail, intrinsic,
                                 norm_cdf, price_bounds)

S, K, T, R, Q, SIG = 450.0, 430.0, 40 / 365, 0.045, 0.013, 0.18


def fd(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


# ----------------------------------------------------------------- pricing
def test_put_call_parity():
    c = bs_price(S, K, T, R, Q, SIG, "C")
    p = bs_price(S, K, T, R, Q, SIG, "P")
    assert c - p == pytest.approx(S * math.exp(-Q * T) - K * math.exp(-R * T), abs=1e-10)


def test_price_is_monotone_increasing_in_volatility():
    """This is what makes the implied-vol root unique."""
    prices = [bs_price(S, K, T, R, Q, v, "P") for v in (0.05, 0.1, 0.2, 0.4, 0.8)]
    assert prices == sorted(prices)


def test_expired_option_is_worth_intrinsic():
    assert bs_price(S, 500.0, 0.0, R, Q, SIG, "P") == pytest.approx(50.0)
    assert bs_price(S, 400.0, 0.0, R, Q, SIG, "P") == pytest.approx(0.0)


def test_zero_volatility_is_worth_intrinsic():
    assert bs_price(S, 500.0, T, R, Q, 0.0, "P") == pytest.approx(50.0)


def test_price_stays_inside_the_no_arbitrage_bounds():
    lo, hi = price_bounds(S, K, T, R, Q, "P")
    for v in (0.01, 0.2, 1.0, 4.9):
        assert lo <= bs_price(S, K, T, R, Q, v, "P") <= hi


def test_bad_option_type_rejected():
    with pytest.raises(ValueError, match="must be 'P' or 'C'"):
        bs_price(S, K, T, R, Q, SIG, "PUT")


# ------------------------------------------------------------------ greeks
@pytest.mark.parametrize("ot", ["P", "C"])
def test_delta_matches_numerical_derivative(ot):
    g = bs_greeks(S, K, T, R, Q, SIG, ot)
    num = fd(lambda s: bs_price(s, K, T, R, Q, SIG, ot), S, 1e-4)
    assert g.delta == pytest.approx(num, abs=1e-8)


@pytest.mark.parametrize("ot", ["P", "C"])
def test_gamma_matches_second_derivative(ot):
    g = bs_greeks(S, K, T, R, Q, SIG, ot)
    h = 1e-3
    num = (bs_price(S + h, K, T, R, Q, SIG, ot) - 2 * bs_price(S, K, T, R, Q, SIG, ot)
           + bs_price(S - h, K, T, R, Q, SIG, ot)) / (h * h)
    assert g.gamma == pytest.approx(num, abs=1e-6)


@pytest.mark.parametrize("ot", ["P", "C"])
def test_vega_matches_derivative_and_is_per_vol_point(ot):
    g = bs_greeks(S, K, T, R, Q, SIG, ot)
    num = fd(lambda v: bs_price(S, K, T, R, Q, v, ot), SIG, 1e-6) / 100.0
    assert g.vega == pytest.approx(num, abs=1e-8)


@pytest.mark.parametrize("ot", ["P", "C"])
def test_theta_matches_derivative_and_is_per_day(ot):
    g = bs_greeks(S, K, T, R, Q, SIG, ot)
    num = -fd(lambda t: bs_price(S, K, t, R, Q, SIG, ot), T, 1e-6) / 365.0
    assert g.theta == pytest.approx(num, abs=1e-8)


@pytest.mark.parametrize("ot", ["P", "C"])
def test_rho_matches_derivative(ot):
    g = bs_greeks(S, K, T, R, Q, SIG, ot)
    num = fd(lambda r: bs_price(S, K, T, r, Q, SIG, ot), R, 1e-7) / 100.0
    assert g.rho == pytest.approx(num, abs=1e-7)


def test_put_delta_is_negative_and_bounded():
    for k in (350.0, 430.0, 450.0, 550.0):
        d = bs_greeks(S, k, T, R, Q, SIG, "P").delta
        assert -1.0 <= d <= 0.0


def test_atm_delta_is_near_half():
    assert bs_greeks(S, S, T, R, Q, SIG, "P").delta == pytest.approx(-0.5, abs=0.05)


def test_delta_is_monotone_in_strike():
    deltas = [bs_greeks(S, float(k), T, R, Q, SIG, "P").delta
              for k in range(350, 560, 10)]
    assert deltas == sorted(deltas, reverse=True)


def test_dividend_yield_moves_delta_materially():
    """q=0 is not a harmless simplification."""
    d0 = bs_greeks(S, K, T, R, 0.0, SIG, "P").delta
    d2 = bs_greeks(S, K, T, R, 0.02, SIG, "P").delta
    assert abs(d2 - d0) > 0.005


def test_rate_regime_moves_delta_materially():
    """0% vs 5% short rates shift the strike the strategy would pick."""
    d_low = bs_greeks(S, K, T, 0.001, Q, SIG, "P").delta
    d_high = bs_greeks(S, K, T, 0.05, Q, SIG, "P").delta
    assert abs(d_high - d_low) > 0.02


def test_expired_greeks_are_zero_with_terminal_delta():
    g = bs_greeks(S, 500.0, 0.0, R, Q, SIG, "P")
    assert g.delta == -1.0 and g.gamma == 0.0 and g.vega == 0.0
    assert bs_greeks(S, 400.0, 0.0, R, Q, SIG, "P").delta == 0.0


# -------------------------------------------------------------- implied vol
def test_iv_round_trip_on_realistic_quotes():
    """Anything a real chain would quote (>= $0.01) must invert precisely."""
    worst, n = 0.0, 0
    for k in range(400, 505, 5):
        for t in (30 / 365, 45 / 365):
            for v in (0.12, 0.18, 0.28, 0.45):
                px = bs_price(S, float(k), t, R, Q, v, "P")
                if px < 0.01:
                    continue
                res = implied_vol_detail(px, S, float(k), t, R, Q, "P")
                assert res.ok, (k, t, v, px, res.reason)
                worst = max(worst, abs(res.iv - v))
                n += 1
    assert n > 50
    assert worst < 1e-6, f"worst IV error {worst:.2e}"


def test_iv_refuses_a_worthless_option_instead_of_guessing():
    """With no time value every sigma fits, so the answer would be the seed."""
    px = bs_price(S, 250.0, 7 / 365, R, Q, 0.30, "P")
    res = implied_vol_detail(px, S, 250.0, 7 / 365, R, Q, "P")
    assert not res.ok
    assert res.reason in ("non_positive_price", "no_time_value", "price_at_lower_bound")


def test_iv_refuses_a_price_below_intrinsic():
    lo, _ = price_bounds(S, 520.0, T, R, Q, "P")
    res = implied_vol_detail(lo - 0.5, S, 520.0, T, R, Q, "P")
    assert res.reason == "price_below_intrinsic"


def test_iv_refuses_a_price_above_the_upper_bound():
    _, hi = price_bounds(S, K, T, R, Q, "P")
    assert implied_vol_detail(hi + 1.0, S, K, T, R, Q, "P").reason == "price_above_upper_bound"


def test_iv_refuses_a_zero_price():
    """The Yahoo case: 501 of 509 contracts quoted bid=ask=0."""
    assert implied_vol_detail(0.0, S, K, T, R, Q, "P").reason == "non_positive_price"


def test_iv_refuses_an_expired_option():
    assert implied_vol_detail(1.0, S, K, 0.0, R, Q, "P").reason == "expired"


def test_iv_refuses_non_finite_input():
    assert implied_vol_detail(float("nan"), S, K, T, R, Q, "P").reason == "non_finite_input"


def test_iv_converges_for_deep_itm_puts_where_newton_alone_diverges():
    for k in (600.0, 700.0):
        px = bs_price(S, k, T, R, Q, 0.30, "P")
        res = implied_vol_detail(px, S, k, T, R, Q, "P")
        assert res.ok, res.reason
        assert res.iv == pytest.approx(0.30, abs=1e-4)


def test_iv_helper_returns_nan_on_failure():
    assert math.isnan(implied_vol(0.0, S, K, T, R, Q, "P"))


def test_greeks_from_price_round_trips():
    px = bs_price(S, K, T, R, Q, SIG, "P")
    g, reason = greeks_from_price(px, S, K, T, R, Q, "P")
    assert reason == ""
    assert g.delta == pytest.approx(bs_greeks(S, K, T, R, Q, SIG, "P").delta, abs=1e-6)


def test_greeks_from_price_reports_failure_rather_than_returning_junk():
    g, reason = greeks_from_price(0.0, S, K, T, R, Q, "P")
    assert g is None and reason == "non_positive_price"


# ------------------------------------------------------------ american flag
def test_early_exercise_flagged_for_deep_itm_puts_near_expiry():
    assert early_exercise_risk(400.0, 500.0, 20 / 365, 0.05, 0.013, "P")


def test_early_exercise_not_flagged_for_the_otm_puts_this_strategy_sells():
    assert not early_exercise_risk(450.0, 430.0, 40 / 365, 0.045, 0.013, "P")


def test_early_exercise_never_flagged_for_calls_here():
    assert not early_exercise_risk(400.0, 500.0, 20 / 365, 0.05, 0.013, "C")


def test_norm_cdf_endpoints():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(-8.0) < 1e-14
    assert norm_cdf(8.0) > 1 - 1e-14


def test_intrinsic():
    assert intrinsic(450.0, 500.0, "P") == 50.0
    assert intrinsic(450.0, 400.0, "P") == 0.0
    assert intrinsic(450.0, 400.0, "C") == 50.0
