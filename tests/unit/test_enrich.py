import math
from datetime import date

import pytest

from data.enrich import (enrich_quote, enrich_quotes, enrichment_summary,
                         needs_greeks)
from data.rates import MarketParams
from models.blackscholes import bs_greeks, bs_price
from tests.conftest import TS, quote

R, Q = 0.045, 0.013
NAN = float("nan")


def greekless(**kw):
    """A quote as a Greek-less vendor would deliver it: prices only."""
    base = dict(delta=NAN, gamma=NAN, theta=NAN, vega=NAN, iv=NAN,
                strike=430.0, underlying_price=450.0, dte=40,
                expiry=date(2024, 2, 11), bid=2.9, ask=3.1)
    base.update(kw)
    return quote(**base)


def priced(strike, sigma, dte=40, spot=450.0):
    """A quote whose mid is a genuine Black-Scholes price, so IV is recoverable."""
    px = bs_price(spot, strike, dte / 365.0, R, Q, sigma, "P")
    return greekless(strike=strike, dte=dte, underlying_price=spot,
                     bid=px - 0.02, ask=px + 0.02)


def test_needs_greeks_detects_missing_fields():
    assert needs_greeks(greekless())
    assert not needs_greeks(quote())


def test_enrich_fills_missing_greeks():
    q = enrich_quote(priced(430.0, 0.18), R, Q)
    assert not math.isnan(q.delta)
    assert -1.0 < q.delta < 0.0
    assert not math.isnan(q.iv)


def test_recovered_delta_matches_the_true_delta():
    """Round trip: price at a known sigma, recover it, and check the delta."""
    sigma = 0.18
    q = enrich_quote(priced(430.0, sigma), R, Q)
    expected = bs_greeks(450.0, 430.0, 40 / 365, R, Q, sigma, "P").delta
    assert q.delta == pytest.approx(expected, abs=1e-3)
    assert q.iv == pytest.approx(sigma, abs=1e-3)


def test_enriched_quotes_are_labelled_as_computed():
    """A modelled delta must never be mistaken for an observed one."""
    assert "greeks_computed" in enrich_quote(priced(430.0, 0.18), R, Q).quality_flags


def test_vendor_greeks_are_never_overwritten():
    q = greekless(delta=-0.33)          # vendor gave delta but no vega
    out = enrich_quote(q, R, Q)
    assert out.delta == -0.33
    assert not math.isnan(out.vega)     # the missing one still got filled


def test_a_complete_quote_passes_through_untouched():
    q = quote()
    assert enrich_quote(q, R, Q) is q


def test_zero_price_is_flagged_not_solved():
    """The Yahoo case: bid=ask=0 on a contract with real volume."""
    out = enrich_quote(greekless(bid=0.0, ask=0.0), R, Q)
    assert math.isnan(out.delta)
    assert any(f.startswith("greeks_failed") for f in out.quality_flags)


def test_missing_underlying_is_flagged():
    out = enrich_quote(greekless(underlying_price=NAN), R, Q)
    assert "greeks_failed_missing_underlying" in out.quality_flags


def test_price_below_intrinsic_is_flagged_not_solved():
    q = greekless(strike=520.0, underlying_price=450.0, bid=50.0, ask=50.2)
    out = enrich_quote(q, R, Q)
    assert math.isnan(out.delta)
    assert "greeks_failed_price_below_intrinsic" in out.quality_flags


def test_deep_itm_put_is_flagged_for_early_exercise_risk():
    """European model, American contract -- flagged rather than silently wrong."""
    q = priced(520.0, 0.20, dte=20)
    out = enrich_quote(q, 0.05, 0.013)
    assert "early_exercise_risk" in out.quality_flags


def test_the_otm_puts_the_strategy_sells_are_not_flagged():
    out = enrich_quote(priced(430.0, 0.18), R, Q)
    assert "early_exercise_risk" not in out.quality_flags


def test_batch_enrichment_uses_date_varying_rates():
    mp = MarketParams.constant(R, Q, date(2024, 1, 1), date(2024, 12, 31))
    out = enrich_quotes([priced(430.0, 0.18), priced(420.0, 0.20)], mp)
    assert all(not math.isnan(q.delta) for q in out)


def test_batch_requires_explicit_rates():
    """There is no safe default for r or q."""
    with pytest.raises(ValueError, match="no safe default"):
        enrich_quotes([priced(430.0, 0.18)])


def test_batch_accepts_flat_rates_when_stated():
    out = enrich_quotes([priced(430.0, 0.18)], r=R, dividend_yield=Q)
    assert not math.isnan(out[0].delta)


def test_summary_reports_what_succeeded_and_what_did_not():
    out = enrich_quotes([priced(430.0, 0.18), greekless(bid=0.0, ask=0.0)],
                        r=R, dividend_yield=Q)
    s = enrichment_summary(out)
    assert s["_total"] == 2
    assert s["_with_delta"] == 1
    assert s["greeks_computed"] == 1


def test_delta_ordering_survives_enrichment():
    """Recovered deltas must stay monotone in strike, or strike selection breaks.

    Put delta grows more negative as the strike rises, so ascending strikes give
    descending deltas.
    """
    out = [enrich_quote(priced(float(k), 0.18), R, Q) for k in range(400, 461, 10)]
    deltas = [q.delta for q in out]
    assert deltas == sorted(deltas, reverse=True)
    assert all(-1.0 < d < 0.0 for d in deltas)


def test_vendor_iv_that_does_not_reprice_the_quote_is_flagged():
    """Yahoo ships an IV column with ~20 distinct values across 500 contracts.

    Greeks are differentiated at the SOLVED iv, so silently keeping a vendor iv
    that disagrees leaves iv and delta describing different volatilities.
    """
    q = priced(430.0, 0.18)
    q = q.__class__(**{**q.__dict__, "iv": 0.95})     # vendor says 95%
    out = enrich_quote(q, R, Q)
    assert "iv_inconsistent_with_price" in out.quality_flags
    assert out.iv == 0.95                              # preserved by default


def test_prefer_solved_iv_keeps_the_record_self_consistent():
    q = priced(430.0, 0.18)
    q = q.__class__(**{**q.__dict__, "iv": 0.95})
    out = enrich_quote(q, R, Q, prefer_solved_iv=True)
    assert out.iv == pytest.approx(0.18, abs=1e-3)
    assert "iv_inconsistent_with_price" in out.quality_flags


def test_agreeing_vendor_iv_is_not_flagged():
    q = priced(430.0, 0.18)
    q = q.__class__(**{**q.__dict__, "iv": 0.181})
    assert "iv_inconsistent_with_price" not in enrich_quote(q, R, Q).quality_flags
