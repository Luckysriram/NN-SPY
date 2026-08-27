from data.validate import (flag_summary, validate_quote, validate_quotes,
                           validate_strike_order)
from tests.conftest import quote


def flags(**kw):
    return validate_quote(quote(**kw)).quality_flags


def test_clean_quote_has_no_flags():
    assert flags() == ()


def test_flags_bid_above_ask():
    assert "bid_above_ask" in flags(bid=3.0, ask=2.0)


def test_flags_negative_iv():
    assert "negative_iv" in flags(iv=-0.1)


def test_zero_iv_is_distinct_from_negative_iv():
    """A missing field and a bad field must not look identical downstream."""
    assert "zero_iv" in flags(iv=0.0)
    assert "negative_iv" not in flags(iv=0.0)


def test_flags_implausible_iv():
    assert "implausible_iv" in flags(iv=9.0)


def test_flags_zero_bid_on_leg():
    assert "zero_bid" in flags(bid=0.0)


def test_flags_wide_spread():
    assert "wide_spread" in flags(bid=2.0, ask=5.0)


def test_flags_missing_underlying():
    assert "missing_underlying" in flags(underlying_price=0.0)


def test_flags_expiry_before_entry():
    from datetime import date
    assert "expiry_before_entry" in flags(expiry=date(2023, 1, 1))


def test_flags_put_with_positive_delta():
    assert "put_delta_positive" in flags(delta=0.25)


def test_flags_implausible_delta():
    assert "implausible_delta" in flags(delta=-4.0)


def test_validate_never_mutates_the_input():
    q = quote(bid=3.0, ask=2.0)
    validate_quote(q)
    assert q.quality_flags == ()


def test_duplicate_contracts_are_flagged_across_the_batch():
    out = validate_quotes([quote(), quote()])
    assert all("duplicate_contract" in q.quality_flags for q in out)


def test_stale_quotes_are_flagged():
    from datetime import timedelta
    from tests.conftest import TS
    same = [quote(timestamp=TS + timedelta(days=i)) for i in range(5)]
    out = validate_quotes(same)
    assert any("stale_quote" in q.quality_flags for q in out)


def test_flag_summary_counts_clean_and_total():
    s = flag_summary(validate_quotes([quote(), quote(bid=3.0, ask=2.0)]))
    assert s["_total_quotes"] == 2


def test_strike_order_rejects_a_long_leg_at_or_above_the_short():
    assert validate_strike_order(400.0, 395.0) == ""
    assert validate_strike_order(400.0, 405.0) == "bad_strike_order"
    assert validate_strike_order(400.0, 400.0) == "bad_strike_order"
