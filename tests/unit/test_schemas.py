from datetime import date, datetime

import pytest

from schemas import (EMBARGO_DAYS, MAX_DTE, MAX_HOLD_DAYS, TIME_EXIT_DTE,
                     Candidate, OptionQuote, with_flags)
from tests.conftest import candidate, quote


def test_option_quote_constructs_with_required_fields():
    q = quote()
    assert q.option_type == "P"
    assert q.quality_flags == ()
    assert q.is_clean


def test_with_flags_returns_a_new_quote_and_dedupes():
    q = quote()
    flagged = with_flags(with_flags(q, ["wide_spread"]), ["wide_spread", "zero_bid"])
    assert flagged.quality_flags == ("wide_spread", "zero_bid")
    assert q.quality_flags == ()          # original untouched: raw data is immutable
    assert not flagged.is_clean


def test_quotes_are_frozen():
    with pytest.raises(Exception):
        quote().bid = 99.0


def test_max_risk_is_width_minus_credit_not_width():
    """A $5 spread sold for $2 risks $3. Dividing by width overstates return ~40%."""
    c = candidate(entry_credit=2.0, width=5.0)
    assert c.max_risk == 3.0


def test_max_risk_never_zero_or_negative():
    assert candidate(entry_credit=5.0, width=5.0).max_risk > 0
    assert candidate(entry_credit=9.0, width=5.0).max_risk > 0


def test_break_even_is_short_strike_less_credit():
    assert candidate(short_strike=400.0, entry_credit=2.0).break_even == 398.0


def test_accepted_flag_tracks_rejection_reason():
    assert candidate().accepted
    assert not candidate(rejection_reason="oi_too_low").accepted


def test_embargo_covers_the_longest_possible_hold():
    """Entered at max DTE and held to the forced exit is the worst case."""
    assert MAX_HOLD_DAYS == MAX_DTE - TIME_EXIT_DTE
    assert EMBARGO_DAYS >= MAX_HOLD_DAYS
