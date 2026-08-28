"""QuoteIndex tests.

The index replaces a full scan of the quote stream, so the property that matters
most is EQUIVALENCE: it must return exactly what `snapshots_from_quotes` returns,
or the speedup has quietly changed the labels.
"""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from labels.quote_index import QuoteIndex, _from_ns, _to_ns
from labels.simulator import snapshots_from_quotes
from tests.conftest import TS, candidate, quote

LATER = [TS + timedelta(days=d) for d in (1, 2, 5, 9)]


def stream(strikes=(400.0, 395.0), times=None, **kw):
    times = times or LATER
    return [quote(timestamp=t, strike=k, bid=k / 1000, ask=k / 1000 + 0.2, **kw)
            for t in times for k in strikes]


# --------------------------------------------------------------- equivalence
def test_matches_snapshots_from_quotes_exactly():
    quotes = stream() + [quote(timestamp=LATER[0], strike=390.0)]
    c = candidate()
    assert QuoteIndex.from_quotes(quotes).snapshots_for(c) == \
        snapshots_from_quotes(c, quotes)


def test_matches_when_a_leg_is_intermittently_missing():
    quotes = stream(strikes=(400.0,)) + [quote(timestamp=LATER[1], strike=395.0)]
    c = candidate()
    assert QuoteIndex.from_quotes(quotes).snapshots_for(c) == \
        snapshots_from_quotes(c, quotes)


def test_matches_when_nothing_is_available():
    c = candidate()
    quotes = [quote(timestamp=LATER[0], strike=123.0)]
    assert QuoteIndex.from_quotes(quotes).snapshots_for(c) == \
        snapshots_from_quotes(c, quotes) == []


# ------------------------------------------------------------------ behaviour
def test_marks_at_or_before_entry_are_excluded():
    quotes = stream(times=[TS - timedelta(days=1), TS, LATER[0]])
    snaps = QuoteIndex.from_quotes(quotes).snapshots_for(candidate())
    assert [s.time for s in snaps] == [LATER[0]]


def test_marks_come_back_in_chronological_order():
    quotes = stream(times=list(reversed(LATER)))
    snaps = QuoteIndex.from_quotes(quotes).snapshots_for(candidate())
    assert [s.time for s in snaps] == sorted(s.time for s in snaps)


def test_uses_short_ask_and_long_bid():
    """Buy back the short at the ask, sell the long at the bid."""
    quotes = [quote(timestamp=LATER[0], strike=400.0, bid=1.5, ask=1.7),
              quote(timestamp=LATER[0], strike=395.0, bid=0.4, ask=0.6)]
    s = QuoteIndex.from_quotes(quotes).snapshots_for(candidate())[0]
    assert s.short_ask == 1.7 and s.long_bid == 0.4


def test_a_timestamp_missing_either_leg_is_skipped():
    """Half a spread is not a price."""
    quotes = [quote(timestamp=LATER[0], strike=400.0),
              quote(timestamp=LATER[1], strike=395.0)]
    assert QuoteIndex.from_quotes(quotes).snapshots_for(candidate()) == []


def test_a_different_expiry_is_never_mixed_in():
    quotes = stream() + [quote(timestamp=LATER[0], strike=400.0,
                               expiry=date(2024, 3, 15)),
                         quote(timestamp=LATER[0], strike=395.0,
                               expiry=date(2024, 3, 15))]
    idx = QuoteIndex.from_quotes(quotes)
    assert len(idx.snapshots_for(candidate())) == len(LATER)


def test_calls_are_excluded_by_default():
    calls = [quote(timestamp=LATER[0], strike=k, option_type="C")
             for k in (400.0, 395.0)]
    assert QuoteIndex.from_quotes(calls).snapshots_for(candidate()) == []


def test_empty_input_builds_an_empty_index():
    idx = QuoteIndex.from_quotes([])
    assert len(idx) == 0
    assert idx.snapshots_for(candidate()) == []


def test_fractional_strikes_do_not_collide():
    """Strikes are keyed as integer cents; float keys compare unreliably."""
    quotes = [quote(timestamp=LATER[0], strike=400.5, bid=1.0, ask=1.2),
              quote(timestamp=LATER[0], strike=395.5, bid=0.3, ask=0.5)]
    c = candidate(short_strike=400.5, long_strike=395.5)
    s = QuoteIndex.from_quotes(quotes).snapshots_for(c)
    assert len(s) == 1 and s[0].short_ask == 1.2


# ---------------------------------------------------------------- from_frame
def frame():
    rows = []
    for t in LATER:
        for k, bid, ask in ((400.0, 1.5, 1.7), (395.0, 0.4, 0.6)):
            rows.append({"timestamp": t, "expiry": pd.Timestamp(2024, 2, 16),
                         "strike": k, "option_type": "P", "bid": bid, "ask": ask,
                         "underlying_price": 450.0})
    return pd.DataFrame(rows)


def test_from_frame_matches_from_quotes():
    a = QuoteIndex.from_frame(frame()).snapshots_for(candidate())
    b = QuoteIndex.from_quotes(stream()).snapshots_for(candidate())
    assert [s.time for s in a] == [s.time for s in b]


def test_from_frame_has_no_timezone_shift():
    """`datetime.timestamp()` applies a local offset; numpy's conversion does not."""
    snaps = QuoteIndex.from_frame(frame()).snapshots_for(candidate())
    assert [s.time for s in snaps] == LATER


def test_ns_round_trip_is_exact():
    for dt in (datetime(2024, 1, 2, 15, 45), datetime(2015, 6, 30, 16, 0)):
        assert _from_ns(_to_ns(dt)) == dt


def test_from_frame_filters_by_option_type():
    df = frame()
    df["option_type"] = "C"
    assert QuoteIndex.from_frame(df, option_type="P").snapshots_for(candidate()) == []


def test_index_reports_its_size():
    idx = QuoteIndex.from_frame(frame())
    assert len(idx) == 8
    assert idx.n_contracts == 2
    assert "8" in repr(idx)


def test_packed_keys_do_not_collide_across_realistic_strikes():
    """One int64 packs expiry and strike; verify that mapping stays 1:1."""
    rows = []
    for k in np.arange(50.0, 1200.0, 0.5):
        rows.append({"timestamp": LATER[0], "expiry": pd.Timestamp(2024, 2, 16),
                     "strike": float(k), "option_type": "P", "bid": 1.0,
                     "ask": 1.1, "underlying_price": 450.0})
    df = pd.DataFrame(rows)
    assert QuoteIndex.from_frame(df).n_contracts == len(df)
