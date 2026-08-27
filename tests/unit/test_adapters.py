import math
from datetime import date

from data.adapters.sample import normalize_option_quotes, normalize_rows

ROW = {
    "timestamp": "2024-01-02 15:45:00", "expiry": "2024-02-16", "strike": "400",
    "type": "P", "bid": "2.50", "ask": "2.70", "last": "2.60", "volume": "100",
    "oi": "500", "iv": "0.18", "delta": "-0.20", "gamma": "0.01",
    "theta": "-0.02", "vega": "0.05", "underlying": "450.0",
}


def test_normalize_maps_a_source_row_to_canonical_quote():
    q = normalize_option_quotes([ROW])[0]
    assert q.symbol == "SPY"
    assert q.strike == 400.0
    assert q.option_type == "P"
    assert q.bid == 2.5
    assert q.expiry == date(2024, 2, 16)
    assert q.dte == (date(2024, 2, 16) - date(2024, 1, 2)).days
    assert q.mid == 2.6


def test_normalize_handles_a_different_vendors_column_names():
    """The adapter is a column map, so a new vendor needs no new module."""
    other = {"quote_date": "2024-01-02", "expiration": "2024-02-16",
             "strike_price": "395", "right": "PUT", "best_bid": "0.80",
             "best_ask": "1.20", "underlying_price": "450.0"}
    q = normalize_option_quotes([other])[0]
    assert q.strike == 395.0 and q.option_type == "P" and q.bid == 0.8


def test_missing_greeks_become_nan_and_a_flag_never_zero():
    row = {k: v for k, v in ROW.items() if k not in ("delta", "iv")}
    q = normalize_option_quotes([row])[0]
    assert math.isnan(q.delta) and math.isnan(q.iv)
    assert "missing_delta" in q.quality_flags and "missing_iv" in q.quality_flags


def test_unparseable_rows_are_returned_as_rejects_not_printed_and_dropped():
    res = normalize_rows([ROW, {"timestamp": "nonsense"}, {"bogus": 1}])
    assert len(res.quotes) == 1
    assert len(res.rejects) == 2
    assert sum(res.reject_reasons.values()) == 2


def test_a_bad_row_does_not_abort_the_whole_load():
    res = normalize_rows([{"bogus": 1}, ROW])
    assert len(res.quotes) == 1


def test_nan_volume_is_flagged_not_dropped():
    """Yahoo returns a blank volume on real contracts. `nan or 0` is nan, and
    int(nan) raises -- which dropped those rows entirely instead of keeping them."""
    row = dict(ROW, volume=float("nan"), oi=float("nan"))
    res = normalize_rows([row])
    assert len(res.quotes) == 1, res.rejects
    q = res.quotes[0]
    assert q.volume == 0 and q.open_interest == 0
    assert "missing_volume" in q.quality_flags
    assert "missing_open_interest" in q.quality_flags
