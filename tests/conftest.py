"""Shared test factories."""
from datetime import date, datetime, timedelta

import pytest

from data.events import EventCalendar
from schemas import Candidate, OptionQuote, UnderlyingBar

TS = datetime(2024, 1, 2, 15, 45)
E1 = date(2024, 2, 16)          # 45 DTE from TS
E2 = date(2024, 2, 2)           # 31 DTE from TS


def quote(**kw) -> OptionQuote:
    base = dict(
        timestamp=TS, symbol="SPY", expiry=E1, dte=45, strike=400.0,
        option_type="P", bid=2.5, ask=2.7, mid=2.6, last=2.6, volume=100,
        open_interest=500, iv=0.18, delta=-0.20, gamma=0.01, theta=-0.02,
        vega=0.05, underlying_price=450.0,
    )
    base.update(kw)
    if "mid" not in kw:
        base["mid"] = (base["bid"] + base["ask"]) / 2.0
    return OptionQuote(**base)


def candidate(**kw) -> Candidate:
    base = dict(
        entry_time=TS, short_strike=400.0, long_strike=395.0, expiry=E1, dte=45,
        entry_credit=2.0, width=5.0, underlying_price=450.0, short_delta=-0.20,
        short_iv=0.20, short_bid=2.0, short_ask=2.4, short_open_interest=1500,
        short_volume=220, short_theta=-0.02, short_vega=0.05, short_gamma=0.01,
        long_bid=0.8, long_ask=1.2, long_delta=-0.08, long_iv=0.22,
        long_open_interest=900, long_volume=140, long_theta=-0.011,
        long_vega=0.03, long_gamma=0.006,
    )
    base.update(kw)
    return Candidate(**base)


def bars(n=80, start_close=450.0, step=-0.5, symbol="SPY"):
    return [
        UnderlyingBar(timestamp=TS - timedelta(days=i), symbol=symbol,
                      open=start_close, high=start_close, low=start_close,
                      close=start_close + step * i, adj_close=start_close,
                      volume=1_000_000)
        for i in range(n - 1, -1, -1)
    ]


def open_calendar(**events) -> EventCalendar:
    """Calendar with wide coverage and no events, so gating never interferes.

    Note that empty event sets make `days_to_*` features NaN, which is correct
    (an unknown calendar is not a clear one) but blocks the paper-trade pipeline.
    Use `dated_calendar()` when the test needs complete features.
    """
    d = {"FOMC": frozenset(), "CPI": frozenset(), "JOBS": frozenset()}
    d.update({k: frozenset(v) for k, v in events.items()})
    return EventCalendar(dates=d, coverage_start=date(2000, 1, 1),
                         coverage_end=date(2099, 12, 31), source="test")


def dated_calendar(**events) -> EventCalendar:
    """Calendar whose events sit far outside any test trade window.

    Event features resolve to real numbers, but nothing falls inside a trade, so
    the event gate stays out of the way.
    """
    far = frozenset([date(2023, 1, 15), date(2025, 6, 15)])
    d = {"FOMC": far, "CPI": far, "JOBS": far}
    d.update({k: frozenset(v) for k, v in events.items()})
    return EventCalendar(dates=d, coverage_start=date(2000, 1, 1),
                         coverage_end=date(2099, 12, 31), source="test")


@pytest.fixture
def cal():
    return open_calendar()


@pytest.fixture
def cand():
    return candidate()
