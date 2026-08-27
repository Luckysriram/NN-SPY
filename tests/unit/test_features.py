import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from data.events import EventCalendar
from features.features import (CALENDAR_COLS, FEATURE_COLS, FEATURE_INDEX,
                               N_FEATURES, assert_no_lookahead, assert_no_nan,
                               build_feature_vector, calendar_features,
                               drop_nan_rows, nan_report, spread_features,
                               to_matrix, underlying_features)
from schemas import UnderlyingBar
from tests.conftest import TS, bars, candidate, open_calendar

AS_OF = datetime(2024, 1, 2, 15, 45)


def bar(days_back, close):
    return UnderlyingBar(timestamp=datetime(2024, 1, 2) - timedelta(days=days_back),
                         symbol="SPY", open=close, high=close, low=close,
                         close=close, adj_close=close, volume=1000)


# ---------------------------------------------------------------- no leakage
def test_underlying_features_no_leakage():
    feat = underlying_features([bar(2, 100.0), bar(1, 102.0), bar(0, 101.0)], AS_OF)
    assert feat["spy_return_1d"] == pytest.approx((101.0 - 102.0) / 102.0)


def test_underlying_features_ignores_future_bars():
    feat = underlying_features([bar(1, 100.0), bar(-1, 200.0)], AS_OF)
    assert feat["spy_close"] == 100.0


def test_assert_no_lookahead_raises_on_a_future_bar():
    with pytest.raises(ValueError, match="lookahead"):
        assert_no_lookahead([bar(1, 100.0), bar(-1, 200.0)], AS_OF)


def test_assert_no_lookahead_passes_on_clean_history():
    assert_no_lookahead([bar(2, 100.0), bar(1, 101.0)], AS_OF)


# --------------------------------------------------- missing is NaN not zero
def test_short_history_leaves_long_window_features_nan():
    feat = underlying_features([bar(i, 100.0 + i) for i in range(10)], AS_OF)
    assert math.isnan(feat["spy_distance_from_ma50"])
    assert math.isnan(feat["spy_drawdown_63d"])


def test_long_history_fills_long_window_features():
    feat = underlying_features(bars(80), TS)
    assert not math.isnan(feat["spy_distance_from_ma50"])
    assert not math.isnan(feat["spy_drawdown_63d"])


def test_no_bars_gives_all_nan_market_features():
    feat = underlying_features([], AS_OF)
    assert all(math.isnan(v) for v in feat.values())


# ------------------------------------------------------------------- spread
def test_spread_features_include_credit_to_width():
    feat = spread_features(candidate(entry_credit=2.0, width=5.0))
    assert feat["credit_to_width"] == pytest.approx(0.4)


def test_bid_ask_pct_is_a_fraction_of_mid_not_of_implied_vol():
    c = candidate(short_bid=2.0, short_ask=2.4)          # mid 2.2
    assert spread_features(c)["short_put_bid_ask_pct"] == pytest.approx(0.4 / 2.2)


def test_greeks_and_liquidity_come_from_the_candidate_not_zero_stubs():
    feat = spread_features(candidate())
    for col in ("short_put_open_interest", "short_put_volume", "long_put_delta",
                "net_theta", "net_vega", "net_gamma"):
        assert feat[col] != 0.0, f"{col} is a zero stub"


def test_net_greeks_have_the_right_signs_for_a_credit_spread():
    feat = spread_features(candidate())
    assert feat["net_theta"] > 0      # time decay helps
    assert feat["net_vega"] < 0       # rising IV hurts
    assert feat["net_delta"] > 0      # bullish position


def test_break_even_distance_is_negative_when_break_even_is_below_spot():
    feat = spread_features(candidate(short_strike=400.0, entry_credit=2.0,
                                     underlying_price=450.0))
    assert feat["break_even_distance_pct"] == pytest.approx((398.0 - 450.0) / 450.0)


# ----------------------------------------------------------------- calendar
def test_calendar_features_emit_every_calendar_column():
    """days_to_jobs_report used to be missing, so it was silently always 0."""
    feat = calendar_features(datetime(2024, 3, 1, 15, 45), open_calendar())
    assert set(CALENDAR_COLS) <= set(feat)


def test_calendar_features_values():
    cal = open_calendar(FOMC=[date(2024, 3, 20)])
    feat = calendar_features(datetime(2024, 3, 1, 15, 45), cal)
    assert feat["month"] == 3
    assert feat["weekday"] == 4                      # Friday
    assert feat["days_to_fomc"] == 19


def test_event_features_are_nan_outside_calendar_coverage():
    """Not 0 -- a model trained on constant zeros learns nothing from the column."""
    blind = EventCalendar(dates={"FOMC": frozenset([date(2030, 1, 1)]),
                                 "CPI": frozenset(), "JOBS": frozenset()},
                          coverage_start=date(2030, 1, 1),
                          coverage_end=date(2031, 1, 1), source="test")
    feat = calendar_features(datetime(2024, 3, 1, 15, 45), blind)
    assert math.isnan(feat["days_to_fomc"])


# -------------------------------------------------------------- integration
def test_build_feature_vector_produces_exactly_the_declared_columns():
    vec = build_feature_vector(candidate(), bars(80), [], TS, open_calendar())
    assert list(vec.keys()) == FEATURE_COLS
    assert len(vec) == N_FEATURES


def test_feature_index_matches_column_order():
    """simple_rule_predict indexes by name through this map, not a literal int."""
    for name, i in FEATURE_INDEX.items():
        assert FEATURE_COLS[i] == name


def test_to_matrix_preserves_column_order():
    v = build_feature_vector(candidate(), bars(80), [], TS, open_calendar())
    X = to_matrix([v])
    assert X.shape == (1, N_FEATURES)
    assert X[0, FEATURE_INDEX["credit_to_width"]] == pytest.approx(v["credit_to_width"])


def test_nan_report_and_assert_no_nan():
    X = np.array([[1.0, np.nan] + [0.0] * (N_FEATURES - 2)])
    assert FEATURE_COLS[1] in nan_report(X)
    with pytest.raises(ValueError, match="NaN features present"):
        assert_no_nan(X)


def test_drop_nan_rows_keeps_only_complete_rows():
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    y = np.array([1, 0, 1])
    Xc, yc, keep = drop_nan_rows(X, y)
    assert Xc.shape == (2, 2)
    assert list(yc) == [1, 1]
    assert list(keep) == [True, False, True]
