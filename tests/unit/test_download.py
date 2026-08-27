from datetime import datetime

import pandas as pd

from data.download import download_underlying, download_vix


def fake_fetch(ticker, start, end):
    """A plain dict -- the shape that used to crash on df.iterrows()."""
    return {"Close": [100.0, 101.0], "Volume": [1000, 1100]}


def test_download_underlying_builds_bars_from_a_dict_fetcher():
    bars = download_underlying("SPY", "2024-01-02", "2024-01-03", fetch_raw=fake_fetch)
    assert len(bars) == 2
    assert bars[0].close == 100.0
    assert bars[0].symbol == "SPY"


def test_download_underlying_handles_multiindex_columns():
    """yfinance returns MultiIndex columns; row["Close"] would be a Series."""
    df = pd.DataFrame({("Close", "SPY"): [470.0], ("Open", "SPY"): [469.0],
                       ("Volume", "SPY"): [8_000_000]},
                      index=pd.to_datetime(["2024-01-02"]))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    bars = download_underlying("SPY", "x", "y", fetch_raw=lambda *a: df)
    assert len(bars) == 1
    assert bars[0].close == 470.0 and bars[0].open == 469.0
    assert isinstance(bars[0].close, float)


def test_download_underlying_returns_empty_on_empty_frame():
    assert download_underlying("SPY", "x", "y", fetch_raw=lambda *a: pd.DataFrame()) == []


def test_bars_come_back_in_chronological_order():
    df = pd.DataFrame({"Close": [3.0, 1.0, 2.0]},
                      index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]))
    bars = download_underlying("SPY", "x", "y", fetch_raw=lambda *a: df)
    assert [b.close for b in bars] == [1.0, 2.0, 3.0]


def test_download_vix_uses_the_vix_ticker():
    seen = {}
    def spy_fetch(ticker, start, end):
        seen["ticker"] = ticker
        return {"Close": [15.0]}
    download_vix("2024-01-02", "2024-01-03", fetch_raw=spy_fetch)
    assert seen["ticker"] == "^VIX"
