"""Underlying and VIX bar download (Yahoo Finance), with an injectable fetcher."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from schemas import UnderlyingBar


def _default_fetch(ticker: str, start: str, end: str):
    import yfinance as yf
    return yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)


def _as_frame(raw) -> pd.DataFrame:
    """Accept a DataFrame or any dict/records shape a test fetcher might return.

    yfinance returns a DataFrame with a MultiIndex column header when given a
    list of tickers, so flatten that too -- otherwise row["Close"] silently
    returns a Series instead of a float.
    """
    df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _cell(row, key, default=None):
    val = row.get(key, default)
    if isinstance(val, pd.Series):
        val = val.iloc[0] if len(val) else default
    return default if val is None or pd.isna(val) else float(val)


def download_underlying(symbol: str, start: str, end: str, fetch_raw=None) -> list[UnderlyingBar]:
    """Fetch daily bars. `fetch_raw(ticker, start, end)` is injectable for tests."""
    df = _as_frame((fetch_raw or _default_fetch)(symbol, start, end))
    if df.empty:
        return []
    bars: list[UnderlyingBar] = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if not isinstance(ts, datetime):
            ts = datetime.combine(pd.to_datetime(ts).date(), datetime.min.time())
        close = _cell(row, "Close")
        if close is None:
            continue
        bars.append(UnderlyingBar(
            timestamp=ts, symbol=symbol,
            open=_cell(row, "Open", close), high=_cell(row, "High", close),
            low=_cell(row, "Low", close), close=close,
            adj_close=_cell(row, "Adj Close", close),
            volume=int(_cell(row, "Volume", 0.0) or 0),
        ))
    bars.sort(key=lambda b: b.timestamp)
    return bars


def download_vix(start: str, end: str, fetch_raw=None) -> list[UnderlyingBar]:
    return download_underlying("^VIX", start, end, fetch_raw=fetch_raw)
