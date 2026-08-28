"""Adapter for optionsDX end-of-day option-chain files.

Format notes, because this one is not the generic shape `sample.py` handles:

* WIDE, NOT LONG. Each row carries BOTH the call and the put for one
  (date, expiry, strike), with `C_` and `P_` prefixes. The canonical schema is
  one row per contract, so this unpivots.
* HEADERS ARE BRACKETED AND PADDED: `[QUOTE_DATE], [UNDERLYING_LAST], ...`.
* NO OPEN INTEREST. The file has `P_SIZE` ("0 x 5240"), which is bid size x ask
  size at the close, not open interest. Downstream `min_oi` filtering must
  therefore be disabled or it rejects every candidate; use volume and quoted
  size as the liquidity screen instead. `open_interest` is set to 0 and the
  quote is flagged `no_open_interest_in_source` so nothing mistakes that 0 for
  an observed value.
* QUOTES ARE STAMPED 16:00, the close. The spec asks for a 3:45 PM decision;
  end-of-day data cannot provide that. Decisions here are made on the closing
  snapshot, 15 minutes later than specified. That is a real difference: a
  strategy that decides at the close cannot be executed at the close.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DECISION_HOUR = 16          # optionsDX EOD snapshot time

# canonical name -> source column
COMMON = {
    "quote_date": "QUOTE_DATE",
    "underlying_price": "UNDERLYING_LAST",
    "expiry": "EXPIRE_DATE",
    "dte": "DTE",
    "strike": "STRIKE",
}
LEG_FIELDS = ("BID", "ASK", "LAST", "DELTA", "GAMMA", "VEGA", "THETA", "RHO",
              "IV", "VOLUME", "SIZE")

CANONICAL_COLUMNS = [
    "timestamp", "symbol", "expiry", "dte", "strike", "option_type",
    "bid", "ask", "mid", "last", "volume", "open_interest", "iv",
    "delta", "gamma", "theta", "vega", "underlying_price",
    "bid_size", "ask_size",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().strip("[]").strip().upper() for c in df.columns]
    return df


def _split_size(series: pd.Series):
    """'0 x 5240' -> (0, 5240). Missing or malformed becomes NaN, never 0."""
    parts = series.astype(str).str.split("x", n=1, expand=True)
    if parts.shape[1] < 2:
        nan = pd.Series(np.nan, index=series.index)
        return nan, nan
    return (pd.to_numeric(parts[0].str.strip(), errors="coerce"),
            pd.to_numeric(parts[1].str.strip(), errors="coerce"))


def read_optionsdx(path, *, option_types=("P",), min_dte: int = 0,
                   max_dte: int = 60, symbol: str = "SPY") -> pd.DataFrame:
    """Read one monthly file into canonical long form.

    `min_dte`/`max_dte` filter during the read. An entry at 45 DTE is marked
    forward until the 7-DTE exit, so a window of roughly 0-60 keeps everything
    the simulator needs while discarding the LEAPS that dominate row count.
    """
    df = clean_columns(pd.read_csv(path, skipinitialspace=True, low_memory=False))

    dte = pd.to_numeric(df["DTE"], errors="coerce")
    df = df[(dte >= min_dte) & (dte <= max_dte)].copy()
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    quote_date = pd.to_datetime(df[COMMON["quote_date"]], errors="coerce")
    frames = []
    for opt in option_types:
        prefix = "P_" if opt == "P" else "C_"
        cols = {f: f"{prefix}{f}" for f in LEG_FIELDS}
        bid = pd.to_numeric(df[cols["BID"]], errors="coerce")
        ask = pd.to_numeric(df[cols["ASK"]], errors="coerce")
        bid_size, ask_size = _split_size(df[cols["SIZE"]])

        out = pd.DataFrame({
            "timestamp": quote_date + pd.Timedelta(hours=DECISION_HOUR),
            "symbol": symbol,
            "expiry": pd.to_datetime(df[COMMON["expiry"]], errors="coerce"),
            "dte": pd.to_numeric(df["DTE"], errors="coerce").round().astype("Int64"),
            "strike": pd.to_numeric(df[COMMON["strike"]], errors="coerce"),
            "option_type": opt,
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0,
            "last": pd.to_numeric(df[cols["LAST"]], errors="coerce"),
            "volume": pd.to_numeric(df[cols["VOLUME"]], errors="coerce"),
            # Not in the source. Left as 0 and flagged downstream; see module docstring.
            "open_interest": 0,
            "iv": pd.to_numeric(df[cols["IV"]], errors="coerce"),
            "delta": pd.to_numeric(df[cols["DELTA"]], errors="coerce"),
            "gamma": pd.to_numeric(df[cols["GAMMA"]], errors="coerce"),
            "theta": pd.to_numeric(df[cols["THETA"]], errors="coerce"),
            "vega": pd.to_numeric(df[cols["VEGA"]], errors="coerce"),
            "underlying_price": pd.to_numeric(df[COMMON["underlying_price"]],
                                              errors="coerce"),
            "bid_size": bid_size,
            "ask_size": ask_size,
        })
        frames.append(out)

    res = pd.concat(frames, ignore_index=True)
    res = res.dropna(subset=["timestamp", "expiry", "strike"])
    return res[CANONICAL_COLUMNS]


def convert_directory(src_dir, out_path, *, option_types=("P",), min_dte: int = 0,
                      max_dte: int = 60, symbol: str = "SPY",
                      progress=print) -> dict:
    """Convert every monthly .txt in `src_dir` into one canonical Parquet file."""
    src, out = Path(src_dir), Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files in {src}")

    # Stream month by month through a ParquetWriter rather than concatenating
    # ~5M rows in memory first. The files are already in chronological order, so
    # the output stays sorted without a global sort.
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    rows_in, days, lo, hi = 0, set(), None, None
    try:
        for i, f in enumerate(files, 1):
            part = read_optionsdx(f, option_types=option_types, min_dte=min_dte,
                                  max_dte=max_dte, symbol=symbol)
            if part.empty:
                continue
            part = part.sort_values(["timestamp", "expiry", "strike"])
            rows_in += len(part)
            days.update(part.timestamp.dt.date.unique())
            lo = part.timestamp.min() if lo is None else min(lo, part.timestamp.min())
            hi = part.timestamp.max() if hi is None else max(hi, part.timestamp.max())

            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="snappy")
            writer.write_table(table)
            if progress and (i % 12 == 0 or i == len(files)):
                progress(f"  {i:3d}/{len(files)} files  {rows_in:,} rows kept")
    finally:
        if writer is not None:
            writer.close()

    return {
        "files": len(files),
        "rows": rows_in,
        "out": str(out),
        "size_mb": out.stat().st_size / 1e6,
        "date_min": lo,
        "date_max": hi,
        "trading_days": len(days),
    }


def to_quote_dicts(df: pd.DataFrame):
    """Yield row dicts in the shape `data.adapters.sample.normalize_rows` expects."""
    for rec in df.to_dict("records"):
        yield {
            "timestamp": rec["timestamp"].isoformat() if isinstance(
                rec["timestamp"], (pd.Timestamp, datetime)) else rec["timestamp"],
            "expiry": (rec["expiry"].date().isoformat()
                       if isinstance(rec["expiry"], (pd.Timestamp, datetime))
                       else rec["expiry"]),
            "strike": rec["strike"], "type": rec["option_type"],
            "bid": rec["bid"], "ask": rec["ask"], "last": rec["last"],
            "volume": rec["volume"], "oi": rec["open_interest"],
            "iv": rec["iv"], "delta": rec["delta"], "gamma": rec["gamma"],
            "theta": rec["theta"], "vega": rec["vega"],
            "underlying": rec["underlying_price"],
        }
