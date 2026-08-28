"""Index quotes by contract so forward marks are a lookup, not a scan.

The problem this solves
-----------------------
`snapshots_from_quotes` walks the entire quote stream for every candidate. On a
300-row fixture that is invisible. On the real dataset it is
67,722 candidates x 3,718,691 quotes, measured at 11.4 hours -- and that is
before the cost of turning 3.7M rows into dataclass instances.

The fix is the obvious one: a spread's marks only ever come from two contracts,
so index by (expiry, strike) once and each candidate becomes two lookups over
~60 rows instead of a full pass over millions.

Storage note: rows are sorted once and each contract's arrays are VIEWS into
those shared sorted arrays. Nothing is copied per contract, so the index costs
roughly the dataframe itself rather than a multiple of it.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np

from labels.simulator import SpreadSnapshot

NS = 1_000_000_000


def _to_ns(dt: datetime) -> int:
    """Naive datetime -> epoch nanoseconds, with NO timezone interpretation.

    `datetime.timestamp()` applies the local UTC offset while pandas'
    datetime64 conversion does not, so mixing the two shifts every mark by
    hours. numpy's conversion is used on both build paths and on the way back.
    """
    return int(np.datetime64(dt, "ns").astype(np.int64))


def _from_ns(ns: int) -> datetime:
    return np.datetime64(int(ns), "ns").astype("datetime64[us]").astype(datetime)


def _strike_key(x) -> int:
    """Strikes as integer cents. Float keys do not compare reliably."""
    return int(round(float(x) * 100))


class QuoteIndex:
    """(expiry, strike) -> chronologically sorted (timestamp, bid, ask, underlying)."""

    __slots__ = ("_legs", "_ts", "_bid", "_ask", "_und", "n_rows", "n_contracts")

    def __init__(self, ts_ns, bid, ask, und, keys):
        order = np.lexsort((ts_ns, keys))
        self._ts = np.ascontiguousarray(ts_ns[order])
        self._bid = np.ascontiguousarray(bid[order])
        self._ask = np.ascontiguousarray(ask[order])
        self._und = np.ascontiguousarray(und[order])
        sorted_keys = keys[order]

        # Slice boundaries per contract; values are (start, stop) into the
        # shared arrays above, so each contract's data is a view.
        uniq, starts = np.unique(sorted_keys, return_index=True)
        stops = np.append(starts[1:], len(sorted_keys))
        self._legs = {int(k): (int(a), int(b))
                      for k, a, b in zip(uniq, starts, stops)}
        self.n_rows = len(sorted_keys)
        self.n_contracts = len(self._legs)

    # ------------------------------------------------------------- builders
    @staticmethod
    def _pack_key(expiry_ordinal, strike_cents) -> np.ndarray:
        # One int64 per contract: expiry ordinal in the high bits, strike in
        # the low. Cheaper to sort and group than tuples of Python objects.
        return (np.asarray(expiry_ordinal, dtype=np.int64) << 24) | np.asarray(
            strike_cents, dtype=np.int64)

    @classmethod
    def from_frame(cls, df, option_type: str = "P") -> "QuoteIndex":
        """Build from the canonical parquet frame. This is the fast path."""
        import pandas as pd

        if "option_type" in df.columns:
            df = df[df.option_type == option_type]
        ts = pd.to_datetime(df["timestamp"]).to_numpy("datetime64[ns]").astype(np.int64)
        expiry = pd.to_datetime(df["expiry"])
        exp_ord = (expiry.dt.year * 10000 + expiry.dt.month * 100
                   + expiry.dt.day).to_numpy(np.int64)
        strikes = np.round(df["strike"].to_numpy(float) * 100).astype(np.int64)
        keys = cls._pack_key(exp_ord, strikes)
        return cls(ts, df["bid"].to_numpy(float), df["ask"].to_numpy(float),
                   df["underlying_price"].to_numpy(float), keys)

    @classmethod
    def from_quotes(cls, quotes, option_type: str = "P") -> "QuoteIndex":
        """Build from OptionQuote objects. For tests and small inputs."""
        rows = [q for q in quotes if q.option_type == option_type]
        if not rows:
            empty_i = np.empty(0, np.int64)
            return cls(empty_i, np.empty(0), np.empty(0), np.empty(0), empty_i)
        ts = np.array([_to_ns(q.timestamp) for q in rows], np.int64)
        exp_ord = np.array([q.expiry.year * 10000 + q.expiry.month * 100
                            + q.expiry.day for q in rows], np.int64)
        strikes = np.array([_strike_key(q.strike) for q in rows], np.int64)
        return cls(ts, np.array([q.bid for q in rows], float),
                   np.array([q.ask for q in rows], float),
                   np.array([q.underlying_price for q in rows], float),
                   cls._pack_key(exp_ord, strikes))

    # -------------------------------------------------------------- lookups
    def _slice(self, expiry: date, strike: float):
        key = int((np.int64(expiry.year * 10000 + expiry.month * 100 + expiry.day)
                   << 24) | np.int64(_strike_key(strike)))
        return self._legs.get(key)

    def contract_marks(self, expiry: date, strike: float):
        """(timestamps_ns, bid, ask, underlying) for one contract, or None."""
        sl = self._slice(expiry, strike)
        if sl is None:
            return None
        a, b = sl
        return self._ts[a:b], self._bid[a:b], self._ask[a:b], self._und[a:b]

    def snapshots_for(self, candidate) -> list:
        """Forward marks for a spread: same contract pair, same timestamps, after entry.

        Equivalent to `labels.simulator.snapshots_from_quotes` and returns marks
        in the same order; only the lookup strategy differs.
        """
        short = self.contract_marks(candidate.expiry, candidate.short_strike)
        long = self.contract_marks(candidate.expiry, candidate.long_strike)
        if short is None or long is None:
            return []

        s_ts, _, s_ask, s_und = short
        l_ts, l_bid, _, _ = long

        # Only timestamps where BOTH legs quote can mark the spread. Half a
        # spread is not a price.
        common, si, li = np.intersect1d(s_ts, l_ts, assume_unique=False,
                                        return_indices=True)
        if common.size == 0:
            return []

        entry_ns = _to_ns(candidate.entry_time)
        keep = common > entry_ns
        if not keep.any():
            return []

        common, si, li = common[keep], si[keep], li[keep]
        return [
            SpreadSnapshot(
                time=_from_ns(t),
                short_ask=float(sa), long_bid=float(lb), underlying_price=float(u),
            )
            for t, sa, lb, u in zip(common, s_ask[si], l_bid[li], s_und[si])
        ]

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        return f"QuoteIndex({self.n_rows:,} rows, {self.n_contracts:,} contracts)"
