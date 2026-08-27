"""Raw-data validation. Flags records, never edits them.

Spec rule: raw data is immutable. A record is either trustworthy or carries a
flag saying why it is not -- we never "repair" a quote, because a repaired quote
is indistinguishable from a real one three months later.
"""
from __future__ import annotations

import math
from collections import Counter

from schemas import OptionQuote, with_flags

SPREAD_PCT_LIMIT = 0.50
MAX_PLAUSIBLE_IV = 5.0        # 500%
STALE_QUOTE_DAYS = 3


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def validate_quote(q: OptionQuote, *, max_spread_pct: float = SPREAD_PCT_LIMIT,
                   max_iv: float = MAX_PLAUSIBLE_IV) -> OptionQuote:
    """Return a copy of `q` with quality flags appended."""
    flags: list[str] = []

    if not _isnan(q.bid) and not _isnan(q.ask) and q.bid > q.ask:
        flags.append("bid_above_ask")
    if q.bid < 0 or q.ask < 0:
        flags.append("negative_quote")
    if q.volume < 0 or q.open_interest < 0:
        flags.append("negative_size")
    if q.dte < 0:
        flags.append("negative_dte")

    # IV: separate the three cases. The old code called a zero IV "negative_iv",
    # which made a missing field and a bad field look identical downstream.
    if _isnan(q.iv):
        flags.append("missing_iv")
    elif q.iv < 0:
        flags.append("negative_iv")
    elif q.iv == 0:
        flags.append("zero_iv")
    elif q.iv > max_iv:
        flags.append("implausible_iv")

    if q.bid == 0.0:
        flags.append("zero_bid")
    if q.ask == 0.0:
        flags.append("zero_ask")

    if not _isnan(q.mid) and q.mid > 0 and q.ask >= q.bid:
        if (q.ask - q.bid) / q.mid > max_spread_pct:
            flags.append("wide_spread")

    if _isnan(q.underlying_price) or q.underlying_price <= 0:
        flags.append("missing_underlying")

    if q.expiry <= q.timestamp.date():
        flags.append("expiry_before_entry")

    if q.option_type not in ("C", "P"):
        flags.append("bad_option_type")

    for name in ("delta", "gamma", "theta", "vega"):
        if _isnan(getattr(q, name)):
            flags.append(f"missing_{name}")
    if not _isnan(q.delta) and not (-1.0 <= q.delta <= 1.0):
        flags.append("implausible_delta")
    if not _isnan(q.delta):
        if q.option_type == "P" and q.delta > 0:
            flags.append("put_delta_positive")
        if q.option_type == "C" and q.delta < 0:
            flags.append("call_delta_negative")

    return with_flags(q, flags)


def validate_quotes(quotes: list[OptionQuote], **kw) -> list[OptionQuote]:
    """Validate each quote, then add list-level flags (duplicates, staleness)."""
    out = [validate_quote(q, **kw) for q in quotes]

    key = lambda q: (q.timestamp, q.symbol, q.expiry, q.strike, q.option_type)  # noqa: E731
    dupes = {k for k, n in Counter(key(q) for q in out).items() if n > 1}
    if dupes:
        out = [with_flags(q, ["duplicate_contract"]) if key(q) in dupes else q for q in out]

    # Staleness: an identical bid/ask on the same contract for N+ consecutive
    # observations usually means the feed stopped updating, not a calm market.
    runs: dict = {}
    stale_ids = set()
    for i, q in enumerate(sorted(out, key=lambda x: (x.strike, x.expiry, x.option_type, x.timestamp))):
        ck = (q.strike, q.expiry, q.option_type)
        prev = runs.get(ck)
        if prev and prev[0] == (q.bid, q.ask):
            run = prev[1] + 1
            if run >= STALE_QUOTE_DAYS:
                stale_ids.add(id(q))
        else:
            run = 1
        runs[ck] = ((q.bid, q.ask), run)
    if stale_ids:
        out = [with_flags(q, ["stale_quote"]) if id(q) in stale_ids else q for q in out]
    return out


def flag_summary(quotes: list[OptionQuote]) -> dict:
    """Count each flag across a batch -- the number you look at after ingest."""
    c: Counter = Counter()
    for q in quotes:
        for f in q.quality_flags:
            c[f] += 1
    c["_total_quotes"] = len(quotes)
    c["_clean_quotes"] = sum(1 for q in quotes if q.is_clean)
    return dict(c)


def validate_strike_order(short_strike: float, long_strike: float) -> str:
    """A put credit spread must buy the LOWER strike. Returns "" when correct."""
    if long_strike >= short_strike:
        return "bad_strike_order"
    return ""
