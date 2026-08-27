"""Generic sample-dataset adapter.

Column names differ between CBOE DataShop, Kaggle dumps and ORATS samples, so the
mapping is data, not code: pass a `colmap` and the same normalizer handles a new
vendor without a new module.
"""
from __future__ import annotations

from datetime import date, datetime

from data.adapters.base import NormalizeResult
from schemas import OptionQuote

# canonical field -> candidate source column names, tried in order
DEFAULT_COLMAP = {
    "timestamp": ("timestamp", "quote_date", "date", "quotedate"),
    "expiry": ("expiry", "expiration", "expirationdate", "exdate"),
    "strike": ("strike", "strike_price", "strikeprice"),
    "option_type": ("type", "option_type", "right", "cp_flag"),
    "bid": ("bid", "best_bid", "bid_price"),
    "ask": ("ask", "best_ask", "ask_price"),
    "last": ("last", "last_price", "close"),
    "volume": ("volume", "vol", "trade_volume"),
    "open_interest": ("oi", "open_interest", "openinterest"),
    "iv": ("iv", "implied_volatility", "impliedvol"),
    "delta": ("delta",),
    "gamma": ("gamma",),
    "theta": ("theta",),
    "vega": ("vega",),
    "underlying_price": ("underlying", "underlying_price", "stock_price", "spot"),
}


def _pick(row: dict, names) -> tuple[str, object] | tuple[None, None]:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for n in names:
        if n in lowered and lowered[n] not in (None, ""):
            return n, lowered[n]
    return None, None


def _to_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v).strip()[:10])


def _to_dt(v) -> datetime:
    if isinstance(v, datetime):
        return v
    s = str(v).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.combine(_to_date(s), datetime.min.time())


def _num(v, cast=float, default=None):
    if v in (None, ""):
        return default
    try:
        return cast(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


def normalize_rows(rows, symbol: str = "SPY", colmap: dict | None = None) -> NormalizeResult:
    """Map arbitrary source rows onto canonical OptionQuote records."""
    cm = colmap or DEFAULT_COLMAP
    res = NormalizeResult()
    for r in rows:
        try:
            _, ts_raw = _pick(r, cm["timestamp"])
            _, exp_raw = _pick(r, cm["expiry"])
            _, strike_raw = _pick(r, cm["strike"])
            _, type_raw = _pick(r, cm["option_type"])
            if ts_raw is None or exp_raw is None or strike_raw is None or type_raw is None:
                res.rejects.append(("missing_key_field", r))
                continue

            ts, expiry = _to_dt(ts_raw), _to_date(exp_raw)
            strike = _num(strike_raw)
            opt = str(type_raw).strip().upper()[:1]
            if opt not in ("C", "P") or strike is None:
                res.rejects.append(("unparseable_key_field", r))
                continue

            bid = _num(_pick(r, cm["bid"])[1], default=None)
            ask = _num(_pick(r, cm["ask"])[1], default=None)
            if bid is None or ask is None:
                res.rejects.append(("missing_quote", r))
                continue

            # Fields that may legitimately be absent become NaN + a flag, never 0.
            flags = []
            greeks = {}
            for name in ("iv", "delta", "gamma", "theta", "vega"):
                val = _num(_pick(r, cm[name])[1], default=None)
                if val is None:
                    flags.append(f"missing_{name}")
                    val = float("nan")
                greeks[name] = val
            und = _num(_pick(r, cm["underlying_price"])[1], default=None)
            if und is None:
                flags.append("missing_underlying")
                und = float("nan")

            res.quotes.append(OptionQuote(
                timestamp=ts, symbol=symbol, expiry=expiry,
                dte=(expiry - ts.date()).days, strike=strike, option_type=opt,
                bid=bid, ask=ask, mid=(bid + ask) / 2.0,
                last=_num(_pick(r, cm["last"])[1], default=float("nan")),
                volume=int(_num(_pick(r, cm["volume"])[1], default=0) or 0),
                open_interest=int(_num(_pick(r, cm["open_interest"])[1], default=0) or 0),
                underlying_price=und, quality_flags=tuple(flags), **greeks,
            ))
        except Exception as e:                      # noqa: BLE001 - row must not kill the load
            res.rejects.append((f"{type(e).__name__}", r))
    return res


def normalize_option_quotes(rows, symbol: str = "SPY", colmap: dict | None = None) -> list:
    """Convenience wrapper returning just the quotes."""
    return normalize_rows(rows, symbol=symbol, colmap=colmap).quotes
