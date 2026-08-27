"""Generate a small, internally consistent synthetic option-chain dataset.

Real sample data is not in the repo, so the tests need something that behaves
like an option chain: deltas that line up with strikes, prices that decay toward
expiry, and a strike grid dense enough to build a $5 spread. Black-Scholes gives
all of that for free and keeps the chain self-consistent, which a table of
hand-typed numbers would not.

This is fixture data for testing the plumbing. It is not market data and no
result computed from it means anything about the strategy.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np

TRADING_DAYS = 252
DECISION = (15, 45)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Black-Scholes put price and greeks. T in years."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(K - S, 0.0)
        return {"price": intrinsic, "delta": -1.0 if K > S else 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return {
        "price": price,
        "delta": _norm_cdf(d1) - 1.0,                       # negative for puts
        "gamma": _norm_pdf(d1) / (S * sigma * math.sqrt(T)),
        "theta": (-(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
                  + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365.0,
        "vega": S * _norm_pdf(d1) * math.sqrt(T) / 100.0,
    }


def make_underlying(n_days: int = 420, start: date = date(2023, 1, 3),
                    s0: float = 400.0, mu: float = 0.07, sigma: float = 0.16,
                    seed: int = 7):
    """Geometric-Brownian-motion daily closes on weekdays only."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    out, s, d = [], s0, start
    while len(out) < n_days:
        if d.weekday() < 5:
            shock = rng.normal((mu - 0.5 * sigma ** 2) * dt, sigma * math.sqrt(dt))
            s *= math.exp(shock)
            out.append((datetime(d.year, d.month, d.day, *DECISION), round(s, 2)))
        d += timedelta(days=1)
    return out


def make_vix(underlying, base: float = 16.0, seed: int = 8):
    """A VIX-like series that rises when the underlying falls."""
    rng = np.random.default_rng(seed)
    out, prev = [], underlying[0][1]
    level = base
    for ts, px in underlying:
        ret = (px - prev) / prev if prev else 0.0
        level = max(9.0, 0.90 * level + 0.10 * base - 260.0 * ret + rng.normal(0, 0.5))
        out.append((ts, round(level, 2)))
        prev = px
    return out


def make_chain_rows(underlying, vix, *, strike_step: float = 5.0,
                    n_strikes: int = 14, dtes=(30, 37, 44), r: float = 0.04,
                    spread_pct: float = 0.06, seed: int = 9):
    """Rows in the raw-source dict shape the adapter consumes."""
    rng = np.random.default_rng(seed)
    vix_by_ts = dict(vix)
    rows = []
    for ts, spot in underlying:
        iv = max(0.08, vix_by_ts.get(ts, 16.0) / 100.0)
        for dte in dtes:
            expiry = (ts + timedelta(days=dte)).date()
            atm = round(spot / strike_step) * strike_step
            for k in range(-n_strikes, 3):
                strike = atm + k * strike_step
                if strike <= 0:
                    continue
                g = bs_put(spot, strike, dte / 365.0, r, iv)
                mid = max(g["price"], 0.01)
                half = max(mid * spread_pct / 2.0, 0.01)
                rows.append({
                    "timestamp": ts.isoformat(),
                    "expiry": expiry.isoformat(),
                    "strike": f"{strike:.2f}",
                    "type": "P",
                    "bid": f"{max(mid - half, 0.01):.2f}",
                    "ask": f"{mid + half:.2f}",
                    "last": f"{mid:.2f}",
                    "volume": str(int(rng.integers(50, 900))),
                    "oi": str(int(rng.integers(400, 9000))),
                    "iv": f"{iv:.4f}",
                    "delta": f"{g['delta']:.4f}",
                    "gamma": f"{g['gamma']:.6f}",
                    "theta": f"{g['theta']:.4f}",
                    "vega": f"{g['vega']:.4f}",
                    "underlying": f"{spot:.2f}",
                })
    return rows


def make_fixture(n_days: int = 420, seed: int = 7, warmup_days: int = 0):
    """Return (underlying_rows, vix_rows, chain_rows) for the whole pipeline.

    `warmup_days` withholds the chain for the first N days while keeping the full
    price history. Features like `vix_percentile_252d` need a year of bars before
    the first decision, so without a warmup every row is dropped as incomplete --
    which is the correct behaviour and a useless fixture. Real usage has the same
    shape: long price history, shorter option-chain sample.
    """
    u = make_underlying(n_days=n_days, seed=seed)
    v = make_vix(u, seed=seed + 1)
    c = make_chain_rows(u[warmup_days:], v[warmup_days:], seed=seed + 2)
    return u, v, c


def to_bars(series, symbol: str = "SPY"):
    from schemas import UnderlyingBar
    return [UnderlyingBar(timestamp=ts, symbol=symbol, open=px, high=px, low=px,
                          close=px, adj_close=px, volume=1_000_000)
            for ts, px in series]


def write_parquet(rows, path) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_parquet(path, index=False)


if __name__ == "__main__":
    u, v, c = make_fixture()
    print(f"underlying bars {len(u)}  vix bars {len(v)}  chain rows {len(c)}")
    print(f"range {u[0][0].date()} .. {u[-1][0].date()}  spot {u[0][1]} -> {u[-1][1]}")
