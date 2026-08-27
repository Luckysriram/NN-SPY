"""Recover missing Greeks on canonical quotes by inverting Black-Scholes.

Ordering note: run this AFTER `validate_quotes` and BEFORE `generate_candidates`.
Validation decides whether a quote is trustworthy; there is no point solving for
implied vol from a bid/ask pair already flagged as crossed or stale.

Two rules this module holds to:

* IT NEVER OVERWRITES VENDOR GREEKS. A field that arrived populated stays as it
  arrived. Only NaNs get filled.
* A COMPUTED GREEK IS LABELLED AS ONE. Every enriched quote carries a
  `greeks_computed` flag, so no downstream analysis can mistake a modelled delta
  for an observed one. That distinction matters: the whole -0.20 delta strike
  selection rests on it, and a computed delta inherits every error in the rate
  and dividend assumptions behind it.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

from models.blackscholes import (bs_greeks, early_exercise_risk,
                                 implied_vol_detail)
from schemas import OptionQuote, with_flags

DAYS_PER_YEAR = 365.0
GREEK_FIELDS = ("delta", "gamma", "theta", "vega")


def _isnan(x) -> bool:
    return isinstance(float(x), float) and math.isnan(float(x))


def needs_greeks(q: OptionQuote) -> bool:
    return any(_isnan(getattr(q, f)) for f in GREEK_FIELDS) or _isnan(q.iv)


IV_DISAGREEMENT_TOL = 0.02      # 2 vol points


def enrich_quote(q: OptionQuote, r: float, dividend_yield: float, *,
                 use_mid: bool = True, prefer_solved_iv: bool = False) -> OptionQuote:
    """Fill NaN Greeks/IV on one quote. Returns it unchanged when nothing is missing.

    When a vendor supplies IV but no Greeks -- the common case, and exactly what
    Yahoo does -- the Greeks here are differentiated at the SOLVED IV, not the
    vendor's. If the two disagree the record would be self-contradictory: an `iv`
    field and a `delta` field describing different volatilities. Any material
    disagreement is therefore flagged `iv_inconsistent_with_price`.

    Set `prefer_solved_iv=True` to keep the record internally consistent by
    storing the solved value instead. That is the right choice when the vendor
    IV has failed a sanity check -- e.g. only a handful of distinct values across
    thousands of contracts, which means the column is a placeholder rather than a
    surface.
    """
    if not needs_greeks(q):
        return q

    T = q.dte / DAYS_PER_YEAR
    S = q.underlying_price
    price = q.mid if use_mid else q.last

    if _isnan(S) or S <= 0:
        return with_flags(q, ["greeks_failed_missing_underlying"])
    if _isnan(price) or price <= 0:
        return with_flags(q, ["greeks_failed_no_price"])

    res = implied_vol_detail(price, S, q.strike, T, r, dividend_yield, q.option_type)
    if not res.ok:
        return with_flags(q, [f"greeks_failed_{res.reason}"])

    g = bs_greeks(S, q.strike, T, r, dividend_yield, res.iv, q.option_type)

    # Fill only what is missing; vendor values win wherever they exist.
    updates = {}
    flags = ["greeks_computed"]

    if _isnan(q.iv):
        updates["iv"] = res.iv
    elif abs(q.iv - res.iv) > IV_DISAGREEMENT_TOL:
        # The vendor's IV does not reprice this quote. The Greeks below come
        # from the solved IV, so leaving the vendor value in place produces a
        # record whose iv and delta describe different volatilities.
        flags.append("iv_inconsistent_with_price")
        if prefer_solved_iv:
            updates["iv"] = res.iv

    for field, value in (("delta", g.delta), ("gamma", g.gamma),
                         ("theta", g.theta), ("vega", g.vega)):
        if _isnan(getattr(q, field)):
            updates[field] = value

    if early_exercise_risk(S, q.strike, T, r, dividend_yield, q.option_type):
        # European model, American contract. Flagged rather than corrected --
        # correcting it needs a binomial/PDE pricer, which is a bigger change
        # than this module should make silently.
        flags.append("early_exercise_risk")

    return with_flags(replace(q, **updates), flags)


def enrich_quotes(quotes, market_params=None, *, r=None, dividend_yield=None,
                  use_mid: bool = True, prefer_solved_iv: bool = False) -> list:
    """Enrich a batch. Supply either `market_params` (date-varying) or flat r/q.

    `market_params` is strongly preferred: a flat rate across a multi-year
    dataset biases every delta in the same direction.
    """
    if market_params is None and (r is None or dividend_yield is None):
        raise ValueError(
            "supply market_params (see data/rates.py) or explicit r and "
            "dividend_yield -- there is no safe default for either"
        )
    out = []
    for q in quotes:
        if market_params is not None:
            rate, div = market_params.at(q.timestamp.date())
        else:
            rate, div = r, dividend_yield
        out.append(enrich_quote(q, rate, div, use_mid=use_mid,
                                prefer_solved_iv=prefer_solved_iv))
    return out


def enrichment_summary(quotes) -> dict:
    """What the enrichment pass actually managed. Read this before modelling.

    A large `greeks_failed_*` count means a big slice of the chain has no
    usable delta, which silently shrinks the candidate universe.
    """
    c: Counter = Counter()
    for q in quotes:
        for f in q.quality_flags:
            if (f.startswith("greeks_") or f == "early_exercise_risk"
                    or f == "iv_inconsistent_with_price"):
                c[f] += 1
    c["_total"] = len(quotes)
    c["_with_delta"] = sum(1 for q in quotes if not _isnan(q.delta))
    return dict(c)
