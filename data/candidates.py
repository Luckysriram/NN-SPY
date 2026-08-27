"""Build put-credit-spread entry candidates from an option chain.

Every candidate is returned -- accepted and rejected alike -- so the filters are
auditable. `rejection_reason == ""` means accepted; use `accepted_only()` when
you want just the tradable ones.
"""
from __future__ import annotations

import math
from datetime import date

from data.events import event_in_window, load_event_calendar
from data.validate import validate_strike_order
from schemas import (MAX_DTE, MIN_DTE, SPREAD_WIDTH, TARGET_DELTA, Candidate,
                     OptionQuote)

DELTA_TOLERANCE = 0.07     # a "20-delta" strike 12 deltas away is not the trade


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _nz(x, default=0.0):
    return x if _finite(x) else default


def generate_candidates(quotes, decision_time, config: dict | None = None,
                        calendar=None) -> list[Candidate]:
    """One candidate per eligible expiry at `decision_time`."""
    cfg = config or {}
    slippage_per_leg = cfg.get("slippage_per_leg", 0.025)
    min_oi = cfg.get("min_oi", 100)
    min_volume = cfg.get("min_volume", 0)
    min_credit = cfg.get("min_credit", 0.20)
    max_spread_pct = cfg.get("max_spread_pct", 0.50)
    gate_on_events = cfg.get("gate_on_events", True)
    require_clean = cfg.get("require_clean_quotes", True)
    calendar = calendar if calendar is not None else load_event_calendar()

    puts = [q for q in quotes
            if q.option_type == "P"
            and MIN_DTE <= q.dte <= MAX_DTE
            and (q.is_clean or not require_clean)]

    by_expiry: dict[date, list[OptionQuote]] = {}
    for q in puts:
        by_expiry.setdefault(q.expiry, []).append(q)

    out: list[Candidate] = []
    for expiry in sorted(by_expiry):
        chain = by_expiry[expiry]
        with_delta = [q for q in chain if _finite(q.delta)]
        if not with_delta:
            continue                      # no delta anywhere: nothing to select on

        short = min(with_delta, key=lambda q: abs(q.delta - TARGET_DELTA))
        long_strike = round(short.strike - SPREAD_WIDTH, 2)
        long = next((q for q in chain if abs(q.strike - long_strike) < 1e-6), None)

        # Credit is conservative on both legs: sell at the bid, buy at the ask,
        # then pay slippage on each leg. Never assume a mid fill.
        entry_credit = (0.0 if long is None
                        else short.bid - long.ask - 2 * slippage_per_leg)

        reason = _first_rejection(
            short=short, long=long, expiry=expiry, decision_time=decision_time,
            entry_credit=entry_credit, calendar=calendar,
            min_oi=min_oi, min_volume=min_volume, min_credit=min_credit,
            max_spread_pct=max_spread_pct, gate_on_events=gate_on_events,
        )

        out.append(Candidate(
            entry_time=decision_time, short_strike=short.strike,
            long_strike=long.strike if long else long_strike,
            expiry=expiry, dte=short.dte,
            entry_credit=max(entry_credit, 0.0) if reason else entry_credit,
            width=(short.strike - long.strike) if long else SPREAD_WIDTH,
            underlying_price=_nz(short.underlying_price),
            short_delta=_nz(short.delta), short_iv=_nz(short.iv),
            short_bid=short.bid, short_ask=short.ask,
            short_open_interest=short.open_interest, short_volume=short.volume,
            short_theta=_nz(short.theta), short_vega=_nz(short.vega),
            short_gamma=_nz(short.gamma),
            long_bid=long.bid if long else 0.0, long_ask=long.ask if long else 0.0,
            long_delta=_nz(long.delta) if long else 0.0,
            long_iv=_nz(long.iv) if long else 0.0,
            long_open_interest=long.open_interest if long else 0,
            long_volume=long.volume if long else 0,
            long_theta=_nz(long.theta) if long else 0.0,
            long_vega=_nz(long.vega) if long else 0.0,
            long_gamma=_nz(long.gamma) if long else 0.0,
            rejection_reason=reason,
        ))
    return out


def _first_rejection(*, short, long, expiry, decision_time, entry_credit, calendar,
                     min_oi, min_volume, min_credit, max_spread_pct,
                     gate_on_events) -> str:
    """Return the first filter the candidate fails, or "" if it passes them all.

    Delta is checked before the long leg deliberately. Put deltas shrink as
    strikes fall, so the strike nearest a -0.20 target is often the lowest one
    quoted -- which is also the one with no strike $5 below it. Checking the long
    leg first would report "long_leg_missing" for a chain whose real problem is
    that it quotes no 20-delta put at all.
    """
    if abs(short.delta - TARGET_DELTA) > DELTA_TOLERANCE:
        return "delta_off_target"
    if long is None:
        return "long_leg_missing"
    bad_order = validate_strike_order(short.strike, long.strike)
    if bad_order:
        return bad_order
    if abs((short.strike - long.strike) - SPREAD_WIDTH) > 1e-6:
        return "width_not_5"
    if entry_credit < min_credit:
        return "credit_too_small"
    if entry_credit >= (short.strike - long.strike):
        return "credit_exceeds_width"          # impossible pricing
    if short.open_interest < min_oi or long.open_interest < min_oi:
        return "oi_too_low"
    if short.volume < min_volume or long.volume < min_volume:
        return "volume_too_low"
    for leg in (short, long):
        if leg.mid > 0 and (leg.ask - leg.bid) / leg.mid > max_spread_pct:
            return "spread_too_wide"
    if gate_on_events:
        if not calendar.covers(decision_time.date()):
            return "event_calendar_gap"        # unknown, so do not pretend it is clear
        ev = event_in_window(decision_time.date(), expiry, calendar)
        if ev:
            return ev
    return ""


def accepted_only(candidates) -> list[Candidate]:
    return [c for c in candidates if c.accepted]


def rejection_summary(candidates) -> dict:
    """Counts by reason -- the table you read when zero trades come out."""
    out: dict = {}
    for c in candidates:
        k = c.rejection_reason or "_accepted"
        out[k] = out.get(k, 0) + 1
    return out
