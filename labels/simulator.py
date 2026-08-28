"""Walk each candidate forward to its exit and produce the training label.

This module decides what `label_win` means, so every modelling result downstream
inherits its assumptions. Three of them are deliberate and conservative:

1. ORDER OF CHECKS. When a day's range could have touched both the stop and the
   profit target, we assume the STOP hit first. Daily marks cannot resolve
   intraday sequence, and assuming the favourable one manufactures free wins.
2. FILLS. Close cost is short ASK minus long BID, plus slippage. Entry credit
   (set in candidates.py) is short BID minus long ASK, minus slippage. Nothing
   fills at the mid.
3. MISSING DATA IS NOT A WIN. A candidate with no usable marks returns NO_DATA
   and is excluded from training, rather than silently scoring a full-credit win.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from schemas import (PROFIT_TARGET_FRAC, STOP_LOSS_FRAC, TIME_EXIT_DTE,
                     Candidate, TradeOutcome)


@dataclass(frozen=True)
class SpreadSnapshot:
    """One later observation of the spread we are short."""
    time: datetime
    short_ask: float      # cost to buy back the short put
    long_bid: float       # proceeds from selling the long put
    underlying_price: float = float("nan")

    @property
    def usable(self) -> bool:
        return not (math.isnan(self.short_ask) or math.isnan(self.long_bid))


def close_cost(short_ask: float, long_bid: float) -> float:
    """Debit to close the spread: buy back the short, sell the long."""
    return short_ask - long_bid


def _bounded_close_cost(snap, candidate) -> tuple:
    """Close cost clamped to [0, width]. Returns (cost, was_clamped).

    A vertical spread cannot cost more than its width to close, nor less than
    nothing -- both are arbitrage. Real end-of-day chains still violate it: a
    stale or crossed quote on one leg produced a $7.21 close cost on a $5-wide
    spread in this dataset, which the simulator happily booked as a -150% loss
    on a DEFINED-RISK position.

    Clamping to the width is not a cosmetic fix, it is the correct worst case:
    max loss on a credit spread is exactly width - credit. Violations are
    counted so a dataset full of them cannot pass unnoticed.
    """
    raw = close_cost(snap.short_ask, snap.long_bid)
    bounded = min(max(raw, 0.0), candidate.width)
    return bounded, int(bounded != raw)


def intrinsic_at_expiry(candidate: Candidate, underlying: float) -> float:
    """Settlement cost of the spread if held to expiry."""
    if math.isnan(underlying):
        return candidate.width          # assume max loss when we cannot tell
    short_itm = max(candidate.short_strike - underlying, 0.0)
    long_itm = max(candidate.long_strike - underlying, 0.0)
    return min(max(short_itm - long_itm, 0.0), candidate.width)


def simulate_trade(candidate: Candidate, snapshots, *,
                   fees_per_spread: float = 0.0,
                   slippage: float = 0.0) -> TradeOutcome:
    """Walk `snapshots` forward until an exit rule fires.

    `fees_per_spread` is the TOTAL round-trip commission (open + close).
    `slippage` is added to each close cost.
    """
    credit = candidate.entry_credit
    target = credit * PROFIT_TARGET_FRAC
    stop = credit * STOP_LOSS_FRAC

    marks = sorted((s for s in snapshots
                    if s.usable and s.time > candidate.entry_time),
                   key=lambda s: s.time)
    if not marks:
        return _no_data(candidate, fees_per_spread, slippage)

    mae = mfe = 0.0
    exit_reason, exit_time, raw_cost, seen = "", marks[-1].time, float("nan"), 0
    n_clamped = 0

    for seen, snap in enumerate(marks, start=1):
        raw_cost, clamped = _bounded_close_cost(snap, candidate)
        n_clamped += clamped
        cost = raw_cost + slippage
        mae = max(mae, cost - credit)        # worst loss seen, >= 0
        mfe = max(mfe, credit - cost)        # best gain seen, >= 0
        exit_time = snap.time
        dte_now = (candidate.expiry - snap.time.date()).days

        if cost >= stop:                     # (1) adverse case checked first
            exit_reason = "STOP_LOSS"
        elif cost <= target:
            exit_reason = "PROFIT_TARGET"
        elif snap.time.date() >= candidate.expiry:
            exit_reason = "EXPIRY"
            raw_cost = intrinsic_at_expiry(candidate, snap.underlying_price)
        elif dte_now <= TIME_EXIT_DTE:
            exit_reason = "TIME_EXIT"
        if exit_reason:
            break

    if not exit_reason:
        # Ran out of marks before any rule fired: the data ends mid-trade.
        return _no_data(candidate, fees_per_spread, slippage,
                        exit_time=exit_time, n_marks=len(marks))

    exit_debit = raw_cost + slippage
    gross_pnl = credit - raw_cost
    net_pnl = gross_pnl - slippage - fees_per_spread
    return TradeOutcome(
        candidate=candidate, exit_time=exit_time, exit_reason=exit_reason,
        exit_debit=exit_debit, label_win=(exit_reason == "PROFIT_TARGET"),
        gross_pnl=gross_pnl, fees=fees_per_spread, slippage=slippage,
        net_pnl=net_pnl,
        # Return on the capital actually at risk (width - credit), not the width.
        final_return_on_risk=net_pnl / candidate.max_risk,
        max_adverse_excursion=mae, max_favorable_excursion=mfe,
        days_held=(exit_time - candidate.entry_time).total_seconds() / 86400.0,
        n_marks=seen, n_clamped_marks=n_clamped,
    )


def _no_data(candidate, fees, slippage, exit_time=None, n_marks=0) -> TradeOutcome:
    """An unlabelable trade. Excluded from training by `is_labelable`."""
    nan = float("nan")
    return TradeOutcome(
        candidate=candidate, exit_time=exit_time or candidate.entry_time,
        exit_reason="NO_DATA", exit_debit=nan, label_win=False,
        gross_pnl=nan, fees=fees, slippage=slippage,
        net_pnl=nan, final_return_on_risk=nan,
        max_adverse_excursion=nan, max_favorable_excursion=nan,
        days_held=0.0, n_marks=n_marks,
    )


def simulate_all(candidates, snapshots_by_candidate, **kw) -> list[TradeOutcome]:
    """Simulate a list of candidates against a {index: snapshots} mapping."""
    return [simulate_trade(c, snapshots_by_candidate.get(i, []), **kw)
            for i, c in enumerate(candidates)]


def snapshots_from_quotes(candidate: Candidate, quotes) -> list[SpreadSnapshot]:
    """Build the forward mark series for one candidate from a raw quote stream.

    Pairs the short and long leg by (expiry, strike) at each later timestamp.
    A timestamp missing either leg is skipped rather than half-filled.
    """
    legs: dict = {}
    for q in quotes:
        if q.option_type != "P" or q.expiry != candidate.expiry:
            continue
        if q.timestamp <= candidate.entry_time:
            continue
        if abs(q.strike - candidate.short_strike) < 1e-6:
            legs.setdefault(q.timestamp, {})["short"] = q
        elif abs(q.strike - candidate.long_strike) < 1e-6:
            legs.setdefault(q.timestamp, {})["long"] = q

    out = []
    for ts in sorted(legs):
        pair = legs[ts]
        if "short" in pair and "long" in pair:
            out.append(SpreadSnapshot(
                time=ts, short_ask=pair["short"].ask, long_bid=pair["long"].bid,
                underlying_price=pair["short"].underlying_price))
    return out
