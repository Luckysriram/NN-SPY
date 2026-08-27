"""Answer, before any modelling: can this dataset support the strategy at all?

Run this FIRST on any candidate dataset. It reports what is actually present --
date range, DTE coverage, whether Greeks exist, whether 20-delta puts are even
quoted -- so the strategy constants get chosen from the data instead of guessed
and discovered wrong ten tasks later.

    python -m data.adapters.feasibility_check <path.csv|path.parquet>
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from data.adapters.sample import normalize_rows
from data.validate import flag_summary, validate_quotes
from schemas import MAX_DTE, MIN_DTE, SPREAD_WIDTH, TARGET_DELTA


@dataclass
class FeasibilityReport:
    n_rows: int = 0
    n_quotes: int = 0
    reject_reasons: dict = field(default_factory=dict)
    date_min: object = None
    date_max: object = None
    n_trading_days: int = 0
    has_greeks: bool = False
    has_iv: bool = False
    dte_in_range_pct: float = 0.0
    days_with_target_delta: int = 0
    days_with_full_spread: int = 0
    flag_counts: dict = field(default_factory=dict)
    blockers: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.blockers

    def render(self) -> str:
        L = [
            "=" * 62,
            "  DATASET FEASIBILITY REPORT",
            "=" * 62,
            f"  rows read              : {self.n_rows:,}",
            f"  quotes normalized      : {self.n_quotes:,}",
            f"  rejected rows          : {sum(self.reject_reasons.values()):,} {self.reject_reasons or ''}",
            f"  date range             : {self.date_min} .. {self.date_max}",
            f"  distinct trading days  : {self.n_trading_days:,}",
            f"  implied vol present    : {'yes' if self.has_iv else 'NO'}",
            f"  greeks present         : {'yes' if self.has_greeks else 'NO'}",
            f"  quotes in {MIN_DTE}-{MAX_DTE} DTE    : {self.dte_in_range_pct:.1%}",
            f"  days w/ {TARGET_DELTA:+.2f}-delta put : {self.days_with_target_delta:,}",
            f"  days w/ full ${SPREAD_WIDTH:.0f} spread : {self.days_with_full_spread:,}",
            "-" * 62,
            "  quality flags:",
        ]
        for k, v in sorted(self.flag_counts.items()):
            if not k.startswith("_"):
                L.append(f"    {k:<26} {v:,}")
        L.append("-" * 62)
        if self.blockers:
            L.append("  NOT USABLE AS-IS:")
            L += [f"    - {b}" for b in self.blockers]
        else:
            L.append("  USABLE. Set configs/data.yaml start/end to the range above.")
        L.append("=" * 62)
        return "\n".join(L)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def assess(rows, symbol: str = "SPY") -> FeasibilityReport:
    """Build a feasibility report from raw source rows."""
    rows = list(rows)
    res = normalize_rows(rows, symbol=symbol)
    quotes = validate_quotes(res.quotes)
    rep = FeasibilityReport(
        n_rows=len(rows), n_quotes=len(quotes), reject_reasons=res.reject_reasons,
        flag_counts=flag_summary(quotes),
    )
    if not quotes:
        rep.blockers.append("no rows normalized into canonical quotes")
        return rep

    days = sorted({q.timestamp.date() for q in quotes})
    rep.date_min, rep.date_max, rep.n_trading_days = days[0], days[-1], len(days)
    rep.has_iv = any(_finite(q.iv) for q in quotes)
    rep.has_greeks = any(_finite(q.delta) for q in quotes)
    rep.dte_in_range_pct = sum(MIN_DTE <= q.dte <= MAX_DTE for q in quotes) / len(quotes)

    puts_by_day: dict = {}
    for q in quotes:
        if q.option_type == "P" and MIN_DTE <= q.dte <= MAX_DTE:
            puts_by_day.setdefault(q.timestamp.date(), []).append(q)

    for day, chain in puts_by_day.items():
        deltas = [q for q in chain if _finite(q.delta)]
        if not deltas:
            continue
        short = min(deltas, key=lambda q: abs(q.delta - TARGET_DELTA))
        if abs(short.delta - TARGET_DELTA) > 0.07:
            continue
        rep.days_with_target_delta += 1
        strikes = {q.strike for q in chain if q.expiry == short.expiry}
        if round(short.strike - SPREAD_WIDTH, 2) in strikes:
            rep.days_with_full_spread += 1

    # Blockers: things that make the strategy as specified impossible.
    if not rep.has_greeks:
        rep.blockers.append(
            "no delta field -- cannot select a 20-delta short strike. Either pick "
            "a dataset with greeks or compute them (needs a rate + dividend curve)."
        )
    if not rep.has_iv:
        rep.blockers.append("no implied vol -- short_put_iv feature unavailable")
    if rep.dte_in_range_pct == 0:
        rep.blockers.append(f"no quotes in the {MIN_DTE}-{MAX_DTE} DTE window")
    if rep.n_trading_days < 250:
        rep.blockers.append(
            f"only {rep.n_trading_days} trading days -- too few for chronological "
            "walk-forward splits (need multiple years)"
        )
    if rep.days_with_full_spread < max(1, 0.5 * rep.days_with_target_delta):
        rep.blockers.append(
            f"only {rep.days_with_full_spread} of {rep.days_with_target_delta} days "
            f"have both legs of a ${SPREAD_WIDTH:.0f} spread -- strike grid too coarse"
        )
    return rep


def _read_any(path: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return df.to_dict("records")


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"no such file: {path}")
        return 2
    rep = assess(_read_any(path))
    print(rep.render())
    return 0 if rep.usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
