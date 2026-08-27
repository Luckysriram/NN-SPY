"""Risk-free rates and dividend yield -- the two inputs no option file contains.

Black-Scholes needs `r` and `q`, and an option-chain dataset gives you neither.
Guessing them is not free: over 2021-2024 the 3-month T-bill went from about
0.05% to over 5%, and at 40 DTE that swing moves a 20-delta put's delta by
roughly 2-3 delta points. Held constant across a multi-year backtest, that is a
systematic bias in which strike gets sold every single day -- always in the same
direction, so it never averages out.

Both curves follow the same contract as `data/events.py`: they declare the window
they actually cover and refuse to extrapolate outside it, because a silently
flat-extended rate is exactly the failure this module exists to prevent.

Populate from CSV:
    data/raw/rates/risk_free.csv    date,rate       (annualised decimal, e.g. 0.0525)
    data/raw/rates/dividends.csv    date,yield      (annualised decimal, e.g. 0.0132)

Free sources: FRED series DGS1MO/DGS3MO for bills, and SPY's distribution
history for the trailing yield.
"""
from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

RATES_DIR = Path("data/raw/rates")


class CurveCoverageError(ValueError):
    """Raised when a curve is asked for a date it does not cover."""


@dataclass(frozen=True)
class StepCurve:
    """A daily series looked up as a step function: last value at or before `d`.

    Step, not interpolated, on purpose. These are observed daily fixings, and
    interpolating between them invents precision the source does not have.
    """
    dates: tuple
    values: tuple
    name: str = "curve"
    source: str = "unknown"
    max_staleness_days: int = 7

    def __post_init__(self):
        if len(self.dates) != len(self.values):
            raise ValueError(f"{self.name}: {len(self.dates)} dates vs {len(self.values)} values")
        if not self.dates:
            raise ValueError(f"{self.name}: empty curve")
        if list(self.dates) != sorted(self.dates):
            raise ValueError(f"{self.name}: dates must be sorted ascending")

    @property
    def coverage_start(self) -> date:
        return self.dates[0]

    @property
    def coverage_end(self) -> date:
        return self.dates[-1]

    def covers(self, d: date) -> bool:
        return self.coverage_start <= d <= self.coverage_end

    def assert_covers(self, start: date, end: date) -> None:
        if start < self.coverage_start or end > self.coverage_end:
            raise CurveCoverageError(
                f"{self.name} covers {self.coverage_start}..{self.coverage_end} "
                f"(source={self.source}) but was asked for {start}..{end}. "
                f"Extend {RATES_DIR}/ before using this period."
            )

    def at(self, d: date) -> float:
        """Value on `d`. Raises rather than extrapolating past either end."""
        if d < self.coverage_start:
            raise CurveCoverageError(
                f"{self.name}: {d} precedes coverage start {self.coverage_start}")
        if d > self.coverage_end:
            raise CurveCoverageError(
                f"{self.name}: {d} is past coverage end {self.coverage_end}")
        i = bisect.bisect_right(self.dates, d) - 1
        observed = self.dates[i]
        gap = (d - observed).days
        if gap > self.max_staleness_days:
            raise CurveCoverageError(
                f"{self.name}: nearest observation to {d} is {observed}, "
                f"{gap} days stale (limit {self.max_staleness_days}). "
                "The series has a hole here."
            )
        return self.values[i]

    @classmethod
    def from_csv(cls, path, value_column: str, name: str, **kw) -> "StepCurve":
        p = Path(path)
        rows = []
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = (row.get(value_column) or "").strip()
                if not raw or raw in (".", "NA", "null"):
                    continue                      # FRED writes "." on holidays
                rows.append((date.fromisoformat(row["date"].strip()), float(raw)))
        if not rows:
            raise ValueError(f"{p} contained no usable rows")
        rows.sort()
        return cls(tuple(d for d, _ in rows), tuple(v for _, v in rows),
                   name=name, source=str(p), **kw)

    @classmethod
    def constant(cls, value: float, start: date, end: date, name: str) -> "StepCurve":
        """A flat curve. Legitimate ONLY for tests and short windows.

        Named explicitly so a flat rate never enters a multi-year backtest by
        accident -- `source` says so in any error message it produces.
        """
        if end < start:
            raise ValueError(f"{name}: end {end} precedes start {start}")
        # Store BOTH endpoints. A single point would make coverage_end == start,
        # so every lookup past day one would raise "past coverage end".
        span = (end - start).days + 1
        return cls((start, end), (value, value), name=name,
                   source=f"CONSTANT {value:.4%} (approximation)",
                   max_staleness_days=span)


def load_risk_free(path=None) -> StepCurve:
    p = Path(path) if path else RATES_DIR / "risk_free.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"no risk-free curve at {p}. Download FRED DGS3MO as date,rate "
            "(annualised decimal), or pass StepCurve.constant(...) if you "
            "accept a flat rate and its bias."
        )
    return StepCurve.from_csv(p, "rate", "risk_free")


def load_dividend_yield(path=None) -> StepCurve:
    p = Path(path) if path else RATES_DIR / "dividends.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"no dividend curve at {p}. SPY yields roughly 1-2%; q=0 biases put "
            "deltas by several points."
        )
    return StepCurve.from_csv(p, "yield", "dividend_yield", max_staleness_days=95)


@dataclass(frozen=True)
class MarketParams:
    """The (r, q) pair for one decision date."""
    risk_free: StepCurve
    dividend_yield: StepCurve

    def at(self, d: date) -> tuple:
        return self.risk_free.at(d), self.dividend_yield.at(d)

    def assert_covers(self, start: date, end: date) -> None:
        self.risk_free.assert_covers(start, end)
        self.dividend_yield.assert_covers(start, end)

    @classmethod
    def constant(cls, r: float, q: float, start: date, end: date) -> "MarketParams":
        """Flat r and q. For tests, or a window short enough that rates did not move."""
        return cls(StepCurve.constant(r, start, end, "risk_free"),
                   StepCurve.constant(q, start, end, "dividend_yield"))
