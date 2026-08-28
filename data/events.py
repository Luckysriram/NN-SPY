"""Macro event calendar (FOMC / CPI / jobs report).

Design note -- the reason this module carries an explicit coverage window:

A calendar that silently returns 0 for dates it does not know about produces
train/serve skew by construction. If the calendar only holds 2024+ dates but the
model trains on 2016-2022, then `days_to_fomc` is a constant 0 for every training
row and a real number at serving time. The model learns nothing from the feature
in training and is then handed a live value it has never seen.

So: this module reports what it knows (`coverage`), returns None outside it, and
`features` turns that None into NaN. `assert_covers()` lets the training pipeline
refuse to run on a period the calendar cannot support.

Real dates come from a CSV (`data/raw/events/events.csv`, columns: date,event).
The built-in seed below is deliberately small and honestly scoped -- extend it
from the published Fed / BLS calendars rather than trusting these few rows.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# CPI is deliberately ABSENT. BLS (bls.gov) and FRED both refuse programmatic
# access (HTTP 403), so historical CPI release dates could not be sourced for
# 2015-2023. Inventing them -- or approximating "the second week" -- would put a
# wrong feature into every training row, which is worse than having one fewer
# feature. Add "CPI" back here, add its suffix below, and drop the dates into
# data/raw/events/events.csv if you obtain a real schedule.
MAJOR_EVENTS = ("FOMC", "JOBS")

# Feature-name suffix per event. Kept beside MAJOR_EVENTS so the feature columns
# and the calendar can never drift apart (the old bug: calendar emitted
# "days_to_jobs" while the feature list wanted "days_to_jobs_report").
EVENT_FEATURE_SUFFIX = {"FOMC": "fomc", "CPI": "cpi", "JOBS": "jobs_report"}
# (suffixes for events not in MAJOR_EVENTS are kept so CPI can be restored)

_SEED_FOMC = [
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
]
_SEED_COVERAGE = (date(2024, 1, 1), date(2025, 12, 31))


@dataclass(frozen=True)
class EventCalendar:
    """Event dates plus the window over which those dates are actually complete."""
    dates: dict          # event name -> frozenset[date]
    coverage_start: date
    coverage_end: date
    source: str = "seed"

    def covers(self, d: date) -> bool:
        return self.coverage_start <= d <= self.coverage_end

    def assert_covers(self, start: date, end: date) -> None:
        if start < self.coverage_start or end > self.coverage_end:
            raise ValueError(
                f"event calendar covers {self.coverage_start}..{self.coverage_end} "
                f"(source={self.source}) but was asked for {start}..{end}. "
                "Load a fuller calendar into data/raw/events/events.csv before "
                "training on this period -- see data/events.py."
            )

    def for_event(self, event: str) -> frozenset:
        return self.dates.get(event, frozenset())


def _first_friday(year: int, month: int) -> date:
    """BLS releases the employment situation on the first Friday of the month."""
    d = date(year, month, 1)
    return d.replace(day=1 + (4 - d.weekday()) % 7)


def jobs_report_dates(start_year: int, end_year: int) -> frozenset:
    """Rule-derived jobs-report dates. The rule is real but has rare exceptions
    (holiday shifts); replace with the published BLS calendar for live use."""
    return frozenset(
        _first_friday(y, m)
        for y in range(start_year, end_year + 1)
        for m in range(1, 13)
    )


def load_event_calendar(path=None) -> EventCalendar:
    """Load from CSV when available, else return the honestly-scoped seed."""
    p = Path(path) if path else Path("data/raw/events/events.csv")
    if p.exists():
        rows: dict[str, set] = {e: set() for e in MAJOR_EVENTS}
        seen: list[date] = []
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ev = row["event"].strip().upper()
                d = date.fromisoformat(row["date"].strip())
                rows.setdefault(ev, set()).add(d)
                seen.append(d)
        if not seen:
            raise ValueError(f"{p} contains no event rows")
        return EventCalendar(
            dates={k: frozenset(v) for k, v in rows.items()},
            coverage_start=min(seen), coverage_end=max(seen), source=str(p),
        )
    lo, hi = _SEED_COVERAGE
    return EventCalendar(
        dates={
            "FOMC": frozenset(_SEED_FOMC),
            "JOBS": jobs_report_dates(lo.year, hi.year),
        },
        coverage_start=lo, coverage_end=hi, source="seed",
    )


def days_to_event(d: date, calendar) -> int | None:
    """Signed days from `d` to the nearest event date.

    Positive when the event is ahead, 0 on the day, negative when it has passed.
    Returns None for an empty calendar -- callers turn that into NaN rather than
    a 0 that would be indistinguishable from "the event is today".
    """
    dates = list(calendar)
    if not dates:
        return None
    # Ties (an event equally far behind and ahead) resolve to the future one.
    return min(((ed - d).days for ed in dates), key=lambda n: (abs(n), -n))


def event_in_window(entry: date, expiry: date, calendar: EventCalendar) -> str:
    """Name the first major event landing inside (entry, expiry], else ""."""
    for event in MAJOR_EVENTS:
        for ed in sorted(calendar.for_event(event)):
            if entry < ed <= expiry:
                return f"event_{event}"
    return ""
