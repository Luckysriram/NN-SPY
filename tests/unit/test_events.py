from datetime import date
from pathlib import Path

import pytest

from data.events import (EVENT_FEATURE_SUFFIX, MAJOR_EVENTS, days_to_event,
                         event_in_window, jobs_report_dates, load_event_calendar)
from tests.conftest import open_calendar

CAL = {date(2024, 3, 20)}


def test_days_to_event_positive_when_future():
    assert days_to_event(date(2024, 3, 1), CAL) == 19


def test_days_to_event_zero_on_the_day():
    assert days_to_event(date(2024, 3, 20), CAL) == 0


def test_days_to_event_negative_when_past():
    """The old implementation filtered to ed >= d, so it could never go negative."""
    assert days_to_event(date(2024, 3, 25), CAL) == -5


def test_days_to_event_returns_none_for_an_empty_calendar():
    """None, not 0 -- 0 already means 'the event is today'."""
    assert days_to_event(date(2024, 3, 25), set()) is None


def test_days_to_event_picks_the_nearest_and_prefers_the_future_on_a_tie():
    assert days_to_event(date(2024, 3, 20), {date(2024, 3, 15), date(2024, 3, 25)}) == 5


def test_load_calendar_has_keys_for_all_major_events():
    assert set(seed_calendar().dates.keys()) == set(MAJOR_EVENTS)


def seed_calendar():
    """The built-in seed, independent of any events.csv on disk.

    These tests must not depend on data/raw/events/events.csv -- it is
    gitignored, so a test that reads it passes locally and behaves differently
    in CI.
    """
    return load_event_calendar(path=Path("does_not_exist_events.csv"))


def test_calendar_declares_its_coverage_window():
    cal = seed_calendar()
    assert cal.coverage_start < cal.coverage_end
    assert cal.covers(date(2024, 6, 1))
    assert not cal.covers(date(2016, 6, 1))


def test_assert_covers_refuses_a_period_the_calendar_cannot_support():
    """The guard against training on 2016-2022 with a 2024+ calendar."""
    with pytest.raises(ValueError, match="event calendar covers"):
        seed_calendar().assert_covers(date(2016, 1, 1), date(2022, 12, 31))


def test_event_feature_suffix_covers_every_major_event():
    """Every active event needs a feature name. Extra suffixes (CPI) are fine --
    they are kept so the event can be restored without touching the mapping."""
    assert set(MAJOR_EVENTS) <= set(EVENT_FEATURE_SUFFIX)


def test_jobs_reports_land_on_first_fridays():
    d = sorted(jobs_report_dates(2024, 2024))
    assert d[0] == date(2024, 1, 5)
    assert all(x.weekday() == 4 for x in d)
    assert all(x.day <= 7 for x in d)


def test_event_in_window_detects_an_event_inside_the_trade():
    cal = open_calendar(FOMC=[date(2024, 1, 31)])
    assert event_in_window(date(2024, 1, 2), date(2024, 2, 16), cal) == "event_FOMC"


def test_event_in_window_ignores_events_outside_the_trade():
    cal = open_calendar(FOMC=[date(2024, 5, 1)])
    assert event_in_window(date(2024, 1, 2), date(2024, 2, 16), cal) == ""
