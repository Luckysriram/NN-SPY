import tempfile
from datetime import date
from pathlib import Path

import pytest

from data.rates import (CurveCoverageError, MarketParams, StepCurve,
                        load_dividend_yield, load_risk_free)

DATES = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 8))
VALUES = (0.0530, 0.0532, 0.0525)


def curve(**kw) -> StepCurve:
    base = dict(dates=DATES, values=VALUES, name="risk_free", source="test")
    base.update(kw)
    return StepCurve(**base)


def test_lookup_returns_the_observation_on_the_date():
    assert curve().at(date(2024, 1, 3)) == 0.0532


def test_lookup_steps_forward_from_the_last_observation():
    """Step, not interpolated -- these are daily fixings, not a smooth curve."""
    assert curve().at(date(2024, 1, 5)) == 0.0532


def test_lookup_refuses_to_extrapolate_before_coverage():
    with pytest.raises(CurveCoverageError, match="precedes coverage start"):
        curve().at(date(2023, 12, 31))


def test_lookup_refuses_to_extrapolate_past_coverage():
    """A silently flat-extended rate is the failure this module exists to stop."""
    with pytest.raises(CurveCoverageError, match="past coverage end"):
        curve().at(date(2024, 6, 1))


def test_lookup_refuses_a_stale_observation():
    c = StepCurve(dates=(date(2024, 1, 2), date(2024, 3, 1)),
                  values=(0.05, 0.05), name="risk_free", max_staleness_days=7)
    with pytest.raises(CurveCoverageError, match="days stale"):
        c.at(date(2024, 2, 20))


def test_assert_covers_guards_a_whole_period():
    c = curve()
    c.assert_covers(date(2024, 1, 2), date(2024, 1, 8))
    with pytest.raises(CurveCoverageError, match="covers"):
        c.assert_covers(date(2020, 1, 1), date(2024, 1, 8))


def test_unsorted_dates_are_rejected():
    with pytest.raises(ValueError, match="sorted ascending"):
        StepCurve(dates=(date(2024, 1, 3), date(2024, 1, 2)), values=(0.1, 0.2))


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="dates vs"):
        StepCurve(dates=DATES, values=(0.1, 0.2))


def test_empty_curve_is_rejected():
    with pytest.raises(ValueError, match="empty curve"):
        StepCurve(dates=(), values=())


def test_from_csv_reads_and_sorts():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.csv"
        p.write_text("date,rate\n2024-01-08,0.0525\n2024-01-02,0.0530\n", encoding="utf-8")
        c = StepCurve.from_csv(p, "rate", "risk_free")
    assert c.coverage_start == date(2024, 1, 2)
    assert c.at(date(2024, 1, 2)) == 0.0530


def test_from_csv_skips_fred_holiday_markers():
    """FRED writes '.' for non-trading days."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.csv"
        p.write_text("date,rate\n2024-01-01,.\n2024-01-02,0.0530\n", encoding="utf-8")
        c = StepCurve.from_csv(p, "rate", "risk_free")
    assert len(c.dates) == 1


def test_from_csv_rejects_a_file_with_no_usable_rows():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.csv"
        p.write_text("date,rate\n2024-01-01,.\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no usable rows"):
            StepCurve.from_csv(p, "rate", "risk_free")


def test_constant_curve_names_itself_an_approximation():
    """So a flat rate can never enter a multi-year backtest unnoticed."""
    c = StepCurve.constant(0.05, date(2024, 1, 1), date(2024, 3, 1), "risk_free")
    assert "CONSTANT" in c.source and "approximation" in c.source
    assert c.at(date(2024, 2, 1)) == 0.05


def test_missing_files_raise_with_a_pointer_to_the_source():
    with pytest.raises(FileNotFoundError, match="FRED"):
        load_risk_free(Path("does_not_exist.csv"))
    with pytest.raises(FileNotFoundError, match="1-2%"):
        load_dividend_yield(Path("does_not_exist.csv"))


def test_market_params_returns_both_rates():
    mp = MarketParams.constant(0.045, 0.013, date(2024, 1, 1), date(2024, 12, 31))
    r, q = mp.at(date(2024, 6, 1))
    assert (r, q) == (0.045, 0.013)


def test_market_params_coverage_guard_covers_both_curves():
    mp = MarketParams.constant(0.045, 0.013, date(2024, 1, 1), date(2024, 12, 31))
    mp.assert_covers(date(2024, 1, 1), date(2024, 12, 31))
    with pytest.raises(CurveCoverageError):
        mp.assert_covers(date(2020, 1, 1), date(2024, 12, 31))
