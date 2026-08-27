import json
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from reports.daily_report import append_decision_log, daily_signal_json
from reports.monthly_report import (by_group, by_prob_bucket, dte_bucket,
                                    monthly_report, render, vix_bucket)
from tests.conftest import candidate

SIGNAL_CAND = candidate(
    entry_time=datetime(2026, 8, 26, 15, 45), short_strike=640.0, long_strike=635.0,
    expiry=date(2026, 9, 25), dte=30, entry_credit=1.55, underlying_price=650.0,
)


def test_daily_signal_json_matches_spec_shape():
    js = daily_signal_json(SIGNAL_CAND, proba=0.71, expected_value=12.40,
                           decision="PAPER_TRADE",
                           reasons=["passes probability threshold"])
    assert js["strategy"] == "SPY put credit spread"
    assert js["short_strike"] == 640
    assert js["model_probability"] == 0.71
    assert js["decision"] == "PAPER_TRADE"
    assert js["reasons"] == ["passes probability threshold"]
    assert js["expiration"] == "2026-09-25"


def test_daily_signal_records_risk_and_a_disclaimer():
    js = daily_signal_json(SIGNAL_CAND, 0.71, 12.40, "PAPER_TRADE", [])
    assert js["max_risk"] == pytest.approx(3.45)      # width 5 - credit 1.55
    assert "Paper trading only" in js["disclaimer"]


def test_fractional_strikes_are_not_truncated_to_int():
    js = daily_signal_json(candidate(short_strike=640.5, long_strike=635.5),
                           0.5, 1.0, "NO_TRADE", [])
    assert js["short_strike"] == 640.5


def test_no_trade_decisions_are_recorded_too():
    js = daily_signal_json(SIGNAL_CAND, 0.30, -2.0, "NO_TRADE", ["below_threshold"])
    assert js["decision"] == "NO_TRADE"
    assert js["reasons"] == ["below_threshold"]


def test_invalid_decision_is_rejected():
    with pytest.raises(ValueError, match="PAPER_TRADE or NO_TRADE"):
        daily_signal_json(SIGNAL_CAND, 0.5, 1.0, "BUY", [])


def test_decision_log_appends_json_lines():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "decisions.jsonl"
        for dec in ("PAPER_TRADE", "NO_TRADE"):
            append_decision_log(daily_signal_json(SIGNAL_CAND, 0.6, 1.0, dec, []), p)
        lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["decision"] == "NO_TRADE"


def test_monthly_report_includes_calibration_buckets():
    rep = monthly_report([None] * 4, proba=[0.2, 0.8, 0.2, 0.8], y=[0, 1, 0, 1])
    assert rep["trades"] == 4
    assert any(b[0] > 0.7 for b in rep["calibration"])


def test_by_prob_bucket_is_actually_produced():
    """The old interface promised by_prob_bucket and returned only calibration."""
    rep = monthly_report([None] * 4, proba=[0.2, 0.8, 0.2, 0.8], y=[0, 1, 0, 1])
    assert "by_prob_bucket" in rep and rep["by_prob_bucket"]


def test_by_prob_bucket_names_the_calibration_gap():
    """Spec's failure case: the 0.70 bucket wins 52% of the time."""
    p = [0.72] * 100
    y = [1] * 52 + [0] * 48
    rows = [r for r in by_prob_bucket(p, y) if r["n"] == 100]
    assert rows[0]["calibration_gap"] == pytest.approx(0.52 - 0.72, abs=1e-6)


def test_by_prob_bucket_totals_pnl_when_given():
    rows = by_prob_bucket([0.8, 0.8], [1, 0], pnl=[100.0, -200.0])
    assert rows[0]["total_pnl"] == pytest.approx(-100.0)


def test_by_group_breaks_trades_down():
    rows = by_group(["low_vix", "high_vix", "low_vix"], [10.0, -5.0, 20.0])
    assert {r["group"] for r in rows} == {"low_vix", "high_vix"}
    low = next(r for r in rows if r["group"] == "low_vix")
    assert low["n"] == 2 and low["total_pnl"] == pytest.approx(30.0)


def test_regime_bucket_helpers():
    assert vix_bucket(12.0) == "low_vix"
    assert vix_bucket(20.0) == "med_vix"
    assert vix_bucket(31.0) == "high_vix"
    assert vix_bucket(float("nan")) == "unknown"
    assert dte_bucket(31) == "30-35" and dte_bucket(44) == "41-45"


def test_monthly_report_with_breakdowns_and_pnl():
    rng = np.random.default_rng(0)
    n = 60
    p = rng.uniform(0.4, 0.9, n)
    y = (rng.uniform(size=n) < p).astype(float)
    pnl = np.where(y > 0, 100.0, -200.0)
    rep = monthly_report([None] * n, p, y, pnl=pnl,
                         groups={"vix": ["low_vix"] * 30 + ["high_vix"] * 30})
    assert rep["trading"]["n_trades"] == n
    assert len(rep["breakdowns"]["vix"]) == 2
    assert "ece" in rep


def test_render_produces_readable_text():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.4, 0.9, 40)
    y = (rng.uniform(size=40) < p).astype(float)
    text = render(monthly_report([None] * 40, p, y, pnl=np.where(y > 0, 100.0, -200.0)))
    assert "CALIBRATION AUDIT" in text
    assert "trades: 40" in text
