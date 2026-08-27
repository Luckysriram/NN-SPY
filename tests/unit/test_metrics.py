import math

import numpy as np
import pytest

from reports.metrics import (accuracy, base_rate, brier_score, calibration_bins,
                             classification_report, expected_calibration_error,
                             max_drawdown, profit_factor, pr_auc, roc_auc,
                             sharpe, trading_report, win_rate)


def test_brier_score_perfect():
    assert brier_score([0, 1, 1], [0.0, 1.0, 1.0]) < 1e-9


def test_roc_auc_perfect_is_one():
    assert roc_auc([0, 1, 1, 0], [0.1, 0.8, 0.9, 0.2]) == 1.0


def test_roc_auc_is_nan_with_a_single_class_not_a_misleading_half():
    assert math.isnan(roc_auc([1, 1, 1], [0.2, 0.5, 0.9]))
    assert math.isnan(pr_auc([0, 0], [0.2, 0.5]))


def test_calibration_bins_single_bin():
    bins = calibration_bins([0, 1], [0.0, 1.0], bins=1)
    assert bins[0][1] == 0.5 and bins[0][2] == 0.5
    assert bins[0][3] == 2                       # count is reported too


def test_calibration_bins_skips_empty_bins():
    assert len(calibration_bins([0, 1], [0.05, 0.95], bins=10)) == 2


def test_expected_calibration_error_is_zero_for_a_calibrated_model():
    y = [0] * 50 + [1] * 50
    p = [0.5] * 100
    assert expected_calibration_error(y, p, bins=10) == pytest.approx(0.0)


def test_expected_calibration_error_catches_overconfidence():
    """The spec's failure case: says 0.9, wins half the time."""
    y = [1] * 50 + [0] * 50
    p = [0.9] * 100
    assert expected_calibration_error(y, p, bins=10) == pytest.approx(0.4, abs=0.01)


def test_base_rate_is_the_number_accuracy_must_beat():
    y = [1] * 85 + [0] * 15
    assert base_rate(y) == pytest.approx(0.85)
    always_win = [0.99] * 100
    assert accuracy(y, always_win) == pytest.approx(0.85)


def test_profit_factor_positive():
    assert profit_factor([10.0, -4.0, 6.0]) == pytest.approx(16.0 / 4.0)


def test_profit_factor_with_no_losses_is_inf():
    assert profit_factor([1.0, 2.0]) == float("inf")


def test_metrics_ignore_nan_entries():
    assert profit_factor([10.0, -4.0, float("nan")]) == pytest.approx(2.5)
    assert win_rate([1.0, -1.0, float("nan")]) == pytest.approx(0.5)


def test_max_drawdown_is_negative_and_measures_peak_to_trough():
    assert max_drawdown([10.0, -5.0, -5.0, 20.0]) == pytest.approx(-10.0)
    assert max_drawdown([1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_sharpe_is_nan_without_variation():
    assert math.isnan(sharpe([1.0, 1.0, 1.0]))


def test_classification_report_shape():
    rep = classification_report([0, 1, 1, 0], [0.1, 0.8, 0.9, 0.2])
    assert set(rep) == {"n", "base_rate", "roc_auc", "pr_auc", "brier", "ece", "accuracy"}
    assert rep["n"] == 4


def test_trading_report_shape():
    rep = trading_report([10.0, -4.0, 6.0])
    assert rep["n_trades"] == 3
    assert rep["total_pnl"] == pytest.approx(12.0)
    assert rep["win_rate"] == pytest.approx(2 / 3)


def test_trading_report_on_no_trades():
    rep = trading_report([])
    assert rep["n_trades"] == 0 and rep["total_pnl"] == 0.0
