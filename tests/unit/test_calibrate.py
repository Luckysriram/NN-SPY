import numpy as np
import pytest

from models.calibrate import Calibrator, CalibratorFrozenError, calibrate_proba
from models.select_threshold import expected_value


def synthetic(n=400, seed=0):
    """Overconfident probabilities: the model says 0.9, reality is nearer 0.6."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.55, 0.95, n)
    y = (rng.uniform(size=n) < (p - 0.25)).astype(float)
    return p, y


def test_calibrate_proba_returns_target_shaped_probabilities():
    p, y = synthetic()
    out = calibrate_proba(p[:300], y[:300], p[300:])
    assert out.shape == (100,)
    assert ((out >= 0) & (out <= 1)).all()


def test_calibration_reduces_expected_calibration_error():
    from reports.metrics import expected_calibration_error
    p, y = synthetic(600)
    val_p, val_y = p[:400], y[:400]
    test_p, test_y = p[400:], y[400:]
    before = expected_calibration_error(test_y, test_p)
    after = expected_calibration_error(test_y, calibrate_proba(val_p, val_y, test_p))
    assert after < before


def test_calibrator_refuses_to_refit_once_frozen():
    """Refitting on the test period is exactly the leak this guards."""
    p, y = synthetic()
    c = Calibrator().fit(p[:200], y[:200]).freeze()
    with pytest.raises(CalibratorFrozenError, match="already frozen"):
        c.fit(p[200:], y[200:])


def test_transform_before_fit_raises():
    with pytest.raises(CalibratorFrozenError, match="must be fit"):
        Calibrator().transform([0.5])


def test_freeze_before_fit_raises():
    with pytest.raises(CalibratorFrozenError, match="unfit"):
        Calibrator().freeze()


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        Calibrator().fit([0.1, 0.2, 0.3], [0, 1])


def test_single_class_validation_falls_back_to_identity():
    c = Calibrator().fit([0.2, 0.6, 0.9], [1, 1, 1])
    assert np.allclose(c.transform([0.2, 0.6, 0.9]), [0.2, 0.6, 0.9])


def test_platt_method_also_works():
    p, y = synthetic()
    out = Calibrator("platt").fit(p[:300], y[:300]).transform(p[300:])
    assert ((out >= 0) & (out <= 1)).all()


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="unknown calibration method"):
        Calibrator("magic")


def test_expected_value_formula():
    assert expected_value(0.6, 10.0, 5.0, costs=1.0) == pytest.approx(0.6 * 10 - 0.4 * 5 - 1.0)
