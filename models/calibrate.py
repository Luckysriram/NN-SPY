"""Probability calibration, fit on VALIDATION and then frozen.

Spec section 9.4 is explicit: calibrate on validation only. Fitting the
calibrator on training probabilities calibrates against data the model has
already overfit, so the reliability curve looks excellent in-sample and the
70%-bucket still wins 52% out of sample -- exactly the failure the spec's
calibration audit is designed to catch.

`Calibrator` therefore refuses to fit twice.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratorFrozenError(RuntimeError):
    pass


class Calibrator:
    """Isotonic (default) or Platt calibration, fit once on validation."""

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError(f"unknown calibration method: {method!r}")
        self.method = method
        self._model = None
        self._frozen = False

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def fit(self, proba_val, y_val) -> "Calibrator":
        """Fit on VALIDATION predictions and labels. Never on training."""
        if self._frozen:
            raise CalibratorFrozenError(
                "calibrator already frozen -- refit would leak the test period"
            )
        p = np.asarray(proba_val, dtype=float).ravel()
        y = np.asarray(y_val, dtype=float).ravel()
        if len(p) != len(y):
            raise ValueError(f"length mismatch: {len(p)} probabilities, {len(y)} labels")
        if len(np.unique(y)) < 2:
            # One class in validation: identity calibration beats a constant map.
            self._model = "identity"
            return self
        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        else:
            self._model = LogisticRegression().fit(p.reshape(-1, 1), y.astype(int))
        return self

    def freeze(self) -> "Calibrator":
        if not self.fitted:
            raise CalibratorFrozenError("cannot freeze an unfit calibrator")
        self._frozen = True
        return self

    def transform(self, proba) -> np.ndarray:
        if not self.fitted:
            raise CalibratorFrozenError("calibrator must be fit before transform")
        p = np.asarray(proba, dtype=float).ravel()
        if isinstance(self._model, str):          # identity
            return np.clip(p, 0.0, 1.0)
        if self.method == "isotonic":
            return np.clip(self._model.predict(p), 0.0, 1.0)
        return self._model.predict_proba(p.reshape(-1, 1))[:, 1]


def calibrate_proba(proba_val, y_val, proba_test, method: str = "isotonic") -> np.ndarray:
    """Fit a calibrator on VALIDATION, apply it to test probabilities.

    Argument order is (validation, validation labels, target) -- deliberately not
    (train, train labels, val), which is what fitting on training looks like.
    """
    return Calibrator(method).fit(proba_val, y_val).freeze().transform(proba_test)
