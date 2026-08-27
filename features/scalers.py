"""A scaler that cannot be refit once frozen.

The point of the class is to make one specific mistake impossible: fitting
normalisation statistics on data that includes the validation or test period.
The previous version had a `freeze()` that set a flag nothing ever read, so the
guarantee it advertised did not exist. Here `fit()` after `freeze()` raises.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class ScalerFrozenError(RuntimeError):
    """Raised when something tries to refit a frozen scaler."""


class FitScaler:
    """Fit once on the training period, freeze, then only ever transform."""

    def __init__(self, ddof: int = 0):
        # ddof=0 matches sklearn's StandardScaler (population std). Stated
        # explicitly because the difference silently changes every scaled value.
        self._scaler = StandardScaler()
        self._frozen = False
        self._fitted = False
        self._n_features: int | None = None
        self.ddof = ddof

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def n_features(self) -> int | None:
        return self._n_features

    def fit(self, X) -> "FitScaler":
        if self._frozen:
            raise ScalerFrozenError(
                "refusing to refit a frozen scaler -- this is the guard against "
                "fitting normalisation on validation or test data"
            )
        X = np.asarray(X, dtype=float)
        if np.isnan(X).any():
            raise ValueError(
                "NaN in scaler training data. StandardScaler would propagate it "
                "into the mean and poison every transform. Clean the rows first."
            )
        self._scaler.fit(X)
        self._fitted = True
        self._n_features = X.shape[1]
        return self

    def freeze(self) -> "FitScaler":
        if not self._fitted:
            raise ScalerFrozenError("cannot freeze a scaler that was never fit")
        self._frozen = True
        return self

    def fit_freeze(self, X) -> "FitScaler":
        return self.fit(X).freeze()

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise ScalerFrozenError("scaler must be fit before transform")
        X = np.asarray(X, dtype=float)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(
                f"scaler was fit on {self._n_features} features but received "
                f"{X.shape[1]}. Feature list and scaler are out of sync."
            )
        return self._scaler.transform(X)

    @property
    def mean_(self):
        return self._scaler.mean_

    @property
    def scale_(self):
        return self._scaler.scale_
