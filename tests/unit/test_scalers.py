import numpy as np
import pytest

from features.scalers import FitScaler, ScalerFrozenError

X_TRAIN = np.array([[0.0], [10.0], [20.0]])


def test_scaler_fits_train_transforms_test_without_refit():
    sc = FitScaler().fit(X_TRAIN).freeze()
    # StandardScaler uses the POPULATION std (ddof=0) = 8.165, not the sample
    # std of 10.0. Asserting against 10.0 is what made the old test fail.
    expected = (5.0 - 10.0) / np.std([0.0, 10.0, 20.0])
    assert sc.transform([[5.0]])[0, 0] == pytest.approx(expected)


def test_freeze_actually_blocks_a_refit():
    """The whole point of the class. The old flag was set but never read."""
    sc = FitScaler().fit(X_TRAIN).freeze()
    with pytest.raises(ScalerFrozenError, match="refusing to refit"):
        sc.fit(np.array([[100.0], [200.0]]))


def test_transform_before_fit_raises():
    with pytest.raises(ScalerFrozenError, match="must be fit"):
        FitScaler().transform([[1.0]])


def test_freeze_before_fit_raises():
    with pytest.raises(ScalerFrozenError, match="never fit"):
        FitScaler().freeze()


def test_feature_count_mismatch_is_caught():
    sc = FitScaler().fit(np.zeros((3, 30))).freeze()
    with pytest.raises(ValueError, match="fit on 30 features"):
        sc.transform(np.zeros((1, 33)))


def test_nan_in_training_data_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        FitScaler().fit(np.array([[1.0], [np.nan]]))


def test_fit_freeze_chains():
    sc = FitScaler().fit_freeze(X_TRAIN)
    assert sc.fitted and sc.frozen
