import numpy as np
import pytest
import torch

from models.mlp import (TorchModel, build_mlp, predict_proba, set_all_seeds,
                        train_mlp)


def data(n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 5)
    y = (X[:, 0] + 0.2 * rng.rand(n) > 0.6).astype(float)
    return X, y


def test_build_mlp_outputs_a_single_value_per_row():
    out = build_mlp(n_features=5)(torch.randn(2, 5))
    assert out.shape == (2, 1)


def test_build_mlp_returns_logits_not_probabilities():
    """Logits + BCEWithLogitsLoss is the numerically stable pairing."""
    out = build_mlp(n_features=5, seed=1)(torch.randn(64, 5) * 20)
    assert (out.abs() > 1.0).any()          # unbounded, unlike a sigmoid output


def test_predict_proba_is_bounded_and_uses_all_not_bool():
    """bool() on a multi-element tensor raises; .all() is the correct check."""
    p = predict_proba(build_mlp(5), np.random.rand(4, 5))
    assert ((p > 0) & (p < 1)).all()
    assert p.shape == (4,)


def test_train_mlp_records_every_seed():
    X, y = data(64)
    model, hist = train_mlp(build_mlp(5), X, y, epochs=5, batch=16, seed=42)
    assert "seed" in hist and hist["seed"] == 42
    assert set(hist.seeds) == {"python", "numpy", "torch"}
    assert len(hist["train_loss"]) > 0


def test_early_stopping_monitors_validation_loss_when_given_one():
    """Watching train loss stops when fitting stalls -- the wrong signal."""
    X, y = data(300)
    Xv, yv = data(100, seed=1)
    _, hist = train_mlp(build_mlp(5), X, y, X_val=Xv, y_val=yv,
                        epochs=10, batch=32, patience=3)
    assert hist.early_stopping_signal == "val_loss"
    assert len(hist.val_loss) == len(hist.train_loss)


def test_missing_validation_set_is_stated_not_hidden():
    X, y = data(64)
    _, hist = train_mlp(build_mlp(5), X, y, epochs=3, batch=16)
    assert "NO VALIDATION SET" in hist.early_stopping_signal
    assert hist.val_loss == []


def test_early_stopping_actually_triggers():
    X, y = data(200)
    Xv, yv = X[:50], 1.0 - y[:50]          # validation deliberately anti-correlated
    _, hist = train_mlp(build_mlp(5), X, y, X_val=Xv, y_val=yv,
                        epochs=60, batch=32, patience=2)
    assert hist.stopped_early
    assert hist.epochs_run < 60


def test_best_weights_are_restored_not_the_last_epoch():
    X, y = data(200)
    Xv, yv = data(80, seed=3)
    model, hist = train_mlp(build_mlp(5), X, y, X_val=Xv, y_val=yv,
                            epochs=25, batch=32, patience=30)
    import torch.nn as nn
    with torch.no_grad():
        final = float(nn.BCEWithLogitsLoss()(
            model(torch.tensor(Xv, dtype=torch.float32)),
            torch.tensor(yv, dtype=torch.float32).reshape(-1, 1)))
    assert final == pytest.approx(min(hist.val_loss), abs=1e-5)


def test_training_learns_something():
    X, y = data(400)
    model, hist = train_mlp(build_mlp(5), X, y, epochs=40, batch=32, lr=1e-2)
    assert hist.train_loss[-1] < hist.train_loss[0]
    from reports.metrics import roc_auc
    assert roc_auc(y, predict_proba(model, X)) > 0.7


def test_same_seed_reproduces_the_run():
    """A recorded seed that does not reproduce the run is not a record of anything."""
    X, y = data(200)
    def run(seed):
        m = build_mlp(5, seed=seed)
        m, _ = train_mlp(m, X, y, epochs=6, batch=32, seed=seed)
        return predict_proba(m, X)
    assert np.allclose(run(7), run(7))
    assert not np.allclose(run(7), run(8))


def test_nan_features_are_rejected_before_training():
    X, y = data(32)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        train_mlp(build_mlp(5), X, y, epochs=1)


def test_dropout_is_off_during_prediction():
    m = build_mlp(5, seed=0)
    X = np.random.rand(8, 5)
    assert np.allclose(predict_proba(m, X), predict_proba(m, X))


def test_torch_model_adapter_matches_the_sklearn_interface():
    X, y = data(64)
    m, hist = train_mlp(build_mlp(5), X, y, epochs=3, batch=16)
    proba = TorchModel(m, hist).predict_proba(X)
    assert proba.shape == (64, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_set_all_seeds_reports_what_it_set():
    assert set_all_seeds(11) == {"python": 11, "numpy": 11, "torch": 11}
