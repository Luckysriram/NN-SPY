import numpy as np
import pytest

from models.numpy_net import NumpyMLP

X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
Y = np.array([0.0, 1.0, 0.0, 1.0])      # separable by feature 0


def test_gradient_check_confirms_backprop_matches_the_derivative():
    """The test that actually proves the hand-written backward pass is right."""
    net = NumpyMLP([2, 4, 1], seed=42)
    assert net.gradient_check(X, Y) < 1e-6


def test_gradient_check_on_a_deeper_network():
    net = NumpyMLP([3, 5, 4, 1], seed=1)
    rng = np.random.default_rng(0)
    Xd, yd = rng.normal(size=(8, 3)), (rng.uniform(size=8) > 0.5).astype(float)
    assert net.gradient_check(Xd, yd) < 1e-6


def test_learns_a_linearly_separable_problem():
    net = NumpyMLP([2, 4, 1], seed=42)
    net.fit(X, Y, epochs=400, lr=0.5)
    preds = net.predict_proba(X).ravel()
    assert np.mean((preds > 0.5).astype(float) == Y) == 1.0


def test_loss_decreases_during_training():
    net = NumpyMLP([2, 4, 1], seed=42)
    history = net.fit(X, Y, epochs=200, lr=0.5)
    assert history[-1] < history[0]


def test_bce_loss_positive():
    net = NumpyMLP([2, 4, 1], seed=42)
    preds, _ = net.forward(np.array([[0.0, 0.0]]))
    assert net.bce_loss(preds, np.array([1.0])) > 0


def test_bce_loss_near_zero_for_a_confident_correct_prediction():
    assert NumpyMLP.bce_loss(np.array([[0.999]]), np.array([1.0])) < 0.01


def test_forward_returns_probabilities():
    p, cache = NumpyMLP([2, 4, 1], seed=42).forward(X)
    assert p.shape == (4, 1)
    assert ((p > 0) & (p < 1)).all()
    assert len(cache) == 3               # input, hidden, output


def test_sigmoid_does_not_overflow_on_extreme_inputs():
    net = NumpyMLP([1, 2, 1], seed=0)
    p = net.predict_proba(np.array([[1e6], [-1e6]]))
    assert np.isfinite(p).all()


def test_same_seed_reproduces_the_same_weights():
    a, b = NumpyMLP([2, 4, 1], seed=7), NumpyMLP([2, 4, 1], seed=7)
    assert np.allclose(a.W[0], b.W[0])
    assert not np.allclose(a.W[0], NumpyMLP([2, 4, 1], seed=8).W[0])


def test_rejects_a_non_binary_output_layer():
    with pytest.raises(ValueError, match="final dim must be 1"):
        NumpyMLP([2, 4, 2])


def test_rejects_a_degenerate_layer_spec():
    with pytest.raises(ValueError, match="at least an input"):
        NumpyMLP([1])
