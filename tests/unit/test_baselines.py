import numpy as np
import pytest

from features.features import FEATURE_INDEX, N_FEATURES
from models.baselines import (CREDIT_TO_WIDTH, always_trade_predict,
                              no_trade_predict, predict_proba,
                              simple_rule_predict, train_gradient_boosting,
                              train_logistic)

X2 = np.array([[0.1, 0.2], [0.9, 0.8], [0.2, 0.1], [0.8, 0.9]])
Y2 = np.array([0, 1, 0, 1])


def test_logistic_trains_and_predicts_proba():
    p = train_logistic(X2, Y2).predict_proba(X2)[:, 1]
    assert p.shape == (4,)
    assert ((p >= 0) & (p <= 1)).all()


def test_gradient_boosting_trains():
    m = train_gradient_boosting(np.array([[0.1], [0.9], [0.2], [0.8]]), Y2)
    assert m.predict(np.array([[0.1], [0.9], [0.2], [0.8]])).shape == (4,)


def test_models_are_seeded_and_reproducible():
    """Global constraint: record all random seeds. An unseeded model cannot be."""
    a = train_gradient_boosting(X2, Y2, seed=7).predict_proba(X2)[:, 1]
    b = train_gradient_boosting(X2, Y2, seed=7).predict_proba(X2)[:, 1]
    assert np.allclose(a, b)


def test_simple_rule_reads_credit_to_width_by_name():
    """The old rule hard-coded column 20, which was `credit`, not credit_to_width."""
    X = np.zeros((3, N_FEATURES))
    # Values straddle the rebased cutoff (0.096). The old test used 0.1/0.5/0.9,
    # which are all above it -- and were all BELOW the old unreachable 0.40.
    X[:, FEATURE_INDEX["credit_to_width"]] = [0.05, 0.12, 0.19]
    p = simple_rule_predict(X)
    assert list(p) == [0.4, 0.6, 0.6]


def test_simple_rule_is_unaffected_by_the_neighbouring_credit_column():
    X = np.zeros((2, N_FEATURES))
    X[:, FEATURE_INDEX["credit"]] = [0.9, 0.9]          # high credit
    X[:, FEATURE_INDEX["credit_to_width"]] = [0.05, 0.05]  # ratio below cutoff
    assert list(simple_rule_predict(X)) == [0.4, 0.4]


def test_credit_to_width_index_matches_the_feature_list():
    assert CREDIT_TO_WIDTH == FEATURE_INDEX["credit_to_width"]


def test_simple_rule_rejects_a_truncated_feature_matrix():
    with pytest.raises(ValueError, match="full feature matrix"):
        simple_rule_predict(np.zeros((2, 3)))


def test_no_trade_and_always_trade_baselines():
    X = np.zeros((5, N_FEATURES))
    assert list(no_trade_predict(X)) == [0.0] * 5
    assert list(always_trade_predict(X)) == [1.0] * 5


def test_predict_proba_accepts_a_plain_callable():
    assert list(predict_proba(lambda X: np.full(len(X), 0.7), X2)) == [0.7] * 4


def test_simple_rule_cutoff_is_reachable_on_real_data():
    """The old 0.40 cutoff was unreachable: a $5-wide 20-delta SPY spread
    collects ~10% of its width, so the rule was a constant."""
    from models.baselines import DEFAULT_CTW_CUTOFF
    assert 0.04 < DEFAULT_CTW_CUTOFF < 0.21


def test_simple_rule_actually_discriminates_at_the_new_cutoff():
    X = np.zeros((4, N_FEATURES))
    X[:, FEATURE_INDEX["credit_to_width"]] = [0.05, 0.09, 0.11, 0.20]
    p = simple_rule_predict(X)
    assert len(set(p)) == 2, "rule must split the population, not return a constant"
    assert p[0] == 0.4 and p[-1] == 0.6


def test_train_simple_rule_fits_the_cutoff_from_training_data_only():
    from models.baselines import train_simple_rule
    X = np.zeros((100, N_FEATURES))
    X[:, FEATURE_INDEX["credit_to_width"]] = np.linspace(0.05, 0.15, 100)
    m = train_simple_rule(X)
    assert m.cutoff == pytest.approx(np.median(X[:, FEATURE_INDEX["credit_to_width"]]))


def test_train_simple_rule_returns_an_sklearn_shaped_model():
    from models.baselines import train_simple_rule
    X = np.zeros((10, N_FEATURES))
    X[:, FEATURE_INDEX["credit_to_width"]] = np.linspace(0.05, 0.15, 10)
    proba = train_simple_rule(X).predict_proba(X)
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_a_training_fold_cutoff_does_not_see_the_test_fold():
    from models.baselines import train_simple_rule
    train = np.zeros((50, N_FEATURES)); train[:, FEATURE_INDEX["credit_to_width"]] = 0.08
    test = np.zeros((50, N_FEATURES)); test[:, FEATURE_INDEX["credit_to_width"]] = 0.99
    assert train_simple_rule(train).cutoff == pytest.approx(0.08)
