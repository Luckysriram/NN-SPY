"""Split tests, centred on the embargo.

The leak the embargo prevents is invisible to the usual checks: train and test
index sets are disjoint, no row is duplicated, and every date is on the correct
side of the boundary. What leaks is the LABEL -- a trade entered just before the
cut is resolved with quotes from just after it.
"""
from datetime import datetime, timedelta

import pytest

from features.splits import (assert_split_is_clean, chronological_split,
                             describe_folds, walk_forward_folds)
from schemas import EMBARGO_DAYS

DAILY = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(760)]
YEARLY = [datetime(y, 6, 1) for y in range(2016, 2026)]
BOUNDS = (datetime(2020, 12, 31), datetime(2021, 6, 30), datetime(2021, 12, 31))


def test_chronological_split_is_contiguous_and_disjoint():
    tr, va, te = chronological_split(YEARLY, datetime(2022, 12, 31),
                                     datetime(2023, 12, 31), datetime(2025, 12, 31),
                                     embargo_days=0)
    assert set(tr) & set(va) == set()
    assert set(va) & set(te) == set()
    assert len(tr) == 7 and len(va) == 1 and len(te) == 2


def test_split_never_reorders_or_shuffles():
    tr, _, _ = chronological_split(DAILY, *BOUNDS)
    assert tr == sorted(tr)


def test_embargo_opens_a_gap_between_train_and_validation():
    tr, va, _ = chronological_split(DAILY, *BOUNDS)
    gap = (min(DAILY[i] for i in va) - max(DAILY[i] for i in tr)).days
    assert gap > EMBARGO_DAYS


def test_without_an_embargo_a_training_trade_is_still_open_at_the_boundary():
    """This is the leak, made explicit."""
    tr, va, _ = chronological_split(DAILY, *BOUNDS, embargo_days=0)
    with pytest.raises(ValueError, match="still open"):
        assert_split_is_clean(DAILY, tr, va)


def test_with_the_embargo_the_same_split_is_clean():
    tr, va, _ = chronological_split(DAILY, *BOUNDS)
    assert_split_is_clean(DAILY, tr, va)


def test_assert_split_is_clean_uses_exact_exit_dates_when_given():
    tr, va, _ = chronological_split(DAILY, *BOUNDS, embargo_days=0)
    same_day_exits = list(DAILY)          # every trade closes the day it opens
    assert_split_is_clean(DAILY, tr, va, exit_dates=same_day_exits)


def test_assert_split_is_clean_rejects_overlapping_index_sets():
    with pytest.raises(ValueError, match="overlap"):
        assert_split_is_clean(DAILY, [1, 2, 3], [3, 4])


def test_walk_forward_folds_grow_train():
    folds = walk_forward_folds(YEARLY, datetime(2020, 12, 31),
                               datetime(2025, 12, 31), fold_years=1, embargo_days=0)
    assert len(folds) == 5
    for tr, te in folds:
        assert set(tr) & set(te) == set()
    assert len(folds[0][0]) == 5
    assert len(folds[-1][0]) == 9
    assert [len(tr) for tr, _ in folds] == sorted(len(tr) for tr, _ in folds)


def test_every_walk_forward_fold_respects_the_embargo():
    folds = walk_forward_folds(DAILY, datetime(2020, 12, 31), datetime(2021, 12, 31))
    assert folds
    for tr, te in folds:
        assert_split_is_clean(DAILY, tr, te)


def test_walk_forward_never_tests_before_it_trains():
    folds = walk_forward_folds(DAILY, datetime(2020, 12, 31), datetime(2021, 12, 31))
    for tr, te in folds:
        assert max(DAILY[i] for i in tr) < min(DAILY[i] for i in te)


def test_describe_folds_reports_the_gap():
    folds = walk_forward_folds(DAILY, datetime(2020, 12, 31), datetime(2021, 12, 31))
    for row in describe_folds(DAILY, folds):
        assert row["gap_days"] > EMBARGO_DAYS
        assert row["train_n"] > 0 and row["test_n"] > 0


def test_no_folds_when_the_window_is_empty():
    assert walk_forward_folds(DAILY, datetime(2021, 12, 31), datetime(2021, 12, 31)) == []
