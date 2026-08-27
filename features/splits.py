"""Chronological splits with a purge embargo. Never shuffles.

Why the embargo exists
----------------------
A spread entered on 2020-12-15 with 45 DTE is not resolved until roughly
2021-01-29. Its label is computed from quotes in January 2021. If the training
period ends 2020-12-31 and testing starts 2021-01-01 with no gap, that training
row's label was built from test-period data -- the model is fit on the answer to
part of its own exam.

The leak is quiet: no test overlaps, no duplicated rows, and cross-validation
looks impeccable. It just inflates every out-of-sample number.

So each split drops training entries whose trade could still have been open when
the next period began. The gap is `EMBARGO_DAYS` (= MAX_DTE - TIME_EXIT_DTE = 38),
the longest a trade can live under the strategy rules.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from schemas import EMBARGO_DAYS


def _as_dt(d):
    return d if isinstance(d, datetime) else datetime.combine(d, datetime.min.time())


def chronological_split(dates, train_end, val_end, test_end,
                        embargo_days: int = EMBARGO_DAYS):
    """Split entry dates into (train, val, test) index lists, purging the embargo.

    Periods are (start, train_end], (train_end, val_end], (val_end, test_end].
    Training entries within `embargo_days` of `train_end` are dropped, as are
    validation entries within `embargo_days` of `val_end`.
    """
    train_end, val_end, test_end = _as_dt(train_end), _as_dt(val_end), _as_dt(test_end)
    gap = timedelta(days=embargo_days)
    train, val, test = [], [], []
    for i, raw in enumerate(dates):
        d = _as_dt(raw)
        if d <= train_end:
            if d <= train_end - gap:
                train.append(i)          # else: purged, trade may span the boundary
        elif d <= val_end:
            if d <= val_end - gap:
                val.append(i)
        elif d <= test_end:
            test.append(i)
    return train, val, test


def walk_forward_folds(dates, first_train_end, last_test_end, fold_years: int = 1,
                       embargo_days: int = EMBARGO_DAYS):
    """Expanding-window folds: train on everything so far, test the next block.

    train 2016-20 -> test '21; train 2016-21 -> test '22; and so on. Each fold's
    training set is purged by `embargo_days` before its test period starts.
    """
    first_train_end, last_test_end = _as_dt(first_train_end), _as_dt(last_test_end)
    gap = timedelta(days=embargo_days)
    folds = []
    train_end = first_train_end
    while train_end < last_test_end:
        test_start = train_end
        test_end = min(_as_dt(datetime(test_start.year + fold_years, 12, 31)), last_test_end)
        if test_end <= test_start:
            break
        train_idx = [i for i, d in enumerate(dates) if _as_dt(d) <= train_end - gap]
        test_idx = [i for i, d in enumerate(dates) if test_start < _as_dt(d) <= test_end]
        if train_idx and test_idx:
            folds.append((train_idx, test_idx))
        train_end = test_end
    return folds


def assert_split_is_clean(dates, train_idx, test_idx, embargo_days: int = EMBARGO_DAYS,
                          exit_dates=None) -> None:
    """Verify no training trade can overlap the test period.

    With `exit_dates` the check is exact; without them it assumes the worst case
    (entry + embargo_days).
    """
    if not train_idx or not test_idx:
        return
    if set(train_idx) & set(test_idx):
        raise ValueError("train and test index sets overlap")

    test_start = min(_as_dt(dates[i]) for i in test_idx)
    for i in train_idx:
        if exit_dates is not None:
            close = _as_dt(exit_dates[i])
        else:
            close = _as_dt(dates[i]) + timedelta(days=embargo_days)
        if close >= test_start:
            raise ValueError(
                f"training row {i} (entered {dates[i]}) is still open at "
                f"{close}, on or after test start {test_start}. Its label uses "
                "test-period quotes. Widen embargo_days."
            )


def describe_folds(dates, folds) -> list[dict]:
    """Human-readable fold table for the experiment log."""
    out = []
    for k, (tr, te) in enumerate(folds, start=1):
        out.append({
            "fold": k,
            "train_n": len(tr), "test_n": len(te),
            "train_start": min(dates[i] for i in tr),
            "train_end": max(dates[i] for i in tr),
            "test_start": min(dates[i] for i in te),
            "test_end": max(dates[i] for i in te),
            "gap_days": (_as_dt(min(dates[i] for i in te))
                         - _as_dt(max(dates[i] for i in tr))).days,
        })
    return out
