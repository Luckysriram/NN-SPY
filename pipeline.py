"""End-to-end wiring: raw source rows in, a modellable dataset out.

    raw rows -> normalize -> validate -> candidates -> forward marks -> simulate
             -> features -> BacktestDataset

Kept as one readable function because the ORDER is the part that matters. Each
step consumes only what precedes it in time, and `build_dataset` never looks at
a quote stamped after the decision it is building features for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backtest.engine import BacktestDataset
from data.adapters.sample import normalize_rows
from data.candidates import accepted_only, generate_candidates, rejection_summary
from data.validate import flag_summary, validate_quotes
from features.features import (FEATURE_COLS, build_feature_vector, nan_report,
                               to_matrix)
from labels.outcomes import label_distribution, outcome_label
from labels.simulator import simulate_trade, snapshots_from_quotes


@dataclass
class BuildReport:
    """What the build actually did -- read this before trusting the dataset."""
    n_rows: int = 0
    n_quotes: int = 0
    quality: dict = field(default_factory=dict)
    rejections: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    nan_features: dict = field(default_factory=dict)
    n_dropped_incomplete: int = 0
    n_final: int = 0

    def render(self) -> str:
        L = ["=" * 62, "  DATASET BUILD REPORT", "=" * 62,
             f"  source rows          : {self.n_rows:,}",
             f"  canonical quotes     : {self.n_quotes:,}",
             f"  clean quotes         : {self.quality.get('_clean_quotes', 0):,}",
             "  candidate rejections :"]
        for k, v in sorted(self.rejections.items(), key=lambda kv: -kv[1]):
            L.append(f"    {k:<28} {v:,}")
        L.append("  label distribution   :")
        for k, v in sorted(self.labels.items()):
            L.append(f"    {k:<28} {v}")
        if self.nan_features:
            L.append("  incomplete features  :")
            for k, v in sorted(self.nan_features.items(), key=lambda kv: -kv[1])[:8]:
                L.append(f"    {k:<28} {v:,}")
        L += [f"  rows dropped (NaN)   : {self.n_dropped_incomplete:,}",
              f"  FINAL TRAINING ROWS  : {self.n_final:,}", "=" * 62]
        return "\n".join(L)


def build_dataset(rows, underlying_bars, vix_bars, *, config=None, calendar=None,
                  fees_per_spread: float = 0.0, slippage: float = 0.0,
                  symbol: str = "SPY", max_days: int | None = None):
    """Turn raw source rows into a (BacktestDataset, BuildReport)."""
    cfg = config or {}
    rows = list(rows)

    quotes = validate_quotes(normalize_rows(rows, symbol=symbol).quotes)
    report = BuildReport(n_rows=len(rows), n_quotes=len(quotes),
                         quality=flag_summary(quotes))

    by_day: dict = {}
    for q in quotes:
        by_day.setdefault(q.timestamp, []).append(q)
    decision_times = sorted(by_day)
    if max_days is not None:
        decision_times = decision_times[:max_days]

    all_candidates, candidates, outcomes, vectors, entry_dates = [], [], [], [], []
    for ts in decision_times:
        day_candidates = generate_candidates(by_day[ts], ts, cfg, calendar=calendar)
        all_candidates.extend(day_candidates)

        for cand in accepted_only(day_candidates):
            # Marks strictly after entry. snapshots_from_quotes enforces this.
            marks = snapshots_from_quotes(cand, quotes)
            outcome = simulate_trade(cand, marks, fees_per_spread=fees_per_spread,
                                     slippage=slippage)
            if outcome.exit_reason == "NO_DATA":
                continue
            # Features use only bars at or before the decision time.
            vec = build_feature_vector(cand, underlying_bars, vix_bars, ts, calendar)
            candidates.append(cand)
            outcomes.append(outcome)
            vectors.append(vec)
            entry_dates.append(ts)

    report.rejections = rejection_summary(all_candidates)
    report.labels = label_distribution(outcomes)

    if not candidates:
        report.n_final = 0
        return BacktestDataset([], [], np.empty((0, len(FEATURE_COLS))),
                               np.empty(0), []), report

    X = to_matrix(vectors)
    y = np.array([outcome_label(o) for o in outcomes], dtype=float)
    report.nan_features = nan_report(X)

    keep = ~np.isnan(X).any(axis=1)
    report.n_dropped_incomplete = int((~keep).sum())
    report.n_final = int(keep.sum())

    dataset = BacktestDataset(
        candidates=[c for c, k in zip(candidates, keep) if k],
        outcomes=[o for o, k in zip(outcomes, keep) if k],
        X=X[keep], y=y[keep],
        entry_dates=[d for d, k in zip(entry_dates, keep) if k],
    )
    return dataset, report
