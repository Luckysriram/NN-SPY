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
from schemas import MAX_DTE, MIN_DTE


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


def build_dataset_from_frame(df, underlying_bars, vix_bars, *, config=None,
                             calendar=None, fees_per_spread: float = 0.0,
                             slippage: float = 0.0, symbol: str = "SPY",
                             min_entry_dte: int = MIN_DTE,
                             max_entry_dte: int = MAX_DTE,
                             max_days: int | None = None, progress=None,
                             sim_kw: dict | None = None):
    """Scale path: build from the canonical parquet frame.

    Two things differ from `build_dataset`, both forced by real data volume:

    * Forward marks come from a `QuoteIndex` built once, instead of rescanning
      the quote stream per candidate. On 3.7M quotes that is the difference
      between 15 seconds and 11 hours.
    * `OptionQuote` objects are materialised one DAY at a time, and only for the
      entry DTE window. Building 3.7M dataclass instances up front costs more
      memory than the dataframe they came from; the index needs plain arrays,
      and candidate generation only ever looks at one day.
    """
    from labels.quote_index import QuoteIndex

    cfg = config or {}
    index = QuoteIndex.from_frame(df)

    entry_rows = df[(df["dte"] >= min_entry_dte) & (df["dte"] <= max_entry_dte)]
    report = BuildReport(n_rows=len(df), n_quotes=len(entry_rows))

    days = sorted(entry_rows["timestamp"].unique())
    if max_days is not None:
        days = days[:max_days]

    all_candidates, candidates, outcomes, vectors, entry_dates = [], [], [], [], []
    n_clean = 0
    for n, ts in enumerate(days, 1):
        day = entry_rows[entry_rows["timestamp"] == ts]
        quotes = validate_quotes(_frame_to_quotes(day, symbol))
        n_clean += sum(1 for q in quotes if q.is_clean)
        decision_time = _as_datetime(ts)

        day_candidates = generate_candidates(quotes, decision_time, cfg,
                                             calendar=calendar)
        all_candidates.extend(day_candidates)

        for cand in accepted_only(day_candidates):
            outcome = simulate_trade(cand, index.snapshots_for(cand),
                                     fees_per_spread=fees_per_spread,
                                     slippage=slippage, **(sim_kw or {}))
            if outcome.exit_reason == "NO_DATA":
                continue
            candidates.append(cand)
            outcomes.append(outcome)
            vectors.append(build_feature_vector(cand, underlying_bars, vix_bars,
                                                decision_time, calendar))
            entry_dates.append(decision_time)

        if progress and (n % 250 == 0 or n == len(days)):
            progress(f"  {n:,}/{len(days):,} days   {len(candidates):,} labelled trades")

    report.quality = {"_total_quotes": len(entry_rows), "_clean_quotes": n_clean}
    return _finish(report, all_candidates, candidates, outcomes, vectors, entry_dates)


def _as_datetime(ts):
    import pandas as pd
    return pd.Timestamp(ts).to_pydatetime()


def _frame_to_quotes(day, symbol: str):
    """Materialise one day's rows as OptionQuote objects."""
    import math

    from schemas import OptionQuote
    out = []
    for r in day.itertuples(index=False):
        nan = float("nan")
        out.append(OptionQuote(
            timestamp=_as_datetime(r.timestamp), symbol=symbol,
            expiry=_as_datetime(r.expiry).date(), dte=int(r.dte),
            strike=float(r.strike), option_type=r.option_type,
            bid=float(r.bid), ask=float(r.ask), mid=float(r.mid),
            last=float(r.last) if not math.isnan(r.last) else nan,
            volume=0 if r.volume != r.volume else int(r.volume),
            open_interest=int(r.open_interest),
            iv=float(r.iv), delta=float(r.delta), gamma=float(r.gamma),
            theta=float(r.theta), vega=float(r.vega),
            underlying_price=float(r.underlying_price),
        ))
    return out


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

    return _finish(report, all_candidates, candidates, outcomes, vectors, entry_dates)


def _finish(report, all_candidates, candidates, outcomes, vectors, entry_dates):
    """Assemble the dataset and fill in the build report. Shared by both paths."""
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
