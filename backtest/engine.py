"""Walk-forward backtest engine.

This is the piece that ties everything together, and it is a real implementation
rather than a stub -- the previous version was `for fold in folds: pass`, which
meant the trade ledger, the cost model and the calibration audit were never
actually exercised by anything.

Per fold, in this order:

  1. carve the fold's training rows into inner-train / inner-validation
     CHRONOLOGICALLY, with the same embargo used between folds
  2. fit the scaler on inner-train only, then freeze it
  3. train the model on inner-train
  4. fit the calibrator on inner-VALIDATION, then freeze it
  5. pick the decision threshold on inner-validation, then freeze it
  6. score the test rows, walk them in time order, apply risk rules, record trades

Steps 2, 4 and 5 all touch only data that precedes the test period. That is the
whole point of the ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from backtest.ledger import TradeLedger
from features.scalers import FitScaler
from features.splits import _as_dt
from models.baselines import predict_proba, train_logistic
from models.calibrate import Calibrator
from models.select_threshold import (expected_value, select_threshold,
                                     spread_payoffs)
from reports.metrics import classification_report, trading_report
from risk.risk import Portfolio, RiskConfig, evaluate_candidate, should_kill
from schemas import CONTRACT_MULTIPLIER, EMBARGO_DAYS


@dataclass
class BacktestDataset:
    """Parallel arrays, one entry per simulated candidate. All same length."""
    candidates: list
    outcomes: list
    X: np.ndarray
    y: np.ndarray
    entry_dates: list

    def __post_init__(self):
        n = len(self.candidates)
        for name in ("outcomes", "X", "y", "entry_dates"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"BacktestDataset length mismatch: candidates={n}, "
                    f"{name}={len(getattr(self, name))}"
                )

    def __len__(self) -> int:
        return len(self.candidates)


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    n_test: int
    threshold: float
    n_traded: int
    classification: dict = field(default_factory=dict)
    trading: dict = field(default_factory=dict)
    killed: str = ""


@dataclass
class BacktestResult:
    ledger: TradeLedger
    folds: list = field(default_factory=list)
    probabilities: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "n_folds": len(self.folds),
            "n_trades": len(self.ledger),
            "total_pnl": self.ledger.total_pnl(),
            "total_fees": self.ledger.total_fees(),
            **trading_report(self.ledger.pnl_series()),
        }


def _inner_split(train_idx, entry_dates, val_frac: float, embargo_days: int):
    """Chronological inner train/validation split with an embargo between them."""
    ordered = sorted(train_idx, key=lambda i: _as_dt(entry_dates[i]))
    if len(ordered) < 4:
        return ordered, []
    cut = int(len(ordered) * (1.0 - val_frac))
    cut = min(max(cut, 1), len(ordered) - 1)
    val = ordered[cut:]
    val_start = _as_dt(entry_dates[val[0]])
    gap = timedelta(days=embargo_days)
    inner = [i for i in ordered[:cut] if _as_dt(entry_dates[i]) <= val_start - gap]
    return (inner or ordered[:cut]), val


def run_walk_forward(dataset: BacktestDataset, folds, model_fn=None, *,
                     risk_config: RiskConfig | None = None,
                     commission_per_contract: float = 0.65,
                     exchange_fee_per_contract: float = 0.0,
                     val_frac: float = 0.2,
                     embargo_days: int = EMBARGO_DAYS,
                     min_trades_for_threshold: int = 30,
                     model_version: str = "logistic_v1",
                     calibrate: bool = True) -> BacktestResult:
    """Run the full walk-forward. `model_fn(X, y)` returns a fitted model."""
    model_fn = model_fn or train_logistic
    rc = risk_config or RiskConfig()
    ledger = TradeLedger(model_version=model_version)
    result = BacktestResult(ledger=ledger)

    if not folds or len(dataset) == 0:
        return result

    for k, (train_idx, test_idx) in enumerate(folds, start=1):
        inner_idx, val_idx = _inner_split(train_idx, dataset.entry_dates,
                                          val_frac, embargo_days)
        if not inner_idx or not test_idx:
            continue

        Xtr, ytr = dataset.X[inner_idx], dataset.y[inner_idx]
        if len(np.unique(ytr)) < 2:
            continue                       # single-class fold: nothing to learn

        # (2) scaler: training rows only, then frozen
        scaler = FitScaler().fit(Xtr).freeze()
        model = model_fn(scaler.transform(Xtr), ytr)

        # (4)(5) calibrate + choose threshold on VALIDATION only
        threshold, calibrator = 0.5, None
        if val_idx:
            p_val = predict_proba(model, scaler.transform(dataset.X[val_idx]))
            y_val = dataset.y[val_idx]
            if calibrate and len(np.unique(y_val)) >= 2:
                calibrator = Calibrator("isotonic").fit(p_val, y_val).freeze()
                p_val = calibrator.transform(p_val)
            val_pnl = np.array([_net_pnl(dataset.outcomes[i], commission_per_contract,
                                         exchange_fee_per_contract) for i in val_idx])
            threshold = select_threshold(
                p_val, y_val, pnl=val_pnl, min_trades=min_trades_for_threshold,
            )

        # (6) score test rows and walk them in time order
        ordered_test = sorted(test_idx, key=lambda i: _as_dt(dataset.entry_dates[i]))
        p_test = predict_proba(model, scaler.transform(dataset.X[ordered_test]))
        if calibrator is not None:
            p_test = calibrator.transform(p_test)

        portfolio = Portfolio(account_size=rc.account_size)
        n_traded, killed = 0, ""
        for pos, i in enumerate(ordered_test):
            cand, outcome = dataset.candidates[i], dataset.outcomes[i]
            proba = float(p_test[pos])
            day = _as_dt(dataset.entry_dates[i])

            stop, why = should_kill(portfolio, rc, day)
            if stop:
                killed = why
                break

            profit, loss = spread_payoffs(cand.entry_credit)
            costs = _fee_per_share(commission_per_contract, exchange_fee_per_contract)
            ev = expected_value(proba, profit, loss, costs)

            allow, _ = evaluate_candidate(
                cand, proba, ev, portfolio, rc, decision_time=day,
                data_is_clean=True,
            )
            if not allow or proba < threshold or outcome.exit_reason == "NO_DATA":
                continue

            from risk.risk import position_size
            n = max(position_size(cand, rc), 1)
            row = ledger.record(
                outcome, contracts=n,
                commission_per_contract=commission_per_contract,
                exchange_fee_per_contract=exchange_fee_per_contract,
                model_probability=proba, fold=k,
            )
            portfolio.record_entry(day, row.max_risk_dollars)
            portfolio.close_trade(outcome.exit_time, row.net_pnl, row.max_risk_dollars)
            result.probabilities.append(proba)
            result.labels.append(int(outcome.label_win))
            n_traded += 1

        fold_rows = [t for t in ledger.trades if t.fold == k]
        result.folds.append(FoldResult(
            fold=k, n_train=len(inner_idx), n_val=len(val_idx), n_test=len(test_idx),
            threshold=threshold, n_traded=n_traded, killed=killed,
            classification=classification_report(
                dataset.y[ordered_test], p_test) if len(ordered_test) else {},
            trading=trading_report([t.net_pnl for t in fold_rows]),
        ))
    return result


def _fee_per_share(commission: float, exchange_fee: float) -> float:
    from backtest.costs import cost_per_spread_per_share
    return cost_per_spread_per_share(commission, exchange_fee)


def _net_pnl(outcome, commission: float, exchange_fee: float) -> float:
    """Dollar net P&L for one spread, used to score thresholds on validation."""
    import math
    if outcome.exit_reason == "NO_DATA" or math.isnan(outcome.gross_pnl):
        return float("nan")
    from backtest.costs import roundtrip_fees
    gross = (outcome.candidate.entry_credit - outcome.exit_debit) * CONTRACT_MULTIPLIER
    return gross - roundtrip_fees(commission, 1, exchange_fee_per_contract=exchange_fee)
