"""Phase 5 -- baselines only. Is there any edge here at all?

No neural network. The question this answers is narrower and more important:
does ANY subset of these trades clear the break-even win rate after costs?

Read the calibration audit before the AUC. On a strategy whose base rate is
57.5%, a model scoring 57% accuracy has learned nothing, and one scoring 66% is
still at break-even before costs.

    python run_phase5.py
"""
from __future__ import annotations

import pickle
from datetime import timedelta

import numpy as np

from backtest.engine import BacktestDataset, run_walk_forward
from features.splits import describe_folds, walk_forward_folds
from models.baselines import (train_gradient_boosting, train_logistic,
                              train_simple_rule)
from models.select_threshold import breakeven_win_rate, spread_payoffs
from reports.metrics import classification_report, roc_auc, trading_report
from reports.monthly_report import by_prob_bucket, dte_bucket, monthly_report, render
from risk.risk import RiskConfig

COMMISSION = 0.65
DATASET = "data/raw/dataset.pkl"


class ConstantModel:
    """Always-trade / never-trade controls."""

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        n = len(np.asarray(X))
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def load() -> BacktestDataset:
    # Pickle is safe here only because this file is produced locally by
    # pipeline.build_dataset_from_frame on this machine and is gitignored --
    # it is a build cache, never an input from elsewhere. Do not point DATASET
    # at a downloaded file; unpickling executes arbitrary code.
    d = pickle.load(open(DATASET, "rb"))
    return BacktestDataset(d["candidates"], d["outcomes"], d["X"], d["y"],
                           d["entry_dates"])


def banner(t):
    print(f"\n{'=' * 74}\n  {t}\n{'=' * 74}")


def main() -> int:
    ds = load()
    y = ds.y
    print(f"dataset: {len(ds):,} trades, {ds.X.shape[1]} features")
    print(f"  {ds.entry_dates[0].date()} .. {ds.entry_dates[-1].date()}")
    print(f"  BASE RATE {y.mean():.4f}  <- the number every result is read against")

    profit, loss = spread_payoffs(1.0)          # payoffs scale with credit
    be = breakeven_win_rate(profit, loss)
    print(f"  BREAK-EVEN win rate {be:.4f} (wins +0.5x credit, losses -1x credit)")
    print(f"  gap to close: {be - y.mean():+.4f}")

    banner("WALK-FORWARD FOLDS")
    folds = walk_forward_folds(ds.entry_dates, ds.entry_dates[0] + timedelta(days=4 * 365),
                               ds.entry_dates[-1])
    for r in describe_folds(ds.entry_dates, folds):
        print(f"  fold {r['fold']}  train {r['train_n']:5,} "
              f"({r['train_start'].date()}..{r['train_end'].date()})   "
              f"test {r['test_n']:5,} ({r['test_start'].date()}..{r['test_end'].date()})  "
              f"gap {r['gap_days']}d")

    models = {
        "no_trade":          lambda X, yy: ConstantModel(0.0),
        "always_trade":      lambda X, yy: ConstantModel(1.0),
        "simple_rule":       lambda X, yy: train_simple_rule(X, yy),
        "logistic":          train_logistic,
        "gradient_boosting": train_gradient_boosting,
    }

    banner("RESULTS  (walk-forward, out of sample, after costs)")
    print(f"  {'model':<19}{'trades':>8}{'AUC':>8}{'Brier':>9}{'ECE':>8}"
          f"{'win%':>8}{'net P&L':>12}{'per trade':>11}")
    print("  " + "-" * 72)

    results = {}
    for name, fn in models.items():
        res = run_walk_forward(ds, folds, model_fn=fn, commission_per_contract=COMMISSION,
                               risk_config=RiskConfig(max_trades_per_day=99),
                               min_trades_for_threshold=50, model_version=name)
        results[name] = res
        pnl = res.ledger.pnl_series()
        tr = trading_report(pnl)
        aucs = [f.classification.get("roc_auc", float("nan")) for f in res.folds]
        auc = float(np.nanmean(aucs)) if aucs else float("nan")
        briers = [f.classification.get("brier", float("nan")) for f in res.folds]
        eces = [f.classification.get("ece", float("nan")) for f in res.folds]
        n = len(res.ledger)
        print(f"  {name:<19}{n:>8,}{auc:>8.3f}{np.nanmean(briers):>9.4f}"
              f"{np.nanmean(eces):>8.4f}"
              f"{(tr['win_rate']*100 if n else float('nan')):>7.1f}%"
              f"{tr['total_pnl']:>12,.0f}"
              f"{(tr['avg_pnl'] if n else 0):>11,.2f}")

    banner("CALIBRATION AUDIT  (read this before the AUC)")
    for name in ("logistic", "gradient_boosting"):
        res = results[name]
        if len(res.labels) < 30:
            print(f"\n  {name}: too few trades to audit ({len(res.labels)})")
            continue
        print(f"\n  {name}")
        print(f"  {'bucket':<13}{'n':>7}{'predicted':>12}{'actual':>10}{'gap':>9}")
        for r in by_prob_bucket(res.probabilities, res.labels, res.ledger.pnl_series()):
            flag = "  <-- miscalibrated" if abs(r["calibration_gap"]) > 0.10 else ""
            print(f"  {r['bucket']:<13}{r['n']:>7}{r['mean_predicted']:>12.3f}"
                  f"{r['actual_win_rate']:>10.3f}{r['calibration_gap']:>+9.3f}{flag}")

    # The engine above applies the full decision layer, so a model that finds no
    # profitable threshold correctly trades nothing. That is the right answer for
    # deployment but hides whether any SIGNAL exists underneath, so score the
    # models directly on every out-of-sample row as well.
    banner("PURE OUT-OF-SAMPLE SIGNAL  (no trading gate)")
    pooled, pooled_idx = pool_predictions(ds, folds, models)
    print(f"  {'model':<19}{'n':>8}{'AUC':>8}{'base':>8}{'top-decile win':>16}"
          f"{'vs break-even':>15}")
    print("  " + "-" * 72)
    for name, (p, lab) in pooled.items():
        if len(np.unique(lab)) < 2:
            continue
        cut = np.quantile(p, 0.9)
        top = lab[p >= cut]
        verdict = "CLEARS" if len(top) and top.mean() > be else "below"
        print(f"  {name:<19}{len(lab):>8,}{roc_auc(lab, p):>8.3f}{lab.mean():>8.3f}"
              f"{(top.mean() if len(top) else float('nan')):>15.1%}"
              f"{verdict:>15}")

    banner("THE QUESTION: does ANY subset clear break-even?")
    for name in ("logistic", "gradient_boosting", "simple_rule"):
        p, lab = pooled[name]
        if len(np.unique(lab)) < 2:
            continue
        print("")
        print(f"  {name}  ({len(lab):,} out-of-sample trades, break-even {be:.1%})")
        print(f"  {'selection':<26}{'n':>7}{'win rate':>11}{'':>4}")
        for q in (0.0, 0.5, 0.75, 0.9, 0.95, 0.99):
            cut = np.quantile(p, q)
            m = p >= cut
            if m.sum() >= 20:
                wr = lab[m].mean()
                mark = "  <-- clears" if wr > be else ""
                label = "everything" if q == 0 else f"top {(1-q)*100:.0f}% (p>={cut:.3f})"
                print(f"  {label:<26}{int(m.sum()):>7}{wr:>11.1%}{mark}")

    banner("IS ANY OF THAT REAL?  (significance and costs)")
    significance_and_costs(ds, pooled_idx, pooled["logistic"], be)
    return 0


def significance_and_costs(ds, idx, pooled_logistic, be) -> None:
    """A win rate above break-even means nothing until it survives two tests:
    is it distinguishable from chance, and does it clear the commission?"""
    from scipy import stats as st

    from backtest.costs import roundtrip_fees
    from schemas import CONTRACT_MULTIPLIER

    p, y = pooled_logistic
    print(f"  {'slice':<14}{'n':>6}{'win':>8}{'95% CI':>18}{'p(win>BE)':>11}   verdict")
    for q in (0.75, 0.90, 0.95, 0.99):
        m = p >= np.quantile(p, q)
        n, k = int(m.sum()), int(y[m].sum())
        if n < 20:
            continue
        lo, hi = st.beta.ppf([.025, .975], k + .5, n - k + .5)
        pv = st.binomtest(k, n, be, alternative="greater").pvalue
        verdict = "significant" if pv < 0.05 else "not distinguishable from noise"
        print(f"  top {(1-q)*100:>3.0f}%{'':<6}{n:>6}{k/n:>8.1%}   "
              f"[{lo:.3f}, {hi:.3f}]{pv:>11.3f}   {verdict}")

    m = p >= np.quantile(p, 0.95)
    outs = [ds.outcomes[i] for i in idx[m]]
    gross = np.array([o.net_pnl for o in outs]) * CONTRACT_MULTIPLIER
    fees = roundtrip_fees(0.65)
    print("")
    print(f"  Best slice (top 5%, {len(outs)} trades, win {y[m].mean():.1%}) "
          "after commission:")
    print(f"    gross ${gross.mean():+.2f}/trade   commission ${fees:.2f}   "
          f"NET ${gross.mean()-fees:+.2f}/trade")
    print(f"    commission is {fees/max(abs(gross.mean()),1e-9)*100:.0f}% of the gross edge")


def pool_predictions(ds, folds, models) -> dict:
    """Train per fold, predict its test rows, pool all out-of-sample predictions.

    Mirrors the engine's fit order -- scaler fit on the training fold only, then
    frozen -- but skips threshold selection and risk gating so the raw signal is
    visible.
    """
    from features.scalers import FitScaler
    from models.baselines import predict_proba

    out = {name: ([], []) for name in models}
    test_order = []
    for train_idx, test_idx in folds:
        Xtr, ytr = ds.X[train_idx], ds.y[train_idx]
        Xte, yte = ds.X[test_idx], ds.y[test_idx]
        if len(np.unique(ytr)) < 2:
            continue
        test_order.append(np.asarray(test_idx))
        scaler = FitScaler().fit(Xtr).freeze()
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        for name, fn in models.items():
            model = fn(Xtr_s, ytr)
            out[name][0].append(predict_proba(model, Xte_s))
            out[name][1].append(yte)
    pooled = {k: (np.concatenate(v[0]), np.concatenate(v[1]))
              for k, v in out.items() if v[0]}
    return pooled, np.concatenate(test_order)


if __name__ == "__main__":
    raise SystemExit(main())
