"""Periodic report: calibration audit plus the regime breakdowns from spec 11.

The calibration audit is the headline. Grouping trades by predicted probability
and comparing predicted to realised frequency answers the one question that
decides whether the model is usable: when it says 70%, do 70% of them win?
"""
from __future__ import annotations

import numpy as np

from reports.metrics import (calibration_bins, classification_report,
                             expected_calibration_error, trading_report)

PROB_BUCKETS = [(0.0, 0.5), (0.5, 0.55), (0.55, 0.60), (0.60, 0.65),
                (0.65, 0.70), (0.70, 0.75), (0.75, 1.01)]


def by_prob_bucket(proba, y, pnl=None) -> list:
    """Realised win rate per predicted-probability bucket -- the audit table.

    A bucket whose predicted probability is 0.70 and whose realised win rate is
    0.52 is the spec's stated failure case, and `calibration_gap` names it.
    """
    p, y = np.asarray(proba, float), np.asarray(y, float)
    pnl = None if pnl is None else np.asarray(pnl, float)
    rows = []
    for lo, hi in PROB_BUCKETS:
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        predicted, actual = float(p[m].mean()), float(y[m].mean())
        rows.append({
            "bucket": f"{lo:.2f}-{min(hi,1.0):.2f}",
            "n": int(m.sum()),
            "mean_predicted": round(predicted, 4),
            "actual_win_rate": round(actual, 4),
            "calibration_gap": round(actual - predicted, 4),
            "total_pnl": (round(float(np.nansum(pnl[m])), 2)
                          if pnl is not None else None),
        })
    return rows


def by_group(values, pnl, y=None, labels=None) -> list:
    """Generic breakdown: group trades by a per-trade key and report each group."""
    values = list(values)
    pnl = np.asarray(pnl, float)
    groups: dict = {}
    for i, v in enumerate(values):
        groups.setdefault(v, []).append(i)
    rows = []
    for key in sorted(groups, key=str):
        idx = groups[key]
        row = {"group": str(key), "n": len(idx), **trading_report(pnl[idx])}
        if y is not None:
            row["win_rate_label"] = float(np.asarray(y, float)[idx].mean())
        rows.append(row)
    return rows


def vix_bucket(vix: float) -> str:
    if np.isnan(vix):
        return "unknown"
    if vix < 15:
        return "low_vix"
    if vix < 25:
        return "med_vix"
    return "high_vix"


def dte_bucket(dte: float) -> str:
    return "30-35" if dte <= 35 else ("36-40" if dte <= 40 else "41-45")


def monthly_report(outcomes, proba, y, *, pnl=None, bins: int = 10,
                   groups: dict | None = None) -> dict:
    """Full report. `groups` maps a name to a per-trade key list, e.g. {"vix": [...]}."""
    proba, y = np.asarray(proba, float), np.asarray(y, float)
    pnl_arr = np.asarray(pnl, float) if pnl is not None else np.array([])

    report = {
        "trades": len(outcomes),
        "calibration": calibration_bins(y, proba, bins=bins),
        "by_prob_bucket": by_prob_bucket(proba, y, pnl),
        "ece": expected_calibration_error(y, proba, bins=bins),
        "classification": classification_report(y, proba),
        "trading": trading_report(pnl_arr) if pnl is not None else {},
        "breakdowns": {},
    }
    if groups and pnl is not None:
        for name, keys in groups.items():
            report["breakdowns"][name] = by_group(keys, pnl_arr, y)
    return report


def render(report: dict) -> str:
    """Plain-text rendering for the console and the experiment log."""
    L = ["=" * 66, "  WALK-FORWARD REPORT", "=" * 66,
         f"  trades: {report['trades']}"]
    c = report.get("classification") or {}
    if c:
        L.append(f"  base rate {c.get('base_rate', float('nan')):.3f} | "
                 f"ROC-AUC {c.get('roc_auc', float('nan')):.3f} | "
                 f"Brier {c.get('brier', float('nan')):.4f} | "
                 f"ECE {c.get('ece', float('nan')):.4f}")
    t = report.get("trading") or {}
    if t.get("n_trades"):
        L.append(f"  net P&L ${t['total_pnl']:,.2f} | win rate {t['win_rate']:.1%} | "
                 f"profit factor {t['profit_factor']:.2f} | "
                 f"max DD ${t['max_drawdown']:,.2f}")
    L += ["-" * 66, "  CALIBRATION AUDIT (predicted vs realised)",
          f"  {'bucket':<12}{'n':>6}{'predicted':>12}{'actual':>10}{'gap':>9}"]
    for r in report.get("by_prob_bucket", []):
        flag = "  <-- miscalibrated" if abs(r["calibration_gap"]) > 0.10 else ""
        L.append(f"  {r['bucket']:<12}{r['n']:>6}{r['mean_predicted']:>12.3f}"
                 f"{r['actual_win_rate']:>10.3f}{r['calibration_gap']:>+9.3f}{flag}")
    for name, rows in (report.get("breakdowns") or {}).items():
        L += ["-" * 66, f"  BY {name.upper()}",
              f"  {'group':<16}{'n':>6}{'net P&L':>12}{'win rate':>10}{'PF':>8}"]
        for r in rows:
            L.append(f"  {r['group']:<16}{r['n']:>6}{r['total_pnl']:>12,.0f}"
                     f"{r['win_rate']:>10.1%}{r['profit_factor']:>8.2f}")
    L.append("=" * 66)
    return "\n".join(L)
