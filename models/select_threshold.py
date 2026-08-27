"""Choose the probability threshold, on validation only, then freeze it.

The economics this has to respect
--------------------------------
A $5-wide spread sold for $2.00 credit:
  win  -> close at 50% of credit  -> +$1.00
  loss -> close at 2x credit      -> -$2.00

Profit and loss are NOT symmetric, so the break-even win rate is
loss/(profit+loss) = 2/(1+2) = 66.7%, not 50%. A threshold search that assumes
1:1 payoffs will happily select a 55%-win-rate bucket and call it profitable.

`select_threshold` also enforces a minimum trade count. Without it the search
maximises a statistic computed on the two luckiest trades in validation.
"""
from __future__ import annotations

import numpy as np

from schemas import PROFIT_TARGET_FRAC, STOP_LOSS_FRAC


def expected_value(p_win: float, expected_profit: float, expected_loss: float,
                   costs: float = 0.0) -> float:
    """p*profit - (1-p)*loss - costs, per the spec."""
    return p_win * expected_profit - (1.0 - p_win) * expected_loss - costs


def spread_payoffs(entry_credit: float) -> tuple:
    """(profit_if_target_hit, loss_if_stopped) per share, from the exit rules."""
    return entry_credit * PROFIT_TARGET_FRAC, entry_credit * (STOP_LOSS_FRAC - 1.0)


def breakeven_win_rate(expected_profit: float, expected_loss: float,
                       costs: float = 0.0) -> float:
    """Win rate at which expected value is exactly zero."""
    denom = expected_profit + expected_loss
    if denom <= 0:
        return float("nan")
    return (expected_loss + costs) / denom


def threshold_table(proba, y, pnl=None, *, expected_profit: float = 1.0,
                    expected_loss: float = 1.0, costs: float = 0.0,
                    grid=None) -> list:
    """Per-threshold diagnostics on validation. Inspect before trusting a pick."""
    p = np.asarray(proba, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    pnl = None if pnl is None else np.asarray(pnl, dtype=float).ravel()
    grid = np.arange(0.05, 0.96, 0.01) if grid is None else np.asarray(grid, dtype=float)

    rows = []
    for t in grid:
        m = p >= t
        n = int(m.sum())
        if n == 0:
            continue
        wr = float(y[m].mean())
        ev = expected_value(wr, expected_profit, expected_loss, costs)
        rows.append({
            "threshold": float(t), "n_trades": n, "win_rate": wr,
            "ev_per_trade": ev, "total_ev": ev * n,
            "realized_pnl": float(np.nansum(pnl[m])) if pnl is not None else float("nan"),
            "realized_avg": float(np.nanmean(pnl[m])) if pnl is not None else float("nan"),
        })
    return rows


def select_threshold(proba, y, *, pnl=None, costs: float = 0.0,
                     expected_profit: float = 1.0, expected_loss: float = 1.0,
                     min_trades: int = 30, ev_min: float = 0.0,
                     objective: str = "total") -> float:
    """Pick the threshold maximising validation profit. Returns 1.01 to never trade.

    objective="total"    -- total expected (or realized) profit across selected trades
    objective="per_trade"-- expected value per trade

    "total" is the default because a per-trade objective rewards trading almost
    nothing at a very high threshold, which is not a usable strategy.
    """
    rows = threshold_table(proba, y, pnl, expected_profit=expected_profit,
                           expected_loss=expected_loss, costs=costs)
    eligible = [r for r in rows if r["n_trades"] >= min_trades]
    if not eligible:
        return 1.01                       # nothing passes the sample-size floor

    use_realized = pnl is not None
    if use_realized:
        key = "realized_pnl" if objective == "total" else "realized_avg"
    else:
        key = "total_ev" if objective == "total" else "ev_per_trade"

    best = max(eligible, key=lambda r: (r[key], -r["threshold"]))
    gate = best["realized_avg"] if use_realized else best["ev_per_trade"]
    if not (gate > ev_min):
        return 1.01                       # best available option is still unprofitable
    return float(best["threshold"])


def decide(proba: float, threshold: float, ev: float, *, ev_min: float = 0.0) -> tuple:
    """Final gate: probability AND expected value must both clear. Returns (bool, reasons)."""
    reasons = []
    if proba < threshold:
        reasons.append("below_threshold")
    if not (ev > ev_min):
        reasons.append("expected_value")
    return (not reasons), (reasons or ["passes probability threshold"])
