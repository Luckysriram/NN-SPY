"""Manual audit of simulated trades -- spec sections 7 and 13.

The spec says: hand-check 50 randomly selected trades before trusting the
simulator. This renders exactly that sample in a readable form, with the
arithmetic spelled out so a wrong exit rule is visible by eye rather than
inferred from an aggregate.

    python -m labels.audit --n 50 --seed 42
"""
from __future__ import annotations

import argparse
import random

from schemas import PROFIT_TARGET_FRAC, STOP_LOSS_FRAC, TIME_EXIT_DTE


def render_trade(outcome, index: int = 0) -> str:
    """One trade, with every number needed to check the exit by hand."""
    c = outcome.candidate
    target = c.entry_credit * PROFIT_TARGET_FRAC
    stop = c.entry_credit * STOP_LOSS_FRAC
    L = [
        f"--- trade #{index} " + "-" * 50,
        f"  entry      {c.entry_time}  expiry {c.expiry} ({c.dte} DTE)",
        f"  structure  short {c.short_strike}P / long {c.long_strike}P  "
        f"width ${c.width:.2f}  underlying ${c.underlying_price:.2f}",
        f"  credit     ${c.entry_credit:.2f}   (short bid {c.short_bid:.2f} "
        f"- long ask {c.long_ask:.2f} - slippage)",
        f"  max risk   ${c.max_risk:.2f}  = width - credit",
        f"  rails      profit target <= ${target:.2f}   stop >= ${stop:.2f}   "
        f"time exit at {TIME_EXIT_DTE} DTE",
        f"  exit       {outcome.exit_reason} at {outcome.exit_time} "
        f"after {outcome.days_held:.1f} days ({outcome.n_marks} marks)",
        f"  exit debit ${outcome.exit_debit:.2f}",
        f"  P&L        gross ${outcome.gross_pnl:+.2f} - fees ${outcome.fees:.2f} "
        f"- slippage ${outcome.slippage:.2f} = net ${outcome.net_pnl:+.2f}",
        f"  return     {outcome.final_return_on_risk:+.1%} of ${c.max_risk:.2f} at risk",
        f"  MAE ${outcome.max_adverse_excursion:.2f}  MFE ${outcome.max_favorable_excursion:.2f}",
        f"  label_win  {outcome.label_win}",
    ]
    checks = consistency_checks(outcome)
    L.append("  checks     " + ("all consistent" if not checks else "FAILED: " + "; ".join(checks)))
    return "\n".join(L)


def consistency_checks(outcome) -> list:
    """Assertions a correct simulation must satisfy. Empty list means clean."""
    import math
    c = outcome.candidate
    problems = []
    if outcome.exit_reason == "NO_DATA":
        return problems
    target = c.entry_credit * PROFIT_TARGET_FRAC
    stop = c.entry_credit * STOP_LOSS_FRAC

    if outcome.label_win != (outcome.exit_reason == "PROFIT_TARGET"):
        problems.append("label_win disagrees with exit_reason")
    if outcome.exit_reason == "PROFIT_TARGET" and outcome.exit_debit > target + 1e-6:
        problems.append(f"exit_debit {outcome.exit_debit:.2f} > target {target:.2f}")
    if outcome.exit_reason == "STOP_LOSS" and outcome.exit_debit < stop - 1e-6:
        problems.append(f"exit_debit {outcome.exit_debit:.2f} < stop {stop:.2f}")
    if outcome.exit_time < c.entry_time:
        problems.append("exit before entry")
    if outcome.max_adverse_excursion < -1e-9 or outcome.max_favorable_excursion < -1e-9:
        problems.append("negative excursion")
    if not math.isnan(outcome.net_pnl) and outcome.net_pnl > c.entry_credit + 1e-6:
        problems.append("net P&L exceeds the credit received")
    # The floor is max_risk plus BOTH costs. Omitting slippage made this fire on
    # trades that were already correctly clamped to max loss -- the worst case is
    # -(width - credit) - slippage - fees, not -(width - credit) - fees.
    floor = -(c.max_risk + outcome.fees + outcome.slippage + 1e-6)
    if not math.isnan(outcome.net_pnl) and outcome.net_pnl < floor:
        problems.append(
            f"loss {outcome.net_pnl:.4f} exceeds defined risk floor {floor:.4f}")
    if outcome.n_clamped_marks:
        problems.append(
            f"{outcome.n_clamped_marks} mark(s) violated the [0, width] bound "
            "in the source data and were clamped")
    return problems


def audit_sample(outcomes, n: int = 50, seed: int = 42) -> dict:
    """Render `n` random trades and summarise any inconsistencies found."""
    pool = [o for o in outcomes if o.exit_reason != "NO_DATA"]
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    rendered = [render_trade(o, i) for i, o in enumerate(sample, start=1)]
    failures = {i: consistency_checks(o) for i, o in enumerate(sample, start=1)
                if consistency_checks(o)}
    return {
        "n_sampled": len(sample), "n_available": len(pool), "seed": seed,
        "text": "\n".join(rendered), "failures": failures,
        "clean": not failures,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Manual audit of simulated trades")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.parse_args(argv)
    print(__doc__)
    print("Load your simulated outcomes and call audit_sample(outcomes, n, seed).")
    print("Wire this to your generated dataset once data/ is populated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
