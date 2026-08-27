"""Label extraction and training-set filtering."""
from __future__ import annotations

from collections import Counter

from schemas import TradeOutcome


def outcome_label(outcome: TradeOutcome) -> int:
    """Primary NN target: did the spread reach its profit target first?"""
    return 1 if outcome.exit_reason == "PROFIT_TARGET" else 0


def is_labelable(outcome: TradeOutcome) -> bool:
    """NO_DATA trades have no true label and must never enter training."""
    return outcome.exit_reason != "NO_DATA"


def labelable(outcomes) -> list[TradeOutcome]:
    return [o for o in outcomes if is_labelable(o)]


def label_distribution(outcomes) -> dict:
    """Exit-reason mix plus base rate -- check this before trusting any AUC.

    If the base rate is 0.85, a model that always says "win" scores 85% accuracy
    and is worthless. The base rate is the number every metric is read against.
    """
    lab = labelable(outcomes)
    counts: dict = dict(Counter(o.exit_reason for o in outcomes))
    counts["_labelable"] = len(lab)
    counts["_dropped_no_data"] = len(outcomes) - len(lab)
    counts["_base_rate"] = (
        round(sum(outcome_label(o) for o in lab) / len(lab), 4) if lab else 0.0
    )
    return counts
