"""The daily decision record, in the exact JSON shape from spec section 9.4.

Every decision is logged, including NO_TRADE. A file that only records the days
you traded cannot answer "how often did the model stand down, and was it right
to?" -- which is the question the spec's safety criterion is about.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ET_OFFSET = "-04:00"          # EDT. See note in `_stamp` about DST.


def _stamp(dt: datetime) -> str:
    """ISO timestamp with an explicit offset.

    If the datetime is already timezone-aware its real offset is used. Naive
    datetimes are assumed to be Eastern and stamped -04:00, which is EDT and
    therefore wrong by an hour from November to March -- attach a real tzinfo
    (zoneinfo "America/New_York") upstream if that matters to you.
    """
    if dt.tzinfo is not None:
        return dt.isoformat()
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + ET_OFFSET


def daily_signal_json(candidate, proba: float, expected_value: float,
                      decision: str, reasons) -> dict:
    """Build the decision record. `decision` is PAPER_TRADE or NO_TRADE."""
    if decision not in ("PAPER_TRADE", "NO_TRADE"):
        raise ValueError(f"decision must be PAPER_TRADE or NO_TRADE, got {decision!r}")
    return {
        "timestamp": _stamp(candidate.entry_time),
        "strategy": "SPY put credit spread",
        "short_strike": _clean_strike(candidate.short_strike),
        "long_strike": _clean_strike(candidate.long_strike),
        "expiration": candidate.expiry.isoformat(),
        "dte": int(candidate.dte),
        "credit": round(float(candidate.entry_credit), 2),
        "max_risk": round(float(candidate.max_risk), 2),
        "model_probability": round(float(proba), 4),
        "expected_value": round(float(expected_value), 2),
        "decision": decision,
        "reasons": list(reasons),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Research output. Paper trading only. Not a trade recommendation.",
    }


def _clean_strike(x: float):
    """Keep 640 as an int and 640.5 as a float rather than truncating to 640."""
    return int(x) if float(x).is_integer() else round(float(x), 2)


def append_decision_log(record: dict, path="experiments/decisions.jsonl") -> None:
    """Append one decision as JSON Lines. Creates the directory if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
