"""Per-trade ledger carrying every field spec section 11 asks for.

`total_pnl()` sums the actual recorded net P&L. It does not re-derive P&L from a
return-on-risk ratio -- doing that made the ledger's total disagree with the sum
of its own rows.
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime

from backtest.costs import apply_costs, max_risk_dollars
from schemas import TradeOutcome

LEDGER_COLUMNS = [
    "entry_time", "exit_time", "expiry", "dte", "short_strike", "long_strike",
    "width", "contracts", "entry_credit", "exit_debit", "gross_pnl", "fees",
    "slippage", "net_pnl", "return_on_risk", "max_risk_dollars",
    "max_adverse_excursion", "max_favorable_excursion", "days_held",
    "exit_reason", "label_win", "model_probability", "feature_version",
    "model_version", "fold",
]


@dataclass
class LedgerRow:
    entry_time: datetime
    exit_time: datetime
    expiry: object
    dte: int
    short_strike: float
    long_strike: float
    width: float
    contracts: int
    entry_credit: float
    exit_debit: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    return_on_risk: float
    max_risk_dollars: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    days_held: float
    exit_reason: str
    label_win: bool
    model_probability: float = float("nan")
    feature_version: str = ""
    model_version: str = ""
    fold: int = 0


@dataclass
class TradeLedger:
    """Append-only record of executed trades."""
    trades: list = field(default_factory=list)
    feature_version: str = "v1"
    model_version: str = ""

    def record(self, outcome: TradeOutcome, *, contracts: int = 1,
               commission_per_contract: float = 0.0,
               exchange_fee_per_contract: float = 0.0,
               model_probability: float = float("nan"),
               fold: int = 0) -> LedgerRow:
        """Record a completed trade, recomputing costs in dollars."""
        c = outcome.candidate
        if outcome.exit_reason == "NO_DATA":
            raise ValueError("refusing to record a NO_DATA outcome as a trade")

        fees, net = apply_costs(
            c.entry_credit, outcome.exit_debit, commission_per_contract,
            contracts=contracts, exchange_fee_per_contract=exchange_fee_per_contract,
        )
        risk = max_risk_dollars(c.width, c.entry_credit, contracts)
        row = LedgerRow(
            entry_time=c.entry_time, exit_time=outcome.exit_time, expiry=c.expiry,
            dte=c.dte, short_strike=c.short_strike, long_strike=c.long_strike,
            width=c.width, contracts=contracts, entry_credit=c.entry_credit,
            exit_debit=outcome.exit_debit,
            gross_pnl=net + fees, fees=fees, slippage=outcome.slippage,
            net_pnl=net, return_on_risk=(net / risk if risk > 0 else float("nan")),
            max_risk_dollars=risk,
            max_adverse_excursion=outcome.max_adverse_excursion,
            max_favorable_excursion=outcome.max_favorable_excursion,
            days_held=outcome.days_held, exit_reason=outcome.exit_reason,
            label_win=outcome.label_win, model_probability=model_probability,
            feature_version=self.feature_version, model_version=self.model_version,
            fold=fold,
        )
        self.trades.append(row)
        return row

    def __len__(self) -> int:
        return len(self.trades)

    def total_pnl(self) -> float:
        """Sum of recorded net P&L, in dollars."""
        return float(sum(t.net_pnl for t in self.trades if not math.isnan(t.net_pnl)))

    def total_fees(self) -> float:
        return float(sum(t.fees for t in self.trades))

    def pnl_series(self) -> list:
        return [t.net_pnl for t in self.trades]

    def probabilities(self) -> list:
        return [t.model_probability for t in self.trades]

    def labels(self) -> list:
        return [int(t.label_win) for t in self.trades]

    def to_rows(self) -> list:
        return [asdict(t) for t in self.trades]

    def write_csv(self, path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
            w.writeheader()
            for row in self.to_rows():
                w.writerow({k: row.get(k, "") for k in LEDGER_COLUMNS})
