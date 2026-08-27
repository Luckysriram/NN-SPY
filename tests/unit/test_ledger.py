import math
import tempfile
from pathlib import Path

import pytest

from backtest.ledger import LEDGER_COLUMNS, TradeLedger
from schemas import TradeOutcome
from tests.conftest import TS, candidate


def outcome(win=True, credit=2.0):
    c = candidate(entry_credit=credit)
    debit = 1.0 if win else 4.0
    gross = credit - debit
    return TradeOutcome(
        candidate=c, exit_time=TS, exit_reason="PROFIT_TARGET" if win else "STOP_LOSS",
        exit_debit=debit, label_win=win, gross_pnl=gross, fees=0.0, slippage=0.05,
        net_pnl=gross, final_return_on_risk=gross / c.max_risk,
        max_adverse_excursion=0.1, max_favorable_excursion=0.3,
        days_held=3.0, n_marks=3,
    )


def test_ledger_records_and_sums_pnl():
    """total_pnl sums the rows it actually holds -- it does not re-derive P&L."""
    ledger = TradeLedger()
    a = ledger.record(outcome(True), commission_per_contract=0.65)
    b = ledger.record(outcome(False), commission_per_contract=0.65)
    assert len(ledger) == 2
    assert ledger.total_pnl() == pytest.approx(a.net_pnl + b.net_pnl)


def test_recorded_pnl_matches_the_cost_model():
    row = TradeLedger().record(outcome(True), commission_per_contract=0.65)
    assert row.net_pnl == pytest.approx((2.0 - 1.0) * 100 - 2.60)
    assert row.fees == pytest.approx(2.60)


def test_return_on_risk_uses_dollar_risk():
    row = TradeLedger().record(outcome(True), commission_per_contract=0.0)
    assert row.max_risk_dollars == pytest.approx(300.0)
    assert row.return_on_risk == pytest.approx(100.0 / 300.0)


def test_gross_minus_fees_equals_net():
    row = TradeLedger().record(outcome(False), commission_per_contract=0.65)
    assert row.gross_pnl - row.fees == pytest.approx(row.net_pnl)


def test_ledger_refuses_to_record_a_no_data_outcome():
    """An unlabelable trade is not a trade."""
    bad = TradeOutcome(
        candidate=candidate(), exit_time=TS, exit_reason="NO_DATA",
        exit_debit=float("nan"), label_win=False, gross_pnl=float("nan"),
        fees=0.0, slippage=0.0, net_pnl=float("nan"),
        final_return_on_risk=float("nan"), max_adverse_excursion=float("nan"),
        max_favorable_excursion=float("nan"), days_held=0.0, n_marks=0,
    )
    with pytest.raises(ValueError, match="NO_DATA"):
        TradeLedger().record(bad)


def test_ledger_carries_the_spec_11_fields():
    ledger = TradeLedger(model_version="mlp_v1")
    row = ledger.record(outcome(True), model_probability=0.71, fold=2)
    for field in ("entry_time", "exit_time", "entry_credit", "exit_debit",
                  "gross_pnl", "fees", "slippage", "net_pnl",
                  "max_adverse_excursion", "max_favorable_excursion",
                  "exit_reason", "model_probability", "feature_version",
                  "model_version"):
        assert hasattr(row, field), field
    assert row.model_version == "mlp_v1"
    assert row.model_probability == 0.71
    assert row.fold == 2


def test_ledger_accessors():
    ledger = TradeLedger()
    ledger.record(outcome(True), model_probability=0.8)
    ledger.record(outcome(False), model_probability=0.3)
    assert ledger.labels() == [1, 0]
    assert ledger.probabilities() == [0.8, 0.3]
    assert len(ledger.pnl_series()) == 2


def test_ledger_writes_csv_with_every_column():
    ledger = TradeLedger()
    ledger.record(outcome(True))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ledger.csv"
        ledger.write_csv(p)
        header = p.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == LEDGER_COLUMNS
