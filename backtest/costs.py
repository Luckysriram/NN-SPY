"""Transaction costs.

A vertical spread is TWO legs, and a round trip opens and closes both, so one
spread traded once pays commission on four contract-legs. Getting this wrong by
a factor of two is enough to flip a marginal strategy from profitable to not.
"""
from __future__ import annotations

from schemas import CONTRACT_MULTIPLIER, LEGS_PER_SPREAD

DEFAULT_COMMISSION_PER_CONTRACT = 0.65
DEFAULT_EXCHANGE_FEE_PER_CONTRACT = 0.05     # ORF/exchange fees, per contract-leg


def roundtrip_fees(commission_per_contract: float, contracts: int = 1,
                   legs: int = LEGS_PER_SPREAD,
                   exchange_fee_per_contract: float = 0.0) -> float:
    """Total commission for opening AND closing `contracts` spreads, in dollars.

    contract-legs = contracts x legs x 2 (open + close)
    """
    contract_legs = contracts * legs * 2
    return contract_legs * (commission_per_contract + exchange_fee_per_contract)


def apply_costs(entry_credit: float, exit_debit: float,
                commission_per_contract: float, contracts: int = 1,
                legs: int = LEGS_PER_SPREAD,
                exchange_fee_per_contract: float = 0.0):
    """Return (fees_dollars, net_pnl_dollars) for a completed spread trade.

    `entry_credit` and `exit_debit` are per-share; P&L is scaled by the contract
    multiplier (100) so the result is real money the risk engine can size against.
    """
    fees = roundtrip_fees(commission_per_contract, contracts, legs,
                          exchange_fee_per_contract)
    gross = (entry_credit - exit_debit) * CONTRACT_MULTIPLIER * contracts
    return fees, gross - fees


def cost_per_spread_per_share(commission_per_contract: float,
                              exchange_fee_per_contract: float = 0.0,
                              legs: int = LEGS_PER_SPREAD) -> float:
    """Round-trip fees expressed per share, to compare against a credit."""
    return roundtrip_fees(commission_per_contract, 1, legs,
                          exchange_fee_per_contract) / CONTRACT_MULTIPLIER


def max_risk_dollars(width: float, entry_credit: float, contracts: int = 1) -> float:
    """Defined risk of the position, in dollars."""
    return max(width - entry_credit, 0.0) * CONTRACT_MULTIPLIER * contracts
