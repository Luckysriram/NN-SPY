"""Hard risk rules. The model never edits these.

Everything here is a deterministic function of the config and the portfolio
state. No learned parameter, no threshold the optimiser can move. If a rule
blocks a trade the model liked, the rule wins.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from backtest.costs import max_risk_dollars


@dataclass(frozen=True)
class RiskConfig:
    account_size: float = 100_000.0
    max_loss_per_trade_pct: float = 0.01
    max_open_risk_pct: float = 0.05
    max_trades_per_day: int = 2
    kill_daily_loss_pct: float = 0.03
    kill_weekly_loss_pct: float = 0.06
    require_clean_data: bool = True
    paper_trade_only: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "RiskConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (cfg or {}).items() if k in known})


def _as_date(d) -> date:
    return d.date() if isinstance(d, datetime) else d


@dataclass
class Portfolio:
    """Open risk and realised P&L, both in dollars."""
    account_size: float = 100_000.0
    open_risk: float = 0.0
    pnl_by_day: dict = field(default_factory=lambda: defaultdict(float))
    trades_by_day: dict = field(default_factory=lambda: defaultdict(int))
    open_positions: list = field(default_factory=list)

    # -- mutation -------------------------------------------------------
    def add_open_risk(self, amount: float) -> None:
        self.open_risk += amount

    def release_open_risk(self, amount: float) -> None:
        self.open_risk = max(0.0, self.open_risk - amount)

    def record_entry(self, day, risk_dollars: float) -> None:
        self.trades_by_day[_as_date(day)] += 1
        self.add_open_risk(risk_dollars)

    def add_pnl(self, day, pnl_dollars: float) -> None:
        self.pnl_by_day[_as_date(day)] += pnl_dollars

    def close_trade(self, day, pnl_dollars: float, risk_dollars: float) -> None:
        self.add_pnl(day, pnl_dollars)
        self.release_open_risk(risk_dollars)

    # -- queries --------------------------------------------------------
    @property
    def open_risk_pct(self) -> float:
        return self.open_risk / self.account_size if self.account_size else 0.0

    def trades_on(self, day) -> int:
        return self.trades_by_day.get(_as_date(day), 0)

    def daily_pnl(self, day) -> float:
        return self.pnl_by_day.get(_as_date(day), 0.0)

    def daily_pnl_pct(self, day) -> float:
        return self.daily_pnl(day) / self.account_size if self.account_size else 0.0

    def weekly_pnl(self, day) -> float:
        """Trailing 7 calendar days ending on `day`."""
        end = _as_date(day)
        start = end - timedelta(days=6)
        return sum(v for d, v in self.pnl_by_day.items() if start <= d <= end)

    def weekly_pnl_pct(self, day) -> float:
        return self.weekly_pnl(day) / self.account_size if self.account_size else 0.0

    def total_pnl(self) -> float:
        return sum(self.pnl_by_day.values())


def position_size(candidate, config: RiskConfig) -> int:
    """Largest contract count whose defined risk fits the per-trade cap."""
    per_contract = max_risk_dollars(candidate.width, candidate.entry_credit, 1)
    if per_contract <= 0:
        return 0
    cap = config.account_size * config.max_loss_per_trade_pct
    return int(cap // per_contract)


def evaluate_candidate(candidate, proba: float, expected_value: float,
                       portfolio: Portfolio, config: RiskConfig, *,
                       decision_time=None, data_is_clean: bool = True,
                       contracts: int | None = None):
    """Return (allow, reasons). `reasons` lists every rule that blocked it."""
    reasons: list[str] = []

    if expected_value is None or not (expected_value > 0):
        reasons.append("expected_value")
    if proba is None or not (0.0 < proba <= 1.0):
        reasons.append("invalid_probability")
    if config.require_clean_data and not data_is_clean:
        reasons.append("data_quality")

    if portfolio.open_risk_pct >= config.max_open_risk_pct:
        reasons.append("max_open_risk")

    day = decision_time or (candidate.entry_time if candidate is not None else None)
    if day is not None and portfolio.trades_on(day) >= config.max_trades_per_day:
        reasons.append("max_trades_per_day")

    if candidate is not None:
        n = position_size(candidate, config) if contracts is None else contracts
        if n < 1:
            reasons.append("position_too_large_for_risk_cap")
        else:
            risk = max_risk_dollars(candidate.width, candidate.entry_credit, n)
            projected = (portfolio.open_risk + risk) / config.account_size
            if projected > config.max_open_risk_pct:
                reasons.append("max_open_risk")
        if candidate.long_strike >= candidate.short_strike:
            reasons.append("undefined_risk")          # never sell a naked put
        if not candidate.accepted:
            reasons.append(f"candidate_{candidate.rejection_reason}")

    return (not reasons), (reasons or ["passes_risk"])


def should_kill(portfolio: Portfolio, config: RiskConfig, day=None):
    """Kill switch. Returns (kill, reason)."""
    days = [_as_date(day)] if day is not None else list(portfolio.pnl_by_day)
    for d in days:
        if portfolio.daily_pnl_pct(d) <= -config.kill_daily_loss_pct:
            return True, "daily_loss_limit"
        if portfolio.weekly_pnl_pct(d) <= -config.kill_weekly_loss_pct:
            return True, "weekly_loss_limit"
    return False, ""


def feature_drift_kill(train_mean, train_std, live_vector, n_sigma: float = 4.0):
    """Halt when a live feature sits absurdly far from its training distribution.

    A model asked to score an input unlike anything it trained on returns a
    confident number with no basis. Better to stop than to trade it.
    """
    import numpy as np
    tm, ts = np.asarray(train_mean, float), np.asarray(train_std, float)
    x = np.asarray(live_vector, float)

    # A zero-variance training column has no scale to measure against, so a
    # z-score there is meaningless. The feature was constant in training, which
    # makes ANY different live value out-of-distribution -- treat it as drift on
    # its own terms rather than dividing by a made-up standard deviation.
    constant = ts <= 0
    if np.any(constant & ~np.isnan(x) & (np.abs(x - tm) > 1e-9)):
        idx = int(np.argmax(constant & ~np.isnan(x) & (np.abs(x - tm) > 1e-9)))
        return True, f"feature_drift_index_{idx}_constant_in_training"

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs((x - tm) / np.where(constant, np.nan, ts))
    z = np.where(np.isnan(z), 0.0, z)
    worst = int(np.argmax(z))
    if z[worst] > n_sigma:
        return True, f"feature_drift_index_{worst}_z_{z[worst]:.1f}"
    return False, ""
