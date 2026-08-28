"""Canonical data schemas and strategy constants.

Every module downstream of the adapters speaks only these types. Nothing here
knows which vendor produced the data.
"""
from dataclasses import dataclass, field, replace
from datetime import date, datetime

# ---------------------------------------------------------------- strategy
SPREAD_WIDTH = 5.0
MIN_DTE = 30
MAX_DTE = 45
TARGET_DELTA = -0.20
PROFIT_TARGET_FRAC = 0.50      # close when cost <= 50% of credit received
STOP_LOSS_FRAC = 2.0           # close when cost >= 200% of credit received
TIME_EXIT_DTE = 7              # forced exit at 7 days to expiry
DECISION_HOUR_ET = 15
DECISION_MINUTE_ET = 45
CONTRACT_MULTIPLIER = 100      # 1 option contract == 100 shares
LEGS_PER_SPREAD = 2            # short put + long put

# A trade entered at MAX_DTE and held to the forced time exit lives this long.
# The walk-forward embargo must be at least this wide or labels leak across folds.
MAX_HOLD_DAYS = MAX_DTE - TIME_EXIT_DTE   # == 38
EMBARGO_DAYS = MAX_HOLD_DAYS

EXIT_REASONS = ("PROFIT_TARGET", "STOP_LOSS", "TIME_EXIT", "EXPIRY", "NO_DATA")


@dataclass(frozen=True)
class OptionQuote:
    timestamp: datetime
    symbol: str
    expiry: date
    dte: int
    strike: float
    option_type: str               # "C" or "P"
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    underlying_price: float
    quality_flags: tuple = ()

    @property
    def is_clean(self) -> bool:
        return len(self.quality_flags) == 0


@dataclass(frozen=True)
class UnderlyingBar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


@dataclass(frozen=True)
class Candidate:
    """One short-put-spread entry considered at one decision time.

    `rejection_reason == ""` means accepted. Rejected candidates are kept so the
    filter behaviour is auditable rather than invisible.
    """
    entry_time: datetime
    short_strike: float
    long_strike: float
    expiry: date
    dte: int
    entry_credit: float
    width: float
    underlying_price: float
    # short leg
    short_delta: float
    short_iv: float
    short_bid: float
    short_ask: float
    short_open_interest: int = 0
    short_volume: int = 0
    short_theta: float = 0.0
    short_vega: float = 0.0
    short_gamma: float = 0.0
    # long leg
    long_bid: float = 0.0
    long_ask: float = 0.0
    long_delta: float = 0.0
    long_iv: float = 0.0
    long_open_interest: int = 0
    long_volume: int = 0
    long_theta: float = 0.0
    long_vega: float = 0.0
    long_gamma: float = 0.0
    rejection_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.rejection_reason == ""

    @property
    def max_risk(self) -> float:
        """Capital genuinely at risk: width minus the credit collected.

        NOT the width. A $5 spread sold for $2 risks $3, so dividing P&L by the
        width overstates return on risk by ~40%.
        """
        return max(self.width - self.entry_credit, 1e-9)

    @property
    def short_mid(self) -> float:
        return (self.short_bid + self.short_ask) / 2.0

    @property
    def long_mid(self) -> float:
        return (self.long_bid + self.long_ask) / 2.0

    @property
    def break_even(self) -> float:
        return self.short_strike - self.entry_credit


@dataclass(frozen=True)
class TradeOutcome:
    candidate: Candidate
    exit_time: datetime
    exit_reason: str               # one of EXIT_REASONS
    exit_debit: float              # per-share cost paid to close the spread
    label_win: bool                # primary NN target: reached PROFIT_TARGET
    gross_pnl: float               # per-share, before costs
    fees: float
    slippage: float
    net_pnl: float                 # per-share, after costs
    final_return_on_risk: float    # net_pnl / candidate.max_risk
    max_adverse_excursion: float   # worst mark-to-market loss, per share (>= 0)
    max_favorable_excursion: float # best mark-to-market gain, per share (>= 0)
    days_held: float
    n_marks: int                   # how many snapshots the walk actually saw
    # Marks whose close cost violated the [0, width] no-arbitrage bound and were
    # clamped. Non-zero means the source data contains impossible quotes.
    n_clamped_marks: int = 0

    @property
    def net_pnl_dollars(self) -> float:
        return self.net_pnl * CONTRACT_MULTIPLIER


def with_flags(quote: OptionQuote, flags) -> OptionQuote:
    """Return a copy of `quote` carrying `flags` (deduplicated, order-stable)."""
    merged = tuple(dict.fromkeys(tuple(quote.quality_flags) + tuple(flags)))
    return replace(quote, quality_flags=merged)
