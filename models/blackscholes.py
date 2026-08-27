"""Black-Scholes-Merton pricing, Greeks, and an implied-volatility solver.

Why this module exists
----------------------
The strategy is *defined* by a -0.20 delta short strike. A dataset without
Greeks cannot express it, and most free option data has no Greeks. This module
recovers them: solve implied vol from the mid price, then differentiate.

Three things to know before trusting its output
-----------------------------------------------

1. SPY OPTIONS ARE AMERICAN. Black-Scholes prices EUROPEAN options. For the
   out-of-the-money puts this strategy sells the early-exercise premium is
   small, but it grows as the short goes in-the-money -- exactly when a losing
   trade is being marked. Deltas computed here are therefore slightly wrong in
   the direction that matters most, and the error is largest for deep-ITM puts
   near expiry. `early_exercise_risk()` flags the region where it stops being
   negligible.

2. DIVIDENDS ARE NOT OPTIONAL. SPY yields roughly 1-2%. Setting q=0 biases put
   deltas by a few points, which silently shifts which strike gets sold every
   day for the entire backtest.

3. THE RATE MATTERS ACROSS YEARS. Short rates went from ~0% in 2021 to over 5%
   by 2023. A constant rate over a multi-year backtest is a real error, not a
   rounding one. See `data/rates.py`.

Conventions: `T` is in years. `theta` is returned PER CALENDAR DAY and `vega`
PER ONE VOLATILITY POINT (0.01), which is how both are quoted on a trading
screen and how the feature columns expect them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SQRT_2PI = math.sqrt(2.0 * math.pi)
MIN_VOL = 1e-6
MAX_VOL = 5.0            # 500% -- beyond this the quote is not a real option price
# Below this much time value, implied vol is not identifiable from the price.
MIN_TIME_VALUE = 1e-8
# Below this vega, the price barely responds to sigma and the residual is
# uninformative; bisect instead of trusting a Newton step.
VEGA_FLOOR = 1e-10
DAYS_PER_YEAR = 365.0


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def intrinsic(S: float, K: float, option_type: str) -> float:
    return max(K - S, 0.0) if option_type == "P" else max(S - K, 0.0)


# --------------------------------------------------------------------- price
def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
             option_type: str = "P") -> float:
    """Black-Scholes-Merton price with a continuous dividend yield `q`."""
    if option_type not in ("P", "C"):
        raise ValueError(f"option_type must be 'P' or 'C', got {option_type!r}")
    if S <= 0 or K <= 0:
        return float("nan")
    if T <= 0 or sigma <= 0:
        return intrinsic(S, K, option_type)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if option_type == "P":
        return K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1)
    return S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2)


def price_bounds(S: float, K: float, T: float, r: float, q: float,
                 option_type: str = "P"):
    """(lower, upper) no-arbitrage bounds for a European option.

    The price is strictly increasing in sigma between these, which is what makes
    the implied-vol root unique and bisection guaranteed to converge.
    """
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if option_type == "P":
        return max(K * df_r - S * df_q, 0.0), K * df_r
    return max(S * df_q - K * df_r, 0.0), S * df_q


# -------------------------------------------------------------------- greeks
@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float      # per calendar day
    vega: float       # per 1 vol point (0.01)
    rho: float        # per 1 rate point (0.01)

    def as_dict(self) -> dict:
        return {"price": self.price, "delta": self.delta, "gamma": self.gamma,
                "theta": self.theta, "vega": self.vega, "rho": self.rho}


def bs_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float,
              option_type: str = "P") -> Greeks:
    """Analytic Greeks. Returns zeros for an expired or zero-vol option."""
    nan = float("nan")
    if S <= 0 or K <= 0:
        return Greeks(nan, nan, nan, nan, nan, nan)
    if T <= 0 or sigma <= 0:
        px = intrinsic(S, K, option_type)
        itm = px > 0
        delta = (-1.0 if itm else 0.0) if option_type == "P" else (1.0 if itm else 0.0)
        return Greeks(px, delta, 0.0, 0.0, 0.0, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    pdf_d1 = norm_pdf(d1)
    sqrtT = math.sqrt(T)

    gamma = df_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * df_q * pdf_d1 * sqrtT / 100.0
    common_theta = -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrtT)

    if option_type == "P":
        delta = -df_q * norm_cdf(-d1)
        theta = (common_theta + r * K * df_r * norm_cdf(-d2)
                 - q * S * df_q * norm_cdf(-d1))
        rho = -K * T * df_r * norm_cdf(-d2) / 100.0
    else:
        delta = df_q * norm_cdf(d1)
        theta = (common_theta - r * K * df_r * norm_cdf(d2)
                 + q * S * df_q * norm_cdf(d1))
        rho = K * T * df_r * norm_cdf(d2) / 100.0

    return Greeks(
        price=bs_price(S, K, T, r, q, sigma, option_type),
        delta=delta, gamma=gamma, theta=theta / DAYS_PER_YEAR, vega=vega, rho=rho,
    )


# ------------------------------------------------------------- implied vol
@dataclass(frozen=True)
class IVResult:
    iv: float
    reason: str = ""          # "" on success
    iterations: int = 0

    @property
    def ok(self) -> bool:
        return self.reason == "" and not math.isnan(self.iv)


def implied_vol_detail(price: float, S: float, K: float, T: float, r: float,
                       q: float, option_type: str = "P", *,
                       tol: float = 1e-9, max_iter: int = 100) -> IVResult:
    """Invert Black-Scholes for sigma, reporting WHY when it cannot.

    `tol` is a tolerance on SIGMA, not on price.

    Uses safeguarded Newton: take the Newton step when it stays inside the
    bracket and makes progress, otherwise bisect. Pure Newton diverges for deep
    in- or out-of-the-money options because vega collapses toward zero there,
    and those are precisely the strikes a naive solver returns nonsense for.
    """
    nan = float("nan")
    if not all(math.isfinite(x) for x in (price, S, K, T, r, q)):
        return IVResult(nan, "non_finite_input")
    if S <= 0 or K <= 0:
        return IVResult(nan, "bad_spot_or_strike")
    if T <= 0:
        return IVResult(nan, "expired")
    if price <= 0:
        return IVResult(nan, "non_positive_price")

    lo_px, hi_px = price_bounds(S, K, T, r, q, option_type)
    # Strictly outside the no-arbitrage band means no sigma reproduces this
    # price. Returning a number here is how bad quotes become fake Greeks.
    if price < lo_px - 1e-12:
        return IVResult(nan, "price_below_intrinsic")
    if price > hi_px + 1e-12:
        return IVResult(nan, "price_above_upper_bound")
    if abs(price - lo_px) < 1e-12:
        return IVResult(nan, "price_at_lower_bound")

    # Time value is what carries the volatility information. When it rounds to
    # nothing, EVERY sigma reproduces the price to within an absolute tolerance
    # and the solver would "converge" on whatever it started from. That is not a
    # measurement, so refuse it.
    time_value = price - lo_px
    if time_value <= max(MIN_TIME_VALUE, MIN_TIME_VALUE * K):
        return IVResult(nan, "no_time_value")

    lo, hi = MIN_VOL, MAX_VOL
    if bs_price(S, K, T, r, q, hi, option_type) < price:
        return IVResult(nan, "vol_above_max")

    sigma = _initial_guess(price, S, K, T, r, q, option_type)
    sigma = min(max(sigma, lo), hi)

    for i in range(1, max_iter + 1):
        px = bs_price(S, K, T, r, q, sigma, option_type)
        diff = px - price
        vega_raw = S * math.exp(-q * T) * norm_pdf(
            _d1_d2(S, K, T, r, q, sigma)[0]) * math.sqrt(T)

        # Converge in SIGMA, not in price. |diff| / vega is the first-order
        # error in sigma implied by the price residual, so this asks the
        # question we actually care about. A plain price tolerance is far too
        # loose for a deep-ITM put -- there vega is tiny, and a price matched to
        # a millionth of a dollar still leaves sigma wrong in the fourth decimal.
        if vega_raw > VEGA_FLOOR and abs(diff) < vega_raw * tol:
            return IVResult(sigma, "", i)

        # Price is monotone increasing in sigma, so the residual sign always
        # gives a valid bracket update.
        if diff > 0:
            hi = sigma
        else:
            lo = sigma

        if hi - lo < tol:
            mid = 0.5 * (lo + hi)
            # The bracket collapsed. Confirm it actually reproduces the price
            # rather than reporting the edge of a flat region as an answer.
            if abs(bs_price(S, K, T, r, q, mid, option_type) - price) < max(tol * price, 1e-9):
                return IVResult(mid, "", i)
            return IVResult(nan, "price_not_attainable", i)

        if vega_raw > VEGA_FLOOR:
            step = sigma - diff / vega_raw
            sigma = step if lo < step < hi else 0.5 * (lo + hi)
        else:
            sigma = 0.5 * (lo + hi)

    return IVResult(nan, "did_not_converge", max_iter)


def _initial_guess(price, S, K, T, r, q, option_type) -> float:
    """Brenner-Subrahmanyam ATM approximation, a good start near the money."""
    fwd = S * math.exp((r - q) * T)
    guess = math.sqrt(2.0 * math.pi / T) * price / max(fwd * math.exp(-r * T), 1e-12)
    return min(max(guess, 0.05), 2.0)


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                q: float, option_type: str = "P", **kw) -> float:
    """Implied vol, or NaN when no sigma reproduces `price`."""
    return implied_vol_detail(price, S, K, T, r, q, option_type, **kw).iv


def greeks_from_price(price: float, S: float, K: float, T: float, r: float,
                      q: float, option_type: str = "P"):
    """Solve for IV, then differentiate. Returns (Greeks | None, reason)."""
    res = implied_vol_detail(price, S, K, T, r, q, option_type)
    if not res.ok:
        return None, res.reason
    return bs_greeks(S, K, T, r, q, res.iv, option_type), ""


# ----------------------------------------------------------- american caveat
def early_exercise_risk(S: float, K: float, T: float, r: float, q: float,
                        option_type: str = "P") -> bool:
    """True where the European approximation stops being safe to rely on.

    Early exercise of an American put is worth considering when it is deep
    in-the-money, near expiry, and the interest earned on the strike exceeds the
    dividends forgone. This is a coarse screen, not a pricing model -- treat a
    True as "do not trust this delta", not as "exercise now".
    """
    if option_type != "P":
        return False
    moneyness = (K - S) / K if K > 0 else 0.0
    return moneyness > 0.05 and T < 0.25 and r > q
