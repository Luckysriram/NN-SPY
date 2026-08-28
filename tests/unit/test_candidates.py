from datetime import date

from data.candidates import accepted_only, generate_candidates, rejection_summary
from tests.conftest import E1, E2, TS, open_calendar, quote

CFG = {"min_oi": 100, "slippage_per_leg": 0.025, "min_credit": 0.20}


def q(expiry, dte, strike, delta, bid, ask, **kw):
    return quote(expiry=expiry, dte=dte, strike=strike, delta=delta,
                 bid=bid, ask=ask, **kw)


def chain():
    return [
        q(E1, 45, 390.0, -0.10, 1.0, 1.4),
        q(E1, 45, 400.0, -0.20, 2.0, 2.4),
        q(E1, 45, 410.0, -0.35, 4.0, 4.4),
        q(E1, 45, 395.0, -0.14, 0.8, 1.2),
    ]


def test_picks_short_strike_nearest_target_delta_and_long_5_below(cal):
    cands = accepted_only(generate_candidates(chain(), TS, CFG, calendar=cal))
    assert len(cands) == 1
    c = cands[0]
    assert c.short_strike == 400.0
    assert c.long_strike == 395.0
    assert c.width == 5.0
    assert c.entry_credit == 2.0 - 1.2 - 0.05      # short bid - long ask - slippage
    assert c.rejection_reason == ""


def test_every_expiry_is_reported_including_rejected_ones(cal):
    """An expiry missing its long leg is a recorded rejection, not a silent drop."""
    quotes = chain() + [q(E2, 31, 400.0, -0.20, 1.5, 1.9)]
    all_c = generate_candidates(quotes, TS, CFG, calendar=cal)
    assert len(all_c) == 2
    assert rejection_summary(all_c) == {"_accepted": 1, "long_leg_missing": 1}
    assert len(accepted_only(all_c)) == 1


def test_candidate_carries_real_greeks_and_liquidity_from_both_legs(cal):
    c = accepted_only(generate_candidates(chain(), TS, CFG, calendar=cal))[0]
    assert c.short_open_interest == 500
    assert c.short_volume == 100
    assert c.long_delta == -0.14
    assert c.short_theta == -0.02 and c.long_vega == 0.05


def test_rejects_when_credit_too_small(cal):
    quotes = [q(E1, 45, 400.0, -0.20, 0.2, 0.6), q(E1, 45, 395.0, -0.12, 0.1, 0.3)]
    c = generate_candidates(quotes, TS, CFG, calendar=cal)[0]
    assert c.rejection_reason == "credit_too_small"


def test_rejects_when_oi_below_min(cal):
    quotes = [q(E1, 45, 400.0, -0.20, 2.0, 2.4, open_interest=50),
              q(E1, 45, 395.0, -0.12, 0.8, 1.2, open_interest=50)]
    c = generate_candidates(quotes, TS, CFG, calendar=cal)[0]
    assert c.rejection_reason == "oi_too_low"


def test_rejects_when_nearest_delta_is_far_from_target(cal):
    """A '20-delta' strike 20 deltas away is not the trade the spec describes.

    Reported as delta_off_target, not long_leg_missing: the chain's real problem
    is that it quotes no 20-delta put.
    """
    quotes = [q(E1, 45, 400.0, -0.45, 2.0, 2.4), q(E1, 45, 395.0, -0.40, 0.8, 1.2)]
    c = generate_candidates(quotes, TS, CFG, calendar=cal)[0]
    assert c.rejection_reason == "delta_off_target"


def test_rejects_when_a_major_event_falls_inside_the_window():
    cal = open_calendar(FOMC=[date(2024, 1, 31)])
    c = generate_candidates(chain(), TS, {**CFG, "gate_on_events": True},
                            calendar=cal)[0]
    assert c.rejection_reason == "event_FOMC"


def test_rejects_when_the_event_calendar_does_not_cover_the_date():
    """Unknown is not the same as clear."""
    from data.events import EventCalendar
    blind = EventCalendar(dates={"FOMC": frozenset(), "CPI": frozenset(),
                                 "JOBS": frozenset()},
                          coverage_start=date(2030, 1, 1),
                          coverage_end=date(2031, 1, 1), source="test")
    c = generate_candidates(chain(), TS, {**CFG, "gate_on_events": True},
                            calendar=blind)[0]
    assert c.rejection_reason == "event_calendar_gap"


def test_flagged_quotes_are_excluded_by_default(cal):
    from schemas import with_flags
    dirty = [with_flags(x, ["wide_spread"]) for x in chain()]
    assert generate_candidates(dirty, TS, CFG, calendar=cal) == []


def test_dte_outside_the_window_is_ignored(cal):
    far = [q(date(2024, 6, 1), 151, 400.0, -0.20, 2.0, 2.4),
           q(date(2024, 6, 1), 151, 395.0, -0.12, 0.8, 1.2)]
    assert generate_candidates(far, TS, CFG, calendar=cal) == []
