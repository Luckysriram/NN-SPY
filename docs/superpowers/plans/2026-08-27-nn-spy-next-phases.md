# NN SPY — Next Phases

**Status:** Roadmap. Written 2026-08-27, after the v2 implementation landed.
**Prior:** `2026-08-27-nn-spy-options-ml.md` (implementation, 282 tests passing)

---

## Where the build actually stands

The machinery is done and tested. What it has never had is data.

| | |
|---|---|
| Pipeline | complete, 282 tests passing, CI green |
| Real option data | **none** — Yahoo failed feasibility (no Greeks, no history, fake IV) |
| Event calendar | 2024–2025 stub; `assert_covers()` refuses anything earlier |
| Model trained on real data | none |
| Simulator hand-audited | no |
| Result that means anything | none |

Everything below is gated on one thing, so it comes first.

---

## Phase 0 — Choose a data source  ⛔ BLOCKS EVERYTHING

This is a decision, not a task, and it has a cost. Nothing downstream can start
until it is made.

The strategy needs, per day, for several years: SPY put quotes at 30–45 DTE,
with **delta** (or enough to compute it), a strike grid fine enough for $5
spreads, and bid/ask captured at a consistent time of day.

### The options

| Source | Greeks | History | Cost | Notes |
|---|---|---|---|---|
| **University WRDS / OptionMetrics** | yes | decades | free with academic access | The gold standard. If you have a university login, start here and stop reading. |
| **CBOE DataShop** | yes | deep | pay per dataset | The spec's original suggestion. EOD chains are the relevant product. |
| **ORATS** | yes | deep | subscription | Purpose-built for this; clean Greeks. |
| **Polygon.io / ThetaData** | varies by tier | good | subscription | Check the tier actually includes Greeks before paying. |
| **Kaggle SPY options dumps** | sometimes | varies | free | Quality is uneven and coverage often ends years ago. Run feasibility before trusting one. |
| **Yahoo (`yfinance`)** | **no** | snapshot only | free | Already tested. Fails. Do not revisit. |

### Decision criteria

Ask of any candidate, in this order:

1. Does it have **delta**? If not, Phase 2 becomes mandatory.
2. Does it cover **≥ 3 years**? Fewer and walk-forward has too few folds to say
   anything.
3. Is the **strike grid ≤ $5**? Coarser and no spread is constructible.
4. Is the quote **timestamp consistent**? A mix of intraday and settlement
   prices makes the labels incoherent.

### Deliverable

A file on disk, and:

```bash
python -m data.adapters.feasibility_check path/to/chains.parquet
```

exiting 0. Until it does, everything below is blocked.

---

## Phase 1 — Ingest

**Depends on:** Phase 0

1. **Write the adapter.** Most sources need only a `colmap` entry, not a new
   module — `DEFAULT_COLMAP` in `data/adapters/sample.py` already handles four
   naming conventions. Add a real module only if the format is exotic (nested
   JSON, per-day files, compressed archives).
2. **Run feasibility, then set `configs/data.yaml` from what it reports** — not
   from the spec's aspirational 2016–2025 range.
3. **Ingest to Parquet** under `data/raw/option_chains/`, immutable.
4. **Read `flag_summary()` before proceeding.** If a third of quotes carry
   `wide_spread` or `stale_quote`, that shapes every downstream filter.

**Watch for:** the two defects real data already exposed — blank numeric fields
(fixed in `d2b41ec`), and vendor IV columns that are placeholders rather than a
real surface. Check `df.iv.nunique()`. A few dozen distinct values across
thousands of contracts means the column is fake.

**Effort:** hours if the colmap fits; a day if the format is awkward.

---

## Phase 2 — Greeks, if the dataset lacks them  *(conditional)*

**Skip entirely if Phase 1's data has delta.**

Delta is not optional — the strategy is *defined* by a −0.20 delta short strike.
If the source has prices but no Greeks:

1. **Solve for implied vol** from the mid price by bisection or Newton on
   Black-Scholes. Do not trust a vendor IV column that failed the `nunique`
   check above.
2. **Compute Greeks** from that IV. `tests/fixtures/make_fixture.py::bs_put`
   already implements the pricing and Greeks — promote it to
   `models/blackscholes.py` rather than duplicating it.
3. **You need two inputs the dataset will not have:** a risk-free rate curve
   (Treasury yields by tenor) and SPY's dividend yield. Both matter — SPY pays
   roughly 1–2%, and ignoring it biases put deltas.
4. **Validate against something.** If any subset of your data has vendor Greeks,
   reconcile. Otherwise sanity-check that ATM delta ≈ −0.5 and that delta is
   monotonic in strike.

**Risk:** computed Greeks inherit every error in your rate and dividend
assumptions. A delta that is systematically off by 0.03 silently changes which
strike gets sold every single day. Treat this phase as a real component with its
own tests, not a utility function.

**Effort:** 2–3 days including validation.

---

## Phase 3 — Real event calendar

**Depends on:** Phase 1 (you need to know the date range first)

Load actual FOMC, CPI and NFP dates covering the full data range into
`data/raw/events/events.csv` (`date,event`). The Fed publishes its calendar;
BLS publishes release schedules. The jobs-report rule (first Friday) is already
implemented but has holiday exceptions — prefer the published dates.

`load_event_calendar()` picks the CSV up automatically and derives coverage from
it. `assert_covers()` then stops complaining.

**Effort:** an afternoon, mostly transcription.

---

## Phase 4 — Audit the simulator  🚦 GATE

**Depends on:** Phases 1–3

The spec requires this and it has not been done. The simulator defines
`label_win`, so every downstream number inherits its bugs.

```bash
python -m labels.audit --n 50 --seed 42
```

Read all 50 by hand. `consistency_checks()` catches arithmetic contradictions
automatically; you are looking for the things it cannot:

- Does the exit rule fire on the day you would actually have exited?
- Are the fills plausible against the quotes on that date?
- Does a STOP_LOSS trade look like a real adverse move, or a data glitch?
- Do the `NO_DATA` drops cluster suspiciously (a delisted strike, a holiday)?

**Gate:** do not proceed until you have personally read 50 trades and believe
them. This is the cheapest place in the entire project to catch a wrong label,
and the most expensive one to skip.

**Effort:** half a day of actual reading.

---

## Phase 5 — Baselines and the first honest report  🚦 THE REAL GATE

**Depends on:** Phase 4

No neural network in this phase. The question is narrower and more important:
**is there any edge here at all?**

1. Build the dataset (`pipeline.build_dataset`) and read the `BuildReport`.
2. **Write down the base rate before looking at any model.** A 20-delta short
   put wins often by construction — expect something like 70–85%. That number is
   the bar, and it is a high one.
3. Run the walk-forward engine with `no_trade`, `always_trade`, `simple_rule`,
   logistic regression and gradient boosting.
4. Read the **calibration audit before the AUC**.

### Go / no-go

| Signal | Read |
|---|---|
| Gradient boosting AUC ≈ 0.5 | No learnable signal in these features. Do not proceed to a neural net — it will not find what GBM cannot. |
| `always_trade` beats every model after costs | The model adds nothing. The strategy may still work; the ML does not. |
| Calibration gap > 0.10 in populated buckets | Probabilities are not usable for sizing, whatever the AUC says. |
| Everything loses after costs | Stop. Report it. This is a valid and valuable outcome. |

**Be willing to stop here.** SPY put credit spreads are among the most heavily
traded strategies in existence; a large, easily-found edge is unlikely. Finding
none is the expected result, and finding it cheaply is what this whole build was
for.

**Effort:** 2–3 days including analysis.

---

## Phase 6 — Neural network  *(conditional on Phase 5)*

**Only if a baseline showed genuine out-of-sample, after-cost edge.**

The MLP and the NumPy reference implementation are built and tested. This phase
is about *use*, not construction:

1. Train through the walk-forward engine (`model_fn=` a torch wrapper).
2. Compare against gradient boosting on identical folds.
3. **Seed stability:** train 10 seeds. If fold AUC swings by more than ~0.05
   across seeds, the model is fitting noise and the mean is meaningless.
4. Keep it only if it beats GBM out of sample, after costs, on most folds.

The spec's rule stands: if it cannot beat the baselines, it does not get used.

**Effort:** 2–3 days.

---

## Phase 7 — Robustness

**Depends on:** Phase 5 or 6 producing something worth stressing

1. **Cost sensitivity.** Re-run at 1.5× and 2× slippage. An edge that dies at
   1.5× was never real — your fill assumptions are not that precise.
2. **Regime breakdown.** `monthly_report(groups=...)` splits by VIX bucket, DTE
   and exit reason. An edge that lives entirely in one regime is a bet on that
   regime.
3. **Feature ablation.** Drop each feature group; see what actually carried the
   result.
4. **Threshold objective.** Currently `select_threshold` defaults to
   `objective="total"`, which on the demo selected 0.05 — near-indiscriminate
   trading. `"per_trade"` has the opposite failure. Neither is obviously right;
   decide it against real data, where the tradeoff is visible.

---

## Phase 8 — Paper trading

**Depends on:** a frozen model that survived Phase 7

1. Daily runner calling `paper_signal` at 3:45 PM ET, logging every decision
   including NO_TRADE via `append_decision_log`.
2. Persist `train_mean`/`train_std` so `feature_drift_kill` is armed.
3. **Fix the timezone.** `reports/daily_report.py` hard-codes `-04:00`, which is
   wrong by an hour from November to March. Attach real `zoneinfo`
   ("America/New_York") before any log is used for analysis.
4. Weekly: compare logged probabilities against realised outcomes. This is live
   calibration, and it is the number that tells you whether the research held.
5. Decide a retraining cadence **in advance** (monthly or quarterly) and write it
   down, so it is not a reaction to a drawdown.

Run for a meaningful period — quarters, not weeks — before any conclusion.

---

## Known gaps to close along the way

Carried from the v2 review, none of them blocking but all real:

| Gap | Where | Matters when |
|---|---|---|
| Assignment / early exercise not modelled | simulator | A short put going deep ITM can be assigned early. Defined risk bounds the loss, but the cash-flow event is real. |
| Timezone hard-coded to EDT | `reports/daily_report.py` | Phase 8, immediately |
| One candidate per expiry per day | `data/candidates.py` | If you later want multiple strikes or widths |
| No market-impact model | `backtest/costs.py` | Only at size; irrelevant at 1–3 contracts |
| Threshold objective unsettled | `models/select_threshold.py` | Phase 7 |
| `MAX_HOLD_DAYS` = 38 vs spec §2's "21 days" | `schemas.py` | Spec contradiction; code follows the exit rules. Resolve if 21 is a real constraint. |

---

## Honest expected outcome

The most likely result of Phases 4–5 is **no reliable edge after costs.** That is
not a failure of the build — it is the build doing its job, which is to find that
out for the price of some data and a week of analysis rather than a year of live
losses.

The second most likely result is a small edge that vanishes under Phase 7's cost
stress. Also useful. Also cheap.

Design every phase so stopping is a clean, reportable outcome rather than sunk
cost. The gates exist for that.
