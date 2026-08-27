# NN SPY Options ML System — Implementation Plan (v2)

**Status:** IMPLEMENTED AND VERIFIED. 281 tests pass (`python -m pytest -q`).
**Supersedes:** v1 of this plan, dated 2026-08-27.
**Spec:** `docs/specs/2026-08-27-nn-spy-options-ml-design.md`

**Goal:** A research system predicting the calibrated probability that a SPY put
credit spread reaches its 50% profit target before its stop loss or forced exit,
validated on unseen future data. Paper trading only.

**Stack:** Python 3.11 · pandas · NumPy · scikit-learn · PyTorch · pyyaml ·
pyarrow · yfinance · Matplotlib · pytest.

---

## Why there is a v2

v1 was reviewed by running its own code against its own tests. Ten of its
~53 tests failed against the implementations printed beside them, and several
that passed did so while masking defects. Every "Expected: PASS" in this
document has been produced by an actual run.

### Defect register

Failures of v1's own tests against v1's own code:

| # | Where | Defect | Now |
|---|---|---|---|
| 1 | `download_underlying` | Test's `fake_fetch` returned a dict; code called `.iterrows()` → `AttributeError` | `_as_frame` accepts dicts, records and DataFrames; MultiIndex columns flattened |
| 2 | `days_to_event` | Filtered `ed >= d` then returned 0 when empty, so it could never return the negative the interface promised | Signed distance to the nearest event; `None` when the calendar is empty |
| 3 | `generate_candidates` | Test asserted 1 candidate, code returned 2 (second expiry lacked its long leg) | All candidates returned with reasons; `accepted_only()` filters |
| 4 | `simulate_trade` | Profit-target check ran first and swallowed the STOP_LOSS, TIME_EXIT and EXPIRY fixtures — 3 of 6 tests | Stop checked first (conservative); each exit has an unambiguous test |
| 5 | `FitScaler` | Test assumed sample std (10.0); `StandardScaler` uses population std (8.165) | Test asserts against `np.std`; the difference is documented |
| 6 | `apply_costs` | Test wanted `fees == 0.20`, code computed `0.10` — no per-leg multiplier | 4 contract-legs per round trip: 2 legs × (open + close) |
| 7 | `TradeLedger.total_pnl` | Summed `return_on_risk × width`, disagreeing with the test's own P&L arithmetic | Sums the recorded `net_pnl` of its own rows |
| 8 | `test_build_mlp...` | `bool(tensor)` on a 2×1 tensor raises `RuntimeError` | `((out > 0) & (out < 1)).all()` |

Defects that passed their tests, which is worse:

| # | Where | Defect | Now |
|---|---|---|---|
| 9 | `simple_rule_predict` | Read column 20 as `credit_to_width`; `FEATURE_COLS` had 32 entries and column 20 was `credit`. The test used a 30-wide zero matrix, so it passed | Indexes via `FEATURE_INDEX["credit_to_width"]`; a test proves the neighbouring `credit` column does not affect it |
| 10 | `calendar_features` | Emitted `days_to_jobs`; the feature list wanted `days_to_jobs_report`. `feat.get(k, 0.0)` swallowed the mismatch | `EVENT_FEATURE_SUFFIX` is the single source; a test asserts every calendar column is produced |
| 11 | `spread_features` | `short_put_open_interest`, `short_put_volume`, `long_put_delta`, `net_theta`, `net_vega`, `net_gamma` hard-coded to `0.0` — `Candidate` could not carry them | `Candidate` carries both legs' Greeks, OI and volume; a test fails on any zero stub |
| 12 | `spread_features` | `short_put_bid_ask_pct = bid_ask / implied_vol` | Divided by the leg's own mid |
| 13 | `FitScaler.freeze` | Set `_frozen = True`; `fit()` never read it, so the one guarantee the class existed for did not exist | `fit()` after `freeze()` raises `ScalerFrozenError` |
| 14 | `train_mlp` | Early stopping monitored **training** loss | Monitors validation loss; says so explicitly when no validation set is given |
| 15 | `test_paper_signal` | Asserted `decision in ("PAPER_TRADE", "NO_TRADE")` — every possible value — on a fixture the event gate rejected before the model ran, hiding a scaler shape error and a `TypeError` | Tests reach the model; each gate is asserted separately |

Financial and methodological errors:

| # | Where | Defect | Now |
|---|---|---|---|
| 16 | `walk_forward_folds` | `test_start = train_end`, zero gap. A trade entered 2020-12-15 at 45 DTE resolves ~2021-01-29, so its label is built from test-period quotes | `EMBARGO_DAYS = MAX_DTE − TIME_EXIT_DTE = 38`; training rows within the gap are purged; `assert_split_is_clean` proves it |
| 17 | `final_return_on_risk` | Divided by `width`. A $5 spread sold for $2 risks $3 — every return inflated ~40% | `Candidate.max_risk = width − credit` |
| 18 | `select_threshold` | Hard-coded `profit=1.0, loss=1.0`, implying a 50% break-even; real payoffs are +0.5×credit / −1×credit, so break-even is **66.7%**. Also ignored trade count, so it could pick the two luckiest trades | `spread_payoffs()` derives the real asymmetry; `min_trades` floor; returns `1.01` (never trade) when nothing qualifies |
| 19 | `calibrate_proba` | Fit isotonic on **training** probabilities — the data the model overfit — contradicting spec §9.4 | Fit on validation, then frozen; a test shows calibration error falls out of sample |
| 20 | Event calendar | Seeded 2024-03 onward while training targeted 2016–2022, so `days_to_*` was a constant 0 in training and a real number at serving | `EventCalendar` declares coverage; outside it features are NaN; `assert_covers()` refuses unsupported periods |
| 21 | Costs | No contract multiplier; P&L in per-share units while risk limits are in dollars | `CONTRACT_MULTIPLIER = 100` applied in `apply_costs` |
| 22 | `run_walk_forward` | `for fold in folds: pass` — a stub. Spec §11's engine, ledger fields and regime breakdowns were never built | Fully implemented; 15 engine tests |
| 23 | `monthly_report` | Interface promised `by_prob_bucket`; returned only `calibration` and `trades` | `by_prob_bucket` implemented with `calibration_gap`, plus regime breakdowns |
| 24 | Baselines | No `random_state`, violating "record all random seeds" | Every model seeded; reproducibility asserted |
| 25 | Missing modules | `feasibility_check.py` (spec §6.1, "the first script run"), the 50-trade audit (§7/§13), the four YAML configs, `make_fixture.py` | All present |

Found while building v2:

| # | Where | Defect | Now |
|---|---|---|---|
| 26 | `build_mlp` | Weights were initialised before `train_mlp` seeded, so a recorded seed did not reproduce its run | `build_mlp(seed=...)` seeds before initialisation; a test asserts same-seed equality and different-seed difference |
| 27 | `feature_drift_kill` | Substituted `1.0` for a zero training std, making the z-score arbitrary | A constant training feature flags any different live value on its own terms |
| 28 | `sortino` | `std(ddof=1)` on a single downside observation warned and returned nan | Guarded at `len(downside) < 2` |
| 29 | `_first_rejection` | Reported `long_leg_missing` for chains whose real problem was no 20-delta strike (put deltas shrink as strikes fall, so the nearest strike is often the lowest quoted) | Delta checked first |

### Two spec inconsistencies this surfaced

- **§2 says "holding period up to 21 calendar days"**, but entering at 30–45 DTE
  and exiting at 7 DTE implies 23–38 days. The code uses
  `MAX_HOLD_DAYS = MAX_DTE − TIME_EXIT_DTE = 38`, which is what the exit rules
  actually produce. If 21 days is the real constraint, it needs its own exit rule.
- **§12's split dates (train 2016–2022) cannot coexist with a 2024+ event
  calendar.** Resolved by making calendar coverage explicit and enforced rather
  than picking one silently.

---

## Global constraints

- Python ≥ 3.11. Parquet for option chains, never CSV.
- **Raw data is immutable.** Flag with `quality_flags`; never edit a record.
- **Missing is NaN, never zero.** Zero-filling teaches the model that unknown
  equals average.
- **No lookahead.** Features use only `timestamp <= as_of`.
- **Never shuffle a time series across time.** Chronological splits only.
  (Minibatch shuffling *within* an already-split training window is fine and is
  what `train_mlp(shuffle=True)` does.)
- **Purge the embargo.** 38 days between any train and test boundary.
- **Scalers and calibrators freeze.** Calibration is fit on validation only.
- **Baselines before the neural net.** It must beat logistic regression and
  gradient boosting out of sample, after costs, or it is rejected.
- **Conservative fills.** Sell the bid, buy the ask, slippage per leg,
  commission on 4 contract-legs per round trip.
- **Record every seed**, and seed before weight initialisation.
- **The model never sets risk limits.**
- **Paper trade only.**

---

## Tasks

Each task lists its files, the decisions that matter, and the verified test
result. Implementations live in the repo — this document does not inline them,
because inlined code in a plan goes stale the moment the code changes.

### Task 1 — Scaffold, configs, canonical schemas
**Files:** `pyproject.toml`, `Makefile`, `schemas.py`, `config.py`,
`configs/{data,strategy,model,risk}.yaml`, package `__init__.py` files
**Tests:** `tests/unit/test_schemas.py`, `tests/unit/test_config.py`

- `Candidate` carries **both legs'** Greeks, OI and volume — without this, six
  features are structurally impossible (defect 11).
- `Candidate.max_risk = width − credit` (defect 17).
- `EMBARGO_DAYS = MAX_HOLD_DAYS = MAX_DTE − TIME_EXIT_DTE = 38` (defect 16).
- `quality_flags` is a tuple; `with_flags()` returns a copy, so raw records
  cannot be mutated in place.
- `require()` raises on a missing config key rather than defaulting silently.

```bash
python -m pytest tests/unit/test_schemas.py tests/unit/test_config.py -q
```
**Verified: 13 passed.**

### Task 2 — Adapters and downloader
**Files:** `data/adapters/base.py`, `data/adapters/sample.py`, `data/download.py`
**Tests:** `tests/unit/test_adapters.py`, `tests/unit/test_download.py`

- Column mapping is **data** (`DEFAULT_COLMAP`), so a new vendor needs no new
  module — tested against two different column layouts.
- Unparseable rows become `NormalizeResult.rejects`, not a `print()` in a loop
  over a million rows.
- Missing Greeks → NaN plus a flag, never 0.0.
- `_as_frame` accepts dicts and flattens MultiIndex columns (defect 1).

```bash
python -m pytest tests/unit/test_adapters.py tests/unit/test_download.py -q
```
**Verified: 10 passed.**

### Task 3 — Validation
**Files:** `data/validate.py` · **Tests:** `tests/unit/test_validate.py`

Covers every check in spec §6.3. `zero_iv` is distinct from `negative_iv`
(a missing field and a bad field must not look identical). Duplicates and
staleness are batch-level. Nothing is ever repaired.

```bash
python -m pytest tests/unit/test_validate.py -q
```
**Verified: 16 passed.**

### Task 4 — Event calendar
**Files:** `data/events.py` · **Tests:** `tests/unit/test_events.py`

- `EventCalendar` declares `coverage_start`/`coverage_end`; `assert_covers()`
  refuses periods it cannot support (defect 20).
- `days_to_event` returns signed distance, `None` on an empty calendar (defect 2).
- Jobs reports are rule-derived (first Friday); FOMC is seeded and honestly
  scoped; CPI is empty rather than invented.

```bash
python -m pytest tests/unit/test_events.py -q
```
**Verified: 12 passed.**

### Task 5 — Candidate generation
**Files:** `data/candidates.py` · **Tests:** `tests/unit/test_candidates.py`

Every candidate is returned with its reason (defect 3). `delta_off_target` is
checked before `long_leg_missing` (defect 29). An uncovered calendar date is
`event_calendar_gap` — unknown is not clear.

```bash
python -m pytest tests/unit/test_candidates.py -q
```
**Verified: 10 passed.**

### Task 6 — Outcome simulator and labels
**Files:** `labels/simulator.py`, `labels/outcomes.py`, `labels/audit.py`
**Tests:** `tests/unit/test_simulator.py`

The component the spec calls most important. Three deliberate choices:

1. **Stop before target.** Daily marks cannot resolve intraday sequence;
   assuming the favourable one manufactures free wins (defect 4).
2. **Conservative fills** on both sides.
3. **`NO_DATA` is not a win.** Unlabelable trades are excluded by
   `is_labelable`, and `label_distribution` reports how many were dropped.

`labels/audit.py` renders the spec's 50-trade manual audit with the arithmetic
spelled out, plus `consistency_checks` that assert what a correct simulation
must satisfy.

```bash
python -m pytest tests/unit/test_simulator.py -q
```
**Verified: 18 passed.**

### Task 7 — Features and scalers
**Files:** `features/features.py`, `features/scalers.py`
**Tests:** `tests/unit/test_features.py`, `tests/unit/test_scalers.py`

33 features. `FEATURE_INDEX` maps name → position so nothing indexes by literal
(defect 9). `EVENT_FEATURE_SUFFIX` keeps calendar keys and feature names in sync
(defect 10). Greeks come from the candidate (defect 11). Bid-ask is a fraction
of mid (defect 12). Insufficient history yields NaN, and `assert_no_nan` makes
handling it a decision. `FitScaler.freeze()` is enforced (defect 13).

```bash
python -m pytest tests/unit/test_features.py tests/unit/test_scalers.py -q
```
**Verified: 27 passed.**

### Task 8 — Chronological splits and the embargo
**Files:** `features/splits.py` · **Tests:** `tests/unit/test_splits.py`

The most important methodology fix (defect 16). One test constructs the leak
with `embargo_days=0` and asserts `assert_split_is_clean` raises; another shows
the same split is clean with the embargo on.

```bash
python -m pytest tests/unit/test_splits.py -q
```
**Verified: 12 passed.**

### Task 9 — Metrics and baselines
**Files:** `reports/metrics.py`, `models/baselines.py`
**Tests:** `tests/unit/test_metrics.py`, `tests/unit/test_baselines.py`

`base_rate` is reported beside accuracy so 85% accuracy on an 85% base rate is
visible for what it is. `expected_calibration_error` answers the spec's question
directly. Single-class AUC is NaN, not a misleading 0.5. All models seeded
(defect 24).

```bash
python -m pytest tests/unit/test_metrics.py tests/unit/test_baselines.py -q
```
**Verified: 25 passed.**

### Task 10 — Costs, ledger, and the backtest engine
**Files:** `backtest/costs.py`, `backtest/ledger.py`, `backtest/engine.py`
**Tests:** `tests/unit/test_costs.py`, `tests/unit/test_ledger.py`, `tests/unit/test_engine.py`

Round-trip fees on 4 contract-legs (defect 6). P&L in dollars via the 100×
multiplier (defect 21). `total_pnl` sums its own rows (defect 7). The ledger
carries every spec §11 field and writes CSV.

**The engine is real** (defect 22). Per fold, in order: inner train/validation
split with the same embargo → fit scaler on inner-train and freeze → train →
calibrate on inner-validation and freeze → pick threshold on inner-validation →
score test rows, walk them chronologically, apply risk rules, record trades.

```bash
python -m pytest tests/unit/test_costs.py tests/unit/test_ledger.py tests/unit/test_engine.py -q
```
**Verified: 32 passed.**

### Task 11 — Hand-written NumPy network
**Files:** `models/numpy_net.py` · **Tests:** `tests/unit/test_numpy_net.py`

The spec's learning exercise. The module docstring derives why `dL/dz = (p − y)`
for sigmoid + BCE. `gradient_check()` compares analytic gradients to numeric
ones and is asserted below `1e-6` — the test that actually proves the backward
pass is right, rather than inferring it from a training curve.

```bash
python -m pytest tests/unit/test_numpy_net.py -q
```
**Verified: 11 passed** (gradient check: max relative error 1.51e-09).

### Task 12 — PyTorch MLP
**Files:** `models/mlp.py` · **Tests:** `tests/unit/test_mlp.py`

Logits + `BCEWithLogitsLoss`. Early stopping on validation loss, and it states
`"train_loss (NO VALIDATION SET)"` when there is none rather than hiding it
(defect 14). Best weights restored, not the last epoch. `build_mlp` seeds before
initialisation (defect 26).

```bash
python -m pytest tests/unit/test_mlp.py -q
```
**Verified: 14 passed.**

### Task 13 — Calibration and threshold
**Files:** `models/calibrate.py`, `models/select_threshold.py`
**Tests:** `tests/unit/test_calibrate.py`, `tests/unit/test_select_threshold.py`

Calibration fit on validation, then frozen (defect 19); a test asserts ECE falls
out of sample. `spread_payoffs` derives the true payoffs, `breakeven_win_rate`
returns 66.7% for a $2 credit, and a test confirms a 55% win rate is rejected —
it clears a 50% break-even but not the real one (defect 18).

```bash
python -m pytest tests/unit/test_calibrate.py tests/unit/test_select_threshold.py -q
```
**Verified: 21 passed.**

### Task 14 — Risk engine
**Files:** `risk/risk.py` · **Tests:** `tests/unit/test_risk.py`

Deterministic. `max_trades_per_day` is now actually enforced. `position_size`
respects the per-trade loss cap. Kill switches on daily and weekly loss, and
`feature_drift_kill` halts on out-of-distribution inputs (defect 27).

```bash
python -m pytest tests/unit/test_risk.py -q
```
**Verified: 19 passed.**

### Task 15 — Reports
**Files:** `reports/daily_report.py`, `reports/monthly_report.py`
**Tests:** `tests/unit/test_reports.py`

`by_prob_bucket` implemented with `calibration_gap` (defect 23), plus regime
breakdowns and a text renderer that flags buckets off by more than 10 points.
Every decision is logged including NO_TRADE. The ET offset is hard-coded `-04:00`
and the docstring says so — it is wrong by an hour from November to March unless
a real `tzinfo` is attached upstream.

```bash
python -m pytest tests/unit/test_reports.py -q
```
**Verified: 14 passed.**

### Task 16 — Paper-trading signal
**Files:** `paper_trade/daily_signal.py` · **Tests:** `tests/unit/test_paper_trade.py`

Gates in order: candidate exists → features complete → no drift → probability
clears the frozen threshold → EV positive → risk allows. Tests reach the model
and assert each gate separately (defect 15). The no-candidate placeholder uses
the decision date, not `date.today()`, so the log is reproducible.

```bash
python -m pytest tests/unit/test_paper_trade.py -q
```
**Verified: 11 passed.**

### Task 17 — Pipeline, fixture, integration test, README
**Files:** `pipeline.py`, `tests/fixtures/make_fixture.py`,
`tests/integration/test_end_to_end.py`, `README.md`
**Tests:** `tests/integration/test_end_to_end.py`

`make_fixture` generates a Black-Scholes-priced synthetic chain so deltas,
prices and decay are internally consistent. Its `warmup_days` parameter withholds
the chain while keeping price history — without it every row is dropped as
incomplete, because `vix_percentile_252d` needs a year of bars.

`pipeline.build_dataset` wires raw rows → dataset and returns a `BuildReport`
showing quote quality, rejection reasons, label distribution and NaN counts.

The integration test asserts cross-module properties, above all that no training
trade is still open when its test period starts — checked with **real exit
dates**, not the worst-case assumption.

```bash
python -m pytest tests/integration/test_end_to_end.py -q
```
**Verified: 16 passed.**

---

## Full-suite verification

```bash
python -m pytest -q
```
**Verified: 281 passed, no warnings.**

| Task | Tests | Task | Tests |
|---|---|---|---|
| 1 Scaffold | 13 | 10 Costs/ledger/engine | 32 |
| 2 Adapters | 10 | 11 NumPy net | 11 |
| 3 Validation | 16 | 12 PyTorch MLP | 14 |
| 4 Events | 12 | 13 Calibration/threshold | 21 |
| 5 Candidates | 10 | 14 Risk | 19 |
| 6 Simulator | 18 | 15 Reports | 14 |
| 7 Features/scalers | 27 | 16 Paper trade | 11 |
| 8 Splits | 12 | 17 Integration | 16 |
| 9 Metrics/baselines | 25 | **Total** | **281** |

---

## What this plan does NOT deliver

Stated plainly, because a plan that implies otherwise is worse than no plan:

1. **No real data has been touched.** `configs/data.yaml` holds placeholder
   dates. Every result so far comes from a synthetic fixture and says nothing
   about whether the strategy works.
2. **The event calendar is a stub** covering 2024–2025. Training on 2016–2022
   requires real Fed/BLS dates in `data/raw/events/events.csv`. The code refuses
   rather than guessing.
3. **No model has been trained on anything real**, so no claim about edge,
   calibration or profitability is available yet.
4. **The 50-trade manual audit has not been performed.** The tooling exists;
   the spec requires a human to actually read them.
5. **Assignment, early exercise and settlement mechanics are not modelled.**
   Defined-risk spreads bound the loss, but early assignment on the short leg is
   a real operational event this system does not simulate.

## Next steps, in order

1. **Run the feasibility check on a real dataset** — before anything else. It
   decides whether the strategy is buildable at all:
   ```bash
   python -m data.adapters.feasibility_check path/to/chains.parquet
   ```
   Free sample data often ships without Greeks, which makes 20-delta selection
   impossible and invalidates a third of the feature list. Find out first.
2. **Set `configs/data.yaml` from what it reports**, not from the spec's
   aspirational 2016–2025 range.
3. **Load a real event calendar** into `data/raw/events/events.csv`.
4. **Build the dataset and read the `BuildReport`** — especially the rejection
   summary and base rate — before training anything.
5. **Audit 50 simulated trades by hand** (`labels/audit.py`). The spec requires
   this before the simulator is trusted, and it is the cheapest place to catch a
   wrong exit rule.
6. **Run baselines through the walk-forward engine.** Read the calibration audit
   before the AUC.
7. **Only then** train the MLP, and only keep it if it beats the baselines out
   of sample after costs.
