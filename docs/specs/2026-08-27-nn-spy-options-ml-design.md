# NN SPY — Neural-Network Research System for SPY Put Credit Spreads

**Date:** 2026-08-27
**Status:** Approved design (user reviewed sections 1–4)
**Location:** `D:\PROJECTS\Option\nn-spy\`

---

## 1. Purpose

Build a **research system** — explicitly *not* a live trading bot — that predicts, for SPY put credit spreads, the **calibrated probability that the spread reaches its 50% profit target before its stop loss or forced exit**. The output is a calibrated probability (e.g. `0.67`), not a BUY/SELL signal, and the system is validated rigorously on unseen future data before any money is at stake.

> Options can lose money quickly; model output is not a trade recommendation. Users should read the current OCC Options Disclosure Document before trading.

## 2. Prediction Target

| Property | Value |
|---|---|
| Instrument | SPY weekly/monthly put credit spreads |
| Decision time | Once per day, 3:45 PM ET |
| Holding period | Up to 21 calendar days |
| Entry | Sell a put around 15–25 delta; buy a lower-strike put |
| Spread width | $5 initially |
| Exit rules | Take profit at 50% of initial credit; stop loss at 2× initial credit; exit at 7 days to expiry; exit at expiry if no earlier rule hit |

**Neural-network target (binary classification):**
- `label_win = 1` if the spread reaches its 50% profit target before stop loss / time exit / expiry.
- `label_win = 0` otherwise.

## 3. Success Criteria (before any model is trusted)

1. **Classification:** better ROC-AUC and precision than baselines.
2. **Calibration:** predicted 60% trades win about 60% of the time.
3. **Trading test:** positive results *after* bid–ask spread, fees, and slippage.
4. **Risk:** max drawdown stays inside the pre-set limit.
5. **Robustness:** works across calm, volatile, bull, bear, and sideways markets.
6. **Safety:** the system can choose NO TRADE often.

Do **not** accept a model because training accuracy looks high.

## 4. Data Strategy

**Free sample dataset first** (user-selected approach), made source-agnostic:

- A **data adapter layer** normalizes any source (CBOE DataShop EOD chains, Kaggle datasets, ORATS samples) into **one canonical Parquet schema**.
- A **feasibility check** (`data/adapters/feasibility_check.py`) runs before committing to any dataset: reports date range, coverage, and which fields exist.
- Downstream modules (simulator, features, models) are written once against the canonical schema and never depend on the source.
- **Upgrade path:** when ready for real-money research, add a paid-provider adapter (ThetaData / Polygon / ORATS). Only one small module changes.
- Underlying + VIX data comes free from Yahoo Finance via `yfinance`.
- Raw data is **immutable**; all cleaning happens downstream by stamping `data_quality_flag` rather than silently fixing records.

## 5. Architecture Overview

```
[FREE SAMPLE DATASET(s)]
        │
        ▼
DATA/ADAPTERS ─── normalizes ANY source into ONE canonical Parquet schema
        ▼
DATA/RAW/  (immutable, timestamped, never edited after ingest)
        ▼
DATA/VALIDATE ─── flags bad records (data_quality_flag), never silently fixes
        ▼
DATA/CANDIDATES ─ 20-delta short put, $5 wide, 30–45 DTE; rejected ones saved WITH reason
        ▼
FEATURES/ ──────── 30–40 timestamped features (market, option, calendar/event)
        ▼
LABELS/SIMULATOR ─ walks each candidate forward → PROFIT_TARGET / STOP_LOSS /
                   TIME_EXIT / EXPIRY  →  label_win (1/0) + secondary labels
        ▼
SPLIT ──────────── chronological walk-forward (train 2016–20 → test 2021, …)
        ▼
MODELS/ ────────── baselines (logistic regression, gradient boosting) → small MLP
                   → calibrated probability, never raw logit
        ▼
BACKTEST/ ──────── event-driven walk-forward engine + trade ledger (fees/slippage)
        ▼
RISK/ ──────────── hard-coded rules: max loss/trade, max open risk, kill switches
        ▼
REPORTS/ ───────── calibration curves, ROC-AUC, profit factor, regime breakdown
        ▼
PAPER_TRADE/ ───── daily: build today's features → signal → log decision (incl. NO TRADE)
```

**Governing rule:** a record is either in the immutable raw store or it isn't; features may only reference data with a timestamp ≤ the decision time. This makes "no future data leakage" enforceable by construction.

### Canonical option-quote schema
`timestamp · symbol · expiry · dte · strike · option_type (C/P) · bid · ask · mid · last · volume · open_interest · iv · delta · gamma · theta · vega · underlying_price · quality_flags`

### Canonical underlying-bar schema
`timestamp · symbol · open · high · low · close · adj_close · volume · returns_1d_5d_10d_21d_63d · ma_5_10_20_50_200 · realized_vol_5_10_21_63 · max_drawdown_63d · distance_from_ma20_ma50_ma200`

## 6. Data Layer

### 6.1 Adapters (`data/adapters/`)
- One module per source (`cboe_adapter.py`, `kaggle_adapter.py`, …). Each outputs canonical Parquet.
- `feasibility_check.py` — first script run; reports what a candidate dataset actually contains before we commit.

### 6.2 Download (`data/download.py`)
- Fetch SPY daily bars, VIX daily, and the chosen sample option-chain dataset.

### 6.3 Validation (`data/validate.py`)
Flag (never fix) records with:
- Bid greater than ask
- Negative quotes, IV, DTE, volume, or open interest
- Missing underlying price
- Very wide option spreads
- Stale quotes
- Zero bid on either leg
- Expiry before entry
- Incorrect strike ordering
- Duplicate contracts/timestamps
- Impossible Greeks or IV values

Every suspicious record gets a `data_quality_flag`.

### 6.4 Candidates (`data/candidates.py`)
At each eligible timestamp:
1. Get the SPY option chain.
2. Select expiries with 30–45 DTE.
3. Find short puts closest to delta −0.20.
4. Choose the long put $5 below the short strike.
5. Compute a **conservative entry credit** (mid − slippage).
6. Reject a candidate (recording the reason) if: width ≠ $5, credit too small, bid–ask too wide, OI/volume below rule, or a major event inside the strategy window (unless explicitly modeled).

All candidates — including rejected ones — are saved with their rejection reason for auditability.

### 6.5 Events calendar (`data/events.py`)
Dates for FOMC, CPI, and jobs reports (days-to-event used as features; events inside the strategy window gate candidates).

## 7. Label Simulation (`labels/`) — the most important component

For each entry candidate, walk forward through every later option quote until one condition happens:

```
if close_cost <= entry_credit * 0.50:  outcome = PROFIT_TARGET
elif close_cost >= entry_credit * 2.00: outcome = STOP_LOSS
elif dte <= 7:                          outcome = TIME_EXIT
elif expiry:                            outcome = EXPIRY
```

- **Primary target:** `label_win = 1 if outcome == PROFIT_TARGET else 0`
- **Secondary labels** (captured per trade): final return on risk, max adverse excursion, max favorable excursion, days held, assignment-risk proxy, worst observed mark-to-market loss, exit reason.
- **Conservative fills everywhere:** entry credit below mid; exit debit above mid; per-contract commissions; regulatory/exchange fees where applicable; slippage. Do **not** fill at the mid by default.
- **Skip** trades with unusable quotes (logged).
- Output is **audited manually on 50 randomly selected trades** before the simulator is trusted.

## 8. Feature Engineering (`features/`)

Start with 25–40 understandable features. **Every feature is timestamped**; features for a 3:45 PM decision use only data known at or before 3:45 PM. Scalers are fit on the training period only and applied frozen to validation/test.

**Market:** `spy_return_1d`, `spy_return_5d`, `spy_return_21d`, `spy_realized_vol_5d`, `spy_realized_vol_21d`, `spy_distance_from_ma20`, `spy_distance_from_ma50`, `spy_drawdown_63d`, `vix_close`, `vix_change_5d`, `vix_percentile_252d`

**Option and spread:** `dte`, `short_put_delta`, `short_put_iv`, `short_put_bid_ask_pct`, `short_put_open_interest`, `short_put_volume`, `long_put_delta`, `spread_width`, `credit`, `credit_to_width`, `break_even_distance_pct`, `net_theta`, `net_vega`, `net_gamma`, `spread_bid_ask_pct`

**Calendar/event:** `weekday`, `month`, `days_to_fomc`, `days_to_cpi`, `days_to_jobs_report`

Avoid hundreds of technical indicators — more features means more ways to overfit.

## 9. Models (`models/`)

### 9.1 Baselines — always first
1. **No-trade baseline** — report what happens if no trades occur.
2. **Simple rule** — sell 20-delta, 30–45 DTE, $5-wide SPY put spreads whenever filters pass.
3. **Logistic regression** — transparent probability baseline.
4. **Random forest or gradient boosting** — strong tabular benchmark.

If the neural network cannot beat these after costs and out of sample, **do not use it.**

### 9.2 Neural network (small MLP, not LSTM/Transformer at first)
```
Input: 30–40 normalized numerical features
Dense: 64 units + ReLU · Dropout: 0.15
Dense: 32 units + ReLU · Dropout: 0.10
Dense: 1 unit + Sigmoid
Output: probability of profit target before loss/time exit
```

- Loss: binary cross-entropy · Optimizer: AdamW · LR: 0.0003 · Batch: 128 or 256 · Epochs: up to 100 · Early stopping: patience 10 · Regularization: weight decay + dropout · Class weights only if imbalance requires it · **Record all random seeds.**
- Build a small **NumPy network manually first** to understand forward propagation, loss, and backpropagation before using PyTorch.

### 9.3 Metrics tracked
ROC-AUC · Precision-recall AUC · Brier score · Calibration curve · Accuracy (secondary) · Profit factor after costs · Average return on risk · Win rate · Max drawdown · Sharpe/Sortino (cautiously) · Number of trades · Results by market regime.

### 9.4 Calibration + decision layer
- Calibrate outputs (Platt scaling or isotonic regression), fit on **validation only**, frozen before final test.
- Decide to trade only when **all** pass: `model_probability >= threshold` **and** `expected_value > 0` **and** acceptable spread liquidity **and** portfolio risk limit not exceeded **and** daily trade limit not exceeded **and** event policy passes.
- `expected_value = p_win * expected_profit - (1 - p_win) * expected_loss - estimated_costs`
- Threshold chosen on validation, then **frozen**.

Example output:
```json
{
  "timestamp": "2026-08-26T15:45:00-04:00",
  "strategy": "SPY put credit spread",
  "short_strike": 640,
  "long_strike": 635,
  "expiration": "2026-09-25",
  "model_probability": 0.71,
  "expected_value": 12.40,
  "decision": "PAPER_TRADE",
  "reasons": ["passes probability threshold", "passes liquidity filter"]
}
```

## 10. Risk Engine (`risk/`) — rules outside the model

The model never controls risk limits. Hard-coded rules:

- Paper trade only until prolonged forward testing is complete.
- Max defined loss per trade: 0.5%–1% of paper account.
- Max total open risk: 3%–5%.
- Max trades per day: 1–3.
- No averaging down. No naked options. No new trades when data quality fails.
- **Kill switches:** after daily/weekly loss limits; when live feature distribution differs sharply from training data; when quotes are stale or bid–ask spreads blow out.

Use defined-risk spreads initially. Assignment, early exercise, liquidity, and settlement mechanics still need to be understood.

## 11. Backtest (`backtest/`)

Event-driven enough to process entries, exits, and portfolio state in time order.

Per trade, store: `entry_time · exit_time · entry_credit · exit_debit · gross_pnl · fees · slippage · net_pnl · max_adverse_excursion · max_favorable_excursion · exit_reason · model_probability · feature_version · model_version`

Reports grouped by: month/year · bull/bear/sideways regime · low/med/high VIX · DTE bucket · delta bucket · credit/width bucket · before/after major macro events · probability bucket (50–55%, 55–60%, …, 75%+).

**Calibration audit:** if a claimed 70% probability group wins only 52% out of sample, the model is poorly calibrated.

## 12. Validation Methodology

- **Chronological splits** (config-driven; defaults per spec, adjusted to dataset range):
  Training: 2016–2022 · Validation: 2023 · Test: 2024–2025 · Paper: 2026 onward.
- **Walk-forward:** train 2016–20 → test '21; train 2016–21 → test '22; … The final test period is never used to pick features, thresholds, or hyperparameters.
- **No random shuffling** of financial time series.
- **Leakage guards:** scalers fit on training only; timestamp every raw field and derived feature; use only historical values for IV rank / rolling volatility; never use revised macro data or future option-chain data to fill missing past values.

## 13. Testing Strategy (`tests/`)

- **Unit tests** per module: adapter normalization, validator flags, candidate filters, simulator outcomes, feature no-leakage, split logic, calibration.
- **Integration test:** run the entire pipeline on a tiny synthetic dataset in `tests/fixtures/` and assert outputs — validates every build end-to-end, cheaply.
- **Manual audit:** 50 randomly selected simulated trades reviewed by hand before trusting the simulator.

## 14. Directory Structure

```
D:\PROJECTS\Option\nn-spy\
├── README.md · pyproject.toml · Makefile
├── configs/          data.yaml · strategy.yaml · model.yaml · risk.yaml
├── data/
│   ├── adapters/     cboe_adapter.py · kaggle_adapter.py · feasibility_check.py
│   ├── download.py · validate.py · candidates.py · events.py
│   └── raw/          underlying/ · option_chains/ · vix/ · events/   (immutable)
├── features/         features.py · scalers.py
├── labels/           simulator.py · outcomes.py
├── models/           baselines.py · numpy_net.py · mlp.py · calibrate.py · select_threshold.py
├── backtest/         engine.py · ledger.py · costs.py
├── risk/             risk.py
├── reports/          metrics.py · charts.py · daily_report.py · monthly_report.py
├── paper_trade/      daily_signal.py
├── experiments/      logs: seeds · metrics · model versions
└── tests/            unit/ · integration/ · fixtures/ (tiny synthetic dataset)
```

Use Parquet for option-chain histories (not CSV). Tooling: Python 3.11, PyTorch, pandas/Polars, NumPy, scikit-learn, DuckDB, Matplotlib, Git.

## 15. Build Order (one controlled change at a time)

1. **Scaffold** — repo, virtual environment, dependencies, configs.
2. **Data** — feasibility check → download sample dataset → adapters → canonical schema → validation.
3. **Candidates + simulator** — candidate filters, event calendar, outcome simulation, audit 50 trades.
4. **Features + splits** — timestamped features, walk-forward splitter.
5. **Baselines** — rule → logistic regression → gradient boosting; first walk-forward report.
6. **Neural net** — NumPy net → PyTorch MLP → calibration → frozen threshold.
7. **Full validation** — stress costs/slippage, regime analysis, feature drift, freeze model.
8. **Paper trading** — daily signals, logged decisions (incl. NO TRADE), scheduled retraining (e.g. monthly/quarterly).

Only consider live execution after a meaningful paper-trading period and after verifying that broker data, fills, and realized results match the research assumptions.

## 16. MVP Scope (Section 15 of the source spec)

Before expanding scope: SPY · $5-wide put credit spread · 30–45 DTE · short strike near 20 delta · daily decisions · logistic regression → gradient boosting → small MLP · target = 50% profit before 2×-credit stop or 7-DTE exit · chronological walk-forward validation · paper execution only. Once robust, add call spreads, alternative DTEs, volatility forecasting, or other underlyings — one controlled change at a time.
