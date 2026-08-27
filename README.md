# nn-spy

A research system that estimates the **calibrated probability** that a SPY put
credit spread reaches its 50% profit target before its stop loss or forced exit.

The output is a probability, not a signal. **Paper trading only — no live
execution, and nothing here is a trade recommendation.** Options can lose money
quickly; read the current OCC Options Disclosure Document before trading.

---

## The strategy being modelled

| | |
|---|---|
| Instrument | SPY put credit spread, $5 wide |
| Entry | Short put near −0.20 delta, long put $5 below, 30–45 DTE |
| Decision | Once per day, 3:45 PM ET |
| Take profit | Close at 50% of the credit received |
| Stop loss | Close at 2× the credit received |
| Time exit | Close at 7 days to expiry |
| Label | `1` if the profit target is reached first, else `0` |

At $2.00 credit that pays **+$1.00** on a win and **−$2.00** on a loss, so the
break-even win rate is **66.7%**, not 50%. Every threshold and expected-value
calculation in the codebase uses that asymmetry.

## Pipeline

```
raw source rows
  └─ data/adapters      normalize any vendor into one canonical schema
  └─ data/validate      flag bad records, never edit them
  └─ data/candidates    20-delta / $5-wide / 30-45 DTE, rejections recorded
  └─ labels/simulator   walk forward to PROFIT_TARGET | STOP_LOSS | TIME_EXIT
                        | EXPIRY | NO_DATA  ->  label_win
  └─ features           33 timestamped features, missing values stay NaN
  └─ features/splits    chronological walk-forward with a 38-day embargo
  └─ models             baselines -> MLP -> calibrate on validation -> freeze
  └─ backtest/engine    walk-forward, real costs, full trade ledger
  └─ risk               hard limits and kill switches, outside the model
  └─ reports            calibration audit and regime breakdowns
  └─ paper_trade        daily decision record, including NO_TRADE
```

## Quick start

```bash
make setup
```

```bash
make test
```

281 tests, unit and integration. They run on every push and pull request via
`.github/workflows/ci.yml`, which also smoke-tests the full pipeline end to end:

```bash
python run_demo.py
```

`run_demo.py` generates a synthetic Black-Scholes chain and walks it through
every stage. It is an installation check, not a backtest — the P&L it prints
describes a random walk, not SPY.

Before committing to any dataset, check whether it can support the strategy:

```bash
python -m data.adapters.feasibility_check path/to/chains.parquet
```

It reports date range, DTE coverage, whether Greeks and IV exist, and how many
days actually quote both legs of a $5 spread — and names the blockers if not.
Free sample option data frequently has no Greeks, which makes 20-delta strike
selection impossible. Find that out first, not after building on top of it.

## Design rules

These are enforced in code, not just documented:

- **Raw data is immutable.** `validate.py` stamps `quality_flags`; nothing edits
  a quote.
- **Missing is NaN, never zero.** A 50-day moving average with 10 bars of
  history is unknown, not "zero distance from the average". `assert_no_nan`
  makes handling it a decision.
- **No lookahead.** Features filter to `timestamp <= as_of` before computing.
  `assert_no_lookahead` fails loudly otherwise.
- **The embargo is real.** A trade entered at 45 DTE resolves up to 38 days
  later, so training rows within 38 days of a fold boundary are purged. Without
  that gap their labels are built from test-period quotes.
- **Scalers and calibrators freeze.** `FitScaler.freeze()` and
  `Calibrator.freeze()` raise on refit. Calibration is fit on **validation**,
  never on training.
- **Conservative fills.** Sell the bid, buy the ask, pay slippage on each leg,
  and charge commission on 4 contract-legs per round trip.
- **Return on risk uses `width − credit`.** A $5 spread sold for $2 risks $3.
- **The model never sets risk limits.** `risk/risk.py` is deterministic.
- **Baselines first.** The MLP must beat logistic regression and gradient
  boosting out of sample, after costs, or it is not used.
- **Every seed is recorded** — and `build_mlp` seeds before weight
  initialisation, so a recorded seed actually reproduces its run.

## Reading a result

Check these before believing any number:

1. **Base rate.** `label_distribution()` reports it. If it is 0.75, then 75%
   accuracy is what you get for always predicting "win".
2. **Calibration.** `by_prob_bucket()` compares predicted to realised win rate
   per bucket. A 0.70 bucket that wins 52% of the time is a broken model, however
   good its AUC.
3. **Costs.** Every ledger row carries its own fees. A strategy that is
   profitable before costs and not after is not a strategy.
4. **Trade count.** A threshold chosen on nine validation trades is noise;
   `select_threshold` enforces a minimum.

## Status

The pipeline, the models and the test suite are complete and passing (281
tests). What is **not** done, and what to do next:

- **No real data.** `configs/data.yaml` carries placeholder dates. Run the
  feasibility check on a real dataset and set the range from what it reports.
- **The event calendar is a stub.** `data/events.py` ships a small seed covering
  2024–2025 and declares that window; `assert_covers()` refuses periods outside
  it. Load real Fed/BLS dates into `data/raw/events/events.csv` before training
  on anything earlier.
- **No result on real data means anything yet.** The synthetic fixture exercises
  the plumbing. It says nothing about whether the strategy works.

## Layout

```
schemas.py · config.py · pipeline.py
configs/     data · strategy · model · risk  (YAML)
data/        adapters/ (base, sample, feasibility_check) · download · validate
             candidates · events
features/    features · scalers · splits
labels/      simulator · outcomes · audit
models/      baselines · numpy_net · mlp · calibrate · select_threshold
backtest/    engine · ledger · costs
risk/        risk
reports/     metrics · daily_report · monthly_report
paper_trade/ daily_signal
tests/       unit/ · integration/ · fixtures/
```
