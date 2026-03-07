# Cross-Sectional LSTM Equity Selector (with Walk-Forward Backtesting)

## Quant Research Project (Risk Engines + LSTM Alpha + Walk-Forward)

A single-user, local-first quant research project that:

- Pulls daily equity data via **yfinance**
- Caches OHLCV data as **Parquet**
- Runs **cost-aware backtests** with turnover + exposure metrics
- Implements **risk engines** (trend/vol-targeting, inverse-vol baseline)
- Implements an **LSTM alpha module** for **cross-sectional stock ranking**
- Validates with **walk-forward retraining** (annual refits, out-of-sample years)

> Research sandbox only — not a live trading system. **This is not financial advice**

---

## What does this project do?

### 1) Data ingestion + caching

- Downloads price history for a ticker universe (yfinance)
- Normalizes columns and stores per-ticker Parquet caches
- Loads cached data into a unified `(date, ticker)` MultiIndex for research and modeling

#### 1A) Universe + horizon + rebalance frequency

- **Universe:** user-defined list in `src/universe/custom.py`
- **Label horizon:** 5 trading days (`fwd_ret_5d`)
- **Rebalance:** weekly (Fri close → trade next session)
- **Portfolio:** Top-N long-only, rank-weighted, buffer rule
- **Risk:** volatility targeting + leverage cap + SPY SMA regime filter

### 2) Backtest engine (cost-aware)

- Converts sparse “rebalance-day targets” into daily holdings (ffill)
- Applies transaction costs (bps) based on turnover
- Reports core metrics:
  - Sharpe, Max Drawdown, Total Return
  - Avg gross exposure, Avg net exposure
  - Avg weekly turnover, Total cost paid

### 3) Risk engines (baseline portfolios)

- **Trend-following (long-only) + vol targeting**
  - Uses an SMA regime filter (SPY) to reduce exposure in down regimes
  - Targets a desired annualized volatility using trailing realized vol
- **Inverse-volatility baseline (risk parity-ish) + regime + vol targeting**
  - Allocates by inverse trailing volatility
  - Optionally gates exposure using SPY regime
  - Targets volatility similarly to the trend risk engine
  - **I use inverse-vol weighting, not equal-risk-contribution optimization using the full covariance (correlations).**

### 4) LSTM alpha (cross-sectional ranking)

- Builds a feature table (MultiIndex: `(date, ticker)`) with:
  - recent returns, rolling vol, SMA-relative price features, drawdown features, etc.
- Trains an LSTM to predict **forward returns** (e.g., `fwd_ret_5d`)
- Uses predictions to rank stocks cross-sectionally (best → worst)
- Constructs a **Top-N long-only portfolio** with:
  - weekly rebalancing
  - **buffer/hysteresis rule** (reduces churn)
  - rank-weighting within holdings
  - volatility targeting
  - transaction costs
- Compares performance vs **SPY** and **QQQ**, including **risk-matched benchmark** curves

### 5) Walk-forward evaluation

- Retrains a fresh model each year (train/val → test year)
- Produces year-by-year out-of-sample metrics and an aggregated summary
- Computes benchmark-aligned metrics including IR vs QQQ

---

## Key concepts implemented

- **Cross-sectional ranking**: predict relative winners/losers across many tickers on each rebalance date
- **Alpha vs risk**:
  - *Alpha module*: produces scores/ranks (predicted forward return)
  - *Risk engine*: controls exposure (vol targeting, regime filters, leverage caps, turnover-aware sizing)
- **Turnover-aware portfolio construction**:
  - buffer/hysteresis rule keeps names unless they fall below rank `N + buffer`
  - only buys new names when they enter the top `N`
- **Cost-aware backtesting**: costs scale with turnover (bps model)
- **Walk-forward validation**: avoids “train once, test once” overfitting patterns

---

## Setup / How to run

> Run commands from the project root with your venv active.

### Data download + cache

```bash
python3 -m src.data.download_prices
python3 -m src.data.inspect_cache
python3 -m src.data.validate_prices
```

### Risk Engines (baselines)

```bash
python3 -m src.experiments.run_trend_vol
python3 -m src.experiments.run_risk_parity
```

### LSTM Pipeline

```bash
python3 -m src.experiments.run_make_dataset
python3 -m src.experiments.run_build_lstm_dataset
python3 -m src.experiments.run_lstm_train
python3 -m src.experiments.run_lstm_eval_ic
python3 -m src.experiments.run_lstm_topn_backtest
```

### Mainline evaluation (default / headline results)

Use the Top-N baseline as the primary portfolio construction layer:

```bash
python3 -m src.experiments.run_walk_forward
# or
make run-walkforward
```

Outputs a shareable HTML report at:
`reports/walk_forward/run_YYYYMMDD_HHMMSS/report.html`
Each run now also writes `run_manifest.json` with config, git commit/dirty state,
runtime metadata, seeds, and input data fingerprint.
Profiling artifacts are also saved as `profiling_summary.csv` and `profiling_stages.png`.

### Experimental optimization layer (R&D only)

```bash
python3 -m src.experiments.run_lstm_optimize_portfolio
python3 -m src.experiments.run_lstm_optimize_grid
# or
make run-opt-rd
make run-opt-grid
```

This optimizer path is kept for research and skill demonstration. It is
currently not the default headline strategy because baseline Top-N has been
more robust in recent runs.

Reports:
- `reports/lstm_optimize/run_YYYYMMDD_HHMMSS/report.html`
- `reports/lstm_optimize_grid/run_YYYYMMDD_HHMMSS/grid_summary.csv`
- `run_manifest.json` in each run folder for reproducibility/provenance
- `profiling_summary.csv` + `profiling_stages.png` in each run folder

### Tests

```bash
pytest -q
# or
make test
```

CI runs this test suite automatically on push/PR via GitHub Actions.

### Reproducibility controls

- All main experiments emit `run_manifest.json` with:
  - script name + run ID + UTC timestamp
  - git commit hash + branch + tracked-dirty flag
  - runtime context (Python/platform/cwd)
  - seeds/config used for the run
  - fingerprint of input data/model paths
- One-command wrappers are provided in `Makefile` for common workflows.

---
## Current state of the project

Implemented so far:

- Local data pipeline + Parquet caching
- Backtest engine with costs + turnover + exposure stats
- Trend/vol-target risk engine baseline(s)
- Inverse-volatility baseline
- LSTM alpha training + IC evaluation
- LSTM Top-N strategy with:
  - buffer/hysteresis
  - rank weighting
  - vol targeting
  - benchmark + risk-matched comparisons
- Walk-forward evaluation across multiple years (annual refits)
- Mean-variance optimizer layer (cvxpy) implemented as experimental R&D module

## Research integrity (anti-lookahead + realism)
- **No lookahead execution:** targets are applied **T+1** (rebalance signals shifted by 1 trading day).
- **Transaction costs:** bps cost model applied on turnover.
- **Walk-forward:** model is retrained each year using only past data (train/val → test year).
- **Turnover controls:** buffer/hysteresis keeps names unless rank deteriorates past `N + buffer`.

## Practical Notes

- Training on Apple Silicon can hit memory limits depending on batch size + dataset size, if so:
  - reduce `batch` size in training,
  - reduce universe size,
  - or force CPU training.

----------------------------------

## Quickstart
> Run commands from the project root with your venv active.

### Requirements
- Python 3.11+ (tested on 3.12)
- Key libs: pandas, numpy, torch, yfinance, matplotlib

### Download + cache data

```bash
python3 -m src.data.download_prices
```

### Main strategy (Walk-forward OOS)

```bash
python3 -m src.experiments.run_walk_forward
```

Outputs saved to: `reports/walk_forward/run_<timestamp>/` (equity PNG/CSV + summary CSVs)
