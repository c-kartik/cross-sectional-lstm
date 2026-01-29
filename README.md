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

Outputs saved to: `reports/walk_forward_<timestamp>/` (equity PNG/CSV + summary CSVs)

