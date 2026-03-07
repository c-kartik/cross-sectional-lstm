# PROJECT_BRIEF

## Project
Cross-Sectional LSTM Equity Research Platform (local-first, reproducible)

## Objective
Build an end-to-end quant equity research system that:
- produces cross-sectional alpha forecasts from daily equity data,
- converts forecasts into executable portfolios with realistic frictions,
- validates performance out-of-sample with walk-forward retraining,
- and maintains engineering discipline (tests, CI, reproducibility, profiling).
- supports full lifecycle ownership: research workflow, implementation, reporting, and operational repeatability.

## System scope
- Data ingestion and cache:
  - yfinance daily OHLCV ingestion to per-ticker Parquet cache.
  - Data quality validation with NYSE trading-calendar-aware checks.
- Research pipeline:
  - Feature engineering on `(date, ticker)` panel.
  - LSTM model to predict forward 5-day returns for cross-sectional ranking.
- Portfolio construction:
  - Headline strategy: Top-N long-only with buffer/hysteresis, rank weighting, volatility targeting, T+1 execution.
  - Experimental module: mean-variance optimizer (cvxpy), rank-aware constraints, covariance shrinkage.
- Evaluation:
  - Cost-aware backtesting with turnover, gross/net exposure, and drawdown.
  - Walk-forward yearly retraining/testing with multi-seed aggregation.
  - HTML reports for walk-forward and optimizer experiments.

## Role alignment (Quantitative Developer)
- End-to-end software ownership:
  - data pipeline, model training/inference, portfolio construction, backtest engine, reporting, and CI.
- Collaboration-ready outputs:
  - run artifacts and HTML reports designed for PM/research review.
- Engineering judgment in investment context:
  - baseline and optimizer both implemented, then selected baseline for headline path based on out-of-sample evidence.

## Research integrity controls
- No-lookahead execution: rebalance targets shifted by one trading day.
- Cost model: bps costs applied on turnover.
- Walk-forward protocol: train/val strictly precedes each OOS test year.
- Data QA: missing data, stale data, duplicate timestamps, non-positive prices, extreme-move flags.

## Reproducibility and engineering controls
- Run manifests (`run_manifest.json`) emitted for main experiments with:
  - script name, UTC timestamp, run ID,
  - git commit + branch + tracked-dirty flag,
  - runtime metadata (Python/platform/cwd),
  - config/seeds and input-data fingerprint.
- CI:
  - GitHub Actions runs `pytest -q` on push/PR.
- Tests:
  - Backtest invariants, optimizer constraint checks, data-validation checks, profiling utility tests.
- Profiling artifacts:
  - stage-level runtime and memory snapshots (`profiling_summary.csv`, `profiling_stages.png`).

## Technology stack
- Core: Python, pandas, numpy, matplotlib.
- ML: PyTorch (LSTM alpha model).
- Optimization: cvxpy (mean-variance R&D path).
- Data: yfinance + Parquet local cache.
- Dev quality: pytest + GitHub Actions CI.

## Results snapshot (current)

### Walk-forward (headline Top-N baseline, 3 seeds x 4 years)
Source: `reports/walk_forward/run_20260303_041320`
- Overall:
  - mean Sharpe: 1.173
  - mean IR vs QQQ: 0.530
  - win rate vs QQQ: 66.7%
  - win rate vs SPY: 41.7%
- By year:
  - 2022: beats QQQ in all seeds but negative absolute return year.
  - 2023: strong outperformance vs both QQQ and SPY.
  - 2024: mixed edge.
  - 2025: underperforms QQQ/SPY in all seeds.

### Optimizer vs baseline (single OOS test window)
Source: `reports/lstm_optimize/run_20260307_011648`
- Optimizer:
  - Sharpe 2.103, total return 39.66%, max drawdown -13.80%.
- Baseline Top-N:
  - Sharpe 2.958, total return 63.52%, max drawdown -11.62%.
- Decision:
  - Keep optimizer as R&D module; keep Top-N baseline as headline strategy.

## Key technical tradeoff decisions
- Did not force optimizer into production path because empirical OOS evidence favored baseline.
- Prioritized anti-lookahead and realistic execution over headline in-sample gains.
- Added reproducibility/profiling infrastructure before expanding model complexity.

## Known limitations
- OOS sample length is still limited; confidence in long-run edge is moderate.
- Universe and alpha calibration can drift across regimes (2025 underperformance vs QQQ).
- Optimizer objective remains sensitive to covariance estimation noise.

## Next technical milestones
- Extend walk-forward horizon and rolling windows for stronger statistical confidence.
- Add attribution layer (name/factor contribution by year).
- Add configurable optimizer hyperparameter sweeps to reports (not only CSV).
- Add e2e integration test for artifact contract validation.

## Interview talking points
- Why baseline remains primary:
  - optimizer improved flexibility but did not beat baseline risk-adjusted OOS performance; kept as R&D, not production path.
- How model risk is handled:
  - anti-lookahead execution, turnover-aware cost modeling, walk-forward retraining, and explicit known-limit documentation.
- How engineering quality is enforced:
  - CI test gate, deterministic run manifests, and profiling artifacts for bottleneck visibility.

## Resume-ready bullets
- Built a local-first quant equity research platform in Python spanning data ingestion, feature engineering, LSTM alpha modeling, and cost-aware walk-forward backtesting.
- Implemented anti-lookahead portfolio execution (T+1), turnover-based transaction costs, and volatility-targeted portfolio sizing.
- Designed reproducibility controls via run manifests capturing git state, config/seeds, and input-data fingerprints for every experiment.
- Added CI and automated tests for backtest math invariants, optimizer constraints, and data-quality validation.
- Developed experiment reporting with HTML dashboards and profiling artifacts (runtime/memory by stage) to support model-risk and performance review.
- Evaluated mean-variance optimization (cvxpy) against Top-N baseline; retained baseline as primary strategy based on out-of-sample risk-adjusted performance.
