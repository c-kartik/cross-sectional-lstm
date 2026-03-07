from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.backtest.engine import run_backtest
from src.data.load_prices import compute_returns, load_prices
from src.datasets.make_dataset import DatasetConfig, make_sequences
from src.datasets.splits import time_split_dates
from src.features.build_features import build_features
from src.models.lstm import LSTMRegressor
from src.optimize.mean_variance import MVConfig, mean_variance_weights
from src.reporting.optimizer_report import generate_optimizer_report
from src.signals.trend import above_sma
from src.universe.custom import get_universe
from src.utils.profiling import RunProfiler
from src.utils.run_manifest import write_run_manifest

TRADING_DAYS = 252


def annualized_vol_from_returns(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.Series(index=index, data=1).resample("W-FRI").last().index


def load_lstm_scores_test(df_feat_label: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    dates = df_feat_label.index.get_level_values("date").unique().sort_values()
    train_end, val_end = time_split_dates(dates)

    Xte, _, keys = make_sequences(
        df_feat_label,
        cfg,
        dates_keep=(val_end + pd.Timedelta(days=1), pd.Timestamp("2100-01-01")),
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LSTMRegressor(n_features=Xte.shape[-1], hidden_size=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load("data/datasets/lstm_best.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(Xte).to(device)).cpu().numpy()

    pred_df = pd.DataFrame(
        {"pred": preds.astype(float)},
        index=pd.MultiIndex.from_tuples(keys, names=["date", "ticker"]),
    ).sort_index()
    return pred_df


if __name__ == "__main__":
    print("[experimental] Running LSTM + optimizer R&D workflow (not headline baseline).")
    profiler = RunProfiler("run_lstm_optimize_portfolio")
    # ------------------------
    # Data
    # ------------------------
    with profiler.stage("load_data"):
        prices = load_prices()
        price_col = "adj_close" if "adj_close" in prices.columns else "close"
        px_all = prices[price_col].unstack("ticker").sort_index()
        rets_all = compute_returns(prices, price_col=price_col).unstack("ticker").sort_index()

        universe = [t for t in get_universe() if t in px_all.columns]
        px = px_all[universe].dropna(how="all")

    with profiler.stage("build_features_and_predictions"):
        # Build features table (same as before)
        Xtab = build_features(px)
        df_feat = Xtab.copy()
        df_feat["fwd_ret_5d"] = 0.0  # dummy label for sequences
        cfg = DatasetConfig(seq_len=60, label_col="fwd_ret_5d")
        pred_df = load_lstm_scores_test(df_feat, cfg)

    trade_tickers = [t for t in universe if t not in ("SPY", "QQQ")]

    # ------------------------
    # Optimization config
    # ------------------------
    mv_cfg = MVConfig(
        risk_aversion=0.75,
        turnover_penalty=1.0,
        max_weight=0.20,
        long_only=True,
        leverage=1.0,
        alpha_scale=5.0,
        cov_shrinkage=0.25,
        min_weight_top=0.01,
    )

    top_n = 15
    buffer_k = 10
    opt_top_k = 30
    target_vol_annual = 0.25
    vol_lookback = 60
    max_leverage = 2.0
    cost_bps = 10.0
    sma_window_regime = 100

    # Restrict to test dates (dates where we have predictions)
    test_dates = pred_df.index.get_level_values("date").unique().sort_values()
    start_dt = test_dates.min()
    end_dt = test_dates.max()

    # SPY regime series (daily)
    if "SPY" in px_all.columns:
        spy_ok = above_sma(px_all[["SPY"]], window=sma_window_regime)["SPY"]
    else:
        spy_ok = pd.Series(True, index=px_all.index)

    # Rebalance dates (weekly) within test window
    rebals = weekly_rebalance_dates(px_all.loc[start_dt:end_dt].index)

    # sparse weights (optimizer)
    w = pd.DataFrame(np.nan, index=px_all.loc[start_dt:end_dt].index, columns=trade_tickers)
    prev_w = pd.Series(0.0, index=trade_tickers)

    # sparse weights (baseline top-N)
    w_base = pd.DataFrame(np.nan, index=px_all.loc[start_dt:end_dt].index, columns=trade_tickers)
    current_holdings: set[str] = set()

    with profiler.stage("build_portfolios"):
        for dt in rebals:
            if dt not in w.index:
                prev = w.index[w.index <= dt]
                if len(prev) == 0:
                    continue
                dt = prev[-1]

            if not bool(spy_ok.loc[dt]):
                w.loc[dt] = 0.0
                prev_w = pd.Series(0.0, index=trade_tickers)
                w_base.loc[dt] = 0.0
                current_holdings = set()
                continue

            preds_today = pred_df.xs(dt, level="date")["pred"].reindex(trade_tickers).fillna(0.0)

            # ----- baseline top-N with buffer -----
            ranks = preds_today.rank(ascending=False, method="first")
            sell_cutoff = top_n + buffer_k
            to_keep = {t for t in current_holdings if (t in ranks.index and ranks.loc[t] <= sell_cutoff)}
            topN = ranks[ranks <= top_n].sort_values().index.tolist()
            new_holdings = set(to_keep)
            for t in topN:
                if len(new_holdings) >= top_n:
                    break
                new_holdings.add(t)
            if len(new_holdings) < top_n:
                for t in ranks.sort_values().index.tolist():
                    if len(new_holdings) >= top_n:
                        break
                    new_holdings.add(t)
            current_holdings = new_holdings

            hold_list = list(current_holdings)
            scores = preds_today.reindex(hold_list)
            r = scores.rank(ascending=False, method="first")
            raw = (len(hold_list) + 1 - r).astype(float).clip(lower=0.0)
            base = pd.Series(0.0, index=trade_tickers)
            if raw.sum() > 0:
                base.loc[hold_list] = (raw / raw.sum()).values
            else:
                base.loc[hold_list] = 1.0 / max(len(hold_list), 1)

            hist = rets_all.loc[:dt, trade_tickers].dropna(how="all")
            if len(hist) >= vol_lookback + 1:
                hist_window = hist.tail(vol_lookback)
                port_hist = (hist_window.fillna(0.0) * base).sum(axis=1)
                vol_daily = float(port_hist.std(ddof=0))
                target_daily = target_vol_annual / np.sqrt(252)
                lev = min(max_leverage, target_daily / vol_daily) if vol_daily > 0 else 1.0
            else:
                lev = 1.0
            w_base.loc[dt] = base * lev

            # ----- optimizer weights (restricted to top-K by alpha) -----
            opt_names = ranks.sort_values().index.tolist()[:opt_top_k]
            top_names = ranks[ranks <= top_n].sort_values().index.tolist()
            hist = rets_all.loc[:dt, opt_names].dropna(how="all")
            if len(hist) < vol_lookback + 1:
                continue

            hist_window = hist.tail(vol_lookback).fillna(0.0)
            cov = hist_window.cov().values
            alpha = preds_today.reindex(opt_names).values

            prev_sub = prev_w.reindex(opt_names).fillna(0.0).values
            top_mask = np.array([t in top_names for t in opt_names], dtype=bool)
            weights = mean_variance_weights(alpha, cov, w_prev=prev_sub, top_mask=top_mask, cfg=mv_cfg)
            base = pd.Series(0.0, index=trade_tickers)
            base.loc[opt_names] = weights

            port_hist = (hist_window * base).sum(axis=1)
            vol_daily = float(port_hist.std(ddof=0))
            target_daily = target_vol_annual / np.sqrt(252)
            lev = min(max_leverage, target_daily / vol_daily) if vol_daily > 0 else 1.0

            w.loc[dt] = base * lev
            prev_w = w.loc[dt].fillna(0.0)

    with profiler.stage("backtest"):
        rets_test = rets_all.loc[w.index, trade_tickers]
        targets = w[trade_tickers].shift(1)
        result = run_backtest(rets_test, targets, cost_bps=cost_bps)

        rets_test_base = rets_all.loc[w_base.index, trade_tickers]
        targets_base = w_base[trade_tickers].shift(1)
        result_base = run_backtest(rets_test_base, targets_base, cost_bps=cost_bps)

        # Benchmark equity (same dates as optimizer)
        qqq_eq = (1.0 + rets_all["QQQ"].loc[w.index].fillna(0.0)).cumprod()
        spy_eq = (1.0 + rets_all["SPY"].loc[w.index].fillna(0.0)).cumprod()

    run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("reports") / "lstm_optimize" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with profiler.stage("write_outputs"):
        eq = result["equity"]
        eq.to_csv(run_dir / "equity.csv", header=True)
        plt.figure()
        eq.plot(title="LSTM + Mean-Variance Optimized Equity")
        plt.xlabel("Date")
        plt.ylabel("Equity (start=1.0)")
        plt.tight_layout()
        plt.savefig(run_dir / "equity.png", dpi=150)
        plt.close()

        eq_base = result_base["equity"]
        eq_base.to_csv(run_dir / "equity_baseline.csv", header=True)
        plt.figure()
        eq_base.plot(title="LSTM Top-N Baseline Equity")
        plt.xlabel("Date")
        plt.ylabel("Equity (start=1.0)")
        plt.tight_layout()
        plt.savefig(run_dir / "equity_baseline.png", dpi=150)
        plt.close()

        qqq_eq.to_csv(run_dir / "equity_qqq.csv", header=True)
        spy_eq.to_csv(run_dir / "equity_spy.csv", header=True)

    summary = {
        "risk_aversion": mv_cfg.risk_aversion,
        "turnover_penalty": mv_cfg.turnover_penalty,
        "max_weight": mv_cfg.max_weight,
        "long_only": mv_cfg.long_only,
        "leverage": mv_cfg.leverage,
        "target_vol_annual": target_vol_annual,
        "vol_lookback": vol_lookback,
        "max_leverage": max_leverage,
        "cost_bps": cost_bps,
        "top_n": top_n,
        "buffer_k": buffer_k,
        "opt_top_k": opt_top_k,
        "alpha_scale": mv_cfg.alpha_scale,
        "cov_shrinkage": mv_cfg.cov_shrinkage,
        "min_weight_top": mv_cfg.min_weight_top,
    }
    (run_dir / "config.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "stats.json").write_text(json.dumps(result["stats"], indent=2))
    (run_dir / "stats_baseline.json").write_text(json.dumps(result_base["stats"], indent=2))
    # Save turnover series if present
    if "daily_turnover" in result:
        result["daily_turnover"].to_csv(run_dir / "turnover.csv", header=True)
    if "daily_turnover" in result_base:
        result_base["daily_turnover"].to_csv(run_dir / "turnover_baseline.csv", header=True)

    manifest_path = write_run_manifest(
        run_dir,
        script_name="src.experiments.run_lstm_optimize_portfolio",
        run_id=run_id,
        config=summary,
        input_paths=[Path("data/prices"), Path("data/datasets/lstm_best.pt")],
        extra={"price_col": price_col, "universe_count": len(universe), "trade_ticker_count": len(trade_tickers)},
    )
    print(f"Saved run manifest to: {manifest_path}")
    report_path = generate_optimizer_report(run_dir)
    print(f"Saved optimizer HTML report to: {report_path}")
    csv_path, png_path = profiler.write_artifacts(run_dir)
    print(f"Saved profiling summary to: {csv_path}")
    print(f"Saved profiling chart to: {png_path}")
    print(f"Saved optimized run to: {run_dir}")
