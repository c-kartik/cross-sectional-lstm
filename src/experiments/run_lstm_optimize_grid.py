from __future__ import annotations

import json
from pathlib import Path

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
from src.signals.trend import above_sma
from src.universe.custom import get_universe


TRADING_DAYS = 252


def weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.Series(index=index, data=1).resample("W-FRI").last().index


def load_lstm_scores_test(df_feat_label: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    dates = df_feat_label.index.get_level_values("date").unique().sort_values()
    _, val_end = time_split_dates(dates)

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


def build_baseline_weights(
    pred_df: pd.DataFrame,
    rets_all: pd.DataFrame,
    trade_tickers: list[str],
    spy_ok: pd.Series,
    top_n: int,
    buffer_k: int,
    target_vol_annual: float,
    vol_lookback: int,
    max_leverage: float,
) -> pd.DataFrame:
    test_dates = pred_df.index.get_level_values("date").unique().sort_values()
    start_dt = test_dates.min()
    end_dt = test_dates.max()
    rebals = weekly_rebalance_dates(rets_all.loc[start_dt:end_dt].index)

    w_base = pd.DataFrame(np.nan, index=rets_all.loc[start_dt:end_dt].index, columns=trade_tickers)
    current_holdings: set[str] = set()

    for dt in rebals:
        if dt not in w_base.index:
            prev = w_base.index[w_base.index <= dt]
            if len(prev) == 0:
                continue
            dt = prev[-1]

        if not bool(spy_ok.loc[dt]):
            w_base.loc[dt] = 0.0
            current_holdings = set()
            continue

        preds_today = pred_df.xs(dt, level="date")["pred"].reindex(trade_tickers).fillna(0.0)
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

    return w_base


def run_optimizer_once(
    pred_df: pd.DataFrame,
    rets_all: pd.DataFrame,
    trade_tickers: list[str],
    spy_ok: pd.Series,
    mv_cfg: MVConfig,
    opt_top_k: int,
    top_n: int,
    target_vol_annual: float,
    vol_lookback: int,
    max_leverage: float,
) -> pd.DataFrame:
    test_dates = pred_df.index.get_level_values("date").unique().sort_values()
    start_dt = test_dates.min()
    end_dt = test_dates.max()
    rebals = weekly_rebalance_dates(rets_all.loc[start_dt:end_dt].index)

    w = pd.DataFrame(np.nan, index=rets_all.loc[start_dt:end_dt].index, columns=trade_tickers)
    prev_w = pd.Series(0.0, index=trade_tickers)

    for dt in rebals:
        if dt not in w.index:
            prev = w.index[w.index <= dt]
            if len(prev) == 0:
                continue
            dt = prev[-1]

        if not bool(spy_ok.loc[dt]):
            w.loc[dt] = 0.0
            prev_w = pd.Series(0.0, index=trade_tickers)
            continue

        preds_today = pred_df.xs(dt, level="date")["pred"].reindex(trade_tickers).fillna(0.0)
        ranks = preds_today.rank(ascending=False, method="first")
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

        port_hist = (hist_window * base.reindex(opt_names)).sum(axis=1)
        vol_daily = float(port_hist.std(ddof=0))
        target_daily = target_vol_annual / np.sqrt(252)
        lev = min(max_leverage, target_daily / vol_daily) if vol_daily > 0 else 1.0

        w.loc[dt] = base * lev
        prev_w = w.loc[dt].fillna(0.0)

    return w


if __name__ == "__main__":
    print("[experimental] Running optimizer grid search for R&D only (not headline baseline).")
    prices = load_prices()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px_all = prices[price_col].unstack("ticker").sort_index()
    rets_all = compute_returns(prices, price_col=price_col).unstack("ticker").sort_index()

    universe = [t for t in get_universe() if t in px_all.columns]
    px = px_all[universe].dropna(how="all")

    Xtab = build_features(px)
    df_feat = Xtab.copy()
    df_feat["fwd_ret_5d"] = 0.0
    cfg = DatasetConfig(seq_len=60, label_col="fwd_ret_5d")
    pred_df = load_lstm_scores_test(df_feat, cfg)

    trade_tickers = [t for t in universe if t not in ("SPY", "QQQ")]
    spy_ok = above_sma(px_all[["SPY"]], window=100)["SPY"] if "SPY" in px_all.columns else pd.Series(True, index=px_all.index)

    top_n = 15
    buffer_k = 10
    opt_top_k = 30
    target_vol_annual = 0.25
    vol_lookback = 60
    max_leverage = 2.0
    cost_bps = 10.0

    w_base = build_baseline_weights(
        pred_df,
        rets_all,
        trade_tickers,
        spy_ok,
        top_n=top_n,
        buffer_k=buffer_k,
        target_vol_annual=target_vol_annual,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
    )
    rets_test_base = rets_all.loc[w_base.index, trade_tickers]
    targets_base = w_base[trade_tickers].shift(1)
    result_base = run_backtest(rets_test_base, targets_base, cost_bps=cost_bps)

    grid = [
        {"risk_aversion": 0.5, "turnover_penalty": 1.0, "alpha_scale": 10.0, "cov_shrinkage": 0.4, "min_weight_top": 0.02},
        {"risk_aversion": 0.4, "turnover_penalty": 0.5, "alpha_scale": 10.0, "cov_shrinkage": 0.4, "min_weight_top": 0.02},
        {"risk_aversion": 0.3, "turnover_penalty": 0.5, "alpha_scale": 15.0, "cov_shrinkage": 0.5, "min_weight_top": 0.03},
        {"risk_aversion": 0.5, "turnover_penalty": 0.5, "alpha_scale": 15.0, "cov_shrinkage": 0.3, "min_weight_top": 0.02},
        {"risk_aversion": 0.4, "turnover_penalty": 0.2, "alpha_scale": 20.0, "cov_shrinkage": 0.5, "min_weight_top": 0.03},
        {"risk_aversion": 0.6, "turnover_penalty": 0.5, "alpha_scale": 10.0, "cov_shrinkage": 0.25, "min_weight_top": 0.02},
    ]

    rows = []
    for params in grid:
        mv_cfg = MVConfig(
            risk_aversion=params["risk_aversion"],
            turnover_penalty=params["turnover_penalty"],
            max_weight=0.20,
            long_only=True,
            leverage=1.0,
            alpha_scale=params["alpha_scale"],
            cov_shrinkage=params["cov_shrinkage"],
            min_weight_top=params["min_weight_top"],
        )

        w = run_optimizer_once(
            pred_df,
            rets_all,
            trade_tickers,
            spy_ok,
            mv_cfg=mv_cfg,
            opt_top_k=opt_top_k,
            top_n=top_n,
            target_vol_annual=target_vol_annual,
            vol_lookback=vol_lookback,
            max_leverage=max_leverage,
        )

        rets_test = rets_all.loc[w.index, trade_tickers]
        targets = w[trade_tickers].shift(1)
        result = run_backtest(rets_test, targets, cost_bps=cost_bps)

        rows.append(
            {
                **params,
                "total_return": result["stats"]["total_return"],
                "sharpe": result["stats"]["sharpe"],
                "max_drawdown": result["stats"]["max_drawdown"],
                "return_diff_vs_base": result["stats"]["total_return"] - result_base["stats"]["total_return"],
                "sharpe_diff_vs_base": result["stats"]["sharpe"] - result_base["stats"]["sharpe"],
            }
        )

    out = pd.DataFrame(rows).sort_values("return_diff_vs_base", ascending=False)

    run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("reports") / "lstm_optimize_grid" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "grid_summary.csv").write_text(out.to_csv(index=False))
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "top_n": top_n,
                "buffer_k": buffer_k,
                "opt_top_k": opt_top_k,
                "target_vol_annual": target_vol_annual,
                "vol_lookback": vol_lookback,
                "max_leverage": max_leverage,
                "cost_bps": cost_bps,
                "grid": grid,
            },
            indent=2,
        )
    )

    print(f"Saved grid summary to: {run_dir / 'grid_summary.csv'}")
