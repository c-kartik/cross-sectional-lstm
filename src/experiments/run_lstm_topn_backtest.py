from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data.load_prices import load_prices, compute_returns
from src.universe.custom import get_universe
from src.features.build_features import build_features
from src.datasets.make_dataset import DatasetConfig, make_sequences
from src.datasets.splits import time_split_dates
from src.models.lstm import LSTMRegressor
from src.signals.trend import above_sma
from src.backtest.engine import run_backtest

TRADING_DAYS = 252

def annualized_return_from_equity(eq: pd.Series) -> float:
    eq = eq.dropna()
    if len(eq) < 2:
        return np.nan
    years = len(eq) / TRADING_DAYS
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0)

def annualized_vol_from_returns(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))

def information_ratio(active_ret: pd.Series) -> float:
    # IR = mean(active) / std(active) annualized
    active_ret = active_ret.dropna()
    if len(active_ret) < 2:
        return np.nan
    mu = float(active_ret.mean() * TRADING_DAYS)
    sig = float(active_ret.std(ddof=0) * np.sqrt(TRADING_DAYS))
    return np.nan if sig == 0 else mu / sig

def weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Every Friday (or last trading day available up to Friday)
    return pd.Series(index=index, data=1).resample("W-FRI").last().index


def load_lstm_scores_test(df_feat_label: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """
    Build sequences on TEST range, load saved model, and return a DataFrame:
      index: (date, ticker)
      columns: pred
    """
    # same split logic as training
    dates = df_feat_label.index.get_level_values("date").unique().sort_values()
    train_end, val_end = time_split_dates(dates)

    Xte, yte, keys = make_sequences(
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
    # ------------------------
    # Data
    # ------------------------
    prices = load_prices()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px_all = prices[price_col].unstack("ticker").sort_index()

    rets_all = compute_returns(prices, price_col=price_col).unstack("ticker").sort_index()

    universe = [t for t in get_universe() if t in px_all.columns]
    px = px_all[universe].dropna(how="all")

    # Build features table (same as before)
    Xtab = build_features(px)
    # label col isn’t needed for inference, but make_sequences expects it; we can create dummy 0 label
    df_feat = Xtab.copy()
    df_feat["fwd_ret_5d"] = 0.0  # dummy

    cfg = DatasetConfig(seq_len=60, label_col="fwd_ret_5d")

    # Get test predictions
    pred_df = load_lstm_scores_test(df_feat, cfg)

    # Trade universe: exclude benchmarks
    trade_tickers = [t for t in universe if t not in ("SPY", "QQQ")]

    # ------------------------
    # Build Top-N weights weekly on the TEST period
    # ------------------------
    DEBUG_HOLDINGS = False
    
    top_n = 15
    buffer_k = 10  # NEW: Top-N buffer rule (sell if rank > N+buffer, buy if rank <= N)
    target_vol_annual = 0.25
    vol_lookback = 60
    max_leverage = 2.0
    cost_bps = 10.0
    sma_window_regime = 100
    current_holdings: set[str] = set()  # NEW: state across rebalances
    prev_holdings: set[str] = set() # NEW: for churn debug printing

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

    # sparse weights
    w = pd.DataFrame(np.nan, index=px_all.loc[start_dt:end_dt].index, columns=trade_tickers)

    for dt in rebals:
        if dt not in w.index:
            prev = w.index[w.index <= dt]
            if len(prev) == 0:
                continue
            dt = prev[-1]

        # if we don't have preds for this dt, skip
        if dt not in test_dates:
            # choose latest prediction date <= dt
            prev_pred = test_dates[test_dates <= dt]
            if len(prev_pred) == 0:
                continue
            dt_pred = prev_pred[-1]
        else:
            dt_pred = dt

        if not bool(spy_ok.loc[dt]):
            # risk-off: go flat
            w.loc[dt] = 0.0
            continue

        # take top-N by predicted score
        preds_today = pred_df.xs(dt_pred, level="date")["pred"]
        preds_today = preds_today.reindex(trade_tickers).dropna()
        if len(preds_today) < top_n:
            w.loc[dt] = 0.0
            continue
        
        # rank 1 = best
        ranks = preds_today.rank(ascending=False, method="first")

        # 1) SELL: drop holdings that fell below N+buffer (rank > N+buffer)
        sell_cutoff = top_n + buffer_k
        to_keep = {t for t in current_holdings if (t in ranks.index and ranks.loc[t] <= sell_cutoff)}

        # 2) BUY: candidates are strictly top-N (rank <= N)
        topN = ranks[ranks <= top_n].sort_values().index.tolist()

        # Add buys until we have N names (or run out)
        new_holdings = set(to_keep)
        for t in topN:
            if len(new_holdings) >= top_n:
                break
            new_holdings.add(t)

        # If still short (can happen early on), fill from best ranks beyond topN
        if len(new_holdings) < top_n:
            for t in ranks.sort_values().index.tolist():
                if len(new_holdings) >= top_n:
                    break
                new_holdings.add(t)

        holdings_changed = (new_holdings != current_holdings)
        current_holdings = new_holdings

        # If holdings didn't change, don't force a rebalance (reduces turnover)
        if not holdings_changed:
            continue  # leave w.loc[dt] as NaN; engine will carry forward
        
        added = current_holdings - prev_holdings
        removed = prev_holdings - current_holdings
        
        if DEBUG_HOLDINGS and (len(added) + len(removed) > 0):
            print(f"{dt.date()} | holdings={len(current_holdings)} | +{len(added)} -{len(removed)}")
        prev_holdings = set(current_holdings)

        # rank-weight across holdings (higher score -> higher weight)
        hold_list = list(current_holdings)
        scores = preds_today.reindex(hold_list)

        # convert scores -> ranks (1 best)
        r = scores.rank(ascending=False, method="first")

        # weights proportional to (N+1-rank): best gets N, worst gets 1
        raw = (len(hold_list) + 1 - r).astype(float)
        raw = raw.clip(lower=0.0)
        # equal weight across holdings
        base = pd.Series(0.0, index=trade_tickers)
        base.loc[hold_list] = (raw / raw.sum()).values
        ##########################################################################


        # vol targeting using trailing realized returns of selected basket
        hist = rets_all.loc[:dt, trade_tickers].dropna(how="all")
        if len(hist) >= vol_lookback + 1:
            hist_window = hist.tail(vol_lookback)
            port_hist = (hist_window.fillna(0.0) * base).sum(axis=1)
            vol_daily = float(port_hist.std(ddof=0))
            target_daily = target_vol_annual / np.sqrt(252)
            if vol_daily > 0:
                lev = min(max_leverage, target_daily / vol_daily)
            else:
                lev = 1.0
        else:
            lev = 1.0

        w.loc[dt] = base * lev

    # ------------------------
    # Backtest on test window
    # ------------------------
    rets_test = rets_all.loc[w.index, trade_tickers]
    result = run_backtest(rets_test, w[trade_tickers], cost_bps=cost_bps)

    # Benchmarks aligned
    spy_eq = (1.0 + rets_all["SPY"].loc[w.index].fillna(0.0)).cumprod()
    qqq_eq = (1.0 + rets_all["QQQ"].loc[w.index].fillna(0.0)).cumprod()

    s = result["stats"]
    print("Top-N:", top_n, " | cost_bps:", cost_bps)
    print("Days:", s["days"])
    print("Sharpe:", round(s["sharpe"], 3))
    print("Max Drawdown:", round(s["max_drawdown"], 3))
    print("Total Return:", round(s["total_return"], 3))
    print("Avg Gross Exposure:", round(s["avg_gross"], 3))
    print("Avg Net Exposure:", round(s["avg_net"], 3))
    if "avg_weekly_turnover" in s:
        print("Avg Weekly Turnover:", round(s["avg_weekly_turnover"], 4))
        print("Total Cost Paid:", round(s["total_cost_paid"], 4))
    print("Annualized Return:", round(annualized_return_from_equity(result["equity"]), 4))       

    # Plot
    eq = result["equity"]

    # --- Strategy daily returns (aligned to eq dates) ---
    strat_ret = eq.pct_change().fillna(0.0)

    # Benchmarks aligned equity curves
    spy_eq_aligned = spy_eq.reindex(eq.index).ffill()
    qqq_eq_aligned = qqq_eq.reindex(eq.index).ffill()

    # --- Benchmark daily returns ---
    spy_ret = spy_eq_aligned.pct_change().fillna(0.0)
    qqq_ret = qqq_eq_aligned.pct_change().fillna(0.0)

    # --- Risk-match QQQ (and SPY optionally) to strategy vol ---
    # Use realized vol over the test window
    eps = 1e-12
    strat_vol = float(strat_ret.std(ddof=0))
    qqq_vol = float(qqq_ret.std(ddof=0))
    spy_vol = float(spy_ret.std(ddof=0))

    qqq_scale = (strat_vol / (qqq_vol + eps))
    spy_scale = (strat_vol / (spy_vol + eps))

    qqq_rm_eq = (1.0 + qqq_ret * qqq_scale).cumprod()
    spy_rm_eq = (1.0 + spy_ret * spy_scale).cumprod()
    
    # --- Research-grade metrics ---
    # Make sure these exist already:
    # eq, strat_ret, qqq_ret aligned to eq.index

    cagr = annualized_return_from_equity(eq)
    vol_ann = annualized_vol_from_returns(strat_ret)

    # Active returns vs QQQ (same dates)
    active_vs_qqq = strat_ret.align(qqq_ret, join="inner")[0] - strat_ret.align(qqq_ret, join="inner")[1] 
    # EARLIER: active_vs_qqq = (strat_ret - qqq_ret).dropna()
    ir_vs_qqq = information_ratio(active_vs_qqq)

    print("CAGR:", round(cagr, 3))
    print("Annualized Vol:", round(vol_ann, 3))
    print("IR vs QQQ:", round(ir_vs_qqq, 3))
    #--- End research-grade metrics ---

    # Print scales so you can explain them in writeups/interviews
    print("Risk-match scales (to strategy vol):")
    print("  QQQ scale:", round(qqq_scale, 3))
    print("  SPY scale:", round(spy_scale, 3))
    print("Strategy daily vol:", round(strat_vol, 5),
        "| QQQ daily vol:", round(qqq_vol, 5),
        "| SPY daily vol:", round(spy_vol, 5))

    # --- Plot everything ---
    ax = eq.plot(label="Strategy (LSTM Top-N)", title="Strategy vs Benchmarks (raw and risk-matched)")
    spy_eq_aligned.plot(ax=ax, label="SPY Buy & Hold")
    qqq_eq_aligned.plot(ax=ax, label="QQQ Buy & Hold")
    qqq_rm_eq.plot(ax=ax, label="QQQ Risk-Matched")
    # Optional:
    # spy_rm_eq.plot(ax=ax, label="SPY Risk-Matched")

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.show()