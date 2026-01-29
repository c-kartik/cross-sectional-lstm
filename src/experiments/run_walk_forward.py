from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random

from src.data.load_prices import load_prices, compute_returns
from src.universe.custom import get_universe
from src.features.build_features import build_features

from src.datasets.make_dataset import DatasetConfig, make_sequences
from src.models.lstm import LSTMRegressor
from src.signals.trend import above_sma
from src.backtest.engine import run_backtest
import torch

TRADING_DAYS = 252

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_fwd_return_label(px: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    px: DataFrame indexed by date, columns=tickers, price levels
    returns: DataFrame indexed by (date, ticker) with column fwd_ret_{horizon}d
    """
    fwd = px.pct_change(periods=horizon).shift(-horizon)  # forward return over next horizon days
    out = fwd.stack().to_frame(name=f"fwd_ret_{horizon}d")
    out.index.set_names(["date", "ticker"], inplace=True)
    return out

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
    active_ret = active_ret.dropna()
    if len(active_ret) < 2:
        return np.nan
    mu = float(active_ret.mean() * TRADING_DAYS)
    sig = float(active_ret.std(ddof=0) * np.sqrt(TRADING_DAYS))
    return np.nan if sig == 0 else mu / sig


def weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.Series(index=index, data=1).resample("W-FRI").last().index


def train_lstm(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, ckpt_path: str) -> None:
    device = torch.device("cpu")
    model = LSTMRegressor(n_features=Xtr.shape[-1], hidden_size=64, num_layers=2, dropout=0.2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    best = float("inf")
    batch = 256
    epochs = 10

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(Xtr_t.shape[0], device=device)

        for i in range(0, Xtr_t.shape[0], batch):
            idx = perm[i : i + batch]
            xb = Xtr_t[idx]
            yb = ytr_t[idx]

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t)
            val_loss = float(loss_fn(val_pred, yva_t).item())

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), ckpt_path)

    # done


def predict_lstm(Xte: np.ndarray, ckpt_path: str, n_features: int) -> np.ndarray:
    device = torch.device("cpu")
    model = LSTMRegressor(n_features=n_features, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(Xte).to(device)).cpu().numpy()
    return preds.astype(float)


if __name__ == "__main__":
    # ------------------------
    # Build full data table once
    # ------------------------
    prices = load_prices()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px_all = prices[price_col].unstack("ticker").sort_index()
    rets_all = compute_returns(prices, price_col=price_col).unstack("ticker").sort_index()

    universe = [t for t in get_universe() if t in px_all.columns]
    trade_tickers = [t for t in universe if t not in ("SPY", "QQQ")]

    # Build features (MultiIndex: date, ticker)
    Xtab = build_features(px_all[universe].dropna(how="all"))
    # Build REAL labels from price levels
    label_col = "fwd_ret_5d"
    labels = make_fwd_return_label(px_all[universe], horizon=5)
    # Merge label into feature table
    Xtab = Xtab.join(labels, how="left")
    # Drop rows without label (last 5 days for each ticker will be NaN)
    Xtab = Xtab.dropna(subset=[label_col])
    cfg = DatasetConfig(seq_len=60, label_col=label_col)
    
    # ------------------------
    # Run output directory
    # ------------------------
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("reports") / f"walk_forward_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure checkpoint directory exists
    ckpt_dir = Path("data") / "datasets"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path("data/datasets").mkdir(parents=True, exist_ok=True)
    
    print("Label stats:", Xtab[label_col].mean(), Xtab[label_col].std())
    print("Label sample:", Xtab[[label_col]].head())

    dates = Xtab.index.get_level_values("date").unique().sort_values()
    
    # Making the run reproducible
    config = {
        "test_years": [2025, 2024, 2023, 2022],
        "seq_len": 60,
        "label_horizon_days": 5,
        "top_n": 15,
        "buffer_k": 10,
        "target_vol_annual": 0.25,
        "vol_lookback": 60,
        "max_leverage": 2.0,
        "cost_bps": 10.0,
        "sma_window_regime": 100,
        "device": "cpu",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    # ------------------------
    # ------------------------

    # ------------------------
    # Walk-forward splits
    # Example: test years 2022, 2023, 2024, 2025 (adjust based on your data)
    # ------------------------
    test_years = [2022, 2023, 2024, 2025]
    seeds = [2, 4, 6]
    
    rows = []

    for seed in seeds:
        print(f"\n====================")
        print(f"SEED = {seed}")
        print(f"====================")

        for yr in test_years:
            # deterministic per seed+year
            run_seed = seed * 10_000 + yr
            set_seed(run_seed)
            
            test_start = pd.Timestamp(f"{yr}-01-01")
            test_end = pd.Timestamp(f"{yr}-12-31")

            # Train/val end right before test starts
            train_end = test_start - pd.Timedelta(days=1)
            val_start = train_end - pd.Timedelta(days=365)  # last year as validation
            train_start = dates.min()

            # Build sequences for train/val/test
            Xtr, ytr, _ = make_sequences(Xtab, cfg, dates_keep=(train_start, val_start - pd.Timedelta(days=1)))
            Xva, yva, _ = make_sequences(Xtab, cfg, dates_keep=(val_start, train_end))
            Xte, yte, keys_te = make_sequences(Xtab, cfg, dates_keep=(test_start, test_end))

            if len(Xte) == 0 or len(Xtr) == 0 or len(Xva) == 0:
                print("Skipping", yr, "due to insufficient data.")
                continue

            ckpt = ckpt_dir / f"lstm_best_seed{seed}_{yr}.pt"
            train_lstm(Xtr, ytr, Xva, yva, ckpt_path=str(ckpt))

            preds = predict_lstm(Xte, ckpt_path=str(ckpt), n_features=Xte.shape[-1])
            pred_df = pd.DataFrame(
                {"pred": preds},
                index=pd.MultiIndex.from_tuples(keys_te, names=["date", "ticker"]),
            ).sort_index()

            # ------------------------
            # Portfolio construction (Top-N + buffer + sticky rebalances)
            # ------------------------
            top_n = 15
            buffer_k = 10
            target_vol_annual = 0.25
            vol_lookback = 60
            max_leverage = 2.0
            cost_bps = 10.0
            sma_window_regime = 100

            idx = pred_df.index.get_level_values("date").unique().sort_values()
            if len(idx) == 0:
                continue

            start_dt, end_dt = idx.min(), idx.max()

            # SPY regime
            if "SPY" in px_all.columns:
                spy_ok = above_sma(px_all[["SPY"]], window=sma_window_regime)["SPY"]
            else:
                spy_ok = pd.Series(True, index=px_all.index)

            rebals = weekly_rebalance_dates(px_all.loc[start_dt:end_dt].index)
            w = pd.DataFrame(np.nan, index=px_all.loc[start_dt:end_dt].index, columns=trade_tickers)

            current_holdings: set[str] = set()

            test_dates = pred_df.index.get_level_values("date").unique().sort_values()

            for dt in rebals:
                if dt not in w.index:
                    prev = w.index[w.index <= dt]
                    if len(prev) == 0:
                        continue
                    dt = prev[-1]

                if dt not in test_dates:
                    prev_pred = test_dates[test_dates <= dt]
                    if len(prev_pred) == 0:
                        continue
                    dt_pred = prev_pred[-1]
                else:
                    dt_pred = dt

                if not bool(spy_ok.loc[dt]):
                    w.loc[dt] = 0.0
                    current_holdings = set()
                    continue

                preds_today = pred_df.xs(dt_pred, level="date")["pred"]
                preds_today = preds_today.reindex(trade_tickers).dropna()
                if len(preds_today) < top_n:
                    w.loc[dt] = 0.0
                    current_holdings = set()
                    continue

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

                holdings_changed = (new_holdings != current_holdings)
                current_holdings = new_holdings
                if not holdings_changed:
                    continue

                # rank-weight base
                hold_list = list(current_holdings)
                scores = preds_today.reindex(hold_list)
                r = scores.rank(ascending=False, method="first")
                raw = (len(hold_list) + 1 - r).astype(float).clip(lower=0.0)

                base = pd.Series(0.0, index=trade_tickers)
                base.loc[hold_list] = (raw / raw.sum()).values

                hist = rets_all.loc[:dt, trade_tickers].dropna(how="all")
                if len(hist) >= vol_lookback + 1:
                    hist_window = hist.tail(vol_lookback)
                    port_hist = (hist_window.fillna(0.0) * base).sum(axis=1)
                    vol_daily = float(port_hist.std(ddof=0))
                    target_daily = target_vol_annual / np.sqrt(252)
                    lev = min(max_leverage, target_daily / vol_daily) if vol_daily > 0 else 1.0
                else:
                    lev = 1.0

                w.loc[dt] = base * lev

            rets_test = rets_all.loc[w.index, trade_tickers]

            # Execute targets on the next trading day (anti-lookahead safety)
            targets = w[trade_tickers].shift(1)
            result = run_backtest(rets_test, targets, cost_bps=cost_bps)

            eq = result["equity"]
            eq.to_csv(run_dir / f"equity_seed{seed}_{yr}.csv", header=True) # 
            # Plot and save (per-year)
            plt.figure()
            eq.plot(title=f"Walk-forward Equity (Strategy) - Seed={seed} | Year={yr}")
            plt.xlabel("Date")
            plt.ylabel("Equity (start=1.0)")
            plt.tight_layout()
            plt.savefig(run_dir / f"equity_seed{seed}_{yr}.png", dpi=150)   # 
            plt.close()
            strat_ret = eq.pct_change().fillna(0.0)

            qqq_eq = (1.0 + rets_all["QQQ"].loc[w.index].fillna(0.0)).cumprod()
            qqq_ret = qqq_eq.pct_change().fillna(0.0)

            cagr = annualized_return_from_equity(eq)
            vol_ann = annualized_vol_from_returns(strat_ret)
            ir = information_ratio((strat_ret - qqq_ret).dropna())
            
            # Benchmark stats on the same dates
            spy_eq = (1.0 + rets_all["SPY"].loc[w.index].fillna(0.0)).cumprod()
            spy_ret = spy_eq.pct_change().fillna(0.0)

            def sharpe(r: pd.Series) -> float:
                r = r.dropna()
                if len(r) < 2:
                    return np.nan
                mu = float(r.mean() * TRADING_DAYS)
                sig = float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))
                return np.nan if sig == 0 else mu / sig

            qqq_total = float(qqq_eq.iloc[-1] - 1.0)
            spy_total = float(spy_eq.iloc[-1] - 1.0)
            qqq_sh = sharpe(qqq_ret)
            spy_sh = sharpe(spy_ret)
            
            s = result["stats"]
            rows.append({
                "seed": seed,
                "year": yr,
                "days": s["days"],
                "total_return": s["total_return"],
                "sharpe": s["sharpe"],
                "max_dd": s["max_drawdown"],
                "cagr": cagr,
                "ann_vol": vol_ann,
                "ir_vs_qqq": ir,
                "avg_weekly_turnover": s.get("avg_weekly_turnover", np.nan),
                "total_cost_paid": s.get("total_cost_paid", np.nan),
                "qqq_total_return": qqq_total,
                "qqq_sharpe": qqq_sh,
                "spy_total_return": spy_total,
                "spy_sharpe": spy_sh,
            })

            print(f"[{yr}] StratSharpe={s['sharpe']:.3f} StratTot={s['total_return']:.3f} | "f"QQQSharpe={qqq_sh:.3f} QQQTot={qqq_total:.3f} | IRvsQQQ={ir:.3f}")

    out = pd.DataFrame(rows).sort_values(["year", "seed"])
    print("\nWalk-forward (seed/year) rows:")
    print(out.to_string(index=False))
    
    # win rates
    out["beat_qqq"] = out["total_return"] > out["qqq_total_return"]
    out["beat_spy"] = out["total_return"] > out["spy_total_return"]

    by_year = out.groupby("year").agg(
        n=("seed", "count"),
        sharpe_mean=("sharpe", "mean"),
        sharpe_std=("sharpe", "std"),
        ir_vs_qqq_mean=("ir_vs_qqq", "mean"),
        ir_vs_qqq_std=("ir_vs_qqq", "std"),
        win_vs_qqq=("beat_qqq", "mean"),
        win_vs_spy=("beat_spy", "mean"),
    ).reset_index()

    overall = pd.DataFrame([{
        "rows": len(out),
        "sharpe_mean": out["sharpe"].mean(),
        "sharpe_std": out["sharpe"].std(),
        "ir_vs_qqq_mean": out["ir_vs_qqq"].mean(),
        "ir_vs_qqq_std": out["ir_vs_qqq"].std(),
        "win_vs_qqq": out["beat_qqq"].mean(),
        "win_vs_spy": out["beat_spy"].mean(),
    }])

    print("\nSummary by year (3 seeds):")
    print(by_year.to_string(index=False))
    print("\nOverall (3 seeds):")
    print(overall.to_string(index=False))
    
    # Save summary
    out.to_csv(run_dir / "walk_forward_summary.csv", index=False)
    (run_dir / "walk_forward_summary.txt").write_text(out.to_string(index=False))
    print(f"\nSaved walk-forward artifacts to: {run_dir}")
    
    out.to_csv(run_dir / "seed_sweep_rows.csv", index=False)
    by_year.to_csv(run_dir / "seed_sweep_by_year.csv", index=False)
    overall.to_csv(run_dir / "seed_sweep_overall.csv", index=False)