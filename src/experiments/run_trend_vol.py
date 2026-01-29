from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_prices import load_prices, compute_returns
from src.signals.trend import above_sma
from src.backtest.engine import run_backtest


def monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Last trading day of each month
    return pd.Series(index=index, data=1).resample("ME").last().index


def build_trend_weights(
    prices: pd.DataFrame,            # date x ticker
    rets: pd.DataFrame,              # date x ticker
    tickers: list[str],
    sma_window: int = 100,
    target_vol_annual: float = 0.10,
    vol_lookback: int = 60,
    max_leverage: float = 1.5,
    use_spy_regime: bool = True,
    defensive_target_vol_annual: float = 0.08,
    defensive_min_gross: float = 0.20,
    defensive_max_gross: float = 0.30,
) -> pd.DataFrame:
    """
    Returns sparse rebalance-day weights (NaN on non-rebalance days).
    Engine will ffill and apply costs.
    """
    print("SMA window being used:", sma_window)
    
    px = prices[tickers].copy()
    #trend_ok = above_sma(px, window=sma_window)  # True/False per ticker per day

    # Optional regime filter: only invest if SPY is in uptrend
    if use_spy_regime and "SPY" in prices.columns:
        spy_ok = above_sma(prices[["SPY"]], window=sma_window)["SPY"]
    else:
        spy_ok = pd.Series(True, index=px.index)

    rebals = monthly_rebalance_dates(px.index)

    # weights: NaN means "no new target that day"
    w = pd.DataFrame(np.nan, index=px.index, columns=tickers)

    for dt in rebals:
        if dt not in px.index:
            # use last available trading day before month-end timestamp
            prev = px.index[px.index <= dt]
            if len(prev) == 0:
                continue
            dt = prev[-1]

        if not bool(spy_ok.loc[dt]):
            # Defensive mode: keep at least a baseline exposure,
            # but scale down in high vol and cap overall exposure.
            base = pd.Series(1.0 / len(tickers), index=tickers)

            hist = rets.loc[:dt, tickers].dropna(how="all")
            lev_def = defensive_min_gross  # default floor

            if len(hist) >= vol_lookback + 1:
                hist_window = hist.tail(vol_lookback)
                port_hist = (hist_window.fillna(0.0) * base).sum(axis=1)
                vol_daily = float(port_hist.std(ddof=0))
                target_daily = defensive_target_vol_annual / np.sqrt(252)

                if vol_daily > 0:
                    lev_def = target_daily / vol_daily  # vol-based suggestion

            # Apply floor and cap
            lev_def = max(defensive_min_gross, lev_def)
            lev_def = min(defensive_max_gross, lev_def)

            w.loc[dt] = base * lev_def
            continue

        # eligible = trend_ok.loc[dt]
        # names = eligible[eligible].index.tolist()
        # Risk-on: invest in the whole basket (SPY regime only)
        names = tickers


        if len(names) == 0:
            w.loc[dt] = 0.0
            continue

        # equal weight across names that are above SMA
        base = pd.Series(0.0, index=tickers)
        base.loc[names] = 1.0 / len(names)

        # Vol targeting: scale overall exposure using realized portfolio vol
        # Use trailing returns up to dt (no lookahead)
        hist = rets.loc[:dt, tickers].dropna(how="all")
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

        w.loc[dt] = (base * lev)

    return w


if __name__ == "__main__":
    prices = load_prices()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px_all = prices[price_col].unstack("ticker").sort_index()

    rets = compute_returns(prices, price_col=price_col).unstack("ticker").sort_index()

    # Benchmark: SPY buy & hold equity curve
    spy_eq = (1.0 + rets["SPY"].dropna()).cumprod()
    qqq_eq = (1.0 + rets["QQQ"].dropna()).cumprod()

    # Trade universe: your stocks; keep SPY for regime/benchmark but don’t have to trade it
    trade_tickers = [c for c in px_all.columns if c not in ("SPY", "QQQ")]
    # If you want to include QQQ as a tradable asset, remove it from the exclude list.

    targets = build_trend_weights(
        prices=px_all,
        rets=rets,
        tickers=trade_tickers,
        # sma_window=200,
        target_vol_annual=0.15,
        vol_lookback=60,
        max_leverage=1.5,
        use_spy_regime=True,
    )

    result = run_backtest(rets[trade_tickers], targets[trade_tickers], cost_bps=10.0)
    
    daily_w = targets.ffill().fillna(0.0).shift(1).fillna(0.0)
    print("Avg invested days exposure:", daily_w.abs().sum(axis=1).mean())
    print("Pct days fully flat:", (daily_w.abs().sum(axis=1) == 0).mean())
    gross = daily_w.abs().sum(axis=1)
    print("Avg gross exposure (daily):", gross.mean())
    print("Gross exposure 10/50/90 pct:", gross.quantile([0.1, 0.5, 0.9]).to_dict())
    
    s = result["stats"]
    print("Days:", s["days"])
    print("Sharpe:", round(s["sharpe"], 3))
    print("Max Drawdown:", round(s["max_drawdown"], 3))
    print("Total Return:", round(s["total_return"], 3))
    print("Avg Gross Exposure:", round(s["avg_gross"], 3))
    print("Avg Net Exposure:", round(s["avg_net"], 3))
    if "avg_weekly_turnover" in s:
        print("Avg Weekly Turnover:", round(s["avg_weekly_turnover"], 4))
        print("Total Cost Paid:", round(s["total_cost_paid"], 4))

    # this prints just the ren_trend_vol.py equity curve
    # result["equity"].plot(title="Trend-Following (Long-only) + Vol Targeting")
    
    eq = result["equity"]
    spy_eq_aligned = spy_eq.reindex(eq.index).ffill()
    qqq_eq_aligned = qqq_eq.reindex(eq.index).ffill()
    ax = eq.plot(label="Strategy", title="Trend (SPY Regime) + Vol Targeting vs SPY")
    spy_eq_aligned.plot(ax=ax, label="SPY Buy & Hold")
    qqq_eq_aligned.plot(ax=ax, label="QQQ Buy & Hold")
    plt.legend()
    
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.show()