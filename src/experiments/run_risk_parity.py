from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_prices import load_prices, compute_returns
from src.signals.trend import above_sma
from src.backtest.engine import run_backtest


def monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Last trading day of each month (month-end)
    return pd.Series(index=index, data=1).resample("ME").last().index


def inverse_vol_weights(
    rets: pd.DataFrame,  # date x ticker
    lookback: int = 60,
) -> pd.Series:
    """
    Compute inverse-vol weights using trailing lookback window.
    Returns a Series indexed by ticker, summing to 1.
    """
    # Use trailing window vol
    vol = rets.tail(lookback).std(ddof=0)
    vol = vol.replace([0.0, np.inf, -np.inf], np.nan).dropna()
    if len(vol) == 0:
        return pd.Series(dtype=float)

    inv = 1.0 / vol
    w = inv / inv.sum()
    return w

def min_var_weights(
    rets: pd.DataFrame,          # date x ticker
    lookback: int = 60,
    ridge: float = 1e-6,
) -> pd.Series:
    """
    Covariance-aware minimum-variance weights:
      w ∝ Σ^{-1} 1
    Then project to long-only by clipping negatives to 0 and renormalizing.
    """
    X = rets.tail(lookback).dropna(how="any")
    if len(X) < 5:
        return pd.Series(dtype=float)

    cov = X.cov()

    # Ridge for numerical stability (especially with correlated assets)
    cov = cov + np.eye(cov.shape[0]) * ridge

    ones = np.ones(cov.shape[0])

    try:
        w = np.linalg.solve(cov.values, ones)  # Σ^{-1} 1
    except np.linalg.LinAlgError:
        return pd.Series(dtype=float)

    w = pd.Series(w, index=cov.index)

    # Long-only projection: clip negatives and renormalize
    w = w.clip(lower=0.0)
    s = float(w.sum())
    if s <= 0:
        return pd.Series(dtype=float)

    return w / s

def build_risk_parity_weights(
    prices: pd.DataFrame,            # date x ticker (not strictly needed except for index/regime)
    rets: pd.DataFrame,              # date x ticker
    tickers: list[str],
    sma_window: int = 100,           # for SPY regime
    vol_lookback: int = 60,          # inverse-vol lookback
    target_vol_annual: float = 0.15, # portfolio vol target in risk-on
    max_leverage: float = 1.5,
    use_spy_regime: bool = True,
    defensive_target_vol_annual: float = 0.08,
    defensive_min_gross: float = 0.20,
    defensive_max_gross: float = 0.30,
) -> pd.DataFrame:
    """
    Monthly rebalanced inverse-vol (risk parity-ish) long-only portfolio,
    with SPY regime gating and defensive floor/cap in risk-off.
    Returns sparse rebalance-day weights (NaN on non-rebalance days).
    """
    print("MinVar lookback:", vol_lookback, " | SPY SMA window:", sma_window)

    px = prices[tickers].copy()

    # SPY regime
    if use_spy_regime and "SPY" in prices.columns:
        spy_ok = above_sma(prices[["SPY"]], window=sma_window)["SPY"]
    else:
        spy_ok = pd.Series(True, index=px.index)

    rebals = monthly_rebalance_dates(px.index)

    # Sparse target weights
    w = pd.DataFrame(np.nan, index=px.index, columns=tickers)

    for dt in rebals:
        # ensure dt is a trading day in index
        if dt not in px.index:
            prev = px.index[px.index <= dt]
            if len(prev) == 0:
                continue
            dt = prev[-1]

        # slice history up to dt (no lookahead)
        hist = rets.loc[:dt, tickers].dropna(how="all")

        # If insufficient data, go flat for now
        if len(hist) < vol_lookback + 1:
            w.loc[dt] = 0.0
            continue

        base = inverse_vol_weights(hist, lookback=vol_lookback)

        # If something went wrong computing base, go flat
        if base.empty:
            w.loc[dt] = 0.0
            continue

        # Ensure base has all tickers (fill missing with 0)
        base_full = pd.Series(0.0, index=tickers)
        base_full.loc[base.index] = base.values
        base = base_full

        # --- Defensive mode (SPY regime off): floor + cap + vol-aware ---
        if not bool(spy_ok.loc[dt]):
            lev_def = defensive_min_gross

            hist_window = hist.tail(vol_lookback)
            port_hist = (hist_window.fillna(0.0) * base).sum(axis=1)
            vol_daily = float(port_hist.std(ddof=0))
            target_daily = defensive_target_vol_annual / np.sqrt(252)

            if vol_daily > 0:
                lev_def = target_daily / vol_daily

            lev_def = max(defensive_min_gross, lev_def)
            lev_def = min(defensive_max_gross, lev_def)

            w.loc[dt] = base * lev_def
            continue

        # --- Risk-on mode: portfolio vol targeting ---
        hist_window = hist.tail(vol_lookback)
        port_hist = (hist_window.fillna(0.0) * base).sum(axis=1)
        vol_daily = float(port_hist.std(ddof=0))
        target_daily = target_vol_annual / np.sqrt(252)

        if vol_daily > 0:
            lev = min(max_leverage, target_daily / vol_daily)
        else:
            lev = 1.0

        w.loc[dt] = base * lev

    return w


if __name__ == "__main__":
    prices = load_prices()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px_all = prices[price_col].unstack("ticker").sort_index()

    rets = compute_returns(prices, price_col=price_col).unstack("ticker").sort_index()

    # Benchmarks
    spy_eq = (1.0 + rets["SPY"].dropna()).cumprod()
    qqq_eq = (1.0 + rets["QQQ"].dropna()).cumprod()

    # Trade universe (same as your trend file)
    trade_tickers = [c for c in px_all.columns if c not in ("SPY", "QQQ")]

    targets = build_risk_parity_weights(
        prices=px_all,
        rets=rets,
        tickers=trade_tickers,
        sma_window=100,
        vol_lookback=60,
        target_vol_annual=0.15,
        max_leverage=1.5,
        use_spy_regime=True,
        defensive_target_vol_annual=0.08,
        defensive_min_gross=0.20,
        defensive_max_gross=0.30,
    )

    result = run_backtest(rets[trade_tickers], targets[trade_tickers], cost_bps=10.0)

    # exposure diagnostics (shift 1 day to match execution)
    daily_w = targets.ffill().fillna(0.0).shift(1).fillna(0.0)
    gross = daily_w.abs().sum(axis=1)
    print("Avg gross exposure (daily):", float(gross.mean()))
    print("Pct days fully flat:", float((gross == 0).mean()))
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

    # Plot vs benchmarks
    eq = result["equity"]
    spy_eq_aligned = spy_eq.reindex(eq.index).ffill()
    qqq_eq_aligned = qqq_eq.reindex(eq.index).ffill()

    ax = eq.plot(label="Strategy (MinVar)", title="Min-Variance + Regime + Vol Targeting")
    spy_eq_aligned.plot(ax=ax, label="SPY Buy & Hold")
    qqq_eq_aligned.plot(ax=ax, label="QQQ Buy & Hold")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.show()