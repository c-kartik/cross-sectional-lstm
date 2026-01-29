from __future__ import annotations

import numpy as np
import pandas as pd


def forward_fill_weights(rebalance_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Takes sparse target weights (mostly zeros except rebalance dates)
    and forward-fills holdings until the next rebalance.
    """
    # Replace all-zero rows with NaN so ffill works properly
    w = rebalance_weights.copy()
    zero_rows = (w.abs().sum(axis=1) == 0)
    w.loc[zero_rows] = np.nan
    w = w.ffill().fillna(0.0)
    return w


def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    """
    Standard turnover approximation: 0.5 * sum(|delta_w|)
    """
    delta = (new_w - prev_w).abs().sum()
    return float(0.5 * delta)


def run_backtest(
    returns_wide: pd.DataFrame,
    rebalance_weights: pd.DataFrame,
    cost_bps: float = 10.0,
) -> dict:
    """
    returns_wide: DataFrame indexed by date, columns=tickers (daily returns)
    rebalance_weights: DataFrame indexed by date, columns=tickers (targets on rebalance dates only)
    cost_bps: transaction cost in basis points per $ traded (e.g., 10 bps = 0.10%)

    Returns dict with portfolio returns, equity curve, and stats.
    """

    # Convert sparse targets to daily holdings
    daily_w = forward_fill_weights(rebalance_weights)

    # No lookahead: use weights decided at t-1 for returns at t
    daily_w = daily_w.shift(1).fillna(0.0)

    # Align with returns
    daily_w, returns_wide = daily_w.align(returns_wide, join="inner", axis=0)
    daily_w, returns_wide = daily_w.align(returns_wide, join="inner", axis=1)

    # Portfolio gross/net diagnostics
    gross = daily_w.abs().sum(axis=1)
    net = daily_w.sum(axis=1)

    # Raw portfolio return
    port_ret = (daily_w * returns_wide).sum(axis=1)

    # Apply transaction costs on days where weights change (i.e., rebalance impact shows up at next day due to shift)
    # We compute turnover between yesterday's weights and today's weights.
    # Costs are deducted from today's return.
    costs = []
    turnovers = []
    prev = daily_w.iloc[0]
    for i in range(len(daily_w)):
        cur = daily_w.iloc[i]
        turnovers.append(turnover(prev, cur))
        to = turnover(prev, cur)
        cost = to * (cost_bps / 10000.0)
        costs.append(cost)
        prev = cur
    costs = pd.Series(costs, index=daily_w.index, name="cost")
    turnovers = pd.Series(turnovers, index=daily_w.index, name="turnover")

    port_ret_net = port_ret - costs

    equity = (1.0 + port_ret_net).cumprod()

    # Stats
    def sharpe(x: pd.Series) -> float:
        if x.std() == 0:
            return 0.0
        return float(np.sqrt(252) * x.mean() / x.std())

    def max_drawdown(eq: pd.Series) -> float:
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        return float(dd.min())

    out = {
        "returns": port_ret_net,
        "equity": equity,
        "costs": costs,
        "gross_exposure": gross,
        "net_exposure": net,
        "stats": {
            "days": int(len(port_ret_net)),
            "sharpe": sharpe(port_ret_net),
            "max_drawdown": max_drawdown(equity),
            "total_return": float(equity.iloc[-1] - 1.0),
            "avg_gross": float(gross.mean()),
            "avg_net": float(net.mean()),
        },
    }
    out["turnover"] = turnovers
    out["stats"]["avg_daily_turnover"] = float(turnovers.mean())
    out["stats"]["avg_weekly_turnover"] = float(turnovers.resample("W-FRI").sum().mean())
    out["stats"]["total_cost_paid"] = float(costs.sum())

    return out