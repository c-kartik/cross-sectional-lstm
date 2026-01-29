from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_prices import load_prices, compute_returns


def equity_curve_from_returns(daily_port_ret: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1.0 + daily_port_ret).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def sharpe(daily_ret: pd.Series, rf_daily: float = 0.0) -> float:
    x = daily_ret - rf_daily
    if x.std() == 0:
        return 0.0
    return float(np.sqrt(252) * x.mean() / x.std())


if __name__ == "__main__":
    prices = load_prices()  # loads all cached tickers
    rets = compute_returns(prices)  # Series indexed by (date, ticker)

    # Equal-weight portfolio: average returns across tickers each day
    ret_table = rets.unstack("ticker").dropna(how="all")
    port_ret = ret_table.mean(axis=1).dropna()

    eq = equity_curve_from_returns(port_ret)

    print("Days:", len(port_ret))
    print("Sharpe:", round(sharpe(port_ret), 3))
    print("Max Drawdown:", round(max_drawdown(eq), 3))
    print("Total Return:", round(eq.iloc[-1] - 1.0, 3))

    eq.plot(title="Equal-Weight Portfolio (Baseline)")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.show()