from __future__ import annotations

import matplotlib.pyplot as plt

from src.data.load_prices import load_prices, compute_returns
from src.signals.mean_reversion import zscore_signal, weekly_market_neutral_weights_from_z
from src.backtest.engine import run_backtest


if __name__ == "__main__":
    prices = load_prices()

    # Use adj_close if present, else close
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px = prices[price_col].unstack("ticker").sort_index()

    # Daily returns table (date x ticker)
    rets = compute_returns(prices, price_col=price_col).unstack("ticker").sort_index()

    # Signal
    #z = zscore_signal(px, lookback=20)
    # Use returns z-score (short-horizon mean reversion)
    ret_1d = rets  # already daily returns (date x ticker)
    mu = ret_1d.rolling(20, min_periods=20).mean()
    sig = ret_1d.rolling(20, min_periods=20).std(ddof=0)
    z = (ret_1d - mu) / sig

    # Rebalance targets (weekly)
    targets = weekly_market_neutral_weights_from_z(z, k=3, long_gross=0.5, short_gross=0.5)

    # Backtest
    result = run_backtest(rets, targets, cost_bps=10.0)

    s = result["stats"]
    print("Days:", s["days"])
    print("Sharpe:", round(s["sharpe"], 3))                             # Risk-adjusted return
    print("Max Drawdown:", round(s["max_drawdown"], 3))                 # Maximum peak-to-trough decline
    print("Total Return:", round(s["total_return"], 3))                 # Overall return over the backtest period
    print("Avg Gross Exposure:", round(s["avg_gross"], 3))              # Should be close to 1.0 (0.5 long + 0.5 short)
    print("Avg Net Exposure:", round(s["avg_net"], 3))                  # Should be close to 0 for market-neutral
    print("Avg Daily Turnover:", round(s["avg_daily_turnover"], 4))     # How much of the portfolio is being traded per day on average
    print("Avg Weekly Turnover:", round(s["avg_weekly_turnover"], 4))   # How much of the portfolio is being traded per week on average
    print("Total Cost Paid:", round(s["total_cost_paid"], 4))           # How much of the initial equity is lost to transaction costs

    # Plot equity
    result["equity"].plot(title="Mean Reversion (Market-Neutral, Weekly Rebalance, Costs)")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.show()