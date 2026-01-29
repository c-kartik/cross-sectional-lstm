from __future__ import annotations

import pandas as pd


def zscore_signal(prices_wide: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    prices_wide: DataFrame indexed by date, columns=tickers (values are prices)
    returns: z-score per ticker per date
    """
    roll_mean = prices_wide.rolling(lookback, min_periods=lookback).mean()
    roll_std = prices_wide.rolling(lookback, min_periods=lookback).std(ddof=0)
    z = (prices_wide - roll_mean) / roll_std
    return z


def weekly_market_neutral_weights_from_z(
    z: pd.DataFrame,
    k: int = 3,
    long_gross: float = 0.8,
    short_gross: float = 0.2,
) -> pd.DataFrame:
    """
    Build weekly (rebalance) weights from z-scores:
      - long k most negative z
      - short k most positive z
    Net exposure ≈ 0, gross exposure = long_gross + short_gross

    Output weights indexed by date (same index as z), columns=tickers.
    These are "rebalance-day targets" (we will forward-fill between rebalances).
    """

    # Rebalance weekly on Friday (or last trading day of week available in index)
    # We'll use W-FRI sampling of the z-score index.
    rebalance_dates = z.resample("W-FRI").last().index

    weights = pd.DataFrame(0.0, index=z.index, columns=z.columns)

    for dt in rebalance_dates:
        if dt not in z.index:
            # If the exact Friday isn't a trading day, find the previous available date
            prev = z.index[z.index <= dt]
            if len(prev) == 0:
                continue
            dt_use = prev[-1]
        else:
            dt_use = dt

        row = z.loc[dt_use].dropna()
        # only consider sufficiently extreme z-scores
        z_threshold = 0.2
        row = row[row.abs() >= z_threshold]
        if len(row) < 2:
            continue

        longs = row.nsmallest(k).index
        shorts = row.nlargest(k).index

        w = pd.Series(0.0, index=z.columns)
        if len(longs) > 0:
            w.loc[longs] = long_gross / len(longs)
        if len(shorts) > 0:
            w.loc[shorts] = -short_gross / len(shorts)

        weights.loc[dt_use] = w

    # Keep only rebalance dates as non-zero targets; forward-fill later in engine.
    return weights