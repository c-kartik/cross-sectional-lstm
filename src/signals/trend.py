from __future__ import annotations
import pandas as pd

def sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window, min_periods=window).mean()

def above_sma(prices: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    prices: date x ticker (levels)
    returns: boolean DataFrame: True if price > SMA(window)
    """
    return prices > prices.rolling(window, min_periods=window).mean()