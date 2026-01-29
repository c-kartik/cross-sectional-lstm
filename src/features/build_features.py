from __future__ import annotations
import numpy as np
import pandas as pd

def build_features(px: pd.DataFrame) -> pd.DataFrame:
    """
    px: date x ticker adjusted close (preferred) or close.
    returns: MultiIndex DataFrame indexed by (date, ticker), columns = features
    """
    # daily returns
    r1 = px.pct_change()

    feats = {}

    # Momentum-like features (just facts, not "hype")
    feats["ret_1d"] = r1
    feats["ret_5d"] = px.pct_change(5)
    feats["ret_20d"] = px.pct_change(20)

    # Volatility features
    feats["vol_20d"] = r1.rolling(20).std(ddof=0)
    feats["vol_60d"] = r1.rolling(60).std(ddof=0)

    # Trend / distance to moving averages
    sma20 = px.rolling(20).mean()
    sma100 = px.rolling(100).mean()
    feats["px_sma20"] = (px / sma20) - 1.0
    feats["px_sma100"] = (px / sma100) - 1.0

    # Drawdown from rolling max
    roll_max_60 = px.rolling(60).max()
    feats["dd_60"] = (px / roll_max_60) - 1.0

    # Combine to MultiIndex (date, ticker)
    out = []
    for name, df in feats.items():
        tmp = df.stack().rename(name)
        out.append(tmp)

    X = pd.concat(out, axis=1)
    X.index.names = ["date", "ticker"]
    return X.replace([np.inf, -np.inf], np.nan)