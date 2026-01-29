from __future__ import annotations
import numpy as np
import pandas as pd

def forward_return(px: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    px: date x ticker prices
    return: Series indexed by (date, ticker) of forward returns over `horizon` days
    label at date t = (P[t+h]/P[t]) - 1
    """
    fwd = px.shift(-horizon) / px - 1.0
    y = fwd.stack().rename(f"fwd_ret_{horizon}d")
    y.index.names = ["date", "ticker"]
    return y.replace([np.inf, -np.inf], np.nan)