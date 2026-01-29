from __future__ import annotations
import pandas as pd

def time_split_dates(
    dates: pd.DatetimeIndex,
    train_end: str = "2022-12-31",
    val_end: str = "2024-12-31",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Returns (train_end_ts, val_end_ts). Test is anything after val_end.
    """
    return pd.Timestamp(train_end), pd.Timestamp(val_end)