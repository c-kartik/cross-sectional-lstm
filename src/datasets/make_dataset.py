from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    seq_len: int = 60
    horizon: int = 5
    feature_cols: tuple[str, ...] = (
        "ret_1d", "ret_5d", "ret_20d",
        "vol_20d", "vol_60d",
        "px_sma20", "px_sma100",
        "dd_60",
    )
    label_col: str = "fwd_ret_5d"


def zscore_by_date(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score per date (standard for ranking alphas).
    For each feature, at each date, normalize across tickers.
    """
    def _z(g: pd.DataFrame) -> pd.DataFrame:
        mu = g.mean(axis=0)
        sd = g.std(axis=0, ddof=0).replace(0.0, np.nan)
        return (g - mu) / sd

    return X.groupby(level="date", group_keys=False).apply(_z).replace([np.inf, -np.inf], np.nan)


def make_sequences(
    df: pd.DataFrame,
    cfg: DatasetConfig,
    dates_keep: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[pd.Timestamp, str]]]:
    """
    df: MultiIndex (date, ticker) with feature cols + label col
    returns:
      X: [N, seq_len, F]
      y: [N]
      keys: list of (date, ticker) aligned with X/y
    """
    # Ensure sorted
    df = df.sort_index()

    # Optionally filter date range
    if dates_keep is not None:
        start, end = dates_keep
        mask = (df.index.get_level_values("date") >= start) & (df.index.get_level_values("date") <= end)
        df = df.loc[mask]

    # Split into X and y
    Xtab = df[list(cfg.feature_cols)].copy()
    ytab = df[cfg.label_col].copy()

    # Cross-sectional normalization per date (recommended for ranking)
    Xtab = zscore_by_date(Xtab)
    df2 = Xtab.join(ytab, how="inner").dropna()

    X_list = []
    y_list = []
    keys = []

    for ticker, g in df2.groupby(level="ticker"):
        g = g.droplevel("ticker")
        g = g.sort_index()

        Xg = g[list(cfg.feature_cols)].values
        yg = g[cfg.label_col].values
        idx = g.index

        # rolling sequences
        for i in range(cfg.seq_len, len(g)):
            # sequence ends at time i-1, label at time i
            x_seq = Xg[i - cfg.seq_len:i, :]
            y_val = yg[i]
            dt = idx[i]

            if np.isnan(x_seq).any() or np.isnan(y_val):
                continue

            X_list.append(x_seq.astype(np.float32))
            y_list.append(np.float32(y_val))
            keys.append((pd.Timestamp(dt), str(ticker)))

    X = np.stack(X_list) if X_list else np.empty((0, cfg.seq_len, len(cfg.feature_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, keys