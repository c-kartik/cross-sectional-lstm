from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.load_prices import load_prices
from src.universe.custom import get_universe
from src.features.build_features import build_features
from src.labels.forward_returns import forward_return
from src.datasets.make_dataset import DatasetConfig, make_sequences
from src.datasets.splits import time_split_dates
from src.models.lstm import LSTMRegressor


def spearman_ic_by_date(df: pd.DataFrame, pred_col: str, label_col: str) -> pd.Series:
    """
    df indexed by (date, ticker) with pred_col and label_col.
    returns IC per date (Spearman corr across tickers).
    """
    def _ic(g: pd.DataFrame) -> float:
        if g[pred_col].nunique() < 2 or g[label_col].nunique() < 2:
            return np.nan
        return float(g[pred_col].rank().corr(g[label_col].rank()))
    return df.groupby(level="date", group_keys=False).apply(_ic)


if __name__ == "__main__":
    # Build df like training did
    prices = load_prices()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px_all = prices[price_col].unstack("ticker").sort_index()

    universe = [t for t in get_universe() if t in px_all.columns]
    px = px_all[universe].dropna(how="all")

    Xtab = build_features(px)
    yser = forward_return(px, horizon=5)
    df = Xtab.join(yser, how="inner").dropna()

    cfg = DatasetConfig(seq_len=60, label_col="fwd_ret_5d")
    train_end, val_end = time_split_dates(px.index)

    # Only test set sequences + keys
    Xte, yte, keys = make_sequences(
        df, cfg,
        dates_keep=(val_end + pd.Timedelta(days=1), pd.Timestamp("2100-01-01"))
    )

    print("Test sequences:", Xte.shape, yte.shape)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = LSTMRegressor(n_features=Xte.shape[-1], hidden_size=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load("data/datasets/lstm_best.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(Xte).to(device)).cpu().numpy()

    # Build prediction frame indexed by (date,ticker)
    pred_df = pd.DataFrame(
        {
            "pred": preds.astype(float),
            "y": yte.astype(float),
        },
        index=pd.MultiIndex.from_tuples(keys, names=["date", "ticker"])
    ).sort_index()

    ic = spearman_ic_by_date(pred_df, "pred", "y").dropna()

    print("IC mean:", round(float(ic.mean()), 4))
    print("IC std :", round(float(ic.std(ddof=0)), 4))
    print("% IC > 0:", round(float((ic > 0).mean()), 4))
    print("Num dates:", len(ic))