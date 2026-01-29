from __future__ import annotations
import numpy as np
import pandas as pd

from src.data.load_prices import load_prices
from src.universe.custom import get_universe
from src.features.build_features import build_features
from src.labels.forward_returns import forward_return
from src.datasets.make_dataset import DatasetConfig, make_sequences
from src.datasets.splits import time_split_dates


if __name__ == "__main__":
    prices = load_prices()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px_all = prices[price_col].unstack("ticker").sort_index()

    universe = [t for t in get_universe() if t in px_all.columns]
    px = px_all[universe].dropna(how="all")

    X = build_features(px)
    y = forward_return(px, horizon=5)
    df = X.join(y, how="inner").dropna()

    cfg = DatasetConfig(seq_len=60, horizon=5, label_col="fwd_ret_5d")

    train_end, val_end = time_split_dates(px.index)
    # train: start -> train_end
    Xtr, ytr, _ = make_sequences(df, cfg, dates_keep=(pd.Timestamp("1900-01-01"), train_end))
    # val: (train_end -> val_end]
    Xva, yva, _ = make_sequences(df, cfg, dates_keep=(train_end + pd.Timedelta(days=1), val_end))
    # test: (val_end -> end]
    Xte, yte, _ = make_sequences(df, cfg, dates_keep=(val_end + pd.Timedelta(days=1), pd.Timestamp("2100-01-01")))

    print("Train X:", Xtr.shape, " y:", ytr.shape)
    print("Val   X:", Xva.shape, " y:", yva.shape)
    print("Test  X:", Xte.shape, " y:", yte.shape)
    print("Label stats (train): mean", float(np.mean(ytr)), "std", float(np.std(ytr)))