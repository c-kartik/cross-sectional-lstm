from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_prices import load_prices
from src.universe.custom import get_universe
from src.features.build_features import build_features
from src.labels.forward_returns import forward_return
from src.datasets.make_dataset import DatasetConfig, make_sequences
from src.datasets.splits import time_split_dates
from src.models.lstm import LSTMRegressor


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def rmse(yhat: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((yhat - y) ** 2)).item())


if __name__ == "__main__":
    # ---- Build dataset (same as before) ----
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

    Xtr, ytr, _ = make_sequences(df, cfg, dates_keep=(pd.Timestamp("1900-01-01"), train_end))
    Xva, yva, _ = make_sequences(df, cfg, dates_keep=(train_end + pd.Timedelta(days=1), val_end))
    Xte, yte, _ = make_sequences(df, cfg, dates_keep=(val_end + pd.Timedelta(days=1), pd.Timestamp("2100-01-01")))

    print("Train:", Xtr.shape, ytr.shape, " Val:", Xva.shape, yva.shape, " Test:", Xte.shape, yte.shape)

    # ---- Torch setup ----
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    model = LSTMRegressor(n_features=Xtr.shape[-1], hidden_size=64, num_layers=2, dropout=0.2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    train_loader = make_loader(Xtr, ytr, batch_size=256, shuffle=True)
    val_loader = make_loader(Xva, yva, batch_size=512, shuffle=False)

    # ---- Training loop ----
    best_val = float("inf")
    best_path = "data/datasets/lstm_best.pt"

    for epoch in range(1, 11):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_losses.append(float(loss.item()))

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = model(xb)
                val_losses.append(float(loss_fn(yhat, yb).item()))

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d} | train MSE {tr:.6f} | val MSE {va:.6f}")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), best_path)
            print("  saved:", best_path)

    # ---- Test evaluation ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    Xte_t = torch.from_numpy(Xte).to(device)
    yte_t = torch.from_numpy(yte).to(device)

    with torch.no_grad():
        yhat = model(Xte_t)
        test_mse = float(torch.mean((yhat - yte_t) ** 2).item())
        test_rmse = rmse(yhat, yte_t)

    print("Test MSE:", round(test_mse, 6), " Test RMSE:", round(test_rmse, 6))