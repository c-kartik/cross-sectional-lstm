from __future__ import annotations
import pandas as pd

from src.data.load_prices import load_prices
from src.universe.custom import get_universe
from src.features.build_features import build_features
from src.labels.forward_returns import forward_return

if __name__ == "__main__":
    prices = load_prices()
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px_all = prices[price_col].unstack("ticker").sort_index()

    universe = [t for t in get_universe() if t in px_all.columns]
    px = px_all[universe].dropna(how="all")

    X = build_features(px)
    y = forward_return(px, horizon=5)

    df = X.join(y, how="inner").dropna()

    print("Universe tickers:", len(universe))
    print("Feature columns:", list(X.columns))
    print("Rows (date,ticker):", len(df))
    print("Date range:", df.index.get_level_values("date").min(), "→", df.index.get_level_values("date").max())
    print(df.head())