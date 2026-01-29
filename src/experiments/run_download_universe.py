from __future__ import annotations
from src.universe.custom import get_universe
from src.data.download_prices import DownloadConfig, download_and_cache_prices

if __name__ == "__main__":
    tickers = get_universe()
    cfg = DownloadConfig(start="2016-01-01", end="2026-01-01")
    download_and_cache_prices(tickers, cfg, force=False)