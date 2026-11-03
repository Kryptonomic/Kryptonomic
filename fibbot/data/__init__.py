"""Data layer - the only part of fibbot that touches the network.

- universe: CoinGecko market-cap ranking filtered to Binance spot pairs with enough history.
- candles: Binance klines (OKX fallback), the CSV.gz candle cache, and resampling.
"""
from .candles import CandleStore, FetchError, fetch_klines, resample
from .universe import get_universe, rank_universe

__all__ = ["CandleStore", "FetchError", "fetch_klines", "get_universe", "rank_universe", "resample"]
