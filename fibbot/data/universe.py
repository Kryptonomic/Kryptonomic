"""Tradable universe: the top-N coins by market cap with a Binance spot pair and enough history.

`rank_universe` always uses the network; `get_universe` is the cached entry point that never fails.
Network calls go through `candles.http_json` (looked up at call time, so tests can patch it).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..config import excluded_bases, home_path
from . import candles

log = logging.getLogger(__name__)

EXCHANGE_CACHE_MAX_AGE = 86400   # seconds the Binance TRADING symbol set is reused
GECKO_TRIES = 2                  # CoinGecko /coins/markets attempts per ranking (re-ranking is never urgent)
GECKO_MAX_DELAY = 10.0           # longest wait between them; a longer Retry-After gives up at once
RANK_RETRY_SECONDS = 6 * 3600    # after a failed ranking, serve the fallback list this long before retrying


def _read_cache(path: Path) -> dict | None:
    """A {"as_of", "symbols": [...], ...} cache file, or None if missing or malformed."""
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(d, dict) and isinstance(d.get("symbols"), list) and isinstance(d.get("as_of"), (int, float)):
        return d
    return None


def _write_cache(path: Path, symbols: list[str], **extra) -> None:
    candles.atomic_write(path, json.dumps({"as_of": int(time.time()), "symbols": symbols, **extra}).encode())


def binance_spot_symbols(cfg: dict, max_age: float = EXCHANGE_CACHE_MAX_AGE) -> set[str]:
    """Symbols with status TRADING on Binance spot; cached in home/exchange_symbols.json for `max_age` s."""
    path = home_path(cfg, "exchange_symbols.json")
    cached = _read_cache(path)
    if cached and time.time() - cached["as_of"] < max_age:
        return set(cached["symbols"])
    info = candles.http_json(f"{cfg['data']['binance_base']}/api/v3/exchangeInfo", {"permissions": "SPOT"})
    try:
        symbols = sorted(s["symbol"] for s in info["symbols"] if s.get("status") == "TRADING")
    except (KeyError, TypeError, AttributeError) as e:
        raise RuntimeError(f"unexpected Binance exchangeInfo response: {e!r}") from e
    _write_cache(path, symbols)
    return set(symbols)


def first_daily_t(cfg: dict, symbol: str) -> int | None:
    """Open time (s) of the symbol's first Binance daily kline (~listing date); None if it has none."""
    rows = candles.http_json(f"{cfg['data']['binance_base']}/api/v3/klines",
                             {"symbol": symbol, "interval": "1d", "startTime": 0, "limit": 1})
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected Binance klines response for {symbol}: {str(rows)[:200]}")
    return int(rows[0][0]) // 1000 if rows else None


def rank_universe(cfg: dict) -> list[str]:
    """Top `universe.top_n` symbols f"{BASE}{quote}" in CoinGecko market-cap order.

    Drops excluded bases (stablecoins, wrapped/staked assets, user extras), coins without a TRADING
    Binance spot pair, and coins with less than `universe.min_history_days` of daily klines.
    Raises RuntimeError (FetchError) on network failure.
    """
    u = cfg["universe"]
    quote, top_n = u["quote"].upper(), int(u["top_n"])
    markets = candles.http_json(f"{cfg['data']['coingecko_base']}/api/v3/coins/markets",
                                {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 100, "page": 1},
                                tries=GECKO_TRIES, max_delay=GECKO_MAX_DELAY)
    if not isinstance(markets, list):
        raise RuntimeError(f"unexpected CoinGecko response: {str(markets)[:200]}")
    excluded = excluded_bases(cfg)
    tradable = binance_spot_symbols(cfg)
    listed_before = time.time() - float(u["min_history_days"]) * 86400
    out: list[str] = []
    seen: set[str] = set()
    for coin in markets:
        base = str(coin.get("symbol") or "").upper() if isinstance(coin, dict) else ""
        sym = base + quote
        if not base or base in excluded or sym in seen or sym not in tradable:
            continue
        seen.add(sym)
        first = first_daily_t(cfg, sym)
        if first is None or first > listed_before:
            continue
        out.append(sym)
        if len(out) >= top_n:
            break
    return out


def _params(cfg: dict) -> dict:
    """Settings a cached ranking depends on; a cache made with different ones is stale."""
    u = cfg["universe"]
    return {"top_n": int(u["top_n"]), "quote": u["quote"].upper(),
            "min_history_days": u["min_history_days"], "exclude": sorted(excluded_bases(cfg))}


def _filter(cfg: dict, symbols) -> list[str]:
    """Keep quote-currency symbols whose base is not excluded; truncate to top_n."""
    quote = cfg["universe"]["quote"].upper()
    excluded = excluded_bases(cfg)
    keep = [s for s in symbols if s.endswith(quote) and s[:-len(quote)] and s[:-len(quote)] not in excluded]
    return keep[:int(cfg["universe"]["top_n"])]


def _record_failure(path: Path, cached: dict | None, params: dict, now: float, err: Exception) -> None:
    """Note a failed ranking in universe.json: "failed_t", "failed_params", "failed_error" are added
    and as_of / symbols / params stay as they were, so the age of the cached ranking stays visible.
    Without a cache the record has as_of 0 and no symbols. Never raises."""
    d = dict(cached) if cached else {"as_of": 0, "symbols": []}
    d.update(failed_t=int(now), failed_params=params, failed_error=str(err)[:300])
    try:
        candles.atomic_write(path, json.dumps(d).encode())
    except OSError as e:
        log.warning("could not record the failed universe ranking in %s: %s", path, e)


def _failed_recently(cached: dict | None, params: dict, now: float) -> bool:
    """True while a ranking with these settings failed less than RANK_RETRY_SECONDS ago."""
    failed_t = cached.get("failed_t") if cached else None
    return (isinstance(failed_t, (int, float)) and cached.get("failed_params") == params
            and 0 <= now - failed_t < RANK_RETRY_SECONDS)


def get_universe(cfg: dict, refresh: bool = False) -> list[str]:
    """Current universe, cached in home/universe.json {"as_of": t, "symbols": [...], "params": {...}}.

    Re-ranks when the cache is older than universe.refresh_days, was made with different universe
    settings, or refresh=True. If ranking fails (or yields nothing) falls back to the cached list,
    else universe.static_fallback - both filtered by excluded bases and truncated to top_n. A failed
    ranking is recorded in the cache file ("failed_t") and, unless refresh=True, not retried for
    RANK_RETRY_SECONDS (every scheduled run is a new process, so it would otherwise retry hourly).
    """
    path = home_path(cfg, "universe.json")
    cached = _read_cache(path)
    params = _params(cfg)
    max_age = float(cfg["universe"]["refresh_days"]) * 86400
    now = time.time()
    if cached and not refresh and cached.get("params") == params and now - cached["as_of"] < max_age:
        return list(cached["symbols"])
    fallback = _filter(cfg, cached["symbols"]) if cached else []
    if not refresh and _failed_recently(cached, params, now):
        log.info("universe ranking failed %.1f h ago (retry after %.0f h); using the %s list",
                 (now - cached["failed_t"]) / 3600, RANK_RETRY_SECONDS / 3600,
                 "cached" if fallback else "static")
        return fallback or _filter(cfg, cfg["universe"]["static_fallback"])
    try:
        symbols = rank_universe(cfg)
        if not symbols:
            raise RuntimeError("ranking returned no symbols")
    except Exception as e:  # any failure -> fallback list; the universe must always be available
        log.warning("universe ranking failed (%s); using the %s list", e, "cached" if fallback else "static")
        _record_failure(path, cached, params, now, e)
        return fallback or _filter(cfg, cfg["universe"]["static_fallback"])
    _write_cache(path, symbols, params=params)
    return symbols
