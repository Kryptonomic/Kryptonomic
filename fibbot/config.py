"""Configuration: built-in defaults deep-merged with an optional JSON file.

Lookup order for the JSON file: explicit path > $FIBBOT_CONFIG > ./fibbot.json (if present).
All state (candles, SQLite DB, brain notebook, reports) lives under cfg["home"]
(default ./.fibbot, override with $FIBBOT_HOME).
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

STABLECOINS = {
    "USDT", "USDC", "DAI", "USDS", "USDE", "FDUSD", "TUSD", "PYUSD", "USD1", "USDD", "BUSD",
    "USDP", "GUSD", "FRAX", "LUSD", "EURC", "EURT", "USDX", "USDB", "USD0", "RLUSD", "USDF",
    "BFUSD", "USDTB", "USDG", "SUSDE", "SUSDS", "XAUT", "PAXG",
}
# Wrapped / staked / bridged duplicates of other assets, and exchange IOUs.
WRAPPED = {
    "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBBTC", "RETH", "METH", "EZETH", "RSETH",
    "WBETH", "BSC-USD", "JITOSOL", "MSOL", "BNSOL", "LBTC", "SOLVBTC", "TBTC", "WBT", "BTCB",
    "CLBTC", "OSETH", "WTRX", "STSOL",
}

DEFAULTS: dict = {
    "home": None,                       # resolved by load_config: $FIBBOT_HOME or ./.fibbot
    "universe": {
        "top_n": 12,                    # how many coins (by market cap) to trade
        "quote": "USDT",
        "min_history_days": 365,        # skip coins with less listing history than this
        "exclude": sorted(STABLECOINS | WRAPPED),
        "extra_exclude": [],            # user additions, e.g. ["TRX"]
        "static_fallback": [            # used when CoinGecko / exchangeInfo are unreachable
            "BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT",
            "TRXUSDT", "LINKUSDT", "AVAXUSDT", "XLMUSDT", "BCHUSDT", "LTCUSDT", "DOTUSDT",
        ],
        "refresh_days": 7,              # re-rank the universe this often
    },
    "data": {
        "history_start": "2020-01-01",  # earliest candle to download
        "timeframes": ["1h", "4h", "1d"],
        "binance_base": "https://data-api.binance.vision",
        "okx_base": "https://www.okx.com",
        "coingecko_base": "https://api.coingecko.com",
    },
    "costs": {"fee_rate": 0.001, "slippage": 0.0005},
    "portfolio": {                       # used by backtests AND paper accounts
        "starting_equity": 10000.0,
        "risk_per_trade": 0.005,         # 0.5% of equity at risk per trade (to the initial stop)
        "max_open_positions": 6,
        "max_total_risk": 0.03,          # sum of open initial-stop risk <= 3% of equity
        "max_notional_frac": 0.25,       # no single position larger than 25% of equity (no leverage)
        "daily_loss_limit": 0.02,        # stop opening trades for the UTC day after -2% realized
    },
    "research": {
        "validation_months": 12,         # most recent N months = locked validation window
        "folds": 4,                      # walk-forward folds inside the research window
        "min_trades": 30,                # fewer trades than this = "insufficient" result
        "max_backtest_units_per_cycle": 200,   # 1 unit = one spec evaluated on the research window
        "max_validations_per_cycle": 3,
        "max_grid_size": 64,
        "workers": 0,                    # process pool size for multi-symbol runs (0 = cpu_count-1, 1 = serial)
        "validation_gate": {
            "min_trades": 30,
            "min_profit_factor": 1.2,
            "min_expectancy_r": 0.05,
            "min_deflated_sharpe": 0.90,
            "max_drawdown": 0.20,
            "min_symbol_win_frac": 0.5,  # fraction of traded symbols with positive total R
        },
    },
    "paper": {
        "max_roster": 4,                 # strategies paper trading at once
        "starting_equity": 10000.0,      # per strategy sub-account
    },
    "graduation": {
        "min_days": 60,
        "min_trades": 40,
        "min_profit_factor": 1.3,
        "max_drawdown": 0.15,
        "ci_level": 0.95,                # bootstrap CI on mean R must have lower bound > 0
        "min_positive_month_frac": 0.6,
        "min_deflated_sharpe": 0.95,
        "max_shortfall_vs_validation_r": 0.25,  # paper mean R may trail validation mean R by at most this
    },
    "brain": {
        "enabled": True,
        "model": "claude-opus-5-5",
        "effort": "high",                # research/review cycles
        "signal_effort": "medium",       # per-signal reviews
        "max_tokens": 16000,
        "max_iterations": 60,            # tool-runner turns per cycle
        "fallbacks": "default",          # server-side refusal fallback (beta server-side-fallback-2026-07-01)
        "signal_review": "shadow",       # off | shadow (record only) | active (veto legs in paper)
        "max_signal_reviews_per_day": 40,
        "research_every_hours": 168,     # weekly research cycle (immediately if roster is empty)
        "review_every_hours": 24,        # daily review of paper results
        "notebook_max_chars": 24000,
    },
    "scheduler": {
        "tick_every_seconds": 3600,      # paper tick cadence (hourly, just after the candle close)
        "tick_offset_seconds": 90,
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    """Defaults <- JSON file <- overrides. Resolves cfg['home'] to an absolute path string."""
    cfg = copy.deepcopy(DEFAULTS)
    p = path or os.environ.get("FIBBOT_CONFIG")
    if not p and Path("fibbot.json").exists():
        p = "fibbot.json"
    if p:
        with open(p) as f:
            cfg = deep_merge(cfg, json.load(f))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    home = cfg.get("home") or os.environ.get("FIBBOT_HOME") or ".fibbot"
    cfg["home"] = str(Path(home).expanduser().resolve())
    return cfg


def home_path(cfg: dict, *parts: str) -> Path:
    """Path under the state directory; creates parent directories."""
    p = Path(cfg["home"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def excluded_bases(cfg: dict) -> set[str]:
    u = cfg["universe"]
    return {s.upper() for s in list(u.get("exclude", [])) + list(u.get("extra_exclude", []))}
