"""Configuration: built-in defaults deep-merged with an optional JSON file.

Lookup order for the JSON file: explicit path > $FIBBOT_CONFIG > ./fibbot.json (if present).
All state (candles, SQLite DB, brain notebook, reports) lives under cfg["home"]: the "home"
override (CLI --home) > $FIBBOT_HOME > "home" in the JSON file > ./.fibbot. The environment
wins over the file so a deployment that sets FIBBOT_HOME (the Docker image, the Actions
workflow) keeps its state where it expects it whatever a mounted or committed config says.

load_config() validates the result (validate_config): an unknown key (a typo such as "efort"),
a wrong type or an out-of-range or unknown value raises ValueError naming the key, instead of
being silently ignored or misread (e.g. "signal_review": false would keep paying for reviews).
"""
from __future__ import annotations

import copy
import difflib
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

from .types import TF_SECONDS

log = logging.getLogger(__name__)

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
    "costs": {
        "fee_rate": 0.001,               # per side, on notional
        "slippage": 0.0005,              # on market and stop fills
        "limit_through": 0.0002,         # a resting limit entry fills only when the bar trades this
                                         # fraction through its price (0 = fill on touch)
    },
    "portfolio": {                       # used by backtests AND paper accounts
        "starting_equity": 10000.0,
        "risk_per_trade": 0.005,         # 0.5% of equity at risk per trade (to the initial stop)
        "max_open_positions": 6,
        "max_total_risk": 0.03,          # sum of open initial-stop risk <= 3% of equity
        "max_notional_frac": 0.25,       # no single position larger than 25% of equity (no leverage)
        "daily_loss_limit": 0.02,        # stop opening trades for the UTC day after -2% realized
        "max_total_notional": 1.0,       # total open notional <= this x equity: no leverage
    },
    "research": {
        "validation_months": 12,         # most recent N months = locked validation window
        "folds": 4,                      # walk-forward folds inside the research window
        "min_trades": 30,                # fewer trades than this = "insufficient" result
        "max_backtest_units_per_cycle": 200,   # 1 unit = one spec evaluated on the research window
        "max_validations_per_cycle": 3,
        "max_validations_per_week": 3,   # validations recorded in the last 7 days across brain cycles
                                         # and the autopilot (CLI ones count but are not capped); null disables
        "validation_universe": "point_in_time",  # or "current" (today's universe): validation symbols
                                         # are ranked by 90-day quote volume before validation_start
        "max_grid_size": 64,
        "workers": 0,                    # process pool size for multi-symbol runs (0 = cpu_count-1, 1 = serial)
        "validation_gate": {
            "min_trades": 30,
            "min_profit_factor": 1.2,
            "min_expectancy_r": 0.05,
            "min_deflated_sharpe": 0.90,
            "max_drawdown": 0.20,
            "min_symbol_win_frac": 0.5,  # fraction of traded symbols with positive total R
            "min_validation_edge_r": 0.0,  # validation mean R minus the mean of random-entry runs
                                           # on the validation window (null disables)
            "min_validation_edge_z": 1.65, # z score of that edge (one-sided 5%; null disables)
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
        "ci_level": 0.95,                # anytime-valid confidence sequence of mean R (trades clustered
                                         # by UTC entry week) must have lower bound > 0
        "min_positive_month_frac": 0.6,
        "min_deflated_sharpe": 0.95,
        "max_shortfall_vs_validation_r": 0.25,  # check value = validation mean R - paper mean R;
                                                # passes when <= this
        "min_edge_r": 0.0,               # paper mean R minus random entries' mean R over the paper
                                         # period (null disables)
        "min_edge_z": 1.65,              # z score of that edge (one-sided 5%; null disables)
    },
    "brain": {
        "enabled": True,
        "model": "claude-opus-5-5",
        "effort": "high",                # research/review cycles
        "signal_effort": "medium",       # per-signal reviews
        "max_tokens": 64000,             # per reply, thinking included; streamed, so it may exceed 21k
        "max_iterations": 60,            # tool-runner turns per research cycle
        "max_retries": 8,                # SDK retries per request (honours retry-after)
        "signal_max_tokens": 16000,      # per-signal review reply (not streamed: at most 21000)
        "review_max_iterations": 20,     # turn cap of review cycles (at most max_iterations)
        "review_effort": "medium",       # effort of review cycles
        "max_monthly_output_tokens": None,  # optional spend cap on output tokens per UTC month (None = off)
        "fallbacks": "default",          # server-side refusal fallback (beta server-side-fallback-2026-07-01)
        "signal_review": "shadow",       # off | shadow (record only) | active (veto legs in paper)
        "max_signal_reviews_per_day": 40,
        "research_every_hours": 168,     # weekly research cycle (immediately if roster is empty)
        "review_every_hours": 24,        # daily review of paper results
        "notebook_max_chars": 24000,
        "max_cycle_minutes": 60,         # wall-clock limit of one research/review cycle (0 = none)
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
    """Defaults <- JSON file <- overrides, validated (validate_config: ValueError naming the bad
    key). cfg['home'] is resolved to an absolute path string: overrides["home"] >
    $FIBBOT_HOME > the file's "home" > ".fibbot"."""
    cfg, file_cfg = copy.deepcopy(DEFAULTS), {}
    p = path or os.environ.get("FIBBOT_CONFIG")
    if not p and Path("fibbot.json").exists():
        p = "fibbot.json"
    if p:
        with open(p) as f:
            file_cfg = json.load(f)
        if not isinstance(file_cfg, dict):
            raise ValueError(f"{p}: the config file must hold a JSON object, got {type(file_cfg).__name__}")
        cfg = deep_merge(cfg, file_cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    over_home, env_home, file_home = (overrides or {}).get("home"), os.environ.get("FIBBOT_HOME"), \
        file_cfg.get("home")
    for where, h in (("home", over_home), (f"{p}: home", file_home)):
        if h is not None and not isinstance(h, (str, os.PathLike)):
            raise ValueError(f"{where}: must be a path string, got {h!r}")
    if env_home and file_home and not over_home and _abs(env_home) != _abs(file_home):
        log.warning("config %s sets home %r, but $FIBBOT_HOME=%r takes precedence", p, file_home, env_home)
    cfg["home"] = _abs(over_home or env_home or file_home or ".fibbot")
    try:
        validate_config(cfg)
    except ValueError as e:
        raise ValueError(f"{p}: {e}" if p else str(e)) from None
    return cfg


def _abs(path) -> str:
    return str(Path(path).expanduser().resolve())


# -- validation --------------------------------------------------------------------------------

EFFORTS = ("low", "medium", "high", "xhigh", "max")   # output_config.effort values the API accepts
SIGNAL_REVIEW_MODES = ("off", "shadow", "active")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _num(lo=None, hi=None, *, lo_open=False, hi_open=False, integer=False, null=False):
    """Rule: a finite number (an int when `integer`, never a bool) within [lo, hi] (open ends
    with lo_open / hi_open); None too when `null`."""
    what = "an integer" if integer else "a number"
    lo_s = "" if lo is None else f"{'>' if lo_open else '>='} {lo:g}"
    hi_s = "" if hi is None else f"{'<' if hi_open else '<='} {hi:g}"
    need = what + (f" {lo_s}" if lo_s else "") + (" and" if lo_s and hi_s else "") + (
        f" {hi_s}" if hi_s else "") + (" (or null)" if null else "")

    def check(v):
        if v is None and null:
            return None
        ok = isinstance(v, int) and not isinstance(v, bool) if integer else _is_num(v)
        if ok and lo is not None:
            ok = v > lo if lo_open else v >= lo
        if ok and hi is not None:
            ok = v < hi if hi_open else v <= hi
        return None if ok else f"must be {need}, got {v!r}"
    return check


def _enum(*choices, null=False):
    def check(v):
        if (v is None and null) or (isinstance(v, str) and v in choices):
            return None
        return f"must be one of {'|'.join(choices)}{' (or null)' if null else ''}, got {v!r}"
    return check


def _str(null=False, prefix: tuple[str, ...] = ()):
    def check(v):
        if v is None and null:
            return None
        if not isinstance(v, str) or not v.strip():
            return f"must be a non-empty string, got {v!r}"
        if prefix and not v.startswith(prefix):
            return f"must start with {' or '.join(prefix)}, got {v!r}"
        return None
    return check


def _bool(v):
    return None if isinstance(v, bool) else f"must be true or false, got {v!r}"


def _list(item, nonempty=False):
    def check(v):
        if not isinstance(v, list):
            return f"must be a list, got {v!r}"
        if nonempty and not v:
            return "must not be empty"
        for x in v:
            msg = item(x)
            if msg:
                return f"item {msg}"
        return None
    return check


def _day(v):
    try:
        datetime.strptime(v, "%Y-%m-%d")
    except (TypeError, ValueError):
        return f"must be a date 'YYYY-MM-DD', got {v!r}"
    return None


def _fallbacks(v):
    if v is None or v is False or isinstance(v, str):
        return None
    if isinstance(v, list) and all(isinstance(x, (str, dict)) for x in v):
        return None
    return f"must be a string (e.g. \"default\"), a list of models, or null/false, got {v!r}"


_FRAC = _num(0, 1, lo_open=True)               # (0, 1]
_UNIT = _num(0, 1)                             # [0, 1]
_COUNT = _num(0, integer=True)                 # 0, 1, 2, ...
_POS_INT = _num(1, integer=True)               # 1, 2, ...
_URL = _str(prefix=("http://", "https://"))

# Every key of DEFAULTS (dotted path) -> rule(value) -> error message or None.
RULES: dict = {
    "home": _str(null=True),
    "universe.top_n": _POS_INT,
    "universe.quote": _str(),
    "universe.min_history_days": _num(0),
    "universe.exclude": _list(_str()),
    "universe.extra_exclude": _list(_str()),
    "universe.static_fallback": _list(_str()),
    "universe.refresh_days": _num(0, lo_open=True),
    "data.history_start": _day,
    "data.timeframes": _list(_enum(*TF_SECONDS), nonempty=True),
    "data.binance_base": _URL,
    "data.okx_base": _URL,
    "data.coingecko_base": _URL,
    "costs.fee_rate": _num(0, 0.1),
    "costs.slippage": _num(0, 0.1),
    "costs.limit_through": _num(0, 0.05),       # the range types.Costs accepts
    "portfolio.starting_equity": _num(0, lo_open=True),
    "portfolio.risk_per_trade": _FRAC,
    "portfolio.max_open_positions": _POS_INT,
    "portfolio.max_total_risk": _FRAC,
    "portfolio.max_notional_frac": _FRAC,
    "portfolio.daily_loss_limit": _UNIT,         # 0 disables the rule
    "portfolio.max_total_notional": _num(0, lo_open=True),
    "research.validation_months": _num(0, lo_open=True),
    "research.folds": _POS_INT,
    "research.min_trades": _COUNT,
    "research.max_backtest_units_per_cycle": _COUNT,
    "research.max_validations_per_cycle": _COUNT,
    "research.max_validations_per_week": _num(0, integer=True, null=True),   # null = no cap
    "research.validation_universe": _enum("point_in_time", "current"),
    "research.max_grid_size": _POS_INT,
    "research.workers": _COUNT,                  # 0 = cpu_count - 1
    "research.validation_gate.min_trades": _COUNT,
    "research.validation_gate.min_profit_factor": _num(0),
    "research.validation_gate.min_expectancy_r": _num(),
    "research.validation_gate.min_deflated_sharpe": _UNIT,
    "research.validation_gate.max_drawdown": _FRAC,
    "research.validation_gate.min_symbol_win_frac": _UNIT,
    "research.validation_gate.min_validation_edge_r": _num(null=True),       # null = no check
    "research.validation_gate.min_validation_edge_z": _num(null=True),       # null = no check
    "paper.max_roster": _COUNT,
    "paper.starting_equity": _num(0, lo_open=True),
    "graduation.min_days": _num(0),
    "graduation.min_trades": _COUNT,
    "graduation.min_profit_factor": _num(0),
    "graduation.max_drawdown": _FRAC,
    "graduation.ci_level": _num(0, 1, lo_open=True, hi_open=True),
    "graduation.min_positive_month_frac": _UNIT,
    "graduation.min_deflated_sharpe": _UNIT,
    "graduation.max_shortfall_vs_validation_r": _num(0),
    "graduation.min_edge_r": _num(null=True),
    "graduation.min_edge_z": _num(null=True),
    "brain.enabled": _bool,
    "brain.model": _str(),
    "brain.effort": _enum(*EFFORTS),
    "brain.signal_effort": _enum(*EFFORTS),
    "brain.max_tokens": _POS_INT,
    "brain.max_iterations": _POS_INT,
    "brain.max_retries": _num(0, integer=True, null=True),
    "brain.signal_max_tokens": _num(1, 21000, integer=True, null=True),
    "brain.review_max_iterations": _num(1, integer=True, null=True),
    "brain.review_effort": _enum(*EFFORTS, null=True),
    "brain.max_monthly_output_tokens": _num(0, integer=True, null=True),
    "brain.fallbacks": _fallbacks,
    "brain.signal_review": _enum(*SIGNAL_REVIEW_MODES),
    "brain.max_signal_reviews_per_day": _COUNT,
    "brain.research_every_hours": _num(0, lo_open=True),
    "brain.review_every_hours": _num(0, lo_open=True),
    "brain.notebook_max_chars": _POS_INT,
    "brain.max_cycle_minutes": _num(0, null=True),
    "scheduler.tick_every_seconds": _POS_INT,
    "scheduler.tick_offset_seconds": _COUNT,
}

# Keys other modules read with their own default (not in DEFAULTS): accepted when present.
# A module that starts reading a new config key must add it to DEFAULTS or here, otherwise
# load_config rejects it as unknown.
OPTIONAL_KEYS: dict = {}


def known_keys(prefix: str = "") -> list[str]:
    """Every accepted dotted key (DEFAULTS and OPTIONAL_KEYS), or those directly under `prefix`."""
    keys = sorted({*RULES, *OPTIONAL_KEYS, *(_paths(DEFAULTS))})
    if not prefix:
        return keys
    return [k for k in keys if k.startswith(prefix + ".") and "." not in k[len(prefix) + 1:]]


def _paths(d: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in d.items():
        out.append(prefix + k)
        if isinstance(v, dict):
            out += _paths(v, prefix + k + ".")
    return out


def validate_config(cfg: dict) -> dict:
    """Check a merged config: no unknown keys (with a "did you mean" hint), the right types and
    allowed values (RULES / OPTIONAL_KEYS). Values are never normalised (e.g. "Off" is rejected,
    not lower-cased), so what status shows is what runs. Returns cfg; raises ValueError listing
    every problem."""
    errors: list[str] = []
    _walk(cfg, DEFAULTS, "", errors)
    if errors:
        raise ValueError("invalid config: " + "; ".join(errors))
    return cfg


def _walk(node, defaults: dict, prefix: str, errors: list[str]) -> None:
    if not isinstance(node, dict):
        errors.append(f"{prefix[:-1] or 'config'}: must be a JSON object, got {node!r}")
        return
    optional = {k[len(prefix):] for k in OPTIONAL_KEYS
                if k.startswith(prefix) and "." not in k[len(prefix):]}
    for key, value in node.items():
        path = prefix + str(key)
        if key in defaults and isinstance(defaults[key], dict):
            _walk(value, defaults[key], path + ".", errors)
            continue
        if key not in defaults and key not in optional:
            near = difflib.get_close_matches(str(key), [*defaults, *optional], n=1, cutoff=0.6)
            errors.append(f"{path}: unknown key" + (f" (did you mean {near[0]!r}?)" if near else ""))
            continue
        rule = RULES.get(path) or OPTIONAL_KEYS.get(path)
        msg = rule(value) if rule else None
        if msg:
            errors.append(f"{path}: {msg}")


def home_path(cfg: dict, *parts: str) -> Path:
    """Path under the state directory; creates parent directories."""
    p = Path(cfg["home"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def excluded_bases(cfg: dict) -> set[str]:
    u = cfg["universe"]
    return {s.upper() for s in list(u.get("exclude", [])) + list(u.get("extra_exclude", []))}
