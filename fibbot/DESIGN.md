# fibbot design contract

This file is the contract between modules. Implementations must match the names, signatures and
semantics below. When this file and a module disagree, fix the module (or update this file in the
same change, deliberately).

Goal: a bot that paper trades Fibonacci-retracement setups on high market-cap coins, researches
which setups work (backtest -> walk-forward -> locked validation -> forward paper trading), and only
declares a strategy "reliably profitable" when a deterministic statistical gate passes on PAPER
results. A Claude "brain" runs the research loop through tools; deterministic code enforces every
risk limit, budget and gate. **No live trading code exists in this package.**

## Conventions

- Python >= 3.10. Core package is **standard library only**. `anthropic` is imported lazily, only
  inside `fibbot/brain/`; requirements.txt pins `anthropic==2.4.0` and its whole dependency tree
  (upgrades are deliberate: bump the pins, reinstall, run the tests).
- Time: integer Unix **seconds**, UTC. `Bars.t[i]` is the bar's OPEN time. Only closed bars exist.
- `side`: +1 long, -1 short.
- Missing indicator values are `float('nan')` (lists stay index-aligned with bars).
- Every module that can be pure is pure. No module except `fibbot/data/*` touches the network.
- Tests: `tests/test_<module>.py`, written with `unittest` (runnable by `python -m unittest` and
  `pytest`), no network, use synthetic data. Test files may build bars with `Bars.from_rows`.
- JSON returned to the brain must be compact (<= ~12k chars per tool result) and JSON-serializable
  (`json.dumps(x, default=str)`), floats rounded to 4 significant decimals where sensible.

Core types live in `fibbot/types.py`: `Bars`, `Trade`, `Signal`, `Costs`, `TF_SECONDS`,
`BARS_PER_YEAR`. `Costs(fee_rate=0.001, slippage=0.0005, limit_through=0.0002)`: limit_through is
a fraction in [0, 0.05] (ValueError otherwise); a resting limit ENTRY fills only when the bar
trades at least this far through its price, 0 restores fill-on-touch. `Trade.risk_pct` is
`side * (entry - initial stop) / entry`, or the order's planned risk for a fill at or beyond the
stop (see Accounting).

Config lives in `fibbot/config.py`: `load_config()`, `home_path()`, `DEFAULTS`, `excluded_bases()`,
`validate_config()`, `RULES`, `OPTIONAL_KEYS`, `known_keys()`. `load_config(path, overrides)` merges
DEFAULTS <- JSON file <- overrides and validates the result: an unknown key (with a "did you mean"
hint), a wrong type, an out-of-range or unknown enum value raises ValueError naming the key (the
CLI prints it and exits 1). Values are case-sensitive and never normalised (signal_review exactly
off|shadow|active, efforts low|medium|high|xhigh|max). Accepted keys are DEFAULTS (every leaf has a
rule in RULES) plus OPTIONAL_KEYS (keys a module reads with its own default; currently empty). Home
precedence: overrides["home"] (CLI --home) > $FIBBOT_HOME > the file's "home" > ./.fibbot; a warning
is logged when the file and the environment disagree. `run_forever` validates a config built in
code too.

## Module map

```
fibbot/
  types.py config.py                    foundation: core types, config defaults and validation
  data/universe.py data/candles.py      network: CoinGecko ranking, Binance klines (+OKX fallback), CSV.gz cache
  indicators.py fib.py                  pure math: EMA/ATR/RSI/ADX, causal ZigZag pivots, fib levels
  metrics.py portfolio.py               stats (PF, expectancy, DD, Sharpe, clustering, confidence sequence, deflated Sharpe), sizing sim
  store.py                              SQLite persistence
  strategy.py                           StrategySpec, PARAM_SPACE, PRESETS, FibEngine (bar-by-bar state machine)
  backtest.py research.py               windows, backtests, walk-forward, random baseline, validation gate
  paper.py graduation.py                forward paper trading (replay design), graduation gate
  lab.py autopilot.py report.py         facade used by CLI + brain; deterministic fallback research; REPORT.md
  brain/client.py tools.py prompts.py cycles.py   Claude brain
  scheduler.py cli.py __main__.py       run loop and command line
tests/
```

---

## fibbot/data/universe.py

```python
def rank_universe(cfg) -> list[str]
    """Network. CoinGecko /api/v3/coins/markets (vs_currency=usd, order=market_cap_desc, per_page=100),
    drop excluded bases (config.excluded_bases), map base -> f"{BASE}{quote}" and keep symbols that are
    TRADING spot pairs on Binance (GET {binance_base}/api/v3/exchangeInfo?permissions=SPOT, cache the
    symbol set), keep those with >= min_history_days of daily klines, return the first top_n in
    market-cap order. Raises RuntimeError on network failure."""

def get_universe(cfg, refresh: bool = False) -> list[str]
    """Cached in home/universe.json {"as_of": t, "symbols": [...], "params"} ; re-rank when older than
    universe.refresh_days, made with other settings, or refresh=True. On any network failure (or an
    empty ranking) fall back to the cached list, else universe.static_fallback (filtered by excluded
    bases) truncated to top_n. A failed ranking is recorded in universe.json as failed_t,
    failed_params, failed_error (as_of, symbols, params stay; with no cache the record is
    {as_of: 0, symbols: []}). For RANK_RETRY_SECONDS (6 h) after that, a call with the same settings
    serves the cached / static list without touching the network; refresh=True ignores this and a
    settings change retries once. rank_universe calls CoinGecko with at most 2 tries, waiting at most
    10 s."""
```

## fibbot/data/candles.py

```python
def fetch_klines(cfg, symbol: str, tf: str, start_t: int, end_t: int | None = None,
                 fallback_start_t: int | None = None) -> list[tuple]
    """Network. Binance GET {binance_base}/api/v3/klines?symbol=&interval=&startTime=ms&limit=1000,
    paginated until end_t/now. Returns [(t, o, h, l, c, v), ...] with t in SECONDS, CLOSED bars only
    (drop any bar whose close time >= now - CLOSED_MARGIN_MS, 5 s for clock skew). If Binance fails
    for a recent range, try OKX /api/v5/market/history-candles (instId BASE-QUOTE, bar 1H/4H/1Dutc) as
    a fallback, from max(start_t, fallback_start_t); that start also decides whether OKX may be used."""

def http_json(url, params=None, timeout=30, tries=5, max_delay=60)
    # exponential backoff that honours Retry-After; no single wait exceeds max_delay and a
    # Retry-After longer than max_delay gives up at once. HTTP 400, 401, 403, 404, 418 and 451 fail
    # at once and are never retried. Raises FetchError(status=...).

class CandleStore:
    """Local cache of closed candles: home/candles/{SYMBOL}_{tf}.csv.gz with header t,o,h,l,c,v."""
    def __init__(self, cfg): ...
    def get(self, symbol: str, tf: str) -> Bars            # from disk (+ in-memory cache); empty Bars if none; no network
    def update(self, symbol: str, tf: str) -> Bars         # incremental fetch (or from data.history_start), save, return
        # The fetch starts AT the last cached bar, which Binance serves again: if it differs (e.g.
        # stored while still forming because the local clock was fast) it is replaced with a
        # warning; an identical re-fetch rewrites nothing. The OKX fallback starts at the next bar
        # and never revises a cached bar.
    def update_all(self, symbols, tfs) -> dict[tuple[str, str], int]   # {(sym, tf): n_new_bars}; per-symbol errors logged, not raised
    def load_many(self, symbols, tf) -> dict[str, Bars]    # get() for each; skips empty

def resample(bars: Bars, tf: str) -> Bars   # aggregate to a coarser tf aligned to UTC multiples of TF_SECONDS[tf]; drops a trailing incomplete bucket
```

## fibbot/indicators.py  (pure)

All take/return `list[float]` aligned with input, `nan` during warmup.

- `sma(x, n)`; `ema(x, n)`: alpha = 2/(n+1), seeded with the SMA of the first n values at index n-1.
- `true_range(h, l, c)`: tr[0] = h0 - l0; tr[i] = max(h-l, |h-c[i-1]|, |l-c[i-1]|).
- `atr(h, l, c, n)`: Wilder. First value at index n-1 = mean(tr[0..n-1]); then (prev*(n-1)+tr)/n.
- `rsi(c, n=14)`: Wilder; first value at index n. avg_loss == 0 -> 100 (and both 0 -> 50).
- `adx(h, l, c, n=14)`: Wilder's ADX; first value at index 2n-1.

## fibbot/fib.py  (pure)

```python
RETRACE_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.65, 0.705, 0.786, 0.886)
EXTENSION_LEVELS = (1.272, 1.414, 1.618, 2.0, 2.618)

@dataclass(frozen=True)
class Pivot:
    idx: int          # bar index of the extreme
    t: int            # bar open time of the extreme
    price: float      # the high (kind 'H') or low (kind 'L')
    kind: str         # 'H' or 'L'
    confirm_idx: int  # bar index at whose CLOSE the pivot became known (> idx)

class PivotTracker:
    """Causal ZigZag. mode 'atr': reversal threshold = mult * atr_i ; mode 'pct': pct * extreme price."""
    def __init__(self, mult: float = 3.0, mode: str = "atr", pct: float = 0.05): ...
    def update(self, i: int, t: int, high: float, low: float, atr_i: float) -> Pivot | None
        # returns the pivot confirmed at bar i, if any. Pivots strictly alternate H/L.

def zigzag(bars: Bars, atr_values: list[float], mult=3.0, mode="atr", pct=0.05) -> list[Pivot]
    # = feeding every bar to one PivotTracker.

def retrace_price(anchor: float, extreme: float, r: float) -> float
    # extreme - r * (extreme - anchor). Works for both directions:
    # up-leg: anchor=swing low, extreme=running high ; down-leg: anchor=swing high, extreme=running low.
    # r=0 -> extreme, r=1 -> anchor.
def extension_price(anchor: float, extreme: float, e: float) -> float
    # anchor + e * (extreme - anchor). e=1 -> extreme ; e=1.272 -> beyond the extreme.
    # Note extension e == retracement (1 - e).
def fib_table(anchor, extreme) -> dict[str, float]   # {"0.618": price, ..., "1.272": price, ...} for display
```

PivotTracker algorithm (state: direction 0/+1/-1, hi/hi_idx, lo/lo_idx; thr invalid when atr_i is nan in atr mode):
- direction 0: update hi (h > hi) and lo (l < lo) extremes. If thr valid: if `hi - low_i >= thr` and
  `hi_idx < i` -> confirm H at hi_idx, direction=-1, lo=low_i, lo_idx=i. Elif `high_i - lo >= thr` and
  `lo_idx < i` -> confirm L at lo_idx, direction=+1, hi=high_i, hi_idx=i. (If both, confirm the older extreme.)
- direction +1: if high_i > hi: extend (hi, hi_idx = high_i, i). **elif** thr valid and `hi - low_i >= thr`:
  confirm H at hi_idx (confirm_idx=i); direction=-1; lo, lo_idx = low_i, i.
- direction -1: mirror (extend on low_i < lo, else confirm L when `high_i - lo >= thr`).
- The `elif` means a bar that extends the extreme never confirms a reversal on the same bar.

## fibbot/metrics.py  (pure)

```python
def trade_stats(trades: list[Trade]) -> dict
    # keys: n, wins, win_rate, avg_r (== expectancy_r), median_r, avg_win_r, avg_loss_r, total_r,
    # profit_factor (gross winning R / gross losing R = sum of positive r_multiple / |sum of negative
    # r_multiple|, which matches risk-based sizing; PF > 1 exactly when total R > 0; 99.0 if no
    # losses and n>0; 0.0 if no wins), avg_ret_pct (not a profitability signal: it weighs trades by
    # stop width), avg_bars_held, max_consec_losses, long_n, short_n, exit_reasons {reason: count},
    # sharpe_per_trade (mean/stdev of r_multiple, 0 if n<2). n == 0 -> all numeric keys 0.
    # PF feeds the validation gate, graduation, folds, per-symbol stats and sort_by=profit_factor.
def max_drawdown(values: list[float]) -> float          # positive fraction, e.g. 0.12
def equity_stats(curve: list[tuple[int, float]], start_equity: float) -> dict
    # total_return, cagr, max_drawdown, sharpe, sortino, calmar, n_days. Curve is resampled to daily
    # closes (UTC days, forward-filled) before computing daily returns; annualize with 365.
def monthly_returns(curve, start_equity) -> list[tuple[str, float]]   # [("2024-01", 0.013), ...]
def bootstrap_mean_ci(values, level=0.95, n_boot=2000, seed=7, groups=None) -> tuple[float, float]
    # deterministic; with groups (one cluster label per value) a cluster bootstrap
def entry_week(t) -> int                               # t // 604800: the cluster label of a trade
def design_effect(values, groups) -> float             # CR1 cluster variance of the mean / iid variance, >= 1
def effective_n(values, groups) -> float               # n / design_effect
def clustered_t_stat(values, groups) -> float          # t_stat_mean / sqrt(design_effect)
def confidence_sequence(values, level=0.95, groups=None, target_n=200) -> tuple[float, float]
    # anytime-valid Gaussian-mixture confidence sequence of the mean (Robbins; Waudby-Smith et al.
    # 2024); information time = effective_n when groups are given; (0, 0) if empty; (-99, 99) for
    # one value. May be recomputed after every new value without inflating its error.
def skew_kurt(values) -> tuple[float, float]           # population skew, NON-excess kurtosis (normal=3)
def t_stat_mean(values) -> float
def probabilistic_sharpe(sr, n, skew, kurt, sr_star=0.0) -> float
    # Phi((sr - sr_star) * sqrt(n - 1) / sqrt(1 - skew*sr + (kurt - 1)/4 * sr^2)); guard sqrt of <=0 -> 1e-12;
    # n may be a float effective n
def expected_max_sharpe(n_trials: int, var_sr: float) -> float
    # sqrt(var_sr) * ((1-g)*Phi^-1(1 - 1/N) + g*Phi^-1(1 - 1/(N*e))), g = 0.5772156649 ; 0 if N <= 1
def deflated_sharpe(values, n_trials: int, var_sr: float | None = None, n_eff: float | None = None) -> float
    # PSR of the per-trade Sharpe of `values` against sr_star = expected_max_sharpe(n_trials, var_sr),
    # with n_eff observations (default len(values), clipped to [1, n]); var_sr defaults to 1/n_eff.
    # 0.0 if len(values) < 3.
```
Use `statistics.NormalDist` for Phi and its inverse.

## fibbot/portfolio.py  (pure)

```python
@dataclass
class PortfolioResult:
    start_equity: float
    final_equity: float              # realized + open positions marked (open trades' ret_pct is marked)
    realized_equity: float
    curve: list[tuple[int, float]]   # (t, realized equity) starting with (first t, start_equity)
    taken: list[dict]                # {"trade": Trade, "notional": float, "pnl": float}
    skipped: list[dict]              # {"trade": Trade, "reason": "max_open"|"max_total_risk"|"max_notional"|"daily_loss_limit"|"no_equity"}

def simulate_portfolio(trades: list[Trade], pcfg: dict, start_equity: float | None = None) -> PortfolioResult
def marked_curve(pf: PortfolioResult, bars_by_symbol, fee_rate, slippage, end_t=None) -> list[tuple[int, float]]
    # mark-to-market equity at every bar close a taken position is open (tp1 partials at their fill
    # time, the remainder at the close with exit slippage and fee); its last point = final_equity.
    # Positions without usable bars contribute their pnl at exit_t only. Used for drawdown / Sharpe.
```
Events in time order: each trade has an entry event at `entry_t` and (if closed) an exit at `exit_t`.
At equal timestamps ENTRIES go first (by symbol, then input order), then exits of earlier
positions, then exits of trades that closed on their own entry bar: an entry at t sees only exits
with exit_t < t (slot, risk, sizing equity and daily-loss PnL), because its order was placed at the
close of the bar before. On entry: skip if realized PnL for that UTC day <= -daily_loss_limit *
equity at the day's first event; skip if open count >= max_open_positions; risk_amount =
risk_per_trade * equity; notional = min(risk_amount / risk_pct, max_notional_frac * equity); no
leverage: notional is shrunk to the room left under max_total_notional (default 1.0) x equity minus
the open notional, or skipped with "max_notional" when no room is left; new_risk = notional *
risk_pct; skip if sum(open new_risk) + new_risk > max_total_risk * equity (+1e-9). On exit: pnl =
notional * ret_pct, equity += pnl, append curve point. `equity` here is realized equity.
`start_equity` defaults to pcfg["starting_equity"]. PortfolioResult.curve stays the realized curve.

## fibbot/store.py  (SQLite, stdlib sqlite3, WAL)

```python
class Store:
    def __init__(self, path: str | Path)       # creates schema if missing; migrates older databases in
                                               # place (columns are only added: trials.n_trades, trials.engine)
    def close(self)
    # experiments: kind in {"backtest","optimize","baseline","validate","robustness","sweep"}
    def add_experiment(self, kind, spec: dict | None, spec_id: str | None, window_start: int | None,
                       window_end: int | None, metrics: dict, source: str = "cli", notes: str = "") -> int
    def list_experiments(self, limit=20, kind=None, sort_by="recent", spec_id=None,
                         exclude_sources: tuple[str, ...] = ()) -> list[dict]
        # sort_by: "recent" | "avg_r" | "profit_factor" (read from metrics JSON; missing -> last)
        # spec_id / exclude_sources filter before the limit (the brain never sees source "cli-holdout")
        # rows: {id, created_t, kind, spec_id, spec, window_start, window_end, metrics, source, notes}
    def get_experiment(self, exp_id) -> dict | None
    def latest_validation(self, spec_id) -> dict | None      # newest kind="validate" row for spec_id
    def count_experiments(self, kind=None, since_t=None) -> int
    def count_experiment_specs(self, kind=None) -> int      # distinct spec_ids (kind="validate": specs ever validated)
    # trials: distinct specs evaluated on any window, plus "<spec_id>#look<k>" keys for repeated
    # distinct holdout looks; Sharpe estimates come only from full-universe research-window runs
    def record_trial(self, spec_id: str, sharpe: float | None, n: int | None = None,
                     engine: int = strategy.ENGINE_VERSION) -> None
        # n = closed-trade count. A None/NaN sharpe only counts the trial. A new estimate replaces the
        # stored one unless both come from the same engine, both counts are known and the new one is
        # smaller (a symbol-subset run never overwrites a wider one; an older engine's always yields).
    def trial_stats(self, min_trades=None) -> dict
        # {"n": all distinct specs (any engine version: they were part of the search), "var_sharpe",
        #  "n_var"}. var_sharpe = population variance of the distinct estimates (equal sharpes count
        # once) of the CURRENT engine version with |sharpe| < metrics.CAP and n_trades >= min_trades
        # (default research.min_trades; unknown counts kept), clipped to median +/- 5 robust sigmas
        # (1.4826*MAD) first; 0 for fewer than 2.
    # holdout looks: one row per distinct draw from a held-out window (spec, window, symbol set)
    def record_look(self, key, spec_id, kind ("validate"|"peek"), window_start, window_end,
                    sharpe=None, n=None, engine=ENGINE_VERSION, now_t=None) -> bool   # False: already recorded
    def holdout_stats(self, window_start=None, window_end=None, min_trades=None) -> dict
        # {"k": distinct looks (any spec, validations and peeks) whose window overlaps
        #  [window_start, window_end), "var_sharpe": variance of their validation Sharpes (as
        #  trial_stats; None for fewer than 2), "n_var"}
    # roster
    def add_to_roster(self, strategy_id, name, spec: dict, paper_start_t: int, validation: dict, rationale: str) -> None
        # status "paper". Raises ValueError for ANY existing strategy_id, retired included:
        # retirement is final, paper trades and the equity curve are never deleted.
    def roster(self, status: str | None = "paper") -> list[dict]  # rows incl. spec (dict), validation (dict)
    def get_strategy(self, strategy_id) -> dict | None
    def set_strategy_status(self, strategy_id, status, reason="") -> None   # status: paper|retired|graduated
    def count_ever_on_paper(self) -> int           # distinct ids = paper runs (each id goes on paper once)
    # paper trades, unique (strategy_id, symbol, leg_id)
    def upsert_paper_trade(self, strategy_id, trade: Trade, notional: float | None = None,
                           pnl: float | None = None) -> str   # returns "new" | "updated" | "unchanged"
    def paper_trades(self, strategy_id=None, status=None, limit=None) -> list[dict]
        # status: "open" | "closed" | None ; newest entry first ; rows include all Trade fields + notional, pnl, status
    # signals, unique (strategy_id, symbol, leg_id)
    def add_signal(self, signal: Signal) -> bool              # False if it already existed
    def signals(self, strategy_id=None, unreviewed=False, since_t=None, limit=None) -> list[dict]
    def set_signal_review(self, strategy_id, symbol, leg_id, decision: str, confidence: float, reason: str) -> None
    def vetoed_legs(self, strategy_id) -> set[tuple[str, str]]   # {(symbol, leg_id)} with decision "skip"
    def count_signal_reviews_since(self, t: int) -> int
    # equity
    def set_equity_curve(self, strategy_id, points: list[tuple[int, float]]) -> None   # replace
    def equity_curve(self, strategy_id) -> list[tuple[int, float]]
    # journal (kind: research|review|signal|system|graduation|promotion|retirement)
    def add_journal(self, kind, title, body) -> int
    def journal(self, limit=20, kind=None) -> list[dict]   # newest first: {id, t, kind, title, body}
    # key/value (JSON values)
    def get_kv(self, key, default=None)
    def set_kv(self, key, value) -> None
    # maintenance
    def file_bytes(self) -> int                       # database size after a WAL checkpoint
    def compact(self, now_t=None, keep_days=90, journal_keep_days=30) -> dict
        # experiments older than keep_days (except validate ones and those a roster row refers to)
        # keep id, times, kind, spec_id, spec, source and notes, but their metrics are cut to the
        # headline numbers (+ "compacted": true); "system" journal bodies older than
        # journal_keep_days become ""; trials, looks, roster, paper trades, equity and signals are
        # untouched (n_trials and var_sharpe do not change); VACUUM.
        # Returns {"bytes_before", "bytes_after", "experiments", "journal"}. The scheduler's report
        # step runs it about once a day (kv last_compact_t).
```
Timestamps written by the store (`created_t`, journal `t`) use `int(time.time())` unless a `now_t`
kwarg is given (all add_* methods accept optional `now_t` for tests).

## fibbot/strategy.py  — the heart

### StrategySpec

```python
@dataclass
class StrategySpec:
    name: str = "custom"                 # label, NOT part of spec_id
    timeframe: str = "4h"                # "1h" | "4h" | "1d"
    direction: str = "long"              # "long" | "short" | "both"
    swing_mode: str = "atr"              # "atr" | "pct"
    swing_atr_len: int = 14              # 5..50
    swing_mult: float = 3.0              # 1.0..10.0 (atr mode)
    swing_pct: float = 0.05              # 0.01..0.30 (pct mode)
    min_leg_atr: float = 4.0             # 0..30  leg (extreme-anchor) must be >= this many ATR
    setup_expiry_bars: int = 40          # 3..500 leg dies if its extreme is older than this and unfilled
    trend_filter: str = "ema"            # "none" | "ema" | "ema_slope" | "ema_stack"
    trend_ema: int = 200                 # 20..400
    adx_min: float = 0.0                 # 0..50 (0 disables)
    require_structure: bool = False      # long: anchor low > previous confirmed low ; short: anchor high < previous high
    entry_zone_lo: float = 0.618         # shallow edge of the zone (retracement ratio), 0.236..0.886
    entry_zone_hi: float = 0.65          # deep edge, entry_zone_lo <= hi <= 0.886
    entry_mode: str = "limit"            # "limit" | "confirm"
    entry_at: str = "mid"                # limit mode: "lo" | "mid" | "hi" of the zone
    confirm_type: str = "reclaim"        # confirm mode: "reclaim" | "engulf" | "rsi"
    confirm_window: int = 6              # 1..50 bars after the zone touch to see a confirmation
    stop_ratio: float = 0.886            # retracement ratio of the stop, > entry_zone_hi, <= 1.2
    stop_atr_buffer: float = 0.25        # 0..3 ATR beyond the stop level
    tp1_ext: float = 1.0                 # extension ratio of target 1 (1.0 = the leg extreme), 0.3..3.0
    tp1_frac: float = 0.5                # 0.05..1.0 fraction closed at tp1
    tp2_ext: float = 1.618               # > tp1_ext, <= 4.236 (ignored if tp1_frac == 1)
    move_stop_to_be: bool = True         # after tp1, stop -> breakeven incl. fees (active from next bar)
    trail_atr: float = 0.0               # 0..10; 0 disables; else trail remainder after tp1 at close - k*ATR
    max_hold_bars: int = 120             # 1..2000 time stop
    min_rr: float = 1.5                  # 0..10 planned (blended) reward/risk at arming, before costs
    min_stop_cost_mult: float = 0.0      # 0..20 (step 0.5): stop at least this many round-trip costs away

    def validate(self) -> "StrategySpec"   # raises ValueError("field: reason") ; returns self
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "StrategySpec"   # unknown keys -> ValueError ; missing -> defaults
    def spec_id(self) -> str               # 10 hex chars: sha1 of canonical JSON of to_dict() minus "name"
        # Fields added after the first release (_ID_OPTIONAL: currently min_stop_cost_mult) are left
        # out of the blob while they hold their default, so every earlier spec_id (all presets) is unchanged.
    def with_overrides(self, **kw) -> "StrategySpec"
```

Construction normalizes numbers so equal specs share a spec_id: ints in float fields become floats,
floats are rounded to 10 decimals, integral floats in int fields become ints. `from_dict` and
`with_overrides` only parse (unknown keys -> ValueError); `validate()` (also run by
`FibEngine.__init__`) checks types, ranges and the cross-field rules.

`PARAM_SPACE: dict[str, dict]` — for every field except name: `{"type": "int"|"float"|"bool"|"choice",
"min", "max", "step" (numeric), "choices" (choice), "default", "doc"}`. min/max are the field-comment
ranges (stop_ratio 0.25..1.2, tp2_ext 0.3..4.236); cross-field rules are only checked by `validate()`.
`step` is used for robustness neighbours. `PRESETS: dict[str, StrategySpec]` (shared instances: derive
with `with_overrides`, never mutate) — at least the rows below; parameters a row does not mention keep
their defaults, and a stop given without "+xATR" means `stop_atr_buffer=0`:

| preset | idea |
|---|---|
| `golden_pocket_4h` | 4h long, EMA200 trend, zone 0.618-0.65 limit mid, stop 0.886+0.25ATR, tp1 1.0 (50%), tp2 1.618 |
| `golden_pocket_1h` | same on 1h |
| `golden_pocket_both_4h` | 4h both directions |
| `ote_confirm_1h` | 1h long, zone 0.618-0.786 confirm reclaim, stop 1.0+0.1ATR, tp1 1.0, tp2 1.618 |
| `shallow_trend_4h` | 4h long, ema_stack + adx>=20, zone 0.382-0.5 limit lo, stop 0.786, tp1 1.0, tp2 1.618 |
| `deep_786_4h` | 4h long, no trend filter, zone 0.786-0.886 limit lo, stop 1.0+0.3ATR, tp1 0.618, tp2 1.0 |
| `golden_pocket_1d` | 1d long, zone 0.5-0.65 limit mid, stop 0.886, tp1 1.0, tp2 1.618 |
| `ext_runner_4h` | 4h long, zone 0.618-0.65, tp1 1.0 (30%), trail 2.5 ATR, tp2 2.618 |

`warmup_bars(spec) -> int` = max(3 * trend_ema, 5 * swing_atr_len, 4 * 14, 300).

`ENGINE_VERSION = 2`: a version of the fill / accounting rules (2: entry-bar target credit at the
close, the limit trade-through, planned risk on stop-through fills, min_stop_cost_mult). Results of
different versions are not comparable for the same spec_id: experiments record it, trial variance
pools only the current version, and promotion refuses a validation of another version.
Helpers: `round_trip_cost(costs, entry_mode)` = 2*fee_rate + slippage (limit) or 2*fee_rate +
2*slippage (confirm); `limit_fills(side, px, h, l, through)`: long iff l <= px*(1-through), short
iff h >= px*(1+through).

### FibEngine

```python
@dataclass
class RunResult:
    trades: list[Trade]          # CLOSED trades, entry_t >= trade_start_t, in entry order
    open_trade: Trade | None     # position still open at the last bar (exit_t None, marked at last close)
    signals: list[Signal]        # one per (leg_id) armed at a bar with t >= trade_start_t - TF_SECONDS[tf]
    n_bars: int

class FibEngine:
    def __init__(self, spec: StrategySpec, costs: Costs | None = None, strategy_id: str | None = None)
        # strategy_id defaults to spec.spec_id()
    def run(self, bars: Bars, trade_start_t: int | None = None,
            skip_legs: frozenset[str] | set[str] = frozenset()) -> RunResult
        # ValueError if bars.tf != spec.timeframe; empty bars -> empty RunResult. skip_legs holds
        # leg_ids of THIS symbol (None = empty). No warmup gating inside: only nan indicators and
        # trade_start_t gate trading (callers prepend warmup_bars).
```

Indicators: ATR is Wilder `atr(h, l, c, swing_atr_len)` everywhere (atr-mode pivots, min_leg_atr,
stop buffer, trailing, meta "atr"); ADX and RSI use n=14.

**Unknown intrabar order is resolved against the trade.** OHLC does not say whether a bar's low came
before or after its high, so an ambiguous bar never creates an entry: the bar that sets a new extreme
still kills the leg if its low reaches the new stop, but is never a confirm-mode touch.

**Determinism and causality (hard requirements, tested):** decisions made at the close of bar i use
only bars <= i; orders placed at the close of bar i can only fill on bars >= i+1. Consequence:
for every k, `run(bars.slice(0, k))` yields exactly the trades of `run(bars)` that closed at or
before bar k-1 (same prices), and the same signals with t <= t[k-1]. Appending bars never changes
earlier output.

**Per-bar processing order for bar i:**
1. *Fills on bar i* from orders placed at the close of bar i-1 (only if t[i] >= trade_start_t):
   - open position (entered before i): manage exits (below);
   - pending limit order: long fills if `l[i] <= px*(1 - limit_through)` at `min(o[i], px)` (short:
     `h[i] >= px*(1 + limit_through)` at `max(o[i], px)`), no slippage; a bare touch does not fill
     (limit_fills). If both a long and a short limit trade through, fill the one whose order price is
     closer to o[i] and cancel the other (an exact tie fills the long; since long px < c[i-1] <
     short px, this is the order the open gapped through, else the nearer one). Targets still fill
     on a touch;
   - pending market order (confirm mode): fills at `o[i] * (1 + side*slippage)`;
   - on the entry bar, after the fill: let `x` = the worse of o[i] and the fill price (long: min,
     short: max), i.e. the fill's price before entry slippage. If `x` is at or beyond the stop, exit
     immediately at `x * (1 - side*slippage)` (reason "stop"); else if the bar's range reaches the
     stop (`l[i] <= stop` long / `h[i] >= stop` short) exit at the stop with slippage (reason "stop").
     Intrabar, the entry bar checks only the stop (its high/low may precede the fill). Mark the leg used.
2. *Close of bar i*: on the entry bar, if the position is still open and c[i] is strictly beyond
   tp1 (long c > tp1), tp1_frac fills at tp1 with no slippage (tp1_frac == 1 closes the trade,
   reason "target"), tp1_hit is set and breakeven applies from the next bar; then, if c is also
   beyond tp2, the rest closes at tp2 (reason "target"). A target the entry bar's high reached while
   it closed back below it is not credited. Then update the pivot tracker; update both side setups;
   trailing stop (also at the entry bar's close) and time stop for an open position; then (if flat)
   place orders for bar i+1.

**Setup definition (long; short mirrors every comparison):**
- anchor = the most recent confirmed LOW pivot (confirm_idx <= i). leg_id = `f"L{anchor.t}"`
  (short: most recent confirmed HIGH pivot, leg_id `f"S{anchor.t}"`).
- extreme = highest high over bars (anchor.idx, i] and its index ext_idx (first occurrence). It is
  tracked incrementally and resets when the anchor changes.
- The leg is skipped/dead when: leg_id in used legs or skip_legs; price traded below the anchor
  after the anchor (`min low over (anchor.idx, i] < anchor.price`); or `i - ext_idx > setup_expiry_bars`
  while unfilled; or the low of bar i is <= the leg's stop price while unfilled (price ran through the
  stop; checked on every bar the leg is evaluated, including the bar that confirms the anchor or sets a
  new extreme, with the stop recomputed from that extreme). A dead leg is added to used legs.
- It *qualifies* at bar i when: `extreme - anchor.price >= min_leg_atr * atr[i]`; the trend filter
  passes **evaluated at ext_idx** (long: `ema`: c > EMA(trend_ema); `ema_slope`: c > EMA and
  EMA[ext_idx] > EMA[ext_idx-10]; `ema_stack`: EMA(50) > EMA(trend_ema) and c > EMA(50); `none`: pass;
  nan fails); `adx[ext_idx] >= adx_min` (when adx_min > 0); structure filter if enabled (fails when
  there is no previous confirmed pivot of that kind); and the reward/risk check passes.
- Levels: `retrace(r) = retrace_price(anchor, extreme, r)`, `ext(e) = extension_price(anchor, extreme, e)`.
  zone_top = retrace(entry_zone_lo) (shallow), zone_bottom = retrace(entry_zone_hi) (deep).
  entry ratio = lo | (lo+hi)/2 | hi per entry_at; entry_px = retrace(entry ratio).
  stop_px = retrace(stop_ratio) - stop_atr_buffer * atr[i]  (short: + buffer).
  tp1 = ext(tp1_ext), tp2 = ext(tp2_ext).
  RR = (tp1_frac*(tp1-entry) + (1-tp1_frac)*(tp2-entry)) / (entry-stop) (short: signs flipped);
  requires entry > stop, tp1 > entry and RR >= min_rr (gross of costs; if tp1_frac == 1, RR uses tp1
  only), and `side*(entry - stop)/entry >= min_stop_cost_mult * round_trip_cost(costs, entry_mode)`.
  Both are checked at the nominal entry at every evaluation (Signal, limit order, confirm touch) and
  at c[k] for the confirm market order.
- The first bar at which a leg qualifies emits a `Signal` (entry_px, stop_px, tp1, tp2, meta with
  anchor, extreme, atr, rr, mode, plus anchor_t, ext_t, leg_size and zone = [zone_top, zone_bottom]),
  whether or not an order is placed (flat or in a position, c[i] on either side of entry_px);
  entry_px is the nominal entry in both modes. Dead legs and legs in skip_legs never signal.
  `Trade.setup_t` is that bar's t, also when the signal window hid the Signal.
- **limit mode**: when flat and the leg qualifies, place a limit at entry_px for bar i+1 only if
  `c[i] > entry_px` (short: `c[i] < entry_px`). Orders are re-computed every bar (the extreme may move;
  the Signal keeps its arming-time prices).
- **confirm mode**: the leg is *touched* at bar j when `j > ext_idx` (see intrabar order above), the
  leg qualifies at j, `l[j] <= zone_top` and `l[j] > stop_px`. A new extreme clears the touch (the
  zone moved). Within `confirm_window` bars after the touch (j..j+window, k == j allowed), a
  confirmation at the close of bar k places a market order for bar k+1 if the leg still qualifies
  with entry = c[k]:
  `reclaim`: c[k] > zone_top and c[k] > o[k]; `engulf`: c[k] > o[k], c[k-1] < o[k-1], c[k] >= o[k-1],
  o[k] <= c[k-1]; `rsi`: rsi[k-1] < 40 and rsi[k] > rsi[k-1]. No confirmation in the window ->
  untouched again (can be re-touched later). At each close: an expired touch (i - j > window) is
  cleared first, then bar i may register a new touch, then the confirmation is checked.
- Stop/targets/anchor/extreme are frozen into the position at fill time.

**Position management on bars after the entry bar** (long; short mirrors):
1. `o <= stop`: exit the remainder at `o*(1-slippage)`.
2. elif `l <= stop`: exit the remainder at `stop*(1-slippage)` (stop before targets - worst case).
3. elif not tp1_hit and `h >= tp1`: fill tp1_frac at `max(o, tp1)` (limit, no slippage; the open
   fill happens only on a true gap after a previous close on the near side of tp: a target that a
   previous close had passed was already filled at its level on that bar); tp1_frac==1
   closes the trade (reason "target"). If move_stop_to_be: stop = max(stop, entry*(1+2*fee_rate)) from
   the NEXT bar. Then if `h >= tp2` close the remainder at `max(o, tp2)` (reason "target").
4. elif tp1_hit and `h >= tp2`: close the remainder at `max(o, tp2)` (reason "target").
5. At the close: if trail_atr > 0 and tp1_hit: stop = max(stop, c - trail_atr*atr). If
   `i - entry_idx >= max_hold_bars`: exit the remainder at `c*(1-slippage)` (reason "time").
Exit reason when a stop fills: "stop" if the stop was never moved, "breakeven" if it was moved to
breakeven and not trailed above it, "trail" if the trailing stop raised it (above breakeven, or above
the initial stop when move_stop_to_be is off). A nan ATR does not trail.

**Accounting.** Fills k with fractions f_k (sum 1) at prices p_k; entry fill P (incl. slippage), fee
rate f:
- long: `ret_pct = sum(f_k*p_k)/P*(1-f) - (1+f)`; short: `ret_pct = (1-f) - sum(f_k*p_k)/P*(1+f)`.
- `risk_pct = side*(P - initial_stop)/P`; if the entry's pre-slippage fill price x (the worse of o
  and the fill) was at or beyond the stop (the immediate stop-out rule fired), `risk_pct =
  side*(order_px - stop)/order_px`, the planned risk (order_px = the limit price, or c[k] for a
  confirm market order). `r_multiple = ret_pct / risk_pct` (0 if risk_pct == 0);
  exit_price = sum(f_k*p_k); `Trade.stop_price` is the initial stop.
- `bars_held = exit_idx - entry_idx`; exit_t = t[exit_idx].
- meta: {"anchor", "extreme", "leg_size", "atr", "entry_ratio", "tp1", "tp2", "rr",
  "fills": [[frac, price, reason, t], ...], "entry_mode"}. tp1/tp2 are the frozen targets; atr/rr are
  from the bar the order was placed; confirm-mode entry_ratio is the retracement ratio of c[k]. Fill
  reasons: "tp1", "tp2", "stop" | "breakeven" | "trail", "time", "open"; exit_reason "target" when tp1
  (tp1_frac == 1) or tp2 closes the trade.
- An open position at the last bar is reported as `open_trade` marked at c[-1] (with exit costs and
  slippage), exit_reason "open", exit_t None: its fills are the realized partials plus the mark
  `[remainder, c[-1]*(1 - side*slippage), "open", t[-1]]`, and bars_held = (n-1) - entry_idx.

```python
def manage_position(bars: Bars, atr: list[float], entry_idx: int, side: int, entry_px: float,
                    stop_px: float, tp1: float, tp2: float, spec: StrategySpec, costs: Costs,
                    start_mgmt_idx: int | None = None, plan_px: float | None = None) -> dict
    # Pure helper implementing "position management" + accounting from entry_idx onward (used by the
    # random baseline). Returns {"exit_idx", "exit_reason", "fills", "exit_price", "ret_pct", "risk_pct",
    # "r_multiple", "open": bool}. FibEngine must use the same per-bar exit logic (shared code, not a copy).
    # atr = ATR(swing_atr_len) aligned with bars (only trailing reads it). start_mgmt_idx (default
    # entry_idx) is the first bar processed: bar entry_idx gets the entry-bar stop check (x = the
    # worse of o and entry_px) and, at its close, the targets its close is beyond; a later start
    # skips the entry bar and every bar before it, and the time stop still counts from entry_idx.
    # plan_px: the order price (risk basis for a fill at or beyond the stop; without it |P-stop|/P).
    # ValueError: side not +/-1, entry_idx outside bars, start_mgmt_idx < entry_idx.
```

## fibbot/backtest.py

```python
def run_strategy(spec, bars_by_symbol: dict[str, Bars], start_t: int | None, end_t: int | None,
                 costs: Costs, workers: int = 1, skip_legs: dict[str, set] | None = None,
                 strategy_id: str | None = None) -> dict[str, RunResult]
    # per symbol: FibEngine.run(bars.window(start_t, end_t, warmup_bars(spec)), trade_start_t=start_t)
    # workers > 1 -> concurrent.futures.ProcessPoolExecutor (module-level worker fn for pickling).

@dataclass
class BacktestResult:
    spec: dict; spec_id: str; start_t: int | None; end_t: int | None; symbols: list[str]
    trades: list[Trade]            # closed + open-marked, sorted by entry_t
    portfolio: PortfolioResult
    metrics: dict                  # trade_stats(...) | equity_stats(equity_curve) | {"n_open": k}
    per_symbol: dict[str, dict]    # {sym: {"n", "total_r", "avg_r", "win_rate", "profit_factor"}}
    per_year: dict[str, dict]      # {"2023": {"n", "total_r", "avg_r"}} by entry year
    equity_curve: list             # what equity_stats saw: the MARKED curve when bars are known
    equity_basis: str              # "marked" | "realized" (also in summary()["portfolio"])
    by_side: dict | None           # direction "both" only: {"long", "short"} lite stats of the one combined run
    def summary(self, max_symbols: int = 20) -> dict   # compact JSON-safe dict (no trade lists; by_side when set)

def result_from_trades(spec, trades, symbols, start_t, end_t, cfg, period=None, bars_by_symbol=None) -> BacktestResult
    # with bars_by_symbol: max_drawdown / sharpe / sortino / calmar on portfolio.marked_curve
def costs_from_cfg(cfg) -> Costs   # every Costs field present in cfg["costs"] (fee_rate, slippage, limit_through)
def backtest(spec, bars_by_symbol, start_t, end_t, cfg, workers=None) -> BacktestResult
    # costs from cfg["costs"], portfolio from cfg["portfolio"], workers from cfg["research"]["workers"];
    # always passes its bars, so the validation gate's max_drawdown is mark-to-market
```

## fibbot/research.py

```python
def compute_windows(cfg, bars_by_symbol, validation_end=None) -> dict
    # {"data_start", "data_end", "research_start", "research_end", "validation_start", "validation_end"}
    # data_end = min over symbols of (last_t + tf). Sliding (validation_end None): validation_start =
    # data_end - validation_months*30.44 days (floored to a UTC day), validation = [vs, data_end).
    # Pinned (validation_end given, e.g. Lab's epoch): validation_start = floor_day(validation_end -
    # validation_months), validation = [vs, min(validation_end, data_end)), "pinned": True; bars after
    # it are forward / paper data only. research = [min first t, validation_start) either way.
def fold_bounds(start_t, end_t, folds) -> list[tuple[int, int]]
def fold_stats(trades, bounds) -> list[dict]       # trade_stats per fold by entry_t (n, avg_r, total_r, profit_factor)
def evaluate(spec, bars_by_symbol, cfg, windows, window="research") -> tuple[BacktestResult, dict]
    # window: "research" | "validation" | "all". Returns (result, extras) with
    # extras = {"folds": [...], "fold_positive_frac": x, "min_fold_avg_r": y}
def random_baseline(spec, bars_by_symbol, cfg, start_t, end_t, n_runs=10, seed=1, result=None, regime=True) -> dict
    # For each real trade of the strategy, draw a random entry bar in the same symbol & window (same
    # side), enter at the open with the strategy's own entry cost (+side*slippage for confirm/market
    # entries, none for limit entries), stop at the same risk_pct (the planned risk for stop-through
    # fills), tp1/tp2 at the same R distances, same management via strategy.manage_position.
    # Keys: n_trades, n_runs, seed, strategy_avg_r, baseline_avg_r_mean, baseline_avg_r_std,
    # beats_frac, runs, edge_r (strategy - baseline mean), strategy_se, edge_se, edge_z; with
    # regime=True a second set of draws uses only bars whose previous close passed the spec's
    # trend / ADX filter: regime_filtered, baseline_regime_avg_r_mean/std, edge_regime_r,
    # edge_regime_z (they repeat the uniform numbers for an unfiltered spec). Use min(edge_r,
    # edge_regime_r): it measures entry timing (setup plus filter) against random timing.
def edge_stats(trades, base_mean, base_sd, n_runs) -> dict
    # edge_r = mean R - base_mean; edge_se = sqrt(max(strategy_se, base_sd)^2 + base_sd^2/n_runs) with
    # strategy_se cluster-robust by UTC entry week; edge_z = edge_r / edge_se
def trades_key(result) -> str   # digest of (symbol, side, entry_t, exit_t, R) over all trades
def walk_forward_optimize(base_spec, grid: dict[str, list], bars_by_symbol, cfg, windows,
                          objective="avg_r", on_combo=None) -> dict
    # combos = product(grid) (<= research.max_grid_size else ValueError). Each combo evaluated once on the
    # research window; trades bucketed into folds. Anchored WF: for fold k = 1..F-1 pick the combo with the
    # best objective on folds [0, k) among combos with >= max(5, min_trades // folds) trades there; record
    # its fold-k trades as OOS. Returns {"n_combos", "n_duplicates", "oos": trade_stats(OOS trades),
    # "oos_by_fold": [...], "chosen": [{"fold", "params", "is_avg_r"}], "best": {"params", "spec",
    # "spec_id", ...stats}, "top": [5 best rows {params, n, avg_r, profit_factor, fold_positive_frac}],
    # "sensitivity": {param: {value: mean avg_r}}}. on_combo(spec, result) is called only for combos
    # whose trades differ from every earlier combo's: every distinct outcome is a trial.
    # The folds are part of the research window the researcher also sees, so OOS measures tuning
    # stability, not independent evidence.
def neighbours(spec) -> list[StrategySpec]        # each numeric/choice param moved one step (both ways), valid only
def robustness(spec, bars_by_symbol, cfg, windows, max_neighbours=48, on_combo=None) -> dict
    # the whole one-step neighbourhood by default (under a cap whole parameters, both directions,
    # spread over the list). The base is evaluated too; a neighbour whose trades equal the base's or
    # an earlier neighbour's gets duplicate_of ("base" | spec_id), is not a trial (on_combo not
    # called) and is left out of positive_frac / median / worst. {"n" (evaluated), "n_distinct",
    # "n_duplicates", "n_candidates", "positive_frac", "median_avg_r", "worst_avg_r", "coverage"
    # {param: [values]}, "params_untested", "rows": [...]} on the research window
def validation_gate(result, gate: dict, n_trials: int, var_sr: float | None, baseline=None,
                    holdout_looks=None, var_holdout_sr=None) -> dict
    # {"passed", "checks": [{"name", "value", "threshold", "passed"}], "deflated_sharpe", "dsr_basis",
    #  "dsr_trials", "dsr_var", "sr_star", "n_eff", "t_clustered"[, "edge"]}
    # checks: min_trades, min_profit_factor, min_expectancy_r, min_deflated_sharpe, max_drawdown
    # (portfolio equity, mark-to-market), min_symbol_win_frac; with a baseline (random_baseline on the
    # same window and trades) also min_validation_edge_r (default 0.0) and min_validation_edge_z
    # (default 1.65; the binding of the uniform and regime edges); a threshold of None drops a check.
    # Deflated Sharpe with n_eff = effective_n(R, entry weeks): dsr_basis "research" (holdout_looks
    # None) against n_trials / var_sr; "holdout" against n_trials = holdout_looks and var =
    # max(1/n_eff, var_holdout_sr). The trial count is never below MIN_GATE_TRIALS = 2.
```

## fibbot/paper.py — forward paper trading by deterministic replay

Paper trading re-runs the *same* FibEngine every tick on each symbol from a fixed anchor
(`paper_start_t - warmup_bars(spec) * TF_SECONDS[tf]`) to the newest closed bar with
`trade_start_t = paper_start_t`. While code, settings and data stay the same, new bars only add or
close trades and paper equals a backtest of the same window. The actual rules:
- A replayed trade not on record that entered at or before the newest bar an earlier tick
  replayed for its symbol is refused as a history rewrite; so is a back-dated signal and a new
  trade on a symbol whose recorded open trade the replay lost (per-symbol high-water marks in kv
  `paper_guard:{sid}`). Bars that arrive late after a per-symbol outage still add their trades,
  which keeps paper == backtest. A closed trade's prices and R are final; its dollar notional/pnl
  and the equity curve can be recomputed when late bars add earlier trades ("resized").
- Account settings (costs: every Costs field; portfolio rules; starting equity) are frozen in kv
  `paper_account:{sid}` (at promotion by Lab.promote, else at the first tick);
  `account_settings(store, cfg, row)` reads them back (config keys added later take their live
  values). Editing costs.*, portfolio.* or paper.starting_equity only affects strategies promoted
  afterwards; account() and graduation use the frozen starting equity.
- An engine code change (`engine_fingerprint()`: code_digest of strategy/fib/indicators/portfolio/types
  without comments, docstrings or layout) is reported once per strategy in errors, journaled and
  recorded in engine_changes; the record so far is kept, later trades follow the new code, and the
  user should re-validate. A Python upgrade that changes ast.dump output also counts as a change.
- Stranded open trades: if the replay stops reproducing a recorded open trade (different entry bar
  or side, gone, or closed on an already-replayed bar) for VOID_AFTER_TICKS=3 consecutive ticks, it
  is closed at the latest close with exit_reason "void". If its symbol's bars lag the strategy's
  newest bar by more than max(DATA_GAP_DAYS=3 days, 3 bars) for 3 ticks (or the symbol cannot be
  read and its mark is that old), it is closed at its last close with "data_gap". Both pay the
  engine's exit slippage and fees, are new exits and stay in the R statistics.
- The stored equity curve is marked to market (portfolio.marked_curve over the replayed bars), so
  account_drawdown matches backtest()'s drawdown.

```python
class PaperTrader:
    def __init__(self, cfg, store: Store, candles: CandleStore)
    def tick(self, update_data: bool = True, now_t: int | None = None, universe: list[str] | None = None) -> dict
        # universe defaults to get_universe(cfg) (the scheduler passes the list it already has).
        # For each roster strategy (status "paper"): optionally candles.update_all(universe, {tf});
        # run engine per symbol (skip_legs = store.vetoed_legs(...) only when cfg.brain.signal_review
        # == "active"); upsert trades under the rules above; add new signals; simulate_portfolio with the
        # frozen settings; write notional/pnl and the marked equity curve.
        # Returns {"strategies": {sid: {"new_signals": [Signal dict], "new_entries": [...], "new_exits": [...],
        #          "equity": float, "open": int, "refused": [Trade dict], "resized": [{symbol, leg_id,
        #          notional_before, notional, pnl_before, pnl}], "engine_changed": bool}},
        #          "new_signals": [Signal...], "errors": [...]}
        # Refusals, forced exits, re-sized trades and engine changes are journaled (kind "system"),
        # one entry per strategy per tick when something happened.
    def account(self, strategy_id) -> dict
        # {"strategy_id", "name", "timeframe", "paper_start_t", "days", "equity", "return", "n_closed", "n_open",
        #  "stats": trade_stats(closed), "max_drawdown", "open_positions": [...]}
def force_close(tr, t, reason, costs, step, close_px=None) -> Trade
def account_settings(store, cfg, row) -> dict
def engine_fingerprint() -> str
def code_digest(source) -> str
```
The universe for a strategy is `get_universe(cfg)` at tick time (cached list), so the paper universe
can change when the market-cap ranking changes; trades already recorded are kept. A symbol that
joins after the strategy's first tick trades only from the bar forming when it joined (no hindsight
back-fill; join times in kv `paper_universe:{sid}`); a symbol that leaves is still replayed while
the account holds a position in it (no new trades or signals); in "active" review mode a veto only
blocks legs without a recorded trade.

## fibbot/graduation.py

```python
def evaluate(store: Store, cfg, strategy_id: str, now_t=None, bars_by_symbol=None, cache=None) -> dict
    # {"strategy_id", "status", "checks": [{"name","value","threshold","passed"[, "note"]}], "summary",
    #  "mean_r_ci": [lo, hi], "evidence": {"n_eff", "design_effect", "weeks", "t_clustered"}}
def paper_edge(cfg, row, closed, bars_by_symbol, end_t, cache=None) -> dict | None
```
Inputs: closed paper trades, equity curve, roster row (validation metrics, paper_start_t), and the
strategy's candles (Lab.graduation always passes them). Trades are clustered by UTC entry week.
Checks (thresholds from cfg["graduation"]; every max_* check passes on value <= threshold, every
min_* on value >= threshold):
- `min_days`: days on paper up to min(now, the newest bar end of the strategy's candles), so days
  without candles (an outage) do not count (wall clock without candles; a note shows both);
- `min_trades`; `min_profit_factor` (R-based PF);
- `mean_r_ci_lower`: the lower end of the anytime-valid ci_level confidence sequence of mean R
  (metrics.confidence_sequence, trades clustered by week, target 200 trades) > 0;
- `max_drawdown` of the account (frozen starting equity, marked curve);
- `min_positive_month_frac` (months with >= 1 closed trade);
- `min_deflated_sharpe`: deflated_sharpe(r, n_trials=store.count_ever_on_paper(), n_eff=n_eff);
- with candles: `min_edge_r` (0.0) and `min_edge_z` (1.65): paper mean R against random entries over
  the paper period (paper_edge = random_baseline with 50 runs, uniform and regime draws, the binding
  one); null drops a check;
- `max_shortfall_vs_validation_r`: value = validation mean R - paper mean R, passes when <= the
  threshold (0.25); the note shows both means.
Status (first match): "graduated" (all pass) | "failing" (DD > 1.5x the limit; OR n >= min_trades/2,
PF < 1 and CI upper < 0; OR, once mature (days >= min_days and n >= min_trades), CI upper < validation
mean R - max_shortfall; OR days >= 3 x min_days with mean R <= 0; the summary ends with "; failing:
<reasons>") | "collecting" (not mature) | "lagging" (mature with mean R <= 0, not yet failing) |
"promising" (mature with mean R > 0 that has not passed every check). The scheduler re-checks it on
every pass: because the interval is anytime-valid this does not inflate false graduations (about
2.5% or less of zero-edge strategies ever graduate, versus about 10% with a fixed CI re-tested
hourly); the cost is time (a +0.3R edge at 0.66 trades/day graduates within a year about 55-65% of
the time). Graduation never enables live trading; it is a report.

## fibbot/lab.py — facade used by CLI, scheduler and brain tools

```python
class BudgetExceeded(RuntimeError): ...

class Lab:
    def __init__(self, cfg, store: Store | None = None, candles: CandleStore | None = None)
        # store defaults to Store(home/fibbot.sqlite); candles to CandleStore(cfg)
    cfg; store; candles
    def universe(self) -> list[str]               # also remembers every symbol seen (kv universe_seen)
    def bars(self, tf: str, symbols: list[str] | None = None) -> dict[str, Bars]   # cached per tf, from disk
    def windows(self, tf: str = "4h") -> dict     # compute_windows, pinned to the validation epoch once set
    def validation_bars(self, tf) -> (bars, windows, info)
        # research.validation_universe "point_in_time" (default): the universe.top_n of a pool (today's
        # universe, kv universe_seen, universe.static_fallback, cached candle files; excluded bases
        # left out) by 90-day quote volume before validation_start, with universe.min_history_days of
        # history before it; "current" = today's universe. info {mode, ranked_at, symbols, dropped, added}.
        # Windows still come from today's universe bars.
    def validation_epoch(self) -> dict | None     # kv validation_epoch {validation_end, t, reason}
    def roll_validation_epoch(self, now_t=None) -> dict
        # explicit, journaled operator action (CLI `epoch --roll`): validation_end += validation_months
        # (never past today); ValueError when no epoch is pinned
    # budgets (per brain cycle; CLI runs are unbudgeted unless begin_cycle was called)
    def begin_cycle(self, kind: str, now_t=None) -> None   # counters from cfg.research; validations_left also
        # capped by weekly_validations_left() (BudgetExceeded then names research.max_validations_per_week)
    def end_cycle(self) -> None                   # back to unbudgeted "cli" mode (cycles call it in finally)
    def budget(self) -> dict                      # {"backtest_units_left", "validations_left"} (None = unlimited)
    def weekly_validations_left(self, now_t=None) -> int | None
        # research.max_validations_per_week (default 3, null disables) minus validate experiments of any
        # source in the last 7 days; brain cycles and the autopilot obey it, CLI validations count but
        # are not capped
    # research (records experiments with "engine_version"; source = "brain" inside a cycle else "cli")
    def backtest(self, spec, window="research", symbols=None, trades=0) -> dict   # summary + folds + experiment_id
        # research window: only a full-universe run (symbols None or equal to the universe) records its
        # Sharpe (n = closed trades); a symbol subset counts as a trial with no Sharpe. Window
        # "validation" (point-in-time symbols unless given) or "all" is a holdout peek: a trial and a
        # look at the spec's holdout (store.record_look, "#look<k>" trial keys), recorded with source
        # "cli-holdout" and metrics {window, holdout_looks, holdout: {result: <metrics>}} (hidden from
        # the brain); the result gains "holdout_looks"; kv holdout_peeks counts them.
    def optimize(self, base_spec, grid) -> dict    # every distinct outcome is a trial
    def baseline(self, spec, n_runs=10) -> dict
    def robustness(self, spec) -> dict             # the whole neighbourhood (ROBUSTNESS_NEIGHBOURS = 48)
    def validate(self, spec) -> dict
        # validation window (validation_bars) + gate; consumes a validation. The gate gets the holdout
        # basis (holdout_looks = store.holdout_stats(window)["k"] incl. this look, var_holdout_sr), the
        # validation baseline (random_baseline, VALIDATION_BASELINE_RUNS = 50 runs) and n_eff.
        # ValueError when the spec's latest validation of this engine version has a window
        # overlapping the current one (one look per validation period); the identical window and
        # symbol set is a replay and allowed. Output / metrics gain baseline {n_runs,
        # baseline_avg_r_mean, baseline_avg_r_std, beats_frac, edge_r, edge_z, regime_filtered,
        # baseline_regime_avg_r_mean, edge_regime_r, edge_regime_z}, validation_universe {mode,
        # ranked_at, dropped, added}, symbols, holdout_looks, replay, n_trials (research trials),
        # var_sr (the gate's variance), dsr_basis, dsr_trials, holdout_k, sr_star, n_eff,
        # t_clustered, var_sharpe_research, n_var, engine (paper.engine_fingerprint()).
    def promote(self, spec, name: str, rationale: str, now_t=None) -> dict
        # ValueError when: the spec_id is on the roster with any status (retirement is final: a retired
        # spec can never come back; promote a variant); its latest validation did not pass or was
        # recorded by another ENGINE_VERSION; any validation of it in the last validation_months
        # failed (holdout ledger); it has no research_baseline with min(edge_r, edge_regime_r) > 0;
        # the roster holds paper.max_roster. paper_start_t = start of the next bar after now. On
        # success the account settings are frozen (kv paper_account:{sid}) and the first promotion
        # pins the validation epoch at floor_day(min(paper_start_t, data_end)).
    def research_baseline(self, spec_id) -> dict | None
    def retire(self, strategy_id, reason) -> dict  # final; the paper record is kept
    def prior_validation(self, spec) -> dict | None   # a validation of this engine overlapping the current window
    def holdout_looks(self, spec_id) -> list[dict]
    # paper / reporting
    def paper(self) -> PaperTrader
    def strategy_bars(self, row) -> dict[str, Bars]
    def graduation(self, strategy_id, now_t=None) -> dict   # graduation.evaluate with strategy_bars
    def status(self, now_t=None) -> dict
        # roster (account; "validation" {experiment_id, window, n, avg_r, profit_factor,
        # deflated_sharpe}; graduation {status, summary, failed, mean_r_ci, evidence, checks
        # [[name, value, threshold, passed]]}), budgets, windows (with "pinned"), data per tf (incl.
        # oldest_end, oldest_symbol, oldest_lag_bars), trial_stats(min_trades), validations_7d {used,
        # cap}, holdout_peeks, validation_epoch, data_stall, system_events_7d {n, last, last_t}
    def system_events(self, days=7, now_t=None) -> dict
    def signal_review_stats(self) -> dict
        # brain shadow-review value: closed paper trades split by the brain's decision on their leg:
        # {"reviewed": n, "take": trade_stats-lite, "skip": trade_stats-lite, "unreviewed": ..., "skip_minus_take_avg_r"}
    # notebook (home/BRAIN_NOTEBOOK.md)
    def notebook(self) -> str
    def notebook_append(self, text: str, now_t=None) -> dict
        # one dated section ("## " headings in the entry become "### "); only older DATED sections are
        # trimmed, never the preamble or undated sections; ValueError (nothing written) when condensed
        # sections leave no room for an entry that would fit an empty notebook.
        # {notebook_chars, limit, free_chars, trimmed, truncated}
    def notebook_rewrite(self, text: str) -> dict  # {notebook_chars, limit, free_chars[, warning]}
```
Units: `backtest` = 1, `robustness` = neighbours + 1 (the base), `optimize` = number of combos (not
evaluated ones refunded), `baseline` = 1. Exceeding a budget raises `BudgetExceeded` (the brain sees
it as a tool error).

## fibbot/autopilot.py — deterministic research without Claude

```python
def sweep(lab, presets: list[str] | None = None, promote: bool = False) -> dict
    # evaluate presets (default: those whose timeframe is in data.timeframes) on the research window
    # (+ baseline, fold stability); eligible = >= research.min_trades, avg_r > 0, fold_positive_frac
    # >= 0.5 and min(edge_r, edge_regime_r) > 0 (the row's edge_r is that binding edge); rank by avg_r.
    # Validate the top 3 eligible: a preset on the roster (any status, retired included) is skipped;
    # one whose validation overlaps the current window keeps that verdict (validations[].reused =
    # true, no new experiment, may still be promoted); new validations stop at
    # weekly_validations_left(). Optionally promote passing ones (respecting max_roster).
    # Returns {as_of, presets, ranking, validations, promoted, errors, skipped [{preset, reason}],
    # experiment_id}; the journal body lists validations, promotions, errors and skipped first.
```

## fibbot/report.py

```python
def render_status(lab, now_t=None) -> str      # terminal text
def write_report(lab, now_t=None) -> Path      # home/REPORT.md
```
Both start with alerts: "STALE DATA since <iso>: <reason>" while kv data_stall is set, "STALE
CANDLES" when a roster timeframe's oldest per-symbol candles lag now by more than 3 bars, and a
DATABASE line above 50 MB (GitHub rejects files over 100 MB). REPORT.md: a bold line when there
were system events in the last 7 days; roster table, graduation checks, "### Graduation changes
(last 5)" (journal kind "graduation"), recent trades, the brain's signal-review value, research and
review journal entries (clipped at line boundaries with "[... N more lines]"), "## Recent system
events (last 5)" (kind "system"), the notebook head and an activity line (validations last 7 days
used/cap, CLI holdout peeks). render_status adds "system events (errors, warnings) last 7 days: N".

## fibbot/brain/

- `client.py`: `class BrainUnavailable(RuntimeError)`; `class Brain(cfg, client=None)`:
  - The SDK client is `anthropic.Anthropic(max_retries=brain.max_retries)` (default 8).
  - `available() -> tuple[bool, str]`: enabled, a client can be built, and the installed anthropic has
    beta_tool, lib.tools.ToolError and beta.messages.tool_runner (else "pip install -r requirements.txt").
  - `run_agent(system, prompt, tools, max_iterations=None, effort=None, wrap_up=None, wrap_up_tools=(),
    stop=None) -> dict` calls `client.beta.messages.tool_runner(..., stream=True)` (model, max_tokens
    = brain.max_tokens incl. thinking, cached system block, tools, max_iterations, adaptive thinking,
    output_config={"effort"}, cache_control, betas/fallbacks unless brain.fallbacks is falsy) and
    reads each turn with get_final_message(). Returns {"text", "stop_reason", "turns", "usage",
    "refused", "resumes", "wrapped_up"} (+ "stopped_by"). A transient API error at any turn resumes the
    mirrored conversation (tool results kept, tools not re-run) up to 3 times with 5/60/180 s backoff.
    `stop` is checked after every turn: a non-empty reason closes the runner (stop_reason "stopped",
    "stopped_by"). When the run ends on the turn cap, a stop or max_tokens and wrap_up is given, one
    extra reply runs with the same system and tools; only the wrap_up_tools named in it run.
    ImportError / AttributeError / TypeError before any turn (an incompatible SDK) is BrainUnavailable.
  - `structured(system, prompt, schema, effort=None, max_tokens=None) -> dict | None` via
    `client.beta.messages.create(..., output_config={"effort", "format": {"type": "json_schema"}})`;
    max_tokens defaults to brain.signal_max_tokens or 16000, capped at 21000. None on refusal, on
    stop_reason max_tokens (logged), with no text block, invalid or non-object JSON; `last_failure`
    says which and `last_usage` holds the token usage.
  - BrainUnavailable covers credentials, an account that cannot serve requests (404 not_found, 402
    billing, 400 "credit balance": not retried, the autopilot stands in) and persistent transient
    failures; mid-stream SSE overloaded/api errors and network errors are transient; other 400s
    propagate.
  - Never use `budget_tokens`, `temperature`, forced `tool_choice`, or assistant prefill.
- `tools.py`: `build_tools(lab, cycle: str, actions=None) -> list` of `anthropic.beta_tool` closures,
  each returning a JSON string (<= 12k chars; values under "spec", "params", "base_spec" keep full
  precision so copied specs keep their spec_id). Tools: `get_lab_status`, `describe_strategy_space`,
  `list_experiments(limit, kind, sort_by, spec=None, spec_id=None, params=None)` (spec lookups return
  {spec_id, spec, n_experiments, latest_validation, roster, experiments, omitted}; sort_by other than
  "recent" is rejected for kind "validate" and ranked listings leave validate rows out; validate rows
  hide fold summaries; source "cli-holdout" rows are never listed; long listings are cut to whole
  rows plus {"omitted", "hint"}), `run_backtest(spec, window, symbols)` (research window only; by_side
  for direction "both"), `run_optimization(base_spec, grid)`, `run_baseline(spec)` (adds baseline_z =
  edge_r / baseline_avg_r_std), `run_robustness(spec)`, `validate_strategy(spec)` (only spec_id,
  name, timeframe, direction, window, period, n_symbols, passed, n_trials, experiment_id, headline
  metrics and the gate {passed, deflated_sharpe, checks, n_eff, t_clustered, dsr_trials}),
  `promote_to_paper(name, rationale, spec=None, spec_id=None)` (by spec_id: the spec stored with its
  validation), `retire_strategy(strategy_id, reason)`, `get_paper_trades(strategy_id, limit)`,
  `get_market_context(symbol, timeframe)` (as of the end of the research window: validation-window
  bars are hidden in research and review cycles; signal review uses its own causal context),
  `get_signal_review_stats()`, `read_notebook()`, `append_notebook(entry)` / `rewrite_notebook(content)`
  (return Lab's notebook dicts). Review cycles get the same tools except `run_optimization`.
- `prompts.py`: `SYSTEM_PROMPT` (stable text, cached; no dates or config-dependent numbers; a
  "Reading results" section; validation deflates by the looks at the locked window, graduation by
  the strategies ever on paper; retirement is final; walk-forward OOS measures tuning stability),
  `research_prompt(status_json, notebook, max_turns=None, last_cycle=None, first_cycle=True)`,
  `review_prompt(...)` (same arguments: the turn limit and the previous cycle's journal entry go in
  the user message), `WRAP_UP_NOTE`, `SIGNAL_SYSTEM`, `signal_prompt(signal, context, stats)`,
  `SIGNAL_SCHEMA` = {decision: "take"|"skip", confidence: number 0..1, reason: string}.
- `cycles.py`: `research_cycle(lab, brain, stop=None)`, `review_cycle(lab, brain, stop=None)`,
  `review_signals(lab, brain, signals=None, stop=None) -> list[dict]`.
  - Review cycles use brain.review_max_iterations (default 20, never more than brain.max_iterations)
    and brain.review_effort (default "medium"). A cycle cut off by its turn cap, max_tokens or stop
    gets the wrap-up turn in which only append_notebook runs. Titles: "(stopped early: iteration
    cap | max_tokens | context window | <stop reason>)"; a stopped cycle is ok=True and still sets kv
    last_{kind}_t. Building the tools with an incompatible SDK is BrainUnavailable.
  - kv brain_usage tallies tokens per UTC month; with brain.max_monthly_output_tokens set, cycles
    return ok=False ("<Kind> cycle skipped: brain spend cap reached") and signal reviews are skipped
    once the cap is reached.
  - review_signals: max_signal_reviews_per_day counts review CALLS in the last 24h, failed ones
    included (kv signal_review_attempts; unbilled brain-unavailable probes do not count); a signal is
    given up after 2 failed attempts; legs that already have a paper trade are not reviewed; the
    review record shown is as of the signal bar ("as_of"); a non-BrainUnavailable error fails that
    signal only and 2 in a row stop the batch; stop() is checked before each call.
  Each cycle journals its outcome and stores `kv["last_research_t"]` / `kv["last_review_t"]`.

## fibbot/scheduler.py and fibbot/cli.py

```python
def run_once(cfg, now_t=None, use_brain=True, lab=None, brain=None, stop=None) -> dict
    # universe refresh if stale -> candles.update_all(universe + held symbols, tfs needed by roster or
    # data.timeframes) -> paper tick -> brain.review_signals (if mode != off and brain available) ->
    # graduation for each roster strategy (journal status changes; auto-retire if paper drawdown >
    # 1.5 * graduation.max_drawdown; without a brain also when the status is "failing") -> review cycle
    # if due (skipped while the candle data is stalled) -> research cycle if due (brain available) else
    # autopilot.sweep(promote=True) when the roster is empty -> write_report (+ Store.compact about
    # once a day). Exceptions in any step are journaled (kind "system"; an identical failure of the same
    # step at most once per FAILURE_REPEAT_HOURS = 24, the next entry says how often it repeated) and
    # do not abort later steps. `stop` (checked before each step) is also handed to brain cycles and
    # signal reviews that declare a `stop` parameter, as cycle_stop(cfg, stop): a reason once the cycle
    # has run brain.max_cycle_minutes (default 60; 0/null = no limit) or "the bot is shutting down".
    # Research is due after brain.research_every_hours, or after brain.review_every_hours while the roster
    # is empty (not on every pass: each cycle costs money and validation tries); the empty-roster cadence
    # is deferred while weekly_validations_left() == 0. A brain cycle that started after the last
    # completed one (kv last_{kind}_attempt_t > last_{kind}_t: it failed or crashed) is retried only
    # CYCLE_RETRY_HOURS (6) after it started. The scheduler never sets status "graduated" (that would
    # stop the paper replay); it journals graduation status changes instead.
    # Candle stalls: one failing symbol is only a warning (data step "ok"). When every requested
    # update failed, an open paper position's candles are more than STALE_BARS = 3 bars old, or no
    # symbol has a bar within 3 bars on a timeframe, the data step fails on every pass (run --once
    # exits 1); the stall is journaled once when it starts and once when it recovers (kv data_stall).
def run_forever(cfg, use_brain=True) -> None
    # validates cfg; one pass now, then a pass at every UTC multiple of tick_every_seconds +
    # tick_offset_seconds until SIGINT/SIGTERM (the first signal stops after the current step; a
    # running brain cycle wraps up at its next turn). Logs to stdout and home/fibbot.log (rotated at
    # 5 MB, 3 old files kept).
```
CLI: `python -m fibbot <command>` with global `--config PATH --home DIR`. Commands:
`universe [--refresh]`, `data [--tf ...] [--symbols ...]`, `presets`,
`backtest (--preset N | --spec JSON_OR_FILE) [--window research|validation|all] [--symbols ...]
[--trades N] [--tf {1h,4h,1d}] [--json]` (validation / all print a warning: a trial and a holdout
look), `optimize (--preset|--spec) --grid JSON [--json]`, `baseline (--preset|--spec) [--runs N]
[--json]`, `robustness (--preset|--spec) [--json]`, `validate (--preset|--spec) [--json]`,
`promote (--preset|--spec) [--name NAME] --rationale TEXT`, `retire ID --reason TEXT`,
`sweep [--promote] [--presets NAME ...] [--json]`, `epoch [--roll]`, `tick`,
`brain {research,review,signals}` (research/review get the max_cycle_minutes stop),
`run [--once] [--no-brain]`, `status [--json]`, `report`. Exit codes: 0 ok; 1 user error (bad
arguments, config or spec, unknown strategy, failed downloads, tick errors including candle
download failures, or failed scheduler steps such as a candle stall); 2 brain unavailable;
130 interrupted.
