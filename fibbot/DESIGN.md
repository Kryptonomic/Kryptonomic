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

- Python >= 3.10. Core package is **standard library only**. `anthropic` (>= 2.4.0) is imported
  lazily, only inside `fibbot/brain/`.
- Time: integer Unix **seconds**, UTC. `Bars.t[i]` is the bar's OPEN time. Only closed bars exist.
- `side`: +1 long, -1 short.
- Missing indicator values are `float('nan')` (lists stay index-aligned with bars).
- Every module that can be pure is pure. No module except `fibbot/data/*` touches the network.
- Tests: `tests/test_<module>.py`, written with `unittest` (runnable by `python -m unittest` and
  `pytest`), no network, use synthetic data. Test files may build bars with `Bars.from_rows`.
- JSON returned to the brain must be compact (<= ~12k chars per tool result) and JSON-serializable
  (`json.dumps(x, default=str)`), floats rounded to 4 significant decimals where sensible.

Core types live in `fibbot/types.py` (already written): `Bars`, `Trade`, `Signal`, `Costs`,
`TF_SECONDS`, `BARS_PER_YEAR`. Config lives in `fibbot/config.py` (already written):
`load_config()`, `home_path()`, `DEFAULTS`, `excluded_bases()`.

## Module map

```
fibbot/
  types.py config.py                    foundation (done)
  data/universe.py data/candles.py      network: CoinGecko ranking, Binance klines (+OKX fallback), CSV.gz cache
  indicators.py fib.py                  pure math: EMA/ATR/RSI/ADX, causal ZigZag pivots, fib levels
  metrics.py portfolio.py               stats (PF, expectancy, DD, Sharpe, bootstrap, deflated Sharpe), sizing sim
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
    """Cached in home/universe.json {"as_of": t, "symbols": [...]} ; re-rank when older than
    universe.refresh_days or refresh=True. On any network failure fall back to the cached list, else
    universe.static_fallback (filtered by excluded bases) truncated to top_n."""
```

## fibbot/data/candles.py

```python
def fetch_klines(cfg, symbol: str, tf: str, start_t: int, end_t: int | None = None) -> list[tuple]
    """Network. Binance GET {binance_base}/api/v3/klines?symbol=&interval=&startTime=ms&limit=1000,
    paginated until end_t/now. Returns [(t, o, h, l, c, v), ...] with t in SECONDS, CLOSED bars only
    (drop any bar whose close time >= now). Retries with backoff (5 tries). If Binance fails for a
    recent range, try OKX /api/v5/market/history-candles (instId BASE-QUOTE, bar 1H/4H/1Dutc) as a fallback."""

class CandleStore:
    """Local cache of closed candles: home/candles/{SYMBOL}_{tf}.csv.gz with header t,o,h,l,c,v."""
    def __init__(self, cfg): ...
    def get(self, symbol: str, tf: str) -> Bars            # from disk (+ in-memory cache); empty Bars if none; no network
    def update(self, symbol: str, tf: str) -> Bars         # incremental fetch from last cached bar (or data.history_start), save, return
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
    # profit_factor (sum of positive ret_pct / |sum of negative ret_pct|; 99.0 if no losses and n>0;
    # 0.0 if no wins), avg_ret_pct, avg_bars_held, max_consec_losses, long_n, short_n,
    # exit_reasons {reason: count}, sharpe_per_trade (mean/stdev of r_multiple, 0 if n<2).
    # n == 0 -> all numeric keys 0.
def max_drawdown(values: list[float]) -> float          # positive fraction, e.g. 0.12
def equity_stats(curve: list[tuple[int, float]], start_equity: float) -> dict
    # total_return, cagr, max_drawdown, sharpe, sortino, calmar, n_days. Curve is resampled to daily
    # closes (UTC days, forward-filled) before computing daily returns; annualize with 365.
def monthly_returns(curve, start_equity) -> list[tuple[str, float]]   # [("2024-01", 0.013), ...]
def bootstrap_mean_ci(values, level=0.95, n_boot=2000, seed=7) -> tuple[float, float]   # deterministic
def skew_kurt(values) -> tuple[float, float]           # population skew, NON-excess kurtosis (normal=3)
def t_stat_mean(values) -> float
def probabilistic_sharpe(sr, n, skew, kurt, sr_star=0.0) -> float
    # Phi((sr - sr_star) * sqrt(n - 1) / sqrt(1 - skew*sr + (kurt - 1)/4 * sr^2)); guard sqrt of <=0 -> 1e-12
def expected_max_sharpe(n_trials: int, var_sr: float) -> float
    # sqrt(var_sr) * ((1-g)*Phi^-1(1 - 1/N) + g*Phi^-1(1 - 1/(N*e))), g = 0.5772156649 ; 0 if N <= 1
def deflated_sharpe(values, n_trials: int, var_sr: float | None = None) -> float
    # PSR of the per-trade Sharpe of `values` against sr_star = expected_max_sharpe(n_trials, var_sr);
    # var_sr defaults to 1/len(values). 0.0 if len(values) < 3.
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
    skipped: list[dict]              # {"trade": Trade, "reason": "max_open"|"max_total_risk"|"daily_loss_limit"|"no_equity"}

def simulate_portfolio(trades: list[Trade], pcfg: dict, start_equity: float | None = None) -> PortfolioResult
```
Events in time order: each trade has an entry event at `entry_t` and (if closed) an exit at `exit_t`;
at equal timestamps exits go first. On entry (in order of entry_t, then symbol): skip if realized PnL
for that UTC day <= -daily_loss_limit * equity at the day's first event; skip if open count >=
max_open_positions; risk_amount = risk_per_trade * equity; notional = min(risk_amount / risk_pct,
max_notional_frac * equity); new_risk = notional * risk_pct; skip if sum(open new_risk) + new_risk >
max_total_risk * equity (+1e-9). On exit: pnl = notional * ret_pct, equity += pnl, append curve point.
`equity` here is realized equity. `start_equity` defaults to pcfg["starting_equity"].

## fibbot/store.py  (SQLite, stdlib sqlite3, WAL)

```python
class Store:
    def __init__(self, path: str | Path)       # creates schema if missing
    def close(self)
    # experiments: kind in {"backtest","optimize","baseline","validate","robustness","sweep"}
    def add_experiment(self, kind, spec: dict | None, spec_id: str | None, window_start: int | None,
                       window_end: int | None, metrics: dict, source: str = "cli", notes: str = "") -> int
    def list_experiments(self, limit=20, kind=None, sort_by="recent") -> list[dict]
        # sort_by: "recent" | "avg_r" | "profit_factor" (read from metrics JSON; missing -> last)
        # rows: {id, created_t, kind, spec_id, spec, window_start, window_end, metrics, source, notes}
    def get_experiment(self, exp_id) -> dict | None
    def latest_validation(self, spec_id) -> dict | None      # newest kind="validate" row for spec_id
    def count_experiments(self, kind=None, since_t=None) -> int
    # trials (distinct specs ever evaluated on the research window; for deflated Sharpe)
    def record_trial(self, spec_id: str, sharpe: float | None) -> None   # upsert; keeps latest sharpe
    def trial_stats(self) -> dict    # {"n": distinct spec count, "var_sharpe": population variance of non-null sharpes (0 if <2)}
    # roster
    def add_to_roster(self, strategy_id, name, spec: dict, paper_start_t: int, validation: dict, rationale: str) -> None
        # status "paper". Re-adding a retired id re-activates it with the new paper_start_t.
    def roster(self, status: str | None = "paper") -> list[dict]  # rows incl. spec (dict), validation (dict)
    def get_strategy(self, strategy_id) -> dict | None
    def set_strategy_status(self, strategy_id, status, reason="") -> None   # status: paper|retired|graduated
    def count_ever_on_paper(self) -> int
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
    trail_atr: float = 0.0               # 0 disables; else trail remainder after tp1 at close - k*ATR
    max_hold_bars: int = 120             # 1..2000 time stop
    min_rr: float = 1.5                  # 0..10 planned (blended) reward/risk at arming

    def validate(self) -> "StrategySpec"   # raises ValueError("field: reason") ; returns self
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "StrategySpec"   # unknown keys -> ValueError ; missing -> defaults
    def spec_id(self) -> str               # 10 hex chars: sha1 of canonical JSON of to_dict() minus "name"
    def with_overrides(self, **kw) -> "StrategySpec"
```

`PARAM_SPACE: dict[str, dict]` — for every field except name: `{"type": "int"|"float"|"bool"|"choice",
"min", "max", "step" (numeric), "choices" (choice), "default", "doc"}`. `step` is used for robustness
neighbours. `PRESETS: dict[str, StrategySpec]` — at least:

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
```

**Determinism and causality (hard requirements, tested):** decisions made at the close of bar i use
only bars <= i; orders placed at the close of bar i can only fill on bars >= i+1. Consequence:
for every k, `run(bars.slice(0, k))` yields exactly the trades of `run(bars)` that closed at or
before bar k-1 (same prices), and the same signals with t <= t[k-1]. Appending bars never changes
earlier output.

**Per-bar processing order for bar i:**
1. *Fills on bar i* from orders placed at the close of bar i-1 (only if t[i] >= trade_start_t):
   - open position (entered before i): manage exits (below);
   - pending limit order: long fills if `l[i] <= px` at `min(o[i], px)` (short: `h[i] >= px` at
     `max(o[i], px)`), no slippage. If both a long and a short limit would fill, fill the one whose
     price is closer to o[i] and cancel the other;
   - pending market order (confirm mode): fills at `o[i] * (1 + side*slippage)`;
   - on the entry bar, after the fill: if the fill price is already beyond the stop, exit immediately
     at the fill price (with slippage, reason "stop"); else if the bar's range reaches the stop
     (`l[i] <= stop` long / `h[i] >= stop` short) exit at the stop with slippage (reason "stop").
     **Targets are never credited on the entry bar.** Mark the leg used.
2. *Close of bar i*: update the pivot tracker; update both side setups; trailing stop and time stop
   for an open position; then (if flat) place orders for bar i+1.

**Setup definition (long; short mirrors every comparison):**
- anchor = the most recent confirmed LOW pivot (confirm_idx <= i). leg_id = `f"L{anchor.t}"`
  (short: most recent confirmed HIGH pivot, leg_id `f"S{anchor.t}"`).
- extreme = highest high over bars (anchor.idx, i] and its index ext_idx (first occurrence). It is
  tracked incrementally and resets when the anchor changes.
- The leg is skipped/dead when: leg_id in used legs or skip_legs; price traded below the anchor
  after the anchor (`min low over (anchor.idx, i] < anchor.price`); or `i - ext_idx > setup_expiry_bars`
  while unfilled; or the low of bar i is <= the leg's stop price while unfilled (price ran through the
  stop). A dead leg is added to used legs.
- It *qualifies* at bar i when: `extreme - anchor.price >= min_leg_atr * atr[i]`; the trend filter
  passes **evaluated at ext_idx** (long: `ema`: c > EMA(trend_ema); `ema_slope`: c > EMA and
  EMA[ext_idx] > EMA[ext_idx-10]; `ema_stack`: EMA(50) > EMA(trend_ema) and c > EMA(50); `none`: pass;
  nan fails); `adx[ext_idx] >= adx_min` (when adx_min > 0); structure filter if enabled; and the
  reward/risk check passes.
- Levels: `retrace(r) = retrace_price(anchor, extreme, r)`, `ext(e) = extension_price(anchor, extreme, e)`.
  zone_top = retrace(entry_zone_lo) (shallow), zone_bottom = retrace(entry_zone_hi) (deep).
  entry ratio = lo | (lo+hi)/2 | hi per entry_at; entry_px = retrace(entry ratio).
  stop_px = retrace(stop_ratio) - stop_atr_buffer * atr[i]  (short: + buffer).
  tp1 = ext(tp1_ext), tp2 = ext(tp2_ext).
  RR = (tp1_frac*(tp1-entry) + (1-tp1_frac)*(tp2-entry)) / (entry-stop) (short: signs flipped);
  requires entry > stop, tp1 > entry and RR >= min_rr. (If tp1_frac == 1, RR uses tp1 only.)
- The first bar at which a leg qualifies emits a `Signal` (entry_px, stop_px, tp1, tp2, meta with
  anchor, extreme, atr, rr, mode).
- **limit mode**: when flat and the leg qualifies, place a limit at entry_px for bar i+1 only if
  `c[i] > entry_px` (short: `c[i] < entry_px`). Orders are re-computed every bar (the extreme may move).
- **confirm mode**: the leg is *touched* at bar j when `l[j] <= zone_top` and `l[j] > stop_px`.
  Within `confirm_window` bars after the touch (j..j+window), a confirmation at the close of bar k
  places a market order for bar k+1 if the leg still qualifies with entry = c[k]:
  `reclaim`: c[k] > zone_top and c[k] > o[k]; `engulf`: c[k] > o[k], c[k-1] < o[k-1], c[k] >= o[k-1],
  o[k] <= c[k-1]; `rsi`: rsi[k-1] < 40 and rsi[k] > rsi[k-1]. No confirmation in the window ->
  untouched again (can be re-touched later).
- Stop/targets/anchor/extreme are frozen into the position at fill time.

**Position management on bars after the entry bar** (long; short mirrors):
1. `o <= stop`: exit the remainder at `o*(1-slippage)`.
2. elif `l <= stop`: exit the remainder at `stop*(1-slippage)` (stop before targets - worst case).
3. elif not tp1_hit and `h >= tp1`: fill tp1_frac at `max(o, tp1)` (limit, no slippage); tp1_frac==1
   closes the trade (reason "target"). If move_stop_to_be: stop = max(stop, entry*(1+2*fee_rate)) from
   the NEXT bar. Then if `h >= tp2` close the remainder at `max(o, tp2)` (reason "target").
4. elif tp1_hit and `h >= tp2`: close the remainder at `max(o, tp2)` (reason "target").
5. At the close: if trail_atr > 0 and tp1_hit: stop = max(stop, c - trail_atr*atr). If
   `i - entry_idx >= max_hold_bars`: exit the remainder at `c*(1-slippage)` (reason "time").
Exit reason when a stop fills: "stop" if the stop was never moved, "breakeven" if it was moved to
breakeven and not trailed above it, "trail" if trailed above breakeven.

**Accounting.** Fills k with fractions f_k (sum 1) at prices p_k; entry fill P (incl. slippage), fee
rate f:
- long: `ret_pct = sum(f_k*p_k)/P*(1-f) - (1+f)`; short: `ret_pct = (1-f) - sum(f_k*p_k)/P*(1+f)`.
- `risk_pct = |P - initial_stop| / P`; `r_multiple = ret_pct / risk_pct`; exit_price = sum(f_k*p_k).
- `bars_held = exit_idx - entry_idx`; exit_t = t[exit_idx].
- meta: {"anchor", "extreme", "leg_size", "atr", "entry_ratio", "tp1", "tp2", "rr",
  "fills": [[frac, price, reason, t], ...], "entry_mode"}.
- An open position at the last bar is reported as `open_trade` marked at c[-1] (with exit costs and
  slippage), exit_reason "open", exit_t None.

```python
def manage_position(bars: Bars, atr: list[float], entry_idx: int, side: int, entry_px: float,
                    stop_px: float, tp1: float, tp2: float, spec: StrategySpec, costs: Costs,
                    start_mgmt_idx: int | None = None) -> dict
    # Pure helper implementing "position management" + accounting from entry_idx onward (used by the
    # random baseline). Returns {"exit_idx", "exit_reason", "fills", "ret_pct", "risk_pct", "r_multiple",
    # "open": bool}. FibEngine must use the same per-bar exit logic (shared code, not a copy).
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
    metrics: dict                  # trade_stats(...) | equity_stats(...) | {"n_open": k}
    per_symbol: dict[str, dict]    # {sym: {"n", "total_r", "avg_r", "win_rate", "profit_factor"}}
    per_year: dict[str, dict]      # {"2023": {"n", "total_r", "avg_r"}} by entry year
    def summary(self, max_symbols: int = 20) -> dict   # compact JSON-safe dict (no trade lists)

def backtest(spec, bars_by_symbol, start_t, end_t, cfg, workers=None) -> BacktestResult
    # costs from cfg["costs"], portfolio from cfg["portfolio"], workers from cfg["research"]["workers"]
```

## fibbot/research.py

```python
def compute_windows(cfg, bars_by_symbol) -> dict
    # {"data_start", "data_end", "research_start", "research_end", "validation_start", "validation_end"}
    # data_end = min over symbols of (last_t + tf) ; validation_start = data_end - validation_months*30.44 days
    # (floored to a UTC day); research_start = min first t; research_end = validation_start.
def fold_bounds(start_t, end_t, folds) -> list[tuple[int, int]]
def fold_stats(trades, bounds) -> list[dict]       # trade_stats per fold by entry_t (n, avg_r, total_r, profit_factor)
def evaluate(spec, bars_by_symbol, cfg, windows, window="research") -> tuple[BacktestResult, dict]
    # window: "research" | "validation" | "all". Returns (result, extras) with
    # extras = {"folds": [...], "fold_positive_frac": x, "min_fold_avg_r": y}
def random_baseline(spec, bars_by_symbol, cfg, start_t, end_t, n_runs=10, seed=1, result=None) -> dict
    # For each real trade of the strategy, draw a random entry bar in the same symbol & window (same side),
    # entry at market (open, slippage), stop at the same risk_pct, tp1/tp2 at the same R distances, same
    # management via strategy.manage_position. {"strategy_avg_r", "baseline_avg_r_mean", "baseline_avg_r_std",
    # "beats_frac" (share of runs the strategy beats), "runs": [avg_r...], "edge_r": strategy - baseline mean}
def walk_forward_optimize(base_spec, grid: dict[str, list], bars_by_symbol, cfg, windows,
                          objective="avg_r", on_combo=None) -> dict
    # combos = product(grid) (<= research.max_grid_size else ValueError). Each combo evaluated once on the
    # research window; trades bucketed into folds. Anchored WF: for fold k = 1..F-1 pick the combo with the
    # best objective on folds [0, k) among combos with >= max(5, min_trades // folds) trades there; record
    # its fold-k trades as OOS. Returns {"n_combos", "oos": trade_stats(OOS trades), "oos_by_fold": [...],
    # "chosen": [{"fold", "params", "is_avg_r"}], "best": {"params", "spec", "spec_id", ...stats},
    # "top": [5 best rows {params, n, avg_r, profit_factor, fold_positive_frac}],
    # "sensitivity": {param: {value: mean avg_r}}}. on_combo(spec, result) callback (used to record trials).
def neighbours(spec) -> list[StrategySpec]        # each numeric/choice param moved one step (both ways), valid only
def robustness(spec, bars_by_symbol, cfg, windows, max_neighbours=16) -> dict
    # {"n", "positive_frac", "median_avg_r", "worst_avg_r", "rows": [...]} on the research window
def validation_gate(result: BacktestResult, gate: dict, n_trials: int, var_sr: float) -> dict
    # {"passed": bool, "checks": [{"name", "value", "threshold", "passed"}], "deflated_sharpe": x}
    # checks: min_trades, min_profit_factor, min_expectancy_r, min_deflated_sharpe (deflated_sharpe of
    # r_multiples with n_trials/var_sr), max_drawdown (portfolio equity), min_symbol_win_frac.
```

## fibbot/paper.py — forward paper trading by deterministic replay

Paper trading re-runs the *same* FibEngine every tick on each symbol from a fixed anchor
(`paper_start_t - warmup_bars(spec) * TF_SECONDS[tf]`) to the newest closed bar with
`trade_start_t = paper_start_t`. Because the engine is causal and the anchor is fixed, past paper
trades never change; new bars only add/close trades. This guarantees backtest/paper parity.

```python
class PaperTrader:
    def __init__(self, cfg, store: Store, candles: CandleStore)
    def tick(self, update_data: bool = True, now_t: int | None = None) -> dict
        # For each roster strategy (status "paper"): optionally candles.update_all(universe, {tf});
        # run engine per symbol in get_universe(cfg) (skip_legs = store.vetoed_legs(...) only when
        # cfg.brain.signal_review == "active"); upsert every trade (closed + open);
        # add signals with t >= paper_start_t - TF_SECONDS[tf]; simulate_portfolio over the strategy's
        # trades with starting_equity = cfg.paper.starting_equity; write notional/pnl and the equity curve.
        # Returns {"strategies": {sid: {"new_signals": [Signal dict], "new_entries": [...], "new_exits": [...],
        #          "equity": float, "open": int}}, "new_signals": [Signal...], "errors": [...]}
    def account(self, strategy_id) -> dict
        # {"strategy_id", "name", "timeframe", "paper_start_t", "days", "equity", "return", "n_closed", "n_open",
        #  "stats": trade_stats(closed), "max_drawdown", "open_positions": [...]}
```
The universe for a strategy is `get_universe(cfg)` at tick time (cached list), so the paper universe
can change when the market-cap ranking changes; trades already recorded are kept.

## fibbot/graduation.py

```python
def evaluate(store: Store, cfg, strategy_id: str, now_t: int | None = None) -> dict
    # {"strategy_id", "status", "checks": [{"name","value","threshold","passed"}], "summary": str}
```
Inputs: closed paper trades, equity curve, roster row (validation metrics, paper_start_t).
Checks (thresholds from cfg["graduation"]): `days >= min_days`; `trades >= min_trades`;
`profit_factor >= min_profit_factor`; `bootstrap CI lower bound of mean R > 0` (ci_level);
`max_drawdown <= max_drawdown`; `share of positive calendar months >= min_positive_month_frac`
(months with >= 1 closed trade); `deflated_sharpe(r, n_trials=store.count_ever_on_paper()) >=
min_deflated_sharpe`; `paper mean R >= validation mean R - max_shortfall_vs_validation_r`.
Status: "graduated" (all pass) | "failing" (n >= min_trades/2 and (profit_factor < 1.0 and CI upper
bound of mean R < 0), or max_drawdown > 1.5 * threshold) | "collecting" (days or trade count short)
| "promising" (otherwise). Graduation never enables live trading; it is a report.

## fibbot/lab.py — facade used by CLI, scheduler and brain tools

```python
class BudgetExceeded(RuntimeError): ...

class Lab:
    def __init__(self, cfg, store: Store | None = None, candles: CandleStore | None = None)
        # store defaults to Store(home/fibbot.sqlite); candles to CandleStore(cfg)
    cfg; store; candles
    def universe(self) -> list[str]
    def bars(self, tf: str, symbols: list[str] | None = None) -> dict[str, Bars]   # cached per tf, from disk
    def windows(self, tf: str = "4h") -> dict
    # budgets (per brain cycle; CLI runs are unbudgeted unless begin_cycle was called)
    def begin_cycle(self, kind: str) -> None      # resets counters from cfg.research
    def budget(self) -> dict                      # {"backtest_units_left", "validations_left"} (None = unlimited)
    # research (records experiments + trials; source = "brain" inside a cycle else "cli")
    def backtest(self, spec: StrategySpec | dict, window="research", symbols=None) -> dict   # summary + folds + experiment_id
    def optimize(self, base_spec, grid) -> dict
    def baseline(self, spec, n_runs=10) -> dict
    def robustness(self, spec) -> dict
    def validate(self, spec) -> dict              # validation window + gate; consumes a validation
    def promote(self, spec, name: str, rationale: str, now_t=None) -> dict
        # requires store.latest_validation(spec_id) with passed=True ; roster size < paper.max_roster;
        # paper_start_t = start of the next bar of spec.timeframe after now. ValueError otherwise.
    def retire(self, strategy_id, reason) -> dict
    # paper / reporting
    def paper(self) -> PaperTrader
    def status(self) -> dict    # roster with account + graduation per strategy, budgets, windows, trial stats, data range
    def graduation(self, strategy_id) -> dict
    def signal_review_stats(self) -> dict
        # brain shadow-review value: closed paper trades split by the brain's decision on their leg:
        # {"reviewed": n, "take": trade_stats-lite, "skip": trade_stats-lite, "unreviewed": ..., "skip_minus_take_avg_r"}
    # notebook (home/BRAIN_NOTEBOOK.md)
    def notebook(self) -> str
    def notebook_append(self, text: str) -> None   # timestamped section; trims oldest sections beyond notebook_max_chars
    def notebook_rewrite(self, text: str) -> None
```
Units: `backtest` = 1, `robustness` = number of neighbours, `optimize` = number of combos,
`baseline` = 1. Exceeding a budget raises `BudgetExceeded` (the brain sees it as a tool error).

## fibbot/autopilot.py — deterministic research without Claude

```python
def sweep(lab, presets: list[str] | None = None, promote: bool = False) -> dict
    # evaluate every preset on the research window (+ baseline, fold stability), rank by avg_r among those
    # with >= research.min_trades and fold_positive_frac >= 0.5 and beating the baseline; validate the top 3;
    # optionally promote passing ones (respecting max_roster). Returns a ranking table + validations.
```

## fibbot/report.py

```python
def render_status(lab) -> str          # terminal text
def write_report(lab, path=None) -> Path   # home/REPORT.md : roster table, graduation checks, recent trades,
                                           # brain signal-review value, last research summary, notebook head
```

## fibbot/brain/

- `client.py`: `class BrainUnavailable(RuntimeError)`; `class Brain(cfg, client=None)`:
  - `available() -> tuple[bool, str]`: `anthropic` importable and a client can be built.
  - `run_agent(system: str, prompt: str, tools: list, max_iterations=None, effort=None) -> dict`
    uses `client.beta.messages.tool_runner(model=cfg.brain.model, max_tokens=cfg.brain.max_tokens,
    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": prompt}], tools=tools, max_iterations=...,
    thinking={"type": "adaptive"}, output_config={"effort": effort}, cache_control={"type": "ephemeral"},
    betas=["server-side-fallback-2026-07-01"], fallbacks="default")` (omit betas/fallbacks when
    cfg.brain.fallbacks is falsy). Iterate the runner, collect usage; return {"text", "stop_reason",
    "turns", "usage": {"input_tokens", "output_tokens", "cache_read_input_tokens"}, "refused": bool}.
    Check `stop_reason == "refusal"` before reading content.
  - `structured(system, prompt, schema: dict, effort=None, max_tokens=4000) -> dict | None` via
    `client.beta.messages.create(..., output_config={"effort": e, "format": {"type": "json_schema",
    "schema": schema}})`, parse the first text block as JSON; None on refusal.
  - Never use `budget_tokens`, `temperature`, forced `tool_choice`, or assistant prefill.
- `tools.py`: `build_tools(lab, cycle: str) -> list` of `anthropic.beta_tool`-decorated closures, each
  returning a JSON string (truncated to 12k chars). Tools: `get_lab_status`, `describe_strategy_space`,
  `list_experiments(limit, kind, sort_by)`, `run_backtest(spec, window, symbols)` (window research only
  for the brain; validation access only via validate_strategy), `run_optimization(base_spec, grid)`,
  `run_baseline(spec)`, `run_robustness(spec)`, `validate_strategy(spec)`,
  `promote_to_paper(spec, name, rationale)`, `retire_strategy(strategy_id, reason)`,
  `get_paper_trades(strategy_id, limit)`, `get_market_context(symbol, timeframe)`,
  `get_signal_review_stats()`, `read_notebook()`, `append_notebook(entry)`, `rewrite_notebook(content)`.
  Review cycles get the same tools except `run_optimization` (and a smaller budget).
- `prompts.py`: `SYSTEM_PROMPT` (stable text, cached), `research_prompt(status_json, notebook)`,
  `review_prompt(status_json, notebook)`, `SIGNAL_SYSTEM`, `signal_prompt(signal, context, stats)`,
  `SIGNAL_SCHEMA` = {decision: "take"|"skip", confidence: number 0..1, reason: string}.
- `cycles.py`: `research_cycle(lab, brain) -> dict`, `review_cycle(lab, brain) -> dict`,
  `review_signals(lab, brain, signals: list[dict] | None = None) -> list[dict]` (respects
  max_signal_reviews_per_day; mode "off" -> no-op). Each journals its outcome and stores
  `kv["last_research_t"]` / `kv["last_review_t"]`.

## fibbot/scheduler.py and fibbot/cli.py

```python
def run_once(cfg, now_t=None, use_brain=True, lab=None, brain=None) -> dict
    # universe refresh if stale -> candles.update_all(universe, tfs needed by roster or data.timeframes)
    # -> paper tick -> brain.review_signals (if mode != off and brain available) -> graduation for each
    # roster strategy (journal status changes; auto-retire if paper drawdown > 1.5 * graduation.max_drawdown)
    # -> review cycle if due -> research cycle if due or roster empty (brain available) else
    # autopilot.sweep(promote=True) when the roster is empty -> write_report. Exceptions in any step are
    # journaled (kind "system") and do not abort later steps.
def run_forever(cfg, use_brain=True) -> None   # loop: sleep until next tf boundary + tick_offset_seconds, run_once
```
CLI: `python -m fibbot <command>` with global `--config PATH --home DIR`. Commands: `universe [--refresh]`,
`data [--tf ...] [--symbols ...]`, `presets`, `backtest (--preset N | --spec JSON_OR_FILE) [--window]
[--symbols ...] [--trades N]`, `optimize (--preset|--spec) --grid JSON`, `baseline`, `robustness`,
`validate`, `promote --name --rationale`, `retire ID --reason`, `sweep [--promote]`, `tick`,
`brain {research,review,signals}`, `run [--once] [--no-brain]`, `status`, `report`.
