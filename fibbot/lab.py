"""Lab: the facade the CLI, the scheduler and the brain tools use (see DESIGN.md, "fibbot/lab.py").

Every research call records an experiment. A full-universe evaluation on the research window
records a trial (spec_id, per-trade Sharpe, trade count) for the deflated-Sharpe correction; a
symbol-subset run and every evaluation that looks at the held-out period (validate, or a CLI
backtest on the "validation"/"all" window) count as trials without a Sharpe, so they raise the
trial count but never move var_sharpe. Between begin_cycle() and end_cycle() (a brain cycle) calls
are budgeted, experiments are recorded with source "brain", and the validation window is only
reachable through validate().

Holdout honesty (validate):
- the validation symbols are chosen point-in-time: the top universe.top_n of a candidate pool
  (today's universe, every symbol the lab has seen in a universe, the static fallback list and the
  candle cache) by trailing PIT_VOLUME_DAYS quote volume before validation_start, not today's
  market-cap ranking (which is taken at the END of the validation window);
- the gate also requires the strategy to beat random entries on the validation window
  (min_validation_edge_r), which removes the drift of coins that rallied into the universe;
- one spec gets one look per validation period: re-validating a spec whose recorded validation
  window overlaps the current one is refused (an identical window and symbol set is a
  deterministic replay and allowed); every distinct look at a spec's holdout counts as a trial;
- promote needs every validation of the spec in the last validation_months to have passed, and
  brain cycles / the autopilot share a rolling research.max_validations_per_week cap.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone

from . import research
from .backtest import LITE_KEYS, BacktestResult, as_spec, iso_day, rounded
from .config import excluded_bases, home_path
from .data.candles import CandleStore, atomic_write
from .data.universe import get_universe
from .graduation import evaluate as graduation_evaluate
from .metrics import trade_stats
from .paper import ACCOUNT_KEY, UNIVERSE_KEY, PaperTrader, account_settings, engine_fingerprint
from .store import EXPERIMENT_KINDS, Store
from .strategy import ENGINE_VERSION, StrategySpec
from .types import TF_SECONDS, Bars

NOTEBOOK_FILE = "BRAIN_NOTEBOOK.md"
NOTEBOOK_HEADER = "# Brain notebook"
TRUNCATED = "\n[... truncated to fit brain.notebook_max_chars]"
NOTEBOOK_LOW_ROOM_FRAC = 0.25  # a rewrite leaving less than this share of the limit gets a warning
UNIVERSE_TTL = 3600.0          # seconds a universe lookup is reused (a failing ranking is not retried per call)
MEMO_SIZE = 8                  # research evaluations kept for reuse (e.g. backtest -> baseline of one spec)
ROBUSTNESS_NEIGHBOURS = research.MAX_NEIGHBOURS   # the whole one-step neighbourhood
MAX_BASELINE_RUNS = 100
REVIEW_UNITS_DIVISOR = 4       # review cycles get 1/4 of the research backtest units and one validation
DAY = 86400
WEEK = 7 * DAY
PIT_VOLUME_DAYS = 90           # validation symbols: quote volume over the days before validation_start
VALIDATION_BASELINE_RUNS = 50  # random-entry runs behind the gate's edge checks (enough for the z test)
MAX_VALIDATIONS_PER_WEEK = 3   # default research.max_validations_per_week (brain cycles + autopilot)
EPOCH_KEY = "validation_epoch"  # kv: {"validation_end", "t", "reason"}: the pinned holdout (see _windows)
HOLDOUT_SOURCE = "cli-holdout"  # experiment source of a CLI backtest on the validation / all window
LOOKS_KEY = "holdout_looks:"   # kv prefix: per-spec ledger of evaluations that looked at the holdout
SEEN_KEY = "universe_seen"     # kv: every symbol the lab has seen in a universe (point-in-time pool)
PEEKS_KEY = "holdout_peeks"    # kv: number of CLI backtests on the validation / all window
JOURNAL_SCAN = 200             # newest system journal entries scanned for the 7-day error count
_STAMPED = re.compile(r"## \d{4}-\d\d-\d\dT\d\d:\d\d(:\d\d)?Z")


class BudgetExceeded(RuntimeError):
    """A brain cycle asked for more research than its remaining budget."""


def iso_time(t: int | float | None) -> str | None:
    """UTC 'YYYY-MM-DDTHH:MM:SSZ' of a Unix time (None stays None)."""
    if t is None:
        return None
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now(now_t: int | None) -> int:
    return int(time.time()) if now_t is None else int(now_t)


def _trade_row(tr) -> dict:
    """Compact, rounded, JSON-safe view of a Trade (times as ISO strings, no meta)."""
    return rounded({"symbol": tr.symbol, "side": tr.side, "leg_id": tr.leg_id,
                    "entry_t": iso_time(tr.entry_t), "entry_price": tr.entry_price,
                    "stop_price": tr.stop_price, "exit_t": iso_time(tr.exit_t),
                    "exit_price": tr.exit_price, "exit_reason": tr.exit_reason,
                    "r_multiple": tr.r_multiple, "bars_held": tr.bars_held})


def _sections(text: str) -> tuple[str, list[str]]:
    """(preamble, ["## " sections ...]) of a notebook text."""
    parts = re.split(r"(?m)^(?=## )", text)
    return parts[0], parts[1:]


def _trim_sections(text: str, limit: int) -> tuple[str, list[str]]:
    """Drop the oldest timestamped (appended) sections until text fits in limit. The preamble,
    undated sections (what rewrite_notebook condensed) and the newest section are never dropped,
    so the result can stay over the limit. Returns (text, headings of the dropped sections)."""
    head, sections = _sections(text)
    dropped = []
    while len(head) + sum(map(len, sections)) > limit:
        i = next((i for i, s in enumerate(sections[:-1]) if _STAMPED.match(s)), None)
        if i is None:
            break
        dropped.append(sections.pop(i).split("\n", 1)[0].strip())
    return head + "".join(sections), dropped


def _trim_notebook(text: str, limit: int) -> str:
    """_trim_sections without the dropped headings."""
    return _trim_sections(text, limit)[0]


def _with_header(text: str) -> str:
    return text if text.startswith(NOTEBOOK_HEADER) else f"{NOTEBOOK_HEADER}\n\n{text}"


def _look_key(start, end, symbols) -> tuple:
    """Identity of one look at a holdout period: same window and symbol set = the same draw."""
    return (int(start), int(end), tuple(sorted(symbols or ())))


def _look_id(spec_id: str, result) -> str:
    """store.record_look key of one draw: spec, window and symbol set."""
    start, end, syms = _look_key(result.start_t, result.end_t, result.symbols)
    return f"{spec_id}|{start}|{end}|{','.join(syms)}"


def _engine_of(metrics) -> int:
    """strategy.ENGINE_VERSION an experiment was recorded with (1 before versions were recorded)."""
    v = metrics.get("engine_version") if isinstance(metrics, dict) else None
    return 1 if v is None else int(v)


class Lab:
    """Research, paper trading and reporting over one config, store and candle cache."""

    def __init__(self, cfg: dict, store: Store | None = None, candles: CandleStore | None = None):
        self.cfg = cfg
        self.store = store if store is not None else Store(home_path(cfg, "fibbot.sqlite"))
        self.candles = candles if candles is not None else CandleStore(cfg)
        self.cycle: str | None = None          # kind of the running brain cycle, None outside cycles
        self._left: dict | None = None         # remaining budget inside a cycle
        self._weekly_capped = False            # the cycle's validations were cut by the weekly cap
        self._universe: tuple[float, list[str]] | None = None
        self._memo: OrderedDict = OrderedDict()
        self._edge_cache: dict = {}             # graduation.paper_edge results
        self._paper: PaperTrader | None = None

    # -- data ----------------------------------------------------------------------------------

    def universe(self, refresh: bool = False) -> list[str]:
        """data.universe.get_universe(cfg), reused for UNIVERSE_TTL seconds within this Lab. Every
        symbol seen is remembered (kv "universe_seen") for the point-in-time validation pool."""
        now = time.monotonic()
        if refresh or self._universe is None or now - self._universe[0] > UNIVERSE_TTL:
            self._universe = (now, list(get_universe(self.cfg, refresh=refresh)))
            seen = list(self.store.get_kv(SEEN_KEY, []) or [])
            new = [s for s in self._universe[1] if s not in seen]
            if new:
                self.store.set_kv(SEEN_KEY, seen + new)
        return list(self._universe[1])

    def bars(self, tf: str, symbols: list[str] | None = None) -> dict[str, Bars]:
        """{symbol: Bars} of cached closed candles (no network) for the universe or `symbols`;
        symbols without data are left out. Served from the CandleStore's per (symbol, tf)
        in-memory cache, which is invalidated when a cache file changes on disk."""
        if tf not in TF_SECONDS:
            raise ValueError(f"unknown timeframe {tf!r}; expected one of {sorted(TF_SECONDS)}")
        if isinstance(symbols, str):
            symbols = [symbols]
        names = self.universe() if symbols is None else [str(s).upper() for s in symbols]
        out = {}
        for sym in dict.fromkeys(names):
            b = self.candles.get(sym, tf)
            if len(b):
                out[sym] = b
        return out

    def windows(self, tf: str = "4h") -> dict:
        """research.compute_windows over the universe's cached bars of tf (ValueError when there
        are none or the research window would be empty)."""
        return self._windows(tf, self.bars(tf))

    def validation_epoch(self) -> dict | None:
        """The pinned holdout {"validation_end", "t", "reason"} (set at the first promotion), or
        None while the windows still slide with the data."""
        epoch = self.store.get_kv(EPOCH_KEY)
        return epoch if isinstance(epoch, dict) and epoch.get("validation_end") is not None else None

    def roll_validation_epoch(self, now_t: int | None = None) -> dict:
        """Move the pinned holdout forward by research.validation_months (never past today), so
        the next validation_months of data become the new locked validation window and the old
        one joins the research window. Journaled (kind "system"): the new holdout overlaps paper
        results the brain has seen, so this is a deliberate operator action. ValueError when no
        epoch is pinned yet."""
        epoch = self.validation_epoch()
        if epoch is None:
            raise ValueError("no validation epoch is pinned yet (it is set at the first promotion);"
                             " the windows still slide with the data")
        now = _now(now_t)
        months = self.cfg["research"]["validation_months"]
        old = int(epoch["validation_end"])
        new = int(old + months * research.MONTH_DAYS * DAY)
        new = min(new - new % DAY, now - now % DAY)
        if new <= old:
            raise ValueError(f"the validation epoch {iso_day(old)} cannot move forward before"
                             f" {iso_day(old + DAY)}")
        out = {"validation_end": new, "t": now, "reason": f"rolled from {iso_day(old)}",
               "previous": old}
        self.store.set_kv(EPOCH_KEY, out)
        self.store.add_journal(
            "system", f"Validation epoch rolled: holdout now ends {iso_day(new)}",
            f"The locked validation window now ends {iso_day(new)} (was {iso_day(old)}). Data up to"
            f" {months} months before it is research data. The new holdout covers forward data that"
            " paper results (seen by the brain) came from, so treat early validations on it with"
            " care.", now_t=now)
        self._memo.clear()
        return out

    def _windows(self, tf: str, bars: dict[str, Bars]) -> dict:
        """research.compute_windows over bars, pinned to the validation epoch once one is set
        (bars after it are forward / paper data only)."""
        if not bars:
            raise ValueError(f"no cached {tf} candles for the universe; download them first"
                             " (python -m fibbot data)")
        epoch = self.validation_epoch()
        w = research.compute_windows(self.cfg, bars,
                                     validation_end=None if epoch is None else int(epoch["validation_end"]))
        if w.get("pinned") and w["validation_end"] <= w["validation_start"]:
            raise ValueError(f"the pinned validation window starting {iso_day(w['validation_start'])}"
                             f" has no {tf} data (the data ends {iso_day(w['data_end'])})")
        if w["research_end"] <= w["research_start"]:
            raise ValueError(f"{tf} history ({iso_day(w['data_start'])}..{iso_day(w['data_end'])}) is"
                             f" too short for a research window before the"
                             f" {self.cfg['research']['validation_months']}-month validation window")
        return w

    def _data(self, tf: str, symbols=None) -> tuple[dict[str, Bars], dict]:
        """(bars, windows) for a research call; windows always come from the universe bars."""
        universe_bars = self.bars(tf)
        windows = self._windows(tf, universe_bars)
        bars = universe_bars if symbols is None else self.bars(tf, symbols)
        if not bars:
            raise ValueError(f"no cached {tf} candles for symbols {symbols}")
        return bars, windows

    def _pit_pool(self, tf: str, universe: list[str]) -> list[str]:
        """Candidates of the point-in-time validation universe: today's universe, every symbol
        seen in an earlier universe, universe.static_fallback and the CandleStore's cached tf
        files; quote pairs whose base is not excluded, first-seen order."""
        u = self.cfg["universe"]
        names = [*universe, *(self.store.get_kv(SEEN_KEY, []) or []), *u.get("static_fallback", [])]
        if isinstance(self.candles, CandleStore):
            suffix = f"_{tf}.csv.gz"
            cache_dir = self.candles.path(f"BTC{u.get('quote', 'USDT')}".upper(), tf).parent
            names += sorted(p.name[:-len(suffix)] for p in cache_dir.glob("*" + suffix))
        quote, excluded = str(u.get("quote", "USDT")).upper(), excluded_bases(self.cfg)
        return [s for s in dict.fromkeys(str(n).upper() for n in names)
                if s.isalnum() and s.endswith(quote) and len(s) > len(quote)
                and s[:-len(quote)] not in excluded]

    def validation_bars(self, tf: str) -> tuple[dict[str, Bars], dict, dict]:
        """(bars, windows, info) of the validation universe of tf.

        research.validation_universe "point_in_time" (default): the universe.top_n candidates of
        _pit_pool with the most quote volume (sum of close x volume) over the PIT_VOLUME_DAYS
        before validation_start, among those with bars then and at least universe.min_history_days
        of history before it (a series that starts at the beginning of the data counts as old
        enough). Today's ranking would pick coins BECAUSE they rallied during the validation year.
        "current": today's universe. Falls back to the universe when no candidate qualifies.
        info = {"mode", "ranked_at", "symbols", "dropped" (universe symbols left out), "added"}."""
        universe_bars = self.bars(tf)
        windows = self._windows(tf, universe_bars)
        mode = str(self.cfg["research"].get("validation_universe", "point_in_time"))
        if mode not in ("point_in_time", "current"):
            raise ValueError("research.validation_universe must be 'point_in_time' or 'current',"
                             f" got {mode!r}")
        vs = windows["validation_start"]
        bars = universe_bars
        if mode == "point_in_time":
            u = self.cfg["universe"]
            oldest = max(windows["data_start"], vs - int(u.get("min_history_days", 0)) * DAY)
            ranked = []
            for k, sym in enumerate(self._pit_pool(tf, list(universe_bars))):
                b = universe_bars.get(sym)
                if b is None:
                    try:
                        b = self.candles.get(sym, tf)
                    except ValueError:         # a malformed cached name is not a candidate
                        continue
                if not len(b) or b.t[0] > oldest:
                    continue
                i0, i1 = b.index_at_or_after(vs - PIT_VOLUME_DAYS * DAY), b.index_at_or_after(vs)
                vol = math.fsum(b.c[i] * b.v[i] for i in range(i0, i1))
                if vol > 0 and math.isfinite(vol):
                    ranked.append((-vol, k, sym, b))
            ranked.sort()          # most volume first; then pool order (universe order is kept below)
            picked = {sym: b for _, _, sym, b in sorted(ranked[:int(u["top_n"])], key=lambda r: r[1])}
            if picked:
                bars = picked
            else:
                mode = "current (no point-in-time candidate)"
        info = {"mode": mode, "ranked_at": iso_day(vs), "symbols": list(bars),
                "dropped": [s for s in universe_bars if s not in bars],
                "added": [s for s in bars if s not in universe_bars]}
        return bars, windows, info

    # -- budgets -------------------------------------------------------------------------------

    def weekly_validations_left(self, now_t: int | None = None) -> int | None:
        """research.max_validations_per_week minus the validations recorded in the last 7 days
        (any source); None when the cap is disabled (null). Brain cycles and the autopilot obey
        it, so the locked window cannot be looked at more often by running more cycles."""
        cap = self.cfg["research"].get("max_validations_per_week", MAX_VALIDATIONS_PER_WEEK)
        if cap is None:
            return None
        used = self.store.count_experiments(kind="validate", since_t=_now(now_t) - WEEK)
        return max(0, int(cap) - used)

    def begin_cycle(self, kind: str, now_t: int | None = None) -> None:
        """Start a budgeted brain cycle: counters reset from cfg.research ("review" cycles get
        1/REVIEW_UNITS_DIVISOR of the backtest units and at most one validation); validations are
        further capped by weekly_validations_left()."""
        r = self.cfg["research"]
        units, vals = int(r["max_backtest_units_per_cycle"]), int(r["max_validations_per_cycle"])
        if kind == "review":
            units, vals = max(1, units // REVIEW_UNITS_DIVISOR), min(1, vals)
        weekly = self.weekly_validations_left(now_t)
        self._weekly_capped = weekly is not None and weekly < vals
        if self._weekly_capped:
            vals = weekly
        self.cycle = kind
        self._left = {"backtest_units_left": units, "validations_left": vals}

    def end_cycle(self) -> None:
        """Back to unbudgeted CLI mode."""
        self.cycle, self._left = None, None

    def budget(self) -> dict:
        """{"backtest_units_left", "validations_left"}; None = unlimited (outside a cycle)."""
        if self._left is None:
            return {"backtest_units_left": None, "validations_left": None}
        return dict(self._left)

    @property
    def source(self) -> str:
        """Experiment source: "brain" inside a cycle, else "cli"."""
        return "cli" if self.cycle is None else "brain"

    @contextmanager
    def _charge(self, units: int = 0, validations: int = 0):
        """Take budget for one call (inside a cycle only); it is refunded if the call raises."""
        left = self._left
        if left is not None:
            for key, need in (("backtest_units_left", units), ("validations_left", validations)):
                if need > left[key]:
                    weekly = key == "validations_left" and self._weekly_capped
                    raise BudgetExceeded(
                        f"{self.cycle} cycle budget: this call needs {need}"
                        f" {key.removesuffix('_left').replace('_', ' ')}, {left[key]} left"
                        + (" (research.max_validations_per_week is used up for the last 7 days)"
                           if weekly else ""))
            left["backtest_units_left"] -= units
            left["validations_left"] -= validations
        try:
            yield
        except BaseException:
            self._refund(units, validations)
            raise

    def _refund(self, units: int = 0, validations: int = 0) -> None:
        if self._left is not None:
            self._left["backtest_units_left"] += units
            self._left["validations_left"] += validations

    # -- research ------------------------------------------------------------------------------

    def _evaluate(self, spec: StrategySpec, window: str, symbols=None) -> tuple[BacktestResult, dict]:
        """research.evaluate on a window, reusing one of the last MEMO_SIZE identical evaluations
        (same spec, name, window bounds and symbols)."""
        bars, windows = self._data(spec.timeframe, symbols)
        start, end = research.window_bounds(windows, window)
        key = (spec.spec_id(), spec.name, window, tuple(bars), start, end)
        if key in self._memo:
            self._memo.move_to_end(key)
            return self._memo[key]
        self._memo[key] = research.evaluate(spec, bars, self.cfg, windows, window)
        while len(self._memo) > MEMO_SIZE:
            self._memo.popitem(last=False)
        return self._memo[key]

    def _trial(self, spec: StrategySpec, result: BacktestResult) -> None:
        """Count a full-universe research-window evaluation as a trial with its per-trade Sharpe
        and closed-trade count (on_combo callback signature)."""
        self.store.record_trial(result.spec_id, result.metrics["sharpe_per_trade"],
                                n=result.metrics["n"])

    def _record(self, kind: str, spec: StrategySpec | None, start, end, metrics: dict,
                notes: str = "", source: str | None = None) -> int:
        """Store an experiment; metrics gain "engine_version" (strategy.ENGINE_VERSION): results
        of different fill/accounting rules are not comparable, even for one spec_id."""
        return self.store.add_experiment(kind, None if spec is None else spec.to_dict(),
                                         None if spec is None else spec.spec_id(), start, end,
                                         {**metrics, "engine_version": ENGINE_VERSION},
                                         source=source or self.source, notes=notes)

    # -- holdout looks -------------------------------------------------------------------------

    def holdout_looks(self, spec_id: str) -> list[dict]:
        """Ledger of the evaluations that looked at spec_id's held-out period, oldest first:
        [{"t", "kind" ("validate" | "backtest"), "window", "start", "end", "symbols",
        "experiment_id", "passed" (validate only)}]."""
        return list(self.store.get_kv(LOOKS_KEY + spec_id, []) or [])

    def _look(self, spec_id: str, kind: str, window: str, result: BacktestResult, eid: int,
              passed: bool | None = None) -> None:
        looks = self.holdout_looks(spec_id)
        looks.append({"t": _now(None), "kind": kind, "window": window, "start": result.start_t,
                      "end": result.end_t, "symbols": list(result.symbols), "experiment_id": eid,
                      "passed": passed})
        self.store.set_kv(LOOKS_KEY + spec_id, looks)

    def _count_look(self, spec_id: str, result: BacktestResult) -> tuple[int, bool]:
        """Count a holdout evaluation as a trial: the spec itself once, and every further
        DISTINCT look (another window or symbol set: a new draw from the held-out period) as an
        extra trial "<spec_id>#look<k>", so repeated looks deflate the Sharpe hurdle. No Sharpe is
        recorded (var_sharpe stays research-window only). Returns (distinct looks incl. this one,
        whether this one is new)."""
        seen = {_look_key(lk["start"], lk["end"], lk["symbols"]) for lk in self.holdout_looks(spec_id)}
        new = _look_key(result.start_t, result.end_t, result.symbols) not in seen
        self.store.record_trial(spec_id, None)
        k = len(seen) + new
        if new and k > 1:
            self.store.record_trial(f"{spec_id}#look{k}", None)
        return k, new

    def prior_validation(self, spec: StrategySpec | dict) -> dict | None:
        """The newest validate experiment of spec whose window overlaps the current validation
        window of its timeframe (its verdict stands: validate() refuses a fresh look until the
        validation window has moved past it), else None. A validation recorded by an older
        engine version does not count (see _check_relook)."""
        spec = as_spec(spec)
        _, windows = self._data(spec.timeframe)
        v = self.store.latest_validation(spec.spec_id())
        if v is None or v["window_end"] is None or v["window_end"] <= windows["validation_start"]:
            return None
        if _engine_of(v.get("metrics")) != ENGINE_VERSION:
            return None             # an older engine's verdict is not comparable: validate afresh
        return v

    @staticmethod
    def _metrics(summary: dict, extras: dict) -> dict:
        """Experiment metrics of an evaluation: rounded trade/equity stats + fold stability."""
        return {**summary["metrics"], "fold_positive_frac": extras["fold_positive_frac"],
                "min_fold_avg_r": extras["min_fold_avg_r"], "folds": extras["folds"]}

    def backtest(self, spec: StrategySpec | dict, window: str = "research", symbols=None,
                 trades: int = 0) -> dict:
        """Evaluate spec on a window ("research" | "validation" | "all"; only "research" inside a
        cycle). 1 unit. Returns summary + folds + {"window", "experiment_id"}, plus "trades": the
        last `trades` trades (closed and open, by entry) as compact rows when trades > 0.

        Research window: a full-universe run is a trial with its Sharpe; a symbol-subset run
        counts as a trial but keeps the spec's universe Sharpe. "validation" (on the validation
        universe of validation_bars() unless symbols are given) and "all" are holdout peeks: each
        counts as a trial and a look at the spec's holdout (see holdout_looks), the experiment is
        recorded with source "cli-holdout" and its numbers nested under metrics["holdout"]
        (not top-level), and "holdout_looks" in the result counts the spec's distinct looks."""
        spec = as_spec(spec)
        if self.cycle is not None and window != "research":
            raise ValueError("only the research window is open during a brain cycle;"
                             " the validation window is reached through validate()")
        with self._charge(units=1):
            if window == "validation" and symbols is None:
                symbols = list(self.validation_bars(spec.timeframe)[0])
            result, extras = self._evaluate(spec, window, symbols)
        summary = result.summary()
        notes = f"window={window}" + ("" if symbols is None else f" symbols={','.join(result.symbols)}")
        metrics = self._metrics(summary, extras)
        out = {**summary, **extras, "window": window}
        if window == "research":
            if symbols is None or set(result.symbols) == set(self.bars(spec.timeframe)):
                self._trial(spec, result)
            else:              # a subset is a narrower sample: it must not replace the universe Sharpe
                self.store.record_trial(result.spec_id, None)
            eid = self._record("backtest", spec, result.start_t, result.end_t, metrics, notes)
        else:
            looks, _ = self._count_look(result.spec_id, result)
            self.store.record_look(_look_id(result.spec_id, result), result.spec_id, "peek",
                                   result.start_t, result.end_t,
                                   result.metrics["sharpe_per_trade"] if window == "validation" else None,
                                   n=result.metrics["n"] if window == "validation" else None)
            eid = self._record("backtest", spec, result.start_t, result.end_t,
                               {"window": window, "holdout_looks": looks, "holdout": {"result": metrics}},
                               notes + " (holdout peek: a trial and a look at the locked period)",
                               source=HOLDOUT_SOURCE)
            self._look(result.spec_id, "backtest", window, result, eid)
            self.store.set_kv(PEEKS_KEY, int(self.store.get_kv(PEEKS_KEY, 0) or 0) + 1)
            out["holdout_looks"] = looks
        out["experiment_id"] = eid
        if trades > 0:
            out["trades"] = [_trade_row(tr) for tr in result.trades[-int(trades):]]
        return out

    def optimize(self, base_spec: StrategySpec | dict, grid: dict) -> dict:
        """research.walk_forward_optimize on the research window. Units = grid combos; combos that
        were not evaluated (invalid or duplicate) are refunded. Every distinct outcome is a trial (a
        combo that trades exactly like an earlier one is evaluated but not counted again)."""
        base = as_spec(base_spec)
        if not isinstance(grid, dict):
            raise ValueError("grid must be an object {param: [values, ...]}")
        units = math.prod(len(v) if isinstance(v, (list, tuple)) else 1 for v in grid.values())
        with self._charge(units=units):
            bars, windows = self._data(base.timeframe)
            out = research.walk_forward_optimize(base, grid, bars, self.cfg, windows, on_combo=self._trial)
        self._refund(units - out["n_evaluated"])
        oos = out["oos"]
        start, end = research.window_bounds(windows, "research")
        eid = self._record("optimize", base, start, end,
                           {**out, "n": oos["n"], "avg_r": oos["avg_r"],
                            "profit_factor": oos["profit_factor"]},
                           "grid=" + json.dumps(grid, default=str, separators=(",", ":"))[:500])
        return {**out, "experiment_id": eid}

    def baseline(self, spec: StrategySpec | dict, n_runs: int = 10) -> dict:
        """research.random_baseline on the research window (1 unit, 1..MAX_BASELINE_RUNS runs)."""
        spec = as_spec(spec)
        if not 1 <= int(n_runs) <= MAX_BASELINE_RUNS:
            raise ValueError(f"n_runs must be 1..{MAX_BASELINE_RUNS}, got {n_runs}")
        with self._charge(units=1):
            result, _ = self._evaluate(spec, "research")
            out = research.random_baseline(spec, self.bars(spec.timeframe, result.symbols), self.cfg,
                                           result.start_t, result.end_t, n_runs=int(n_runs),
                                           result=result)
        self._trial(spec, result)
        eid = self._record("baseline", spec, result.start_t, result.end_t, out)
        return {"spec_id": result.spec_id, **out, "experiment_id": eid}

    def robustness(self, spec: StrategySpec | dict) -> dict:
        """research.robustness over the whole one-step neighbourhood (up to ROBUSTNESS_NEIGHBOURS);
        units = neighbours + 1 (the base spec is evaluated too, to spot no-op moves). Every
        distinct neighbour is a trial; a neighbour that trades exactly like the base or an earlier
        neighbour is flagged duplicate_of and is not."""
        spec = as_spec(spec)
        n_nb = min(len(research.neighbours(spec)), ROBUSTNESS_NEIGHBOURS)
        units = n_nb + 1 if n_nb else 0
        with self._charge(units=units):
            bars, windows = self._data(spec.timeframe)
            out = research.robustness(spec, bars, self.cfg, windows,
                                      max_neighbours=ROBUSTNESS_NEIGHBOURS, on_combo=self._trial)
        start, end = research.window_bounds(windows, "research")
        eid = self._record("robustness", spec, start, end, out)
        return {"spec_id": spec.spec_id(), **out, "experiment_id": eid}

    def _check_relook(self, spec_id: str, start: int, end: int, symbols: list[str]) -> None:
        """ValueError when spec_id already has a validation whose window overlaps [start, end)
        and is not the identical window and symbol set (that would re-draw the locked period)."""
        v = self.store.latest_validation(spec_id)
        if v is None or v["window_end"] is None or v["window_end"] <= start:
            return
        if _engine_of(v.get("metrics")) != ENGINE_VERSION:
            return       # recorded by older fill rules: not the same test (still a new look/trial)
        prev = (v.get("metrics") or {}).get("symbols")
        if (v["window_start"], v["window_end"]) == (start, end) and (
                prev is None or sorted(prev) == sorted(symbols)):
            return                                     # identical draw: a deterministic replay
        verdict = (v.get("metrics") or {}).get("passed")
        raise ValueError(
            f"spec {spec_id} was already validated on an overlapping window (experiment {v['id']},"
            f" {iso_day(v['window_start'])}..{iso_day(v['window_end'])}:"
            f" {'PASS' if verdict is True else 'FAIL'}). One look per validation period: a new"
            f" validation of this spec is possible once the validation window starts on or after"
            f" {iso_day(v['window_end'])}; until then that verdict stands. A changed parameter is a"
            f" new spec (and a new trial)")

    def validate(self, spec: StrategySpec | dict) -> dict:
        """Evaluate on the locked validation window (point-in-time symbols, see validation_bars)
        and apply research.validation_gate. Consumes one validation.

        The gate's deflated Sharpe uses the holdout's own reuse (dsr_basis "holdout"): the
        number of distinct looks at validation windows overlapping this one, any spec, this look
        included (store.holdout_stats: validations and CLI peeks), and var = max(1 / n_eff,
        variance of the recorded validation Sharpes). Research-window trials never touched the
        holdout, so they do not select on it (their count, trial_stats()["n"], is reported as
        n_trials). It also gets a random-entry baseline (VALIDATION_BASELINE_RUNS runs on the
        same window, uniform and regime-filtered): mean R must beat it by
        min_validation_edge_r with a z score of at least min_validation_edge_z.

        ValueError when the spec's recorded validation window overlaps this one (see
        _check_relook). Recorded as kind "validate" with metrics incl. "passed", "checks",
        "deflated_sharpe", "baseline", "symbols", "validation_universe", "holdout_looks", "replay",
        the deflation inputs and "engine" (paper.engine_fingerprint())."""
        spec = as_spec(spec)
        sid = spec.spec_id()
        with self._charge(validations=1):
            bars, windows, pit = self.validation_bars(spec.timeframe)
            start, end = research.window_bounds(windows, "validation")
            self._check_relook(sid, start, end, list(bars))
            result, extras = self._evaluate(spec, "validation", list(bars))
            base = research.random_baseline(spec, bars, self.cfg, result.start_t, result.end_t,
                                            n_runs=VALIDATION_BASELINE_RUNS, result=result)
        looks, new_look = self._count_look(sid, result)
        self.store.record_look(_look_id(sid, result), sid, "validate", result.start_t, result.end_t,
                               result.metrics["sharpe_per_trade"], n=result.metrics["n"])
        min_trades = int(self.cfg["research"]["min_trades"])
        ts = self.store.trial_stats(min_trades=min_trades)
        hs = self.store.holdout_stats(start, end, min_trades=min_trades)
        gate = research.validation_gate(result, self.cfg["research"]["validation_gate"], ts["n"],
                                        ts["var_sharpe"], baseline=base, holdout_looks=hs["k"],
                                        var_holdout_sr=hs["var_sharpe"])
        summary = result.summary()
        trials = {"n_trials": ts["n"], "var_sr": gate["dsr_var"], "dsr_basis": gate["dsr_basis"],
                  "dsr_trials": gate["dsr_trials"], "holdout_k": hs["k"], "sr_star": gate["sr_star"],
                  "n_eff": gate["n_eff"], "t_clustered": gate["t_clustered"],
                  "var_sharpe_research": rounded(ts["var_sharpe"]), "n_var": ts["n_var"]}
        baseline = {k: base.get(k) for k in ("n_runs", "baseline_avg_r_mean", "baseline_avg_r_std",
                                             "beats_frac", "edge_r", "edge_z", "regime_filtered",
                                             "baseline_regime_avg_r_mean", "edge_regime_r",
                                             "edge_regime_z")}
        extra = {"baseline": baseline, "validation_universe": {k: v for k, v in pit.items() if k != "symbols"},
                 "holdout_looks": looks, "replay": not new_look}
        eid = self._record("validate", spec, result.start_t, result.end_t,
                           {**self._metrics(summary, extras), "passed": gate["passed"],
                            "checks": gate["checks"], "deflated_sharpe": gate["deflated_sharpe"],
                            **trials, **extra, "symbols": list(result.symbols),
                            "engine": engine_fingerprint()})
        self._look(sid, "validate", "validation", result, eid, passed=gate["passed"])
        return {**summary, **extras, "window": "validation", "passed": gate["passed"], "gate": gate,
                **trials, **extra, "experiment_id": eid}

    def research_baseline(self, spec_id: str) -> dict | None:
        """The newest research-window baseline experiment of spec_id recorded by this engine
        version (its metrics), else None."""
        for r in self.store.list_experiments(limit=None, kind="baseline", spec_id=spec_id):
            if _engine_of(r["metrics"]) == ENGINE_VERSION:
                return {**r["metrics"], "experiment_id": r["id"]}
        return None

    def promote(self, spec: StrategySpec | dict, name: str, rationale: str, now_t: int | None = None) -> dict:
        """Put a validated spec on paper as strategy_id = spec_id, starting at the next bar of its
        timeframe. ValueError when it is already on the roster (any status, retired included:
        retirement is final), has no passing latest validation, its latest validation was
        recorded by another engine version (re-validate), it failed any validation recorded in
        the last research.validation_months (one pass after a failed look is not a pass), it has
        no research-window baseline (research_baseline) beating random entries on both draws
        (min(edge_r, edge_regime_r) > 0, the autopilot's rule), or the roster already holds
        paper.max_roster paper strategies.

        On success the paper account settings are frozen now (paper.account_settings), and the
        first promotion pins the validation epoch (validation_epoch): from then on the locked
        validation window no longer slides over the forward data paper trades on."""
        spec = as_spec(spec)
        sid, name = spec.spec_id(), str(name).strip() or spec.name
        row = self.store.get_strategy(sid)
        if row is not None and row["status"] == "retired":
            raise ValueError(f"strategy {sid} was retired on {iso_day(row['status_t'])}"
                             f" ({row['status_reason'] or 'no reason given'}); retirement is final and"
                             " its paper record stands. Promote a variant instead (a changed parameter"
                             " gives a new spec_id)")
        if row is not None:
            raise ValueError(f"spec {sid} is already on the roster as {row['name']!r} (status {row['status']})")
        val = self.store.latest_validation(sid)
        if val is None:
            raise ValueError(f"spec {sid} has no validation on record; run validate first")
        metrics = val["metrics"] or {}
        if metrics.get("passed") is not True:
            failed = [c["name"] for c in metrics.get("checks", []) if not c.get("passed")]
            raise ValueError(f"the latest validation of spec {sid} (experiment {val['id']}) did not pass"
                             + (f": {', '.join(failed)}" if failed else ""))
        if _engine_of(metrics) != ENGINE_VERSION:
            raise ValueError(f"the engine changed since this validation (experiment {val['id']} used"
                             f" engine version {_engine_of(metrics)}, now {ENGINE_VERSION}); re-validate")
        base = self.research_baseline(sid)
        if base is None:
            raise ValueError(f"spec {sid} has no research-window baseline: run run_baseline (python -m"
                             " fibbot baseline) first; promotion needs an edge over random entries")
        edge = min(float(base.get("edge_r") or 0.0),
                   float(base.get("edge_regime_r", base.get("edge_r")) or 0.0))
        if not edge > 0:
            raise ValueError(f"spec {sid} does not beat random entries on the research window"
                             f" (baseline experiment {base['experiment_id']}: min(edge_r, edge_regime_r)"
                             f" = {edge:+.3f} R)")
        now = _now(now_t)
        months = self.cfg["research"]["validation_months"]
        since = now - months * research.MONTH_DAYS * DAY
        bad = [lk for lk in self.holdout_looks(sid)
               if lk["kind"] == "validate" and lk.get("passed") is False and lk["t"] >= since]
        if bad:
            raise ValueError(f"spec {sid} failed validation (experiment {bad[-1]['experiment_id']},"
                             f" {iso_day(bad[-1]['t'])}) within the last {months} months; every"
                             " validation of a spec in that period must pass")
        cap, on_paper = int(self.cfg["paper"]["max_roster"]), len(self.store.roster("paper"))
        if on_paper >= cap:
            raise ValueError(f"the paper roster is full ({on_paper}/{cap}); retire a strategy first")
        step = TF_SECONDS[spec.timeframe]
        start = (now // step + 1) * step
        validation = {"experiment_id": val["id"], "created_t": val["created_t"],
                      "window_start": val["window_start"], "window_end": val["window_end"],
                      "metrics": metrics}
        self.store.add_to_roster(sid, name, spec.with_overrides(name=name).to_dict(), start,
                                 validation, rationale, now_t=now)
        # freeze the account settings at promotion (not at the first tick), and pin the holdout
        snap = account_settings(self.store, self.cfg, self.store.get_strategy(sid))
        self.store.set_kv(ACCOUNT_KEY.format(sid), {**snap, "t": now})
        if self.validation_epoch() is None:
            end = start
            try:                   # never past the data: the holdout must hold the newest bars
                end = min(end, self.windows(spec.timeframe)["data_end"])
            except ValueError:
                pass
            self.store.set_kv(EPOCH_KEY, {"validation_end": end - end % DAY, "t": now,
                                          "reason": f"first promotion ({sid})"})
            self._memo.clear()
        self.store.add_journal(
            "promotion", f"Promoted {name} ({sid}) to paper",
            f"{rationale}\n\nPaper trading starts {iso_time(start)} on {spec.timeframe}. Validation"
            f" (experiment {val['id']}): n={metrics.get('n')}, mean R={metrics.get('avg_r')},"
            f" PF={metrics.get('profit_factor')}, deflated Sharpe={metrics.get('deflated_sharpe')}.",
            now_t=now)
        return {"strategy_id": sid, "name": name, "timeframe": spec.timeframe,
                "paper_start_t": start, "paper_start": iso_time(start),
                "validation_experiment_id": val["id"], "roster_size": on_paper + 1, "max_roster": cap}

    def retire(self, strategy_id: str, reason: str) -> dict:
        """Stop paper trading a strategy: retirement is final; its paper record is kept (the spec
        can never be promoted again; a variant with a changed parameter is a new strategy).
        KeyError if unknown, ValueError if already retired."""
        row = self.store.get_strategy(strategy_id)
        if row is None:
            raise KeyError(f"unknown strategy {strategy_id}")
        if row["status"] == "retired":
            raise ValueError(f"strategy {strategy_id} is already retired")
        self.store.set_strategy_status(strategy_id, "retired", reason)
        acct = self.paper().account(strategy_id)
        st = acct["stats"]
        self.store.add_journal(
            "retirement", f"Retired {row['name']} ({strategy_id})",
            f"{reason}\n\nAfter {acct['days']:.1f} days on paper: {acct['n_closed']} closed trades,"
            f" mean R {st['avg_r']:+.3f}, PF {st['profit_factor']:.2f}, return {acct['return']:+.2%},"
            f" max drawdown {acct['max_drawdown']:.1%}.")
        return {"strategy_id": strategy_id, "name": row["name"], "status": "retired",
                "reason": reason, "account": acct}

    # -- paper / reporting ---------------------------------------------------------------------

    def paper(self) -> PaperTrader:
        if self._paper is None:
            self._paper = PaperTrader(self.cfg, self.store, self.candles)
        return self._paper

    def strategy_bars(self, row: dict) -> dict[str, Bars]:
        """Cached candles of a roster strategy's timeframe for the symbols it trades: its paper
        universe (paper.UNIVERSE_KEY; today's universe before its first tick) and every symbol
        of its paper trades. Unreadable symbols are left out."""
        tf = StrategySpec.from_dict(row["spec"]).timeframe
        kv = self.store.get_kv(UNIVERSE_KEY.format(row["strategy_id"]))
        names = list((kv or {}).get("symbols") or {}) if isinstance(kv, dict) else []
        if not names:
            try:
                names = self.universe()
            except Exception:  # a failing ranking must not break the graduation report
                names = []
        names += [r["symbol"] for r in self.store.paper_trades(row["strategy_id"])]
        out = {}
        for sym in dict.fromkeys(names):
            try:
                b = self.candles.get(sym, tf)
            except Exception:
                continue
            if len(b):
                out[sym] = b
        return out

    def graduation(self, strategy_id: str, now_t: int | None = None) -> dict:
        """graduation.evaluate with the strategy's candles (strategy_bars): paper days count only
        up to its newest bar, and the paper record must beat random entries over the paper
        period. KeyError for an unknown strategy."""
        row = self.store.get_strategy(strategy_id)
        if row is None:
            raise KeyError(strategy_id)
        return graduation_evaluate(self.store, self.cfg, strategy_id, now_t=now_t,
                                   bars_by_symbol=self.strategy_bars(row), cache=self._edge_cache)

    def _roster_entry(self, row: dict, now: int) -> dict:
        """Status view of one roster strategy: its paper account, the validation it was promoted
        on (what paper is compared with) and its graduation checks as compact
        [name, value, threshold, passed] rows with the anytime-valid interval of mean R and the
        clustering evidence (n_eff, design effect, weeks, clustered t)."""
        sid = row["strategy_id"]
        acct, grad = self.paper().account(sid, now_t=now), self.graduation(sid, now_t=now)
        st = acct["stats"]
        val = row.get("validation") or {}
        vm = val.get("metrics") or {}
        entry = rounded({
            "strategy_id": sid, "name": row["name"], "status": row["status"],
            "timeframe": acct["timeframe"], "paper_start": iso_time(row["paper_start_t"]),
            "account": {"days": acct["days"], "equity": acct["equity"], "return": acct["return"],
                        "n_closed": acct["n_closed"], "n_open": acct["n_open"],
                        "avg_r": st["avg_r"], "total_r": st["total_r"], "win_rate": st["win_rate"],
                        "profit_factor": st["profit_factor"], "max_drawdown": acct["max_drawdown"]},
            "validation": {"experiment_id": val.get("experiment_id"),
                           "window": [iso_day(val.get("window_start")), iso_day(val.get("window_end"))],
                           **{k: vm.get(k) for k in ("n", "avg_r", "profit_factor", "deflated_sharpe")}},
            "graduation": {"status": grad["status"], "summary": grad["summary"],
                           "failed": [c["name"] for c in grad["checks"] if not c["passed"]],
                           "mean_r_ci": grad.get("mean_r_ci"), "evidence": grad.get("evidence"),
                           "checks": [[c["name"], c.get("value"), c.get("threshold"), c["passed"]]
                                      for c in grad["checks"]]}})
        return {**entry, "spec": row["spec"]}    # unrounded: a copied spec keeps its spec_id

    def system_events(self, days: float = 7, now_t: int | None = None) -> dict:
        """{"n": "system" journal entries (scheduler step failures, paper-integrity warnings) in
        the last `days` days, "last": newest title, "last_t": its ISO time} (of the newest
        JOURNAL_SCAN entries)."""
        since = _now(now_t) - days * DAY
        rows = [j for j in self.store.journal(JOURNAL_SCAN, "system") if j["t"] >= since]
        return {"n": len(rows), "last": rows[0]["title"] if rows else None,
                "last_t": iso_time(rows[0]["t"]) if rows else None}

    def status(self, now_t: int | None = None) -> dict:
        """Compact JSON-safe lab overview: as_of, universe, data range and windows per timeframe,
        active roster (account, promoting validation and graduation checks per strategy), retired
        strategies, budget, the weekly validation cap, trial stats, experiment counts, CLI holdout
        peeks, system journal events (errors, warnings) of the last 7 days, last
        research/review/sweep times and the signal-review mode. data[tf] also names the symbol
        whose bars end first ("oldest_end", "oldest_symbol"), and "data_stall" holds the
        scheduler's candle-stall record ({"since", "reason"}) while the data is stalled;
        "validation_epoch" is the pinned holdout (None while the windows slide)."""
        now = _now(now_t)
        rows = self.store.roster(None)
        active = [r for r in rows if r["status"] != "retired"]
        retired = sorted((r for r in rows if r["status"] == "retired"), key=lambda r: r["status_t"])
        tfs = [tf for tf in dict.fromkeys([*self.cfg["data"]["timeframes"],
                                           *(r["spec"].get("timeframe") for r in active)])
               if tf in TF_SECONDS]
        data, windows = {}, {}
        for tf in tfs:
            bars = self.bars(tf)
            if not bars:
                continue
            ends = {sym: b.t[-1] + TF_SECONDS[tf] for sym, b in bars.items()}
            oldest = min(ends, key=ends.get)
            data[tf] = {"symbols": len(bars), "start": iso_day(min(b.t[0] for b in bars.values())),
                        "end": iso_time(max(ends.values())),
                        "oldest_end": iso_time(ends[oldest]), "oldest_symbol": oldest,
                        "oldest_lag_bars": round(max(0, now - ends[oldest]) / TF_SECONDS[tf], 1)}
            try:
                w = self._windows(tf, bars)
            except ValueError as e:
                windows[tf] = {"error": str(e)}
                continue
            windows[tf] = {name: [iso_day(w[f"{name}_start"]), iso_day(w[f"{name}_end"])]
                           for name in ("research", "validation")}
            if w.get("pinned"):
                windows[tf]["pinned"] = True
        last = {k: iso_time(self.store.get_kv(f"last_{k}_t")) for k in ("research", "review", "sweep")}
        epoch, stall = self.validation_epoch(), self.store.get_kv("data_stall")
        return {
            "as_of": iso_time(now), "universe": self.universe(), "data": data, "windows": windows,
            "roster": [self._roster_entry(r, now) for r in active],
            "roster_slots": {"paper": sum(r["status"] == "paper" for r in active),
                             "max": int(self.cfg["paper"]["max_roster"])},
            "retired": [{"strategy_id": r["strategy_id"], "name": r["name"],
                         "reason": r["status_reason"], "since": iso_day(r["status_t"])}
                        for r in retired[-10:]],
            "cycle": self.cycle, "budget": self.budget(),
            "validations_7d": {"used": self.store.count_experiments(kind="validate", since_t=now - WEEK),
                               "cap": self.cfg["research"].get("max_validations_per_week",
                                                               MAX_VALIDATIONS_PER_WEEK)},
            "trial_stats": rounded(self.store.trial_stats(min_trades=self.cfg["research"]["min_trades"])),
            "experiments": {"total": self.store.count_experiments(),
                            **{k: self.store.count_experiments(kind=k) for k in sorted(EXPERIMENT_KINDS)}},
            "holdout_peeks": int(self.store.get_kv(PEEKS_KEY, 0) or 0),
            "validation_epoch": None if epoch is None else {
                "validation_end": iso_day(epoch["validation_end"]), "since": iso_day(epoch.get("t"))},
            "data_stall": stall if isinstance(stall, dict) and stall.get("since") is not None else None,
            "system_events_7d": self.system_events(7, now),
            "last": last, "signal_review": self.cfg["brain"]["signal_review"],
        }

    def signal_review_stats(self) -> dict:
        """Value of the brain's signal reviews: closed paper trades (all strategies) grouped by the
        brain's decision on their leg (take / skip / unreviewed), with trade_stats-lite each."""
        signals = self.store.signals()
        decision = {(s["strategy_id"], s["symbol"], s["leg_id"]): s["decision"] for s in signals}
        groups: dict[str, list] = {"take": [], "skip": [], "unreviewed": []}
        for r in self.store.paper_trades(status="closed"):
            key = decision.get((r["strategy_id"], r["symbol"], r["leg_id"])) or "unreviewed"
            groups[key].append(self.store.trade_from_row(r))
        lite = {}
        for key, trades in groups.items():
            st = trade_stats(trades)
            lite[key] = {k: st[k] for k in LITE_KEYS}
        take, skip = lite["take"], lite["skip"]
        return rounded({
            "mode": self.cfg["brain"]["signal_review"], "reviewed": take["n"] + skip["n"], **lite,
            "skip_minus_take_avg_r": skip["avg_r"] - take["avg_r"] if take["n"] and skip["n"] else None,
            "n_signals": len(signals),
            "n_signals_reviewed": sum(s["decision"] is not None for s in signals)})

    # -- notebook ------------------------------------------------------------------------------

    @property
    def notebook_path(self):
        return home_path(self.cfg, NOTEBOOK_FILE)

    def _notebook_limit(self) -> int:
        return int(self.cfg["brain"]["notebook_max_chars"])

    def notebook(self) -> str:
        """The brain notebook (home/BRAIN_NOTEBOOK.md); "" when it does not exist yet."""
        try:
            return self.notebook_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def notebook_append(self, text: str, now_t: int | None = None) -> dict:
        """Append one dated section "## <ISO time UTC>\\n<text>\\n" ("## " headings inside the
        entry become "### ", so an entry is exactly one section), then drop whole oldest DATED
        sections beyond brain.notebook_max_chars. The preamble and undated sections (what
        rewrite_notebook condensed) are never dropped. An entry longer than an empty notebook
        allows is truncated to the room left; ValueError (nothing written) when the entry would
        fit an empty notebook but the condensed part leaves it no room: rewrite_notebook first.
        Returns {"notebook_chars", "limit", "free_chars", "trimmed": [dropped headings],
        "truncated"}."""
        limit, stamp = self._notebook_limit(), f"## {iso_time(_now(now_t))}\n"
        old = _with_header(self.notebook() or NOTEBOOK_HEADER + "\n").rstrip("\n") + "\n\n"
        body = re.sub(r"(?m)^## ", "### ", str(text).strip())
        head, sections = _sections(old)
        fixed = len(head) + sum(len(s) for s in sections if not _STAMPED.match(s))
        room = limit - fixed - len(stamp) - 1
        alone = limit - len(NOTEBOOK_HEADER) - 2 - len(stamp) - 1   # room in an empty notebook
        truncated = len(body) > room
        if truncated and (len(body) <= alone or room <= len(TRUNCATED)):
            raise ValueError(
                f"notebook full: this {len(body)}-char entry does not fit. {fixed} of the {limit}"
                f" chars (brain.notebook_max_chars) are the preamble and undated sections written by"
                f" rewrite_notebook, which are never trimmed, leaving {max(0, room)} chars for dated"
                f" entries. Condense the notebook with rewrite_notebook (keep it well under the"
                f" limit) or shorten the entry")
        if truncated:
            body = body[:room - len(TRUNCATED)] + TRUNCATED
        text, trimmed = _trim_sections(f"{old}{stamp}{body}\n", limit)
        atomic_write(self.notebook_path, text.encode("utf-8"))
        return {"notebook_chars": len(text), "limit": limit, "free_chars": max(0, limit - len(text)),
                "trimmed": trimmed, "truncated": truncated}

    def notebook_rewrite(self, text: str) -> dict:
        """Replace the notebook (the "# Brain notebook" header is added when missing). ValueError
        when the result exceeds brain.notebook_max_chars. Returns {"notebook_chars", "limit",
        "free_chars"} plus a "warning" when less than NOTEBOOK_LOW_ROOM_FRAC of the limit is left
        for dated entries (undated sections are never trimmed, so appends fail once it is used)."""
        text = _with_header(str(text).strip() + "\n")
        limit = self._notebook_limit()
        if len(text) > limit:
            raise ValueError(f"notebook rewrite is {len(text)} chars, over brain.notebook_max_chars"
                             f" {limit}; condense it")
        atomic_write(self.notebook_path, text.encode("utf-8"))
        out = {"notebook_chars": len(text), "limit": limit, "free_chars": limit - len(text)}
        if limit - len(text) < NOTEBOOK_LOW_ROOM_FRAC * limit:
            out["warning"] = (f"only {limit - len(text)} chars are left for appended entries; undated"
                              " sections are never trimmed, so condense further")
        return out
