"""Lab: the facade the CLI, the scheduler and the brain tools use (see DESIGN.md, "fibbot/lab.py").

Every research call records an experiment; every evaluation on the research window also records
a trial (spec_id, per-trade Sharpe) for the deflated-Sharpe correction. Between begin_cycle() and
end_cycle() (a brain cycle) calls are budgeted, experiments are recorded with source "brain", and
the validation window is only reachable through validate().
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
from .config import home_path
from .data.candles import CandleStore, atomic_write
from .data.universe import get_universe
from .graduation import evaluate as graduation_evaluate
from .metrics import trade_stats
from .paper import PaperTrader
from .store import EXPERIMENT_KINDS, Store
from .strategy import StrategySpec
from .types import TF_SECONDS, Bars

NOTEBOOK_FILE = "BRAIN_NOTEBOOK.md"
NOTEBOOK_HEADER = "# Brain notebook"
TRUNCATED = "\n[... truncated to fit brain.notebook_max_chars]"
UNIVERSE_TTL = 3600.0          # seconds a universe lookup is reused (a failing ranking is not retried per call)
MEMO_SIZE = 8                  # research evaluations kept for reuse (e.g. backtest -> baseline of one spec)
ROBUSTNESS_NEIGHBOURS = 16
MAX_BASELINE_RUNS = 100
REVIEW_UNITS_DIVISOR = 4       # review cycles get 1/4 of the research backtest units and one validation
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


def _trim_notebook(text: str, limit: int) -> str:
    """Drop whole sections ("## " headings) until text fits in limit: the oldest timestamped
    (appended) section first, else the oldest section; the preamble and newest section stay."""
    parts = re.split(r"(?m)^(?=## )", text)
    head, sections = parts[0], parts[1:]
    while len(sections) > 1 and len(head) + sum(map(len, sections)) > limit:
        del sections[next((i for i, s in enumerate(sections[:-1]) if _STAMPED.match(s)), 0)]
    return head + "".join(sections)


def _with_header(text: str) -> str:
    return text if text.startswith(NOTEBOOK_HEADER) else f"{NOTEBOOK_HEADER}\n\n{text}"


class Lab:
    """Research, paper trading and reporting over one config, store and candle cache."""

    def __init__(self, cfg: dict, store: Store | None = None, candles: CandleStore | None = None):
        self.cfg = cfg
        self.store = store if store is not None else Store(home_path(cfg, "fibbot.sqlite"))
        self.candles = candles if candles is not None else CandleStore(cfg)
        self.cycle: str | None = None          # kind of the running brain cycle, None outside cycles
        self._left: dict | None = None         # remaining budget inside a cycle
        self._universe: tuple[float, list[str]] | None = None
        self._memo: OrderedDict = OrderedDict()
        self._paper: PaperTrader | None = None

    # -- data ----------------------------------------------------------------------------------

    def universe(self, refresh: bool = False) -> list[str]:
        """data.universe.get_universe(cfg), reused for UNIVERSE_TTL seconds within this Lab."""
        now = time.monotonic()
        if refresh or self._universe is None or now - self._universe[0] > UNIVERSE_TTL:
            self._universe = (now, list(get_universe(self.cfg, refresh=refresh)))
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

    def _windows(self, tf: str, bars: dict[str, Bars]) -> dict:
        if not bars:
            raise ValueError(f"no cached {tf} candles for the universe; download them first"
                             " (python -m fibbot data)")
        w = research.compute_windows(self.cfg, bars)
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

    # -- budgets -------------------------------------------------------------------------------

    def begin_cycle(self, kind: str) -> None:
        """Start a budgeted brain cycle: counters reset from cfg.research ("review" cycles get
        1/REVIEW_UNITS_DIVISOR of the backtest units and at most one validation)."""
        r = self.cfg["research"]
        units, vals = int(r["max_backtest_units_per_cycle"]), int(r["max_validations_per_cycle"])
        if kind == "review":
            units, vals = max(1, units // REVIEW_UNITS_DIVISOR), min(1, vals)
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
                    raise BudgetExceeded(f"{self.cycle} cycle budget: this call needs {need}"
                                         f" {key.removesuffix('_left').replace('_', ' ')},"
                                         f" {left[key]} left")
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
        """Count a research-window evaluation as a trial (on_combo callback signature)."""
        self.store.record_trial(result.spec_id, result.metrics["sharpe_per_trade"])

    def _record(self, kind: str, spec: StrategySpec | None, start, end, metrics: dict,
                notes: str = "") -> int:
        return self.store.add_experiment(kind, None if spec is None else spec.to_dict(),
                                         None if spec is None else spec.spec_id(), start, end,
                                         metrics, source=self.source, notes=notes)

    @staticmethod
    def _metrics(summary: dict, extras: dict) -> dict:
        """Experiment metrics of an evaluation: rounded trade/equity stats + fold stability."""
        return {**summary["metrics"], "fold_positive_frac": extras["fold_positive_frac"],
                "min_fold_avg_r": extras["min_fold_avg_r"], "folds": extras["folds"]}

    def backtest(self, spec: StrategySpec | dict, window: str = "research", symbols=None,
                 trades: int = 0) -> dict:
        """Evaluate spec on a window ("research" | "validation" | "all"; only "research" inside a
        cycle). 1 unit. Returns summary + folds + {"window", "experiment_id"}, plus "trades": the
        last `trades` trades (closed and open, by entry) as compact rows when trades > 0."""
        spec = as_spec(spec)
        if self.cycle is not None and window != "research":
            raise ValueError("only the research window is open during a brain cycle;"
                             " the validation window is reached through validate()")
        with self._charge(units=1):
            result, extras = self._evaluate(spec, window, symbols)
        if window == "research":
            self._trial(spec, result)
        summary = result.summary()
        notes = f"window={window}" + ("" if symbols is None else f" symbols={','.join(result.symbols)}")
        eid = self._record("backtest", spec, result.start_t, result.end_t,
                           self._metrics(summary, extras), notes)
        out = {**summary, **extras, "window": window, "experiment_id": eid}
        if trades > 0:
            out["trades"] = [_trade_row(tr) for tr in result.trades[-int(trades):]]
        return out

    def optimize(self, base_spec: StrategySpec | dict, grid: dict) -> dict:
        """research.walk_forward_optimize on the research window. Units = grid combos; combos that
        were not evaluated (invalid or duplicate) are refunded. Every evaluated combo is a trial."""
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
        """research.robustness (at most ROBUSTNESS_NEIGHBOURS neighbours = units); every
        neighbour is a trial."""
        spec = as_spec(spec)
        units = min(len(research.neighbours(spec)), ROBUSTNESS_NEIGHBOURS)
        with self._charge(units=units):
            bars, windows = self._data(spec.timeframe)
            out = research.robustness(spec, bars, self.cfg, windows,
                                      max_neighbours=ROBUSTNESS_NEIGHBOURS, on_combo=self._trial)
        start, end = research.window_bounds(windows, "research")
        eid = self._record("robustness", spec, start, end, out)
        return {"spec_id": spec.spec_id(), **out, "experiment_id": eid}

    def validate(self, spec: StrategySpec | dict) -> dict:
        """Evaluate on the locked validation window and apply research.validation_gate with
        n_trials / var_sr from store.trial_stats() (the spec itself always counts as a trial).
        Consumes one validation. Recorded as kind "validate" with metrics incl. "passed"/"checks"."""
        spec = as_spec(spec)
        with self._charge(validations=1):
            result, extras = self._evaluate(spec, "validation")
        self.store.record_trial(result.spec_id, None)
        ts = self.store.trial_stats()
        gate = research.validation_gate(result, self.cfg["research"]["validation_gate"], ts["n"],
                                        ts["var_sharpe"])
        summary = result.summary()
        trials = {"n_trials": ts["n"], "var_sr": rounded(ts["var_sharpe"])}
        eid = self._record("validate", spec, result.start_t, result.end_t,
                           {**self._metrics(summary, extras), "passed": gate["passed"],
                            "checks": gate["checks"], "deflated_sharpe": gate["deflated_sharpe"],
                            **trials})
        return {**summary, **extras, "window": "validation", "passed": gate["passed"], "gate": gate,
                **trials, "experiment_id": eid}

    def promote(self, spec: StrategySpec | dict, name: str, rationale: str, now_t: int | None = None) -> dict:
        """Put a validated spec on paper as strategy_id = spec_id, starting at the next bar of its
        timeframe. ValueError when it is already on the roster, has no passing latest validation,
        or the roster already holds paper.max_roster paper strategies."""
        spec = as_spec(spec)
        sid, name = spec.spec_id(), str(name).strip() or spec.name
        row = self.store.get_strategy(sid)
        if row is not None and row["status"] != "retired":
            raise ValueError(f"spec {sid} is already on the roster as {row['name']!r} (status {row['status']})")
        val = self.store.latest_validation(sid)
        if val is None:
            raise ValueError(f"spec {sid} has no validation on record; run validate first")
        metrics = val["metrics"] or {}
        if metrics.get("passed") is not True:
            failed = [c["name"] for c in metrics.get("checks", []) if not c.get("passed")]
            raise ValueError(f"the latest validation of spec {sid} (experiment {val['id']}) did not pass"
                             + (f": {', '.join(failed)}" if failed else ""))
        cap, on_paper = int(self.cfg["paper"]["max_roster"]), len(self.store.roster("paper"))
        if on_paper >= cap:
            raise ValueError(f"the paper roster is full ({on_paper}/{cap}); retire a strategy first")
        now, step = _now(now_t), TF_SECONDS[spec.timeframe]
        start = (now // step + 1) * step
        validation = {"experiment_id": val["id"], "created_t": val["created_t"],
                      "window_start": val["window_start"], "window_end": val["window_end"],
                      "metrics": metrics}
        self.store.add_to_roster(sid, name, spec.with_overrides(name=name).to_dict(), start,
                                 validation, rationale, now_t=now)
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
        """Stop paper trading a strategy (its record is kept). KeyError if unknown, ValueError if
        already retired."""
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

    def graduation(self, strategy_id: str, now_t: int | None = None) -> dict:
        return graduation_evaluate(self.store, self.cfg, strategy_id, now_t=now_t)

    def _roster_entry(self, row: dict, now: int) -> dict:
        sid = row["strategy_id"]
        acct, grad = self.paper().account(sid, now_t=now), self.graduation(sid, now_t=now)
        st = acct["stats"]
        entry = rounded({
            "strategy_id": sid, "name": row["name"], "status": row["status"],
            "timeframe": acct["timeframe"], "paper_start": iso_time(row["paper_start_t"]),
            "account": {"days": acct["days"], "equity": acct["equity"], "return": acct["return"],
                        "n_closed": acct["n_closed"], "n_open": acct["n_open"],
                        "avg_r": st["avg_r"], "total_r": st["total_r"], "win_rate": st["win_rate"],
                        "profit_factor": st["profit_factor"], "max_drawdown": acct["max_drawdown"]},
            "graduation": {"status": grad["status"], "summary": grad["summary"],
                           "failed": [c["name"] for c in grad["checks"] if not c["passed"]]}})
        return {**entry, "spec": row["spec"]}    # unrounded: a copied spec keeps its spec_id

    def status(self, now_t: int | None = None) -> dict:
        """Compact JSON-safe lab overview: as_of, universe, data range and windows per timeframe,
        active roster (account + graduation per strategy), retired strategies, budget, trial
        stats, experiment counts, last research/review/sweep times and the signal-review mode."""
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
            data[tf] = {"symbols": len(bars), "start": iso_day(min(b.t[0] for b in bars.values())),
                        "end": iso_time(max(b.t[-1] for b in bars.values()) + TF_SECONDS[tf])}
            try:
                w = self._windows(tf, bars)
            except ValueError as e:
                windows[tf] = {"error": str(e)}
                continue
            windows[tf] = {name: [iso_day(w[f"{name}_start"]), iso_day(w[f"{name}_end"])]
                           for name in ("research", "validation")}
        last = {k: iso_time(self.store.get_kv(f"last_{k}_t")) for k in ("research", "review", "sweep")}
        return {
            "as_of": iso_time(now), "universe": self.universe(), "data": data, "windows": windows,
            "roster": [self._roster_entry(r, now) for r in active],
            "roster_slots": {"paper": sum(r["status"] == "paper" for r in active),
                             "max": int(self.cfg["paper"]["max_roster"])},
            "retired": [{"strategy_id": r["strategy_id"], "name": r["name"],
                         "reason": r["status_reason"], "since": iso_day(r["status_t"])}
                        for r in retired[-10:]],
            "cycle": self.cycle, "budget": self.budget(),
            "trial_stats": rounded(self.store.trial_stats()),
            "experiments": {"total": self.store.count_experiments(),
                            **{k: self.store.count_experiments(kind=k) for k in sorted(EXPERIMENT_KINDS)}},
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

    def notebook_append(self, text: str, now_t: int | None = None) -> None:
        """Append "## <ISO time UTC>\\n<text>\\n" (an entry too long to fit on its own is
        truncated), then drop whole oldest sections beyond brain.notebook_max_chars."""
        limit, stamp = self._notebook_limit(), f"## {iso_time(_now(now_t))}\n"
        old = _with_header(self.notebook() or NOTEBOOK_HEADER + "\n").rstrip("\n") + "\n\n"
        head = re.split(r"(?m)^(?=## )", old)[0]
        body, room = str(text).strip(), limit - len(head) - len(stamp) - 1
        if len(body) > room:
            body = body[:max(0, room - len(TRUNCATED))] + TRUNCATED
        atomic_write(self.notebook_path,
                     _trim_notebook(f"{old}{stamp}{body}\n", limit).encode("utf-8"))

    def notebook_rewrite(self, text: str) -> None:
        """Replace the notebook (the "# Brain notebook" header is added when missing). ValueError
        when the result exceeds brain.notebook_max_chars."""
        text = _with_header(str(text).strip() + "\n")
        if len(text) > self._notebook_limit():
            raise ValueError(f"notebook rewrite is {len(text)} chars, over brain.notebook_max_chars"
                             f" {self._notebook_limit()}; condense it")
        atomic_write(self.notebook_path, text.encode("utf-8"))
