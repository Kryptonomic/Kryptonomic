"""SQLite persistence (stdlib sqlite3, WAL mode) for research and paper-trading state.

Tables: experiments, trials, looks, roster, paper_trades, signals (with the brain's review), equity,
journal and kv. JSON columns (spec, metrics, validation, meta, kv values) are stored as TEXT and
decoded on read. SQLite stores NaN as NULL, so NaN prices/returns read back as ``float('nan')``.
Timestamps written by the store use ``int(time.time())`` unless a ``now_t`` kwarg is given.
Older databases are migrated in place on open (columns are only ever added).
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULTS
from .metrics import CAP
from .strategy import ENGINE_VERSION
from .types import Signal, Trade

EXPERIMENT_KINDS = frozenset({"backtest", "optimize", "baseline", "validate", "robustness", "sweep"})
JOURNAL_KINDS = frozenset({"research", "review", "signal", "system", "graduation", "promotion",
                           "retirement"})
STRATEGY_STATUSES = frozenset({"paper", "retired", "graduated"})
SIGNAL_DECISIONS = frozenset({"take", "skip"})
EXPERIMENT_SORTS = ("recent", "avg_r", "profit_factor")

_TRADE_FLOATS = ("entry_price", "stop_price", "exit_price", "ret_pct", "risk_pct", "r_multiple")
_SIGNAL_FLOATS = ("entry_price", "stop_price", "tp1", "tp2")
_LEG_KEY = "strategy_id = ? AND symbol = ? AND leg_id = ?"
TRIAL_WINSOR_MADS = 5.0     # trial sharpes beyond median +/- this many robust sigmas are clipped
# metric keys a compacted experiment keeps (top level, and one level down in nested dicts)
COMPACT_KEYS = frozenset({
    "n", "n_open", "avg_r", "total_r", "win_rate", "profit_factor", "max_drawdown",
    "sharpe_per_trade", "fold_positive_frac", "min_fold_avg_r", "edge_r", "edge_regime_r",
    "edge_z", "beats_frac", "positive_frac", "median_avg_r", "worst_avg_r", "deflated_sharpe",
    "passed", "n_combos", "n_evaluated", "window", "holdout_looks", "engine_version"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_t INTEGER NOT NULL,
    kind TEXT NOT NULL,
    spec_id TEXT,
    spec TEXT,
    window_start INTEGER,
    window_end INTEGER,
    metrics TEXT NOT NULL,
    source TEXT NOT NULL,
    notes TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_experiments_kind ON experiments (kind, created_t);
CREATE INDEX IF NOT EXISTS ix_experiments_spec ON experiments (spec_id, kind, created_t);

CREATE TABLE IF NOT EXISTS trials (
    spec_id TEXT PRIMARY KEY,
    sharpe REAL,
    n_trades INTEGER,
    engine INTEGER
);

CREATE TABLE IF NOT EXISTS looks (
    key TEXT PRIMARY KEY,
    spec_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    window_start INTEGER,
    window_end INTEGER,
    t INTEGER NOT NULL,
    sharpe REAL,
    n_trades INTEGER,
    engine INTEGER
);

CREATE TABLE IF NOT EXISTS roster (
    strategy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    spec TEXT NOT NULL,
    status TEXT NOT NULL,
    paper_start_t INTEGER NOT NULL,
    validation TEXT NOT NULL,
    rationale TEXT NOT NULL,
    added_t INTEGER NOT NULL,
    status_t INTEGER NOT NULL,
    status_reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_trades (
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    leg_id TEXT NOT NULL,
    tf TEXT NOT NULL,
    side INTEGER NOT NULL,
    setup_t INTEGER NOT NULL,
    entry_t INTEGER NOT NULL,
    entry_price REAL,
    stop_price REAL,
    exit_t INTEGER,
    exit_price REAL,
    exit_reason TEXT NOT NULL,
    ret_pct REAL,
    risk_pct REAL,
    r_multiple REAL,
    bars_held INTEGER NOT NULL,
    meta TEXT NOT NULL,
    notional REAL,
    pnl REAL,
    PRIMARY KEY (strategy_id, symbol, leg_id)
);
CREATE INDEX IF NOT EXISTS ix_paper_trades_entry ON paper_trades (strategy_id, entry_t);

CREATE TABLE IF NOT EXISTS signals (
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    leg_id TEXT NOT NULL,
    tf TEXT NOT NULL,
    side INTEGER NOT NULL,
    t INTEGER NOT NULL,
    entry_price REAL,
    stop_price REAL,
    tp1 REAL,
    tp2 REAL,
    meta TEXT NOT NULL,
    created_t INTEGER NOT NULL,
    decision TEXT,
    confidence REAL,
    reason TEXT,
    reviewed_t INTEGER,
    PRIMARY KEY (strategy_id, symbol, leg_id)
);
CREATE INDEX IF NOT EXISTS ix_signals_t ON signals (strategy_id, t);
CREATE INDEX IF NOT EXISTS ix_signals_reviewed ON signals (reviewed_t);

CREATE TABLE IF NOT EXISTS equity (
    strategy_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    t INTEGER NOT NULL,
    equity REAL,
    PRIMARY KEY (strategy_id, seq)
);

CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t INTEGER NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_journal_kind ON journal (kind, t);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now(now_t: int | None) -> int:
    return int(time.time()) if now_t is None else int(now_t)


def _dumps(obj: Any) -> str:
    """Compact JSON; non-JSON values fall back to str()."""
    return json.dumps(obj, separators=(",", ":"), default=str)


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _num(x: float | None) -> float | None:
    """Float for a REAL column; None for None/NaN (SQLite would store NaN as NULL anyway)."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _nan_if_null(d: dict, fields: tuple[str, ...]) -> dict:
    for f in fields:
        if d[f] is None:
            d[f] = math.nan
    return d


def _robust_pvariance(vals: list[float]) -> float:
    """Population variance after clipping to median +/- TRIAL_WINSOR_MADS x 1.4826 x MAD (no
    clipping when the MAD is 0); 0.0 for fewer than 2 values."""
    if len(vals) < 2:
        return 0.0
    med = statistics.median(vals)
    band = TRIAL_WINSOR_MADS * 1.4826 * statistics.median(abs(v - med) for v in vals)
    if band > 0:
        vals = [min(med + band, max(med - band, v)) for v in vals]
    return float(statistics.pvariance(vals))


def _engine(v) -> int:
    """Engine version of a stored row (rows written before versions were recorded are 1)."""
    return 1 if v is None else int(v)


def _estimates(rows, floor: int) -> list[float]:
    """Distinct real Sharpe estimates of the current engine: finite, |s| < CAP, n >= floor
    (unknown n kept)."""
    return sorted({float(f"{r['sharpe']:.10g}") for r in rows
                   if r["sharpe"] is not None and math.isfinite(r["sharpe"]) and abs(r["sharpe"]) < CAP
                   and _engine(r["engine"]) == ENGINE_VERSION
                   and (r["n_trades"] is None or r["n_trades"] >= floor)})


def _headline(metrics) -> dict:
    """What compact() keeps of an experiment's metrics: top-level scalars, and the COMPACT_KEYS
    scalars of nested dicts one level down (per-symbol / per-year / fold / row detail is dropped)."""
    if not isinstance(metrics, dict):
        return {}
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            sub = {kk: vv for kk, vv in v.items() if kk in COMPACT_KEYS
                   and (isinstance(vv, (int, float, str, bool)) or vv is None)}
            if sub:
                out[k] = sub
    return out


def _iso_day(t: int | None) -> str:
    return "?" if t is None else datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d")


def _metric_key(metrics: dict, name: str) -> tuple[int, float]:
    """Sort key: finite numeric metric descending, missing / non-numeric / non-finite last."""
    v = metrics.get(name) if isinstance(metrics, dict) else None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return (0, -float(v))
    return (1, 0.0)


class Store:
    """All persistent fibbot state in one SQLite file (``":memory:"`` works for tests).

    The connection may be shared across threads; every operation runs under an internal lock.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add the columns newer versions introduced to tables created by older ones."""
        with self._lock, self._conn:
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(trials)")}
            for col in ("n_trades", "engine"):
                if col not in cols:
                    self._conn.execute(f"ALTER TABLE trials ADD COLUMN {col} INTEGER")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- helpers -------------------------------------------------------------------------------

    def _write(self, sql: str, args=()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, args)

    def _one(self, sql: str, args=()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    def _rows(self, base: str, conds: list[tuple[str, tuple]], order: str | None = None,
              limit: int | None = None) -> list[sqlite3.Row]:
        """`base` [WHERE all conds (clause, args)] [ORDER BY order] [LIMIT limit]."""
        sql, args = base, []
        if conds:
            sql += " WHERE " + " AND ".join(c for c, _ in conds)
            for _, a in conds:
                args.extend(a)
        if order:
            sql += " ORDER BY " + order
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    @staticmethod
    def _experiment(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["spec"] = _loads(d["spec"])
        d["metrics"] = _loads(d["metrics"])
        return d

    @staticmethod
    def _strategy(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["spec"] = _loads(d["spec"])
        d["validation"] = _loads(d["validation"])
        return d

    @staticmethod
    def _paper_trade(row: sqlite3.Row) -> dict:
        d = _nan_if_null(dict(row), _TRADE_FLOATS)
        d["meta"] = _loads(d["meta"])
        d["status"] = "open" if d["exit_t"] is None else "closed"
        return d

    @staticmethod
    def _signal(row: sqlite3.Row) -> dict:
        d = _nan_if_null(dict(row), _SIGNAL_FLOATS)
        d["meta"] = _loads(d["meta"])
        return d

    # -- experiments ---------------------------------------------------------------------------

    def add_experiment(self, kind: str, spec: dict | None, spec_id: str | None,
                       window_start: int | None, window_end: int | None, metrics: dict,
                       source: str = "cli", notes: str = "", now_t: int | None = None) -> int:
        """Record one research result; returns its id. kind must be in EXPERIMENT_KINDS."""
        if kind not in EXPERIMENT_KINDS:
            raise ValueError(f"unknown experiment kind {kind!r}; expected one of {sorted(EXPERIMENT_KINDS)}")
        cur = self._write(
            "INSERT INTO experiments (created_t, kind, spec_id, spec, window_start, window_end,"
            " metrics, source, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(now_t), kind, spec_id, None if spec is None else _dumps(spec),
             None if window_start is None else int(window_start),
             None if window_end is None else int(window_end),
             _dumps(metrics if metrics is not None else {}), source, notes))
        return int(cur.lastrowid)

    def list_experiments(self, limit: int | None = 20, kind: str | None = None,
                         sort_by: str = "recent", spec_id: str | None = None,
                         exclude_sources: tuple[str, ...] = ()) -> list[dict]:
        """Experiments newest first, or best-first by metrics[sort_by] (missing last, ties newest
        first); optionally only one spec_id and without the given sources (filtered before the
        limit)."""
        if sort_by not in EXPERIMENT_SORTS:
            raise ValueError(f"sort_by must be one of {EXPERIMENT_SORTS}, got {sort_by!r}")
        conds = [("kind = ?", (kind,))] if kind is not None else []
        if spec_id is not None:
            conds.append(("spec_id = ?", (spec_id,)))
        if exclude_sources:
            conds.append((f"source NOT IN ({','.join('?' * len(exclude_sources))})",
                          tuple(exclude_sources)))
        recent = "created_t DESC, id DESC"
        if sort_by == "recent":
            return [self._experiment(r) for r in
                    self._rows("SELECT * FROM experiments", conds, recent, limit)]
        rows = [self._experiment(r) for r in self._rows("SELECT * FROM experiments", conds, recent)]
        rows.sort(key=lambda r: _metric_key(r["metrics"], sort_by))   # stable: ties stay newest first
        return rows if limit is None else rows[:int(limit)]

    def get_experiment(self, exp_id: int) -> dict | None:
        row = self._one("SELECT * FROM experiments WHERE id = ?", (int(exp_id),))
        return None if row is None else self._experiment(row)

    def latest_validation(self, spec_id: str) -> dict | None:
        """Newest kind="validate" experiment for spec_id."""
        row = self._one("SELECT * FROM experiments WHERE kind = 'validate' AND spec_id = ?"
                        " ORDER BY created_t DESC, id DESC LIMIT 1", (spec_id,))
        return None if row is None else self._experiment(row)

    def count_experiments(self, kind: str | None = None, since_t: int | None = None) -> int:
        conds = []
        if kind is not None:
            conds.append(("kind = ?", (kind,)))
        if since_t is not None:
            conds.append(("created_t >= ?", (int(since_t),)))
        return int(self._rows("SELECT COUNT(*) FROM experiments", conds)[0][0])

    def count_experiment_specs(self, kind: str | None = None) -> int:
        """Distinct non-null spec_ids among the experiments (of `kind`), e.g. kind="validate" =
        how many different specs have ever been validated (the number of out-of-sample tests)."""
        conds = [("spec_id IS NOT NULL", ())] + ([("kind = ?", (kind,))] if kind is not None else [])
        return int(self._rows("SELECT COUNT(DISTINCT spec_id) FROM experiments", conds)[0][0])

    # -- trials --------------------------------------------------------------------------------

    def record_trial(self, spec_id: str, sharpe: float | None, n: int | None = None,
                     engine: int = ENGINE_VERSION) -> None:
        """Upsert a distinct evaluated spec (each spec_id counts once in trial_stats()["n"]).

        sharpe is the research-window per-trade Sharpe, n its closed-trade count (None =
        unknown) and engine the strategy.ENGINE_VERSION that produced it. A None/NaN sharpe only
        counts the trial and keeps any stored estimate. A new estimate replaces the stored one
        (latest wins) unless both come from the same engine version, both trade counts are known
        and the new one is smaller: a narrower run (e.g. a symbol subset) never overwrites a wider
        one, and an estimate of an older engine is always replaced.
        """
        sharpe, n = _num(sharpe), None if n is None else int(n)
        with self._lock, self._conn:
            old = self._conn.execute("SELECT sharpe, n_trades, engine FROM trials WHERE spec_id = ?",
                                     (spec_id,)).fetchone()
            if old is None:
                self._conn.execute("INSERT INTO trials (spec_id, sharpe, n_trades, engine)"
                                   " VALUES (?, ?, ?, ?)",
                                   (spec_id, sharpe, None if sharpe is None else n,
                                    None if sharpe is None else int(engine)))
                return
            same_engine = old["sharpe"] is not None and _engine(old["engine"]) == int(engine)
            if sharpe is None or (same_engine and n is not None and old["n_trades"] is not None
                                  and n < old["n_trades"]):
                return
            self._conn.execute("UPDATE trials SET sharpe = ?, n_trades = ?, engine = ? WHERE spec_id = ?",
                               (sharpe, n, int(engine), spec_id))

    def trial_stats(self, min_trades: int | None = None) -> dict:
        """Inputs of the deflated-Sharpe correction over every trial recorded so far.

        {"n": distinct specs (every trial counts, so more search always raises n; this includes
         specs evaluated by an older engine version: they were still part of the search),
         "var_sharpe": variance of the trials' sharpes that are real estimates (0 if < 2),
         "n_var": how many distinct estimates var_sharpe is computed from}.
        An estimate is a finite sharpe with |sharpe| < metrics.CAP (a capped ratio is not an
        estimate) of the current strategy.ENGINE_VERSION (older engines' numbers are not
        comparable) from a trial with at least min_trades closed trades (default
        research.min_trades; trials with an unknown count are kept). Equal sharpes count once (a
        no-op parameter change or a zero-trade spec repeats one value and would otherwise shrink
        the variance), and values beyond median +/- TRIAL_WINSOR_MADS robust sigmas (1.4826 x MAD)
        are clipped to that band before the population variance is taken, so one outlier
        cannot blow the variance up.
        """
        floor = DEFAULTS["research"]["min_trades"] if min_trades is None else int(min_trades)
        with self._lock:
            rows = self._conn.execute("SELECT sharpe, n_trades, engine FROM trials").fetchall()
        vals = _estimates(rows, floor)
        return {"n": len(rows), "var_sharpe": _robust_pvariance(vals), "n_var": len(vals)}

    # -- holdout looks ---------------------------------------------------------------------------

    def record_look(self, key: str, spec_id: str, kind: str, window_start: int | None,
                    window_end: int | None, sharpe: float | None = None, n: int | None = None,
                    engine: int = ENGINE_VERSION, now_t: int | None = None) -> bool:
        """Record one distinct look at a held-out window (key identifies the draw: spec, window
        and symbol set; kind "validate" or "peek"); sharpe / n are the per-trade Sharpe and
        closed-trade count of that window (None for a peek over a mixed window). Returns False
        when the key was already recorded (a deterministic replay is not another look)."""
        cur = self._write(
            "INSERT OR IGNORE INTO looks (key, spec_id, kind, window_start, window_end, t, sharpe,"
            " n_trades, engine) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, spec_id, kind, None if window_start is None else int(window_start),
             None if window_end is None else int(window_end), _now(now_t), _num(sharpe),
             None if n is None else int(n), int(engine)))
        return cur.rowcount > 0

    def holdout_stats(self, window_start: int | None = None, window_end: int | None = None,
                      min_trades: int | None = None) -> dict:
        """Reuse of the held-out period [window_start, window_end): {"k": distinct looks (any
        spec, validations and peeks) whose window overlaps it (all looks when no window is
        given), "var_sharpe": variance of their validation Sharpes (estimates as in trial_stats:
        current engine, |sharpe| < CAP, n >= min_trades; None for fewer than 2), "n_var"}. A look
        at an earlier, non-overlapping window was a draw from other data and does not count."""
        floor = DEFAULTS["research"]["min_trades"] if min_trades is None else int(min_trades)
        conds = []
        if window_start is not None:
            conds.append(("(window_end IS NULL OR window_end > ?)", (int(window_start),)))
        if window_end is not None:
            conds.append(("(window_start IS NULL OR window_start < ?)", (int(window_end),)))
        rows = self._rows("SELECT sharpe, n_trades, engine, kind FROM looks", conds)
        vals = _estimates([r for r in rows if r["kind"] == "validate"], floor)
        return {"k": len(rows), "var_sharpe": _robust_pvariance(vals) if len(vals) >= 2 else None,
                "n_var": len(vals)}

    # -- roster --------------------------------------------------------------------------------

    def add_to_roster(self, strategy_id: str, name: str, spec: dict, paper_start_t: int,
                      validation: dict, rationale: str, now_t: int | None = None) -> None:
        """Put a new strategy on paper (status "paper").

        A strategy_id goes on paper once: re-adding any id already on the roster raises
        ValueError, a retired one included. Retirement is final and its paper record (trades,
        equity curve, signals) is never deleted, so a losing paper run cannot be erased and
        re-drawn; a variant with a changed parameter has a new spec_id and is a new trial.
        """
        now = _now(now_t)
        with self._lock, self._conn:
            row = self._conn.execute("SELECT status, status_t, status_reason FROM roster"
                                     " WHERE strategy_id = ?", (strategy_id,)).fetchone()
            if row is not None and row["status"] == "retired":
                raise ValueError(
                    f"strategy {strategy_id} was retired on {_iso_day(row['status_t'])}"
                    f" ({row['status_reason'] or 'no reason given'}); retirement is final and its"
                    " paper record stands. Promote a variant instead (a changed parameter gives a"
                    " new spec_id)")
            if row is not None:
                raise ValueError(f"strategy {strategy_id} is already on the roster"
                                 f" (status {row['status']})")
            self._conn.execute(
                "INSERT INTO roster (strategy_id, name, spec, status, paper_start_t,"
                " validation, rationale, added_t, status_t, status_reason)"
                " VALUES (?, ?, ?, 'paper', ?, ?, ?, ?, ?, '')",
                (strategy_id, name, _dumps(spec), int(paper_start_t), _dumps(validation),
                 rationale, now, now))

    def roster(self, status: str | None = "paper") -> list[dict]:
        """Roster rows (spec and validation decoded) ordered by paper_start_t; status=None -> all."""
        if status is not None and status not in STRATEGY_STATUSES:
            raise ValueError(f"unknown strategy status {status!r}")
        conds = [("status = ?", (status,))] if status is not None else []
        return [self._strategy(r) for r in
                self._rows("SELECT * FROM roster", conds, "paper_start_t, strategy_id", None)]

    def get_strategy(self, strategy_id: str) -> dict | None:
        row = self._one("SELECT * FROM roster WHERE strategy_id = ?", (strategy_id,))
        return None if row is None else self._strategy(row)

    def set_strategy_status(self, strategy_id: str, status: str, reason: str = "",
                            now_t: int | None = None) -> None:
        """Set status (paper|retired|graduated); KeyError for an unknown strategy."""
        if status not in STRATEGY_STATUSES:
            raise ValueError(f"unknown strategy status {status!r};"
                             f" expected one of {sorted(STRATEGY_STATUSES)}")
        cur = self._write("UPDATE roster SET status = ?, status_reason = ?, status_t = ?"
                          " WHERE strategy_id = ?", (status, reason, _now(now_t), strategy_id))
        if cur.rowcount == 0:
            raise KeyError(strategy_id)

    def count_ever_on_paper(self) -> int:
        """Distinct strategies ever put on paper (each id can go on paper only once)."""
        return int(self._one("SELECT COUNT(*) FROM roster")[0])

    # -- paper trades --------------------------------------------------------------------------

    def upsert_paper_trade(self, strategy_id: str, trade: Trade, notional: float | None = None,
                           pnl: float | None = None) -> str:
        """Insert or refresh the trade keyed by (strategy_id, trade.symbol, trade.leg_id).

        notional/pnl of None keep the stored values. Returns "new", "updated" (any stored value
        differs: exit fields, ret_pct, notional, pnl, ...) or "unchanged" (nothing written).
        """
        vals = {
            "strategy_id": strategy_id, "symbol": trade.symbol, "leg_id": trade.leg_id,
            "tf": trade.tf, "side": int(trade.side), "setup_t": int(trade.setup_t),
            "entry_t": int(trade.entry_t), "entry_price": _num(trade.entry_price),
            "stop_price": _num(trade.stop_price),
            "exit_t": None if trade.exit_t is None else int(trade.exit_t),
            "exit_price": _num(trade.exit_price), "exit_reason": trade.exit_reason,
            "ret_pct": _num(trade.ret_pct), "risk_pct": _num(trade.risk_pct),
            "r_multiple": _num(trade.r_multiple), "bars_held": int(trade.bars_held),
            "meta": _dumps(trade.meta), "notional": _num(notional), "pnl": _num(pnl),
        }
        with self._lock, self._conn:
            old = self._conn.execute(f"SELECT * FROM paper_trades WHERE {_LEG_KEY}",
                                     (strategy_id, trade.symbol, trade.leg_id)).fetchone()
            if old is not None:
                for k in ("notional", "pnl"):
                    if vals[k] is None:
                        vals[k] = old[k]
                if all(old[k] == v for k, v in vals.items()):
                    return "unchanged"
            self._conn.execute(f"INSERT OR REPLACE INTO paper_trades ({', '.join(vals)})"
                               f" VALUES ({', '.join('?' * len(vals))})", tuple(vals.values()))
        return "new" if old is None else "updated"

    def paper_trades(self, strategy_id: str | None = None, status: str | None = None,
                     limit: int | None = None) -> list[dict]:
        """Paper trades newest entry first; rows = Trade fields + notional, pnl, status."""
        conds = [("strategy_id = ?", (strategy_id,))] if strategy_id is not None else []
        if status == "open":
            conds.append(("exit_t IS NULL", ()))
        elif status == "closed":
            conds.append(("exit_t IS NOT NULL", ()))
        elif status is not None:
            raise ValueError(f"status must be 'open', 'closed' or None, got {status!r}")
        rows = self._rows("SELECT * FROM paper_trades", conds, "entry_t DESC, symbol, leg_id", limit)
        return [self._paper_trade(r) for r in rows]

    @staticmethod
    def trade_from_row(row: dict) -> Trade:
        """Rebuild a Trade from a paper_trades() row (extra keys are ignored)."""
        return Trade.from_dict(row)

    # -- signals -------------------------------------------------------------------------------

    def add_signal(self, signal: Signal, now_t: int | None = None) -> bool:
        """Store a signal once per (strategy_id, symbol, leg_id); False if it already existed."""
        cur = self._write(
            "INSERT OR IGNORE INTO signals (strategy_id, symbol, leg_id, tf, side, t, entry_price,"
            " stop_price, tp1, tp2, meta, created_t) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (signal.strategy_id, signal.symbol, signal.leg_id, signal.tf, int(signal.side),
             int(signal.t), _num(signal.entry_price), _num(signal.stop_price), _num(signal.tp1),
             _num(signal.tp2), _dumps(signal.meta), _now(now_t)))
        return cur.rowcount == 1

    def signals(self, strategy_id: str | None = None, unreviewed: bool = False,
                since_t: int | None = None, limit: int | None = None) -> list[dict]:
        """Signals newest first (by t). Rows = Signal fields + created_t, decision, confidence,
        reason, reviewed_t (review fields None until reviewed). since_t filters on t >= since_t."""
        conds = [("strategy_id = ?", (strategy_id,))] if strategy_id is not None else []
        if unreviewed:
            conds.append(("decision IS NULL", ()))
        if since_t is not None:
            conds.append(("t >= ?", (int(since_t),)))
        rows = self._rows("SELECT * FROM signals", conds, "t DESC, symbol, leg_id", limit)
        return [self._signal(r) for r in rows]

    def set_signal_review(self, strategy_id: str, symbol: str, leg_id: str, decision: str,
                          confidence: float, reason: str, now_t: int | None = None) -> None:
        """Record (or overwrite) the brain's take/skip review; KeyError if the signal is unknown."""
        if decision not in SIGNAL_DECISIONS:
            raise ValueError(f"decision must be 'take' or 'skip', got {decision!r}")
        cur = self._write(f"UPDATE signals SET decision = ?, confidence = ?, reason = ?, reviewed_t = ?"
                          f" WHERE {_LEG_KEY}",
                          (decision, _num(confidence), reason, _now(now_t), strategy_id, symbol, leg_id))
        if cur.rowcount == 0:
            raise KeyError((strategy_id, symbol, leg_id))

    def vetoed_legs(self, strategy_id: str) -> set[tuple[str, str]]:
        """{(symbol, leg_id)} the brain reviewed as "skip" for this strategy."""
        with self._lock:
            rows = self._conn.execute("SELECT symbol, leg_id FROM signals WHERE strategy_id = ?"
                                      " AND decision = 'skip'", (strategy_id,)).fetchall()
        return {(r[0], r[1]) for r in rows}

    def count_signal_reviews_since(self, t: int) -> int:
        """Signals whose (latest) review was written at or after t."""
        return int(self._one("SELECT COUNT(*) FROM signals WHERE reviewed_t >= ?", (int(t),))[0])

    # -- equity --------------------------------------------------------------------------------

    def set_equity_curve(self, strategy_id: str, points: list[tuple[int, float]]) -> None:
        """Replace the strategy's curve with `points` (order and duplicate timestamps preserved)."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM equity WHERE strategy_id = ?", (strategy_id,))
            self._conn.executemany("INSERT INTO equity (strategy_id, seq, t, equity) VALUES (?, ?, ?, ?)",
                                   [(strategy_id, i, int(t), _num(e)) for i, (t, e) in enumerate(points)])

    def equity_curve(self, strategy_id: str) -> list[tuple[int, float]]:
        with self._lock:
            rows = self._conn.execute("SELECT t, equity FROM equity WHERE strategy_id = ? ORDER BY seq",
                                      (strategy_id,)).fetchall()
        return [(int(t), math.nan if e is None else float(e)) for t, e in rows]

    # -- journal -------------------------------------------------------------------------------

    def add_journal(self, kind: str, title: str, body: str, now_t: int | None = None) -> int:
        """Append a journal entry; returns its id. kind must be in JOURNAL_KINDS."""
        if kind not in JOURNAL_KINDS:
            raise ValueError(f"unknown journal kind {kind!r}; expected one of {sorted(JOURNAL_KINDS)}")
        cur = self._write("INSERT INTO journal (t, kind, title, body) VALUES (?, ?, ?, ?)",
                          (_now(now_t), kind, title, body))
        return int(cur.lastrowid)

    def journal(self, limit: int | None = 20, kind: str | None = None) -> list[dict]:
        """Newest first: [{id, t, kind, title, body}]."""
        conds = [("kind = ?", (kind,))] if kind is not None else []
        return [dict(r) for r in self._rows("SELECT id, t, kind, title, body FROM journal", conds,
                                            "t DESC, id DESC", limit)]

    # -- maintenance ---------------------------------------------------------------------------

    def file_bytes(self) -> int:
        """Size of the database file (after a WAL checkpoint)."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        try:
            return Path(self.path).stat().st_size
        except OSError:
            return 0

    def compact(self, now_t: int | None = None, keep_days: float = 90,
                journal_keep_days: float = 30) -> dict:
        """Shrink the database for months of unattended running, without changing anything a
        gate reads: experiments older than keep_days keep id, times, kind, spec_id, spec, source
        and notes but their metrics are cut to the headline numbers (_headline, plus
        "compacted": true), except validate experiments and those a roster row refers to (its
        promoting validation); "system" journal bodies older than journal_keep_days are cut to
        "" (the title stays). The trials and looks tables (deflated-Sharpe inputs) and roster,
        paper trades, equity curves and signals are untouched. Ends with VACUUM. Returns
        {"bytes_before", "bytes_after", "experiments", "journal"} (rows changed)."""
        now = _now(now_t)
        before = self.file_bytes()
        keep = {int(eid) for r in self.roster(None)
                for eid in [((r.get("validation") or {}).get("experiment_id"))] if eid is not None}
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT id, metrics FROM experiments WHERE created_t < ? AND kind != 'validate'",
                (int(now - keep_days * 86400),)).fetchall()
            n_exp = 0
            for r in rows:
                m = _loads(r["metrics"])
                if r["id"] in keep or (isinstance(m, dict) and m.get("compacted")):
                    continue
                self._conn.execute("UPDATE experiments SET metrics = ? WHERE id = ?",
                                   (_dumps({**_headline(m), "compacted": True}), r["id"]))
                n_exp += 1
            n_jr = self._conn.execute(
                "UPDATE journal SET body = '' WHERE kind = 'system' AND t < ? AND body != ''",
                (int(now - journal_keep_days * 86400),)).rowcount
        with self._lock:
            self._conn.execute("VACUUM")
        return {"bytes_before": before, "bytes_after": self.file_bytes(), "experiments": n_exp,
                "journal": n_jr}

    # -- key/value -----------------------------------------------------------------------------

    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self._one("SELECT value FROM kv WHERE key = ?", (key,))
        return default if row is None else _loads(row[0])

    def set_kv(self, key: str, value: Any) -> None:
        self._write("INSERT INTO kv (key, value) VALUES (?, ?)"
                    " ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, _dumps(value)))
