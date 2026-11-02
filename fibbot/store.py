"""SQLite persistence (stdlib sqlite3, WAL mode) for research and paper-trading state.

Tables: experiments, trials, roster, paper_trades, signals (with the brain's review), equity,
journal and kv. JSON columns (spec, metrics, validation, meta, kv values) are stored as TEXT and
decoded on read. SQLite stores NaN as NULL, so NaN prices/returns read back as ``float('nan')``.
Timestamps written by the store use ``int(time.time())`` unless a ``now_t`` kwarg is given.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
import threading
import time
from pathlib import Path
from typing import Any

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
    sharpe REAL
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
                         sort_by: str = "recent") -> list[dict]:
        """Experiments newest first, or best-first by metrics[sort_by] (missing last, ties newest first)."""
        if sort_by not in EXPERIMENT_SORTS:
            raise ValueError(f"sort_by must be one of {EXPERIMENT_SORTS}, got {sort_by!r}")
        conds = [("kind = ?", (kind,))] if kind is not None else []
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

    # -- trials --------------------------------------------------------------------------------

    def record_trial(self, spec_id: str, sharpe: float | None) -> None:
        """Upsert a distinct evaluated spec; keeps the latest non-null (non-NaN) sharpe."""
        self._write("INSERT INTO trials (spec_id, sharpe) VALUES (?, ?) ON CONFLICT (spec_id)"
                    " DO UPDATE SET sharpe = COALESCE(excluded.sharpe, trials.sharpe)",
                    (spec_id, _num(sharpe)))

    def trial_stats(self) -> dict:
        """{"n": distinct specs, "var_sharpe": population variance of finite sharpes (0 if < 2)}."""
        with self._lock:
            rows = self._conn.execute("SELECT sharpe FROM trials").fetchall()
        vals = [r[0] for r in rows if r[0] is not None and math.isfinite(r[0])]
        return {"n": len(rows), "var_sharpe": float(statistics.pvariance(vals)) if len(vals) >= 2 else 0.0}

    # -- roster --------------------------------------------------------------------------------

    def add_to_roster(self, strategy_id: str, name: str, spec: dict, paper_start_t: int,
                      validation: dict, rationale: str, now_t: int | None = None) -> None:
        """Put a strategy on paper (status "paper").

        Re-adding a retired id re-activates it with the new paper_start_t, spec, validation and
        rationale, and starts a fresh paper account (its old paper trades and equity curve are
        deleted; signals and their reviews are kept). Re-adding a non-retired id raises ValueError.
        """
        now = _now(now_t)
        with self._lock, self._conn:
            row = self._conn.execute("SELECT status FROM roster WHERE strategy_id = ?",
                                     (strategy_id,)).fetchone()
            if row is not None:
                if row["status"] != "retired":
                    raise ValueError(f"strategy {strategy_id} is already on the roster"
                                     f" (status {row['status']})")
                self._conn.execute("DELETE FROM paper_trades WHERE strategy_id = ?", (strategy_id,))
                self._conn.execute("DELETE FROM equity WHERE strategy_id = ?", (strategy_id,))
            self._conn.execute(
                "INSERT OR REPLACE INTO roster (strategy_id, name, spec, status, paper_start_t,"
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
        """Distinct strategies ever added to the roster (re-activation does not count twice)."""
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

    # -- key/value -----------------------------------------------------------------------------

    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self._one("SELECT value FROM kv WHERE key = ?", (key,))
        return default if row is None else _loads(row[0])

    def set_kv(self, key: str, value: Any) -> None:
        self._write("INSERT INTO kv (key, value) VALUES (?, ?)"
                    " ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, _dumps(value)))
