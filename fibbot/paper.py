"""Forward paper trading by deterministic replay (see DESIGN.md, "fibbot/paper.py").

Every tick re-runs each roster strategy's FibEngine on every symbol from a fixed anchor
(paper_start_t - warmup) to the newest closed bar. The engine is causal and the anchor is fixed,
so while code, settings and data stay the same, new bars only add trades or close open ones, and
paper trades equal a backtest of the same window, trade for trade.

Honesty rules on top of the replay:
- A symbol that joins the universe after a strategy's first tick trades only from the bar that
  was forming when it joined. Nothing is back-filled, because a coin that enters the market-cap
  ranking was selected with hindsight.
- A symbol that leaves the universe is still replayed while the account holds a position in it,
  so that position is managed to its exit; it gets no new trades or signals.
- A closed paper trade is final, and a brain veto ("active" review mode) only blocks legs with no
  recorded trade yet, so a later review can never erase a trade.
- The account settings (cfg.costs, cfg.portfolio, cfg.paper.starting_equity) are frozen at a
  strategy's first tick (kv ACCOUNT_KEY), so editing the config later cannot re-price or re-size
  its record; they apply to strategies promoted afterwards. A change of the engine code (the
  sources of the modules in ENGINE_MODULES) is reported and journaled once.
- History is never rewritten: a replayed trade that is not on record and entered at or before the
  newest bar an earlier tick already replayed for its symbol is refused, and so is a back-dated
  signal (a code or data change produced them, not a new bar). Bars that arrive late after a
  per-symbol outage still add their trades, which keeps paper == backtest; the dollar sizing of
  other closed trades can then change, and those are reported under "resized" (R and prices of a
  closed trade never change).
- A recorded open trade is not left open forever: if the replay stops reproducing it, or its
  symbol's bars stop more than DATA_GAP_DAYS before the strategy's newest bar (delisting, rename),
  for VOID_AFTER_TICKS ticks in a row, it is closed at the latest close with the engine's exit
  costs, exit_reason "void" or "data_gap" respectively.
Refusals, forced exits, re-sized trades and engine changes are journaled (kind "system").
"""
from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import itertools
import logging
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .backtest import costs_from_cfg
from .data.universe import get_universe
from .metrics import max_drawdown, trade_stats
from .portfolio import marked_curve, simulate_portfolio
from .strategy import FibEngine, StrategySpec, warmup_bars
from .types import TF_SECONDS, Costs, Trade

if TYPE_CHECKING:
    from .data.candles import CandleStore
    from .store import Store

log = logging.getLogger(__name__)

UNIVERSE_KEY = "paper_universe:{}"   # kv: {"paper_start_t": P, "symbols": {symbol: trade_start_t}}
ACCOUNT_KEY = "paper_account:{}"     # kv: frozen settings, see account_settings()
GUARD_KEY = "paper_guard:{}"         # kv: {"paper_start_t", "seen": {symbol: newest replayed bar t},
                                     #      "refused": [[symbol, leg_id]], "missing": {"sym|leg": ticks}}
DAY = 86400
DATA_GAP_DAYS = 3          # a held symbol whose bars lag the strategy's newest bar by this much is stale
VOID_AFTER_TICKS = 3       # consecutive ticks an open trade may go unconfirmed before it is closed
ENGINE_MODULES = ("strategy", "fib", "indicators", "portfolio", "types")
RESIZE_TOL = 1e-6


def rounded(x, digits: int = 6):
    """Copy of x with every finite float rounded to `digits` significant digits (recursive)."""
    if isinstance(x, float):
        return float(f"{x:.{digits}g}") if math.isfinite(x) else x
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, digits) for v in x]
    return x


def account_drawdown(start_equity: float, curve: list[tuple[int, float]], trades: list[dict]) -> float:
    """Max drawdown of a paper account over: start_equity; the stored equity curve (marked to
    market at every bar close a position was open, as written by PaperTrader, so it matches the
    backtest's drawdown; when none is stored, start_equity plus the cumulative pnl of closed trades
    by exit time); and finally the current equity with open positions marked. `trades` are
    store.paper_trades() rows."""
    if curve:
        values = [v for _, v in curve]
    else:
        closed = sorted((r for r in trades if r["exit_t"] is not None), key=lambda r: r["exit_t"])
        values = list(itertools.accumulate((r["pnl"] or 0.0 for r in closed), initial=start_equity))
    marked = start_equity + sum(r["pnl"] or 0.0 for r in trades)
    return max_drawdown([start_equity, *values, marked])


def paper_days(row: dict, now_t: int | None = None) -> float:
    """Days since the roster row's paper_start_t (0 before it starts)."""
    now = int(time.time()) if now_t is None else int(now_t)
    return max(0.0, (now - int(row["paper_start_t"])) / DAY)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def code_digest(source: str) -> str:
    """Digest of Python source that ignores comments, docstrings and layout (line numbers)."""
    return hashlib.sha1(ast.dump(_strip_docstrings(ast.parse(source))).encode()).hexdigest()


@functools.lru_cache(maxsize=1)
def engine_fingerprint() -> str:
    """Short hash of the code that decides paper trades and their sizing: code_digest() of each
    module in ENGINE_MODULES (an edit that only touches comments or docstrings keeps it).
    "unknown" when a source cannot be read."""
    h = hashlib.sha1()
    for mod in ENGINE_MODULES:
        try:
            digest = code_digest(Path(__file__).with_name(f"{mod}.py").read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            return "unknown"
        h.update(f"{mod}:{digest};".encode())
    return h.hexdigest()[:12]


def account_settings(store: Store, cfg: dict, row: dict) -> dict:
    """The settings a paper account is booked with: the snapshot frozen at the strategy's first
    tick, else (not ticked yet) the live cfg. {"paper_start_t", "costs", "portfolio",
    "starting_equity", "engine", "t", "engine_changes"}; costs and portfolio are complete (keys
    added to the config after the snapshot take their live values)."""
    start = int(row["paper_start_t"])
    snap = store.get_kv(ACCOUNT_KEY.format(row["strategy_id"]))
    # costs: every Costs field (defaults included), so the snapshot pins e.g. limit_through even
    # when the config does not list it, and a later change of a Costs default cannot re-price it
    live = {"paper_start_t": start, "costs": dataclasses.asdict(costs_from_cfg(cfg)),
            "portfolio": dict(cfg["portfolio"]),
            "starting_equity": float(cfg["paper"]["starting_equity"]), "engine": engine_fingerprint(),
            "t": None, "engine_changes": []}
    if not isinstance(snap, dict) or snap.get("paper_start_t") != start:
        return live
    return {**live, **snap, "costs": {**live["costs"], **(snap.get("costs") or {})},
            "portfolio": {**live["portfolio"], **(snap.get("portfolio") or {})},
            "starting_equity": float(snap.get("starting_equity", live["starting_equity"]))}


def _costs(d: dict) -> Costs:
    return Costs(**{k: float(v) for k, v in d.items() if k in Costs.__dataclass_fields__})


def _key(tr: Trade) -> tuple[str, str]:
    return tr.symbol, tr.leg_id


def _iso(t: int | None) -> str:
    if t is None:
        return "?"
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _error(out: dict, sid: str | None, symbol: str | None, err) -> None:
    msg = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    log.warning("paper tick: strategy=%s symbol=%s: %s", sid, symbol, msg)
    out["errors"].append({"strategy_id": sid, "symbol": symbol, "error": msg})


def _mark_t(tr: Trade, step: int) -> int:
    """Bar time an open trade is marked at (its "open" fill), else entry_t + bars_held bars."""
    fills = tr.meta.get("fills") if isinstance(tr.meta, dict) else None
    if fills and len(fills[-1]) >= 4 and fills[-1][2] == "open":
        return int(fills[-1][3])
    return int(tr.entry_t + tr.bars_held * step)


def force_close(tr: Trade, t: int, reason: str, costs: Costs, step: int,
                close_px: float | None = None) -> Trade:
    """Close an open trade at bar time t with the engine's accounting: the part not yet filled
    exits at close_px less slippage (None: at the trade's current mark, which already includes
    it), then fees on both sides; partial fills already made (tp1) are kept."""
    s, fee = tr.side, costs.fee_rate
    fills = [list(f) for f in (tr.meta.get("fills") or [])] if isinstance(tr.meta, dict) else []
    mark = fills.pop() if fills and len(fills[-1]) >= 4 and fills[-1][2] == "open" else None
    exit_t = max(int(t), int(tr.entry_t))
    held = (exit_t - int(tr.entry_t)) // step
    if close_px is not None:
        px = float(close_px) * (1 - s * costs.slippage)
    elif mark is not None:
        px = float(mark[1])
    else:
        px = None
    p0 = tr.entry_price
    if px is None or not (p0 > 0) or not math.isfinite(px):   # nothing to re-price: keep the mark
        return replace(tr, exit_t=exit_t, exit_reason=reason, bars_held=held)
    rem = max(0.0, 1.0 - sum(float(f[0]) for f in fills))
    gross = sum(float(f[0]) * float(f[1]) for f in fills) + rem * px
    ret = gross / p0 * (1 - fee) - (1 + fee) if s > 0 else (1 - fee) - gross / p0 * (1 + fee)
    return replace(tr, exit_t=exit_t, exit_price=gross, exit_reason=reason, ret_pct=ret,
                   r_multiple=ret / tr.risk_pct if tr.risk_pct > 0 else 0.0, bars_held=held,
                   meta={**tr.meta, "fills": fills + [[rem, px, reason, exit_t]]})


def _confirms(tr: Trade, old: dict | None, mark: int | None) -> bool:
    """Whether a replayed trade continues the recorded OPEN trade `old`: same entry bar and side,
    and not closed on a bar an earlier tick already replayed with the position still open (that
    would rewrite history, like a back-filled trade). Entry prices may follow a data revision."""
    return (old is not None and old["exit_t"] is None and int(tr.entry_t) == int(old["entry_t"])
            and int(tr.side) == int(old["side"])
            and not (tr.exit_t is not None and mark is not None and tr.exit_t <= mark))


def _moved(old: dict, notional: float, pnl: float) -> bool:
    """Whether a stored closed trade's sizing differs from the new one (taken <-> skipped too)."""
    on, op = old["notional"], old["pnl"]
    if on is None or op is None:
        return False
    return ((on > 0) != (notional > 0) or abs(on - notional) > RESIZE_TOL * max(1.0, abs(on))
            or abs(op - pnl) > RESIZE_TOL * max(1.0, abs(op)))


@dataclass
class _Plan:
    """One roster strategy prepared for a tick."""
    row: dict
    spec: StrategySpec
    stored: dict[tuple[str, str], dict]   # (symbol, leg_id) -> paper_trades row before this tick
    starts: dict[str, int]                # symbol to replay -> its trade_start_t
    universe: frozenset[str]              # symbols allowed to open new trades


class PaperTrader:
    """Paper sub-accounts for the roster strategies with status "paper" (one per strategy)."""

    def __init__(self, cfg: dict, store: Store, candles: CandleStore):
        self.cfg, self.store, self.candles = cfg, store, candles

    def tick(self, update_data: bool = True, now_t: int | None = None,
             universe: list[str] | None = None) -> dict:
        """Replay every paper strategy up to the newest bar closed at now_t (default: now).

        universe defaults to get_universe(cfg). Per-strategy and per-symbol failures are reported
        in "errors" as {"strategy_id", "symbol", "error"} and never abort the tick. Returns
        {"strategies": {sid: {"new_signals": [Signal dict], "new_entries": [trade dict],
        "new_exits": [trade dict], "equity": float, "open": int, "refused": [Trade dict],
        "resized": [{"symbol", "leg_id", "notional_before", "notional", "pnl_before", "pnl"}],
        "engine_changed": bool}}, "new_signals": [Signal], "errors": [...]}. Trade dicts are
        Trade.to_dict() plus "notional" and "pnl"; a trade that opened and closed between two
        ticks is both a new entry and a new exit; forced exits ("void", "data_gap") are new exits.
        "refused" lists replayed trades refused this tick as history rewrites, "resized" the
        closed trades whose dollar sizing changed.
        """
        now = int(time.time()) if now_t is None else int(now_t)
        out: dict = {"strategies": {}, "new_signals": [], "errors": []}
        roster = self.store.roster("paper")
        if not roster:
            return out
        symbols = list(get_universe(self.cfg) if universe is None else universe)
        plans = []
        for row in roster:
            try:
                plans.append(self._plan(row, symbols, now))
            except Exception as e:  # a broken roster row must not stop the other strategies
                _error(out, row["strategy_id"], None, e)
        if update_data and plans:
            self._update_data(plans, out)
        for plan in plans:
            sid = plan.row["strategy_id"]
            try:
                res, signals = self._run(plan, now, out)
            except Exception as e:
                _error(out, sid, None, e)
                continue
            out["strategies"][sid] = res
            out["new_signals"].extend(signals)
        return out

    def _plan(self, row: dict, universe: list[str], now: int) -> _Plan:
        """Parse the spec and decide which symbols to replay, each from which trade_start_t."""
        spec = StrategySpec.from_dict(row["spec"]).validate()
        sid, start = row["strategy_id"], int(row["paper_start_t"])
        stored = {(r["symbol"], r["leg_id"]): r for r in self.store.paper_trades(sid)}
        state = self.store.get_kv(UNIVERSE_KEY.format(sid))
        known = state["symbols"] if state and state.get("paper_start_t") == start else {}
        step = TF_SECONDS[spec.timeframe]
        joined = start if not known else max(start, now - now % step)   # the forming bar
        starts = {sym: int(known.get(sym, joined)) for sym in universe}
        for (sym, _), r in stored.items():
            if r["exit_t"] is None:              # keep managing positions on dropped symbols
                starts.setdefault(sym, int(known.get(sym, start)))
        return _Plan(row, spec, stored, starts, frozenset(universe))

    def _update_data(self, plans: list[_Plan], out: dict) -> None:
        """One candles.update_all over every replayed symbol and every roster timeframe."""
        symbols = sorted({sym for p in plans for sym in p.starts})
        tfs = sorted({p.spec.timeframe for p in plans})
        try:
            self.candles.update_all(symbols, tfs)
        except Exception as e:  # cached bars are still worth replaying
            _error(out, None, None, f"data update failed: {e}")
        for (sym, tf), msg in sorted(getattr(self.candles, "last_errors", {}).items()):
            _error(out, None, sym, f"data update {tf}: {msg}")

    def _settings(self, row: dict, now: int, out: dict, notes: list[str]) -> tuple[dict, bool]:
        """account_settings(), frozen into kv on a strategy's first tick; an engine fingerprint
        that differs from the stored one is reported once and recorded. Returns (settings,
        engine_changed)."""
        sid = row["strategy_id"]
        settings = account_settings(self.store, self.cfg, row)
        fp, changed = engine_fingerprint(), False
        if settings["t"] is None:
            settings["t"] = now
        elif settings.get("engine") != fp:
            changed = True
            settings["engine_changes"] = [*settings.get("engine_changes", []),
                                          {"t": now, "from": settings.get("engine"), "to": fp}]
            msg = (f"engine code changed since the last tick ({settings.get('engine')} -> {fp});"
                   " the record so far is kept, later trades follow the new code and replayed"
                   " trades that would rewrite history are refused. Re-validate the spec.")
            _error(out, sid, None, msg)
            notes.append(msg)
        else:
            return settings, False
        settings["engine"] = fp
        self.store.set_kv(ACCOUNT_KEY.format(sid), settings)
        return settings, changed

    def _guard(self, sid: str, start: int) -> dict:
        g = self.store.get_kv(GUARD_KEY.format(sid))
        if not isinstance(g, dict) or g.get("paper_start_t") != start:
            g = {}
        return {"paper_start_t": start, "seen": dict(g.get("seen") or {}),
                "refused": [list(k) for k in g.get("refused") or []],
                "missing": dict(g.get("missing") or {})}

    def _run(self, plan: _Plan, now: int, out: dict) -> tuple[dict, list]:
        """Replay one strategy, record its new signals and its trades with portfolio sizing, and
        write its equity curve. Returns (per-strategy result, new Signal objects)."""
        row, spec = plan.row, plan.spec
        sid, start = row["strategy_id"], int(row["paper_start_t"])
        step = TF_SECONDS[spec.timeframe]
        notes: list[str] = []
        settings, engine_changed = self._settings(row, now, out, notes)
        costs = _costs(settings["costs"])
        engine = FibEngine(spec, costs, strategy_id=sid)
        anchor_t, end_t = start - warmup_bars(spec) * step, now - step + 1   # bars closed by now
        vetoed = (self.store.vetoed_legs(sid) if self.cfg["brain"]["signal_review"] == "active"
                  else set())
        guard = self._guard(sid, start)
        seen, missing = guard["seen"], guard["missing"]
        refused = {tuple(k) for k in guard["refused"]}
        held = {k for k, r in plan.stored.items() if r["exit_t"] is None}
        trades = {k: self.store.trade_from_row(r) for k, r in plan.stored.items()}
        ran, replayed, signals, last, windows = set(), set(), [], {}, {}
        refused_now: list[Trade] = []
        for sym, sym_start in sorted(plan.starts.items()):
            try:
                bars = self.candles.get(sym, spec.timeframe).window(anchor_t, end_t)
                skip = {leg for s, leg in vetoed if s == sym and (s, leg) not in plan.stored}
                res = engine.run(bars, trade_start_t=sym_start, skip_legs=skip)
            except Exception as e:
                _error(out, sid, sym, e)
                continue
            ran.add(sym)
            windows[sym] = bars
            if len(bars):
                last[sym] = (int(bars.t[-1]), float(bars.c[-1]))
            mark = seen.get(sym)
            produced = res.trades + ([res.open_trade] if res.open_trade else [])
            confirmed = {_key(tr) for tr in produced if _confirms(tr, plan.stored.get(_key(tr)), mark)}
            lost = any(k[0] == sym and k not in confirmed for k in held)   # a held position vanished
            for tr in produced:
                k, old = _key(tr), plan.stored.get(_key(tr))
                if old is not None:
                    if k in confirmed:          # closed trades are final
                        trades[k] = tr
                        replayed.add(k)
                    continue
                if sym not in plan.universe or k in refused:
                    continue                    # a dropped symbol only manages held positions
                if lost or (mark is not None and tr.entry_t <= mark):
                    refused.add(k)              # not a new bar's trade: history would change
                    refused_now.append(tr)
                    continue
                trades[k] = tr
            if sym in plan.universe:
                signals += [sig for sig in res.signals if (mark is None or sig.t > mark)
                            and self.store.add_signal(sig, now_t=now)]
            if sym in last:
                seen[sym] = max(int(seen.get(sym, last[sym][0])), last[sym][0])
        if refused_now:
            listed = ", ".join(f"{tr.symbol} {tr.leg_id} (entry {_iso(tr.entry_t)})"
                               for tr in refused_now[:10])
            msg = (f"refused {len(refused_now)} replayed trade(s) that are not on record but entered"
                   f" at or before bars an earlier tick already replayed (code or data changed;"
                   f" history is not back-filled): {listed}" + (" ..." if len(refused_now) > 10 else ""))
            _error(out, sid, None, msg)
            notes.append(msg)
        forced = self._resolve_stranded(plan, held, trades, ran, replayed, last, missing, costs,
                                        step, out, notes)
        self.store.set_kv(UNIVERSE_KEY.format(sid), {"paper_start_t": start, "symbols": plan.starts})
        booked = self._book(sid, start, settings, plan.stored, trades, windows, costs, end_t)
        # only once the trades are stored: a failed tick must not advance the high-water marks
        self.store.set_kv(GUARD_KEY.format(sid), {
            "paper_start_t": start, "seen": seen, "refused": sorted(list(k) for k in refused),
            "missing": missing})
        if booked["resized"]:
            notes.append(f"{len(booked['resized'])} closed trade(s) re-sized after trades from late"
                         " bars were added: " + ", ".join(
                             f"{d['symbol']} {d['leg_id']} pnl {d['pnl_before']:.2f} -> {d['pnl']:.2f}"
                             for d in booked["resized"][:10]))
        if notes:
            what = [label for label, hit in (
                ("engine changed", engine_changed),
                (f"{len(refused_now)} trade(s) refused", refused_now),
                (f"{len(forced)} open trade(s) closed", forced),
                (f"{len(booked['resized'])} trade(s) re-sized", booked["resized"])) if hit]
            self.store.add_journal("system", f"Paper {row['name']} ({sid}): {', '.join(what)}",
                                   "\n\n".join(notes), now_t=now)
        return {"new_signals": [sig.to_dict() for sig in signals], **booked,
                "refused": [tr.to_dict() for tr in refused_now],
                "engine_changed": engine_changed}, signals

    def _resolve_stranded(self, plan: _Plan, held: set, trades: dict, ran: set, replayed: set,
                          last: dict, missing: dict, costs: Costs, step: int, out: dict,
                          notes: list[str]) -> list[Trade]:
        """Count, then close, recorded open trades the replay no longer confirms (see the module
        docstring). Updates trades and missing in place; returns the trades closed this tick."""
        sid, tf = plan.row["strategy_id"], plan.spec.timeframe
        newest = max((t for t, _ in last.values()), default=None)
        gap = max(DATA_GAP_DAYS * DAY, 3 * step)
        forced = []
        for k in sorted(held):
            sym, leg = k
            tag, cur = f"{sym}|{leg}", trades[k]
            if sym in ran and k in replayed:
                stale = newest is not None and sym in last and last[sym][0] < newest - gap
                if cur.exit_t is not None or not stale:
                    missing.pop(tag, None)
                    continue
                reason = "data_gap"
            elif sym in last:
                reason = "void"                 # bars are there but the replay lost the trade
            elif newest is not None and _mark_t(cur, step) < newest - gap:
                reason = "data_gap"             # no bars / unreadable, and its mark is old
            else:
                continue                        # a read error with a recent mark: wait
            missing[tag] = n = int(missing.get(tag, 0)) + 1
            if n < VOID_AFTER_TICKS:
                if reason == "void":
                    _error(out, sid, sym, f"open trade {leg} was not reproduced by the replay;"
                                          " kept as recorded")
                elif sym in last:               # (an unreadable symbol is already reported)
                    _error(out, sid, sym, f"open trade {leg}: no new {tf} bar since"
                                          f" {_iso(_mark_t(cur, step))}; it is closed as data_gap"
                                          f" after {VOID_AFTER_TICKS} such ticks ({n} so far)")
                continue
            if reason == "void" and sym in last and last[sym][0] >= _mark_t(cur, step):
                closed = force_close(cur, last[sym][0], reason, costs, step, close_px=last[sym][1])
            else:                               # stale or unreadable: its last mark is the last close
                closed = force_close(cur, _mark_t(cur, step), reason, costs, step)
            trades[k] = closed
            missing.pop(tag, None)
            forced.append(closed)
            msg = (f"open trade {sym} {leg} closed as {reason} at {closed.exit_price:.6g}"
                   f" ({_iso(closed.exit_t)}, {closed.r_multiple:+.2f} R) after {n} ticks"
                   + (" in which the replay did not reproduce it" if reason == "void" else
                      f" without a new {tf} bar for its symbol"))
            _error(out, sid, sym, msg)
            notes.append(msg)
        return forced

    def _book(self, sid: str, start: int, settings: dict, stored: dict, trades: dict,
              bars_by_symbol: dict | None = None, costs: Costs | None = None,
              end_t: int | None = None) -> dict:
        """Size all trades through the (frozen) portfolio rules, upsert them and write the equity
        curve: mark-to-market (portfolio.marked_curve over the replayed bars, as backtest() uses
        for its drawdown) when the bars and costs are given, else the realized exits. Closed
        trades whose dollar sizing changed are listed under "resized"."""
        start_equity = float(settings["starting_equity"])
        ordered = sorted(trades.values(), key=lambda tr: (tr.entry_t, tr.symbol, tr.leg_id))
        pr = simulate_portfolio(ordered, settings["portfolio"], start_equity=start_equity)
        sized = {_key(rec["trade"]): (rec["notional"], rec["pnl"]) for rec in pr.taken}
        entries, exits, resized = [], [], []
        for tr in ordered:
            notional, pnl = sized.get(_key(tr), (0.0, 0.0))   # 0 = skipped by the risk limits
            old = stored.get(_key(tr))
            if old is not None and old["exit_t"] is not None and _moved(old, notional, pnl):
                resized.append({"symbol": tr.symbol, "leg_id": tr.leg_id,
                                "notional_before": old["notional"], "notional": notional,
                                "pnl_before": old["pnl"], "pnl": pnl})
            status = self.store.upsert_paper_trade(sid, tr, notional, pnl)
            d = {**tr.to_dict(), "notional": notional, "pnl": pnl}
            if status == "new":
                entries.append(d)
            if tr.exit_t is not None and (old is None or old["exit_t"] is None):
                exits.append(d)
        if bars_by_symbol is not None and costs is not None:
            inner = marked_curve(pr, bars_by_symbol, costs.fee_rate, costs.slippage, end_t)
        else:
            inner = pr.curve[1:]
        self.store.set_equity_curve(sid, [(start, start_equity)] + [p for p in inner if p[0] >= start])
        return {"new_entries": entries, "new_exits": exits, "equity": pr.final_equity,
                "open": sum(rec["trade"].exit_t is None for rec in pr.taken), "resized": resized}

    def account(self, strategy_id: str, now_t: int | None = None) -> dict:
        """Paper account summary from the store (no engine run); floats rounded for JSON.

        {"strategy_id", "name", "status", "timeframe", "paper_start_t", "days", "equity",
        "realized_equity", "return", "n_closed", "n_open", "stats": trade_stats(closed trades),
        "max_drawdown", "open_positions": [...]}. n_closed and stats cover every closed paper
        trade (R statistics do not depend on sizing); equity, n_open and open_positions cover the
        positions the portfolio rules actually took. Uses the account's frozen starting equity.
        KeyError for an unknown strategy.
        """
        row = self.store.get_strategy(strategy_id)
        if row is None:
            raise KeyError(strategy_id)
        start_equity = float(account_settings(self.store, self.cfg, row)["starting_equity"])
        rows = self.store.paper_trades(strategy_id)
        closed = [r for r in rows if r["exit_t"] is not None]
        held = [r for r in rows if r["exit_t"] is None and (r["notional"] or 0.0) > 0]
        realized = start_equity + sum(r["pnl"] or 0.0 for r in closed)
        equity = realized + sum(r["pnl"] or 0.0 for r in held)
        return rounded({
            "strategy_id": strategy_id, "name": row["name"], "status": row["status"],
            "timeframe": StrategySpec.from_dict(row["spec"]).timeframe,
            "paper_start_t": row["paper_start_t"], "days": paper_days(row, now_t),
            "equity": equity, "realized_equity": realized, "return": equity / start_equity - 1.0,
            "n_closed": len(closed), "n_open": len(held),
            "stats": trade_stats([self.store.trade_from_row(r) for r in closed]),
            "max_drawdown": account_drawdown(start_equity, self.store.equity_curve(strategy_id), rows),
            "open_positions": [{
                "symbol": r["symbol"], "side": r["side"], "leg_id": r["leg_id"],
                "entry_t": r["entry_t"], "entry_price": r["entry_price"],
                "stop_price": r["stop_price"], "tp1": r["meta"].get("tp1"),
                "tp2": r["meta"].get("tp2"), "mark_price": r["exit_price"],
                "r_multiple": r["r_multiple"], "ret_pct": r["ret_pct"],
                "bars_held": r["bars_held"], "notional": r["notional"], "pnl": r["pnl"],
            } for r in held],
        })
