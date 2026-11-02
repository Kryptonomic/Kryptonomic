"""Forward paper trading by deterministic replay (see DESIGN.md, "fibbot/paper.py").

Every tick re-runs each roster strategy's FibEngine on every symbol from a fixed anchor
(paper_start_t - warmup) to the newest closed bar. The engine is causal and the anchor is fixed,
so recorded trades never change: new bars only add trades or close open ones. Paper trades
therefore equal a backtest of the same window, trade for trade.

Honesty rules on top of the replay:
- A symbol that joins the universe after a strategy's first tick trades only from the bar that
  was forming when it joined. Nothing is back-filled, because a coin that enters the market-cap
  ranking was selected with hindsight.
- A symbol that leaves the universe is still replayed while the account holds a position in it,
  so that position is managed to its exit; it gets no new trades or signals.
- A closed paper trade is final, and a brain veto ("active" review mode) only blocks legs with no
  recorded trade yet, so a later review can never erase a trade.
"""
from __future__ import annotations

import itertools
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .data.universe import get_universe
from .metrics import max_drawdown, trade_stats
from .portfolio import simulate_portfolio
from .strategy import FibEngine, StrategySpec, warmup_bars
from .types import TF_SECONDS, Costs, Trade

if TYPE_CHECKING:
    from .data.candles import CandleStore
    from .store import Store

log = logging.getLogger(__name__)

UNIVERSE_KEY = "paper_universe:{}"   # kv: {"paper_start_t": P, "symbols": {symbol: trade_start_t}}
DAY = 86400


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
    """Max drawdown of a paper account over: start_equity; the stored realized curve (or, when none
    is stored, start_equity plus the cumulative pnl of closed trades by exit time); and finally the
    current equity with open positions marked. `trades` are store.paper_trades() rows."""
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


def _key(tr: Trade) -> tuple[str, str]:
    return tr.symbol, tr.leg_id


def _error(out: dict, sid: str | None, symbol: str | None, err) -> None:
    msg = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    log.warning("paper tick: strategy=%s symbol=%s: %s", sid, symbol, msg)
    out["errors"].append({"strategy_id": sid, "symbol": symbol, "error": msg})


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
        "new_exits": [trade dict], "equity": float, "open": int}}, "new_signals": [Signal],
        "errors": [...]}. Trade dicts are Trade.to_dict() plus "notional" and "pnl"; a trade that
        opened and closed between two ticks is both a new entry and a new exit.
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

    def _run(self, plan: _Plan, now: int, out: dict) -> tuple[dict, list]:
        """Replay one strategy, record its new signals and its trades with portfolio sizing, and
        write its equity curve. Returns (per-strategy result, new Signal objects)."""
        sid, spec, start = plan.row["strategy_id"], plan.spec, int(plan.row["paper_start_t"])
        step = TF_SECONDS[spec.timeframe]
        engine = FibEngine(spec, Costs(**self.cfg["costs"]), strategy_id=sid)
        anchor_t, end_t = start - warmup_bars(spec) * step, now - step + 1   # bars closed by now
        vetoed = (self.store.vetoed_legs(sid) if self.cfg["brain"]["signal_review"] == "active"
                  else set())
        trades = {k: self.store.trade_from_row(r) for k, r in plan.stored.items()}
        ran, replayed, signals = set(), set(), []
        for sym, sym_start in sorted(plan.starts.items()):
            try:
                bars = self.candles.get(sym, spec.timeframe).window(anchor_t, end_t)
                skip = {leg for s, leg in vetoed if s == sym and (s, leg) not in plan.stored}
                res = engine.run(bars, trade_start_t=sym_start, skip_legs=skip)
            except Exception as e:
                _error(out, sid, sym, e)
                continue
            ran.add(sym)
            for tr in res.trades + ([res.open_trade] if res.open_trade else []):
                k, old = _key(tr), plan.stored.get(_key(tr))
                replayed.add(k)
                if old is None and sym not in plan.universe:
                    continue                    # a dropped symbol only manages held positions
                if old is None or old["exit_t"] is None:
                    trades[k] = tr              # closed trades are final
            if sym in plan.universe:
                signals += [sig for sig in res.signals if self.store.add_signal(sig, now_t=now)]
        for (sym, leg), r in plan.stored.items():
            if r["exit_t"] is None and sym in ran and (sym, leg) not in replayed:
                _error(out, sid, sym, f"open trade {leg} was not reproduced by the replay;"
                                      " kept as recorded")
        self.store.set_kv(UNIVERSE_KEY.format(sid), {"paper_start_t": start, "symbols": plan.starts})
        return {"new_signals": [sig.to_dict() for sig in signals],
                **self._book(sid, start, plan.stored, trades)}, signals

    def _book(self, sid: str, start: int, stored: dict, trades: dict) -> dict:
        """Size all trades through the portfolio rules, upsert them and write the equity curve."""
        start_equity = float(self.cfg["paper"]["starting_equity"])
        ordered = sorted(trades.values(), key=lambda tr: (tr.entry_t, tr.symbol, tr.leg_id))
        pr = simulate_portfolio(ordered, self.cfg["portfolio"], start_equity=start_equity)
        sized = {_key(rec["trade"]): (rec["notional"], rec["pnl"]) for rec in pr.taken}
        entries, exits = [], []
        for tr in ordered:
            notional, pnl = sized.get(_key(tr), (0.0, 0.0))   # 0 = skipped by the risk limits
            status = self.store.upsert_paper_trade(sid, tr, notional, pnl)
            d = {**tr.to_dict(), "notional": notional, "pnl": pnl}
            if status == "new":
                entries.append(d)
            old = stored.get(_key(tr))
            if tr.exit_t is not None and (old is None or old["exit_t"] is None):
                exits.append(d)
        self.store.set_equity_curve(sid, [(start, start_equity)] + pr.curve[1:])
        return {"new_entries": entries, "new_exits": exits, "equity": pr.final_equity,
                "open": sum(rec["trade"].exit_t is None for rec in pr.taken)}

    def account(self, strategy_id: str, now_t: int | None = None) -> dict:
        """Paper account summary from the store (no engine run); floats rounded for JSON.

        {"strategy_id", "name", "status", "timeframe", "paper_start_t", "days", "equity",
        "realized_equity", "return", "n_closed", "n_open", "stats": trade_stats(closed trades),
        "max_drawdown", "open_positions": [...]}. n_closed and stats cover every closed paper
        trade (R statistics do not depend on sizing); equity, n_open and open_positions cover the
        positions the portfolio rules actually took. KeyError for an unknown strategy.
        """
        row = self.store.get_strategy(strategy_id)
        if row is None:
            raise KeyError(strategy_id)
        start_equity = float(self.cfg["paper"]["starting_equity"])
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
