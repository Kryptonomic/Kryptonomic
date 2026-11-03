"""Portfolio simulation: sizes a list of single-position trades into one shared account.

Pure and deterministic. Positions are sized from REALIZED equity at entry time and limited by the
risk rules in cfg["portfolio"]; open trades (exit_t None) keep their slot and risk until the end
and are marked to their (already marked) ret_pct in final_equity only.

Account decisions are causal: a trade's entry_t is the open time of the bar it filled in, and its
order was placed at the close of the bar before, so an entry at t is decided on the account as it
stood before bar t. Exits stamped t happened somewhere inside bar t, possibly after that fill, so
they do not free a slot, release risk, add realized PnL for sizing or count toward the daily loss
limit of entries at the same t (see simulate_portfolio).

marked_curve() is the mark-to-market equity series (every bar close a position is open, partial
fills at their fill time) used for drawdown / Sharpe / Sortino; PortfolioResult.curve is the
realized curve.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .config import DEFAULTS
from .types import Bars, Trade

DAY = 86400
# event phases at equal timestamps: entries first (decided on the state before the bar), then
# exits of earlier positions, then exits of trades that closed on their own entry bar
_ENTRY, _EXIT, _SAME_BAR_EXIT = 0, 1, 2
MAX_TOTAL_NOTIONAL = 1.0       # default portfolio.max_total_notional: open notional <= 1x equity


@dataclass
class PortfolioResult:
    start_equity: float
    final_equity: float                     # realized + open positions marked
    realized_equity: float
    curve: list[tuple[int, float]] = field(default_factory=list)   # (t, realized equity)
    taken: list[dict] = field(default_factory=list)                # {"trade", "notional", "pnl"}
    skipped: list[dict] = field(default_factory=list)              # {"trade", "reason"}


def simulate_portfolio(trades: list[Trade], pcfg: dict, start_equity: float | None = None) -> PortfolioResult:
    """Replay entries and exits in time order and apply the sizing / risk rules of DESIGN.md.

    Event order: by time; at equal timestamps entries first (by symbol, then input order), then
    exits of earlier positions, then exits of trades that closed on their own entry bar. An entry
    at t therefore sees only exits with exit_t < t: positions exiting during bar t still hold their
    slot, their risk and their notional, their PnL is not yet in the sizing equity and not in the
    day's realized loss. Missing pcfg keys fall back to config.DEFAULTS["portfolio"];
    daily_loss_limit <= 0 disables that rule.

    No leverage: the open notional (at entry) never exceeds max_total_notional (default
    MAX_TOTAL_NOTIONAL = 1.0) x realized equity. An entry that would exceed it is shrunk to the
    room left (its risk shrinks with it); with no room left it is skipped ("max_notional").
    """
    p = {**DEFAULTS["portfolio"], **(pcfg or {})}
    start = float(p["starting_equity"] if start_equity is None else start_equity)
    cap_x = p.get("max_total_notional")
    max_notional = MAX_TOTAL_NOTIONAL if cap_x is None else float(cap_x)
    events = []
    for k, tr in enumerate(trades):
        events.append((tr.entry_t, _ENTRY, tr.symbol, k))
        if tr.exit_t is not None:
            if tr.exit_t > tr.entry_t:
                events.append((tr.exit_t, _EXIT, tr.symbol, k))
            else:
                events.append((tr.entry_t, _SAME_BAR_EXIT, tr.symbol, k))
    events.sort()

    equity = start
    curve = [(events[0][0], start)] if events else []
    taken, skipped = [], []
    open_pos: dict[int, tuple[dict, float]] = {}   # trade index -> (taken record, initial risk $)
    day, day_equity, day_pnl = None, equity, 0.0
    limit = p["daily_loss_limit"] or 0.0
    for t, phase, _, k in events:
        if t // DAY != day:
            day, day_equity, day_pnl = t // DAY, equity, 0.0
        tr = trades[k]
        if phase != _ENTRY:
            if k in open_pos:
                rec = open_pos.pop(k)[0]
                rec["pnl"] = rec["notional"] * tr.ret_pct
                equity += rec["pnl"]
                day_pnl += rec["pnl"]
                curve.append((t, equity))
            continue
        reason, notional, new_risk = None, 0.0, 0.0
        if equity <= 0:
            reason = "no_equity"
        elif limit > 0 and day_pnl <= -limit * day_equity:
            reason = "daily_loss_limit"
        elif len(open_pos) >= p["max_open_positions"]:
            reason = "max_open"
        else:
            cap = p["max_notional_frac"] * equity
            notional = min(p["risk_per_trade"] * equity / tr.risk_pct, cap) if tr.risk_pct > 0 else cap
            room = max_notional * equity - sum(rec["notional"] for rec, _ in open_pos.values())
            if room <= 1e-9 * equity:
                reason = "max_notional"
            else:
                notional = min(notional, room)
                new_risk = notional * tr.risk_pct
                if sum(r for _, r in open_pos.values()) + new_risk > p["max_total_risk"] * equity + 1e-9:
                    reason = "max_total_risk"
        if reason:
            skipped.append({"trade": tr, "reason": reason})
            continue
        rec = {"trade": tr, "notional": notional, "pnl": 0.0}
        taken.append(rec)
        open_pos[k] = (rec, new_risk)

    marked = 0.0
    for k, (rec, _) in open_pos.items():
        rec["pnl"] = rec["notional"] * trades[k].ret_pct
        marked += rec["pnl"]
    return PortfolioResult(start_equity=start, final_equity=equity + marked, realized_equity=equity,
                           curve=curve, taken=taken, skipped=skipped)


def _mark_ret(side: int, p0: float, done_gross: float, rem: float, close: float, fee: float,
              slip: float) -> float:
    """Net return on the entry notional if the remaining fraction `rem` were closed at `close`
    (with exit slippage and fee) after fills worth done_gross (sum of fraction x price): the
    accounting of strategy._Position.result."""
    gross = done_gross + rem * close * (1 - side * slip)
    if side > 0:
        return gross / p0 * (1 - fee) - (1 + fee)
    return (1 - fee) - gross / p0 * (1 + fee)


def _position_marks(rec: dict, bars: Bars | None, fee: float, slip: float,
                    end_t: int | None) -> list[tuple[int, float]]:
    """[(t, dollar value)] of one taken position: its marked PnL at the close of every bar it is
    open (stamped with the bar's open time, like trade times), then its final pnl at its exit
    bar (an open trade: at the last bar before end_t). Without usable bars (missing symbol, or
    the entry / exit bar is not in them) only the final pnl at exit_t (closed trades) is known."""
    tr, notional, final = rec["trade"], rec["notional"], rec["pnl"]
    if notional == 0:
        return []
    t_list = bars.t if bars is not None else []
    i0 = bisect.bisect_left(t_list, tr.entry_t)
    if tr.exit_t is not None:
        ix = bisect.bisect_left(t_list, tr.exit_t)
        usable = ix < len(t_list) and t_list[ix] == tr.exit_t
    else:
        ix = (len(t_list) if end_t is None else bisect.bisect_left(t_list, end_t)) - 1
        usable = ix >= i0
    if not (usable and i0 < len(t_list) and t_list[i0] == tr.entry_t and tr.entry_price > 0):
        return [(tr.exit_t, final)] if tr.exit_t is not None else []
    fills = [f for f in (tr.meta or {}).get("fills") or [] if len(f) >= 4 and f[2] != "open"]
    s, p0, out = tr.side, tr.entry_price, []
    for i in range(i0, ix):
        ti = t_list[i]
        done = [f for f in fills if f[3] <= ti]
        rem = max(0.0, 1.0 - sum(f[0] for f in done))
        out.append((ti, notional * _mark_ret(s, p0, sum(f[0] * f[1] for f in done), rem,
                                             bars.c[i], fee, slip)))
    out.append((t_list[ix], final))
    return out


def marked_curve(pf: PortfolioResult, bars_by_symbol: dict[str, Bars], fee_rate: float,
                 slippage: float, end_t: int | None = None) -> list[tuple[int, float]]:
    """Mark-to-market equity of the account behind `pf`: one point (t, equity) for every bar time
    at which a taken position was open or closed, equity = start_equity + every position's value
    at the close of that bar. A position is worth its marked PnL while open (fills done so far, tp1
    partials included, at their prices; the rest at the bar's close with exit slippage and the
    exit fee, like an open trade's mark) and its pnl from its exit bar on, so the last point is
    final_equity when bars cover every position. Points carry the bar's open time t (the
    convention of trade times; the value is the bar's close). Sizing and the daily loss rule stay
    on realized equity (simulate_portfolio); this series is for drawdown and return statistics.
    Positions whose bars are missing contribute their pnl at exit_t only (as the realized curve).
    end_t bounds the marks of open trades (their mark is the last close before end_t)."""
    deltas: dict[int, float] = {}
    for rec in pf.taken:
        bars = bars_by_symbol.get(rec["trade"].symbol)
        prev = 0.0
        for t, v in _position_marks(rec, bars, fee_rate, slippage, end_t):
            deltas[t] = deltas.get(t, 0.0) + (v - prev)
            prev = v
    out, equity = [], pf.start_equity
    for t in sorted(deltas):
        equity += deltas[t]
        out.append((t, equity))
    return out
