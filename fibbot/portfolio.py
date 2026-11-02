"""Portfolio simulation: sizes a list of single-position trades into one shared account.

Pure and deterministic. Positions are sized from REALIZED equity at entry time and limited by the
risk rules in cfg["portfolio"]; open trades (exit_t None) keep their slot and risk until the end
and are marked to their (already marked) ret_pct in final_equity only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULTS
from .types import Trade

DAY = 86400
_EXIT, _ENTRY, _SAME_BAR_EXIT = 0, 1, 2   # event phases at equal timestamps


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

    Event order: by time; at equal timestamps exits of earlier positions first, then entries (by
    symbol, then input order), then exits of trades that closed on their own entry bar (those hold
    their slot and risk while the other entries of that timestamp are decided). Missing pcfg keys
    fall back to config.DEFAULTS["portfolio"]; daily_loss_limit <= 0 disables that rule.
    """
    p = {**DEFAULTS["portfolio"], **(pcfg or {})}
    start = float(p["starting_equity"] if start_equity is None else start_equity)
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
