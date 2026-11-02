"""Graduation gate: a deterministic statistical verdict on a strategy's PAPER results.

Graduation is a report. It never enables live trading (fibbot has no live trading code).
See DESIGN.md, "fibbot/graduation.py", for the checks and the status rules.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .metrics import bootstrap_mean_ci, deflated_sharpe, trade_stats
from .paper import account_drawdown, paper_days, rounded

if TYPE_CHECKING:
    from .store import Store


def validation_mean_r(validation) -> float | None:
    """Mean R of the validation stored with a roster row: validation["metrics"]["avg_r"], else
    validation["avg_r"]; None when neither is a finite number."""
    if not isinstance(validation, dict):
        return None
    metrics = validation.get("metrics")
    for v in (metrics.get("avg_r") if isinstance(metrics, dict) else None, validation.get("avg_r")):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            return float(v)
    return None


def monthly_r(trades) -> dict[str, float]:
    """{"YYYY-MM": summed r_multiple} by UTC exit month, for months with >= 1 closed trade."""
    out: dict[str, float] = {}
    for tr in trades:
        month = datetime.fromtimestamp(tr.exit_t, tz=timezone.utc).strftime("%Y-%m")
        out[month] = out.get(month, 0.0) + tr.r_multiple
    return out


def _check(name: str, value, threshold, passed: bool, note: str | None = None) -> dict:
    d = {"name": name, "value": rounded(value), "threshold": rounded(threshold), "passed": bool(passed)}
    if note:
        d["note"] = note
    return d


def evaluate(store: Store, cfg: dict, strategy_id: str, now_t: int | None = None) -> dict:
    """Graduation checks for one roster strategy (KeyError if it is not on the roster).

    Returns {"strategy_id", "status", "checks": [{"name", "value", "threshold", "passed"}],
    "summary", "mean_r_ci": [lo, hi]}; status is "graduated" | "failing" | "collecting" |
    "promising". Trade checks use every closed paper trade (R does not depend on sizing); the
    drawdown uses the sized account (paper.account_drawdown). A positive month is one whose
    closed trades sum to R > 0.
    """
    row = store.get_strategy(strategy_id)
    if row is None:
        raise KeyError(strategy_id)
    g = cfg["graduation"]
    rows = store.paper_trades(strategy_id)
    closed = sorted((store.trade_from_row(r) for r in rows if r["exit_t"] is not None),
                    key=lambda tr: (tr.entry_t, tr.symbol, tr.leg_id))   # chronological
    rs = [tr.r_multiple for tr in closed]
    n, stats, days = len(closed), trade_stats(closed), paper_days(row, now_t)
    ci_lo, ci_hi = bootstrap_mean_ci(rs, level=g["ci_level"])
    dd = account_drawdown(float(cfg["paper"]["starting_equity"]), store.equity_curve(strategy_id), rows)
    months = monthly_r(closed)
    pos_months = sum(r > 0 for r in months.values()) / len(months) if months else 0.0
    dsr = deflated_sharpe(rs, n_trials=store.count_ever_on_paper())
    pf, avg_r = stats["profit_factor"], stats["avg_r"]
    checks = [
        _check("min_days", days, g["min_days"], days >= g["min_days"]),
        _check("min_trades", n, g["min_trades"], n >= g["min_trades"]),
        _check("min_profit_factor", pf, g["min_profit_factor"], pf >= g["min_profit_factor"]),
        _check("mean_r_ci_lower", ci_lo, 0.0, ci_lo > 0),
        _check("max_drawdown", dd, g["max_drawdown"], dd <= g["max_drawdown"]),
        _check("min_positive_month_frac", pos_months, g["min_positive_month_frac"],
               pos_months >= g["min_positive_month_frac"]),
        _check("min_deflated_sharpe", dsr, g["min_deflated_sharpe"], dsr >= g["min_deflated_sharpe"]),
    ]
    val_r = validation_mean_r(row["validation"])
    if val_r is None:
        checks.append(_check("max_shortfall_vs_validation_r", None, None, True,
                             "no validation mean R on record; check skipped"))
    else:
        floor = val_r - g["max_shortfall_vs_validation_r"]
        checks.append(_check("max_shortfall_vs_validation_r", avg_r, floor, avg_r >= floor))
    failed = [c["name"] for c in checks if not c["passed"]]
    if not failed:
        status = "graduated"
    elif (n >= g["min_trades"] / 2 and pf < 1.0 and ci_hi < 0) or dd > 1.5 * g["max_drawdown"]:
        status = "failing"
    elif days < g["min_days"] or n < g["min_trades"]:
        status = "collecting"
    else:
        status = "promising"
    summary = (f"{status}: {len(checks) - len(failed)}/{len(checks)} checks passed; {n} closed"
               f" trades over {days:.1f} days, mean R {avg_r:+.3f}, PF {pf:.2f}, max DD {dd:.1%}")
    if failed:
        summary += "; not yet: " + ", ".join(failed)
    return {"strategy_id": strategy_id, "status": status, "checks": checks, "summary": summary,
            "mean_r_ci": rounded([ci_lo, ci_hi])}
