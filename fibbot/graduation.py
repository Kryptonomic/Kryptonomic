"""Graduation gate: a deterministic statistical verdict on a strategy's PAPER results.

Graduation is a report. It never enables live trading (fibbot has no live trading code).
See DESIGN.md, "fibbot/graduation.py", for the checks and the status rules.

The scheduler re-evaluates every paper strategy on every pass over its growing record, so the
evidence checks must stay honest under continuous monitoring:
- the mean-R interval is an anytime-valid confidence sequence (metrics.confidence_sequence), not
  a fixed-sample CI. The chance that a zero-edge strategy EVER passes "lower end > 0" while it is
  re-checked after every trade stays below about (1 - ci_level) / 2 (2.5% by default); a fixed
  95% CI re-tested on every pass let about 1 in 10 zero-edge strategies graduate within a year.
- trades are clustered by UTC entry week (correlated coins trade together), so the interval uses
  the cluster-robust variance (effective trade count n_eff) and so does the deflated Sharpe.

With the strategy's candles (Lab.graduation always passes them) two more things hold:
- min_days counts paper days only up to the newest bar the strategy could trade, so days without
  candles (a download outage) do not count toward graduation;
- the paper record must beat random entries over the paper period (research.random_baseline with
  the same trades' risk and exits, uniform and trend/ADX-regime draws; the binding one counts):
  min_edge_r and min_edge_z, so a long strategy cannot graduate on a bull market's drift alone.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .backtest import result_from_trades
from .metrics import (confidence_sequence, deflated_sharpe, design_effect, entry_week, t_stat_mean,
                      trade_stats)
from .paper import account_drawdown, account_settings, paper_days, rounded
from .research import _gate_edge, random_baseline
from .strategy import StrategySpec
from .types import TF_SECONDS

if TYPE_CHECKING:
    from .store import Store

CS_TARGET_TRADES = 200   # the anytime-valid mean-R interval is narrowest near this many closed trades
TIMEOUT_FACTOR = 3       # a mature record with mean R <= 0 after TIMEOUT_FACTOR x min_days is failing
EDGE_BASELINE_RUNS = 50  # random-entry runs behind the min_edge_r / min_edge_z checks
EDGE_CACHE_SIZE = 32
DAY = 86400
STATUSES = ("collecting", "promising", "lagging", "failing", "graduated")


def _iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


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
        dt = datetime.fromtimestamp(tr.exit_t, tz=timezone.utc)
        month = f"{dt.year:04d}-{dt.month:02d}"
        out[month] = out.get(month, 0.0) + tr.r_multiple
    return out


def _check(name: str, value, threshold, passed: bool, note: str | None = None) -> dict:
    d = {"name": name, "value": rounded(value), "threshold": rounded(threshold), "passed": bool(passed)}
    if note:
        d["note"] = note
    return d


def paper_edge(cfg: dict, row: dict, closed: list, bars_by_symbol: dict, end_t: int,
               cache: dict | None = None) -> dict | None:
    """Edge of the closed paper trades over random entries on the paper period [paper_start_t,
    end_t) (research.random_baseline, EDGE_BASELINE_RUNS runs, with the trades' own risk and
    exits): research._gate_edge's {"edge_r", "edge_z", ...} (the smaller of the uniform and the
    regime-filtered edge), or None with fewer than 2 closed trades on symbols with bars. `cache`
    (a dict) reuses the result while the record and end_t are unchanged."""
    trades = [tr for tr in closed if tr.symbol in bars_by_symbol]
    if len(trades) < 2:
        return None
    start = int(row["paper_start_t"])
    key = (row["strategy_id"], start, len(trades), max(tr.exit_t for tr in trades), int(end_t))
    if cache is not None and key in cache:
        return cache[key]
    spec = StrategySpec.from_dict(row["spec"])
    symbols = sorted({tr.symbol for tr in trades})
    res = result_from_trades(spec, trades, symbols, start, end_t, cfg)
    base = random_baseline(spec, {s: bars_by_symbol[s] for s in symbols}, cfg, start, end_t,
                           n_runs=EDGE_BASELINE_RUNS, result=res)
    edge = {**_gate_edge(res, base), "baseline_avg_r_mean": base["baseline_avg_r_mean"],
            "edge_regime_r": base["edge_regime_r"]}
    if cache is not None:
        while len(cache) >= EDGE_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        cache[key] = edge
    return edge


def evaluate(store: Store, cfg: dict, strategy_id: str, now_t: int | None = None,
             bars_by_symbol: dict | None = None, cache: dict | None = None) -> dict:
    """Graduation checks for one roster strategy (KeyError if it is not on the roster).

    bars_by_symbol: the strategy's candles (timeframe of its spec). With them, min_days counts
    days up to min(now, the newest bar end) and the min_edge_r / min_edge_z checks (paper_edge;
    skipped for a null threshold) are added; without them days run on the wall clock and the
    edge checks are absent. The drawdown starts from the account's frozen starting equity
    (paper.account_settings).

    Returns {"strategy_id", "status", "checks": [{"name", "value", "threshold", "passed"[, "note"]}],
    "summary", "mean_r_ci": [lo, hi], "evidence": {...}}; status is one of STATUSES. Trade checks
    use every closed paper trade (R does not depend on sizing); the drawdown uses the sized account
    (paper.account_drawdown). A positive month is one whose closed trades sum to R > 0.
    mean_r_ci is the anytime-valid ci_level confidence sequence of mean R with trades clustered by
    UTC entry week; evidence = {"n_eff", "design_effect", "weeks", "t_clustered"}. Every check
    named max_* passes when value <= threshold and every min_* check when value >= threshold.

    Status (first match): "graduated" when every check passes; "failing" when the drawdown is above
    1.5 x max_drawdown, or (n >= min_trades / 2, profit factor < 1 and the interval's upper end < 0),
    or, once mature (days >= min_days and n >= min_trades), when the upper end is below the
    validation floor (validation mean R - max_shortfall_vs_validation_r) or the record is still
    not profitable (mean R <= 0) after TIMEOUT_FACTOR x min_days; "collecting" while not mature;
    "lagging" when mature but mean R <= 0; "promising" otherwise (mature, profitable, not proven).
    """
    row = store.get_strategy(strategy_id)
    if row is None:
        raise KeyError(strategy_id)
    g = cfg["graduation"]
    rows = store.paper_trades(strategy_id)
    closed = sorted((store.trade_from_row(r) for r in rows if r["exit_t"] is not None),
                    key=lambda tr: (tr.entry_t, tr.symbol, tr.leg_id))   # chronological
    rs = [tr.r_multiple for tr in closed]
    weeks = [entry_week(tr.entry_t) for tr in closed]
    now = int(time.time()) if now_t is None else int(now_t)
    step = TF_SECONDS.get(str((row.get("spec") or {}).get("timeframe")), 0)
    ends = [int(b.t[-1]) + step for b in (bars_by_symbol or {}).values() if len(b)]
    # a wall-clock day without candles (an outage) is not a day the strategy could trade
    end_t = now if bars_by_symbol is None else min(now, max(ends, default=int(row["paper_start_t"])))
    n, stats, days = len(closed), trade_stats(closed), paper_days(row, end_t)
    deff = design_effect(rs, weeks)
    n_eff = n / deff
    ci_lo, ci_hi = confidence_sequence(rs, level=g["ci_level"], groups=weeks, target_n=CS_TARGET_TRADES)
    start_equity = float(account_settings(store, cfg, row)["starting_equity"])   # frozen per account
    dd = account_drawdown(start_equity, store.equity_curve(strategy_id), rows)
    months = monthly_r(closed)
    pos_months = sum(r > 0 for r in months.values()) / len(months) if months else 0.0
    dsr = deflated_sharpe(rs, n_trials=store.count_ever_on_paper(), n_eff=n_eff)
    pf, avg_r = stats["profit_factor"], stats["avg_r"]
    t_clustered = t_stat_mean(rs) / math.sqrt(deff)
    checks = [
        _check("min_days", days, g["min_days"], days >= g["min_days"],
               None if now - end_t <= DAY else f"days up to the newest bar ({_iso(end_t)});"
                                               f" {paper_days(row, now):.1f} calendar days"),
        _check("min_trades", n, g["min_trades"], n >= g["min_trades"]),
        _check("min_profit_factor", pf, g["min_profit_factor"], pf >= g["min_profit_factor"]),
        _check("mean_r_ci_lower", ci_lo, 0.0, ci_lo > 0,
               f"anytime-valid {g['ci_level']:.0%} interval, trades clustered by entry week"
               f" (n_eff {n_eff:.1f} of {n})"),
        _check("max_drawdown", dd, g["max_drawdown"], dd <= g["max_drawdown"]),
        _check("min_positive_month_frac", pos_months, g["min_positive_month_frac"],
               pos_months >= g["min_positive_month_frac"]),
        _check("min_deflated_sharpe", dsr, g["min_deflated_sharpe"], dsr >= g["min_deflated_sharpe"]),
    ]
    if bars_by_symbol is not None:
        edge = paper_edge(cfg, row, closed, bars_by_symbol, end_t, cache)
        for name, key, fmt in (("min_edge_r", "edge_r", "{:+.3f} R"), ("min_edge_z", "edge_z", "{:.2f}")):
            threshold = g.get(name)
            if threshold is None:
                continue
            if edge is None:
                checks.append(_check(name, None, threshold, False, "fewer than 2 closed trades"))
                continue
            checks.append(_check(name, edge[key], threshold, edge[key] >= threshold,
                                 f"paper mean R vs random entries over the paper period (binding of"
                                 f" uniform and regime draws): edge {fmt.format(edge[key])}"))
    val_r = validation_mean_r(row["validation"])
    max_short = g["max_shortfall_vs_validation_r"]
    if val_r is None:
        checks.append(_check("max_shortfall_vs_validation_r", None, None, True,
                             "no validation mean R on record; check skipped"))
    else:
        shortfall = val_r - avg_r          # how far paper mean R trails the validation mean R
        checks.append(_check("max_shortfall_vs_validation_r", shortfall, max_short, shortfall <= max_short,
                             f"paper mean R {avg_r:+.3f} vs validation mean R {val_r:+.3f}"))
    failed = [c["name"] for c in checks if not c["passed"]]
    mature = days >= g["min_days"] and n >= g["min_trades"]
    why = []
    if dd > 1.5 * g["max_drawdown"]:
        why.append(f"drawdown {dd:.1%} > 1.5 x max_drawdown")
    if n >= g["min_trades"] / 2 and pf < 1.0 and ci_hi < 0:
        why.append("significant loss (profit factor < 1 and mean R interval below 0)")
    if mature and val_r is not None and ci_hi < val_r - max_short:
        why.append(f"mean R interval below the validation floor {val_r - max_short:+.3f}")
    if mature and days >= TIMEOUT_FACTOR * g["min_days"] and avg_r <= 0:
        why.append(f"mean R <= 0 after {days:.0f} days (>= {TIMEOUT_FACTOR} x min_days)")
    if not failed:
        status = "graduated"
    elif why:
        status = "failing"
    elif not mature:
        status = "collecting"
    elif avg_r <= 0:
        status = "lagging"
    else:
        status = "promising"
    summary = (f"{status}: {len(checks) - len(failed)}/{len(checks)} checks passed; {n} closed"
               f" trades over {days:.1f} days, mean R {avg_r:+.3f} (clustered t {t_clustered:.2f},"
               f" n_eff {n_eff:.0f}), PF {pf:.2f}, max DD {dd:.1%}")
    if failed:
        summary += "; not yet: " + ", ".join(failed)
    if status == "failing":
        summary += "; failing: " + "; ".join(why)
    return {"strategy_id": strategy_id, "status": status, "checks": checks, "summary": summary,
            "mean_r_ci": rounded([ci_lo, ci_hi]),
            "evidence": rounded({"n_eff": n_eff, "design_effect": deff, "weeks": len(set(weeks)),
                                 "t_clustered": t_clustered})}
