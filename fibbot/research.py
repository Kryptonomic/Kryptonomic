"""Research protocol (pure, no network): research/validation windows, walk-forward folds,
evaluation, the random-entry baseline, walk-forward optimization, parameter-neighbour robustness
and the locked-validation gate.

Statistics use CLOSED trades (see backtest.py). Every dict these functions return is JSON-safe
and rounded with backtest.rounded; decisions (selection, pass/fail) are made on unrounded values.
"""
from __future__ import annotations

import itertools
import math
import random
from statistics import fmean, median, stdev

from .backtest import BacktestResult, as_spec, backtest, costs_from_cfg, rounded
from .config import DEFAULTS
from .indicators import atr
from .metrics import deflated_sharpe, trade_stats
from .strategy import PARAM_SPACE, StrategySpec, manage_position, warmup_bars
from .types import TF_SECONDS, Bars, Trade

DAY = 86400
MONTH_DAYS = 30.44
OBJECTIVES = ("avg_r", "profit_factor", "total_r")
FOLD_KEYS = ("n", "avg_r", "total_r", "profit_factor")
# Choice params that change WHAT is traded (other data / other side), not how: never neighbours.
FIXED_PARAMS = frozenset({"timeframe", "direction"})


def compute_windows(cfg: dict, bars_by_symbol: dict[str, Bars]) -> dict:
    """Research / locked validation windows. data_end = min over symbols of (last t + tf);
    validation_start = data_end - validation_months * 30.44 days, floored to a UTC day;
    research = [first t of any symbol, validation_start); validation = [validation_start, data_end)."""
    live = [b for b in bars_by_symbol.values() if len(b)]
    if not live:
        raise ValueError("compute_windows: no bars")
    data_start = min(b.t[0] for b in live)
    data_end = min(b.t[-1] + TF_SECONDS[b.tf] for b in live)
    vs = int(data_end - cfg["research"]["validation_months"] * MONTH_DAYS * DAY)
    vs -= vs % DAY
    return {"data_start": data_start, "data_end": data_end, "research_start": data_start,
            "research_end": vs, "validation_start": vs, "validation_end": data_end}


def window_bounds(windows: dict, window: str) -> tuple[int, int]:
    """(start_t, end_t) of "research" | "validation" | "all" (= data_start..data_end)."""
    keys = {"research": ("research_start", "research_end"),
            "validation": ("validation_start", "validation_end"),
            "all": ("data_start", "data_end")}
    if window not in keys:
        raise ValueError(f"window must be one of {sorted(keys)}, got {window!r}")
    a, b = keys[window]
    return windows[a], windows[b]


def fold_bounds(start_t: int, end_t: int, folds: int) -> list[tuple[int, int]]:
    """`folds` contiguous, equal-length [a, b) intervals covering [start_t, end_t)."""
    if folds < 1:
        raise ValueError(f"folds must be >= 1, got {folds}")
    if end_t <= start_t:
        raise ValueError(f"empty window: end_t {end_t} <= start_t {start_t}")
    edges = [start_t + (end_t - start_t) * k // folds for k in range(folds + 1)]
    return list(zip(edges, edges[1:]))


def _bucket(trades: list[Trade], bounds: list[tuple[int, int]]) -> list[list[Trade]]:
    """Trades per fold by entry_t (a <= entry_t < b); trades outside every fold are dropped."""
    return [[tr for tr in trades if a <= tr.entry_t < b] for a, b in bounds]


def _fold_row(k: int, bound: tuple[int, int], trades: list[Trade]) -> dict:
    s = trade_stats(trades)
    return {"fold": k, "start": bound[0], "end": bound[1], **{key: s[key] for key in FOLD_KEYS}}


def fold_stats(trades: list[Trade], bounds: list[tuple[int, int]]) -> list[dict]:
    """[{"fold", "start", "end", "n", "avg_r", "total_r", "profit_factor"}] per fold (by entry_t)."""
    return [_fold_row(k, b, ts) for k, (b, ts) in enumerate(zip(bounds, _bucket(trades, bounds)))]


def evaluate(spec: StrategySpec | dict, bars_by_symbol: dict[str, Bars], cfg: dict, windows: dict,
             window: str = "research") -> tuple[BacktestResult, dict]:
    """Backtest on a window and split its closed trades into cfg research.folds folds.
    extras = {"folds", "fold_positive_frac" (share of ALL folds with avg_r > 0; an empty fold is not
    positive), "min_fold_avg_r" (over all folds; an empty fold counts as 0)}."""
    start, end = window_bounds(windows, window)
    result = backtest(spec, bars_by_symbol, start, end, cfg)
    folds = fold_stats(result.closed_trades, fold_bounds(start, end, cfg["research"]["folds"]))
    return result, rounded({
        "folds": folds,
        "fold_positive_frac": sum(f["avg_r"] > 0 for f in folds) / len(folds),
        "min_fold_avg_r": min(f["avg_r"] for f in folds)})


def random_baseline(spec: StrategySpec | dict, bars_by_symbol: dict[str, Bars], cfg: dict,
                    start_t: int | None, end_t: int | None, n_runs: int = 10, seed: int = 1,
                    result: BacktestResult | None = None) -> dict:
    """Random-entry baseline with the strategy's own risk and exits. For every closed trade of
    `result` (backtest of spec on the same window when None) each run draws an entry bar uniformly
    from the window bars of the same symbol (any but the last), enters at the open with slippage on
    the same side, puts the stop at the same risk_pct and tp1/tp2 at the same R distances, and
    manages it with strategy.manage_position (an exit-less position is marked at the window end).
    Deterministic for a given seed."""
    spec = as_spec(spec)
    if result is None:
        result = backtest(spec, bars_by_symbol, start_t, end_t, cfg)
    costs = costs_from_cfg(cfg)
    trades = [tr for tr in result.closed_trades if tr.risk_pct > 0]
    windowed: dict[str, tuple[Bars, list[float], int]] = {}
    for sym in sorted({tr.symbol for tr in trades}):
        w = bars_by_symbol[sym].window(start_t, end_t, warmup_bars(spec))
        i0 = 0 if start_t is None else w.index_at_or_after(start_t)
        windowed[sym] = (w, atr(w.h, w.l, w.c, spec.swing_atr_len), i0)
    rng = random.Random(seed)
    runs = []
    for _ in range(n_runs if trades else 0):
        rs = []
        for tr in trades:
            w, atr_w, i0 = windowed[tr.symbol]
            if len(w) - 1 <= i0:
                continue
            s, risk = tr.side, abs(tr.entry_price - tr.stop_price)
            k1, k2 = (s * (tr.meta[k] - tr.entry_price) / risk for k in ("tp1", "tp2"))
            e = rng.randrange(i0, len(w) - 1)
            px = w.o[e] * (1 + s * costs.slippage)
            dist = px * tr.risk_pct
            rs.append(manage_position(w, atr_w, e, s, px, px - s * dist, px + s * k1 * dist,
                                      px + s * k2 * dist, spec, costs)["r_multiple"])
        runs.append(fmean(rs) if rs else 0.0)
    strat = result.metrics["avg_r"]
    mean = fmean(runs) if runs else 0.0
    return rounded({
        "n_trades": len(trades), "n_runs": n_runs, "seed": seed, "strategy_avg_r": strat,
        "baseline_avg_r_mean": mean, "baseline_avg_r_std": stdev(runs) if len(runs) > 1 else 0.0,
        "beats_frac": sum(strat > r for r in runs) / len(runs) if runs else 0.0,
        "runs": runs, "edge_r": strat - mean})


def _row(spec: StrategySpec, keys, result: BacktestResult, extras: dict, min_n: int) -> dict:
    """Ranking row of one evaluated spec (params = the given keys)."""
    m = result.metrics
    return {"params": {k: getattr(spec, k) for k in keys}, "spec_id": result.spec_id,
            "n": m["n"], "avg_r": m["avg_r"], "total_r": m["total_r"],
            "profit_factor": m["profit_factor"], "win_rate": m["win_rate"],
            "max_drawdown": m["max_drawdown"], "fold_positive_frac": extras["fold_positive_frac"],
            "min_fold_avg_r": extras["min_fold_avg_r"], "eligible": m["n"] >= min_n}


def walk_forward_optimize(base_spec: StrategySpec | dict, grid: dict[str, list],
                          bars_by_symbol: dict[str, Bars], cfg: dict, windows: dict,
                          objective: str = "avg_r", on_combo=None) -> dict:
    """Grid search with an anchored walk-forward check on the research window.

    Every combo of `grid` (at most research.max_grid_size, else ValueError) that passes
    validate() is evaluated once (invalid combos are counted and skipped, duplicates evaluated
    once) and on_combo(spec, result) is called. For fold k = 1..F-1 the combo with the best
    objective on folds [0, k) among combos with >= max(5, min_trades // F) trades there (ties:
    more trades, then grid order) contributes its fold-k trades to the out-of-sample record.
    "best"/"top" rank the whole research window: combos with >= research.min_trades trades first,
    then by objective and trade count."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    base, rcfg = as_spec(base_spec), cfg["research"]
    keys = list(grid)
    unknown = sorted(set(keys) - set(PARAM_SPACE))
    if unknown:
        raise ValueError(f"grid: unknown parameters {unknown}")
    if any(not isinstance(grid[k], (list, tuple)) or not grid[k] for k in keys):
        raise ValueError("grid: every parameter needs a non-empty list of values")
    if any(v != base.timeframe for v in grid.get("timeframe", ())):
        raise ValueError("grid: timeframe must stay the base spec's (bars have one timeframe)")
    values = [list(grid[k]) for k in keys]
    n_combos, folds = math.prod(len(v) for v in values), rcfg["folds"]
    if n_combos > rcfg["max_grid_size"]:
        raise ValueError(f"grid has {n_combos} combos > research.max_grid_size {rcfg['max_grid_size']}")
    if folds < 2:
        raise ValueError("walk-forward needs research.folds >= 2")
    bounds = fold_bounds(windows["research_start"], windows["research_end"], folds)
    evaluated, invalid, seen = [], [], set()
    for combo in itertools.product(*values):
        try:
            spec = base.with_overrides(**dict(zip(keys, combo))).validate()
        except ValueError as e:
            invalid.append(str(e))
            continue
        if spec.spec_id() in seen:
            continue
        seen.add(spec.spec_id())
        result, extras = evaluate(spec, bars_by_symbol, cfg, windows, "research")
        if on_combo is not None:
            on_combo(spec, result)
        evaluated.append((spec, result, _row(spec, keys, result, extras, rcfg["min_trades"]),
                          _bucket(result.closed_trades, bounds)))

    min_is = max(5, rcfg["min_trades"] // folds)
    oos, oos_by_fold, chosen = [], [], []
    for k in range(1, folds):
        best = None
        for _, _, row, by_fold in evaluated:
            st = trade_stats([tr for f in by_fold[:k] for tr in f])
            if st["n"] >= min_is and (best is None or (st[objective], st["n"]) > best[0]):
                best = ((st[objective], st["n"]), row, st, by_fold[k])
        if best is None:
            chosen.append({"fold": k, "params": None, "spec_id": None, "is_n": 0, "is_avg_r": None,
                           "is_objective": None})
            oos_by_fold.append(_fold_row(k, bounds[k], []))
            continue
        _, row, st, fold_trades = best
        chosen.append({"fold": k, "params": row["params"], "spec_id": row["spec_id"],
                       "is_n": st["n"], "is_avg_r": st["avg_r"], "is_objective": st[objective]})
        oos_by_fold.append(_fold_row(k, bounds[k], fold_trades))
        oos.extend(fold_trades)

    ranked = sorted(evaluated, key=lambda e: (e[2]["eligible"], e[2][objective], e[2]["n"]),
                    reverse=True)
    sensitivity = {}
    for k in keys:
        groups: dict[str, list[float]] = {}
        for spec, _, row, _ in evaluated:
            groups.setdefault(str(getattr(spec, k)), []).append(row["avg_r"])
        if len(groups) > 1:
            sensitivity[k] = {v: fmean(rs) for v, rs in groups.items()}
    top_keys = ("params", "spec_id", "n", "avg_r", "profit_factor", "fold_positive_frac")
    return rounded({
        "objective": objective, "n_combos": n_combos, "n_invalid": len(invalid),
        "invalid_reasons": list(dict.fromkeys(invalid))[:3], "n_evaluated": len(evaluated),
        "n_folds": folds, "min_is_trades": min_is,
        "oos": trade_stats(oos), "oos_by_fold": oos_by_fold, "chosen": chosen,
        "best": {**ranked[0][2], "spec": ranked[0][0].to_dict()} if ranked else None,
        "top": [{k: e[2][k] for k in top_keys} for e in ranked[:5]],
        "sensitivity": sensitivity})


def _inactive(spec: StrategySpec) -> set[str]:
    """Parameters that cannot change this spec's trades (moving them would duplicate it)."""
    off = {"swing_pct" if spec.swing_mode == "atr" else "swing_mult"}
    if spec.trend_filter == "none":
        off.add("trend_ema")
    if spec.entry_mode == "limit":
        off |= {"confirm_type", "confirm_window"}
        if spec.entry_at == "hi":                 # zone_top only matters for confirm touches
            off.add("entry_zone_lo")
    if spec.entry_at == "lo":                     # the deep edge only enters via the entry ratio
        off.add("entry_zone_hi")
    if spec.tp1_frac >= 1.0:                      # the trade closes at tp1: no tp2, no trailing
        off |= {"tp2_ext", "trail_atr"}
    return off


def neighbours(spec: StrategySpec | dict) -> list[StrategySpec]:
    """Valid specs that move ONE parameter one step: numeric params by -/+ PARAM_SPACE step, choice
    params to the previous/next choice. timeframe, direction, bools and parameters inactive for
    this spec are not moved. Order: PARAM_SPACE order, the lower move first."""
    spec = as_spec(spec)
    skip = FIXED_PARAMS | _inactive(spec)
    out = []
    for k, p in PARAM_SPACE.items():
        if k in skip or p["type"] == "bool":
            continue
        v = getattr(spec, k)
        if p["type"] == "choice":
            i = p["choices"].index(v)
            moves = [p["choices"][j] for j in (i - 1, i + 1) if 0 <= j < len(p["choices"])]
        else:
            moves = [v - p["step"], v + p["step"]]
        for nv in moves:
            try:
                out.append(spec.with_overrides(**{k: nv}).validate())
            except ValueError:
                pass
    return out


def robustness(spec: StrategySpec | dict, bars_by_symbol: dict[str, Bars], cfg: dict, windows: dict,
               max_neighbours: int = 16, on_combo=None) -> dict:
    """Evaluate neighbours(spec) on the research window (an evenly spaced subset of at most
    max_neighbours when there are more). on_combo(spec, result) is called per neighbour.
    positive_frac = share of neighbours with avg_r > 0 (no trades is not positive)."""
    spec = as_spec(spec)
    cands = neighbours(spec)
    m = max(0, max_neighbours)
    picked = cands if len(cands) <= m else [cands[i * len(cands) // m] for i in range(m)]
    base, rows = spec.to_dict(), []
    for nb in picked:
        result, extras = evaluate(nb, bars_by_symbol, cfg, windows, "research")
        if on_combo is not None:
            on_combo(nb, result)
        param = next(k for k, v in nb.to_dict().items() if v != base[k])
        mt = result.metrics
        rows.append({"param": param, "value": getattr(nb, param), "spec_id": result.spec_id,
                     "n": mt["n"], "avg_r": mt["avg_r"], "total_r": mt["total_r"],
                     "profit_factor": mt["profit_factor"],
                     "fold_positive_frac": extras["fold_positive_frac"]})
    avg = [r["avg_r"] for r in rows]
    return rounded({
        "n": len(rows), "n_candidates": len(cands),
        "positive_frac": sum(a > 0 for a in avg) / len(avg) if avg else 0.0,
        "median_avg_r": median(avg) if avg else 0.0, "worst_avg_r": min(avg, default=0.0),
        "rows": rows})


def validation_gate(result: BacktestResult, gate: dict, n_trials: int, var_sr: float | None) -> dict:
    """Deterministic pass/fail of a (validation-window) backtest. Checks, thresholds from `gate`
    (missing keys: config defaults): min_trades (closed), min_profit_factor, min_expectancy_r,
    min_deflated_sharpe (deflated_sharpe of the closed trades' R multiples with n_trials, var_sr),
    max_drawdown (portfolio equity), min_symbol_win_frac (share of symbols WITH closed trades whose
    total R is > 0)."""
    g = {**DEFAULTS["research"]["validation_gate"], **(gate or {})}
    m = result.metrics
    dsr = deflated_sharpe([tr.r_multiple for tr in result.closed_trades], n_trials, var_sr)
    traded = [s for s in result.per_symbol.values() if s["n"] > 0]
    sym_frac = sum(s["total_r"] > 0 for s in traded) / len(traded) if traded else 0.0
    checks = [(name, value, g[name], value <= g[name] if name.startswith("max_") else value >= g[name])
              for name, value in (("min_trades", m["n"]), ("min_profit_factor", m["profit_factor"]),
                                  ("min_expectancy_r", m["expectancy_r"]),
                                  ("min_deflated_sharpe", dsr), ("max_drawdown", m["max_drawdown"]),
                                  ("min_symbol_win_frac", sym_frac))]
    return rounded({"passed": all(c[3] for c in checks), "deflated_sharpe": dsr,
                    "checks": [{"name": n, "value": v, "threshold": t, "passed": p}
                               for n, v, t, p in checks]})
