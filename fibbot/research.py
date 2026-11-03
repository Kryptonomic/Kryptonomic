"""Research protocol (pure, no network): research/validation windows, walk-forward folds,
evaluation, the random-entry baseline, walk-forward optimization, parameter-neighbour robustness
and the locked-validation gate.

Statistics use CLOSED trades (see backtest.py). Every dict these functions return is JSON-safe
and rounded with backtest.rounded; decisions (selection, pass/fail) are made on unrounded values.

Honesty rules implemented here:
- the validation window can be pinned (compute_windows(validation_end=...)): bars after it are
  forward / paper data only, so the holdout never slides over results the brain has seen;
- the random-entry baseline pays the strategy's own entry cost (no slippage for a resting limit)
  and also draws entries only where the strategy's trend / ADX filter passed (regime baseline), so
  the edge measures entry timing, not cost asymmetry or the filter's regime;
- validation_gate can require an edge over random entries on the validation window that is
  significant (z test against the baseline, cluster-robust strategy error), so a bull market's
  drift cannot pass the gate on its own; its deflated Sharpe can use the holdout's own number of
  looks, and never tests a lone first look at less than MIN_GATE_TRIALS trials;
- robustness tests whole parameters (both directions) and neither robustness nor optimize records
  a combo whose trades duplicate the base's or an earlier combo's as a separate trial.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import random
from statistics import fmean, median, stdev

from .backtest import BacktestResult, as_spec, backtest, costs_from_cfg, rounded
from .config import DEFAULTS
from .indicators import adx, atr, ema
from .metrics import (clustered_t_stat, deflated_sharpe, design_effect, effective_n, entry_week,
                      expected_max_sharpe, trade_stats)
from .strategy import PARAM_SPACE, StrategySpec, _filter_ok, manage_position, warmup_bars
from .types import TF_SECONDS, Bars, Trade

DAY = 86400
MONTH_DAYS = 30.44
OBJECTIVES = ("avg_r", "profit_factor", "total_r")
FOLD_KEYS = ("n", "avg_r", "total_r", "profit_factor")
# Choice params that change WHAT is traded (other data / other side), not how: never neighbours.
FIXED_PARAMS = frozenset({"timeframe", "direction"})
MAX_NEIGHBOURS = 48            # >= 2 moves x every movable PARAM_SPACE parameter: the whole neighbourhood
GOLDEN = 0.6180339887498949    # low-discrepancy order of parameter groups when robustness is capped
MIN_GATE_TRIALS = 2            # the gate's deflated Sharpe never treats a look as the only idea
MIN_VALIDATION_EDGE_R = 0.0    # default validation_gate.min_validation_edge_r (with a baseline)
MIN_VALIDATION_EDGE_Z = 1.65   # default validation_gate.min_validation_edge_z (one-sided 5%)


def compute_windows(cfg: dict, bars_by_symbol: dict[str, Bars],
                    validation_end: int | None = None) -> dict:
    """Research / locked validation windows. data_end = min over symbols of (last t + tf).

    Sliding (validation_end None): validation = [validation_start, data_end) with
    validation_start = data_end - validation_months * 30.44 days, floored to a UTC day.
    Pinned (validation_end = an epoch's end, e.g. stored at the first promotion):
    validation_start = validation_end - validation_months * 30.44 days (floored), validation =
    [validation_start, min(validation_end, data_end)) and the result has "pinned": True; bars at
    or after validation_end belong to neither window (forward / paper data only), so new data
    never slides the holdout over paper results. research = [first t of any symbol,
    validation_start) in both cases."""
    live = [b for b in bars_by_symbol.values() if len(b)]
    if not live:
        raise ValueError("compute_windows: no bars")
    data_start = min(b.t[0] for b in live)
    data_end = min(b.t[-1] + TF_SECONDS[b.tf] for b in live)
    end = data_end if validation_end is None else int(validation_end)
    vs = int(end - cfg["research"]["validation_months"] * MONTH_DAYS * DAY)
    vs -= vs % DAY
    out = {"data_start": data_start, "data_end": data_end, "research_start": data_start,
           "research_end": vs, "validation_start": vs, "validation_end": min(end, data_end)}
    if validation_end is not None:
        out["pinned"] = True
    return out


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


def edge_stats(trades: list[Trade], base_mean: float, base_sd: float, n_runs: int) -> dict:
    """Edge of a strategy's closed trades over a random-entry baseline and its z score.

    edge_r = mean R - base_mean. Under "no timing skill" the strategy's mean R is one draw of a
    random-entry run, so its null spread is at least the run-to-run sd of the baseline (base_sd)
    and at least its own cluster-robust standard error (strategy_se: sd/sqrt(n) inflated by the
    design effect of UTC entry weeks, since correlated coins enter together); base_mean itself is
    known to base_sd/sqrt(n_runs). edge_se = sqrt(max(strategy_se, base_sd)^2 + base_sd^2/n_runs),
    edge_z = edge_r / edge_se (0 when edge_se is 0). Unrounded floats."""
    rs = [tr.r_multiple for tr in trades]
    n = len(rs)
    mean = fmean(rs) if rs else 0.0
    se = 0.0
    if n >= 2:
        se = stdev(rs) / math.sqrt(n) * math.sqrt(design_effect(rs, [entry_week(tr.entry_t) for tr in trades]))
    base_sd = max(0.0, float(base_sd or 0.0))
    edge_se = math.sqrt(max(se, base_sd) ** 2 + base_sd ** 2 / max(1, int(n_runs or 1)))
    edge = mean - float(base_mean)
    return {"strategy_se": se, "edge_r": edge, "edge_se": edge_se,
            "edge_z": edge / edge_se if edge_se > 0 else 0.0}


def _filters_entries(spec: StrategySpec) -> bool:
    """Whether the spec's trend / ADX filter restricts when it can trade."""
    return spec.trend_filter != "none" or spec.adx_min > 0


def _regime_bars(spec: StrategySpec, w: Bars, sides, i0: int) -> dict[int, list[int]]:
    """Per side: entry bars e in [max(i0, 1), len(w) - 1) whose previous close passed the spec's
    trend / ADX filter (indicators as FibEngine.run computes them on the same window)."""
    c = w.c
    ema_t = ema(c, spec.trend_ema) if spec.trend_filter != "none" else None
    ema50 = ema(c, 50) if spec.trend_filter == "ema_stack" else None
    adx_v = adx(w.h, w.l, c, 14) if spec.adx_min > 0 else None
    out = {}
    for s in sides:
        ok = _filter_ok(spec, s, c, ema_t, ema50, adx_v)
        out[s] = [e for e in range(max(i0, 1), len(w) - 1) if ok[e - 1]]
    return out


def random_baseline(spec: StrategySpec | dict, bars_by_symbol: dict[str, Bars], cfg: dict,
                    start_t: int | None, end_t: int | None, n_runs: int = 10, seed: int = 1,
                    result: BacktestResult | None = None, regime: bool = True) -> dict:
    """Random-entry baseline with the strategy's own risk and exits. For every closed trade of
    `result` (backtest of spec on the same window when None) each run draws an entry bar uniformly
    from the window bars of the same symbol (any but the last), enters at the open with the
    strategy's own entry cost (a confirm-mode market entry pays slippage on the same side; a
    limit entry pays none, like a limit filled at the open of a bar that gapped through it), puts
    the stop at the same risk_pct and tp1/tp2 at the same R distances, and manages it with
    strategy.manage_position (an exit-less position is marked at the window end).

    Keys: n_trades, n_runs, seed, strategy_avg_r, baseline_avg_r_mean, baseline_avg_r_std,
    beats_frac, runs, edge_r (strategy - baseline mean), plus strategy_se, edge_se, edge_z
    (edge_stats). With regime=True a second set of n_runs draws only uses bars whose previous
    close passed the spec's trend / ADX filter for that side (the strategy can only trade then;
    a symbol-side with no such bar falls back to the uniform draw): baseline_regime_avg_r_mean,
    baseline_regime_avg_r_std, edge_regime_r, edge_regime_z, regime_filtered (False when the spec
    has no filter: the regime keys then repeat the uniform ones). Use min(edge_r, edge_regime_r):
    the edge over both is what entry timing adds; either alone can be the lenient one.
    Deterministic for a given seed (the uniform draws do not depend on `regime`)."""
    spec = as_spec(spec)
    if result is None:
        result = backtest(spec, bars_by_symbol, start_t, end_t, cfg)
    costs = costs_from_cfg(cfg)
    entry_slip = costs.slippage if spec.entry_mode == "confirm" else 0.0
    trades = [tr for tr in result.closed_trades if tr.risk_pct > 0]
    filtered = bool(regime) and _filters_entries(spec)
    windowed: dict[str, tuple[Bars, list[float], int, dict]] = {}
    for sym in sorted({tr.symbol for tr in trades}):
        w = bars_by_symbol[sym].window(start_t, end_t, warmup_bars(spec))
        i0 = 0 if start_t is None else w.index_at_or_after(start_t)
        elig = (_regime_bars(spec, w, sorted({tr.side for tr in trades if tr.symbol == sym}), i0)
                if filtered else {})
        windowed[sym] = (w, atr(w.h, w.l, w.c, spec.swing_atr_len), i0, elig)

    def run_once(rng: random.Random, conditioned: bool) -> float:
        rs = []
        for tr in trades:
            w, atr_w, i0, elig = windowed[tr.symbol]
            if len(w) - 1 <= i0:
                continue
            # risk_pct is the planned risk for a fill at or beyond the stop (strategy._Position):
            # the fill-to-stop distance would give negative or absurd target R-distances
            s, risk = tr.side, tr.risk_pct * tr.entry_price
            k1, k2 = (s * (tr.meta[k] - tr.entry_price) / risk for k in ("tp1", "tp2"))
            pool = elig.get(s) if conditioned else None
            e = pool[rng.randrange(len(pool))] if pool else rng.randrange(i0, len(w) - 1)
            px = w.o[e] * (1 + s * entry_slip)
            dist = px * tr.risk_pct
            rs.append(manage_position(w, atr_w, e, s, px, px - s * dist, px + s * k1 * dist,
                                      px + s * k2 * dist, spec, costs)["r_multiple"])
        return fmean(rs) if rs else 0.0

    n = n_runs if trades else 0
    rng = random.Random(seed)
    runs = [run_once(rng, False) for _ in range(n)]
    strat = result.metrics["avg_r"]
    mean = fmean(runs) if runs else 0.0
    sd = stdev(runs) if len(runs) > 1 else 0.0
    edge = edge_stats(result.closed_trades, mean, sd, n_runs)
    if filtered:
        rng_r = random.Random(f"regime:{seed}")
        r_runs = [run_once(rng_r, True) for _ in range(n)]
        r_mean = fmean(r_runs) if r_runs else 0.0
        r_sd = stdev(r_runs) if len(r_runs) > 1 else 0.0
        r_edge = edge_stats(result.closed_trades, r_mean, r_sd, n_runs)
    else:
        r_mean, r_sd, r_edge = mean, sd, edge
    return rounded({
        "n_trades": len(trades), "n_runs": n_runs, "seed": seed, "strategy_avg_r": strat,
        "baseline_avg_r_mean": mean, "baseline_avg_r_std": sd,
        "beats_frac": sum(strat > r for r in runs) / len(runs) if runs else 0.0,
        "runs": runs, "edge_r": strat - mean, "strategy_se": edge["strategy_se"],
        "edge_se": edge["edge_se"], "edge_z": edge["edge_z"], "regime_filtered": filtered,
        "baseline_regime_avg_r_mean": r_mean, "baseline_regime_avg_r_std": r_sd,
        "edge_regime_r": strat - r_mean, "edge_regime_z": r_edge["edge_z"]})


def trades_key(result: BacktestResult) -> str:
    """Digest of a result's trades (symbol, side, entry/exit time, R): two specs with the same key
    traded identically on this window (a no-op parameter change), so the second one is the same
    evidence, not another trial."""
    rows = sorted((tr.symbol, tr.side, tr.entry_t, tr.exit_t, round(tr.r_multiple, 9))
                  for tr in result.trades)
    return hashlib.sha1(repr(rows).encode()).hexdigest()


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
    once) and on_combo(spec, result) is called, except for a combo whose trades are identical to
    an earlier combo's (trades_key; counted in "n_duplicates"): a parameter value that changes
    nothing is not another trial. For fold k = 1..F-1 the combo with the best
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
    evaluated, invalid, seen, outcomes, n_dup = [], [], set(), set(), 0
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
        key = trades_key(result)
        if key in outcomes:
            n_dup += 1
        else:
            outcomes.add(key)
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
        "n_duplicates": n_dup, "n_folds": folds, "min_is_trades": min_is,
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


def _moved_param(spec: StrategySpec, nb: StrategySpec) -> str:
    base = spec.to_dict()
    return next(k for k, v in nb.to_dict().items() if v != base[k])


def _pick_neighbours(spec: StrategySpec, cands: list[StrategySpec], m: int) -> list[StrategySpec]:
    """All candidates when they fit in m; otherwise whole parameters (both of a parameter's moves
    or neither), taken in a low-discrepancy order over the PARAM_SPACE order (so a cap spreads
    over the parameter list instead of cutting its tail) while they fit; PARAM_SPACE order kept."""
    if len(cands) <= m:
        return list(cands)
    groups: dict[str, list[StrategySpec]] = {}
    for nb in cands:
        groups.setdefault(_moved_param(spec, nb), []).append(nb)
    names = list(groups)
    chosen, left = set(), m
    for i in sorted(range(len(names)), key=lambda i: ((i * GOLDEN) % 1.0, i)):
        if len(groups[names[i]]) <= left:
            chosen.add(names[i])
            left -= len(groups[names[i]])
    return [nb for name in names if name in chosen for nb in groups[name]]


def robustness(spec: StrategySpec | dict, bars_by_symbol: dict[str, Bars], cfg: dict, windows: dict,
               max_neighbours: int = MAX_NEIGHBOURS, on_combo=None) -> dict:
    """Evaluate the one-step neighbourhood neighbours(spec) on the research window: every
    neighbour when there are at most max_neighbours (the default covers every parameter), else
    whole parameters (both directions) up to the cap (see _pick_neighbours; "coverage" and
    "params_untested" say what was tested).

    The base spec is evaluated too (not recorded). A neighbour whose trades are identical to the
    base's or an earlier neighbour's (trades_key) is a no-op move: its row has "duplicate_of"
    ("base" or that spec_id), on_combo is not called for it and it is left out of the aggregates.
    on_combo(spec, result) is called per distinct neighbour. n = neighbours evaluated (rows),
    n_distinct = those that are not duplicates, n_duplicates = n - n_distinct, n_candidates =
    len(neighbours(spec)); positive_frac = share of the distinct ones with avg_r > 0 (no trades is
    not positive), median_avg_r / worst_avg_r over the distinct ones."""
    spec = as_spec(spec)
    cands = neighbours(spec)
    picked = _pick_neighbours(spec, cands, max(0, max_neighbours))
    rows, outcomes = [], {}
    if picked:
        base_result, _ = evaluate(spec, bars_by_symbol, cfg, windows, "research")
        outcomes[trades_key(base_result)] = "base"
    for nb in picked:
        result, extras = evaluate(nb, bars_by_symbol, cfg, windows, "research")
        key = trades_key(result)
        dup = outcomes.get(key)
        if dup is None:
            outcomes[key] = result.spec_id
            if on_combo is not None:
                on_combo(nb, result)
        param = _moved_param(spec, nb)
        mt = result.metrics
        rows.append({"param": param, "value": getattr(nb, param), "spec_id": result.spec_id,
                     "n": mt["n"], "avg_r": mt["avg_r"], "total_r": mt["total_r"],
                     "profit_factor": mt["profit_factor"],
                     "fold_positive_frac": extras["fold_positive_frac"], "duplicate_of": dup})
    avg = [r["avg_r"] for r in rows if r["duplicate_of"] is None]
    coverage: dict[str, list] = {}
    for r in rows:
        coverage.setdefault(r["param"], []).append(r["value"])
    movable = list(dict.fromkeys(_moved_param(spec, nb) for nb in cands))
    return rounded({
        "n": len(rows), "n_distinct": len(avg), "n_duplicates": len(rows) - len(avg),
        "n_candidates": len(cands),
        "positive_frac": sum(a > 0 for a in avg) / len(avg) if avg else 0.0,
        "median_avg_r": median(avg) if avg else 0.0, "worst_avg_r": min(avg, default=0.0),
        "coverage": coverage, "params_untested": [k for k in movable if k not in coverage],
        "rows": rows})


def _gate_edge(result: BacktestResult, baseline: dict) -> dict:
    """edge_stats of the result's closed trades against the uniform baseline and, when present,
    the regime baseline; edge_r and edge_z are the smaller (binding) of the two."""
    n_runs = baseline.get("n_runs") or 1
    rows = [edge_stats(result.closed_trades, baseline.get("baseline_avg_r_mean", 0.0),
                       baseline.get("baseline_avg_r_std", 0.0), n_runs)]
    if baseline.get("regime_filtered") and baseline.get("baseline_regime_avg_r_mean") is not None:
        rows.append(edge_stats(result.closed_trades, baseline["baseline_regime_avg_r_mean"],
                               baseline.get("baseline_regime_avg_r_std", 0.0), n_runs))
    return {"edge_r": min(r["edge_r"] for r in rows), "edge_z": min(r["edge_z"] for r in rows),
            "edge_se": max(r["edge_se"] for r in rows), "strategy_se": rows[0]["strategy_se"],
            "baselines": len(rows)}


def validation_gate(result: BacktestResult, gate: dict, n_trials: int, var_sr: float | None,
                    baseline: dict | None = None, holdout_looks: int | None = None,
                    var_holdout_sr: float | None = None) -> dict:
    """Deterministic pass/fail of a (validation-window) backtest. Checks, thresholds from `gate`
    (missing keys: config defaults): min_trades (closed), min_profit_factor, min_expectancy_r,
    min_deflated_sharpe, max_drawdown (portfolio equity, marked to market), min_symbol_win_frac
    (share of symbols WITH closed trades whose total R is > 0).

    Deflated Sharpe of the closed trades' R multiples (dsr_basis), with the effective number of
    trades n_eff (trades clustered by UTC entry week: correlated coins enter together, so n
    trades are worth fewer independent ones; metrics.effective_n) as the sample size:
    - "research" (holdout_looks None): against n_trials / var_sr, the research-window trial count
      and Sharpe variance (store.trial_stats);
    - "holdout" (holdout_looks given): against the holdout's own reuse, n_trials = holdout_looks
      (distinct looks at validation windows) and var = max(1 / n_eff, var_holdout_sr) (the null
      variance of a per-trade Sharpe estimate; var_holdout_sr = variance of the recorded
      validation Sharpes). Research trials never touched the holdout, so they do not select on
      it; the bar then falls with more validation trades and rises with more looks.
    Either way the trial count is at least MIN_GATE_TRIALS: a lone first look is not a 10% test.
    sr_star = the deflation hurdle used and dsr_var its variance (None = the 1 / n_eff default);
    n_eff and t_clustered (cluster-robust t of mean R) are reported.

    baseline (a random_baseline on the SAME window and trades, e.g. random_baseline(...,
    result=result)) adds min_validation_edge_r (edge over random entries, R) and
    min_validation_edge_z (edge_stats z score; with a regime baseline the smaller of the two
    edges is used): without them a long strategy passes in a bull year on drift alone. Their
    thresholds default to MIN_VALIDATION_EDGE_R / MIN_VALIDATION_EDGE_Z; a threshold set to None
    drops that check. "edge" holds the numbers."""
    g = {**DEFAULTS["research"]["validation_gate"], **(gate or {})}
    m = result.metrics
    closed = result.closed_trades
    rs_ = [tr.r_multiple for tr in closed]
    weeks = [entry_week(tr.entry_t) for tr in closed]
    n_eff = effective_n(rs_, weeks)
    if holdout_looks is None:
        basis, trials, var = "research", max(int(n_trials or 0), MIN_GATE_TRIALS), var_sr
    else:
        basis, trials = "holdout", max(int(holdout_looks), MIN_GATE_TRIALS)
        var = max(1.0 / max(n_eff, 1.0), float(var_holdout_sr or 0.0)) if rs_ else None
    dsr = deflated_sharpe(rs_, trials, var, n_eff=n_eff if rs_ else None)
    sr_star = expected_max_sharpe(trials, var if var is not None and var > 0
                                  else 1.0 / max(1.0, n_eff))
    traded = [s for s in result.per_symbol.values() if s["n"] > 0]
    sym_frac = sum(s["total_r"] > 0 for s in traded) / len(traded) if traded else 0.0
    values = [("min_trades", m["n"]), ("min_profit_factor", m["profit_factor"]),
              ("min_expectancy_r", m["expectancy_r"]), ("min_deflated_sharpe", dsr),
              ("max_drawdown", m["max_drawdown"]), ("min_symbol_win_frac", sym_frac)]
    thresholds = {name: g[name] for name, _ in values}
    edge = None
    if baseline is not None:
        edge = _gate_edge(result, baseline)
        for name, key, default in (("min_validation_edge_r", "edge_r", MIN_VALIDATION_EDGE_R),
                                   ("min_validation_edge_z", "edge_z", MIN_VALIDATION_EDGE_Z)):
            threshold = g.get(name, default)
            if threshold is not None:
                values.append((name, edge[key]))
                thresholds[name] = threshold
    checks = [(name, value, thresholds[name],
               value <= thresholds[name] if name.startswith("max_") else value >= thresholds[name])
              for name, value in values]
    out = {"passed": all(c[3] for c in checks), "deflated_sharpe": dsr,
           "checks": [{"name": n, "value": v, "threshold": t, "passed": p} for n, v, t, p in checks],
           "dsr_basis": basis, "dsr_trials": trials, "dsr_var": var, "sr_star": sr_star, "n_eff": n_eff,
           "t_clustered": clustered_t_stat(rs_, weeks) if len(rs_) >= 2 else 0.0}
    if edge is not None:
        out["edge"] = edge
    return rounded(out)
