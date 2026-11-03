"""Tools the brain uses during research and review cycles, plus the market-context summary.

build_tools(lab, cycle) wraps the Lab facade in `anthropic.beta_tool` closures. Every tool returns
compact JSON (floats rounded, <= MAX_RESULT_CHARS). A failing tool raises ToolError with
"ExceptionType: message", which the tool runner hands back to Claude as an is_error result (budget,
gate and validation errors are expected and steer the brain). Budgets, gates and risk limits live in
Lab and below, never here.
"""
from __future__ import annotations

import inspect
import json
import logging
import math
from datetime import datetime, timezone

from ..backtest import rounded
from ..fib import fib_table, zigzag
from ..indicators import adx, atr, ema, rsi
from ..strategy import PARAM_SPACE, PRESETS, StrategySpec
from ..types import TF_SECONDS, Bars

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 12000
CYCLES = ("research", "review")
CONTEXT_BARS = 1000          # bars used for indicators/pivots in market_context (EMA200 settles)
RETURN_WINDOWS = (1, 7, 30)  # bars
METRIC_KEYS = ("n", "avg_r", "total_r", "win_rate", "profit_factor", "max_drawdown",
               "sharpe_per_trade", "fold_positive_frac", "min_fold_avg_r", "edge_r", "beats_frac",
               "positive_frac", "median_avg_r", "worst_avg_r", "deflated_sharpe", "passed", "n_combos")
BRIEF_KEYS = ("spec_id", "strategy_id", "experiment_id", "passed")
EXACT_KEYS = frozenset({"spec", "params", "base_spec"})   # spec values copied back by the brain: never rounded
# validate_strategy shows the brain only pass/fail, the gate checks and these headline numbers: a
# per-symbol / per-year / per-fold breakdown of the held-out window would let it steer research
# toward what happened to work there (the full result stays in the experiment row for humans).
VALIDATION_KEYS = ("spec_id", "name", "timeframe", "direction", "window", "period", "n_symbols",
                   "passed", "n_trials", "experiment_id")
VALIDATION_METRICS = ("n", "n_open", "avg_r", "total_r", "win_rate", "profit_factor", "max_drawdown",
                      "sharpe_per_trade")
VALIDATION_HIDDEN = ("fold_positive_frac", "min_fold_avg_r")    # left out of validate rows too
ROWS_RESERVE = 400           # chars kept free for the header/omitted note of a list_experiments reply
SPEC_RULES = (
    "entry_zone_lo <= entry_zone_hi < stop_ratio",
    "tp1_ext < tp2_ext unless tp1_frac == 1 (then tp2_ext and trail_atr are unused)",
    "min_rr is reward/risk before costs; min_stop_cost_mult (0-20) requires the stop to be at least"
    " that many round-trip costs away (2*fee + slippage; + entry slippage in confirm mode): the lever"
    " against tight 1h stops that costs eat",
    "swing_mult is used in atr swing mode, swing_pct in pct mode",
    "entry_at applies to limit entries; confirm_type and confirm_window to confirm entries",
    "trend_ema is unused when trend_filter is 'none'; adx_min 0 disables the ADX filter",
    "run_optimization grids map parameter names to lists of values; the grid size (product of list "
    "lengths) is capped and timeframe/direction cannot vary inside a grid",
)


def iso(t: int | None) -> str | None:
    """Compact UTC timestamp 'YYYY-MM-DDTHH:MMZ' (None stays None)."""
    if t is None:
        return None
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _prepare(x, exact: bool = False):
    """rounded(x), except that values under EXACT_KEYS keep full precision (nan/inf -> None), so
    a spec copied from a tool result keeps its spec_id."""
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return x if exact else rounded(x)
    if isinstance(x, dict):
        return {k: _prepare(v, exact or k in EXACT_KEYS) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_prepare(v, exact) for v in x]
    return x


def to_json(obj, limit: int = MAX_RESULT_CHARS) -> str:
    """Compact JSON of obj with floats rounded (nan -> null; spec values under EXACT_KEYS keep
    full precision), cut to at most `limit` chars."""
    s = json.dumps(_prepare(obj), default=str, separators=(",", ":"))
    if len(s) <= limit:
        return s
    keep = limit - 60
    return s[:keep] + f"...[truncated {len(s) - keep} chars; request less detail]"


def parse_spec(d: dict) -> StrategySpec:
    """Validated StrategySpec from a tool argument: StrategySpec fields, optionally on top of a
    preset via {"preset": name}. ValueError on unknown presets/keys or invalid values."""
    d = dict(d)
    preset = d.pop("preset", None)
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; presets: {', '.join(PRESETS)}")
    base = PRESETS[preset] if preset is not None else StrategySpec()
    return base.with_overrides(**d).validate()


_DEFAULTS = StrategySpec().to_dict()


def spec_diff(d: dict | None) -> dict:
    """The parameters of a spec dict that differ from the StrategySpec defaults (name dropped)."""
    return {k: v for k, v in (d or {}).items() if k != "name" and _DEFAULTS.get(k, object()) != v}


def pick_metrics(m) -> dict:
    """The headline METRIC_KEYS found in a metrics dict (top level first, then nested dicts)."""
    if not isinstance(m, dict):
        return {}
    out: dict = {}
    for src in [m, *(v for v in m.values() if isinstance(v, dict))]:
        for k in METRIC_KEYS:
            if k not in out and isinstance(src.get(k), (int, float)):
                out[k] = src[k]
    return out


def experiment_row(r: dict) -> dict:
    """Compact view of a store.list_experiments row (validate rows: headline metrics only, no
    fold summaries of the held-out window)."""
    spec = r.get("spec") or {}
    metrics = pick_metrics(r["metrics"])
    if r["kind"] == "validate":
        metrics = {k: v for k, v in metrics.items() if k not in VALIDATION_HIDDEN}
    return {"id": r["id"], "kind": r["kind"], "created": iso(r["created_t"]), "spec_id": r["spec_id"],
            "name": spec.get("name"), "params": spec_diff(spec),
            "window": [iso(r["window_start"]), iso(r["window_end"])],
            "metrics": metrics, "notes": (r.get("notes") or "")[:200]}


def fit_rows(rows: list, limit: int = MAX_RESULT_CHARS - ROWS_RESERVE) -> tuple[list, int]:
    """(the leading rows whose JSON fits in `limit` chars, how many were left out): whole rows
    only, so a long listing is never cut mid-row into invalid JSON."""
    kept, size = [], 2
    for r in rows:
        size += len(to_json(r, 10 ** 9)) + 1
        if size > limit:
            break
        kept.append(r)
    return kept, len(rows) - len(kept)


def validation_view(out: dict) -> dict:
    """What the brain sees of a Lab.validate result: pass/fail, the gate checks and headline
    numbers (VALIDATION_KEYS / VALIDATION_METRICS); no per-symbol, per-year, fold or portfolio
    breakdown of the held-out window."""
    gate = out.get("gate") if isinstance(out.get("gate"), dict) else {}
    m = out.get("metrics") if isinstance(out.get("metrics"), dict) else {}
    checks = [{k: c.get(k) for k in ("name", "value", "threshold", "passed")}
              for c in gate.get("checks") or [] if isinstance(c, dict)]
    return {**{k: out[k] for k in VALIDATION_KEYS if k in out},
            "metrics": {k: m[k] for k in VALIDATION_METRICS if k in m},
            "gate": {"passed": gate.get("passed", out.get("passed")),
                     "deflated_sharpe": gate.get("deflated_sharpe"), "checks": checks,
                     **{k: gate[k] for k in ("n_eff", "t_clustered", "dsr_trials") if k in gate}},
            "note": "only pass/fail, the gate checks and headline numbers are shown, by design"}


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _params_filter(params: dict) -> dict:
    """Validated {StrategySpec field: value} filter for list_experiments(params=...)."""
    if not isinstance(params, dict) or not params:
        raise ValueError("params must be a non-empty object {parameter: value}")
    unknown = sorted(set(params) - set(_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown spec parameters: {', '.join(unknown)} (use spec={{\"preset\": ...}} "
                         "for a preset; the \"name\" parameter matches the preset a spec started from)")
    return params


HIDDEN_SOURCES = ("cli-holdout",)   # operator backtests of the validation / all window


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _listing(store, **kw) -> list[dict]:
    """store.list_experiments(**kw) without HIDDEN_SOURCES rows (filtered before the limit)."""
    fn = store.list_experiments
    if _accepts(fn, "exclude_sources"):
        return fn(**kw, exclude_sources=HIDDEN_SOURCES)
    limit = kw.pop("limit", None)
    rows = [r for r in fn(**kw, limit=None) if r.get("source") not in HIDDEN_SOURCES]
    return rows if limit is None else rows[:int(limit)]


def _experiments_of(store, spec_id: str, kind: str | None) -> list[dict]:
    """Every experiment of one spec_id (of `kind`), newest first (no HIDDEN_SOURCES rows)."""
    if _accepts(store.list_experiments, "spec_id"):
        return _listing(store, limit=None, kind=kind, spec_id=spec_id)
    return [r for r in _listing(store, limit=None, kind=kind) if r.get("spec_id") == spec_id]


def spec_lookup(store, spec_id: str, kind: str | None = None, limit: int = 50,
                spec: dict | None = None) -> dict:
    """One spec's record: its exact parameters (copy them to reuse the spec_id), its latest
    validation (passed, failed checks), its roster status and its experiments (newest first)."""
    rows = _experiments_of(store, spec_id, kind)
    val = store.latest_validation(spec_id)
    roster = store.get_strategy(spec_id)
    if spec is None:
        spec = next((r["spec"] for r in rows if r.get("spec")), None) or \
            (val or {}).get("spec") or (roster or {}).get("spec")
    vm = (val or {}).get("metrics") or {}
    kept, omitted = fit_rows([experiment_row(r) for r in rows[:limit]],
                             MAX_RESULT_CHARS - ROWS_RESERVE - len(to_json(spec, 10 ** 9)))
    return {"spec_id": spec_id, "spec": spec, "n_experiments": len(rows),
            "latest_validation": None if val is None else {
                "experiment_id": val["id"], "created": iso(val["created_t"]),
                "passed": vm.get("passed"),
                "failed_checks": [c.get("name") for c in vm.get("checks") or []
                                  if isinstance(c, dict) and not c.get("passed")]},
            "roster": None if roster is None else {
                "strategy_id": roster["strategy_id"], "name": roster["name"],
                "status": roster["status"], "status_reason": roster.get("status_reason") or ""},
            "experiments": kept, "omitted": omitted + max(0, len(rows) - limit)}


def paper_trade_row(r: dict) -> dict:
    """Compact view of a store.paper_trades row."""
    return {"symbol": r["symbol"], "side": r["side"], "leg_id": r["leg_id"], "status": r["status"],
            "entry_t": iso(r["entry_t"]), "entry_price": r["entry_price"], "stop_price": r["stop_price"],
            "exit_t": iso(r["exit_t"]), "exit_price": r["exit_price"], "exit_reason": r["exit_reason"],
            "r_multiple": r["r_multiple"], "ret_pct": r["ret_pct"], "bars_held": r["bars_held"],
            "notional": r["notional"], "pnl": r["pnl"]}


def strategy_space(cfg: dict) -> dict:
    """Parameter space, cross-field rules, presets (as differences from the defaults) and limits."""
    return {"params": PARAM_SPACE, "rules": list(SPEC_RULES),
            "presets": {name: spec_diff(p.to_dict()) for name, p in PRESETS.items()},
            "research": {k: cfg["research"].get(k) for k in
                         ("validation_months", "folds", "min_trades", "max_grid_size")},
            "validation_gate": cfg["research"].get("validation_gate"),
            "graduation": cfg.get("graduation"), "costs": cfg.get("costs"),
            "portfolio": cfg.get("portfolio"), "timeframes": cfg["data"].get("timeframes")}


def _ratio(a: float, b: float) -> float | None:
    return a / b - 1.0 if b == b and b else None


def _current_leg(b: Bars, pivots: list, last: int, atr_last: float) -> dict | None:
    """The leg from the latest confirmed pivot to the running extreme after it, with its fib table."""
    if not pivots:
        return None
    p = pivots[-1]
    seg = range(p.idx + 1, last + 1)
    up = p.kind == "L"
    ext_i = max(seg, key=lambda i: b.h[i]) if up else min(seg, key=lambda i: b.l[i])
    extreme = b.h[ext_i] if up else b.l[ext_i]
    size = abs(extreme - p.price)
    if size == 0:
        return None
    return {"direction": "up" if up else "down", "anchor": p.price, "anchor_t": iso(p.t),
            "extreme": extreme, "extreme_t": iso(b.t[ext_i]),
            "size_atr": size / atr_last if atr_last == atr_last and atr_last else None,
            "close_retracement": (extreme - b.c[last]) / (extreme - p.price),
            "levels": fib_table(p.price, extreme)}


def market_context(bars: Bars, upto_t: int | None = None, swing_mode: str = "atr",
                   swing_mult: float = 3.0, swing_pct: float = 0.05, atr_len: int = 14,
                   n_closes: int = 30) -> dict:
    """Rounded, JSON-safe summary of one symbol/timeframe using only bars with t <= upto_t.

    Keys: symbol, timeframe, n_bars, t (last bar open), close, returns over 1/7/30 bars, atr_pct,
    close vs EMA50/EMA200, ema50_above_ema200, rsi14, adx14, range (high/low of the last n_closes
    bars), closes (last n_closes), pivots (last 4 confirmed ZigZag pivots with the given swing
    settings) and leg (the current leg from the latest pivot with its fib_table).
    """
    if upto_t is not None:
        bars = bars.window(None, int(upto_t) + 1)
    b = bars.slice(max(0, len(bars) - CONTEXT_BARS), len(bars))
    out: dict = {"symbol": b.symbol, "timeframe": b.tf, "n_bars": len(b)}
    if len(b) < 2:
        return {**out, "error": "not enough bars"}
    last, c = len(b) - 1, b.c
    a = atr(b.h, b.l, c, atr_len)
    e50, e200 = ema(c, 50)[last], ema(c, 200)[last]
    pivots = zigzag(b, a, swing_mult, swing_mode, swing_pct)
    out.update({
        "t": iso(b.t[last]), "close": c[last],
        "returns": {f"{k}_bars": c[last] / c[last - k] - 1.0 if last >= k else None
                    for k in RETURN_WINDOWS},
        "atr_pct": a[last] / c[last],
        "close_vs_ema50": _ratio(c[last], e50), "close_vs_ema200": _ratio(c[last], e200),
        "ema50_above_ema200": e50 > e200 if e50 == e50 and e200 == e200 else None,
        "rsi14": rsi(c)[last], "adx14": adx(b.h, b.l, c)[last],
        "range": {"bars": min(n_closes, len(b)), "high": max(b.h[-n_closes:]),
                  "low": min(b.l[-n_closes:])},
        "closes": c[-n_closes:],
        "pivots": [{"kind": p.kind, "price": p.price, "t": iso(p.t), "bars_ago": last - p.idx}
                   for p in pivots[-4:]],
        "leg": _current_leg(b, pivots, last, a[last]),
    })
    return rounded(out)


def _symbols(symbols: list[str] | None) -> list[str] | None:
    return [s.strip().upper() for s in symbols] if symbols else None


def build_tools(lab, cycle: str, actions: list | None = None) -> list:
    """beta_tool closures over `lab` for a "research" or "review" cycle (review: no optimization).

    Every call is logged to `actions` (if given) as {"tool", "ok", ...}: on success the result's
    spec_id/strategy_id/experiment_id/passed plus call details; on failure "error".
    """
    from anthropic import beta_tool
    from anthropic.lib.tools import ToolError

    if cycle not in CYCLES:
        raise ValueError(f"cycle must be one of {CYCLES}, got {cycle!r}")
    record = actions if actions is not None else []

    def call(tool: str, fn, limit: int = MAX_RESULT_CHARS, **info) -> str:
        try:
            out = fn()
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            log.info("brain tool %s failed: %s", tool, msg)
            record.append({"tool": tool, "ok": False, "error": msg[:300], **info})
            raise ToolError(msg) from e
        brief = {k: out[k] for k in BRIEF_KEYS if isinstance(out, dict) and k in out}
        record.append({"tool": tool, "ok": True, **brief, **info})
        return to_json(out, limit)

    @beta_tool
    def get_lab_status() -> str:
        """Current lab status: roster, cycle budget, data windows and trial statistics.

        The roster lists each paper strategy with its paper account, its "validation"
        {experiment_id, window, n, avg_r, profit_factor, deflated_sharpe} and its graduation:
        status, summary, "checks" as [name, value, threshold, passed], "mean_r_ci" and
        "evidence" (n_eff, design_effect, weeks, t_clustered). The status also has
        validations_7d {used, cap}, holdout_peeks, validation_epoch (the pinned holdout, if
        any), data_stall and system_events_7d (errors and warnings of the last 7 days).
        """
        return call("get_lab_status", lab.status)

    @beta_tool
    def describe_strategy_space() -> str:
        """The strategy parameter space, presets, gates and limits.

        Lists every spec parameter with type, range, step, default and meaning; the cross-field
        rules; the presets (as differences from the defaults); and the research limits,
        validation gate, graduation thresholds, costs and portfolio rules.
        """
        return call("describe_strategy_space", lambda: strategy_space(lab.cfg))

    @beta_tool
    def list_experiments(limit: int = 10, kind: str | None = None, sort_by: str = "recent",
                         spec: dict | None = None, spec_id: str | None = None,
                         params: dict | None = None) -> str:
        """Past experiments with their spec parameters and headline metrics.

        Check this before testing an idea so you do not repeat work. With spec or spec_id it
        returns that spec's whole record: exact parameters, every experiment, latest validation
        and roster status. With params it lists the experiments of every spec whose parameters
        include those values. Validation results cannot be ranked: sort_by other than "recent"
        rejects kind "validate" and leaves validation rows out. Operator backtests of the locked
        validation period are never listed.

        Args:
            limit: Number of rows, 1-50.
            kind: Optional filter: backtest, optimize, baseline, validate, robustness or sweep.
            sort_by: "recent" (newest first), "avg_r" or "profit_factor" (best first).
            spec: Optional spec (same format as run_backtest's spec) to look up.
            spec_id: Optional spec_id to look up.
            params: Optional parameter values every listed spec must have, e.g.
                {"name": "golden_pocket_4h", "trend_ema": 100}.
        """
        def run():
            n = max(1, min(int(limit), 50))
            if kind == "validate" and sort_by != "recent":
                raise ValueError("validation results cannot be ranked (use sort_by 'recent' with kind "
                                 "'validate'): ranking held-out results would make the validation "
                                 "window a search target")
            if spec is not None or spec_id is not None:
                parsed = parse_spec(spec) if spec is not None else None
                sid = parsed.spec_id() if parsed is not None else str(spec_id).strip()
                if parsed is not None and spec_id is not None and str(spec_id).strip() != sid:
                    raise ValueError(f"spec has spec_id {sid}, not {spec_id}")
                return spec_lookup(lab.store, sid, kind, n,
                                   None if parsed is None else parsed.to_dict())
            if params is not None:
                want = _params_filter(params)
                rows = [r for r in _listing(lab.store, limit=None, kind=kind)
                        if all(_same((r.get("spec") or {}).get(k), v) for k, v in want.items())]
            elif kind is None and sort_by != "recent":
                rows = [r for r in _listing(lab.store, limit=None, sort_by=sort_by)
                        if r["kind"] != "validate"]
            else:
                rows = _listing(lab.store, limit=n, kind=kind, sort_by=sort_by)
            kept, omitted = fit_rows([experiment_row(r) for r in rows[:n]])
            omitted += max(0, len(rows) - n)
            if omitted:
                kept.append({"omitted": omitted, "hint": "more rows match: ask for fewer, filter by "
                                                          "kind, or look up one spec"})
            return kept
        return call("list_experiments", run)

    @beta_tool
    def run_backtest(spec: dict, window: str = "research", symbols: list[str] | None = None) -> str:
        """Backtest a spec on the research window and record the experiment.

        Runs across the universe (or a symbol subset). Returns trade statistics (n, avg_r,
        win_rate, profit_factor, ...), portfolio equity stats, per-symbol and per-year results and
        walk-forward fold results. A direction "both" spec also gets "by_side" {long, short: n,
        total_r, avg_r, win_rate, profit_factor}: rows of the one combined run (one position per
        symbol), so before acting on one side backtest a direction-only spec (a new trial).
        Costs 1 backtest unit. A symbol subset is a diagnostic only: validation, promotion and
        paper trading always use the whole universe.

        Args:
            spec: Strategy parameters as a JSON object, e.g. {"timeframe": "4h",
                "entry_zone_lo": 0.5}. Omitted parameters take defaults;
                {"preset": "golden_pocket_4h", ...} starts from a preset.
            window: Must be "research". The held-out validation window is reachable only
                through validate_strategy.
            symbols: Optional subset of universe symbols such as ["BTCUSDT", "ETHUSDT"];
                default is the whole universe.
        """
        def run():
            if window != "research":
                raise ValueError("only the research window is available here; the validation window "
                                 "is reachable only through validate_strategy")
            return lab.backtest(parse_spec(spec), window="research", symbols=_symbols(symbols))
        return call("run_backtest", run)

    @beta_tool
    def run_optimization(base_spec: dict, grid: dict[str, list]) -> str:
        """Anchored walk-forward optimization of a small parameter grid on the research window.

        Every combination is evaluated once; for each fold the best combination on earlier folds
        is traded on that fold ("oos": walk-forward, held-out-fold stats). Returns those stats, the
        chosen params per fold, the best and top combinations and per-parameter sensitivity. The
        folds are part of the research window, which run_backtest also shows you, so the
        walk-forward result measures tuning stability, not independent evidence: only the locked
        validation and paper trades are unseen. Costs one backtest unit per combination; every
        combination with a distinct outcome counts as a trial, so prefer few, meaningful values.

        Args:
            base_spec: The spec the grid varies, same format as run_backtest's spec.
            grid: Parameter name -> list of values, e.g.
                {"entry_zone_lo": [0.5, 0.618], "min_rr": [1.0, 1.5]}.
        """
        return call("run_optimization", lambda: lab.optimize(parse_spec(base_spec), grid))

    @beta_tool
    def run_baseline(spec: dict) -> str:
        """Compare a spec with random entries on the research window.

        Each real trade is replayed from a random entry bar in the same symbol with the same side,
        risk, entry cost and management. Returns strategy_avg_r, the baseline mean/std, beats_frac
        (share of random runs the strategy beats), edge_r and baseline_z = edge_r / baseline std.
        For a spec with a trend / ADX filter a second set of random entries is drawn only where
        the filter passed: baseline_regime_avg_r_mean/std, edge_regime_r and edge_regime_z (they
        repeat the uniform numbers when regime_filtered is false). edge_z / edge_regime_z are z
        scores that also count the strategy's own (week-clustered) error. Judge the edge on
        min(edge_r, edge_regime_r): either alone can be the lenient one; promotion requires it to
        be > 0. A real edge has baseline_z of about 2 or more and beats_frac of at least 0.9.
        Costs 1 backtest unit.

        Args:
            spec: Strategy parameters, same format as run_backtest's spec.
        """
        def run():
            out = lab.baseline(parse_spec(spec))
            edge, std = out.get("edge_r"), out.get("baseline_avg_r_std")
            ok = isinstance(edge, (int, float)) and isinstance(std, (int, float)) and std > 0
            return {**out, "baseline_z": edge / std if ok else None}
        return call("run_baseline", run)

    @beta_tool
    def run_robustness(spec: dict) -> str:
        """Re-test a spec with every parameter moved one step both ways (research window).

        Up to 48 neighbours (see coverage / params_untested). Neighbours with the same trades as
        the base or an earlier neighbour are flagged duplicate_of and excluded from positive_frac
        (they are no-op moves, not trials). Returns the share of distinct neighbours with positive
        avg_r, their median and worst avg_r and one row per neighbour. Costs one backtest unit per
        neighbour plus one for the base spec.

        Args:
            spec: Strategy parameters, same format as run_backtest's spec.
        """
        return call("run_robustness", lambda: lab.robustness(parse_spec(spec)))

    @beta_tool
    def validate_strategy(spec: dict) -> str:
        """Run a spec once on the locked validation window and apply the validation gate.

        The validation window is the most recent months, never used in research. Consumes one of
        the cycle's scarce validation tries and is recorded permanently. Only a spec that passes
        can be promoted. Returns pass/fail, the gate checks and headline numbers only, by design.

        Args:
            spec: Strategy parameters, same format as run_backtest's spec.
        """
        return call("validate_strategy", lambda: validation_view(lab.validate(parse_spec(spec))))

    @beta_tool
    def promote_to_paper(name: str, rationale: str, spec: dict | None = None,
                         spec_id: str | None = None) -> str:
        """Put a validated spec on forward paper trading.

        Paper trading starts at the next bar of its timeframe. Requires a passed validation of
        exactly this spec and a free roster slot. Give the spec, or just the spec_id of the
        validated spec (its exact validated parameters are then used).

        Args:
            name: Short descriptive name, e.g. "gp_4h_trend".
            rationale: Why this strategy deserves paper capital: the evidence and what it adds
                to the roster.
            spec: The validated strategy parameters (same spec_id as the validation).
            spec_id: The spec_id of the validated spec, instead of spec.
        """
        def run():
            if spec is None and spec_id is None:
                raise ValueError("give the validated spec or its spec_id")
            if spec is not None:
                parsed = parse_spec(spec)
                if spec_id is not None and str(spec_id).strip() != parsed.spec_id():
                    raise ValueError(f"spec has spec_id {parsed.spec_id()}, not {spec_id}")
            else:
                sid = str(spec_id).strip()
                val = lab.store.latest_validation(sid)
                if val is None or not val.get("spec"):
                    raise ValueError(f"spec {sid} has no validation on record; run validate first")
                parsed = StrategySpec.from_dict(val["spec"]).validate()
                if parsed.spec_id() != sid:
                    raise ValueError(f"the stored validation of {sid} holds spec {parsed.spec_id()}")
            return lab.promote(parsed, name, rationale)
        return call("promote_to_paper", run, name=name)

    @beta_tool
    def retire_strategy(strategy_id: str, reason: str) -> str:
        """Stop paper trading a roster strategy; retirement is final for this spec_id.

        Its paper record is kept, and the spec can never be promoted again (a variant with a
        changed parameter is a new strategy).

        Args:
            strategy_id: The roster strategy_id.
            reason: The evidence for retiring it.
        """
        return call("retire_strategy", lambda: lab.retire(strategy_id, reason),
                    strategy_id=strategy_id, reason=reason[:200])

    @beta_tool
    def get_paper_trades(strategy_id: str, limit: int = 20) -> str:
        """Paper trades of one strategy, newest entry first.

        Open and closed trades with R multiples, exit reasons and position sizes.

        Args:
            strategy_id: The roster strategy_id.
            limit: Number of trades, 1-100.
        """
        def run():
            if lab.store.get_strategy(strategy_id) is None:
                raise KeyError(f"unknown strategy_id {strategy_id!r}")
            rows = lab.store.paper_trades(strategy_id, limit=max(1, min(int(limit), 100)))
            return {"strategy_id": strategy_id, "trades": [paper_trade_row(r) for r in rows]}
        return call("get_paper_trades", run)

    @beta_tool
    def get_market_context(symbol: str, timeframe: str = "4h") -> str:
        """Market summary for one symbol as of the end of the research window.

        Built from cached closed candles; the locked validation period is hidden from research
        and review cycles. Returns over 1/7/30 bars, ATR%, close vs EMA50/EMA200, RSI, ADX, the 30-bar range and
        closes, the latest confirmed swing pivots and the current leg with its Fibonacci levels.

        Args:
            symbol: Universe symbol such as "BTCUSDT".
            timeframe: "1h", "4h" or "1d".
        """
        def run():
            sym = symbol.strip().upper()
            if timeframe not in TF_SECONDS:
                raise ValueError(f"timeframe must be one of {sorted(TF_SECONDS)}, got {timeframe!r}")
            bars = lab.bars(timeframe, [sym]).get(sym)
            if not bars:
                raise ValueError(f"no cached {timeframe} candles for {sym}")
            upto = int(lab.windows(timeframe)["validation_start"]) - 1
            return {**market_context(bars, upto_t=upto),
                    "as_of": iso(upto + 1) + " (end of the research window)"}
        return call("get_market_context", run)

    @beta_tool
    def get_signal_review_stats() -> str:
        """How your take/skip signal reviews compare with taking every signal.

        Closed paper trades split by your decision on their leg, and skip_minus_take_avg_r
        (negative = your skips avoided worse trades).
        """
        return call("get_signal_review_stats", lab.signal_review_stats)

    notebook_limit = max(MAX_RESULT_CHARS, int(lab.cfg["brain"].get("notebook_max_chars", 0)) + 1000)

    @beta_tool
    def read_notebook() -> str:
        """Read the research notebook (your memory between cycles)."""
        return call("read_notebook", lambda: {"notebook": lab.notebook()}, notebook_limit)

    @beta_tool
    def append_notebook(entry: str) -> str:
        """Append a timestamped entry to the research notebook.

        Dated entries are trimmed oldest first when the notebook exceeds its size limit;
        undated sections written by rewrite_notebook are never trimmed. When they leave no room
        the append fails: condense with rewrite_notebook. "## " headings inside an entry become
        "### ". Returns notebook_chars, limit, free_chars, trimmed (dropped headings), truncated.

        Args:
            entry: Markdown text: hypotheses, results with numbers, dead ends, next steps.
        """
        return call("append_notebook", lambda: {"ok": True, **lab.notebook_append(entry)})

    @beta_tool
    def rewrite_notebook(content: str) -> str:
        """Replace the whole research notebook, e.g. to condense it.

        Keep every conclusion, open hypothesis, dead end and next step that still matters, and
        keep it well under the limit: undated sections are never trimmed. Returns notebook_chars,
        limit, free_chars and a warning when little room is left.

        Args:
            content: The complete new notebook in Markdown.
        """
        return call("rewrite_notebook", lambda: {"ok": True, **lab.notebook_rewrite(content)})

    tools = [get_lab_status, describe_strategy_space, list_experiments, run_backtest,
             run_optimization, run_baseline, run_robustness, validate_strategy, promote_to_paper,
             retire_strategy, get_paper_trades, get_market_context, get_signal_review_stats,
             read_notebook, append_notebook, rewrite_notebook]
    return tools if cycle == "research" else [t for t in tools if t is not run_optimization]
