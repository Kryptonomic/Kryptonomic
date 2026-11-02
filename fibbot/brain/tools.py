"""Tools the brain uses during research and review cycles, plus the market-context summary.

build_tools(lab, cycle) wraps the Lab facade in `anthropic.beta_tool` closures. Every tool returns
compact JSON (floats rounded, <= MAX_RESULT_CHARS). A failing tool raises ToolError with
"ExceptionType: message", which the tool runner hands back to Claude as an is_error result (budget,
gate and validation errors are expected and steer the brain). Budgets, gates and risk limits live in
Lab and below, never here.
"""
from __future__ import annotations

import json
import logging
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
SPEC_RULES = (
    "entry_zone_lo <= entry_zone_hi < stop_ratio",
    "tp1_ext < tp2_ext unless tp1_frac == 1 (then tp2_ext and trail_atr are unused)",
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


def to_json(obj, limit: int = MAX_RESULT_CHARS) -> str:
    """Compact JSON of obj with floats rounded (nan -> null), cut to at most `limit` chars."""
    s = json.dumps(rounded(obj), default=str, separators=(",", ":"))
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
    """Compact view of a store.list_experiments row."""
    spec = r.get("spec") or {}
    return {"id": r["id"], "kind": r["kind"], "created": iso(r["created_t"]), "spec_id": r["spec_id"],
            "name": spec.get("name"), "params": spec_diff(spec),
            "window": [iso(r["window_start"]), iso(r["window_end"])],
            "metrics": pick_metrics(r["metrics"]), "notes": (r.get("notes") or "")[:200]}


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

        The roster lists each paper strategy with its paper account and graduation checks.
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
    def list_experiments(limit: int = 10, kind: str | None = None, sort_by: str = "recent") -> str:
        """Past experiments with their spec parameters and headline metrics.

        Check this before testing an idea so you do not repeat work.

        Args:
            limit: Number of rows, 1-50.
            kind: Optional filter: backtest, optimize, baseline, validate, robustness or sweep.
            sort_by: "recent" (newest first), "avg_r" or "profit_factor" (best first).
        """
        n = max(1, min(int(limit), 50))
        return call("list_experiments", lambda: [
            experiment_row(r) for r in lab.store.list_experiments(limit=n, kind=kind, sort_by=sort_by)])

    @beta_tool
    def run_backtest(spec: dict, window: str = "research", symbols: list[str] | None = None) -> str:
        """Backtest a spec on the research window and record the experiment.

        Runs across the universe (or a symbol subset). Returns trade statistics (n, avg_r,
        win_rate, profit_factor, ...), portfolio equity stats, per-symbol and per-year results and
        walk-forward fold results. Costs 1 backtest unit.

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
        is traded out-of-sample. Returns the out-of-sample stats, the chosen params per fold, the
        best and top combinations and per-parameter sensitivity. Costs one backtest unit per
        combination and every combination counts as a trial, so prefer few, meaningful values.

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
        risk and management. Returns strategy_avg_r, the baseline mean/std, beats_frac (share of
        random runs the strategy beats) and edge_r. A real edge beats the baseline clearly. Costs
        1 backtest unit.

        Args:
            spec: Strategy parameters, same format as run_backtest's spec.
        """
        return call("run_baseline", lambda: lab.baseline(parse_spec(spec)))

    @beta_tool
    def run_robustness(spec: dict) -> str:
        """Re-test a spec with each active parameter moved one step either way (research window).

        Returns the share of neighbours with positive avg_r, their median and worst avg_r and one
        row per neighbour. Costs one backtest unit per neighbour.

        Args:
            spec: Strategy parameters, same format as run_backtest's spec.
        """
        return call("run_robustness", lambda: lab.robustness(parse_spec(spec)))

    @beta_tool
    def validate_strategy(spec: dict) -> str:
        """Run a spec once on the locked validation window and apply the validation gate.

        The validation window is the most recent months, never used in research. Consumes one of
        the cycle's scarce validation tries and is recorded permanently. Only a spec that passes
        can be promoted.

        Args:
            spec: Strategy parameters, same format as run_backtest's spec.
        """
        return call("validate_strategy", lambda: lab.validate(parse_spec(spec)))

    @beta_tool
    def promote_to_paper(spec: dict, name: str, rationale: str) -> str:
        """Put a validated spec on forward paper trading.

        Paper trading starts at the next bar of its timeframe. Requires a passed validation of
        exactly this spec and a free roster slot.

        Args:
            spec: The validated strategy parameters (same spec_id as the validation).
            name: Short descriptive name, e.g. "gp_4h_trend".
            rationale: Why this strategy deserves paper capital: the evidence and what it adds
                to the roster.
        """
        return call("promote_to_paper", lambda: lab.promote(parse_spec(spec), name, rationale),
                    name=name)

    @beta_tool
    def retire_strategy(strategy_id: str, reason: str) -> str:
        """Stop paper trading a roster strategy. Its paper record is kept.

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
        """Current market summary for one symbol from cached closed candles.

        Returns over 1/7/30 bars, ATR%, close vs EMA50/EMA200, RSI, ADX, the 30-bar range and
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
            return market_context(bars)
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

        The oldest entries are trimmed when the notebook grows beyond its size limit.

        Args:
            entry: Markdown text: hypotheses, results with numbers, dead ends, next steps.
        """
        def run():
            lab.notebook_append(entry)
            return {"ok": True, "notebook_chars": len(lab.notebook())}
        return call("append_notebook", run)

    @beta_tool
    def rewrite_notebook(content: str) -> str:
        """Replace the whole research notebook, e.g. to condense it.

        Keep every conclusion, open hypothesis, dead end and next step that still matters.

        Args:
            content: The complete new notebook in Markdown.
        """
        def run():
            lab.notebook_rewrite(content)
            return {"ok": True, "notebook_chars": len(lab.notebook())}
        return call("rewrite_notebook", run)

    tools = [get_lab_status, describe_strategy_space, list_experiments, run_backtest,
             run_optimization, run_baseline, run_robustness, validate_strategy, promote_to_paper,
             retire_strategy, get_paper_trades, get_market_context, get_signal_review_stats,
             read_notebook, append_notebook, rewrite_notebook]
    return tools if cycle == "research" else [t for t in tools if t is not run_optimization]
