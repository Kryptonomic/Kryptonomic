"""Multi-symbol backtests: run one StrategySpec over many symbols, size the trades into one account
and summarize (pure, no network).

run_strategy replays FibEngine per symbol on bars.window(start_t, end_t, warmup_bars(spec)) with
trade_start_t=start_t, optionally in a process pool. backtest() adds the portfolio simulation and
the statistics. Trade statistics (metrics, per_symbol, per_year) use CLOSED trades; a position
still open at the end of the window is kept in `trades`, sized by the portfolio and marked into
final_equity, and counted in metrics["n_open"].
"""
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

from .metrics import equity_stats, trade_stats
from .portfolio import PortfolioResult, simulate_portfolio
from .strategy import FibEngine, RunResult, StrategySpec, warmup_bars
from .types import TF_SECONDS, Bars, Costs, Trade

LITE_KEYS = ("n", "total_r", "avg_r", "win_rate", "profit_factor")


def rounded(x, digits: int = 4):
    """JSON-safe copy of x: floats rounded to `digits` decimals (|x| >= 1) or `digits` significant
    digits (|x| < 1), nan/inf -> None, tuples -> lists; dicts and lists are copied recursively."""
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return round(x, digits) if abs(x) >= 1 else float(f"{x:.{digits}g}")
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, digits) for v in x]
    return x


def iso_day(t: int | None) -> str | None:
    """UTC date 'YYYY-MM-DD' of a Unix time (None stays None)."""
    return None if t is None else datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")


def as_spec(spec: StrategySpec | dict) -> StrategySpec:
    """A validated StrategySpec from a spec or its dict (ValueError if invalid)."""
    return (spec if isinstance(spec, StrategySpec) else StrategySpec.from_dict(spec)).validate()


def costs_from_cfg(cfg: dict) -> Costs:
    """Costs from cfg["costs"] (missing keys keep the Costs defaults)."""
    c = cfg.get("costs") or {}
    return Costs(**{k: float(c[k]) for k in ("fee_rate", "slippage") if k in c})


def resolve_workers(n: int | None) -> int:
    """Process count for research.workers: <= 0 or None -> os.cpu_count() - 1 (at least 1)."""
    n = int(n or 0)
    return n if n > 0 else max(1, (os.cpu_count() or 2) - 1)


def _run_task(task) -> RunResult:
    """Process-pool worker: (spec, costs, strategy_id, windowed bars, trade_start_t, skip_legs)."""
    spec, costs, strategy_id, bars, start_t, skip = task
    return FibEngine(spec, costs, strategy_id).run(bars, trade_start_t=start_t, skip_legs=skip)


def run_strategy(spec: StrategySpec | dict, bars_by_symbol: dict[str, Bars], start_t: int | None,
                 end_t: int | None, costs: Costs, workers: int = 1,
                 skip_legs: dict[str, set] | None = None,
                 strategy_id: str | None = None) -> dict[str, RunResult]:
    """FibEngine.run(bars.window(start_t, end_t, warmup_bars(spec)), trade_start_t=start_t) for every
    symbol, in input order. skip_legs maps symbol -> leg_ids to ignore. workers > 1 (and more than
    one symbol) runs the symbols in a process pool (default start method; scripts on platforms
    that spawn need the `if __name__ == "__main__":` guard); results equal a serial run."""
    spec = as_spec(spec)
    warm, skip_legs = warmup_bars(spec), skip_legs or {}
    tasks = [(spec, costs, strategy_id, bars.window(start_t, end_t, warm), start_t,
              frozenset(skip_legs.get(sym, ()))) for sym, bars in bars_by_symbol.items()]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(min(workers, len(tasks))) as ex:
            results = list(ex.map(_run_task, tasks))
    else:
        results = [_run_task(task) for task in tasks]
    return dict(zip(bars_by_symbol, results))


def _lite(trades: list[Trade]) -> dict:
    """{"n", "total_r", "avg_r", "win_rate", "profit_factor"} of a trade list."""
    s = trade_stats(trades)
    return {k: s[k] for k in LITE_KEYS}


def data_period(bars_by_symbol: dict[str, Bars], start_t: int | None,
                end_t: int | None) -> tuple[int | None, int | None]:
    """Time span a backtest covers: [max(start_t, first bar), min(end_t, last bar close))."""
    live = [b for b in bars_by_symbol.values() if len(b)]
    lo = min((b.t[0] for b in live), default=None)
    hi = max((b.t[-1] + TF_SECONDS[b.tf] for b in live), default=None)
    if start_t is not None:
        lo = start_t if lo is None else max(lo, start_t)
    if end_t is not None:
        hi = end_t if hi is None else min(hi, end_t)
    return lo, hi


@dataclass
class BacktestResult:
    """One spec over many symbols and one window. metrics = trade_stats(closed trades) |
    equity_stats(portfolio curve from period start to end, ending at final_equity) | {"n_open"}."""
    spec: dict
    spec_id: str
    start_t: int | None
    end_t: int | None
    symbols: list[str]
    trades: list[Trade]                 # closed + open-marked, sorted by (entry_t, symbol, leg_id)
    portfolio: PortfolioResult
    metrics: dict
    per_symbol: dict[str, dict]         # every symbol: {"n", "total_r", "avg_r", "win_rate", "profit_factor"}
    per_year: dict[str, dict]           # {"2023": {"n", "total_r", "avg_r"}} by UTC entry year
    period: tuple[int | None, int | None] = (None, None)   # data_period(...)

    @property
    def closed_trades(self) -> list[Trade]:
        return [tr for tr in self.trades if tr.exit_t is not None]

    def summary(self, max_symbols: int = 20) -> dict:
        """Compact, rounded, JSON-safe overview (no trade lists). With more than max_symbols
        symbols only the best ceil(max/2) and worst floor(max/2) by total_r are listed."""
        rows = sorted(self.per_symbol.items(), key=lambda kv: (-kv[1]["total_r"], kv[0]))
        omitted = max(0, len(rows) - max(0, max_symbols))
        if omitted:
            rows = rows[:max_symbols - max_symbols // 2] + rows[len(rows) - max_symbols // 2:]
        pf = self.portfolio
        return rounded({
            "spec_id": self.spec_id, "name": self.spec.get("name"),
            "timeframe": self.spec.get("timeframe"), "direction": self.spec.get("direction"),
            "period": {"start": iso_day(self.period[0]), "end": iso_day(self.period[1])},
            "n_symbols": len(self.symbols), "n_open": self.metrics["n_open"],
            "metrics": self.metrics,
            "portfolio": {"start_equity": pf.start_equity, "final_equity": pf.final_equity,
                          "realized_equity": pf.realized_equity, "n_taken": len(pf.taken),
                          "skipped": dict(Counter(s["reason"] for s in pf.skipped))},
            "per_symbol": dict(rows), "per_symbol_omitted": omitted,
            "per_year": self.per_year,
        })


def result_from_trades(spec: StrategySpec | dict, trades: list[Trade], symbols: list[str],
                       start_t: int | None, end_t: int | None, cfg: dict,
                       period: tuple[int | None, int | None] | None = None) -> BacktestResult:
    """Assemble a BacktestResult: sort the trades, simulate the portfolio (cfg["portfolio"]) and
    compute metrics. The equity curve for equity_stats runs from period[0] (start equity) through
    the realized exits to period[1] (final, marked equity); period defaults to (start_t, end_t)."""
    spec = as_spec(spec)
    trades = sorted(trades, key=lambda tr: (tr.entry_t, tr.symbol, tr.leg_id))
    closed = [tr for tr in trades if tr.exit_t is not None]
    pf = simulate_portfolio(trades, cfg.get("portfolio"))
    lo, hi = period or (start_t, end_t)
    curve = ([(lo, pf.start_equity)] if lo is not None else []) + list(pf.curve)
    if hi is not None:
        curve.append((hi, pf.final_equity))
    metrics = {**trade_stats(closed), **equity_stats(curve, pf.start_equity),
               "n_open": len(trades) - len(closed)}
    by_sym, by_year = defaultdict(list), defaultdict(list)
    for tr in closed:
        by_sym[tr.symbol].append(tr)
        by_year[datetime.fromtimestamp(tr.entry_t, tz=timezone.utc).strftime("%Y")].append(tr)
    per_year = {y: {k: v for k, v in _lite(ts).items() if k in ("n", "total_r", "avg_r")}
                for y, ts in sorted(by_year.items())}
    return BacktestResult(spec=spec.to_dict(), spec_id=spec.spec_id(), start_t=start_t, end_t=end_t,
                          symbols=list(symbols), trades=trades, portfolio=pf, metrics=metrics,
                          per_symbol={s: _lite(by_sym.get(s, [])) for s in symbols},
                          per_year=per_year, period=(lo, hi))


def backtest(spec: StrategySpec | dict, bars_by_symbol: dict[str, Bars], start_t: int | None,
             end_t: int | None, cfg: dict, workers: int | None = None) -> BacktestResult:
    """Run spec on every symbol over [start_t, end_t) and build the BacktestResult. Costs come
    from cfg["costs"], sizing from cfg["portfolio"], workers (None) from cfg["research"]["workers"]."""
    spec = as_spec(spec)
    if workers is None:
        workers = cfg.get("research", {}).get("workers", 0)
    runs = run_strategy(spec, bars_by_symbol, start_t, end_t, costs_from_cfg(cfg),
                        workers=resolve_workers(workers))
    trades = [tr for r in runs.values() for tr in r.trades + ([r.open_trade] if r.open_trade else [])]
    return result_from_trades(spec, trades, list(bars_by_symbol), start_t, end_t, cfg,
                              period=data_period(bars_by_symbol, start_t, end_t))
