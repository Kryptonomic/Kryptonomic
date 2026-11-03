"""Command line: python -m fibbot [--config PATH] [--home DIR] <command> (see DESIGN.md,
"fibbot/scheduler.py and fibbot/cli.py"). Research and PAPER trading only: no command can place a
real order.

Output is plain-text tables; backtest, optimize, baseline, robustness, validate, status and sweep
print JSON with --json. Exit codes: 0 ok; 1 user error (bad arguments, config or spec, unknown
strategy, failed downloads, tick errors or failed scheduler steps, e.g. a candle stall), printed
without a traceback; 2 brain unavailable; 130 interrupted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from . import autopilot, scheduler
from .backtest import iso_day
from .config import load_config
from .data.candles import CandleStore
from .data.universe import get_universe
from .lab import BudgetExceeded, Lab, iso_time
from .report import render_status, write_report
from .strategy import PARAM_SPACE, PRESETS, StrategySpec
from .types import TF_SECONDS

EXIT_USER, EXIT_BRAIN, EXIT_INTERRUPTED = 1, 2, 130
WINDOWS = ("research", "validation", "all")
BRAIN_HINT = ("The brain needs the 'anthropic' package and Anthropic credentials (ANTHROPIC_API_KEY)."
              " Without them, 'python -m fibbot sweep' runs the deterministic research autopilot"
              " and 'python -m fibbot run --no-brain' the scheduler.")
R, PCT, X2 = "{:+.3f}", "{:.1%}", "{:.2f}"


class UserError(Exception):
    """Bad input: the message is printed without a traceback (exit code 1)."""


class BrainMissing(Exception):
    """Claude cannot be used right now (exit code 2)."""


# -- formatting --------------------------------------------------------------------------------

def _f(x, fmt: str = "{:.4g}") -> str:
    """Format a number; '-' for None."""
    return "-" if x is None else fmt.format(x)


def _yes(x) -> str:
    return "yes" if x else "no"


def table(headers: list[str], rows: list[list]) -> str:
    """Aligned plain-text table: first column left-aligned, the others right-aligned."""
    cells = [[str(h) for h in headers], *([str(c) for c in row] for row in rows)]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]

    def line(r):
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                         for i, (c, w) in enumerate(zip(r, widths))).rstrip()
    return "\n".join([line(cells[0]), line(["-" * w for w in widths]), *map(line, cells[1:])])


def _params(p: dict | None) -> str:
    return "-" if p is None else " ".join(f"{k}={v}" for k, v in p.items())


def _header(title: str, out: dict) -> str:
    per = out["period"]
    return (f"{title}: {out['name']} [{out['spec_id']}] {out['timeframe']} {out['direction']},"
            f" {out['window']} window {per['start']}..{per['end']}, {out['n_symbols']} symbols")


def _metrics(m: dict) -> list[str]:
    reasons = ", ".join(f"{k} {v}" for k, v in sorted(m["exit_reasons"].items())) or "-"
    return [
        f"trades {m['n']} closed + {m['n_open']} open | win rate {_f(m['win_rate'], PCT)} | mean R"
        f" {_f(m['avg_r'], R)} | median R {_f(m['median_r'], R)} | total R {_f(m['total_r'], '{:+.1f}')}"
        f" | PF {_f(m['profit_factor'], X2)}",
        f"avg win {_f(m['avg_win_r'], R)} R, avg loss {_f(m['avg_loss_r'], R)} R, longest losing"
        f" streak {m['max_consec_losses']}, avg bars held {_f(m['avg_bars_held'], '{:.1f}')},"
        f" long/short {m['long_n']}/{m['short_n']}, Sharpe per trade {_f(m['sharpe_per_trade'], '{:.3f}')}",
        f"equity: return {_f(m['total_return'], '{:+.1%}')}, CAGR {_f(m['cagr'], '{:+.1%}')}, max drawdown"
        f" {_f(m['max_drawdown'], PCT)}, Sharpe {_f(m['sharpe'], X2)}, Sortino {_f(m['sortino'], X2)},"
        f" Calmar {_f(m['calmar'], X2)} over {m['n_days']} days",
        f"exit reasons: {reasons}",
    ]


def _portfolio(p: dict) -> str:
    skipped = ", ".join(f"{k} {v}" for k, v in sorted(p["skipped"].items())) or "none"
    return (f"account: {p['start_equity']:,.0f} -> {p['final_equity']:,.2f} (realized"
            f" {p['realized_equity']:,.2f}), {p['n_taken']} trades taken, skipped by risk limits: {skipped}")


def _folds(out: dict) -> list[str]:
    rows = [[f["fold"], iso_day(f["start"]), iso_day(f["end"]), f["n"], _f(f["avg_r"], R),
             _f(f["total_r"], "{:+.1f}"), _f(f["profit_factor"], X2)] for f in out["folds"]]
    return [table(["fold", "start", "end", "n", "mean R", "total R", "PF"], rows),
            f"positive folds {_f(out['fold_positive_frac'], '{:.0%}')}, worst fold mean R"
            f" {_f(out['min_fold_avg_r'], R)}"]


def _lite_rows(d: dict) -> list[list]:
    return [[k, v["n"], _f(v["avg_r"], R), _f(v["total_r"], "{:+.1f}"),
             _f(v.get("win_rate"), "{:.0%}"), _f(v.get("profit_factor"), X2)] for k, v in d.items()]


def show_backtest(out: dict) -> str:
    lines = [_header("Backtest", out), *_metrics(out["metrics"]), _portfolio(out["portfolio"]), "",
             *_folds(out), "", table(["symbol", "n", "mean R", "total R", "win", "PF"],
                                     _lite_rows(out["per_symbol"]))]
    if out["per_symbol_omitted"]:
        lines.append(f"({out['per_symbol_omitted']} middle symbols not shown)")
    lines += ["", table(["year", "n", "mean R", "total R"],
                        [row[:4] for row in _lite_rows(out["per_year"])])]
    if out.get("trades"):
        lines += ["", table(["symbol", "side", "leg", "entry (UTC)", "entry", "stop", "exit (UTC)",
                             "exit", "reason", "R", "bars"],
                            [[t["symbol"], "long" if t["side"] > 0 else "short", t["leg_id"],
                              t["entry_t"], _f(t["entry_price"], "{:.6g}"), _f(t["stop_price"], "{:.6g}"),
                              t["exit_t"] or "-", _f(t["exit_price"], "{:.6g}"), t["exit_reason"],
                              _f(t["r_multiple"], R), t["bars_held"]] for t in out["trades"]])]
    if out.get("holdout_looks"):
        lines.append(f"holdout peek: look #{out['holdout_looks']} at this spec's locked period"
                     " (counted as a trial)")
    lines.append(f"experiment #{out['experiment_id']}")
    return "\n".join(lines)


def show_validate(out: dict) -> str:
    gate = out["gate"]
    checks = [[c["name"], _f(c["value"]), _f(c["threshold"]), "yes" if c["passed"] else "NO"]
              for c in gate["checks"]]
    base, pit = out.get("baseline") or {}, out.get("validation_universe") or {}
    lines = [_header("Validation", out),
             f"gate {'PASSED' if out['passed'] else 'FAILED'}: deflated Sharpe"
             f" {_f(gate['deflated_sharpe'], '{:.3f}')} against {out.get('dsr_trials', out['n_trials'])}"
             f" {out.get('dsr_basis', 'research')} trials (Sharpe variance {_f(out['var_sr'])};"
             f" effective trades {_f(out.get('n_eff'), '{:.1f}')}; {out['n_trials']} research trials)",
             table(["check", "value", "threshold", "passed"], checks)]
    if base:
        lines.append(f"random entries on this window: mean R {_f(base.get('baseline_avg_r_mean'), R)}"
                     f" ({base.get('n_runs')} runs); edge {_f(base.get('edge_r'), R)} R"
                     f" (z {_f(base.get('edge_z'), X2)}), regime-filtered edge"
                     f" {_f(base.get('edge_regime_r'), R)} R; beats {_f(base.get('beats_frac'), '{:.0%}')}"
                     " of the runs")
    if pit:
        lines.append(f"validation universe {pit.get('mode')} (ranked at {pit.get('ranked_at')}):"
                     f" dropped {', '.join(pit.get('dropped') or []) or '-'}; added"
                     f" {', '.join(pit.get('added') or []) or '-'}")
    if "holdout_looks" in out:
        lines.append(f"distinct looks at this spec's holdout: {out['holdout_looks']}"
                     + (" (identical draw replayed)" if out.get("replay") else ""))
    lines += ["", *_metrics(out["metrics"]), _portfolio(out["portfolio"]), "", *_folds(out)]
    if out["passed"]:
        lines.append("passed: put it on paper with 'promote' (same --preset/--spec) --rationale TEXT")
    lines.append(f"experiment #{out['experiment_id']}")
    return "\n".join(lines)


def show_optimize(out: dict) -> str:
    oos, best = out["oos"], out["best"]
    folds = [[c["fold"], _params(c["params"]), c["is_n"], _f(c["is_avg_r"], R), f["n"], _f(f["avg_r"], R)]
             for c, f in zip(out["chosen"], out["oos_by_fold"])]
    lines = [f"Walk-forward optimization: {out['n_combos']} combos ({out['n_evaluated']} evaluated,"
             f" {out['n_invalid']} invalid), {out['n_folds']} folds, objective {out['objective']},"
             f" chosen combos need >= {out['min_is_trades']} in-sample trades",
             *(f"  invalid: {r}" for r in out["invalid_reasons"]),
             f"walk-forward held-out folds: {oos['n']} trades, mean R {_f(oos['avg_r'], R)}, PF"
             f" {_f(oos['profit_factor'], X2)}, win rate {_f(oos['win_rate'], PCT)}, total R"
             f" {_f(oos['total_r'], '{:+.1f}')}", "",
             table(["fold", "chosen on earlier folds", "IS n", "IS mean R", "OOS n", "OOS mean R"], folds),
             "", "top on the whole research window:",
             table(["params", "spec_id", "n", "mean R", "PF", "positive folds"],
                   [[_params(t["params"]), t["spec_id"], t["n"], _f(t["avg_r"], R),
                     _f(t["profit_factor"], X2), _f(t["fold_positive_frac"], "{:.0%}")] for t in out["top"]])]
    if best:
        lines.append(f"best [{best['spec_id']}] {_params(best['params'])}: n {best['n']}, mean R"
                     f" {_f(best['avg_r'], R)}, PF {_f(best['profit_factor'], X2)}, enough trades"
                     f" {_yes(best['eligible'])}")
    for param, by_value in out["sensitivity"].items():
        lines.append(f"sensitivity {param}: " + ", ".join(f"{v} {_f(r, R)}" for v, r in by_value.items()))
    lines.append(f"experiment #{out['experiment_id']}")
    return "\n".join(lines)


def show_baseline(out: dict) -> str:
    return (f"Random-entry baseline [{out['spec_id']}]: strategy mean R {_f(out['strategy_avg_r'], R)}"
            f" over {out['n_trades']} trades vs random entries {_f(out['baseline_avg_r_mean'], R)}"
            f" +/- {_f(out['baseline_avg_r_std'], '{:.3f}')} ({out['n_runs']} runs, seed {out['seed']})\n"
            f"edge {_f(out['edge_r'], R)} R; the strategy beats {_f(out['beats_frac'], '{:.0%}')} of"
            f" the runs\nexperiment #{out['experiment_id']}")


def show_robustness(out: dict) -> str:
    rows = [[r["param"], r["value"], r["n"], _f(r["avg_r"], R), _f(r["total_r"], "{:+.1f}"),
             _f(r["profit_factor"], X2), _f(r["fold_positive_frac"], "{:.0%}")] for r in out["rows"]]
    return "\n".join([
        f"Robustness [{out['spec_id']}]: {out['n']} of {out['n_candidates']} one-step neighbours, positive"
        f" mean R {_f(out['positive_frac'], '{:.0%}')}, median {_f(out['median_avg_r'], R)}, worst"
        f" {_f(out['worst_avg_r'], R)}",
        table(["param", "value", "n", "mean R", "total R", "PF", "positive folds"], rows),
        f"experiment #{out['experiment_id']}"])


def show_sweep(out: dict) -> str:
    lines = [f"Autopilot sweep {out['as_of']}: {len(out['ranking'])} presets ranked", "",
             table(["preset", "tf", "n", "mean R", "PF", "positive folds", "edge R", "eligible"],
                   [[r["preset"], r["timeframe"], r["n"], _f(r["avg_r"], R), _f(r["profit_factor"], X2),
                     _f(r["fold_positive_frac"], "{:.0%}"), _f(r["edge_r"], R), _yes(r["eligible"])]
                    for r in out["ranking"]])]
    if out["validations"]:
        lines += ["", table(["validated", "passed", "n", "mean R", "PF", "deflated Sharpe", "failed checks"],
                            [[v["preset"] + (" (kept)" if v.get("reused") else ""), _yes(v["passed"]),
                              v["n"], _f(v["avg_r"], R), _f(v["profit_factor"], X2),
                              _f(v["deflated_sharpe"], "{:.3f}"), ", ".join(v["failed"]) or "-"]
                             for v in out["validations"]])]
        if any(v.get("reused") for v in out["validations"]):
            lines.append("(kept) = verdict of an earlier validation on the current window (one look per period)")
    lines += [f"not validated {s['preset']}: {s['reason']}" for s in out.get("skipped", [])]
    lines += [f"promoted {p['preset']} [{p['strategy_id']}], paper from {p['paper_start']}"
              for p in out["promoted"]]
    lines += [f"error: {e['preset']} {e['step']}: {e['error']}" for e in out["errors"]]
    lines.append(f"experiment #{out['experiment_id']}")
    return "\n".join(lines)


def show_tick(out: dict) -> str:
    if not out["strategies"] and not out["errors"]:
        return "no strategies on paper (promote a validated spec first)"
    rows = [[sid, _f(s["equity"], "{:,.2f}"), s["open"], len(s["new_signals"]), len(s["new_entries"]),
             len(s["new_exits"])] for sid, s in out["strategies"].items()]
    lines = [table(["strategy", "equity", "open", "new signals", "entries", "exits"], rows)]
    for kind in ("new_entries", "new_exits"):
        for sid, s in out["strategies"].items():
            lines += [f"{sid} {kind[4:-1]}: {d['symbol']} {'long' if d['side'] > 0 else 'short'} {d['leg_id']}"
                      f" at {_f(d['entry_price'], '{:.6g}')}"
                      + ("" if kind == "new_entries" else f" -> {d['exit_reason']} {_f(d['r_multiple'], R)} R")
                      for d in s[kind]]
    lines += [f"error: strategy {e['strategy_id']} symbol {e['symbol']}: {e['error']}" for e in out["errors"]]
    return "\n".join(lines)


def show_cycle(res: dict) -> str:
    u = res["usage"]
    head = (f"{res['kind']} cycle: {res['turns']} turns, stop {res['stop_reason']},"
            f" {len(res['promoted'])} promoted, {len(res['retired'])} retired (journal #{res['journal_id']})"
            f"\ntokens: input {u['input_tokens']}, output {u['output_tokens']}, cache read"
            f" {u['cache_read_input_tokens']}")
    return head + ("\nthe model refused this cycle" if res["refused"] else "") + "\n\n" + (res["text"] or "")


def show_signal_reviews(results: list[dict]) -> str:
    if not results:
        return "no unreviewed signals (or today's review quota is used up)"
    return "\n".join(f"{r['symbol']} {r['leg_id']} [{r['strategy_id']}]: "
                     + (f"{r['decision']} ({r['confidence']:.2f}) {r['reason']}" if "decision" in r
                        else f"ERROR {r['error']}") for r in results)


def show_presets() -> str:
    rows = []
    for name, s in PRESETS.items():
        entry = f"limit {s.entry_at}" if s.entry_mode == "limit" else f"confirm {s.confirm_type}"
        stop = f"{s.stop_ratio:g}" + (f"+{s.stop_atr_buffer:g}ATR" if s.stop_atr_buffer else "")
        targets = (f"{s.tp1_ext:g} ({s.tp1_frac:.0%})" + ("" if s.tp1_frac == 1 else f" / {s.tp2_ext:g}")
                   + (f" trail {s.trail_atr:g}ATR" if s.trail_atr else ""))
        trend = s.trend_filter + ("" if s.trend_filter == "none" else f" {s.trend_ema}") + (
            f" adx>={s.adx_min:g}" if s.adx_min else "")
        rows.append([name, s.timeframe, s.direction, f"{s.entry_zone_lo:g}-{s.entry_zone_hi:g}", entry,
                     stop, targets, trend, s.spec_id()])
    return table(["preset", "tf", "side", "zone", "entry", "stop", "targets", "trend", "spec_id"], rows)


# -- inputs ------------------------------------------------------------------------------------

def read_json(text: str, what: str):
    """Inline JSON (text starting with '{' or '['), else the contents of the JSON file `text`."""
    s = text.strip()
    if not s.startswith(("{", "[")):
        path = Path(s).expanduser()
        if not path.is_file():
            raise UserError(f"{what}: not inline JSON and no such file: {text}")
        s = path.read_text(encoding="utf-8")
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise UserError(f"{what}: invalid JSON ({e})") from None


def load_spec(args) -> StrategySpec:
    """Validated spec from --preset NAME or --spec JSON_OR_FILE ({"preset": NAME, ...} applies the
    other keys to a preset), with --tf (backtest) overriding the timeframe."""
    if args.preset:
        spec = PRESETS[args.preset]
    else:
        d = read_json(args.spec, "--spec")
        if not isinstance(d, dict):
            raise UserError("--spec must be a JSON object of strategy parameters")
        d = dict(d)
        base = d.pop("preset", None)
        if base is not None and base not in PRESETS:
            raise UserError(f"--spec: unknown preset {base!r}; known: {', '.join(PRESETS)}")
        spec = PRESETS[base].with_overrides(**d) if base else StrategySpec.from_dict(d)
    if getattr(args, "tf", None):
        spec = spec.with_overrides(timeframe=args.tf)
    return spec.validate()


@contextmanager
def open_lab(cfg: dict):
    lab = Lab(cfg)
    try:
        yield lab
    finally:
        lab.store.close()


def require_brain(cfg: dict):
    """An available Brain, else BrainMissing with the reason."""
    brain = scheduler.make_brain(cfg)
    ok, why = brain.available()
    if not ok:
        raise BrainMissing(why)
    return brain


def emit(args, obj, show) -> None:
    """Print obj as JSON with --json, else show(obj)."""
    print(json.dumps(obj, indent=2, default=str) if getattr(args, "json", False) else show(obj))


# -- commands ----------------------------------------------------------------------------------

def cmd_universe(args, cfg) -> int:
    symbols = get_universe(cfg, refresh=args.refresh)
    u = cfg["universe"]
    print(f"universe: top {u['top_n']} by market cap, {u['quote']} spot pairs with >="
          f" {u['min_history_days']} days of history")
    print("\n".join(f"{i:3d}. {sym}" for i, sym in enumerate(symbols, 1)))
    return 0


def cmd_data(args, cfg) -> int:
    symbols = [s.upper() for s in args.symbols] if args.symbols else get_universe(cfg)
    tfs = list(dict.fromkeys(args.tf or cfg["data"]["timeframes"]))
    pairs = list(itertools.product(symbols, tfs))
    candles, failed = CandleStore(cfg), 0
    print(f"updating {len(symbols)} symbols x {', '.join(tfs)} (cache {cfg['home']}/candles)", flush=True)
    for i, (sym, tf) in enumerate(pairs, 1):
        tag = f"[{i}/{len(pairs)}] {sym} {tf}"
        try:
            before = len(candles.get(sym, tf))
            bars = candles.update(sym, tf)
        except Exception as e:  # report and go on: one bad symbol must not stop the download
            failed += 1
            print(f"{tag}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        span = (f"{iso_day(bars.t[0])}..{iso_time(bars.t[-1] + TF_SECONDS[tf])}" if len(bars)
                else "no data")
        print(f"{tag}: +{len(bars) - before} bars, {len(bars)} cached ({span})", flush=True)
    if failed:
        print(f"{failed} of {len(pairs)} updates failed", file=sys.stderr)
        return EXIT_USER
    return 0


def cmd_presets(args, cfg) -> int:
    print(show_presets())
    return 0


def cmd_backtest(args, cfg) -> int:
    spec = load_spec(args)
    if args.window in ("validation", "all"):
        print("warning: this looks at the locked validation period; it counts as a trial and as a"
              " look at this spec's holdout (see FIBBOT.md)", file=sys.stderr)
    with open_lab(cfg) as lab:
        emit(args, lab.backtest(spec, args.window, args.symbols, trades=args.trades), show_backtest)
    return 0


def cmd_optimize(args, cfg) -> int:
    spec, grid = load_spec(args), read_json(args.grid, "--grid")
    with open_lab(cfg) as lab:
        emit(args, lab.optimize(spec, grid), show_optimize)
    return 0


def cmd_baseline(args, cfg) -> int:
    spec = load_spec(args)
    with open_lab(cfg) as lab:
        emit(args, lab.baseline(spec, n_runs=args.runs), show_baseline)
    return 0


def cmd_robustness(args, cfg) -> int:
    spec = load_spec(args)
    with open_lab(cfg) as lab:
        emit(args, lab.robustness(spec), show_robustness)
    return 0


def cmd_validate(args, cfg) -> int:
    spec = load_spec(args)
    with open_lab(cfg) as lab:
        emit(args, lab.validate(spec), show_validate)
    return 0


def cmd_promote(args, cfg) -> int:
    spec = load_spec(args)
    with open_lab(cfg) as lab:
        p = lab.promote(spec, args.name or args.preset or spec.name, args.rationale)
    print(f"promoted {p['name']} [{p['strategy_id']}] {p['timeframe']}: paper trading starts"
          f" {p['paper_start']} (roster {p['roster_size']}/{p['max_roster']}, validation"
          f" experiment #{p['validation_experiment_id']})")
    return 0


def cmd_retire(args, cfg) -> int:
    with open_lab(cfg) as lab:
        out = lab.retire(args.strategy_id, args.reason)
    a = out["account"]
    print(f"retired {out['name']} [{out['strategy_id']}]: {out['reason']}\nafter {a['days']:.1f} days:"
          f" {a['n_closed']} closed trades, mean R {_f(a['stats']['avg_r'], R)}, return"
          f" {_f(a['return'], '{:+.2%}')}, max drawdown {_f(a['max_drawdown'], PCT)}")
    return 0


def cmd_sweep(args, cfg) -> int:
    with open_lab(cfg) as lab:
        emit(args, autopilot.sweep(lab, presets=args.presets, promote=args.promote), show_sweep)
    return 0


def cmd_epoch(args, cfg) -> int:
    """Show the pinned validation epoch, or --roll it forward by research.validation_months."""
    with open_lab(cfg) as lab:
        if args.roll:
            out = lab.roll_validation_epoch()
            print(f"validation epoch rolled: the locked validation window now ends"
                  f" {iso_day(out['validation_end'])} (was {iso_day(out['previous'])}); journaled")
            return 0
        epoch = lab.validation_epoch()
    if epoch is None:
        print("no validation epoch is pinned: the validation window slides with the data until the"
              " first promotion")
    else:
        print(f"validation epoch: the locked validation window ends {iso_day(epoch['validation_end'])}"
              f" ({epoch.get('reason', '')}); 'epoch --roll' moves it forward by"
              f" {cfg['research']['validation_months']} months")
    return 0


def cmd_tick(args, cfg) -> int:
    """Exit 1 when the tick reported errors (a strategy, a symbol or a candle download failed)."""
    with open_lab(cfg) as lab:
        out = lab.paper().tick(update_data=True)
    print(show_tick(out))
    if out["errors"]:
        print(f"{len(out['errors'])} errors during the tick (listed above)", file=sys.stderr)
        return EXIT_USER
    return 0


def cmd_brain(args, cfg) -> int:
    brain = require_brain(cfg)
    from .brain import cycles
    with open_lab(cfg) as lab:
        if args.action == "signals":
            if cfg["brain"]["signal_review"] == "off":
                print("brain.signal_review is off: nothing to review")
                return 0
            results = cycles.review_signals(lab, brain)
            print(show_signal_reviews(results))
            lost = next((r["error"] for r in results if r.get("error", "").startswith("brain unavailable")),
                        None)
            if lost:
                raise BrainMissing(lost.removeprefix("brain unavailable: "))
            return 0
        fn = cycles.research_cycle if args.action == "research" else cycles.review_cycle
        kw = {"stop": scheduler.cycle_stop(cfg)} if scheduler.accepts(fn, "stop") else {}
        res = fn(lab, brain, **kw)
    if not res["ok"]:
        raise BrainMissing(res["error"])
    print(show_cycle(res))
    return 0


def cmd_run(args, cfg) -> int:
    use_brain = not args.no_brain
    if not args.once:
        scheduler.run_forever(cfg, use_brain=use_brain)
        return 0
    scheduler.setup_logging(cfg)
    out = scheduler.run_once(cfg, use_brain=use_brain)
    return EXIT_USER if any(s["status"] == "failed" for s in out["steps"]) else 0


def cmd_status(args, cfg) -> int:
    with open_lab(cfg) as lab:
        print(json.dumps(lab.status(), indent=2, default=str) if args.json else render_status(lab))
    return 0


def cmd_report(args, cfg) -> int:
    with open_lab(cfg) as lab:
        print(f"wrote {write_report(lab)}")
    return 0


# -- parser ------------------------------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    """Usage errors raise UserError (exit code 1; argparse's own 2 means 'brain unavailable' here)."""

    def error(self, message):
        self.print_usage(sys.stderr)
        raise UserError(f"{self.prog}: {message}")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="fibbot", description="Fibonacci-retracement PAPER-trading research bot for high"
                " market-cap coins, with a Claude research brain. No live trading.")
    p.add_argument("--config", metavar="PATH", help="JSON config (default: $FIBBOT_CONFIG or ./fibbot.json)")
    p.add_argument("--home", metavar="DIR", help="state directory (overrides config home and $FIBBOT_HOME)")
    sub = p.add_subparsers(dest="command", metavar="COMMAND", required=True)

    spec = _Parser(add_help=False)
    g = spec.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", choices=list(PRESETS), metavar="NAME", help="a preset (see 'presets')")
    g.add_argument("--spec", metavar="JSON_OR_FILE",
                   help='strategy parameters as inline JSON or a JSON file; {"preset": NAME, ...} overrides a preset')
    js = _Parser(add_help=False)
    js.add_argument("--json", action="store_true", help="print JSON instead of text")

    def add(name, func, text, parents=()):
        sp = sub.add_parser(name, help=text, description=text, parents=list(parents))
        sp.set_defaults(func=func)
        return sp

    add("universe", cmd_universe, "show the coin universe").add_argument(
        "--refresh", action="store_true", help="re-rank by market cap now")
    sp = add("data", cmd_data, "download / update cached candles")
    sp.add_argument("--tf", nargs="+", choices=list(TF_SECONDS), help="timeframes (default: data.timeframes)")
    sp.add_argument("--symbols", nargs="+", metavar="SYMBOL", help="symbols (default: the universe)")
    add("presets", cmd_presets, "list the strategy presets")
    sp = add("backtest", cmd_backtest, "backtest a spec on a window", (spec, js))
    sp.add_argument("--window", choices=WINDOWS, default="research")
    sp.add_argument("--symbols", nargs="+", metavar="SYMBOL", help="symbols (default: the universe)")
    sp.add_argument("--trades", type=int, default=0, metavar="N", help="also list the last N trades")
    sp.add_argument("--tf", choices=PARAM_SPACE["timeframe"]["choices"], help="override the spec timeframe")
    add("optimize", cmd_optimize, "walk-forward grid search on the research window", (spec, js)).add_argument(
        "--grid", required=True, metavar="JSON_OR_FILE", help='{"param": [values, ...], ...}')
    add("baseline", cmd_baseline, "compare with random entries (same side, risk and management)",
        (spec, js)).add_argument("--runs", type=int, default=10, metavar="N")
    add("robustness", cmd_robustness, "re-test with every parameter moved one step both ways"
        " (duplicates of the base flagged)", (spec, js))
    add("validate", cmd_validate, "run the locked validation window and the validation gate", (spec, js))
    sp = add("promote", cmd_promote, "start paper trading a spec that passed validation", (spec,))
    sp.add_argument("--name", help="strategy name (default: the preset or spec name)")
    sp.add_argument("--rationale", required=True, help="why this strategy deserves a paper slot")
    sp = add("retire", cmd_retire, "stop paper trading a strategy")
    sp.add_argument("strategy_id")
    sp.add_argument("--reason", required=True)
    sp = add("sweep", cmd_sweep, "deterministic research without Claude: rank, validate presets", (js,))
    sp.add_argument("--promote", action="store_true", help="put presets that pass on paper")
    sp.add_argument("--presets", nargs="+", choices=list(PRESETS), metavar="NAME",
                    help="default: every preset whose timeframe is in data.timeframes")
    add("epoch", cmd_epoch, "show the pinned validation epoch (or roll it forward)").add_argument(
        "--roll", action="store_true",
        help="move the locked validation window forward by research.validation_months (journaled)")
    add("tick", cmd_tick, "update candles and advance the paper accounts once")
    add("brain", cmd_brain, "run a Claude brain cycle").add_argument(
        "action", choices=("research", "review", "signals"))
    sp = add("run", cmd_run, "run the scheduler (hourly passes; Ctrl-C stops after the current step)")
    sp.add_argument("--once", action="store_true", help="one pass, then exit (1 if a step failed)")
    sp.add_argument("--no-brain", action="store_true", help="never call Claude (autopilot research)")
    add("status", cmd_status, "lab and paper roster overview", (js,))
    add("report", cmd_report, "write home/REPORT.md")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run one command (argv defaults to sys.argv[1:]) and return its exit code."""
    try:
        args = build_parser().parse_args(argv)
        cfg = load_config(args.config, {"home": args.home} if args.home else None)
        return args.func(args, cfg)
    except BrainMissing as e:
        print(f"brain unavailable: {e}\n{BRAIN_HINT}", file=sys.stderr)
        return EXIT_BRAIN
    except (UserError, ValueError, KeyError, OSError, BudgetExceeded) as e:
        print(f"error: {e.args[0] if isinstance(e, KeyError) and e.args else e}", file=sys.stderr)
        return EXIT_USER
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
