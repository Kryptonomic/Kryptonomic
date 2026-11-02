"""Deterministic research without Claude (see DESIGN.md, "fibbot/autopilot.py").

sweep() evaluates presets on the research window with a random-entry baseline and fold stability,
ranks the robust ones, validates the best on the locked validation window and can promote those
that pass. It needs no API key; the scheduler runs it when the brain is unavailable.
"""
from __future__ import annotations

import time

from .backtest import rounded
from .lab import BudgetExceeded, Lab, iso_time
from .strategy import PRESETS

TOP_VALIDATE = 3               # how many eligible presets are validated
MIN_FOLD_POSITIVE = 0.5        # eligible: share of research folds with mean R > 0


def _row(name: str, bt: dict, base: dict, min_trades: int) -> dict:
    """Ranking row; eligible = enough trades, positive mean R, stable across folds and beating
    random entries (a preset that loses money is never worth a validation try)."""
    m = bt["metrics"]
    return {"preset": name, "spec_id": bt["spec_id"], "timeframe": bt["timeframe"], "n": m["n"],
            "avg_r": m["avg_r"], "profit_factor": m["profit_factor"],
            "fold_positive_frac": bt["fold_positive_frac"], "edge_r": base["edge_r"],
            "eligible": (m["n"] >= min_trades and m["avg_r"] > 0
                         and bt["fold_positive_frac"] >= MIN_FOLD_POSITIVE and base["edge_r"] > 0)}


def _journal_body(out: dict) -> str:
    """One bullet line per ranked preset, validation, promotion and error (plain text that also
    reads as a Markdown list)."""
    lines = [f"- {r['preset']} ({r['timeframe']}): n {r['n']}, mean R {r['avg_r']:+.3f},"
             f" PF {r['profit_factor']:.2f}, positive folds {r['fold_positive_frac']:.2f},"
             f" edge vs random {r['edge_r']:+.3f} R: {'eligible' if r['eligible'] else 'not eligible'}"
             for r in out["ranking"]]
    for v in out["validations"]:
        lines.append(f"- validation {v['preset']}: {'PASS' if v['passed'] else 'FAIL'} (n {v['n']},"
                     f" mean R {v['avg_r']:+.3f}, PF {v['profit_factor']:.2f},"
                     f" deflated Sharpe {v['deflated_sharpe']:.2f})"
                     + (f"; failed {', '.join(v['failed'])}" if v["failed"] else ""))
    lines += [f"- promoted {p['preset']} ({p['strategy_id']}), paper from {p['paper_start']}"
              for p in out["promoted"]]
    lines += [f"- {e['step']} {e['preset']} failed: {e['error']}" for e in out["errors"]]
    return "\n".join(lines)


def sweep(lab: Lab, presets: list[str] | None = None, promote: bool = False) -> dict:
    """Evaluate presets (default: the PRESETS whose timeframe is in data.timeframes) on the
    research window, each with a baseline; rank eligible presets (>= research.min_trades trades,
    mean R > 0, fold_positive_frac >= 0.5, baseline edge_r > 0) first, each group by mean R;
    validate the top TOP_VALIDATE eligible ones; with promote=True put passing ones on paper
    while the roster has room. Failures of single presets
    (no data, budget, roster full) are listed in "errors". Recorded as a "sweep" experiment and a
    "research" journal entry; kv "last_sweep_t" is set.

    Returns {"as_of", "presets", "ranking": [{preset, spec_id, timeframe, n, avg_r, profit_factor,
    fold_positive_frac, edge_r, eligible}], "validations": [{preset, spec_id, passed, n, avg_r,
    profit_factor, deflated_sharpe, failed, experiment_id}], "promoted": [...], "errors":
    [{preset, step, error}], "experiment_id"}.
    """
    tfs = set(lab.cfg["data"]["timeframes"])
    names = ([n for n, s in PRESETS.items() if s.timeframe in tfs] if presets is None
             else list(dict.fromkeys(presets)))
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise ValueError(f"unknown presets {unknown}; known: {sorted(PRESETS)}")
    min_trades = int(lab.cfg["research"]["min_trades"])
    rows, validations, promoted, errors = [], [], [], []
    for name in names:
        try:
            bt = lab.backtest(PRESETS[name], "research")
            base = lab.baseline(PRESETS[name])
        except (ValueError, BudgetExceeded) as e:
            errors.append({"preset": name, "step": "evaluate", "error": str(e)})
            continue
        rows.append(_row(name, bt, base, min_trades))
    rows.sort(key=lambda r: (not r["eligible"], -r["avg_r"]))       # stable: ties keep preset order

    for row in [r for r in rows if r["eligible"]][:TOP_VALIDATE]:
        try:
            v = lab.validate(PRESETS[row["preset"]])
        except (ValueError, BudgetExceeded) as e:
            errors.append({"preset": row["preset"], "step": "validate", "error": str(e)})
            continue
        m = v["metrics"]
        validations.append({"preset": row["preset"], "spec_id": v["spec_id"], "passed": v["passed"],
                            "n": m["n"], "avg_r": m["avg_r"], "profit_factor": m["profit_factor"],
                            "deflated_sharpe": v["gate"]["deflated_sharpe"],
                            "failed": [c["name"] for c in v["gate"]["checks"] if not c["passed"]],
                            "experiment_id": v["experiment_id"]})

    for v in validations if promote else []:
        if not v["passed"]:
            continue
        research_row = next(r for r in rows if r["preset"] == v["preset"])
        rationale = (f"Autopilot sweep: research mean R {research_row['avg_r']:+.3f} over"
                     f" {research_row['n']} trades (edge over random entries"
                     f" {research_row['edge_r']:+.3f} R); validation mean R {v['avg_r']:+.3f} over"
                     f" {v['n']} trades, deflated Sharpe {v['deflated_sharpe']:.2f}.")
        try:
            p = lab.promote(PRESETS[v["preset"]], v["preset"], rationale)
        except ValueError as e:
            errors.append({"preset": v["preset"], "step": "promote", "error": str(e)})
            continue
        promoted.append({"preset": v["preset"], "strategy_id": p["strategy_id"],
                         "paper_start": p["paper_start"]})

    now = int(time.time())
    out = rounded({"as_of": iso_time(now), "presets": names, "ranking": rows,
                   "validations": validations, "promoted": promoted, "errors": errors})
    out["experiment_id"] = lab.store.add_experiment(
        "sweep", None, None, None, None,
        {k: out[k] for k in ("ranking", "validations", "promoted", "errors")},
        source="autopilot", notes=f"presets={','.join(names)} promote={promote}", now_t=now)
    n_ok = sum(v["passed"] for v in validations)
    lab.store.add_journal(
        "research", f"Autopilot sweep: {sum(r['eligible'] for r in rows)}/{len(names)} presets eligible,"
                    f" {n_ok}/{len(validations)} passed validation, {len(promoted)} promoted",
        _journal_body(out), now_t=now)
    lab.store.set_kv("last_sweep_t", now)
    return out
