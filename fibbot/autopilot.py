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
    random entries (a preset that loses money is never worth a validation try). edge_r is the
    binding edge: min(uniform, regime-filtered) of the baseline (either alone can be lenient; the
    same rule Lab.promote applies)."""
    m = bt["metrics"]
    edge = min(base["edge_r"], base.get("edge_regime_r", base["edge_r"]))
    return {"preset": name, "spec_id": bt["spec_id"], "timeframe": bt["timeframe"], "n": m["n"],
            "avg_r": m["avg_r"], "profit_factor": m["profit_factor"],
            "fold_positive_frac": bt["fold_positive_frac"], "edge_r": edge,
            "eligible": (m["n"] >= min_trades and m["avg_r"] > 0
                         and bt["fold_positive_frac"] >= MIN_FOLD_POSITIVE and edge > 0)}


def _num(x, fmt: str) -> str:
    return "-" if x is None else fmt.format(x)


def _journal_body(out: dict) -> str:
    """One bullet line per validation, promotion, error, skipped preset and ranked preset, in
    that order (outcomes first, so a clipped report still shows them; plain text that also reads
    as a Markdown list)."""
    lines = []
    for v in out["validations"]:
        kept = f"kept from experiment {v['experiment_id']}: " if v.get("reused") else ""
        lines.append(f"- validation {v['preset']}: {'PASS' if v['passed'] else 'FAIL'} ({kept}n {v['n']},"
                     f" mean R {_num(v['avg_r'], '{:+.3f}')}, PF {_num(v['profit_factor'], '{:.2f}')},"
                     f" deflated Sharpe {_num(v['deflated_sharpe'], '{:.2f}')})"
                     + (f"; failed {', '.join(v['failed'])}" if v["failed"] else ""))
    lines += [f"- promoted {p['preset']} ({p['strategy_id']}), paper from {p['paper_start']}"
              for p in out["promoted"]]
    lines += [f"- {e['step']} {e['preset']} failed: {e['error']}" for e in out["errors"]]
    lines += [f"- not validated {s['preset']}: {s['reason']}" for s in out.get("skipped", [])]
    lines += [f"- {r['preset']} ({r['timeframe']}): n {r['n']}, mean R {r['avg_r']:+.3f},"
              f" PF {r['profit_factor']:.2f}, positive folds {r['fold_positive_frac']:.2f},"
              f" edge vs random {r['edge_r']:+.3f} R: {'eligible' if r['eligible'] else 'not eligible'}"
              for r in out["ranking"]]
    return "\n".join(lines)


def _validation_row(preset: str, spec_id: str, m: dict, eid, reused: bool) -> dict:
    """Sweep validation row from validate experiment metrics (a new or an earlier validation)."""
    checks = m.get("checks") or []
    return {"preset": preset, "spec_id": spec_id, "passed": m.get("passed") is True, "n": m.get("n"),
            "avg_r": m.get("avg_r"), "profit_factor": m.get("profit_factor"),
            "deflated_sharpe": m.get("deflated_sharpe"),
            "failed": [c["name"] for c in checks if not c.get("passed")],
            "experiment_id": eid, "reused": reused}


def sweep(lab: Lab, presets: list[str] | None = None, promote: bool = False) -> dict:
    """Evaluate presets (default: the PRESETS whose timeframe is in data.timeframes) on the
    research window, each with a baseline; rank eligible presets (>= research.min_trades trades,
    mean R > 0, fold_positive_frac >= 0.5, baseline edge_r > 0) first, each group by mean R;
    validate the top TOP_VALIDATE eligible ones; with promote=True put passing ones on paper
    while the roster has room. Failures of single presets
    (no data, budget, roster full) are listed in "errors". Recorded as a "sweep" experiment and a
    "research" journal entry; kv "last_sweep_t" is set.

    The locked window is never re-drawn: an eligible preset whose recorded validation overlaps
    the current validation window (Lab.prior_validation) keeps that verdict ("reused": true, no
    new validation, it may still be promoted when it passed); presets already on the roster or
    retired are not validated; new validations stop at Lab.weekly_validations_left(). Those
    left out are listed in "skipped" ({preset, reason}).

    Returns {"as_of", "presets", "ranking": [{preset, spec_id, timeframe, n, avg_r, profit_factor,
    fold_positive_frac, edge_r, eligible}], "validations": [{preset, spec_id, passed, n, avg_r,
    profit_factor, deflated_sharpe, failed, experiment_id, reused}], "promoted": [...], "errors":
    [{preset, step, error}], "skipped": [{preset, reason}], "experiment_id"}.
    """
    tfs = set(lab.cfg["data"]["timeframes"])
    names = ([n for n, s in PRESETS.items() if s.timeframe in tfs] if presets is None
             else list(dict.fromkeys(presets)))
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise ValueError(f"unknown presets {unknown}; known: {sorted(PRESETS)}")
    min_trades = int(lab.cfg["research"]["min_trades"])
    rows, validations, promoted, errors, skipped = [], [], [], [], []
    for name in names:
        try:
            bt = lab.backtest(PRESETS[name], "research")
            base = lab.baseline(PRESETS[name])
        except (ValueError, BudgetExceeded) as e:
            errors.append({"preset": name, "step": "evaluate", "error": str(e)})
            continue
        rows.append(_row(name, bt, base, min_trades))
    rows.sort(key=lambda r: (not r["eligible"], -r["avg_r"]))       # stable: ties keep preset order

    left, picked = lab.weekly_validations_left(), 0
    for row in [r for r in rows if r["eligible"]]:
        if picked >= TOP_VALIDATE:
            break
        name, spec = row["preset"], PRESETS[row["preset"]]
        on_roster = lab.store.get_strategy(row["spec_id"])
        if on_roster is not None:
            skipped.append({"preset": name, "reason": f"on the roster (status {on_roster['status']})"
                            + (f": {on_roster['status_reason']}" if on_roster["status_reason"] else "")})
            continue
        picked += 1
        try:
            prior = lab.prior_validation(spec)
        except ValueError as e:
            errors.append({"preset": name, "step": "validate", "error": str(e)})
            continue
        if prior is not None:               # the locked window was already looked at: keep that verdict
            validations.append(_validation_row(name, row["spec_id"], prior["metrics"] or {},
                                               prior["id"], reused=True))
            continue
        if left is not None and left <= 0:
            skipped.append({"preset": name, "reason": "research.max_validations_per_week reached"
                                                      " for the last 7 days"})
            continue
        try:
            v = lab.validate(spec)
        except (ValueError, BudgetExceeded) as e:
            errors.append({"preset": name, "step": "validate", "error": str(e)})
            continue
        left = None if left is None else left - 1
        gate = v["gate"]
        validations.append(_validation_row(name, v["spec_id"],
                                           {**v["metrics"], "passed": v["passed"], "checks": gate["checks"],
                                            "deflated_sharpe": gate["deflated_sharpe"]},
                                           v["experiment_id"], reused=False))

    for v in validations if promote else []:
        if not v["passed"]:
            continue
        research_row = next(r for r in rows if r["preset"] == v["preset"])
        rationale = (f"Autopilot sweep: research mean R {research_row['avg_r']:+.3f} over"
                     f" {research_row['n']} trades (edge over random entries"
                     f" {research_row['edge_r']:+.3f} R); validation (experiment {v['experiment_id']})"
                     f" mean R {_num(v['avg_r'], '{:+.3f}')} over {v['n']} trades, deflated Sharpe"
                     f" {_num(v['deflated_sharpe'], '{:.2f}')}.")
        try:
            p = lab.promote(PRESETS[v["preset"]], v["preset"], rationale)
        except ValueError as e:
            errors.append({"preset": v["preset"], "step": "promote", "error": str(e)})
            continue
        promoted.append({"preset": v["preset"], "strategy_id": p["strategy_id"],
                         "paper_start": p["paper_start"]})

    now = int(time.time())
    out = rounded({"as_of": iso_time(now), "presets": names, "ranking": rows,
                   "validations": validations, "promoted": promoted, "errors": errors,
                   "skipped": skipped})
    out["experiment_id"] = lab.store.add_experiment(
        "sweep", None, None, None, None,
        {k: out[k] for k in ("ranking", "validations", "promoted", "errors", "skipped")},
        source="autopilot", notes=f"presets={','.join(names)} promote={promote}", now_t=now)
    n_ok = sum(v["passed"] for v in validations)
    lab.store.add_journal(
        "research", f"Autopilot sweep: {sum(r['eligible'] for r in rows)}/{len(names)} presets eligible,"
                    f" {n_ok}/{len(validations)} passed validation, {len(promoted)} promoted",
        _journal_body(out), now_t=now)
    lab.store.set_kv("last_sweep_t", now)
    return out
