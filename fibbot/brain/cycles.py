"""Brain cycles: research and review (agent loops over the lab tools) and per-signal reviews.

Each journals its outcome. A BrainUnavailable error is journaled and returned as {"ok": False};
other errors propagate to the caller (the scheduler journals them).

Spend guards: review cycles get their own, smaller turn cap and effort (brain.review_max_iterations,
brain.review_effort); every call's token usage is tallied per UTC month in kv "brain_usage", and an
optional brain.max_monthly_output_tokens stops all brain calls for the rest of the month. Signal
reviews count CALLS, not stored reviews, toward brain.max_signal_reviews_per_day (kv
"signal_review_attempts"), and a signal whose review failed MAX_SIGNAL_ATTEMPTS times is not
offered again.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime, timezone

from ..backtest import LITE_KEYS, rounded
from ..graduation import validation_mean_r
from ..metrics import trade_stats
from ..strategy import StrategySpec
from ..types import TF_SECONDS
from .client import INCOMPATIBLE, USAGE_KEYS, BrainUnavailable, _incompatible
from .prompts import (SIGNAL_SCHEMA, SIGNAL_SYSTEM, SYSTEM_PROMPT, WRAP_UP_NOTE, research_prompt,
                      review_prompt, signal_prompt)
from .tools import build_tools, iso, market_context, spec_diff, to_json

log = logging.getLogger(__name__)

DAY = 86400
STATUS_MAX_CHARS = 30000        # cap on the status JSON embedded in a cycle prompt
SIGNAL_MAX_AGE = 7 * DAY        # older unreviewed signals are no longer offered for review
MAX_SIGNAL_ATTEMPTS = 2         # failed review attempts per signal before it is given up
MAX_CALL_ERRORS = 2             # consecutive unexpected review-call errors that stop a batch
ATTEMPTS_KEY = "signal_review_attempts"   # kv: [{"t", "key", "outcome"}] of the last SIGNAL_MAX_AGE
QUOTA_OUTCOMES = ("pending", "ok", "failed")        # attempts that (may have) made a billed call
GIVE_UP_OUTCOMES = ("pending", "failed", "error")   # attempts that count against one signal
USAGE_KEY = "brain_usage"       # kv: {"YYYY-MM": {token sums..., "calls"}} (last USAGE_MONTHS months)
USAGE_MONTHS = 13
REVIEW_MAX_ITERATIONS = 20      # default brain.review_max_iterations (at most brain.max_iterations)
REVIEW_EFFORT = "medium"        # default brain.review_effort
WRAP_UP_TOOLS = ("append_notebook",)
EARLY_STOPS = {"tool_use": "iteration cap", "max_tokens": "max_tokens",
               "model_context_window_exceeded": "context window"}
PROMPTS = {"research": research_prompt, "review": review_prompt}


def research_cycle(lab, brain, stop=None) -> dict:
    """One research cycle: the agent loop with every tool inside lab.begin_cycle("research") ..
    lab.end_cycle(); journal kind "research"; kv last_research_t."""
    return _cycle(lab, brain, "research", stop)


def review_cycle(lab, brain, stop=None) -> dict:
    """One review cycle (Lab gives "review" cycles a smaller budget; no optimization tool; its
    own turn cap and effort, see cycle_limits); journal kind "review"; kv last_review_t."""
    return _cycle(lab, brain, "review", stop)


def cycle_limits(bcfg: dict, kind: str) -> tuple[int, str]:
    """(max_iterations, effort) for a cycle of `kind`: brain.max_iterations / brain.effort for
    research; for review brain.review_max_iterations (default REVIEW_MAX_ITERATIONS; never more
    than brain.max_iterations) and brain.review_effort (default REVIEW_EFFORT)."""
    its, effort = int(bcfg["max_iterations"]), bcfg["effort"]
    if kind == "review":
        cap = bcfg.get("review_max_iterations")
        its = min(int(cap) if cap else REVIEW_MAX_ITERATIONS, its)
        effort = bcfg.get("review_effort") or REVIEW_EFFORT
    return its, effort


# -- spend tally -----------------------------------------------------------------------------------

def _month(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m")


def record_usage(store, usage: dict | None, now: int | None = None, calls: int = 1) -> None:
    """Add one call's (or one cycle's) token usage to kv "brain_usage" under its UTC month."""
    if not isinstance(usage, dict):
        return
    now = int(time.time()) if now is None else int(now)
    tally = store.get_kv(USAGE_KEY) or {}
    month = tally.setdefault(_month(now), {})
    for k in USAGE_KEYS:
        month[k] = int(month.get(k, 0)) + int(usage.get(k) or 0)
    month["calls"] = int(month.get("calls", 0)) + int(calls)
    store.set_kv(USAGE_KEY, {m: tally[m] for m in sorted(tally)[-USAGE_MONTHS:]})


def spend_block(store, bcfg: dict, now: int) -> str | None:
    """Why no brain call may be made now (brain.max_monthly_output_tokens reached), else None."""
    cap = bcfg.get("max_monthly_output_tokens")
    if not cap:
        return None
    month = _month(now)
    used = int(((store.get_kv(USAGE_KEY) or {}).get(month) or {}).get("output_tokens", 0))
    if used < int(cap):
        return None
    return (f"brain.max_monthly_output_tokens reached: {used} output tokens used in {month} "
            f"(cap {int(cap)}); brain calls resume next month")


# -- research / review cycles ------------------------------------------------------------------

def _actions_text(actions: list[dict]) -> str:
    calls = Counter(a["tool"] for a in actions)
    fails = Counter(a["tool"] for a in actions if not a["ok"])
    lines = ["Tools: " + (", ".join(f"{t} x{n}" + (f" ({fails[t]} failed)" if fails[t] else "")
                                    for t, n in calls.items()) or "none")]
    for a in actions:
        if a["ok"] and a["tool"] == "validate_strategy":
            lines.append(f"Validated {a.get('spec_id')}: passed={a.get('passed')}")
        elif a["ok"] and a["tool"] == "promote_to_paper":
            lines.append(f"Promoted {a.get('name')} ({a.get('strategy_id') or a.get('spec_id')})")
        elif a["ok"] and a["tool"] == "retire_strategy":
            lines.append(f"Retired {a.get('strategy_id')}: {a.get('reason')}")
    return "\n".join(lines)


def _previous_entry(store, kind: str) -> str | None:
    """The newest journal entry of `kind` (time, title, body) for the next cycle's prompt."""
    rows = store.journal(limit=1, kind=kind)
    if not rows:
        return None
    j = rows[0]
    return f"{iso(j['t'])} {j['title']}\n{j['body']}"


def _first_cycle(store, status: dict) -> bool:
    """No research/review cycle ran and no experiment was recorded yet."""
    exps = status.get("experiments") if isinstance(status, dict) else None
    total = exps.get("total") if isinstance(exps, dict) else None
    return not total and not any(store.journal(limit=1, kind=k) for k in PROMPTS)


def _cycle(lab, brain, kind: str, stop=None) -> dict:
    """Run one agent cycle of `kind`, journal it and (unless the brain was unavailable) set
    kv last_{kind}_t to the cycle's start time. stop (optional callable, see
    Brain.run_agent) can end the cycle early between turns; such a cycle still counts as run
    (ok=True, journaled "(stopped early: <reason>)"). An anthropic SDK that cannot build the
    tools (missing beta_tool, ...) is BrainUnavailable, so the autopilot can stand in."""
    now, label, actions = int(time.time()), kind.capitalize(), []
    bcfg = lab.cfg["brain"]
    blocked = spend_block(lab.store, bcfg, now)
    if blocked:
        jid = lab.store.add_journal(kind, f"{label} cycle skipped: brain spend cap reached", blocked)
        return {"kind": kind, "ok": False, "error": blocked, "actions": actions, "journal_id": jid}
    its, effort = cycle_limits(bcfg, kind)
    previous = _previous_entry(lab.store, kind)
    lab.begin_cycle(kind)
    try:
        status = lab.status()
        prompt = PROMPTS[kind](to_json(status, STATUS_MAX_CHARS), lab.notebook(), max_turns=its,
                               last_cycle=previous, first_cycle=_first_cycle(lab.store, status))
        try:
            tools = build_tools(lab, kind, actions)
        except INCOMPATIBLE as e:          # e.g. anthropic.beta_tool missing
            raise _incompatible(e) from e
        extra = {} if stop is None else {"stop": stop}
        res = brain.run_agent(SYSTEM_PROMPT, prompt, tools, max_iterations=its, effort=effort,
                              wrap_up=WRAP_UP_NOTE, wrap_up_tools=WRAP_UP_TOOLS, **extra)
    except BrainUnavailable as e:
        jid = lab.store.add_journal(kind, f"{label} cycle failed: brain unavailable",
                                    f"{e}\n\n{_actions_text(actions)}")
        return {"kind": kind, "ok": False, "error": str(e), "actions": actions, "journal_id": jid}
    finally:
        lab.end_cycle()
    record_usage(lab.store, res.get("usage"), now, calls=res.get("turns", 0))
    promoted = [a for a in actions if a["ok"] and a["tool"] == "promote_to_paper"]
    retired = [a for a in actions if a["ok"] and a["tool"] == "retire_strategy"]
    if res["refused"]:
        title = f"{label} cycle refused by the model"
    else:
        title = f"{label} cycle: {len(promoted)} promoted, {len(retired)} retired"
        early = (res.get("stopped_by") or "stop requested" if res["stop_reason"] == "stopped"
                 else EARLY_STOPS.get(res["stop_reason"]))
        if early:
            title += f" (stopped early: {early})"
    u = res["usage"]
    body = "\n\n".join((
        res["text"] or "(no final summary)", _actions_text(actions),
        f"stop_reason={res['stop_reason']} turns={res['turns']} (cap {its}, effort {effort}) "
        f"resumes={res.get('resumes', 0)} wrapped_up={res.get('wrapped_up', False)} "
        f"tokens: in={u['input_tokens']} out={u['output_tokens']} "
        f"cache_read={u['cache_read_input_tokens']} cache_write={u['cache_creation_input_tokens']}"))
    jid = lab.store.add_journal(kind, title, body)
    lab.store.set_kv(f"last_{kind}_t", now)
    return {"kind": kind, "ok": True, **res, "promoted": promoted, "retired": retired,
            "actions": actions, "journal_id": jid}


# -- signal reviews ----------------------------------------------------------------------------

def _div(a, b):
    return a / b if a is not None and b else None


def signal_setup(sig: dict, close: float | None) -> dict:
    """Geometry of a signal: prices, risk, targets in R, leg size, zone, bar counts and (given the
    close of the signal bar) where price stands relative to the entry and the leg."""
    side, m = int(sig["side"]), sig.get("meta") or {}
    entry, stop = sig["entry_price"], sig["stop_price"]
    anchor, extreme, a = m.get("anchor"), m.get("extreme"), m.get("atr")
    risk = (entry - stop) * side
    leg = abs(extreme - anchor) if anchor is not None and extreme is not None else None
    tf_s = TF_SECONDS[sig["tf"]]
    out = {"side": "long" if side > 0 else "short", "entry_mode": m.get("mode"),
           "entry": entry, "stop": stop, "tp1": sig["tp1"], "tp2": sig["tp2"],
           "risk_pct": _div(risk, entry), "tp1_r": _div((sig["tp1"] - entry) * side, risk),
           "tp2_r": _div((sig["tp2"] - entry) * side, risk), "planned_rr": m.get("rr"),
           "anchor": anchor, "extreme": extreme, "zone": m.get("zone"),
           "leg_atr": _div(leg, a), "leg_pct": _div(leg, anchor), "atr_pct": _div(a, entry),
           "bars_since_anchor": _div(sig["t"] - m["anchor_t"], tf_s) if "anchor_t" in m else None,
           "bars_since_extreme": _div(sig["t"] - m["ext_t"], tf_s) if "ext_t" in m else None}
    if close is not None:
        out.update(close=close, close_vs_entry_atr=_div((close - entry) * side, a),
                   close_retracement=_div(extreme - close, extreme - anchor) if leg else None)
    return rounded(out)


def signal_context(sig: dict, bars, spec: StrategySpec) -> dict:
    """{"setup", "market"} for a signal, using only bars that had closed when it fired."""
    if bars:
        past = bars.window(None, int(sig["t"]) + 1)
        close = past.c[-1] if len(past) else None
        market = market_context(past, swing_mode=spec.swing_mode, swing_mult=spec.swing_mult,
                                swing_pct=spec.swing_pct, atr_len=spec.swing_atr_len)
    else:
        close, market = None, {"error": "no cached candles"}
    return {"setup": signal_setup(sig, close), "market": market}


def signal_stats(lab, row: dict, until_t: int, review_record: dict) -> dict:
    """The strategy's track record as known at until_t: closed paper trades that exited by then,
    its validation mean R, plus the brain's own review record (review_record_until)."""
    rows = lab.store.paper_trades(row["strategy_id"], status="closed")
    closed = [lab.store.trade_from_row(r) for r in rows if r["exit_t"] <= until_t]
    ts = trade_stats(closed)
    return rounded({"strategy": {"name": row["name"], "params": spec_diff(row["spec"]),
                                 "paper_start": iso(row["paper_start_t"])},
                    "paper_closed_before_signal": {k: ts[k] for k in LITE_KEYS},
                    "validation_avg_r": validation_mean_r(row.get("validation")),
                    "your_review_record": review_record})


def review_record_until(store, signals: list[dict], closed: list[dict], until_t: int,
                        mode: str) -> dict:
    """The brain's signal-review record as it stood at until_t (Lab.signal_review_stats shape,
    plus "as_of"): closed paper trades of all strategies that exited by until_t, split by the
    decision on their leg if that review was written by until_t (else "unreviewed"). A later
    trade, e.g. the one of the signal under review, never shows up in it."""
    decided = {(s["strategy_id"], s["symbol"], s["leg_id"]): s["decision"] for s in signals
               if s.get("decision") is not None and s.get("reviewed_t") is not None
               and int(s["reviewed_t"]) <= until_t}
    groups: dict[str, list] = {"take": [], "skip": [], "unreviewed": []}
    for r in closed:
        if r.get("exit_t") is None or int(r["exit_t"]) > until_t:
            continue
        key = decided.get((r["strategy_id"], r["symbol"], r["leg_id"])) or "unreviewed"
        groups[key].append(store.trade_from_row(r))
    lite = {}
    for key, trades in groups.items():
        st = trade_stats(trades)
        lite[key] = {k: st[k] for k in LITE_KEYS}
    take, skip = lite["take"], lite["skip"]
    known = [s for s in signals if int(s["t"]) <= until_t]
    return rounded({
        "mode": mode, "as_of": iso(until_t), "reviewed": take["n"] + skip["n"], **lite,
        "skip_minus_take_avg_r": skip["avg_r"] - take["avg_r"] if take["n"] and skip["n"] else None,
        "n_signals": len(known),
        "n_signals_reviewed": sum((s["strategy_id"], s["symbol"], s["leg_id"]) in decided
                                  for s in known)})


def _leg(s: dict) -> tuple:
    return s["strategy_id"], s["symbol"], s["leg_id"]


def _attempt_log(store, now: int) -> list[dict]:
    """Signal-review attempts of the last SIGNAL_MAX_AGE (older ones cannot matter: older signals
    are not offered)."""
    rows = store.get_kv(ATTEMPTS_KEY) or []
    return [a for a in rows if isinstance(a, dict) and int(a.get("t") or 0) >= now - SIGNAL_MAX_AGE]


def review_quota(store, cap: int, now: int, attempts: list[dict] | None = None) -> int:
    """Calls left today: cap minus the review calls of the last 24h (attempts that made or may
    have made a billed call), never less than cap minus the reviews stored in the last 24h."""
    attempts = _attempt_log(store, now) if attempts is None else attempts
    calls = sum(a["t"] >= now - DAY and a.get("outcome") in QUOTA_OUTCOMES for a in attempts)
    return max(0, int(cap) - max(calls, store.count_signal_reviews_since(now - DAY)))


def review_signals(lab, brain, signals: list[dict] | None = None, stop=None) -> list[dict]:
    """Ask Claude take/skip for paper signals and record each review (store.set_signal_review).

    Mode cfg.brain.signal_review: "off" -> []. Signals default to the unreviewed ones of roster
    strategies from the last SIGNAL_MAX_AGE, newest first; given signals (dicts or Signal objects)
    are also limited to roster strategies. Left out: legs that already have a paper trade (a
    review could neither veto them nor be scored without hindsight) and signals whose review
    failed MAX_SIGNAL_ATTEMPTS times. At most the remaining daily quota (review_quota: CALLS of
    the last 24h, failed ones included, count against brain.max_signal_reviews_per_day) is
    attempted, and each attempt is logged before its call. Each prompt is causal: market context,
    paper record and the brain's own review record as of the signal bar.

    Returns one dict per attempt: {"strategy_id", "symbol", "leg_id"} + {"decision",
    "confidence", "reason"} or {"error"}. BrainUnavailable stops the batch ("brain unavailable:
    ..."); any other error of a review call fails that signal only, and MAX_CALL_ERRORS of them in
    a row stop the batch. stop (optional callable): checked before each review call; once it
    returns a reason the reviews gathered so far are returned. One "signal" journal entry per
    non-empty batch; kv last_signal_review_t.
    """
    b = lab.cfg["brain"]
    mode = b["signal_review"]
    if mode == "off":
        return []
    store, now = lab.store, int(time.time())
    blocked = spend_block(store, b, now)
    if blocked:
        log.warning("signal reviews skipped: %s", blocked)
        return []
    roster = {r["strategy_id"]: r for r in store.roster()}
    if signals is None:
        signals = store.signals(unreviewed=True, since_t=now - SIGNAL_MAX_AGE)
    signals = [s.to_dict() if hasattr(s, "to_dict") else s for s in signals]
    attempts = _attempt_log(store, now)
    quota = review_quota(store, b["max_signal_reviews_per_day"], now, attempts)
    failures = Counter(a.get("key") for a in attempts if a.get("outcome") in GIVE_UP_OUTCOMES)
    traded = {_leg(r) for r in store.paper_trades()}
    batch = [s for s in signals if s["strategy_id"] in roster and _leg(s) not in traded
             and failures["|".join(_leg(s))] < MAX_SIGNAL_ATTEMPTS][:quota]
    if not batch:
        return []
    every_signal, closed = store.signals(), store.paper_trades(status="closed")

    def save(entry: dict, outcome: str) -> None:
        entry["outcome"] = outcome
        store.set_kv(ATTEMPTS_KEY, attempts)

    results, call_errors = [], 0
    for sig in batch:
        if stop is not None and stop():
            log.info("signal reviews stopped early: %s", stop())
            break
        key = {"strategy_id": sig["strategy_id"], "symbol": sig["symbol"], "leg_id": sig["leg_id"]}
        entry = {"t": now, "key": "|".join(_leg(sig)), "outcome": "pending"}
        attempts.append(entry)
        store.set_kv(ATTEMPTS_KEY, attempts)       # logged before the call: a crash still counts
        n = failures[entry["key"]] + 1
        retry = " (given up)" if n >= MAX_SIGNAL_ATTEMPTS else " (will be retried)"
        try:
            row = roster[sig["strategy_id"]]
            spec = StrategySpec.from_dict(row["spec"])
            bars = lab.bars(sig["tf"], [sig["symbol"]]).get(sig["symbol"])
            head = {**key, "strategy_name": row["name"], "timeframe": sig["tf"],
                    "signal_bar": iso(sig["t"])}
            record = review_record_until(store, every_signal, closed, int(sig["t"]), mode)
            prompt = signal_prompt(head, signal_context(sig, bars, spec),
                                   signal_stats(lab, row, int(sig["t"]), record))
        except Exception as e:  # a broken signal must not block the others
            log.exception("could not build the review prompt for %s", entry["key"])
            save(entry, "error")
            results.append({**key, "error": f"could not build the review prompt: "
                                            f"{type(e).__name__}: {e}"[:300] + retry})
            continue
        try:
            out = brain.structured(SIGNAL_SYSTEM, prompt, SIGNAL_SCHEMA, effort=b["signal_effort"])
        except BrainUnavailable as e:
            save(entry, "unavailable")
            results.append({**key, "error": f"brain unavailable: {e}"})
            break
        except Exception as e:  # e.g. a 400: counted; repeated ones (systematic) stop the batch
            log.exception("signal review call failed for %s", entry["key"])
            save(entry, "failed")
            results.append({**key, "error": f"review call failed: {type(e).__name__}: {e}"[:300]
                                            + retry})
            call_errors += 1
            if call_errors >= MAX_CALL_ERRORS:
                break
            continue
        call_errors = 0
        record_usage(store, getattr(brain, "last_usage", None), now)
        if out is None or out.get("decision") not in ("take", "skip"):
            save(entry, "failed")
            why = getattr(brain, "last_failure", None) or (
                "refused or malformed reply" if out is None
                else f"decision {out.get('decision')!r} is not take/skip")
            results.append({**key, "error": f"no usable review ({why}){retry}"})
            continue
        review = {"decision": out["decision"],
                  "confidence": min(1.0, max(0.0, float(out.get("confidence") or 0.0))),
                  "reason": str(out.get("reason", ""))[:500]}
        try:
            store.set_signal_review(sig["strategy_id"], sig["symbol"], sig["leg_id"], **review)
        except KeyError:
            save(entry, "failed")
            results.append({**key, "error": "signal not found in the store"})
            continue
        save(entry, "ok")
        results.append({**key, **review})
    done = [r for r in results if "decision" in r]
    n_take = sum(r["decision"] == "take" for r in done)
    title = f"Signal reviews ({mode}): {n_take} take, {len(done) - n_take} skip"
    if len(done) < len(results):
        title += f", {len(results) - len(done)} failed"
    body = "\n".join(f"- {r['symbol']} {r['leg_id']} [{r['strategy_id']}]: " +
                     (f"{r['decision']} ({r['confidence']:.2f}) {r['reason']}" if "decision" in r
                      else f"ERROR {r['error']}") for r in results)
    store.add_journal("signal", title, body)
    store.set_kv("last_signal_review_t", now)
    return results
