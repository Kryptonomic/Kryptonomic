"""Brain cycles: research and review (agent loops over the lab tools) and per-signal reviews.

Each journals its outcome. A BrainUnavailable error is journaled and returned as {"ok": False};
other errors propagate to the caller (the scheduler journals them).
"""
from __future__ import annotations

import time
from collections import Counter

from ..backtest import LITE_KEYS, rounded
from ..graduation import validation_mean_r
from ..metrics import trade_stats
from ..strategy import StrategySpec
from ..types import TF_SECONDS
from .client import BrainUnavailable
from .prompts import (SIGNAL_SCHEMA, SIGNAL_SYSTEM, SYSTEM_PROMPT, research_prompt, review_prompt,
                      signal_prompt)
from .tools import build_tools, iso, market_context, spec_diff, to_json

DAY = 86400
STATUS_MAX_CHARS = 30000        # cap on the status JSON embedded in a cycle prompt
SIGNAL_MAX_AGE = 7 * DAY        # older unreviewed signals are no longer offered for review
PROMPTS = {"research": research_prompt, "review": review_prompt}


def research_cycle(lab, brain) -> dict:
    """One research cycle: the agent loop with every tool inside lab.begin_cycle("research") ..
    lab.end_cycle(); journal kind "research"; kv last_research_t."""
    return _cycle(lab, brain, "research")


def review_cycle(lab, brain) -> dict:
    """One review cycle (Lab gives "review" cycles a smaller budget; no optimization tool);
    journal kind "review"; kv last_review_t."""
    return _cycle(lab, brain, "review")


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


def _cycle(lab, brain, kind: str) -> dict:
    """Run one agent cycle of `kind`, journal it and (unless the brain was unavailable) set
    kv last_{kind}_t to the cycle's start time."""
    now, label, actions = int(time.time()), kind.capitalize(), []
    lab.begin_cycle(kind)
    try:
        prompt = PROMPTS[kind](to_json(lab.status(), STATUS_MAX_CHARS), lab.notebook())
        res = brain.run_agent(SYSTEM_PROMPT, prompt, build_tools(lab, kind, actions))
    except BrainUnavailable as e:
        jid = lab.store.add_journal(kind, f"{label} cycle failed: brain unavailable",
                                    f"{e}\n\n{_actions_text(actions)}")
        return {"kind": kind, "ok": False, "error": str(e), "actions": actions, "journal_id": jid}
    finally:
        lab.end_cycle()
    promoted = [a for a in actions if a["ok"] and a["tool"] == "promote_to_paper"]
    retired = [a for a in actions if a["ok"] and a["tool"] == "retire_strategy"]
    if res["refused"]:
        title = f"{label} cycle refused by the model"
    else:
        title = f"{label} cycle: {len(promoted)} promoted, {len(retired)} retired"
        early = {"tool_use": "iteration cap", "max_tokens": "max_tokens"}.get(res["stop_reason"])
        if early:
            title += f" (stopped early: {early})"
    u = res["usage"]
    body = "\n\n".join((
        res["text"] or "(no final summary)", _actions_text(actions),
        f"stop_reason={res['stop_reason']} turns={res['turns']} tokens: in={u['input_tokens']} "
        f"out={u['output_tokens']} cache_read={u['cache_read_input_tokens']} "
        f"cache_write={u['cache_creation_input_tokens']}"))
    jid = lab.store.add_journal(kind, title, body)
    lab.store.set_kv(f"last_{kind}_t", now)
    return {"kind": kind, "ok": True, **res, "promoted": promoted, "retired": retired,
            "actions": actions, "journal_id": jid}


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
    its validation mean R, plus the brain's own review record (get_signal_review_stats)."""
    rows = lab.store.paper_trades(row["strategy_id"], status="closed")
    closed = [lab.store.trade_from_row(r) for r in rows if r["exit_t"] <= until_t]
    ts = trade_stats(closed)
    return rounded({"strategy": {"name": row["name"], "params": spec_diff(row["spec"]),
                                 "paper_start": iso(row["paper_start_t"])},
                    "paper_closed_before_signal": {k: ts[k] for k in LITE_KEYS},
                    "validation_avg_r": validation_mean_r(row.get("validation")),
                    "your_review_record": review_record})


def review_signals(lab, brain, signals: list[dict] | None = None) -> list[dict]:
    """Ask Claude take/skip for paper signals and record each review (store.set_signal_review).

    Mode cfg.brain.signal_review: "off" -> []. Signals default to the unreviewed ones of roster
    strategies from the last SIGNAL_MAX_AGE, newest first; given signals (dicts or Signal objects)
    are also limited to roster strategies. At most the remaining daily quota
    (max_signal_reviews_per_day minus reviews in the last 24h) is attempted. Returns one dict per
    attempt: {"strategy_id", "symbol", "leg_id"} + {"decision", "confidence", "reason"} or
    {"error"}. One "signal" journal entry per non-empty batch; kv last_signal_review_t.
    """
    b = lab.cfg["brain"]
    mode = b["signal_review"]
    if mode == "off":
        return []
    store, now = lab.store, int(time.time())
    roster = {r["strategy_id"]: r for r in store.roster()}
    if signals is None:
        signals = store.signals(unreviewed=True, since_t=now - SIGNAL_MAX_AGE)
    signals = [s.to_dict() if hasattr(s, "to_dict") else s for s in signals]
    quota = max(0, int(b["max_signal_reviews_per_day"]) - store.count_signal_reviews_since(now - DAY))
    batch = [s for s in signals if s["strategy_id"] in roster][:quota]
    if not batch:
        return []
    record = lab.signal_review_stats()
    results = []
    for sig in batch:
        key = {"strategy_id": sig["strategy_id"], "symbol": sig["symbol"], "leg_id": sig["leg_id"]}
        row = roster[sig["strategy_id"]]
        spec = StrategySpec.from_dict(row["spec"])
        bars = lab.bars(sig["tf"], [sig["symbol"]]).get(sig["symbol"])
        head = {**key, "strategy_name": row["name"], "timeframe": sig["tf"], "signal_bar": iso(sig["t"])}
        prompt = signal_prompt(head, signal_context(sig, bars, spec),
                               signal_stats(lab, row, int(sig["t"]), record))
        try:
            out = brain.structured(SIGNAL_SYSTEM, prompt, SIGNAL_SCHEMA, effort=b["signal_effort"])
        except BrainUnavailable as e:
            results.append({**key, "error": f"brain unavailable: {e}"})
            break
        if out is None or out.get("decision") not in ("take", "skip"):
            results.append({**key, "error": "no usable review (refused or malformed reply)"})
            continue
        review = {"decision": out["decision"],
                  "confidence": min(1.0, max(0.0, float(out.get("confidence") or 0.0))),
                  "reason": str(out.get("reason", ""))[:500]}
        try:
            store.set_signal_review(sig["strategy_id"], sig["symbol"], sig["leg_id"], **review)
        except KeyError:
            results.append({**key, "error": "signal not found in the store"})
            continue
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
