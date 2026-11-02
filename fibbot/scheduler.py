"""Run loop (see DESIGN.md, "fibbot/scheduler.py and fibbot/cli.py").

run_once() is one pass: universe -> candles -> paper tick -> brain signal reviews -> graduation
(journal status changes, auto-retire on a deep paper drawdown) -> brain review cycle -> brain
research cycle -> autopilot sweep (stands in for research when the brain is unavailable and the
roster is empty) -> REPORT.md. Every step is isolated: an exception is logged, journaled (kind
"system") and the next step still runs. run_forever() repeats the pass every
scheduler.tick_every_seconds, tick_offset_seconds after each UTC multiple of that period, until
SIGINT/SIGTERM. Paper trading only: nothing here can place a real order.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager

from . import autopilot
from .config import home_path
from .lab import Lab, iso_time
from .report import write_report
from .types import TF_SECONDS

log = logging.getLogger(__name__)

LOG_FILE = "fibbot.log"
HOUR = 3600
AUTO_RETIRE_FACTOR = 1.5          # retire when paper drawdown > this x graduation.max_drawdown
GRADUATION_KEY = "graduation_status:{}"   # kv: {"status", "paper_start_t", "t"} of the last check
ATTEMPT_KEY = "last_{}_attempt_t"         # kv: start of the latest scheduled research/review cycle
CYCLE_RETRY_HOURS = 6             # a cycle that failed or crashed is retried after this long
TRACEBACK_CHARS = 4000
STEPS = ("universe", "data", "paper", "signals", "graduation", "review", "research", "autopilot",
         "report")
_handlers: list[logging.Handler] = []


def setup_logging(cfg: dict, stream=None) -> list[logging.Handler]:
    """Send the "fibbot" loggers' INFO+ records to `stream` (default stdout) and append them to
    home/fibbot.log, with UTC timestamps. Replaces the handlers of an earlier call; returns the
    new handlers."""
    logger = logging.getLogger("fibbot")
    for h in _handlers:
        logger.removeHandler(h)
        h.close()
    _handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    fmt.converter = time.gmtime
    for h in (logging.StreamHandler(sys.stdout if stream is None else stream),
              logging.FileHandler(home_path(cfg, LOG_FILE), encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
        _handlers.append(h)
    logger.setLevel(logging.INFO)
    return list(_handlers)


def make_brain(cfg: dict):
    """A fibbot.brain.client.Brain (imported lazily: the scheduler runs without anthropic)."""
    from .brain.client import Brain
    return Brain(cfg)


def due(last_t: float | None, every_hours: float, now: float) -> bool:
    """True when nothing ran yet (last_t None) or last_t is at least every_hours old."""
    return last_t is None or now - float(last_t) >= float(every_hours) * HOUR


def retrying(store, kind: str, now: float) -> bool:
    """A scheduled `kind` cycle started after the last completed one (kv last_{kind}_t), i.e. it
    failed or crashed, less than CYCLE_RETRY_HOURS ago: wait instead of paying for it again."""
    attempt, done = store.get_kv(ATTEMPT_KEY.format(kind)), store.get_kv(f"last_{kind}_t")
    return (attempt is not None and (done is None or attempt > done)
            and not due(attempt, CYCLE_RETRY_HOURS, now))


def research_due(store, cfg: dict, now: float, roster_empty: bool) -> bool:
    """kv last_research_t is unset or older than brain.research_every_hours, or the roster is
    empty and it is older than brain.review_every_hours: an empty roster gets research in the
    review slot (there is nothing to review) instead of a costly cycle on every pass."""
    b, last = cfg["brain"], store.get_kv("last_research_t")
    return due(last, b["research_every_hours"], now) or (
        roster_empty and due(last, b["review_every_hours"], now))


def review_due(store, cfg: dict, now: float, roster: list[dict]) -> bool:
    """kv last_review_t (or, before the first review, the roster's earliest paper_start_t) is
    older than brain.review_every_hours. Never due with an empty roster."""
    if not roster:
        return False
    last = store.get_kv("last_review_t")
    ref = last if last is not None else min(int(r["paper_start_t"]) for r in roster)
    return due(ref, cfg["brain"]["review_every_hours"], now)


def autopilot_due(store, cfg: dict, now: float) -> bool:
    """At most one autopilot sweep per brain.research_every_hours (kv last_autopilot_t)."""
    return due(store.get_kv("last_autopilot_t"), cfg["brain"]["research_every_hours"], now)


def next_run_t(now: float, period: int, offset: int) -> int:
    """First time after `now` that lies `offset` seconds past a UTC multiple of `period` (> 0)."""
    return int((now - offset) // period + 1) * period + offset


class _Skip(Exception):
    """A step that had nothing to do (the message says why)."""


class _Failed(Exception):
    """A step that failed and has already been journaled elsewhere."""


class _Pass:
    """One run_once pass: shared state plus one method per step (STEPS)."""

    def __init__(self, cfg: dict, lab, brain, now_t: int | None, use_brain: bool):
        self.cfg, self.lab, self.store, self.now_t = cfg, lab, lab.store, now_t
        self.now = int(time.time()) if now_t is None else int(now_t)
        self.universe: list[str] | None = None
        self.brain, self.brain_reason = None, "disabled by --no-brain"
        if use_brain:
            try:
                brain = make_brain(cfg) if brain is None else brain
                ok, self.brain_reason = brain.available()
                self.brain = brain if ok else None
            except Exception as e:  # a broken brain setup must not stop paper trading
                self.brain_reason = f"{type(e).__name__}: {e}"

    def roster(self) -> list[dict]:
        return self.store.roster("paper")

    def need_brain(self) -> None:
        if self.brain is None:
            raise _Skip(f"brain unavailable ({self.brain_reason})")

    def lose_brain(self, why: str) -> None:
        """The brain failed mid-pass: later steps treat it as unavailable."""
        log.warning("brain became unavailable: %s", why)
        self.brain, self.brain_reason = None, why

    def journal_failure(self, step: str, exc: BaseException) -> None:
        log.error("%s step failed: %s: %s", step, type(exc).__name__, exc, exc_info=exc)
        body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            self.store.add_journal("system", f"Scheduler step {step} failed: {type(exc).__name__}:"
                                   f" {exc}"[:300], body[-TRACEBACK_CHARS:], now_t=self.now_t)
        except Exception as e:  # e.g. the database itself is broken; the log still has it
            log.error("could not journal the %s failure: %s", step, e)

    # -- steps ---------------------------------------------------------------------------------

    def universe_step(self) -> dict:
        """data.universe.get_universe via Lab (re-ranks when the cached ranking is stale)."""
        self.universe = self.lab.universe()
        return {"symbols": len(self.universe)}

    def data_step(self) -> dict:
        """Incremental candle update: universe + symbols held by paper strategies, for
        data.timeframes plus every roster timeframe. Per-symbol failures are counted, not raised."""
        roster = self.roster()
        ids = {r["strategy_id"] for r in roster}
        held = sorted({r["symbol"] for r in self.store.paper_trades(status="open")
                       if r["strategy_id"] in ids})
        symbols = list(dict.fromkeys([*(self.universe or []), *held]))
        tfs = [tf for tf in dict.fromkeys([*self.cfg["data"]["timeframes"],
                                           *(r["spec"].get("timeframe") for r in roster)])
               if tf in TF_SECONDS]
        counts = self.lab.candles.update_all(symbols, tfs)
        failed = getattr(self.lab.candles, "last_errors", {})
        return {"symbols": len(symbols), "timeframes": tfs, "new_bars": sum(counts.values()),
                "failed": len(failed)}

    def paper_step(self) -> dict:
        """PaperTrader.tick on the data just updated (update_data=False)."""
        if not self.roster():
            raise _Skip("no strategies on paper")
        out = self.lab.paper().tick(update_data=False, now_t=self.now_t, universe=self.universe)
        entries = exits = 0
        for sid, res in out["strategies"].items():
            for d in res["new_entries"]:
                log.info("paper %s: entry %s %s %s at %.6g (stop %.6g)", sid, d["symbol"],
                         "long" if d["side"] > 0 else "short", d["leg_id"], d["entry_price"],
                         d["stop_price"])
            for d in res["new_exits"]:
                log.info("paper %s: exit %s %s %s, %+.2f R", sid, d["symbol"], d["leg_id"],
                         d["exit_reason"], d["r_multiple"])
            entries, exits = entries + len(res["new_entries"]), exits + len(res["new_exits"])
        return {"strategies": len(out["strategies"]), "new_signals": len(out["new_signals"]),
                "new_entries": entries, "new_exits": exits, "errors": len(out["errors"])}

    def signals_step(self) -> dict:
        """brain.cycles.review_signals over the unreviewed recent signals (quota-limited)."""
        if self.cfg["brain"]["signal_review"] == "off":
            raise _Skip("brain.signal_review is off")
        self.need_brain()
        from .brain import cycles
        results = cycles.review_signals(self.lab, self.brain)
        done = [r for r in results if "decision" in r]
        lost = next((r["error"] for r in results if r.get("error", "").startswith("brain unavailable")),
                    None)
        if lost:
            self.lose_brain(lost)
        return {"reviewed": len(done), "take": sum(r["decision"] == "take" for r in done),
                "skip": sum(r["decision"] == "skip" for r in done), "failed": len(results) - len(done)}

    def graduation_step(self) -> dict:
        """graduation.evaluate per paper strategy; journal (kind "graduation") each status change
        and retire a strategy whose paper drawdown exceeds AUTO_RETIRE_FACTOR x the limit. One
        strategy's failure is journaled and does not stop the others."""
        limit = AUTO_RETIRE_FACTOR * float(self.cfg["graduation"]["max_drawdown"])
        statuses, retired, failed = {}, [], 0
        for row in self.roster():
            sid = row["strategy_id"]
            try:
                g = self.lab.graduation(sid, now_t=self.now_t)
                statuses[sid] = g["status"]
                self._journal_change(row, g)
                dd = next((c["value"] for c in g["checks"] if c["name"] == "max_drawdown"), None)
                if dd is not None and dd > limit:
                    self.lab.retire(sid, f"auto-retired by the scheduler: paper max drawdown {dd:.1%}"
                                         f" > {AUTO_RETIRE_FACTOR:g} x graduation.max_drawdown"
                                         f" ({limit:.1%})")
                    log.warning("auto-retired %s (%s): drawdown %.1f%%", row["name"], sid, 100 * dd)
                    retired.append(sid)
            except Exception as e:
                self.journal_failure(f"graduation {sid}", e)
                failed += 1
        return {"statuses": statuses, "retired": retired, "failed": failed}

    def _journal_change(self, row: dict, g: dict) -> None:
        """Journal the graduation status when it differs from the last one recorded for this
        paper run (a re-activated strategy starts afresh)."""
        key, start = GRADUATION_KEY.format(row["strategy_id"]), int(row["paper_start_t"])
        prev = self.store.get_kv(key)
        before = prev["status"] if isinstance(prev, dict) and prev.get("paper_start_t") == start else None
        if g["status"] == before:
            return
        change = f"{before} -> {g['status']}" if before else f"is {g['status']}"
        checks = "\n".join(f"- [{'x' if c['passed'] else ' '}] {c['name']}: {c['value']}"
                           f" (threshold {c['threshold']})" for c in g["checks"])
        self.store.add_journal("graduation", f"{row['name']} ({row['strategy_id']}) {change}",
                               f"{g['summary']}\n\n{checks}", now_t=self.now_t)
        self.store.set_kv(key, {"status": g["status"], "paper_start_t": start, "t": self.now})
        log.info("graduation: %s %s", row["name"], change)

    def review_step(self) -> dict:
        """brain.cycles.review_cycle when review_due()."""
        self.need_brain()
        if not review_due(self.store, self.cfg, self.now, self.roster()):
            raise _Skip("not due")
        return self._run_cycle("review")

    def research_step(self) -> dict:
        """brain.cycles.research_cycle when research_due()."""
        self.need_brain()
        if not research_due(self.store, self.cfg, self.now, not self.roster()):
            raise _Skip("not due")
        return self._run_cycle("research")

    def _run_cycle(self, kind: str) -> dict:
        """Run a due brain cycle unless retrying() a failed one; record the attempt first."""
        if retrying(self.store, kind, self.now):
            raise _Skip(f"the last {kind} cycle did not finish; retrying {CYCLE_RETRY_HOURS}h after it started")
        from .brain import cycles
        self.store.set_kv(ATTEMPT_KEY.format(kind), self.now)
        return self._cycle_result(getattr(cycles, f"{kind}_cycle")(self.lab, self.brain))

    def _cycle_result(self, res: dict) -> dict:
        if not res["ok"]:
            self.lose_brain(res["error"])       # the cycle journaled the failure itself
            raise _Failed(f"brain cycle failed: {res['error']}")
        log.info("%s cycle: %d turns, stop %s, %d promoted, %d retired", res["kind"], res["turns"],
                 res["stop_reason"], len(res["promoted"]), len(res["retired"]))
        return {"turns": res["turns"], "stop_reason": res["stop_reason"], "refused": res["refused"],
                "promoted": len(res["promoted"]), "retired": len(res["retired"]),
                "journal_id": res["journal_id"]}

    def autopilot_step(self) -> dict:
        """autopilot.sweep(promote=True) when the brain is unavailable, the roster is empty and
        no sweep that ranked any preset ran within brain.research_every_hours."""
        if self.brain is not None:
            raise _Skip("the brain does the research")
        if self.roster():
            raise _Skip("the roster is not empty")
        if not autopilot_due(self.store, self.cfg, self.now):
            raise _Skip("swept within brain.research_every_hours")
        out = autopilot.sweep(self.lab, promote=True)
        if out["ranking"]:      # nothing ranked (e.g. no candles yet): try again next pass
            self.store.set_kv("last_autopilot_t", self.now)
        return {"ranked": len(out["ranking"]), "validated": len(out["validations"]),
                "promoted": [p["strategy_id"] for p in out["promoted"]], "errors": len(out["errors"])}

    def report_step(self) -> dict:
        return {"path": str(write_report(self.lab, now_t=self.now_t))}


def run_once(cfg: dict, now_t: int | None = None, use_brain: bool = True, lab=None, brain=None,
             stop=None) -> dict:
    """One scheduler pass (module docstring). lab defaults to a Lab(cfg) that is closed again;
    brain to make_brain(cfg) when use_brain. stop: optional callable checked before each step;
    once it returns True the remaining steps are skipped.

    Returns {"t", "brain": {"available", "reason"} (at the start of the pass), "steps": [{"step",
    "status": "ok"|"skipped"|"failed", "seconds", "detail": dict (ok) | reason | error}],
    "stopped": bool}.
    """
    own = lab is None
    lab = Lab(cfg) if own else lab
    try:
        p = _Pass(cfg, lab, brain, now_t, use_brain)
        out = {"t": p.now, "brain": {"available": p.brain is not None, "reason": p.brain_reason},
               "steps": [], "stopped": False}
        log.info("run started at %s; brain %s (%s)", iso_time(p.now),
                 "available" if p.brain is not None else "unavailable", p.brain_reason)
        for name in STEPS:
            if stop is not None and stop():
                out["stopped"] = True
                log.warning("stop requested: skipping the remaining steps from %s", name)
                break
            t0 = time.monotonic()
            try:
                status, detail = "ok", getattr(p, f"{name}_step")()
            except _Skip as e:
                status, detail = "skipped", str(e)
            except _Failed as e:
                status, detail = "failed", str(e)
            except Exception as e:
                p.journal_failure(name, e)
                status, detail = "failed", f"{type(e).__name__}: {e}"
            seconds = round(time.monotonic() - t0, 3)
            log.log(logging.WARNING if status == "failed" else logging.INFO, "step %s %s in %.1fs: %s",
                    name, status, seconds, detail)
            out["steps"].append({"step": name, "status": status, "seconds": seconds, "detail": detail})
        n = Counter(r["status"] for r in out["steps"])
        log.info("run finished: %d steps ok, %d skipped, %d failed", n["ok"], n["skipped"], n["failed"])
        return out
    finally:
        if own:
            lab.store.close()


@contextmanager
def stop_on_signals(stop: threading.Event):
    """While active, SIGINT/SIGTERM set `stop` (the loop ends after the current step); a second
    signal raises KeyboardInterrupt. Only the main thread can install handlers; elsewhere this
    does nothing. Previous handlers are restored on exit."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        log.warning("%s received: stopping after the current step (send it again to abort now)",
                    signal.Signals(signum).name)

    old = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def sleep_until(wake_t: float, stop: threading.Event) -> None:
    """Sleep until wake_t (time.time() clock) or until stop is set, in steps of at most 1 s."""
    while not stop.is_set():
        left = wake_t - time.time()
        if left <= 0:
            return
        stop.wait(min(left, 1.0))


def run_forever(cfg: dict, use_brain: bool = True) -> None:
    """Log to stdout and home/fibbot.log, run a pass now, then one pass at every
    next_run_t(scheduler.tick_every_seconds, tick_offset_seconds) until SIGINT/SIGTERM. One Lab
    (and brain) serves every pass."""
    setup_logging(cfg)
    period = int(cfg["scheduler"]["tick_every_seconds"])
    offset = int(cfg["scheduler"]["tick_offset_seconds"])
    if period <= 0:
        raise ValueError(f"scheduler.tick_every_seconds must be > 0, got {period}")
    stop = threading.Event()
    lab = Lab(cfg)
    brain = make_brain(cfg) if use_brain else None
    log.info("scheduler started (home %s): a pass every %ds at +%ds, paper trading only",
             cfg["home"], period, offset)
    try:
        with stop_on_signals(stop):
            while not stop.is_set():
                try:
                    run_once(cfg, use_brain=use_brain, lab=lab, brain=brain, stop=stop.is_set)
                except Exception:   # steps are isolated already; keep the loop alive regardless
                    log.exception("scheduler pass failed")
                if stop.is_set():
                    break
                wake = next_run_t(time.time(), period, offset)
                log.info("next pass at %s", iso_time(wake))
                sleep_until(wake, stop)
    finally:
        lab.store.close()
        log.info("scheduler stopped")
