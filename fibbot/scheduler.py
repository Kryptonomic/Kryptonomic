"""Run loop (see DESIGN.md, "fibbot/scheduler.py and fibbot/cli.py").

run_once() is one pass: universe -> candles -> paper tick -> brain signal reviews -> graduation
(journal status changes, auto-retire on a deep paper drawdown) -> brain review cycle -> brain
research cycle -> autopilot sweep (stands in for research when the brain is unavailable and the
roster is empty) -> REPORT.md. Every step is isolated: an exception is logged, journaled (kind
"system"; an identical failure of the same step at most once per FAILURE_REPEAT_HOURS) and the
next step still runs. run_forever() repeats the pass every scheduler.tick_every_seconds,
tick_offset_seconds after each UTC multiple of that period, until SIGINT/SIGTERM. Paper trading
only: nothing here can place a real order.

Candle stalls: one symbol failing to download is only a warning, but when every requested
update failed, an open paper position's candles, or a whole timeframe, are more than STALE_BARS
bars old, the data step fails (so `run --once` exits 1). The stall is journaled once when it
starts and once when it clears (kv DATA_STALL_KEY), and the review cycle waits for fresh data.

Brain cycles that accept a `stop` callable get one: it returns a reason once the cycle has run
brain.max_cycle_minutes or a stop signal arrived, so the cycle can wrap up between turns instead
of being killed by a job timeout or `docker stop`.
"""
from __future__ import annotations

import inspect
import logging
import logging.handlers
import signal
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager

from . import autopilot
from .config import home_path, validate_config
from .lab import Lab, iso_time
from .report import write_report
from .types import TF_SECONDS

log = logging.getLogger(__name__)

LOG_FILE = "fibbot.log"
LOG_MAX_BYTES = 5_000_000         # fibbot.log is rotated at this size ...
LOG_BACKUPS = 3                   # ... keeping fibbot.log.1 .. .3
HOUR = 3600
AUTO_RETIRE_FACTOR = 1.5          # retire when paper drawdown > this x graduation.max_drawdown
GRADUATION_KEY = "graduation_status:{}"   # kv: {"status", "paper_start_t", "t"} of the last check
ATTEMPT_KEY = "last_{}_attempt_t"         # kv: start of the latest scheduled research/review cycle
CYCLE_RETRY_HOURS = 6             # a cycle that failed or crashed is retried after this long
TRACEBACK_CHARS = 4000
FAILURE_KEY = "last_failure:{}"   # kv: {"title", "t", "repeats"} of a step's newest journaled failure
FAILURE_REPEAT_HOURS = 24         # the same failure of the same step is journaled at most this often
STALE_BARS = 3                    # candles whose newest bar ended more than this many bars ago are stale
DATA_STALL_KEY = "data_stall"     # kv: {"since", "reason"} while the candle data is stalled, else null
COMPACT_KEY = "last_compact_t"    # kv: last Store.compact() run (report step, about once a day)
COMPACT_EVERY_HOURS = 24
STEPS = ("universe", "data", "paper", "signals", "graduation", "review", "research", "autopilot",
         "report")
_handlers: list[logging.Handler] = []


def setup_logging(cfg: dict, stream=None) -> list[logging.Handler]:
    """Send the "fibbot" loggers' INFO+ records to `stream` (default stdout) and append them to
    home/fibbot.log (rotated at LOG_MAX_BYTES, LOG_BACKUPS old files kept), with UTC timestamps.
    Replaces the handlers of an earlier call; returns the new handlers."""
    logger = logging.getLogger("fibbot")
    for h in _handlers:
        logger.removeHandler(h)
        h.close()
    _handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    fmt.converter = time.gmtime
    for h in (logging.StreamHandler(sys.stdout if stream is None else stream),
              logging.handlers.RotatingFileHandler(home_path(cfg, LOG_FILE), maxBytes=LOG_MAX_BYTES,
                                                   backupCount=LOG_BACKUPS, encoding="utf-8")):
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


def accepts(fn, name: str) -> bool:
    """fn declares a parameter called `name` (a bare **kwargs does not count)."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def cycle_stop(cfg: dict, stop=None, clock=None):
    """The `stop` callable handed to a brain cycle: "" while it may go on, else why it must wrap
    up (a stop signal via `stop`, or brain.max_cycle_minutes since this call; 0/None = no limit).
    clock defaults to time.monotonic."""
    clock = clock or time.monotonic
    minutes = cfg["brain"].get("max_cycle_minutes")
    deadline = clock() + float(minutes) * 60 if minutes else None

    def should_stop() -> str:
        if stop is not None and stop():
            return "the bot is shutting down"
        if deadline is not None and clock() >= deadline:
            return f"the cycle has run for brain.max_cycle_minutes ({minutes:g} minutes)"
        return ""
    return should_stop


def newest_bar_end(candles, symbol: str, tf: str) -> int | None:
    """End time of the newest cached (symbol, tf) bar; None without bars (or a cache that
    cannot say)."""
    try:
        t = candles.get(symbol, tf).t
        return int(t[-1]) + TF_SECONDS[tf] if len(t) else None
    except Exception:
        return None


class _Skip(Exception):
    """A step that had nothing to do (the message says why)."""


class _Failed(Exception):
    """A step that failed and has already been journaled elsewhere."""


class _Pass:
    """One run_once pass: shared state plus one method per step (STEPS)."""

    def __init__(self, cfg: dict, lab, brain, now_t: int | None, use_brain: bool, stop=None):
        self.cfg, self.lab, self.store, self.now_t, self.stop = cfg, lab, lab.store, now_t, stop
        self.now = int(time.time()) if now_t is None else int(now_t)
        self.universe: list[str] | None = None
        self.data_stalled: str | None = None     # why the candle data is stalled (data_step)
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
        """Log the failure; journal it (kind "system", with the traceback) unless the same step
        failed with the same title less than FAILURE_REPEAT_HOURS ago (then only count it), so a
        persistent failure adds one entry a day, not one an hour."""
        log.error("%s step failed: %s: %s", step, type(exc).__name__, exc, exc_info=exc)
        title = f"Scheduler step {step} failed: {type(exc).__name__}: {exc}"[:300]
        key = FAILURE_KEY.format(step)
        try:
            prev = self.store.get_kv(key)
            same = isinstance(prev, dict) and prev.get("title") == title
            if same and not due(prev.get("t"), FAILURE_REPEAT_HOURS, self.now):
                self.store.set_kv(key, {**prev, "repeats": int(prev.get("repeats") or 0) + 1})
                log.info("same %s failure as the one journaled at %s: not journaled again", step,
                         iso_time(int(prev["t"])))
                return
            body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-TRACEBACK_CHARS:]
            if same and prev.get("repeats"):
                body = (f"The same failure also happened {prev['repeats']} more times since"
                        f" {iso_time(int(prev['t']))} (not journaled separately).\n\n{body}")
            self.store.add_journal("system", title, body, now_t=self.now_t)
            self.store.set_kv(key, {"title": title, "t": self.now, "repeats": 0})
        except Exception as e:  # e.g. the database itself is broken; the log still has it
            log.error("could not journal the %s failure: %s", step, e)

    # -- steps ---------------------------------------------------------------------------------

    def universe_step(self) -> dict:
        """data.universe.get_universe via Lab (re-ranks when the cached ranking is stale)."""
        self.universe = self.lab.universe()
        return {"symbols": len(self.universe)}

    def data_step(self) -> dict:
        """Incremental candle update: universe + symbols held by paper strategies, for
        data.timeframes plus every roster timeframe. Per-symbol failures are counted, not raised;
        a stall (_data_problems) fails the step (_data_stall)."""
        roster = self.roster()
        tf_of = {r["strategy_id"]: r["spec"].get("timeframe") for r in roster}
        held = sorted({(r["symbol"], tf_of[r["strategy_id"]])
                       for r in self.store.paper_trades(status="open") if r["strategy_id"] in tf_of})
        symbols = list(dict.fromkeys([*(self.universe or []), *(s for s, _ in held)]))
        tfs = [tf for tf in dict.fromkeys([*self.cfg["data"]["timeframes"], *tf_of.values()])
               if tf in TF_SECONDS]
        counts = self.lab.candles.update_all(symbols, tfs)
        failed = dict(getattr(self.lab.candles, "last_errors", None) or {})
        detail = {"symbols": len(symbols), "timeframes": tfs, "new_bars": sum(counts.values()),
                  "failed": len(failed)}
        self._data_stall(self._data_problems(symbols, tfs, held, failed), failed, detail)
        return detail

    def _data_problems(self, symbols: list[str], tfs: list[str], held: list[tuple],
                       failed: dict) -> list[str]:
        """Why the candle data is stalled: every requested update failed, an open paper
        position's (symbol, timeframe) has no bar that ended within STALE_BARS bars, or no symbol
        has one on a timeframe. One failing symbol among many is not a stall."""
        problems, n_pairs = [], len(symbols) * len(tfs)
        if n_pairs and len(failed) >= n_pairs:
            problems.append(f"all {n_pairs} candle updates failed")
        ends = {(s, tf): newest_bar_end(self.lab.candles, s, tf) for s in symbols for tf in tfs}
        stale = [f"{sym} {tf} ({'no bars' if ends[(sym, tf)] is None else 'ends ' + iso_time(ends[(sym, tf)])})"
                 for sym, tf in held if (sym, tf) in ends
                 and (ends[(sym, tf)] is None or self.now - ends[(sym, tf)] > STALE_BARS * TF_SECONDS[tf])]
        if stale:
            more = f" and {len(stale) - 3} more" if len(stale) > 3 else ""
            problems.append(f"stale candles for open paper positions: {', '.join(stale[:3])}{more}")
        for tf in tfs:
            known = [e for (_, t), e in ends.items() if t == tf and e is not None]
            if known and self.now - max(known) > STALE_BARS * TF_SECONDS[tf]:
                problems.append(f"no {tf} bar of any symbol after {iso_time(max(known))}")
        return problems

    def _data_stall(self, problems: list[str], failed: dict, detail: dict) -> None:
        """Journal a stall when it starts and when it clears (kv DATA_STALL_KEY), and fail the
        step on every pass while it lasts (_Failed: journaled here, not by run_once)."""
        prev = self.store.get_kv(DATA_STALL_KEY)
        prev = prev if isinstance(prev, dict) and prev.get("since") is not None else None
        if not problems:
            if prev is not None:
                hours = (self.now - int(prev["since"])) / HOUR
                self.store.add_journal("system", f"Candle data recovered after {hours:.1f} h",
                                       f"Stalled since {iso_time(int(prev['since']))}: {prev['reason']}",
                                       now_t=self.now_t)
                self.store.set_kv(DATA_STALL_KEY, None)
                log.info("candle data recovered after %.1f h", hours)
            return
        errors = [f"{s} {tf}: {msg}" for (s, tf), msg in sorted(failed.items())[:3]]
        reason = "; ".join(problems) + (f" (e.g. {'; '.join(errors)})" if errors else "")
        since = self.now if prev is None else int(prev["since"])
        if prev is None:
            self.store.add_journal(
                "system", f"Candle data stalled: {reason}"[:300],
                f"{reason}\n\nPaper accounts cannot advance without new bars. Check the network"
                " and the exchange (HTTP 451/403 means it blocks this region); the data step fails"
                " on every pass until bars arrive, and the recovery is journaled.", now_t=self.now_t)
            self.store.set_kv(DATA_STALL_KEY, {"since": since, "reason": reason})
        self.data_stalled = reason
        raise _Failed(f"candle data stalled since {iso_time(since)}: {reason} ({detail['new_bars']}"
                      f" new bars, {detail['failed']} failed updates)")

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
        kw = {"stop": cycle_stop(self.cfg, self.stop)} if accepts(cycles.review_signals, "stop") else {}
        results = cycles.review_signals(self.lab, self.brain, **kw)
        done = [r for r in results if "decision" in r]
        lost = next((r["error"] for r in results if r.get("error", "").startswith("brain unavailable")),
                    None)
        if lost:
            self.lose_brain(lost)
        return {"reviewed": len(done), "take": sum(r["decision"] == "take" for r in done),
                "skip": sum(r["decision"] == "skip" for r in done), "failed": len(results) - len(done)}

    def graduation_step(self) -> dict:
        """graduation.evaluate per paper strategy; journal (kind "graduation") each status change
        and retire a strategy whose paper drawdown exceeds AUTO_RETIRE_FACTOR x the limit. Without
        a brain (no-brain mode or the brain is unavailable) a strategy whose status is "failing"
        is retired too: nobody else would, and it would block the autopilot's research. One
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
                elif self.brain is None and g["status"] == "failing":
                    self.lab.retire(sid, "auto-retired by the scheduler (no brain): graduation status"
                                         f" failing - {g['summary']}")
                    log.warning("auto-retired %s (%s): graduation status failing", row["name"], sid)
                    retired.append(sid)
            except Exception as e:
                self.journal_failure(f"graduation {sid}", e)
                failed += 1
        return {"statuses": statuses, "retired": retired, "failed": failed}

    def _journal_change(self, row: dict, g: dict) -> None:
        """Journal the graduation status when it differs from the last one recorded for this
        paper run (keyed by paper_start_t)."""
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
        """brain.cycles.review_cycle when review_due() and the candle data is not stalled (the
        paper accounts cannot have changed)."""
        self.need_brain()
        if not review_due(self.store, self.cfg, self.now, self.roster()):
            raise _Skip("not due")
        if self.data_stalled:
            raise _Skip("waiting for fresh candles (the data is stalled)")
        return self._run_cycle("review")

    def research_step(self) -> dict:
        """brain.cycles.research_cycle when research_due(). A cycle that is due only through the
        empty-roster cadence (brain.review_every_hours) is deferred while the rolling weekly
        validation cap is used up: it could neither validate nor promote anything."""
        self.need_brain()
        roster_empty = not self.roster()
        if not research_due(self.store, self.cfg, self.now, roster_empty):
            raise _Skip("not due")
        weekly = getattr(self.lab, "weekly_validations_left", None)
        if (roster_empty and callable(weekly)
                and not due(self.store.get_kv("last_research_t"),
                            self.cfg["brain"]["research_every_hours"], self.now)
                and weekly(self.now) == 0):
            raise _Skip("research deferred: weekly validation cap reached")
        return self._run_cycle("research")

    def _run_cycle(self, kind: str) -> dict:
        """Run a due brain cycle unless retrying() a failed one; record the attempt first. A
        cycle function with a `stop` parameter gets cycle_stop() (time limit and stop signal)."""
        if retrying(self.store, kind, self.now):
            raise _Skip(f"the last {kind} cycle did not finish; retrying {CYCLE_RETRY_HOURS}h after it started")
        from .brain import cycles
        self.store.set_kv(ATTEMPT_KEY.format(kind), self.now)
        fn = getattr(cycles, f"{kind}_cycle")
        kw = {"stop": cycle_stop(self.cfg, self.stop)} if accepts(fn, "stop") else {}
        return self._cycle_result(fn(self.lab, self.brain, **kw))

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
        """Write REPORT.md; about once a day (COMPACT_EVERY_HOURS) also Store.compact() the
        database (old experiment detail and system journal bodies; gate inputs are kept)."""
        out = {"path": str(write_report(self.lab, now_t=self.now_t))}
        compact = getattr(self.store, "compact", None)
        if callable(compact) and due(self.store.get_kv(COMPACT_KEY), COMPACT_EVERY_HOURS, self.now):
            self.store.set_kv(COMPACT_KEY, self.now)
            try:
                res = compact(now_t=self.now)
                log.info("database compacted: %s -> %s bytes", res["bytes_before"], res["bytes_after"])
            except Exception as e:  # maintenance must never fail the report
                log.warning("database compaction failed: %s", e)
        return out


def run_once(cfg: dict, now_t: int | None = None, use_brain: bool = True, lab=None, brain=None,
             stop=None) -> dict:
    """One scheduler pass (module docstring). lab defaults to a Lab(cfg) that is closed again;
    brain to make_brain(cfg) when use_brain. stop: optional callable checked before each step;
    once it returns True the remaining steps are skipped (a brain cycle that accepts `stop` also
    sees it between its turns, see cycle_stop).

    Returns {"t", "brain": {"available", "reason"} (at the start of the pass), "steps": [{"step",
    "status": "ok"|"skipped"|"failed", "seconds", "detail": dict (ok) | reason | error}],
    "stopped": bool}.
    """
    own = lab is None
    lab = Lab(cfg) if own else lab
    try:
        p = _Pass(cfg, lab, brain, now_t, use_brain, stop)
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
    period = int(cfg["scheduler"]["tick_every_seconds"])
    offset = int(cfg["scheduler"]["tick_offset_seconds"])
    if period <= 0:
        raise ValueError(f"scheduler.tick_every_seconds must be > 0, got {period}")
    validate_config(cfg)            # load_config validates too; this covers configs built in code
    setup_logging(cfg)
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
