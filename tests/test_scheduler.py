"""Tests for fibbot.scheduler: step order, error isolation (a failing step is journaled and later
steps still run), the research/review/autopilot due logic and the brain-unavailable fallback,
graduation journaling and auto-retire, logging, and run_forever's sleep/signal handling.

No network and no Claude: the Lab, brain cycles, autopilot and report are fakes, except for one
end-to-end pass on the real Lab with synthetic candles.
"""
import io
import json
import logging
import os
import re
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fibbot import scheduler
from fibbot.config import DEFAULTS, deep_merge
from fibbot.lab import Lab
from fibbot.scheduler import (ATTEMPT_KEY, AUTO_RETIRE_FACTOR, CYCLE_RETRY_HOURS, GRADUATION_KEY, STEPS,
                              autopilot_due, next_run_t, retrying,
                              research_due, review_due, run_forever, run_once, setup_logging)
from fibbot.store import Store
from fibbot.strategy import PRESETS
from fibbot.types import TF_SECONDS, Bars, Trade
from tests.test_lab import SYMS, FakeCandles as LabCandles, make_cfg as lab_cfg

NOW = 1_700_000_000
HOUR = 3600
SPEC = PRESETS["golden_pocket_4h"]


def cfg_(home, **over):
    return deep_merge(deep_merge(DEFAULTS, {"home": str(home)}), over)


def grad(status, dd=0.01):
    return {"status": status, "summary": f"{status}: summary",
            "checks": [{"name": "min_days", "value": 3.0, "threshold": 60, "passed": False},
                       {"name": "max_drawdown", "value": dd, "threshold": 0.15, "passed": dd <= 0.15}]}


def open_trade(sid, symbol):
    return Trade(sid, symbol, "4h", 1, "L1", NOW - 10 * HOUR, NOW - 8 * HOUR, 100.0, 95.0, None,
                 101.0, "open", 0.01, 0.05, 0.2, 2, {})


class FakeBrain:
    def __init__(self, ok=True, why="model x"):
        self.ok, self.why = ok, why

    def available(self):
        return self.ok, self.why


class FakeCandles:
    """update_all "fails" for BADUSDT only; get serves one bar per (symbol, tf) that ends at
    `ends.get((symbol, tf), self.end)` (None = no bars), fresh at NOW by default."""

    def __init__(self, lab):
        self.lab, self.last_errors, self.end, self.ends, self.errors = lab, {}, NOW, {}, None

    def update_all(self, symbols, tfs):
        self.lab.record("update_all", list(symbols), list(tfs))
        if self.errors is not None:          # e.g. every request answered HTTP 451
            self.last_errors = {(s, tf): self.errors for s in symbols for tf in tfs}
            return {}
        self.last_errors = {("BADUSDT", "4h"): "boom"}
        return {(s, tf): 2 for s in symbols for tf in tfs}

    def get(self, symbol, tf):
        end = self.ends.get((symbol, tf), self.end)
        if end is None:
            return Bars(symbol, tf)
        t = end - TF_SECONDS[tf]
        return Bars.from_rows(symbol, tf, [(t, 1.0, 1.0, 1.0, 1.0, 1.0)])


class FakePaper:
    def __init__(self, lab):
        self.lab = lab

    def tick(self, update_data=True, now_t=None, universe=None):
        self.lab.record("tick", update_data, now_t, universe)
        entry = {**open_trade("s1", "AAAUSDT").to_dict(), "notional": 10.0, "pnl": 0.1}
        exit_ = {**entry, "exit_t": NOW, "exit_reason": "target", "r_multiple": 1.5}
        return {"strategies": {"s1": {"new_signals": [{}], "new_entries": [entry], "new_exits": [exit_],
                                      "equity": 10001.0, "open": 1}},
                "new_signals": [object()], "errors": [{"strategy_id": "s1", "symbol": None, "error": "x"}]}


class FakeLab:
    """Records calls in order; a call whose name is in `fail` raises RuntimeError."""

    def __init__(self, cfg, store):
        self.cfg, self.store, self.calls, self.fail = cfg, store, [], set()
        self.candles, self._paper, self.grades = FakeCandles(self), FakePaper(self), {}

    def record(self, name, *args):
        self.calls.append((name, *args))
        if name in self.fail:
            raise RuntimeError(f"{name} exploded")

    def names(self):
        return [c[0] for c in self.calls]

    def universe(self):
        self.record("universe")
        return list(SYMS)

    def paper(self):
        return self._paper

    def graduation(self, sid, now_t=None):
        self.record("graduation", sid)
        return self.grades.get(sid, grad("collecting"))

    def retire(self, sid, reason):
        self.record("retire", sid, reason)
        self.store.set_strategy_status(sid, "retired", reason)


class SchedulerCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.store = Store(self.home / "fibbot.sqlite")
        self.addCleanup(self.store.close)
        self.cfg = cfg_(self.home, data={"timeframes": ["4h"]})
        self.lab = FakeLab(self.cfg, self.store)
        self.sweep_out = {"ranking": [{"preset": "x"}], "validations": [], "promoted": [], "errors": []}
        self.cycle_out = {"ok": True, "turns": 3, "stop_reason": "end_turn", "refused": False,
                          "promoted": [], "retired": [], "journal_id": 1}
        self.reviews = [{"decision": "take"}, {"decision": "skip"}, {"error": "no usable review"}]
        for target, fn in (("fibbot.brain.cycles.review_signals", self._review_signals),
                           ("fibbot.brain.cycles.review_cycle", self._cycle("review_cycle")),
                           ("fibbot.brain.cycles.research_cycle", self._cycle("research_cycle")),
                           ("fibbot.autopilot.sweep", self._sweep),
                           ("fibbot.scheduler.write_report", self._report)):
            p = mock.patch(target, side_effect=fn)
            p.start()
            self.addCleanup(p.stop)

    def _review_signals(self, lab, brain):
        lab.record("review_signals")
        return self.reviews

    def _cycle(self, name):
        def run(lab, brain):
            lab.record(name)
            return {**self.cycle_out, "kind": name.split("_")[0]}
        return run

    def _sweep(self, lab, promote=False):
        lab.record("sweep", promote)
        return self.sweep_out

    def _report(self, lab, now_t=None):
        lab.record("write_report")
        return self.home / "REPORT.md"

    def add(self, sid="s1", start=NOW - 30 * 86400, spec=SPEC):
        self.store.add_to_roster(sid, f"name-{sid}", spec.to_dict(), start, {}, "because", now_t=start)

    def go(self, brain=None, now=NOW, **kw):
        return run_once(self.cfg, now_t=now, lab=self.lab, brain=FakeBrain() if brain is None else brain, **kw)

    def steps(self, out):
        return {s["step"]: s for s in out["steps"]}

    def system_journal(self):
        return self.store.journal(kind="system")


class TestOrderAndIsolation(SchedulerCase):
    def test_full_pass_runs_every_step_in_order(self):
        self.add()
        out = self.go()
        self.assertEqual(self.lab.names(), ["universe", "update_all", "tick", "review_signals",
                                            "graduation", "review_cycle", "research_cycle", "write_report"])
        st = self.steps(out)
        self.assertEqual(list(st), list(STEPS))
        self.assertEqual({k for k, v in st.items() if v["status"] != "ok"}, {"autopilot"})
        self.assertEqual(st["autopilot"]["detail"], "the brain does the research")
        self.assertEqual(out["brain"], {"available": True, "reason": "model x"})
        self.assertFalse(out["stopped"])
        self.assertEqual(st["universe"]["detail"], {"symbols": len(SYMS)})
        self.assertEqual(st["paper"]["detail"], {"strategies": 1, "new_signals": 1, "new_entries": 1,
                                                 "new_exits": 1, "errors": 1})
        self.assertEqual(st["signals"]["detail"], {"reviewed": 2, "take": 1, "skip": 1, "failed": 1})
        self.assertEqual(st["report"]["detail"], {"path": str(self.home / "REPORT.md")})
        self.assertEqual(st["research"]["detail"]["turns"], 3)
        self.assertEqual(self.system_journal(), [])
        json.dumps(out)

    def test_tick_uses_the_updated_data_and_universe(self):
        self.add()
        self.go()
        tick = next(c for c in self.lab.calls if c[0] == "tick")
        self.assertEqual(tick, ("tick", False, NOW, list(SYMS)))

    def test_failing_step_is_journaled_and_later_steps_run(self):
        self.add()
        self.lab.fail = {"tick"}
        out = self.go()
        st = self.steps(out)
        self.assertEqual(st["paper"]["status"], "failed")
        self.assertIn("tick exploded", st["paper"]["detail"])
        self.assertEqual(self.lab.names()[-5:], ["review_signals", "graduation", "review_cycle",
                                                 "research_cycle", "write_report"])
        j, = self.system_journal()
        self.assertIn("paper", j["title"])
        self.assertIn("tick exploded", j["title"])
        self.assertIn("Traceback", j["body"])
        self.assertEqual(j["t"], NOW)

    def test_every_step_failing_still_completes_the_pass(self):
        self.add()
        self.lab.fail = {"universe", "update_all", "tick", "review_signals", "graduation",
                         "review_cycle", "research_cycle", "write_report"}
        out = self.go()
        st = self.steps(out)
        self.assertEqual([k for k, v in st.items() if v["status"] == "failed"],
                         ["universe", "data", "paper", "signals", "review", "research", "report"])
        self.assertEqual(self.steps(out)["graduation"]["detail"]["failed"], 1)   # isolated per strategy
        titles = [j["title"] for j in self.system_journal()]
        self.assertEqual(len(titles), 8)
        self.assertTrue(any("graduation s1" in t for t in titles))

    def test_universe_failure_keeps_held_symbols_and_lets_tick_resolve_the_universe(self):
        self.add()
        self.store.upsert_paper_trade("s1", open_trade("s1", "OLDUSDT"))
        self.lab.fail = {"universe"}
        self.go()
        upd = next(c for c in self.lab.calls if c[0] == "update_all")
        self.assertEqual(upd[1], ["OLDUSDT"])
        self.assertIsNone(next(c for c in self.lab.calls if c[0] == "tick")[3])

    def test_journal_failure_never_raises(self):
        self.lab.fail = {"universe"}
        with mock.patch.object(self.store, "add_journal", side_effect=RuntimeError("db gone")):
            out = self.go()
        self.assertEqual(self.steps(out)["universe"]["status"], "failed")
        self.assertEqual(self.steps(out)["report"]["status"], "ok")

    def test_stop_skips_the_remaining_steps(self):
        self.add()
        out = self.go(stop=lambda: "tick" in self.lab.names())
        self.assertTrue(out["stopped"])
        self.assertEqual([s["step"] for s in out["steps"]], ["universe", "data", "paper"])
        self.assertNotIn("write_report", self.lab.names())

    def test_run_once_opens_and_closes_its_own_lab(self):
        store = mock.Mock()
        lab = FakeLab(self.cfg, store)
        store.roster.return_value, store.get_kv.return_value = [], None
        with mock.patch("fibbot.scheduler.Lab", return_value=lab) as make:
            run_once(self.cfg, now_t=NOW, use_brain=False)
        make.assert_called_once_with(self.cfg)
        store.close.assert_called_once_with()


class TestDataStep(SchedulerCase):
    def test_symbols_and_timeframes(self):
        self.add("s1", spec=SPEC.with_overrides(timeframe="1h"))
        self.add("s2", spec=SPEC)
        self.add("gone")
        self.store.upsert_paper_trade("s1", open_trade("s1", "OLDUSDT"))
        self.store.upsert_paper_trade("s1", open_trade("s1", SYMS[0]))
        self.store.upsert_paper_trade("gone", open_trade("gone", "RETIREDUSDT"))
        self.store.set_strategy_status("gone", "retired")
        out = self.go()
        upd = next(c for c in self.lab.calls if c[0] == "update_all")
        self.assertEqual(upd[1], [*SYMS, "OLDUSDT"])
        self.assertEqual(upd[2], ["4h", "1h"])                # data.timeframes, then roster extras
        self.assertEqual(self.steps(out)["data"]["detail"],
                         {"symbols": 4, "timeframes": ["4h", "1h"], "new_bars": 16, "failed": 1})

    def test_paper_is_skipped_without_strategies(self):
        out = self.go(brain=FakeBrain(False, "no key"))
        self.assertEqual(self.steps(out)["paper"], {**self.steps(out)["paper"], "status": "skipped",
                                                    "detail": "no strategies on paper"})
        self.assertNotIn("tick", self.lab.names())


class TestDueLogic(SchedulerCase):
    def test_research_due(self):
        b = self.cfg["brain"]
        r, v = b["research_every_hours"] * HOUR, b["review_every_hours"] * HOUR
        self.assertTrue(research_due(self.store, self.cfg, NOW, roster_empty=False))     # never ran
        for age, empty, expected in ((r - 1, False, False), (r, False, True), (v - 1, True, False),
                                     (v, True, True), (v, False, False)):
            self.store.set_kv("last_research_t", NOW - age)
            self.assertEqual(research_due(self.store, self.cfg, NOW, empty), expected, (age, empty))

    def test_review_due(self):
        v = self.cfg["brain"]["review_every_hours"] * HOUR
        self.assertFalse(review_due(self.store, self.cfg, NOW, []))
        rows = [{"paper_start_t": NOW - v + 1}, {"paper_start_t": NOW + 3600}]
        self.assertFalse(review_due(self.store, self.cfg, NOW, rows))          # first review: from paper start
        rows[0]["paper_start_t"] = NOW - v
        self.assertTrue(review_due(self.store, self.cfg, NOW, rows))
        self.store.set_kv("last_review_t", NOW - 10)
        self.assertFalse(review_due(self.store, self.cfg, NOW, rows))
        self.store.set_kv("last_review_t", NOW - v)
        self.assertTrue(review_due(self.store, self.cfg, NOW, rows))

    def test_autopilot_due(self):
        r = self.cfg["brain"]["research_every_hours"] * HOUR
        self.assertTrue(autopilot_due(self.store, self.cfg, NOW))
        self.store.set_kv("last_autopilot_t", NOW - r + 1)
        self.assertFalse(autopilot_due(self.store, self.cfg, NOW))
        self.store.set_kv("last_autopilot_t", NOW - r)
        self.assertTrue(autopilot_due(self.store, self.cfg, NOW))

    def test_cycles_not_due_are_skipped(self):
        self.add(start=NOW - 3600)
        self.store.set_kv("last_research_t", NOW - 3600)
        out = self.go()
        st = self.steps(out)
        self.assertEqual((st["review"]["status"], st["review"]["detail"]), ("skipped", "not due"))
        self.assertEqual((st["research"]["status"], st["research"]["detail"]), ("skipped", "not due"))
        self.assertNotIn("review_cycle", self.lab.names())
        self.assertNotIn("research_cycle", self.lab.names())

    def test_empty_roster_research_runs_at_most_once_per_review_period(self):
        out = self.go()
        self.assertIn("research_cycle", self.lab.names())
        self.assertEqual(self.steps(out)["review"]["detail"], "not due")      # nothing to review
        self.store.set_kv("last_research_t", NOW)
        self.lab.calls.clear()
        self.go(now=NOW + HOUR)
        self.assertNotIn("research_cycle", self.lab.names())
        self.go(now=NOW + self.cfg["brain"]["review_every_hours"] * HOUR)
        self.assertIn("research_cycle", self.lab.names())

    def test_retrying(self):
        self.assertFalse(retrying(self.store, "research", NOW))                 # never attempted
        self.store.set_kv(ATTEMPT_KEY.format("research"), NOW - 10)
        self.assertTrue(retrying(self.store, "research", NOW))                  # attempted, never finished
        self.assertFalse(retrying(self.store, "review", NOW))                   # per kind
        self.store.set_kv("last_research_t", NOW - 10)                          # that attempt finished
        self.assertFalse(retrying(self.store, "research", NOW))
        self.store.set_kv("last_research_t", NOW - 20)                          # an older one finished
        self.assertTrue(retrying(self.store, "research", NOW))
        self.assertFalse(retrying(self.store, "research", NOW - 10 + CYCLE_RETRY_HOURS * HOUR))

    def test_a_failed_cycle_is_retried_after_the_backoff_not_every_pass(self):
        self.cycle_out = {"ok": False, "error": "API down", "actions": [], "journal_id": 1}
        self.go()
        self.assertEqual(self.store.get_kv(ATTEMPT_KEY.format("research")), NOW)
        self.lab.calls.clear()
        out = self.go(now=NOW + HOUR)
        self.assertNotIn("research_cycle", self.lab.names())
        self.assertIn("did not finish", self.steps(out)["research"]["detail"])
        self.go(now=NOW + CYCLE_RETRY_HOURS * HOUR)
        self.assertIn("research_cycle", self.lab.names())

    def test_a_crashing_review_cycle_backs_off_too(self):
        self.add()
        with mock.patch("fibbot.brain.cycles.review_cycle", side_effect=RuntimeError("boom")):
            out = self.go()
            self.assertEqual(self.steps(out)["review"]["status"], "failed")
            out = self.go(now=NOW + HOUR)
        self.assertIn("did not finish", self.steps(out)["review"]["detail"])

    def test_signal_review_off(self):
        self.add()
        self.cfg["brain"]["signal_review"] = "off"
        out = self.go()
        self.assertEqual(self.steps(out)["signals"]["detail"], "brain.signal_review is off")
        self.assertNotIn("review_signals", self.lab.names())


class TestBrainFallback(SchedulerCase):
    def test_unavailable_brain_and_empty_roster_runs_the_autopilot_once_per_period(self):
        out = self.go(brain=FakeBrain(False, "no key"))
        st = self.steps(out)
        self.assertEqual(out["brain"], {"available": False, "reason": "no key"})
        self.assertEqual(st["autopilot"]["detail"], {"ranked": 1, "validated": 0, "promoted": [], "errors": 0})
        self.assertIn(("sweep", True), self.lab.calls)
        for name in ("signals", "review", "research"):
            self.assertEqual(st[name]["detail"], "brain unavailable (no key)")
        self.assertEqual(self.store.get_kv("last_autopilot_t"), NOW)
        self.lab.calls.clear()
        out = self.go(brain=FakeBrain(False, "no key"), now=NOW + 3600)
        self.assertNotIn("sweep", self.lab.names())
        self.assertEqual(self.steps(out)["autopilot"]["detail"], "swept within brain.research_every_hours")
        self.go(brain=FakeBrain(False), now=NOW + self.cfg["brain"]["research_every_hours"] * HOUR)
        self.assertIn("sweep", self.lab.names())

    def test_a_sweep_that_ranked_nothing_is_retried_next_pass(self):
        self.sweep_out = {**self.sweep_out, "ranking": []}
        self.go(brain=FakeBrain(False))
        self.assertIsNone(self.store.get_kv("last_autopilot_t"))
        self.lab.calls.clear()
        self.go(brain=FakeBrain(False), now=NOW + 3600)
        self.assertIn("sweep", self.lab.names())

    def test_no_autopilot_when_the_roster_is_not_empty(self):
        self.add()
        out = self.go(brain=FakeBrain(False))
        self.assertEqual(self.steps(out)["autopilot"]["detail"], "the roster is not empty")
        self.assertNotIn("sweep", self.lab.names())
        self.assertEqual(self.lab.names(), ["universe", "update_all", "tick", "graduation", "write_report"])

    def test_no_brain_never_builds_one(self):
        with mock.patch("fibbot.scheduler.make_brain", side_effect=AssertionError("built")):
            out = run_once(self.cfg, now_t=NOW, use_brain=False, lab=self.lab)
        self.assertEqual(out["brain"], {"available": False, "reason": "disabled by --no-brain"})
        self.assertIn(("sweep", True), self.lab.calls)

    def test_default_brain_is_built_and_a_broken_setup_counts_as_unavailable(self):
        with mock.patch("fibbot.scheduler.make_brain", return_value=FakeBrain()) as make:
            out = run_once(self.cfg, now_t=NOW, lab=self.lab)
        make.assert_called_once_with(self.cfg)
        self.assertTrue(out["brain"]["available"])
        with mock.patch("fibbot.scheduler.make_brain", side_effect=ImportError("nope")):
            out = run_once(self.cfg, now_t=NOW, lab=self.lab)
        self.assertEqual(out["brain"], {"available": False, "reason": "ImportError: nope"})

    def test_make_brain_needs_no_anthropic(self):
        with mock.patch.dict("sys.modules", {"anthropic": None}), mock.patch.dict(os.environ, clear=True):
            ok, why = scheduler.make_brain(self.cfg).available()
        self.assertFalse(ok)
        self.assertIn("anthropic", why)

    def test_research_cycle_losing_the_brain_falls_back_to_the_autopilot(self):
        self.cycle_out = {"ok": False, "error": "API down", "actions": [], "journal_id": 1}
        out = self.go()
        st = self.steps(out)
        self.assertEqual(st["research"]["status"], "failed")
        self.assertIn("API down", st["research"]["detail"])
        self.assertIn(("sweep", True), self.lab.calls)
        self.assertEqual(self.system_journal(), [])        # the cycle journals its own failure

    def test_signal_reviews_losing_the_brain_skip_the_cycles(self):
        self.add()
        self.reviews = [{"error": "brain unavailable: 529 overloaded"}]
        out = self.go()
        st = self.steps(out)
        self.assertEqual(st["signals"]["detail"]["failed"], 1)
        self.assertEqual(st["review"]["detail"], "brain unavailable (brain unavailable: 529 overloaded)")
        self.assertNotIn("review_cycle", self.lab.names())
        self.assertNotIn("research_cycle", self.lab.names())


class TestGraduationStep(SchedulerCase):
    def graduation_journal(self):
        return self.store.journal(kind="graduation")

    def test_status_changes_are_journaled_once(self):
        self.add()
        self.go()
        self.go()
        j, = self.graduation_journal()
        self.assertEqual(j["title"], "name-s1 (s1) is collecting")
        self.assertIn("collecting: summary", j["body"])
        self.assertIn("- [ ] min_days: 3.0 (threshold 60)", j["body"])
        self.lab.grades["s1"] = grad("promising")
        out = self.go()
        self.assertEqual(self.graduation_journal()[0]["title"], "name-s1 (s1) collecting -> promising")
        self.assertEqual(self.steps(out)["graduation"]["detail"],
                         {"statuses": {"s1": "promising"}, "retired": [], "failed": 0})
        self.assertEqual(self.store.get_kv(GRADUATION_KEY.format("s1"))["status"], "promising")

    def test_a_status_of_another_paper_run_is_not_carried_over(self):
        # The store no longer re-activates a retired id (retirement is final), so a status
        # recorded for an earlier paper run (another paper_start_t) stands in for one here.
        self.add("s0")
        self.store.set_strategy_status("s0", "retired")
        with self.assertRaises(ValueError):
            self.add("s0", start=NOW - 86400)
        self.add()
        self.store.set_kv(GRADUATION_KEY.format("s1"),
                          {"status": "promising", "paper_start_t": NOW - 99 * 86400, "t": NOW - 86400})
        self.go()
        self.assertEqual([j["title"] for j in self.graduation_journal()],
                         ["name-s1 (s1) is collecting"])

    def test_auto_retire_beyond_the_drawdown_limit(self):
        limit = AUTO_RETIRE_FACTOR * self.cfg["graduation"]["max_drawdown"]
        self.add("s1")
        self.add("s2")
        self.lab.grades = {"s1": grad("failing", dd=limit + 1e-6), "s2": grad("failing", dd=limit)}
        out = self.go()
        retire, = [c for c in self.lab.calls if c[0] == "retire"]
        self.assertEqual(retire[1], "s1")
        self.assertIn("drawdown 22.5%", retire[2])
        self.assertEqual(self.steps(out)["graduation"]["detail"]["retired"], ["s1"])
        self.assertEqual([r["strategy_id"] for r in self.store.roster()], ["s2"])

    def test_one_failing_strategy_does_not_block_the_others(self):
        self.add("s1")
        self.add("s2")
        real = self.lab.graduation

        def graduation(sid, now_t=None):
            if sid == "s1":
                raise KeyError(sid)
            return real(sid, now_t)
        self.lab.graduation = graduation
        out = self.go()
        self.assertEqual(self.steps(out)["graduation"]["detail"],
                         {"statuses": {"s2": "collecting"}, "retired": [], "failed": 1})
        self.assertIn("graduation s1", self.system_journal()[0]["title"])


class TestHelpers(unittest.TestCase):
    def test_next_run_t(self):
        p, off = 3600, 90
        base = 1_700_002_800                    # a UTC hour boundary
        self.assertEqual(base % p, 0)
        for now, expected in ((base, base + off), (base + 30, base + off), (base + off, base + p + off),
                              (base + off + 1, base + p + off), (base + p - 1, base + p + off)):
            self.assertEqual(next_run_t(now, p, off), expected, now - base)
        self.assertEqual(next_run_t(base + 5, 14400, 0) % 14400, 0)

    def test_setup_logging_writes_utc_lines_to_stream_and_file(self):
        with tempfile.TemporaryDirectory() as home:
            cfg = cfg_(home)
            stream = io.StringIO()
            setup_logging(cfg, stream)
            handlers = setup_logging(cfg, stream)                   # replaces, never duplicates
            logger = logging.getLogger("fibbot")
            try:
                self.assertEqual(sum(h in logger.handlers for h in handlers), 2)
                self.assertEqual(len([h for h in logger.handlers if h in scheduler._handlers]), 2)
                logging.getLogger("fibbot.data.candles").info("hello %s", "world")
            finally:
                for h in handlers:
                    logger.removeHandler(h)
                    h.close()
                scheduler._handlers.clear()
            pattern = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ INFO fibbot\.data\.candles: hello world$"
            self.assertRegex(stream.getvalue().strip(), pattern)
            log_text = (Path(home) / "fibbot.log").read_text().strip()
            self.assertRegex(log_text, pattern)
            self.assertEqual(log_text.count("hello world"), 1)


@unittest.skipUnless(threading.current_thread() is threading.main_thread(), "needs the main thread")
class TestRunForever(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cfg = cfg_(tmp.name)
        self.lab = mock.Mock()
        for target, kw in (("fibbot.scheduler.setup_logging", {}),
                           ("fibbot.scheduler.Lab", {"return_value": self.lab}),
                           ("fibbot.scheduler.make_brain", {"return_value": "brain"})):
            p = mock.patch(target, **kw)
            p.start()
            self.addCleanup(p.stop)
        self.passes = []

    def forever(self, on_pass, wake_in=0.0):
        def fake_run_once(cfg, use_brain, lab, brain, stop):
            self.passes.append((use_brain, lab, brain, stop()))
            on_pass(len(self.passes), stop)
        with mock.patch("fibbot.scheduler.run_once", side_effect=fake_run_once), \
                mock.patch("fibbot.scheduler.next_run_t", side_effect=lambda now, p, o: now + wake_in):
            run_forever(self.cfg)

    def test_signal_stops_after_the_current_pass_and_restores_handlers(self):
        before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        seen = []

        def on_pass(n, stop):
            if n == 2:
                os.kill(os.getpid(), signal.SIGTERM)
                seen.append(stop())
        self.forever(on_pass)
        self.assertEqual(len(self.passes), 2)
        self.assertEqual(seen, [True])                      # run_once sees the stop request
        self.assertEqual(self.passes[0], (True, self.lab, "brain", False))
        self.lab.store.close.assert_called_once_with()
        self.assertEqual((signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)), before)

    def test_a_signal_while_sleeping_ends_the_loop(self):
        timer = threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGINT))
        self.addCleanup(timer.cancel)
        timer.start()
        t0 = time.monotonic()
        self.forever(lambda n, stop: None, wake_in=30.0)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertEqual(len(self.passes), 1)

    def test_second_signal_aborts_at_once(self):
        def on_pass(n, stop):
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)
        with self.assertRaises(KeyboardInterrupt):
            self.forever(on_pass)
        self.lab.store.close.assert_called_once_with()

    def test_a_crashing_pass_does_not_end_the_loop(self):
        def on_pass(n, stop):
            if n == 1:
                raise RuntimeError("bug")
            os.kill(os.getpid(), signal.SIGTERM)
        with self.assertLogs("fibbot.scheduler", "ERROR"):
            self.forever(on_pass)
        self.assertEqual(len(self.passes), 2)

    def test_bad_period_is_rejected_before_any_pass(self):
        def abort(n, stop):
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)
        self.cfg["scheduler"]["tick_every_seconds"] = 0
        with self.assertRaises(ValueError):
            self.forever(abort)
        self.assertEqual(self.passes, [])


class TestRealLab(unittest.TestCase):
    """One brainless pass on the real Lab (synthetic candles): the autopilot fills the empty roster,
    graduation is journaled and REPORT.md is written; the next pass ticks the new paper account."""

    def test_end_to_end_without_the_brain(self):
        class Candles(LabCandles):
            def update_all(self, symbols, tfs):
                self.updated = (list(symbols), list(tfs))
                return {}

        with tempfile.TemporaryDirectory() as home, \
                mock.patch("fibbot.lab.get_universe", return_value=list(SYMS)), \
                mock.patch.dict("sys.modules", {"anthropic": None}):
            cfg = lab_cfg(home, research={"validation_gate": {"min_deflated_sharpe": 0.0,
                                                              "min_profit_factor": 1.0,
                                                              "min_expectancy_r": 0.0,
                                                              "min_validation_edge_z": None}},
                          paper={"max_roster": 1})
            store, candles = Store(Path(home) / "fibbot.sqlite"), Candles()
            self.addCleanup(store.close)
            lab = Lab(cfg, store, candles)
            # "now" is just after the synthetic history (a later clock would be a candle stall)
            now = max(int(b.t[-1]) for b in candles.bars.values()) + TF_SECONDS["4h"] + 90
            out = run_once(cfg, lab=lab, use_brain=False, now_t=now)
            st = {s["step"]: s for s in out["steps"]}
            self.assertEqual({k for k, v in st.items() if v["status"] == "failed"}, set(), out)
            self.assertEqual(len(st["autopilot"]["detail"]["promoted"]), 1)
            sid, = [r["strategy_id"] for r in store.roster()]
            self.assertEqual(candles.updated, (list(SYMS), cfg["data"]["timeframes"]))
            self.assertTrue((Path(home) / "REPORT.md").read_text().startswith("# fibbot report"))
            self.assertEqual(store.journal(kind="research")[0]["title"][:15], "Autopilot sweep")

            out = run_once(cfg, lab=lab, use_brain=False, now_t=now + 3600)
            st = {s["step"]: s for s in out["steps"]}
            self.assertEqual(st["paper"]["status"], "ok", out)
            self.assertEqual(st["graduation"]["detail"]["statuses"], {sid: "collecting"})
            self.assertEqual(st["autopilot"]["detail"], "the roster is not empty")
            self.assertTrue(re.match(r"\S+ \(" + sid + r"\) is collecting",
                                     store.journal(kind="graduation")[0]["title"]))


if __name__ == "__main__":
    unittest.main()
