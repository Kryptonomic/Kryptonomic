"""Integration tests for the cross-group changes of the review round: costs parity (F18), baseline
risk on stop-through fills (F44), engine-versioned trials (F39), final retirement through the brain
tools (F08), the holdout-based validation gate (F01/F06/F09/F22) with its random-entry checks
(F02), frozen paper accounts (F31), the pinned validation epoch (F04), graduation on candle days
and paper edge (F02/F33), the brain's stop callable and SDK fallbacks (F13/F35), compaction and
report alerts (F33/F48), the no-brain scheduler (F14/F23) and the autopilot's binding edge (F40)."""
from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from fibbot import cli, research, scheduler
from fibbot.backtest import costs_from_cfg, result_from_trades
from fibbot.config import DEFAULTS, deep_merge, load_config, validate_config
from fibbot.graduation import evaluate
from fibbot.lab import EPOCH_KEY
from fibbot.paper import ACCOUNT_KEY, _costs, account_settings
from fibbot.report import render_status, write_report
from fibbot.store import Store
from fibbot.strategy import ENGINE_VERSION, PRESETS
from fibbot.types import TF_SECONDS, Bars, Costs, Trade
from tests.test_autopilot import PERMISSIVE_GATE, StubLab
from tests.test_lab import BARS, H4, OTHER, SPEC, SYMS, T0, FakeCandles, LabCase, mk_trade, record_validation

DAY = 86400


# -- F18: one trade-through margin everywhere -------------------------------------------------------

class TestLimitThroughParity(unittest.TestCase):
    def test_backtest_and_paper_read_the_same_costs(self):
        cfg = {"costs": {"limit_through": 0.0005}}
        self.assertEqual(costs_from_cfg(cfg).limit_through, 0.0005)
        self.assertEqual(_costs(cfg["costs"]).limit_through, 0.0005)
        self.assertEqual(costs_from_cfg(DEFAULTS), Costs(**DEFAULTS["costs"]))
        self.assertEqual(DEFAULTS["costs"]["limit_through"], Costs().limit_through)

    def test_config_accepts_and_bounds_it(self):
        validate_config(deep_merge(DEFAULTS, {"home": "/x", "costs": {"limit_through": 0.0}}))
        with self.assertRaisesRegex(ValueError, "costs.limit_through"):
            validate_config(deep_merge(DEFAULTS, {"home": "/x", "costs": {"limit_through": 0.06}}))

    def test_a_paper_account_freezes_every_cost_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "s.sqlite")
            self.addCleanup(store.close)
            cfg = deep_merge(DEFAULTS, {"home": tmp, "costs": {"fee_rate": 0.001, "slippage": 0.0005}})
            del cfg["costs"]["limit_through"]                      # a config that does not list it
            store.add_to_roster("s1", "x", SPEC.to_dict(), T0, {}, "r")
            snap = account_settings(store, cfg, store.get_strategy("s1"))
            self.assertEqual(snap["costs"]["limit_through"], Costs().limit_through)


# -- F44: the baseline's target distances use the planned risk --------------------------------------

class TestBaselineRiskOnStopThroughFills(unittest.TestCase):
    def test_target_distances_stay_positive(self):
        bars = {"AUSDT": Bars.from_rows("AUSDT", "4h", [(T0 + i * H4, 100.0, 101.0, 99.0, 100.0, 1.0)
                                                        for i in range(400)])}
        # a long limit at 100 (stop 90, planned risk 10%) that filled at 89 below its stop
        tr = Trade("s", "AUSDT", "4h", 1, "L1", T0 + 299 * H4, T0 + 300 * H4, 89.0, 90.0, T0 + 300 * H4,
                   89.0 * 0.9995, "stop", -0.012, 0.1, -0.12, 0, {"tp1": 110.0, "tp2": 120.0})
        res = result_from_trades(SPEC, [tr], ["AUSDT"], T0 + 250 * H4, T0 + 400 * H4, DEFAULTS)
        seen = []

        def capture(w, atr_w, e, side, px, stop, tp1, tp2, spec, costs, **kw):
            seen.append((px, stop, tp1, tp2))
            return {"r_multiple": 0.0}
        with mock.patch("fibbot.research.manage_position", side_effect=capture):
            research.random_baseline(SPEC, bars, DEFAULTS, T0 + 250 * H4, T0 + 400 * H4, n_runs=2,
                                     result=res, regime=False)
        for px, stop, tp1, tp2 in seen:
            self.assertLess(stop, px)
            self.assertLess(px, tp1)
            self.assertLess(tp1, tp2)
            self.assertAlmostEqual((px - stop) / px, 0.1)


# -- F39: trials of different engine versions ----------------------------------------------------------

class TestEngineVersions(LabCase):
    def test_variance_pools_only_the_current_engine(self):
        self.store.record_trial("a", 0.5, n=100, engine=ENGINE_VERSION - 1)
        self.store.record_trial("b", 0.1, n=100)
        self.store.record_trial("c", 0.3, n=100)
        st = self.store.trial_stats(min_trades=5)
        self.assertEqual((st["n"], st["n_var"]), (3, 2))                # an old trial still counts
        self.store.record_trial("a", 0.2, n=50)                         # re-evaluated: replaces it
        self.assertEqual(self.store.trial_stats(min_trades=5)["n_var"], 3)

    def test_experiments_record_the_engine_and_old_validations_do_not_promote(self):
        out = self.lab.backtest(SPEC)
        self.assertEqual(self.store.get_experiment(out["experiment_id"])["metrics"]["engine_version"],
                         ENGINE_VERSION)
        self.store.add_experiment("baseline", SPEC.to_dict(), SPEC.spec_id(), T0, T0 + H4,
                                  {"edge_r": 0.2, "edge_regime_r": 0.2, "engine_version": ENGINE_VERSION})
        self.store.add_experiment("validate", SPEC.to_dict(), SPEC.spec_id(), T0, T0 + 3000 * H4,
                                  {"passed": True, "checks": [], "engine_version": ENGINE_VERSION - 1})
        with self.assertRaisesRegex(ValueError, "engine changed since this validation"):
            self.lab.promote(SPEC, "old", "x")
        self.assertIsNone(self.lab.prior_validation(SPEC))             # not a verdict that stands


# -- F08: a retired spec cannot come back through the brain either ---------------------------------

class TestRetiredSpecThroughTheBrain(LabCase):
    def test_promote_tool_reports_retirement(self):
        try:
            from fibbot.brain.tools import build_tools
        except ImportError:                                             # pragma: no cover
            self.skipTest("anthropic not installed")
        record_validation(self.store, SPEC)
        sid = self.lab.promote(SPEC, "deep", "x", now_t=1_700_000_000)["strategy_id"]
        self.lab.retire(sid, "losing")
        tools = {t.name: t for t in build_tools(self.lab, "research")}
        from anthropic.lib.tools import ToolError
        with self.assertRaisesRegex(ToolError, "retired"):
            tools["promote_to_paper"].call({"name": "again", "rationale": "x", "spec_id": sid})
        self.assertEqual(self.store.get_strategy(sid)["status"], "retired")


# -- F01/F06/F09/F22/F02: the validation gate ------------------------------------------------------------

class TestValidationGateOnTheHoldout(LabCase):
    def setUp(self):
        super().setUp()
        self.lab = self.make_lab(research={"validation_gate": PERMISSIVE_GATE})

    def test_holdout_basis_and_variance_floor(self):
        out = self.lab.validate(SPEC)
        self.assertEqual(out["dsr_basis"], "holdout")
        self.assertEqual(out["holdout_k"], 1)
        self.assertEqual(out["dsr_trials"], research.MIN_GATE_TRIALS)
        self.assertGreaterEqual(out["var_sr"], 1.0 / out["metrics"]["n"] - 1e-5)   # >= 1 / n_val (rounded)
        self.assertGreaterEqual(out["var_sr"], 1.0 / out["n_eff"] - 1e-5)
        names = [c["name"] for c in out["gate"]["checks"]]
        self.assertIn("min_validation_edge_r", names)
        self.assertNotIn("min_validation_edge_z", names)                   # disabled by PERMISSIVE_GATE
        self.assertEqual(out["baseline"]["n_runs"], 50)
        second = self.lab.validate(OTHER)                                   # another spec, same window
        self.assertEqual((second["holdout_k"], second["dsr_trials"]), (2, 2))

    def test_research_padding_never_moves_the_holdout_gate(self):
        base = self.lab.validate(SPEC)
        for k in range(30):                                                 # zero-trade / duplicate trials
            self.store.record_trial(f"pad{k}", 0.0, n=0)
            self.store.record_trial(f"dup{k}", 0.05, n=500)
        self.lab.backtest(OTHER, symbols=SYMS[:1])                          # a symbol-subset run
        again = self.lab.validate(SPEC)                                     # identical draw: a replay
        self.assertTrue(again["replay"])
        for k in ("deflated_sharpe", "var_sr", "dsr_trials", "passed"):
            self.assertEqual(again["gate"].get(k, again.get(k)), base["gate"].get(k, base.get(k)), k)

    def test_the_edge_z_check_is_on_by_default(self):
        lab = self.make_lab(research={"validation_gate": {k: v for k, v in PERMISSIVE_GATE.items()
                                                          if k != "min_validation_edge_z"}})
        check = {c["name"]: c for c in lab.validate(SPEC)["gate"]["checks"]}["min_validation_edge_z"]
        self.assertEqual(check["threshold"], 1.65)

    def test_promotion_needs_a_research_edge_over_random_entries(self):
        self.store.add_experiment("validate", SPEC.to_dict(), SPEC.spec_id(), T0, T0 + H4,
                                  {"passed": True, "checks": [], "engine_version": ENGINE_VERSION})
        with self.assertRaisesRegex(ValueError, "no research-window baseline"):
            self.lab.promote(SPEC, "x", "x")
        self.store.add_experiment("baseline", SPEC.to_dict(), SPEC.spec_id(), T0, T0 + H4,
                                  {"edge_r": 0.3, "edge_regime_r": -0.01, "engine_version": ENGINE_VERSION})
        with self.assertRaisesRegex(ValueError, "does not beat random entries"):
            self.lab.promote(SPEC, "x", "x")


# -- F31 / F04: promotion freezes the account and pins the holdout ---------------------------------------

class TestPromotionPinsTheAccountAndTheEpoch(LabCase):
    NOW = 1_700_000_123

    def test_settings_and_epoch(self):
        record_validation(self.store, SPEC)
        before = self.lab.windows("4h")
        self.assertNotIn("pinned", before)
        sid = self.lab.promote(SPEC, "deep", "x", now_t=self.NOW)["strategy_id"]
        snap = self.store.get_kv(ACCOUNT_KEY.format(sid))
        self.assertEqual((snap["t"], snap["starting_equity"]), (self.NOW, 10000.0))
        epoch = self.lab.validation_epoch()
        self.assertEqual(epoch["validation_end"] % DAY, 0)
        self.assertLessEqual(epoch["validation_end"], before["data_end"])
        w = self.lab.windows("4h")
        self.assertTrue(w["pinned"])
        self.assertEqual(w["validation_end"], epoch["validation_end"])
        # more data no longer slides the holdout; the new bars are forward data only
        more = {s: Bars.from_rows(s, "4h", list(zip(b.t, b.o, b.h, b.l, b.c, b.v))
                                  + [(b.t[-1] + k * H4, b.c[-1], b.c[-1], b.c[-1], b.c[-1], 1.0)
                                     for k in range(1, 200)]) for s, b in BARS.items()}
        later = self.make_lab(candles=FakeCandles(more))
        w2 = later.windows("4h")
        self.assertEqual((w2["validation_start"], w2["validation_end"]),
                         (w["validation_start"], w["validation_end"]))
        self.assertGreater(w2["data_end"], w["data_end"])
        self.assertTrue(self.lab.status(now_t=self.NOW)["windows"]["4h"]["pinned"])

    def test_graduation_uses_the_frozen_starting_equity(self):
        record_validation(self.store, SPEC)
        sid = self.lab.promote(SPEC, "deep", "x", now_t=self.NOW)["strategy_id"]
        start = self.store.get_strategy(sid)["paper_start_t"]
        self.store.upsert_paper_trade(sid, mk_trade(sid, SYMS[0], "L1", start + H4, -1.0), 2500.0, -500.0)
        self.store.set_equity_curve(sid, [(start, 10000.0), (start + 4 * H4, 9500.0)])
        dd = lambda cfg: next(c["value"] for c in evaluate(self.store, cfg, sid, now_t=start + DAY)["checks"]
                              if c["name"] == "max_drawdown")
        edited = deep_merge(self.cfg, {"paper": {"starting_equity": 1_000_000.0}})
        self.assertAlmostEqual(dd(self.cfg), dd(edited))
        self.assertAlmostEqual(dd(self.cfg), 0.05, places=6)

    def test_rolling_the_epoch_is_explicit_and_journaled(self):
        with self.assertRaisesRegex(ValueError, "no validation epoch"):
            self.lab.roll_validation_epoch()
        self.store.set_kv(EPOCH_KEY, {"validation_end": T0 + 1000 * H4 - (T0 + 1000 * H4) % DAY})
        old = self.lab.validation_epoch()["validation_end"]
        out = self.lab.roll_validation_epoch(now_t=T0 + 3000 * H4)
        self.assertGreater(out["validation_end"], old)
        self.assertEqual(self.store.journal(kind="system")[0]["title"][:24], "Validation epoch rolled:")
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(cli.cmd_epoch(mock.Mock(roll=False), self.cfg | {"home": str(self.home)}), 0)


# -- F02 / F33: graduation counts candle days and needs an edge over random entries ------------------------

class TestGraduationWithCandles(LabCase):
    def setUp(self):
        super().setUp()
        self.store.add_to_roster("g", "g", SPEC.to_dict(), T0 + 1500 * H4, {"metrics": {"avg_r": 0.3}}, "r")

    def bars_until(self, end_t):
        return {s: b.window(None, end_t) for s, b in BARS.items()}

    def test_days_without_candles_do_not_count(self):
        start = T0 + 1500 * H4
        res = evaluate(self.store, self.cfg, "g", now_t=start + 60 * DAY,
                       bars_by_symbol=self.bars_until(start + 30 * DAY))
        days = next(c for c in res["checks"] if c["name"] == "min_days")
        self.assertAlmostEqual(days["value"], 30.0, places=4)
        self.assertFalse(days["passed"])
        self.assertIn("60.0 calendar days", days["note"])
        wall = evaluate(self.store, self.cfg, "g", now_t=start + 60 * DAY)
        self.assertEqual(next(c["value"] for c in wall["checks"] if c["name"] == "min_days"), 60.0)

    def test_edge_checks_are_added_with_candles(self):
        start = T0 + 1500 * H4
        for k in range(12):
            self.store.upsert_paper_trade("g", mk_trade("g", SYMS[k % 3], f"L{k}", start + 10 * k * H4,
                                                        (1.5, -1.0)[k % 2]), 100.0, 1.0)
        res = evaluate(self.store, self.cfg, "g", now_t=start + 90 * DAY,
                       bars_by_symbol=self.bars_until(start + 90 * DAY))
        checks = {c["name"]: c for c in res["checks"]}
        self.assertEqual((checks["min_edge_r"]["threshold"], checks["min_edge_z"]["threshold"]), (0.0, 1.65))
        self.assertIsInstance(checks["min_edge_z"]["value"], float)
        lab_res = self.lab.graduation("g", now_t=start + 90 * DAY)
        self.assertIn("min_edge_z", [c["name"] for c in lab_res["checks"]])
        self.assertNotIn("min_edge_z", [c["name"] for c in
                                        evaluate(self.store, self.cfg, "g", now_t=start + 90 * DAY)["checks"]])


# -- F33 / F48: compaction and report alerts ------------------------------------------------------------

class TestCompactAndAlerts(LabCase):
    def test_compact_keeps_every_gate_input(self):
        for k in range(40):
            self.store.record_trial(f"t{k}", 0.01 * k, n=100)
            self.store.add_experiment("backtest", SPEC.to_dict(), f"t{k}", T0, T0 + H4,
                                      {"metrics": {"n": 100, "avg_r": 0.1}, "per_symbol": {f"S{i}": {"n": i}
                                       for i in range(200)}, "rows": list(range(500))}, now_t=T0)
        eid = record_validation(self.store, SPEC, now_t=T0)
        self.lab.promote(SPEC, "deep", "x", now_t=1_700_000_000)
        self.store.add_journal("system", "old failure", "trace " * 2000, now_t=T0)
        stats, val = self.store.trial_stats(), self.store.get_experiment(eid)
        roster = self.store.roster(None)
        out = self.store.compact(now_t=T0 + 400 * DAY)
        self.assertEqual(self.store.trial_stats(), stats)
        self.assertEqual(self.store.get_experiment(eid), val)
        self.assertEqual(self.store.roster(None), roster)
        self.assertLess(out["bytes_after"], out["bytes_before"])
        self.assertGreaterEqual(out["experiments"], 40)
        row = self.store.list_experiments(limit=None, kind="backtest")[0]
        self.assertEqual((row["metrics"]["compacted"], row["metrics"]["metrics"]["avg_r"]), (True, 0.1))
        self.assertNotIn("per_symbol", row["metrics"])
        self.assertEqual(self.store.journal(kind="system")[-1]["body"], "")

    def test_stale_data_and_database_size_are_shown_first(self):
        record_validation(self.store, SPEC)
        self.lab.promote(SPEC, "deep", "x", now_t=1_700_000_000)
        self.store.set_kv("data_stall", {"since": 1_700_000_000, "reason": "all 3 candle updates failed"})
        with mock.patch("fibbot.report._db_bytes", return_value=60_000_000):
            text = render_status(self.lab, now_t=1_700_100_000)
            md = write_report(self.lab, now_t=1_700_100_000).read_text()
        for out in (text, md):
            self.assertIn("STALE DATA since 2023-11-14T22:13:20Z: all 3 candle updates failed", out)
            self.assertIn("STALE CANDLES: S0USDT 4h", out)
            self.assertIn("DATABASE 60 MB", out)
        self.assertLess(md.index("STALE DATA"), md.index("## Paper roster"))


# -- F13 / F35: the brain's stop callable and SDK compatibility ---------------------------------------------

try:
    import anthropic
    from tests.test_brain import FakeClient, FakeLab, cfg_, msg, text, tool_use
    from fibbot.brain import Brain, BrainUnavailable
    from fibbot.brain import client as client_mod
    from fibbot.brain.cycles import research_cycle, review_signals
    HAVE_SDK = True
except unittest.SkipTest:                                               # pragma: no cover
    HAVE_SDK = False


@unittest.skipUnless(HAVE_SDK, "anthropic SDK not installed")
class TestBrainStop(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cfg = cfg_(tmp.name)

    @staticmethod
    def stopper(after: int, reason="the cycle has run for brain.max_cycle_minutes (60 minutes)"):
        calls = []

        def stop():
            calls.append(1)
            return reason if len(calls) >= after else ""
        return stop, calls

    def test_run_agent_stops_and_wraps_up(self):
        script = [msg([tool_use("get_lab_status", {}, f"t{k}")], "tool_use") for k in range(10)]
        fc = FakeClient(scripts=[script, [msg([text("wrapped up")])]])
        stop, _ = self.stopper(3)
        lab = FakeLab(self.cfg)
        from fibbot.brain.tools import build_tools
        res = Brain(self.cfg, client=fc).run_agent("S", "p", build_tools(lab, "research"),
                                                   wrap_up="Stop here: {reason}.", stop=stop)
        self.assertEqual((res["turns"], res["stop_reason"]), (4, "stopped"))       # 3 + the wrap-up
        self.assertIn("max_cycle_minutes", res["stopped_by"])
        self.assertTrue(res["wrapped_up"])
        self.assertEqual(len(fc.runner_kwargs), 2)                                   # one wrap-up request
        self.assertTrue(fc.runners[0].closed)
        self.assertIn("Stop here: the cycle has run", json.dumps(fc.runner_kwargs[1]["messages"][-1]))

    def test_research_cycle_stopped_early_counts_as_run(self):
        script = [msg([tool_use("get_lab_status", {}, f"t{k}")], "tool_use") for k in range(10)]
        fc = FakeClient(scripts=[script, [msg([text("summary")])]])
        lab = FakeLab(self.cfg)
        stop, _ = self.stopper(3, "the bot is shutting down")
        res = research_cycle(lab, Brain(self.cfg, client=fc), stop=stop)
        self.assertTrue(res["ok"])
        self.assertEqual(res["stop_reason"], "stopped")
        self.assertIn("(stopped early: the bot is shutting down)", lab.store.journal(kind="research")[0]["title"])
        self.assertIsNotNone(lab.store.get_kv("last_research_t"))

    def test_review_signals_honour_stop(self):
        lab = FakeLab(self.cfg)
        lab.cfg = deep_merge(self.cfg, {"brain": {"signal_review": "shadow"}})
        brain = mock.Mock()
        lab.store.add_to_roster("s", "s", PRESETS["golden_pocket_4h"].to_dict(), 0, {}, "r")
        sig = {"strategy_id": "s", "symbol": "BTCUSDT", "leg_id": "L1", "tf": "4h", "t": 1}
        self.assertEqual(review_signals(lab, brain, signals=[sig], stop=lambda: "stop"), [])
        brain.structured.assert_not_called()

    def test_classify_without_credentials_error(self):
        err = anthropic.RateLimitError("429", response=mock.Mock(status_code=429, headers={}), body=None)
        with mock.patch.object(anthropic, "CredentialsError", create=False, new=None):
            self.assertEqual(client_mod._classify(err, anthropic), "transient")
        with mock.patch.dict(anthropic.__dict__):
            del anthropic.__dict__["CredentialsError"]
            self.assertEqual(client_mod._classify(err, anthropic), "transient")

    def test_an_incompatible_sdk_hands_research_to_the_autopilot(self):
        for breakage in ("runner", "beta_tool"):
            with self.subTest(breakage):
                lab = FakeLab(self.cfg)
                lab.universe = lambda: ["BTCUSDT"]
                lab.candles = mock.Mock(update_all=lambda symbols, tfs: {}, last_errors={})
                lab.paper = lambda: mock.Mock(tick=lambda **kw: {"strategies": {}, "new_signals": [],
                                                                 "errors": []})
                fc = FakeClient(scripts=[TypeError("unexpected keyword argument 'fallbacks'")])
                sweep = mock.Mock(return_value={"ranking": [{"preset": "x"}], "validations": [],
                                                "promoted": [], "errors": []})
                patches = [mock.patch("fibbot.autopilot.sweep", sweep),
                           mock.patch("fibbot.scheduler.write_report", return_value="REPORT.md")]
                if breakage == "beta_tool":
                    patches.append(mock.patch.object(anthropic, "beta_tool", None))
                with patches[0], patches[1], (patches[2] if len(patches) > 2 else mock.patch("time.sleep")):
                    out = scheduler.run_once(self.cfg, lab=lab, brain=Brain(self.cfg, client=fc))
                steps = {s["step"]: (s["status"], s["detail"]) for s in out["steps"]}
                self.assertEqual(steps["research"][0], "failed")
                self.assertIn("incompatible anthropic SDK", steps["research"][1])
                self.assertEqual(steps["autopilot"][0], "ok")
                sweep.assert_called_once()
                self.assertEqual(lab.store.journal(kind="research")[0]["title"],
                                 "Research cycle failed: brain unavailable")


# -- F14 / F23: the scheduler without a brain ----------------------------------------------------------------

from tests.test_scheduler import NOW, SchedulerCase, grad  # noqa: E402


class TestNoBrainScheduler(SchedulerCase):
    def test_a_failing_strategy_is_retired_without_a_brain(self):
        self.add()
        self.lab.grades["s1"] = grad("failing", dd=0.05)
        out = run_once_steps(self.cfg, self.lab, use_brain=False)
        self.assertEqual(out["graduation"][1]["retired"], ["s1"])
        self.assertEqual(self.store.get_strategy("s1")["status"], "retired")
        self.assertIn("failing", self.store.get_strategy("s1")["status_reason"])
        self.assertEqual(out["autopilot"][0], "ok")          # no longer "the roster is not empty"

    def test_with_a_brain_failing_strategies_are_left_to_it(self):
        self.add()
        self.lab.grades["s1"] = grad("failing", dd=0.05)
        with mock.patch("fibbot.scheduler.make_brain", return_value=FakeBrainOK()):
            out = run_once_steps(self.cfg, self.lab, use_brain=True)
        self.assertEqual(out["graduation"][1]["retired"], [])

    def test_research_is_deferred_when_the_weekly_cap_is_used(self):
        self.store.set_kv("last_research_t", NOW - 30 * 3600)               # due only by the daily cadence
        self.lab.weekly_validations_left = lambda now_t=None: 0
        with mock.patch("fibbot.scheduler.make_brain", return_value=FakeBrainOK()):
            out = run_once_steps(self.cfg, self.lab, use_brain=True)
        self.assertEqual(out["research"], ("skipped", "research deferred: weekly validation cap reached"))


class FakeBrainOK:
    def available(self):
        return True, "model x"


def run_once_steps(cfg, lab, use_brain):
    out = scheduler.run_once(cfg, now_t=NOW, lab=lab, use_brain=use_brain)
    return {s["step"]: (s["status"], s["detail"]) for s in out["steps"]}


# -- F40: the autopilot ranks on the binding edge ----------------------------------------------------------

class TestAutopilotBindingEdge(LabCase):
    def test_a_regime_edge_at_or_below_zero_is_not_eligible(self):
        from fibbot import autopilot
        name = "golden_pocket_4h"
        lab = StubLab(self.store, self.cfg, {name: (100, 0.2, 0.75, 0.3)})
        real = lab.baseline
        lab.baseline = lambda spec: {**real(spec), "edge_regime_r": -0.01}
        out = autopilot.sweep(lab, presets=[name])
        row, = out["ranking"]
        self.assertEqual((row["edge_r"], row["eligible"]), (-0.01, False))


# -- F16 / F42: CLI warning and tool wording ----------------------------------------------------------------

class TestWording(unittest.TestCase):
    def test_cli_warns_before_a_holdout_peek(self):
        err = io.StringIO()
        lab = mock.MagicMock()
        lab.__enter__ = lambda s: s
        with mock.patch("fibbot.cli.open_lab") as ol, redirect_stderr(err), redirect_stdout(io.StringIO()):
            ol.return_value.__enter__.return_value.backtest.return_value = {}
            args = mock.Mock(window="all", symbols=None, trades=0, json=True, preset="golden_pocket_4h",
                             spec=None, tf=None)
            cli.cmd_backtest(args, load_config(overrides={"home": tempfile.gettempdir()}))
        self.assertIn("counts as a trial and as a look at this spec's holdout", err.getvalue())

    @unittest.skipUnless(HAVE_SDK, "anthropic SDK not installed")
    def test_walk_forward_is_not_called_independent_evidence(self):
        from fibbot.brain.prompts import SYSTEM_PROMPT
        from fibbot.brain.tools import build_tools
        with tempfile.TemporaryDirectory() as tmp:
            tools = {t.name: t for t in build_tools(FakeLab(cfg_(tmp)), "research")}
        desc = tools["run_optimization"].to_dict()["description"]
        self.assertIn("part of the research window", desc)
        self.assertIn("not independent evidence", SYSTEM_PROMPT)
        self.assertIn("Retiring a strategy is permanent", SYSTEM_PROMPT)
        self.assertIn("min_stop_cost_mult", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()


class TestDesignListsEveryCliFlag(unittest.TestCase):
    def test_every_long_flag_is_in_the_cli_paragraph(self):
        """F54: DESIGN.md's CLI paragraph names every subcommand and every long option."""
        import argparse
        text = (Path(__file__).resolve().parent.parent / "fibbot" / "DESIGN.md").read_text()
        para = text[text.index("CLI: `python -m fibbot <command>`"):]
        parser = cli.build_parser()
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        for name, sp in sub.choices.items():
            self.assertIn(f"`{name}", para.replace("\n", " "), name)
            for action in sp._actions:
                for opt in action.option_strings:
                    if opt.startswith("--") and opt != "--help":
                        self.assertIn(opt, para, f"{name} {opt}")
