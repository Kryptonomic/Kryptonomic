"""Regression tests for the brain review findings (F10, F11, F12, F15, F17, F24, F26, F28, F29,
F47). No network and no Claude: fake SDK clients, or the real SDK over an in-memory transport."""
from __future__ import annotations

import json
import re
import time
import unittest
from collections import Counter
from unittest import mock

try:
    import anthropic
except ImportError:  # the SDK is optional
    raise unittest.SkipTest("anthropic SDK not installed")

from fibbot import scheduler
from fibbot.brain import Brain, BrainUnavailable, build_tools, review_signals
from fibbot.brain import client as client_mod
from fibbot.brain import cycles
from fibbot.brain.cycles import (MAX_SIGNAL_ATTEMPTS, research_cycle, review_cycle, review_quota,
                                 review_record_until)
from fibbot.brain.prompts import SYSTEM_PROMPT, WRAP_UP_NOTE, research_prompt, review_prompt
from fibbot.brain.tools import MAX_RESULT_CHARS, iso, to_json
from fibbot.strategy import PRESETS, StrategySpec
from fibbot.types import Trade
from tests.test_brain import (H4, FakeClient, FakeLab, TmpCase, api_error, api_message, cfg_,
                              make_bars, make_signal, msg, review, sdk_client, text)


def tu(uid: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": uid, "name": name, "input": inp}


def trade(sid: str, leg: str, entry_t: int, exit_t: int | None, r: float = -1.0) -> Trade:
    status = "open" if exit_t is None else "stop"
    return Trade(sid, "BTCUSDT", "4h", 1, leg, 0, entry_t, 100.0, 95.0, exit_t,
                 None if exit_t is None else 95.0, status, -0.05, 0.05, r, 2, {})


class CountingClient:
    """A FakeClient-like SDK stand-in whose beta.messages.create always answers with `reply`
    (or raises it) and records the (patched) time of each call."""

    def __init__(self, reply):
        from types import SimpleNamespace
        self.reply, self.calls, self.api_key = reply, [], "sk-test"
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append((time.time(), kw))
        r = self.reply(kw) if callable(self.reply) else self.reply
        if isinstance(r, BaseException):
            raise r
        return r


def leg_of(prompt: str) -> str:
    return json.loads(re.search(r"<signal>\n(.*?)\n</signal>", prompt, re.S).group(1))["leg_id"]


# -- F10: the signal-review quota counts calls, failed ones included --------------------------

class TestSignalQuota(TmpCase):
    def setUp(self):
        super().setUp()
        self.bars = make_bars("BTCUSDT", 300, seed=9)
        self.lab = FakeLab(cfg_(self.tmp.name, max_signal_reviews_per_day=3), {"BTCUSDT": self.bars})
        st = self.lab.store
        st.add_to_roster("live", "gp_live", PRESETS["golden_pocket_4h"].to_dict(), self.bars.t[200],
                         {"metrics": {"avg_r": 0.25}}, "r")
        self.sigs = [make_signal("live", "BTCUSDT", k, self.bars) for k in (280, 285, 290, 295)]
        for s in self.sigs:
            st.add_signal(s)

    def passes(self, client, hours: int) -> None:
        t0 = time.time()
        brain = Brain(self.lab.cfg, client=client)
        for h in range(hours):
            with mock.patch.object(cycles.time, "time", return_value=t0 + h * 3600):
                review_signals(self.lab, brain)

    def test_failed_replies_count_toward_the_daily_cap(self):
        for bad in (msg([text('{"decision": "ta')], "max_tokens"), msg([text("no")], "refusal")):
            with self.subTest(stop=bad.stop_reason):
                self.setUp()
                fc = CountingClient(bad)
                self.passes(fc, 96)                         # four days of hourly passes
                times = [t for t, _ in fc.calls]
                for t in times:   # never more than the cap within any rolling 24h
                    self.assertLessEqual(sum(t <= u < t + 86400 for u in times), 3)
                legs = Counter(leg_of(kw["messages"][0]["content"]) for _, kw in fc.calls)
                self.assertTrue(all(n <= MAX_SIGNAL_ATTEMPTS for n in legs.values()), legs)
                self.assertEqual(sum(legs.values()), 4 * MAX_SIGNAL_ATTEMPTS)   # then given up
                self.assertEqual(self.lab.store.count_signal_reviews_since(0), 0)

    def test_a_failing_signal_does_not_block_the_others(self):
        newest = self.sigs[-1].leg_id

        def reply(kw):
            if leg_of(kw["messages"][0]["content"]) == newest:
                return api_error(anthropic.BadRequestError, 400)
            return review("take")
        fc = CountingClient(reply)
        res = review_signals(self.lab, Brain(self.lab.cfg, client=fc))
        self.assertEqual(res[0]["leg_id"], newest)
        self.assertIn("review call failed: BadRequestError", res[0]["error"])
        self.assertEqual([r.get("decision") for r in res[1:]], ["take", "take"])   # same pass
        self.assertEqual(review_quota(self.lab.store, 3, int(time.time())), 0)     # failure counted
        j = self.lab.store.journal(kind="signal")[0]
        self.assertEqual(j["title"], "Signal reviews (shadow): 2 take, 0 skip, 1 failed")

    def test_unavailable_brain_does_not_use_up_signals(self):
        fc = CountingClient(api_error(anthropic.AuthenticationError, 401))
        self.passes(fc, 6)
        self.assertEqual(len(fc.calls), 6)                  # one probe per pass, then the batch stops
        fc = CountingClient(review("take"))
        res = review_signals(self.lab, Brain(self.lab.cfg, client=fc))
        self.assertEqual(len([r for r in res if "decision" in r]), 3)  # nothing was given up or counted

    def test_structured_max_tokens_over_http(self):
        reply = api_message([{"type": "text", "text": '{"decision":"ta'}], "max_tokens")
        requests = []
        brain = Brain(self.cfg, client=sdk_client([reply], requests))
        from fibbot.brain.prompts import SIGNAL_SCHEMA
        with self.assertLogs("fibbot.brain.client", "WARNING") as logs:
            self.assertIsNone(brain.structured("S", "p", SIGNAL_SCHEMA))
        self.assertIn("cut off at max_tokens", logs.output[0])
        self.assertEqual(brain.last_failure, "max_tokens")
        self.assertGreaterEqual(requests[0][1]["max_tokens"], 16000)
        self.assertEqual(brain.last_usage["output_tokens"], 7)


# -- F11: the agent loop streams, so max_tokens can be large; cut-off stops are labelled --------

class TestStreamingAgent(TmpCase):
    def test_large_max_tokens_streams(self):
        cfg = cfg_(self.tmp.name, max_tokens=64000)
        lab = FakeLab(cfg)
        responses = [api_message([tu("t1", "get_lab_status", {})], "tool_use"),
                     api_message([{"type": "text", "text": "done"}])]
        requests = []
        res = Brain(cfg, client=sdk_client(responses, requests)).run_agent(
            "S", "go", build_tools(lab, "research"))
        self.assertEqual((res["text"], res["turns"], res["stop_reason"]), ("done", 2, "end_turn"))
        self.assertEqual(res["usage"]["output_tokens"], 14)
        self.assertEqual([(b["stream"], b["max_tokens"]) for _, b in requests], [(True, 64000)] * 2)
        self.assertEqual([c for c in lab.calls if c[0] == "status"], [("status",)])

    def test_max_tokens_cut_tool_call_is_not_run(self):
        lab = FakeLab(self.cfg)
        lab._notebook = "# Brain notebook\n\nkeep me\n"
        cut = {**tu("t2", "rewrite_notebook", {}), "partial_json": '{"content": "# Notebook\\n- part'}
        responses = [api_message([tu("t1", "read_notebook", {})], "tool_use"),
                     api_message([cut], "max_tokens"),
                     api_message([{"type": "text", "text": "Summary after the cut."}])]
        requests = []
        res = research_cycle(lab, Brain(self.cfg, client=sdk_client(responses, requests)))
        self.assertTrue(res["ok"])
        self.assertNotIn("notebook_rewrite", [c[0] for c in lab.calls])
        self.assertEqual(lab.notebook(), "# Brain notebook\n\nkeep me\n")
        j = lab.store.journal(kind="research")[0]
        self.assertEqual(j["title"], "Research cycle: 0 promoted, 0 retired (stopped early: max_tokens)")
        self.assertTrue(j["body"].startswith("Summary after the cut."))
        wrap = requests[2][1]["messages"]
        self.assertFalse(any(b.get("name") == "rewrite_notebook" for m in wrap for b in m["content"]
                             if isinstance(b, dict)), "the cut-off reply is dropped")
        last = wrap[-1]["content"]
        self.assertEqual(last[0]["type"], "tool_result")          # tool results stay first
        self.assertIn("cut off at the output token limit", last[-1]["text"])
        self.assertIsNotNone(lab.store.get_kv("last_research_t"))

    def test_context_window_stop_is_labelled(self):
        lab = FakeLab(self.cfg)
        fc = FakeClient(scripts=[[msg([text("partial")], "model_context_window_exceeded")]])
        research_cycle(lab, Brain(self.cfg, client=fc))
        self.assertEqual(lab.store.journal(kind="research")[0]["title"],
                         "Research cycle: 0 promoted, 0 retired (stopped early: context window)")
        self.assertEqual(len(fc.runner_kwargs), 1)                # no wrap-up into a full context


# -- F12: exhausted credits / retired model mean "brain unavailable" ------------------------------

class TestAccountErrors(TmpCase):
    CASES = ((404, "not_found_error", "model: claude-x"), (402, "billing_error", "payment required"),
             (400, "invalid_request_error", "Your credit balance is too low to access the API."))

    def test_account_errors_are_unavailable_without_retry(self):
        from fibbot.brain.prompts import SIGNAL_SCHEMA
        for status, etype, message in self.CASES:
            with self.subTest(status=status):
                for call in (lambda b: b.structured("S", "p", SIGNAL_SCHEMA),
                             lambda b: b.run_agent("S", "p", [])):
                    requests = []
                    brain = Brain(self.cfg, client=sdk_client([status] * 3, requests,
                                                              {"type": etype, "message": message}))
                    with mock.patch.object(client_mod.time, "sleep") as sleep:
                        with self.assertRaisesRegex(BrainUnavailable, "credits exhausted or model"):
                            call(brain)
                    self.assertEqual(len(requests), 1)
                    sleep.assert_not_called()
        requests = []
        brain = Brain(self.cfg, client=sdk_client([400], requests,
                                                  {"type": "invalid_request_error", "message": "bad field"}))
        with self.assertRaises(anthropic.BadRequestError):
            brain.structured("S", "p", SIGNAL_SCHEMA)

    def _scheduler_lab(self):
        lab = FakeLab(self.cfg)
        lab.universe = lambda: ["BTCUSDT"]
        lab.candles = mock.Mock(update_all=lambda symbols, tfs: {}, last_errors={})
        lab.paper = lambda: mock.Mock(tick=lambda **kw: {"strategies": {}, "new_signals": [], "errors": []})
        lab.graduation = lambda sid, now_t=None: {"status": "collecting", "summary": "s", "checks": []}
        return lab

    def run_pass(self, lab):
        credit = {"type": "invalid_request_error", "message": "Your credit balance is too low"}
        brain = Brain(self.cfg, client=sdk_client([400] * 5, [], credit))
        sweep = mock.Mock(return_value={"ranking": [{"preset": "x"}], "validations": [],
                                        "promoted": [], "errors": []})
        with mock.patch("fibbot.autopilot.sweep", sweep), \
                mock.patch("fibbot.scheduler.write_report", return_value="REPORT.md"):
            out = scheduler.run_once(self.cfg, lab=lab, brain=brain)
        return {s["step"]: (s["status"], s["detail"]) for s in out["steps"]}, sweep

    def test_credit_exhaustion_hands_research_to_the_autopilot(self):
        lab = self._scheduler_lab()
        steps, sweep = self.run_pass(lab)
        self.assertEqual(steps["research"][0], "failed")
        self.assertIn("brain cycle failed", steps["research"][1])
        self.assertEqual(steps["autopilot"][0], "ok")
        sweep.assert_called_once()
        self.assertEqual(lab.store.journal(kind="research")[0]["title"],
                         "Research cycle failed: brain unavailable")
        self.assertEqual(lab.store.journal(kind="system"), [])

    def test_credit_exhaustion_in_signal_reviews(self):
        lab = self._scheduler_lab()
        bars = make_bars("BTCUSDT", 300, seed=9)
        lab._bars = {"BTCUSDT": bars}
        lab.store.add_to_roster("live", "gp", PRESETS["golden_pocket_4h"].to_dict(), bars.t[200], {}, "r")
        lab.store.add_signal(make_signal("live", "BTCUSDT", 295, bars))
        steps, _ = self.run_pass(lab)
        self.assertEqual(steps["signals"][0], "ok")
        self.assertEqual(steps["signals"][1]["failed"], 1)
        for step in ("review", "research"):
            self.assertEqual(steps[step][0], "skipped")
            self.assertIn("brain unavailable", steps[step][1])
        self.assertEqual(lab.store.journal(kind="system"), [])


# -- F15: the brain sees only pass/fail and headline numbers of a validation ---------------------

class ValidatingLab(FakeLab):
    def validate(self, spec):
        self._call("validate", spec)
        sid = spec.spec_id()
        metrics = {"n": 42, "avg_r": 0.21, "total_r": 8.8, "win_rate": 0.45, "profit_factor": 1.4,
                   "max_drawdown": 0.08, "sharpe_per_trade": 0.15, "n_open": 1}
        gate = {"passed": True, "deflated_sharpe": 0.93,
                "checks": [{"name": "min_trades", "value": 42, "threshold": 30, "passed": True}]}
        eid = self.store.add_experiment("validate", spec.to_dict(), sid, 0, H4,
                                        {**metrics, "passed": True, "checks": gate["checks"],
                                         "folds": [{"t": 1, "avg_r": 0.3}]})
        return {"spec_id": sid, "name": spec.name, "timeframe": spec.timeframe, "direction": "both",
                "period": {"start": "2026-01-01", "end": "2026-11-01"}, "n_symbols": 6,
                "metrics": metrics, "portfolio": {"final_equity": 11000.0},
                "per_symbol": {"ETHUSDT": {"n": 14, "total_r": 12.75}}, "per_symbol_omitted": 0,
                "per_year": {"2026": {"n": 42}}, "folds": [{"start": "2026-01-01", "avg_r": 0.26}],
                "fold_positive_frac": 0.75, "min_fold_avg_r": -0.2, "window": "validation",
                "passed": True, "gate": gate, "n_trials": 17, "var_sr": 0.01, "experiment_id": eid}


class TestValidationLeak(TmpCase):
    def setUp(self):
        super().setUp()
        self.lab = ValidatingLab(self.cfg)
        self.actions = []
        self.tools = {t.name: t for t in build_tools(self.lab, "research", self.actions)}

    def call(self, tool, **kw):
        return json.loads(self.tools[tool].call(kw))

    def test_validate_strategy_hides_the_breakdown(self):
        out = self.call("validate_strategy", spec={"preset": "golden_pocket_4h"})
        for k in ("per_symbol", "per_year", "folds", "portfolio", "fold_positive_frac",
                  "min_fold_avg_r", "per_symbol_omitted"):
            self.assertNotIn(k, out)
        self.assertTrue(out["passed"])
        self.assertEqual(out["gate"]["checks"][0]["name"], "min_trades")
        self.assertEqual((out["metrics"]["n"], out["metrics"]["avg_r"], out["metrics"]["profit_factor"]),
                         (42, 0.21, 1.4))
        self.assertEqual(out["n_trials"], 17)
        stored = self.lab.store.latest_validation(out["spec_id"])
        self.assertEqual(stored["id"], out["experiment_id"])
        self.assertIn("folds", stored["metrics"])                  # kept for humans / REPORT.md
        self.assertEqual(self.actions[-1]["passed"], True)

    def test_validation_results_cannot_be_ranked(self):
        from anthropic.lib.tools import ToolError
        st = self.lab.store
        st.add_experiment("validate", StrategySpec().to_dict(), "v1", 0, H4,
                          {"n": 40, "avg_r": 0.9, "fold_positive_frac": 0.75, "min_fold_avg_r": -0.1})
        st.add_experiment("backtest", StrategySpec().to_dict(), "b1", 0, H4, {"n": 40, "avg_r": 0.1})
        with self.assertRaisesRegex(ToolError, "cannot be ranked"):
            self.tools["list_experiments"].call({"kind": "validate", "sort_by": "avg_r"})
        ranked = self.call("list_experiments", sort_by="avg_r")
        self.assertEqual([r["kind"] for r in ranked], ["backtest"])
        recent = self.call("list_experiments", kind="validate")
        self.assertEqual(recent[0]["metrics"], {"n": 40, "avg_r": 0.9})   # headline kept for reviews


# -- F17: the signal prompt's review record is causal; traded legs are not reviewed --------------

class TestCausalSignalRecord(TmpCase):
    def setUp(self):
        super().setUp()
        self.bars = make_bars("BTCUSDT", 300, seed=9)
        self.lab = FakeLab(cfg_(self.tmp.name, max_signal_reviews_per_day=10), {"BTCUSDT": self.bars})
        st = self.lab.store
        for sid in ("live", "other"):
            st.add_to_roster(sid, sid, PRESETS["golden_pocket_4h"].to_dict(), self.bars.t[200], {}, "r")
        self.sig = make_signal("live", "BTCUSDT", 280, self.bars)
        st.add_signal(self.sig)

    def test_leg_with_a_paper_trade_is_not_reviewed(self):
        for exit_t in (self.sig.t + 2 * H4, None):         # closed after the signal, or still open
            with self.subTest(exit_t=exit_t):
                self.lab.store.upsert_paper_trade("live", trade("live", self.sig.leg_id, self.sig.t + H4,
                                                                exit_t), notional=1.0, pnl=-1.0)
                fc = FakeClient(replies=[review()])
                self.assertEqual(review_signals(self.lab, Brain(self.lab.cfg, client=fc)), [])
                self.assertEqual(fc.create_kwargs, [])
                self.assertEqual(self.lab.store.count_signal_reviews_since(0), 0)

    def test_review_record_only_counts_trades_closed_by_the_signal_bar(self):
        st, t = self.lab.store, self.sig.t
        st.upsert_paper_trade("other", trade("other", "A", t - 3 * H4, t - H4, 0.5), notional=1.0, pnl=1.0)
        st.upsert_paper_trade("other", trade("other", "B", t + H4, t + 2 * H4, -1.0), notional=1.0, pnl=-1.0)
        fc = FakeClient(replies=[review()])
        review_signals(self.lab, Brain(self.lab.cfg, client=fc))
        p = fc.create_kwargs[0]["messages"][0]["content"]
        stats = json.loads(re.search(r"<strategy_stats>\n(.*?)\n</strategy_stats>", p, re.S).group(1))
        rec = stats["your_review_record"]
        self.assertEqual((rec["unreviewed"]["n"], rec["unreviewed"]["total_r"]), (1, 0.5))
        self.assertEqual(rec["as_of"], iso(t))
        # The review-time record (all closed trades) would have shown the later -1R trade too.
        self.assertEqual(len(self.lab.store.paper_trades(status="closed")), 2)

    def test_review_record_until(self):
        st, t = self.lab.store, 10 * H4
        st.upsert_paper_trade("other", trade("other", "A", 0, 2 * H4, 1.0), notional=1.0, pnl=1.0)
        st.upsert_paper_trade("other", trade("other", "B", 0, 20 * H4, -1.0), notional=1.0, pnl=-1.0)
        sigs = [{"strategy_id": "other", "symbol": "BTCUSDT", "leg_id": "A", "t": 0, "decision": "skip",
                 "reviewed_t": H4},
                {"strategy_id": "other", "symbol": "BTCUSDT", "leg_id": "B", "t": 0, "decision": "take",
                 "reviewed_t": 15 * H4}]
        rec = review_record_until(st, sigs, st.paper_trades(status="closed"), t, "shadow")
        self.assertEqual((rec["skip"]["n"], rec["take"]["n"], rec["unreviewed"]["n"]), (1, 0, 0))
        self.assertEqual((rec["n_signals"], rec["n_signals_reviewed"]), (2, 1))   # B's review came later
        rec = review_record_until(st, sigs, st.paper_trades(status="closed"), 30 * H4, "shadow")
        self.assertEqual((rec["skip"]["n"], rec["take"]["n"], rec["reviewed"]), (1, 1, 2))


# -- F24: SDK retries, and a mid-cycle transient failure resumes the conversation ----------------

class TestResume(TmpCase):
    def test_client_is_built_with_retries(self):
        fake = mock.Mock(api_key="sk-test")
        with mock.patch.object(anthropic, "Anthropic", return_value=fake) as ctor:
            Brain(self.cfg).client
        self.assertGreaterEqual(ctor.call_args.kwargs["max_retries"], 5)
        with mock.patch.object(anthropic, "Anthropic", return_value=fake) as ctor:
            Brain(cfg_(self.tmp.name, max_retries=2)).client
        self.assertEqual(ctor.call_args.kwargs["max_retries"], 2)

    @mock.patch.object(client_mod.time, "sleep")
    def test_real_runner_resumes_after_a_529_and_a_mid_stream_error(self, sleep):
        lab = FakeLab(self.cfg)
        overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        responses = [api_message([tu("t1", "get_lab_status", {})], "tool_use"), 529,
                     api_message([overloaded]),
                     api_message([{"type": "text", "text": "done"}])]
        requests = []
        res = Brain(self.cfg, client=sdk_client(responses, requests)).run_agent(
            "S", "go", build_tools(lab, "research"))
        self.assertEqual((res["text"], res["turns"], res["resumes"]), ("done", 2, 2))
        self.assertEqual([c for c in lab.calls if c[0] == "status"], [("status",)])   # ran once
        for _, body in requests[1:]:
            results = [b for m in body["messages"] for b in m["content"]
                       if isinstance(b, dict) and b.get("type") == "tool_result"]
            self.assertEqual([r["tool_use_id"] for r in results], ["t1"])
        self.assertEqual(sleep.call_count, 2)


# -- F26: turn limit in the prompt, wrap-up turn, previous cycle carried forward -----------------

class TestWrapUp(TmpCase):
    def test_capped_cycle_gets_a_wrap_up_turn(self):
        cfg = cfg_(self.tmp.name, max_iterations=3)
        lab = FakeLab(cfg)
        spec = {"preset": "golden_pocket_4h"}
        responses = [api_message([tu("t1", "run_backtest", {"spec": spec})], "tool_use"),
                     api_message([tu("t2", "run_baseline", {"spec": spec})], "tool_use"),
                     api_message([tu("t3", "validate_strategy", {"spec": spec})], "tool_use"),
                     api_message([{"type": "text", "text": "Validated gp; next: shorts."},
                                  tu("w1", "append_notebook", {"entry": "gp validated, PF 1.4"}),
                                  tu("w2", "run_backtest", {"spec": spec})], "tool_use")]
        requests = []
        res = research_cycle(lab, Brain(cfg, client=sdk_client(responses, requests)))
        self.assertEqual(len(requests), 4)
        self.assertTrue(res["wrapped_up"])
        wrap = requests[3][1]
        self.assertEqual(len(wrap["tools"]), len(requests[0][1]["tools"]))   # same tools: cache hit
        last = wrap["messages"][-1]["content"]
        self.assertEqual(last[0]["tool_use_id"], "t3")        # the validation result is seen
        self.assertIn("you have used all 3 model turns", last[-1]["text"])
        self.assertIn("gp validated, PF 1.4", lab.notebook())
        self.assertEqual([c[0] for c in lab.calls].count("backtest"), 1)    # w2 did not run
        j = lab.store.journal(kind="research")[0]
        self.assertTrue(j["body"].startswith("Validated gp; next: shorts."))
        self.assertIn("(stopped early: iteration cap)", j["title"])

    def test_prompts_carry_turn_limit_and_previous_cycle(self):
        p = research_prompt("{}", "", max_turns=60, last_cycle="old title\nbody", first_cycle=False)
        self.assertIn("at most 60 model turns", p)
        self.assertIn("<previous_cycle>\nold title\nbody\n</previous_cycle>", p)
        self.assertNotIn("first cycle", p)
        self.assertIn("first cycle", review_prompt("{}", "", first_cycle=True))
        self.assertIn("{reason}", WRAP_UP_NOTE)
        lab = FakeLab(self.cfg)
        fc = FakeClient(scripts=[[msg([text("Summary one.")])], [msg([text("Summary two.")])]])
        brain = Brain(self.cfg, client=fc)
        research_cycle(lab, brain)
        first = fc.runner_kwargs[0]["messages"][0]["content"]
        self.assertIn("first cycle", first)
        self.assertIn(f"at most {self.cfg['brain']['max_iterations']} model turns", first)
        research_cycle(lab, brain)
        second = fc.runner_kwargs[1]["messages"][0]["content"]
        self.assertIn("Research cycle: 0 promoted, 0 retired", second)
        self.assertIn("Summary one.", second)
        self.assertNotIn("first cycle", second)

    def test_no_first_cycle_claim_once_experiments_exist(self):
        lab = FakeLab(self.cfg)
        lab.status = lambda: {"roster": [], "experiments": {"total": 12}, "budget": {}}
        fc = FakeClient(scripts=[[msg([text("ok")])]])
        research_cycle(lab, Brain(self.cfg, client=fc))
        self.assertNotIn("first cycle", fc.runner_kwargs[0]["messages"][0]["content"])


# -- F28: the system prompt teaches how to read results ------------------------------------------

class TestReadingResults(TmpCase):
    def test_system_prompt(self):
        self.assertIn("## Reading results", SYSTEM_PROMPT)
        self.assertRegex(SYSTEM_PROMPT, r'"oos" block and oos_by_fold are the honest estimate')
        self.assertIn("in-sample", SYSTEM_PROMPT)
        self.assertIn("baseline_z", SYSTEM_PROMPT)
        self.assertIn("always use the whole universe", SYSTEM_PROMPT)
        self.assertIn("strategies ever put on paper", SYSTEM_PROMPT)
        self.assertNotIn("both gates counts every spec ever tried", SYSTEM_PROMPT)
        self.assertIsNone(re.search(r"\b(19|20)\d\d\b|\d\d:\d\d", SYSTEM_PROMPT))

    def test_baseline_z(self):
        lab = FakeLab(self.cfg)
        lab.baseline = lambda spec, n_runs=10: {"spec_id": spec.spec_id(), "edge_r": 0.06,
                                                "baseline_avg_r_std": 0.03, "beats_frac": 0.8}
        tools = {t.name: t for t in build_tools(lab, "research")}
        self.assertAlmostEqual(json.loads(tools["run_baseline"].call({"spec": {}}))["baseline_z"], 2.0)
        lab.baseline = lambda spec, n_runs=10: {"edge_r": 0.06, "baseline_avg_r_std": 0.0}
        self.assertIsNone(json.loads(tools["run_baseline"].call({"spec": {}}))["baseline_z"])


# -- F29: list_experiments looks up a spec; listings are valid JSON; exact specs ------------------

class TestSpecLookup(TmpCase):
    def setUp(self):
        super().setUp()
        self.lab = FakeLab(self.cfg)
        self.tools = {t.name: t for t in build_tools(self.lab, "research")}

    def call(self, tool, **kw):
        return json.loads(self.tools[tool].call(kw))

    def test_lookup_finds_an_old_spec_with_its_status(self):
        st = self.lab.store
        target = PRESETS["golden_pocket_4h"].with_overrides(trend_ema=100)
        st.add_experiment("backtest", target.to_dict(), target.spec_id(), 0, H4, {"n": 30, "avg_r": -0.1},
                          notes="dead end")
        for i in range(320):
            s = StrategySpec().with_overrides(min_rr=1.0 + i / 1000)
            st.add_experiment("backtest", s.to_dict(), s.spec_id(), 0, H4, {"n": 50, "avg_r": 0.01 * i},
                              notes="x" * 300)
        out = self.call("list_experiments", spec={"preset": "golden_pocket_4h", "trend_ema": 100})
        self.assertEqual(out["spec_id"], target.spec_id())
        self.assertEqual([r["notes"] for r in out["experiments"]], ["dead end"])
        self.assertIsNone(out["latest_validation"])
        self.assertIsNone(out["roster"])
        by_params = self.call("list_experiments", params={"name": "golden_pocket_4h", "trend_ema": 100})
        self.assertEqual([r["spec_id"] for r in by_params], [target.spec_id()])
        listing = self.tools["list_experiments"].call({"limit": 50})
        self.assertLessEqual(len(listing), MAX_RESULT_CHARS)
        rows = json.loads(listing)                                   # valid JSON, whole rows
        self.assertGreater(rows[-1]["omitted"], 0)
        self.assertTrue(all("kind" in r for r in rows[:-1]))

    def test_promote_by_spec_id_keeps_exact_params(self):
        st = self.lab.store
        spec = StrategySpec().with_overrides(stop_atr_buffer=0.123456, entry_zone_lo=0.61803)
        sid = spec.spec_id()
        st.add_experiment("validate", spec.to_dict(), sid, 0, H4, {"n": 40, "passed": True, "checks": []})
        out = self.call("list_experiments", spec_id=sid)
        self.assertEqual(out["latest_validation"]["passed"], True)
        self.assertEqual(StrategySpec.from_dict(out["spec"]).spec_id(), sid)       # exact, not rounded
        row = self.call("list_experiments", kind="validate")[0]
        self.assertEqual(StrategySpec().with_overrides(**row["params"]).spec_id(), sid)
        promoted = self.call("promote_to_paper", spec_id=sid, name="exact", rationale="r")
        self.assertEqual(promoted["strategy_id"], sid)
        self.assertEqual(self.call("list_experiments", spec_id=sid)["roster"]["status"], "paper")
        from anthropic.lib.tools import ToolError
        with self.assertRaisesRegex(ToolError, "no validation on record"):
            self.tools["promote_to_paper"].call({"spec_id": "nope", "name": "x", "rationale": "r"})
        with self.assertRaisesRegex(ToolError, "spec or its spec_id"):
            self.tools["promote_to_paper"].call({"name": "x", "rationale": "r"})

    def test_to_json_keeps_spec_values_exact(self):
        out = json.loads(to_json({"spec": {"stop_atr_buffer": 0.123456}, "metrics": {"avg_r": 0.123456},
                                  "params": {"x": float("nan")}}))
        self.assertEqual(out, {"spec": {"stop_atr_buffer": 0.123456}, "metrics": {"avg_r": 0.1235},
                               "params": {"x": None}})


# -- F47: review cycles get their own cap and effort; usage tally and optional monthly cap -------

class TestCycleSpend(TmpCase):
    def test_review_cycle_limits(self):
        cfg = cfg_(self.tmp.name, max_iterations=60)
        fc = FakeClient(scripts=[[msg([text("r")])], [msg([text("x")])], [msg([text("y")])]])
        brain = Brain(cfg, client=fc)
        review_cycle(FakeLab(cfg), brain)
        research_cycle(FakeLab(cfg), brain)
        self.assertEqual((fc.runner_kwargs[0]["max_iterations"], fc.runner_kwargs[0]["output_config"]),
                         (cycles.REVIEW_MAX_ITERATIONS, {"effort": "medium"}))
        self.assertEqual((fc.runner_kwargs[1]["max_iterations"], fc.runner_kwargs[1]["output_config"]),
                         (60, {"effort": "high"}))
        cfg = cfg_(self.tmp.name, review_max_iterations=7, review_effort="low")
        review_cycle(FakeLab(cfg), Brain(cfg, client=fc))
        self.assertEqual((fc.runner_kwargs[2]["max_iterations"], fc.runner_kwargs[2]["output_config"]),
                         (7, {"effort": "low"}))

    def test_usage_tally_and_monthly_cap(self):
        cfg = cfg_(self.tmp.name, max_monthly_output_tokens=30)
        lab = FakeLab(cfg)
        fc = FakeClient(scripts=[[msg([text("a")], usage=(100, 25, 0, 0))],
                                 [msg([text("b")], usage=(100, 25, 0, 0))]])
        brain = Brain(cfg, client=fc)
        self.assertTrue(research_cycle(lab, brain)["ok"])
        month = time.strftime("%Y-%m", time.gmtime())
        self.assertEqual(lab.store.get_kv(cycles.USAGE_KEY)[month]["output_tokens"], 25)
        self.assertTrue(research_cycle(lab, brain)["ok"])          # 25 < 30: still allowed
        res = research_cycle(lab, brain)                            # 50 >= 30: blocked
        self.assertFalse(res["ok"])
        self.assertIn("max_monthly_output_tokens", res["error"])
        self.assertEqual(len(fc.runner_kwargs), 2)
        self.assertEqual(lab.store.journal(kind="research")[0]["title"],
                         "Research cycle skipped: brain spend cap reached")
        bars = make_bars("BTCUSDT", 300, seed=9)
        lab._bars = {"BTCUSDT": bars}
        lab.store.add_to_roster("live", "gp", PRESETS["golden_pocket_4h"].to_dict(), bars.t[200], {}, "r")
        lab.store.add_signal(make_signal("live", "BTCUSDT", 295, bars))
        rc = FakeClient(replies=[review()])
        self.assertEqual(review_signals(lab, Brain(cfg, client=rc)), [])
        self.assertEqual(rc.create_kwargs, [])


if __name__ == "__main__":
    unittest.main()
