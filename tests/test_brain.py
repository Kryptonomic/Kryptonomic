"""Tests for fibbot.brain (client, tools, prompts, cycles). No network: a fake SDK client object
drives the tool loop, and the real SDK runner is exercised through an in-memory HTTP transport."""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import anthropic
    import httpx2
except ImportError:  # the SDK is optional: the rest of fibbot (and its tests) run without it
    raise unittest.SkipTest("anthropic SDK not installed")

from fibbot.brain import Brain, BrainUnavailable, build_tools, review_signals
from fibbot.brain import client as client_mod
from fibbot.brain.cycles import (SIGNAL_MAX_AGE, research_cycle, review_cycle, signal_setup,
                                 signal_stats)
from fibbot.brain.prompts import (SIGNAL_SCHEMA, SIGNAL_SYSTEM, SYSTEM_PROMPT, research_prompt,
                                  review_prompt, signal_prompt)
from fibbot.brain.tools import (MAX_RESULT_CHARS, experiment_row, iso, market_context,
                                parse_spec, pick_metrics, spec_diff, to_json)
from fibbot.config import load_config
from fibbot.fib import fib_table
from fibbot.indicators import ema
from fibbot.store import Store
from fibbot.strategy import PRESETS, StrategySpec
from fibbot.types import TF_SECONDS, Bars, Signal, Trade

H4 = 14400
T_END = (int(time.time()) // H4) * H4 - H4          # open time of the newest closed 4h bar
FORBIDDEN = ("temperature", "top_p", "top_k", "budget_tokens", "tool_choice")


def make_bars(symbol: str, n: int, seed: int = 1, end_t: int = T_END) -> Bars:
    """Deterministic 4h random walk with alternating drift regimes (so swings and legs exist)."""
    rng = random.Random(seed)
    rows, price, t0 = [], 100.0, end_t - (n - 1) * H4
    for i in range(n):
        drift = 0.004 * math.sin(i / 40.0)
        o = price
        c = o * math.exp(drift + rng.gauss(0, 0.01))
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.004)))
        l = min(o, c) * (1 - abs(rng.gauss(0, 0.004)))
        rows.append((t0 + i * H4, o, h, l, c, 1000.0))
        price = c
    return Bars.from_rows(symbol, "4h", rows)


def cfg_(home: str, **brain) -> dict:
    return load_config(overrides={
        "home": home, "data": {"timeframes": ["4h"]},
        "research": {"workers": 1, "validation_months": 3, "min_trades": 5},
        "brain": {"max_iterations": 12, **brain}})


# -- fake SDK objects ----------------------------------------------------------------------------

def text(t: str):
    return SimpleNamespace(type="text", text=t)


def tool_use(name: str, inp: dict, uid: str = "tu"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=uid)


def msg(content, stop="end_turn", usage=(100, 20, 50, 7)):
    u = SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1],
                        cache_read_input_tokens=usage[2], cache_creation_input_tokens=usage[3])
    return SimpleNamespace(content=list(content), stop_reason=stop, usage=u)


_UNSET = object()


class FakeRunner:
    """Mimics BetaToolRunner: yields scripted messages and, after each tool_use reply, runs its
    tool_use blocks through the provided tools (exceptions become error results, as in the SDK).
    Like the SDK, generate_tool_call_response() runs them once and caches the tool_result message,
    so calling it inside the loop body does not run the tools twice."""

    def __init__(self, kw: dict, script: list):
        self.kw, self.script, self.results = kw, script, []
        self._last, self._cached, self.closed = None, _UNSET, False

    def __iter__(self):
        for step in self.script[:self.kw["max_iterations"]]:
            if isinstance(step, BaseException):
                raise step
            self._last, self._cached = step, _UNSET
            yield step
            if step.stop_reason == "tool_use":
                self.generate_tool_call_response()

    def generate_tool_call_response(self):
        if self._cached is _UNSET:
            blocks = []
            for b in self._last.content:
                if b.type != "tool_use":
                    continue
                try:
                    tool = next(t for t in self.kw["tools"] if getattr(t, "name", None) == b.name)
                    out = tool.call(b.input)
                    self.results.append((b.name, out, False))
                    blocks.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
                except Exception as e:
                    self.results.append((b.name, str(e), True))
                    blocks.append({"type": "tool_result", "tool_use_id": b.id, "content": str(e),
                                   "is_error": True})
            self._cached = {"role": "user", "content": blocks} if blocks else None
        return self._cached

    def close(self):
        self.closed = True


class FakeClient:
    """Stands in for anthropic.Anthropic: beta.messages.tool_runner / beta.messages.create."""

    def __init__(self, scripts=(), replies=()):
        self.scripts, self.replies = list(scripts), list(replies)
        self.runner_kwargs, self.create_kwargs, self.runners = [], [], []
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._runner, create=self._create))
        self.api_key = "sk-test"

    def _runner(self, **kw):
        self.runner_kwargs.append(kw)
        script = self.scripts.pop(0)
        if isinstance(script, BaseException):
            raise script
        r = FakeRunner(kw, script)
        self.runners.append(r)
        return r

    def _create(self, **kw):
        self.create_kwargs.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def api_error(cls, status: int):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    if cls is anthropic.APIConnectionError:
        return cls(request=req)
    return cls(f"status {status}", response=httpx2.Response(status, request=req), body=None)


def sse(events: list[dict]) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def sse_message(m: dict) -> str:
    """An api_message dict as the server-sent events of a streamed reply. A tool_use block with
    "partial_json" streams that raw (possibly cut-off) JSON instead of its input; a block
    {"type": "error", "error": {"type", "message"}} ends the stream with that error event (a
    mid-stream failure)."""
    usage = m["usage"]
    events = [{"type": "message_start", "message": {
        **{k: v for k, v in m.items() if k not in ("content", "stop_reason", "usage")},
        "content": [], "stop_reason": None,
        "usage": {**usage, "output_tokens": 1}}}]
    for i, b in enumerate(m["content"]):
        if b["type"] == "error":
            return sse(events + [{"type": "error", "error": b["error"]}])
        if b["type"] == "text":
            events += [{"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                       {"type": "content_block_delta", "index": i,
                        "delta": {"type": "text_delta", "text": b["text"]}}]
        else:
            partial = b.get("partial_json", json.dumps(b["input"]))
            events += [{"type": "content_block_start", "index": i, "content_block": {
                            "type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}},
                       {"type": "content_block_delta", "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": partial}}]
        events.append({"type": "content_block_stop", "index": i})
    events += [{"type": "message_delta", "delta": {"stop_reason": m["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]}},
               {"type": "message_stop"}]
    return sse(events)


def sdk_client(responses: list, requests: list, error: dict | None = None) -> anthropic.Anthropic:
    """A real SDK client whose HTTP layer replays `responses` (dicts = 200 JSON, or server-sent
    events when the request streams; ints = error status) and records each request (headers,
    JSON body). `error` is the {"type", "message"} body of error responses."""
    err = error or {"type": "api_error", "message": "x"}

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.headers, body))
        r = responses.pop(0)
        if isinstance(r, int):
            return httpx2.Response(r, json={"type": "error", "error": err})
        if body.get("stream"):
            return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                   content=sse_message(r).encode())
        return httpx2.Response(200, json=r)
    return anthropic.Anthropic(api_key="sk-test", base_url="http://anthropic.test", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def api_message(content: list, stop: str = "end_turn") -> dict:
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 5,
                      "cache_creation_input_tokens": 2}}


# -- fake lab ------------------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    pass


class FakeLab:
    """Implements the Lab methods the brain calls; records calls."""

    def __init__(self, cfg: dict, bars: dict[str, Bars] | None = None):
        self.cfg, self.store = cfg, Store(":memory:")
        self._bars = bars or {}
        self.calls, self.cycle, self._notebook, self.fail = [], None, "", {}

    def _call(self, name, *args):
        self.calls.append((name, *args))
        if name in self.fail:
            raise self.fail[name]

    def begin_cycle(self, kind):
        self._call("begin_cycle", kind)
        self.cycle = kind

    def end_cycle(self):
        self._call("end_cycle")
        self.cycle = None

    def status(self):
        self._call("status")
        return {"as_of": "2026-11-02T00:00:00Z", "cycle": self.cycle, "roster": [],
                "budget": {"backtest_units_left": 5, "validations_left": 1}, "odd": float("nan")}

    def bars(self, tf, symbols=None):
        return {s: b for s, b in self._bars.items() if b.tf == tf and (symbols is None or s in symbols)}

    def backtest(self, spec, window="research", symbols=None):
        self._call("backtest", spec, window, symbols)
        return {"spec_id": spec.spec_id(), "metrics": {"n": 40, "avg_r": 0.212345678},
                "experiment_id": 1}

    def optimize(self, base, grid):
        self._call("optimize", base, grid)
        return {"n_combos": 2, "best": {"spec_id": base.spec_id()}}

    def baseline(self, spec, n_runs=10):
        self._call("baseline", spec)
        return {"spec_id": spec.spec_id(), "edge_r": 0.1}

    def robustness(self, spec):
        self._call("robustness", spec)
        return {"spec_id": spec.spec_id(), "positive_frac": 0.8}

    def validate(self, spec):
        self._call("validate", spec)
        return {"spec_id": spec.spec_id(), "passed": True, "experiment_id": 2}

    def promote(self, spec, name, rationale, now_t=None):
        self._call("promote", spec, name, rationale)
        sid = spec.spec_id()
        self.store.add_to_roster(sid, name, spec.to_dict(), T_END, {"avg_r": 0.3}, rationale)
        return {"strategy_id": sid, "name": name}

    def retire(self, strategy_id, reason):
        self._call("retire", strategy_id, reason)
        self.store.set_strategy_status(strategy_id, "retired", reason)
        return {"strategy_id": strategy_id, "status": "retired"}

    def signal_review_stats(self):
        self._call("signal_review_stats")
        return {"reviewed": 0, "skip_minus_take_avg_r": None}

    def notebook(self):
        return self._notebook

    def notebook_append(self, text):
        self._call("notebook_append", text)
        self._notebook += f"\n## entry\n{text}\n"
        return {"notebook_chars": len(self._notebook), "limit": 24000,
                "free_chars": 24000 - len(self._notebook), "trimmed": [], "truncated": False}

    def notebook_rewrite(self, text):
        self._call("notebook_rewrite", text)
        self._notebook = text
        return {"notebook_chars": len(text), "limit": 24000, "free_chars": 24000 - len(text)}

    def windows(self, tf="4h"):
        """Research window ends at the newest cached bar minus 30 bars (validation after it)."""
        ts = [b.t[-1] for b in self._bars.values() if b.tf == tf and len(b)]
        vs = (max(ts) if ts else T_END) - 30 * TF_SECONDS[tf]
        return {"validation_start": vs, "research_end": vs}


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = cfg_(self.tmp.name)


# -- prompts -------------------------------------------------------------------------------------

class TestPrompts(unittest.TestCase):
    def test_system_prompt_is_stable_and_covers_the_method(self):
        self.assertIsNone(re.search(r"\b(19|20)\d\d\b|\d\d:\d\d", SYSTEM_PROMPT), "no dates/times")
        for phrase in ("0.618-0.65", "0.62-0.79", "deflated Sharpe", "random-entry baseline",
                       "hypothesis", "notebook", '"failing"', "diverse", "plain-text summary",
                       "enforced by code", "Paper results outrank backtests", "single best grid cell"):
            self.assertIn(phrase, SYSTEM_PROMPT)

    def test_cycle_prompts_embed_status_and_notebook(self):
        for fn, word in ((research_prompt, "RESEARCH"), (review_prompt, "REVIEW")):
            p = fn('{"roster":[]}', "## old note\nkeep going")
            self.assertIn(word, p)
            self.assertIn('<lab_status>\n{"roster":[]}\n</lab_status>', p)
            self.assertIn("## old note\nkeep going", p)
            self.assertIn("first cycle", fn("{}", "   "))
        self.assertIn("no optimization", review_prompt("{}", ""))
        self.assertNotIn("no optimization", research_prompt("{}", ""))

    def test_signal_schema_and_prompt(self):
        self.assertFalse(SIGNAL_SCHEMA["additionalProperties"])
        self.assertEqual(set(SIGNAL_SCHEMA["required"]), {"decision", "confidence", "reason"})
        self.assertEqual(SIGNAL_SCHEMA["properties"]["decision"]["enum"], ["take", "skip"])
        self.assertIn("measured against simply taking every signal", SIGNAL_SYSTEM)
        self.assertIn('Default to "take"', SIGNAL_SYSTEM)
        self.assertIsNone(re.search(r"\b(19|20)\d\d\b|\d\d:\d\d", SIGNAL_SYSTEM))
        p = signal_prompt({"symbol": "BTCUSDT"}, {"setup": {"rr": 2.0}}, {"n": 3})
        blocks = dict(re.findall(r"<(\w+)>\n(.*?)\n</\1>", p, re.S))
        self.assertEqual(json.loads(blocks["signal"]), {"symbol": "BTCUSDT"})
        self.assertEqual(json.loads(blocks["market_context"]), {"setup": {"rr": 2.0}})
        self.assertEqual(json.loads(blocks["strategy_stats"]), {"n": 3})
        self.assertIn("benchmark is taking every signal", p)


# -- pure helpers --------------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_to_json_rounds_and_truncates(self):
        s = to_json({"a": 0.123456789, "b": float("nan"), "c": (1.0, 2)})
        self.assertEqual(json.loads(s), {"a": 0.1235, "b": None, "c": [1.0, 2]})
        big = to_json({"x": "y" * 50000})
        self.assertLessEqual(len(big), MAX_RESULT_CHARS)
        self.assertIn("[truncated", big)
        self.assertTrue(19900 < len(to_json({"x": "y" * 50000}, limit=20000)) <= 20000)

    def test_parse_spec(self):
        s = parse_spec({"preset": "ote_confirm_1h", "min_rr": 2})
        self.assertEqual((s.name, s.entry_mode, s.min_rr), ("ote_confirm_1h", "confirm", 2.0))
        self.assertEqual(parse_spec({}).spec_id(), StrategySpec().spec_id())
        with self.assertRaisesRegex(ValueError, "unknown preset"):
            parse_spec({"preset": "nope"})
        with self.assertRaisesRegex(ValueError, "unknown spec keys"):
            parse_spec({"bogus": 1})
        with self.assertRaisesRegex(ValueError, "stop_ratio"):
            parse_spec({"entry_zone_hi": 0.886, "stop_ratio": 0.8})

    def test_spec_diff_and_pick_metrics(self):
        self.assertEqual(spec_diff(PRESETS["golden_pocket_1h"].to_dict()), {"timeframe": "1h"})
        self.assertEqual(spec_diff(None), {})
        m = {"n": 3, "folds": [1, 2], "gate": {"passed": False, "n": 99}, "metrics": {"avg_r": 0.2}}
        self.assertEqual(pick_metrics(m), {"n": 3, "passed": False, "avg_r": 0.2})
        self.assertEqual(pick_metrics(None), {})

    def test_experiment_row_is_compact(self):
        row = {"id": 4, "kind": "backtest", "created_t": 0, "spec_id": "abc",
               "spec": PRESETS["deep_786_4h"].to_dict(), "window_start": 0, "window_end": H4,
               "metrics": {"n": 31, "avg_r": 0.1, "per_symbol": {"X": {}}}, "source": "brain",
               "notes": "n" * 500}
        r = experiment_row(row)
        self.assertEqual(r["name"], "deep_786_4h")
        self.assertEqual(r["window"], ["1970-01-01T00:00Z", "1970-01-01T04:00Z"])
        self.assertEqual(r["metrics"], {"n": 31, "avg_r": 0.1})
        self.assertNotIn("timeframe", r["params"])
        self.assertEqual(len(r["notes"]), 200)

    def test_iso(self):
        self.assertEqual(iso(86400 + 3600 * 5 + 60), "1970-01-02T05:01Z")
        self.assertIsNone(iso(None))


class TestMarketContext(unittest.TestCase):
    bars = make_bars("BTCUSDT", 700, seed=3)

    def test_fields(self):
        b = self.bars
        ctx = market_context(b)
        json.dumps(ctx, allow_nan=False)
        self.assertEqual((ctx["symbol"], ctx["timeframe"], ctx["n_bars"]), ("BTCUSDT", "4h", 700))
        self.assertEqual(ctx["t"], iso(b.t[-1]))
        self.assertAlmostEqual(ctx["returns"]["1_bars"], b.c[-1] / b.c[-2] - 1, places=3)
        self.assertAlmostEqual(ctx["returns"]["30_bars"], b.c[-1] / b.c[-31] - 1, places=3)
        self.assertAlmostEqual(ctx["close_vs_ema200"], b.c[-1] / ema(b.c, 200)[-1] - 1, places=3)
        self.assertEqual(len(ctx["closes"]), 30)
        self.assertEqual(ctx["range"]["high"], round(max(b.h[-30:]), 4))
        self.assertTrue(1 <= len(ctx["pivots"]) <= 4)
        kinds = [p["kind"] for p in ctx["pivots"]]
        self.assertTrue(all(a != b_ for a, b_ in zip(kinds, kinds[1:])), "pivots alternate")
        leg = ctx["leg"]
        self.assertEqual(leg["direction"], "up" if kinds[-1] == "L" else "down")
        self.assertEqual(set(leg["levels"]), set(fib_table(1.0, 2.0)))
        lo, hi = sorted((leg["anchor"], leg["extreme"]))
        self.assertTrue(lo <= leg["levels"]["0.618"] <= hi)

    def test_causal_cut(self):
        b, k = self.bars, 500
        self.assertEqual(market_context(b, upto_t=b.t[k]), market_context(b.slice(0, k + 1)))
        self.assertEqual(market_context(b, upto_t=b.t[k])["t"], iso(b.t[k]))

    def test_short_input(self):
        self.assertEqual(market_context(self.bars.slice(0, 1))["error"], "not enough bars")
        ctx = market_context(self.bars.slice(0, 20))
        self.assertIsNone(ctx["close_vs_ema200"])
        self.assertIsNone(ctx["returns"]["30_bars"])


# -- client --------------------------------------------------------------------------------------

class TestBrainClient(TmpCase):
    def test_available(self):
        off = Brain(cfg_(self.tmp.name, enabled=False), client=FakeClient())
        self.assertEqual(off.available()[0], False)
        self.assertTrue(Brain(self.cfg, client=FakeClient()).available()[0])
        nocreds = SimpleNamespace(api_key=None, auth_token=None, credentials=None)
        with mock.patch.object(anthropic, "Anthropic", return_value=nocreds):
            ok, why = Brain(self.cfg).available()
        self.assertFalse(ok)
        self.assertIn("ANTHROPIC_API_KEY", why)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            ok, why = Brain(self.cfg).available()
        self.assertTrue(ok)
        self.assertIn("claude-opus-5-5", why)

    def test_sdk_missing(self):
        with mock.patch.dict(sys.modules, {"anthropic": None}):
            ok, why = Brain(self.cfg).available()
        self.assertFalse(ok)
        self.assertIn("pip install", why)

    def test_run_agent_request_shape(self):
        fc = FakeClient(scripts=[[msg([text("done")])]] * 3)
        brain = Brain(self.cfg, client=fc)
        brain.run_agent("SYS", "hello", ["t1"])
        kw = fc.runner_kwargs[0]
        self.assertEqual(kw["model"], "claude-opus-5-5")
        self.assertEqual(kw["max_tokens"], self.cfg["brain"]["max_tokens"])
        self.assertIs(kw["stream"], True)       # streamed, so max_tokens may exceed ~21k
        self.assertEqual(kw["system"], [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(kw["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(kw["tools"], ["t1"])
        self.assertEqual(kw["max_iterations"], 12)
        self.assertEqual(kw["thinking"], {"type": "adaptive"})
        self.assertEqual(kw["output_config"], {"effort": "high"})
        self.assertEqual(kw["cache_control"], {"type": "ephemeral"})
        self.assertEqual(kw["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(kw["fallbacks"], "default")
        for k in FORBIDDEN:
            self.assertNotIn(k, kw)
        brain.run_agent("SYS", "hello", [], max_iterations=3, effort="max")
        self.assertEqual((fc.runner_kwargs[1]["max_iterations"], fc.runner_kwargs[1]["output_config"]),
                         (3, {"effort": "max"}))
        Brain(cfg_(self.tmp.name, fallbacks=None), client=fc).run_agent("S", "p", [])
        self.assertNotIn("betas", fc.runner_kwargs[2])
        self.assertNotIn("fallbacks", fc.runner_kwargs[2])

    def test_list_fallbacks_use_their_own_beta(self):
        fc = FakeClient(scripts=[[msg([text("x")])]])
        fb = [{"model": "claude-opus-4-8"}]
        Brain(cfg_(self.tmp.name, fallbacks=fb), client=fc).run_agent("S", "p", [])
        self.assertEqual(fc.runner_kwargs[0]["betas"], ["server-side-fallback-2026-06-01"])
        self.assertEqual(fc.runner_kwargs[0]["fallbacks"], fb)

    def test_run_agent_collects_usage_and_final_text(self):
        script = [msg([text("thinking aloud"), tool_use("x", {})], "tool_use", (10, 1, 0, 3)),
                  msg([text("Final summary."), text("Next: more.")], "end_turn", (20, 2, 8, None))]
        res = Brain(self.cfg, client=FakeClient(scripts=[script])).run_agent("S", "p", [])
        self.assertEqual(res, {"text": "Final summary.\nNext: more.", "stop_reason": "end_turn",
                               "turns": 2, "refused": False, "resumes": 0, "wrapped_up": False,
                               "usage": {"input_tokens": 30, "output_tokens": 3,
                                         "cache_read_input_tokens": 8, "cache_creation_input_tokens": 3}})

    def test_refusal_is_not_read(self):
        script = [msg([text("partial")], "refusal")]
        res = Brain(self.cfg, client=FakeClient(scripts=[script])).run_agent("S", "p", [])
        self.assertTrue(res["refused"])
        self.assertEqual(res["text"], "")

    @mock.patch.object(client_mod.time, "sleep")
    def test_transient_error_retried_before_any_reply(self, sleep):
        fc = FakeClient(scripts=[[api_error(anthropic.RateLimitError, 429)], [msg([text("ok")])]])
        res = Brain(self.cfg, client=fc).run_agent("S", "p", [])
        self.assertEqual((res["text"], res["resumes"]), ("ok", 1))
        sleep.assert_called_once_with(client_mod.RETRY_SLEEP)
        self.assertEqual(fc.runner_kwargs[1]["messages"], [{"role": "user", "content": "p"}])
        # Persistent failure: every resume (with its backoff) fails, then BrainUnavailable.
        n = len(client_mod.RESUME_SLEEPS) + 1
        fc = FakeClient(scripts=[[api_error(anthropic.InternalServerError, 529)],
                                 [api_error(anthropic.APIConnectionError, 0)]] * n)
        sleep.reset_mock()
        with self.assertRaisesRegex(BrainUnavailable, "after retrying"):
            Brain(self.cfg, client=fc).run_agent("S", "p", [])
        self.assertEqual(len(fc.runner_kwargs), n)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], list(client_mod.RESUME_SLEEPS))

    @mock.patch.object(client_mod.time, "sleep")
    def test_resume_after_tools_ran_keeps_the_conversation(self, sleep):
        tool = CountingTool("x")
        script = [msg([text("checking"), tool_use("x", {"a": 1}, "u1")], "tool_use", (10, 1, 0, 0)),
                  api_error(anthropic.InternalServerError, 500)]
        fc = FakeClient(scripts=[script, [msg([text("done")], "end_turn", (20, 2, 0, 0))]])
        res = Brain(self.cfg, client=fc).run_agent("S", "p", [tool])
        self.assertEqual((res["text"], res["turns"], res["resumes"]), ("done", 2, 1))
        self.assertEqual(res["usage"]["input_tokens"], 30)       # cumulative over both runners
        self.assertEqual(tool.calls, [{"a": 1}])                 # the tool ran once, not again
        resumed = fc.runner_kwargs[1]
        self.assertEqual(resumed["max_iterations"], 11)           # the cap counts earlier turns
        self.assertEqual([m["role"] for m in resumed["messages"]], ["user", "assistant", "user"])
        self.assertEqual(resumed["messages"][2]["content"][0]["tool_use_id"], "u1")
        sleep.assert_called_once_with(client_mod.RESUME_SLEEPS[0])

    @mock.patch.object(client_mod.time, "sleep")
    def test_auth_and_client_errors(self, sleep):
        for cls, code in ((anthropic.AuthenticationError, 401), (anthropic.PermissionDeniedError, 403)):
            fc = FakeClient(scripts=[[api_error(cls, code)]])
            with self.assertRaisesRegex(BrainUnavailable, "ANTHROPIC_API_KEY"):
                Brain(self.cfg, client=fc).run_agent("S", "p", [])
        fc = FakeClient(scripts=[[api_error(anthropic.BadRequestError, 400)]])
        with self.assertRaises(anthropic.BadRequestError):
            Brain(self.cfg, client=fc).run_agent("S", "p", [])
        sleep.assert_not_called()

    def test_structured(self):
        fc = FakeClient(replies=[msg([text('{"decision": "skip", "confidence": 0.7, "reason": "r"}')]),
                                 msg([text("ignored")], "refusal"),
                                 msg([text('{"decision": "ta')], "max_tokens"),
                                 msg([text('["take"]')])])
        brain = Brain(self.cfg, client=fc)
        self.assertEqual(brain.structured("S", "p", SIGNAL_SCHEMA),
                         {"decision": "skip", "confidence": 0.7, "reason": "r"})
        kw = fc.create_kwargs[0]
        self.assertEqual(kw["output_config"], {"effort": "medium",
                                               "format": {"type": "json_schema", "schema": SIGNAL_SCHEMA}})
        # Thinking counts toward max_tokens, so the default leaves room for it (was 4000).
        self.assertEqual((kw["max_tokens"], kw["thinking"], kw["fallbacks"]),
                         (client_mod.SIGNAL_MAX_TOKENS, {"type": "adaptive"}, "default"))
        self.assertGreaterEqual(client_mod.SIGNAL_MAX_TOKENS, 16000)
        for k in FORBIDDEN + ("tools", "cache_control", "stream"):
            self.assertNotIn(k, kw)
        self.assertIsNone(brain.structured("S", "p", SIGNAL_SCHEMA))
        self.assertEqual(brain.last_failure, "refusal")
        with self.assertLogs("fibbot.brain.client", "WARNING") as logs:
            self.assertIsNone(brain.structured("S", "p", SIGNAL_SCHEMA, effort="low"))
        self.assertIn("cut off at max_tokens", logs.output[0])
        self.assertEqual(brain.last_failure, "max_tokens")
        self.assertEqual(fc.create_kwargs[2]["output_config"]["effort"], "low")
        self.assertIsNone(brain.structured("S", "p", SIGNAL_SCHEMA), "non-object JSON")
        self.assertEqual(brain.last_failure, "not an object")


class CountingTool:
    """A minimal runnable tool for FakeRunner: records its inputs."""

    def __init__(self, name: str, out: str = '{"ok":true}'):
        self.name, self.out, self.calls = name, out, []

    def call(self, inp):
        self.calls.append(inp)
        return self.out


class TestRealSdkWiring(TmpCase):
    """The real SDK tool runner and request serialization, over an in-memory HTTP transport."""

    def test_tool_runner_end_to_end(self):
        lab = FakeLab(self.cfg)
        actions = []
        tools = build_tools(lab, "research", actions)
        responses = [
            api_message([{"type": "tool_use", "id": "t1", "name": "get_lab_status", "input": {}},
                         {"type": "tool_use", "id": "t2", "name": "run_backtest",
                          "input": {"spec": {"preset": "golden_pocket_4h", "bogus": 1}}},
                         {"type": "tool_use", "id": "t3", "name": "run_backtest",
                          "input": {"spec": {"preset": "golden_pocket_4h"}, "symbols": ["btcusdt"]}}],
                        "tool_use"),
            api_message([{"type": "text", "text": "Tested golden pocket."}]),
        ]
        requests = []
        brain = Brain(self.cfg, client=sdk_client(responses, requests))
        res = brain.run_agent("SYSTEM", "go", tools)
        self.assertEqual((res["text"], res["turns"], res["stop_reason"]), ("Tested golden pocket.", 2, "end_turn"))
        self.assertEqual(res["usage"]["cache_read_input_tokens"], 10)
        headers, body = requests[0]
        self.assertIn("server-side-fallback-2026-07-01", headers["anthropic-beta"])
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertIs(body["stream"], True)
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "high"})
        self.assertEqual(body["fallbacks"], "default")
        self.assertEqual(body["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
        for k in FORBIDDEN:
            self.assertNotIn(k, body)
        self.assertNotIn("max_iterations", body)
        self.assertEqual(len(body["tools"]), 16)
        results = {b["tool_use_id"]: b for m in requests[1][1]["messages"] for b in m["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_result"}
        self.assertEqual(json.loads(results["t1"]["content"])["budget"]["validations_left"], 1)
        self.assertIsNone(json.loads(results["t1"]["content"])["odd"])
        self.assertTrue(results["t2"]["is_error"])
        self.assertIn("ValueError: unknown spec keys: bogus", json.dumps(results["t2"]["content"]))
        self.assertNotIn("is_error", results["t3"])
        self.assertEqual(lab.calls[-1][2:], ("research", ["BTCUSDT"]))
        self.assertEqual([(a["tool"], a["ok"]) for a in actions],
                         [("get_lab_status", True), ("run_backtest", False), ("run_backtest", True)])

    def test_structured_and_errors_over_http(self):
        reply = api_message([{"type": "text", "text": '{"decision":"take","confidence":0.6,"reason":"ok"}'}])
        requests = []
        brain = Brain(self.cfg, client=sdk_client([reply], requests))
        self.assertEqual(brain.structured("S", "p", SIGNAL_SCHEMA)["decision"], "take")
        body = requests[0][1]
        self.assertEqual(body["output_config"]["format"]["schema"], SIGNAL_SCHEMA)
        self.assertEqual(body["output_config"]["effort"], "medium")
        for k in FORBIDDEN:
            self.assertNotIn(k, body)
        with self.assertRaisesRegex(BrainUnavailable, "credentials"):
            Brain(self.cfg, client=sdk_client([401], [])).structured("S", "p", SIGNAL_SCHEMA)
        with mock.patch.object(client_mod.time, "sleep") as sleep:
            out = Brain(self.cfg, client=sdk_client([529, reply], [])).structured("S", "p", SIGNAL_SCHEMA)
            self.assertEqual(out["reason"], "ok")
            sleep.assert_called_once()
            with self.assertRaises(BrainUnavailable):
                Brain(self.cfg, client=sdk_client([429, 503], [])).structured("S", "p", SIGNAL_SCHEMA)


# -- tools ---------------------------------------------------------------------------------------

class TestTools(TmpCase):
    def setUp(self):
        super().setUp()
        self.lab = FakeLab(self.cfg, {"BTCUSDT": make_bars("BTCUSDT", 400, seed=5)})
        self.actions = []
        self.tools = {t.name: t for t in build_tools(self.lab, "research", self.actions)}

    def call(self, tool, **kw):
        return json.loads(self.tools[tool].call(kw))

    def test_tool_sets(self):
        names = list(self.tools)
        self.assertEqual(len(names), 16)
        review = [t.name for t in build_tools(self.lab, "review")]
        self.assertEqual(review, [n for n in names if n != "run_optimization"])
        with self.assertRaises(ValueError):
            build_tools(self.lab, "sweep")

    def test_descriptions_are_complete(self):
        for t in self.tools.values():
            d = t.to_dict()
            first = d["description"].split("\n\n")[0]
            self.assertTrue(first.endswith("."), f"{t.name}: summary must be one full sentence")
            self.assertNotIn("\n", first, t.name)
            for p, schema in d["input_schema"]["properties"].items():
                self.assertTrue(schema.get("description"), f"{t.name}.{p} lacks a description")

    def test_research_tools_route_to_lab(self):
        out = self.call("run_backtest", spec={"preset": "golden_pocket_4h", "min_rr": 2.0})
        self.assertEqual(out["metrics"]["avg_r"], 0.2123)
        spec = self.lab.calls[-1][1]
        self.assertIsInstance(spec, StrategySpec)
        self.assertEqual(spec.min_rr, 2.0)
        self.call("run_optimization", base_spec={}, grid={"min_rr": [1.0, 2.0]})
        self.assertEqual(self.lab.calls[-1][2], {"min_rr": [1.0, 2.0]})
        self.assertEqual(self.call("run_baseline", spec={})["edge_r"], 0.1)
        self.assertEqual(self.call("run_robustness", spec={})["positive_frac"], 0.8)
        self.assertTrue(self.call("validate_strategy", spec={})["passed"])
        sid = StrategySpec().spec_id()
        self.call("promote_to_paper", spec={}, name="gp", rationale="robust")
        self.assertEqual(self.lab.store.get_strategy(sid)["name"], "gp")
        self.call("retire_strategy", strategy_id=sid, reason="failing")
        self.assertEqual(self.lab.store.get_strategy(sid)["status"], "retired")
        self.assertEqual([a["tool"] for a in self.actions][-4:],
                         ["run_robustness", "validate_strategy", "promote_to_paper", "retire_strategy"])
        val, promo, retire = self.actions[-3:]
        self.assertEqual((val["spec_id"], val["passed"]), (sid, True))
        self.assertEqual((promo["name"], promo["strategy_id"]), ("gp", sid))
        self.assertEqual((retire["strategy_id"], retire["reason"]), (sid, "failing"))

    def test_errors_become_tool_errors(self):
        from anthropic.lib.tools import ToolError
        with self.assertRaisesRegex(ToolError, "only the research window"):
            self.tools["run_backtest"].call({"spec": {}, "window": "validation"})
        self.lab.fail["baseline"] = BudgetExceeded("research cycle budget: 0 left")
        with self.assertRaisesRegex(ToolError, r"^BudgetExceeded: research cycle budget"):
            self.tools["run_baseline"].call({"spec": {}})
        with self.assertRaisesRegex(ToolError, "unknown strategy_id"):
            self.tools["get_paper_trades"].call({"strategy_id": "nope"})
        with self.assertRaisesRegex(ToolError, "no cached 4h candles for ETHUSDT"):
            self.tools["get_market_context"].call({"symbol": "ethusdt"})
        with self.assertRaisesRegex(ToolError, "timeframe"):
            self.tools["get_market_context"].call({"symbol": "BTCUSDT", "timeframe": "2h"})
        self.assertEqual(sum(not a["ok"] for a in self.actions), 5)
        self.assertTrue(all("error" in a for a in self.actions if not a["ok"]))

    def test_info_tools(self):
        space = self.call("describe_strategy_space")
        self.assertIn("entry_zone_lo", space["params"])
        self.assertEqual(space["presets"]["golden_pocket_1h"], {"timeframe": "1h"})
        self.assertEqual(space["research"]["max_grid_size"], 64)
        self.assertEqual(space["validation_gate"]["min_trades"], 30)
        ctx = self.call("get_market_context", symbol="btcusdt", timeframe="4h")
        self.assertEqual(ctx["symbol"], "BTCUSDT")
        self.assertEqual(self.call("get_signal_review_stats")["reviewed"], 0)
        st = self.lab.store
        for i in range(3):
            st.add_experiment("backtest", PRESETS["golden_pocket_1d"].to_dict(), "s1", 0, H4,
                              {"n": 10 + i, "avg_r": 0.1 * i, "trades": list(range(500))})
        rows = self.call("list_experiments", limit=500, sort_by="avg_r")
        self.assertEqual([r["metrics"]["n"] for r in rows], [12, 11, 10])
        self.assertEqual(rows[0]["params"], {"timeframe": "1d", "entry_zone_lo": 0.5, "stop_atr_buffer": 0.0})

    def test_paper_trades(self):
        sid = "abc"
        self.lab.store.add_to_roster(sid, "x", StrategySpec().to_dict(), T_END, {}, "r")
        for i in range(30):
            tr = Trade(sid, "BTCUSDT", "4h", 1, f"L{i}", i, i * H4, 100.0, 95.0, (i + 2) * H4, 110.0,
                       "target", 0.1, 0.05, 2.0, 2, {})
            self.lab.store.upsert_paper_trade(sid, tr, notional=100.0, pnl=10.0)
        out = self.call("get_paper_trades", strategy_id=sid, limit=1000)
        self.assertEqual(len(out["trades"]), 30)
        self.assertEqual(out["trades"][0]["leg_id"], "L29")
        self.assertEqual(out["trades"][0]["status"], "closed")
        self.assertEqual(len(self.call("get_paper_trades", strategy_id=sid, limit=0)["trades"]), 1)

    def test_notebook_tools(self):
        self.call("append_notebook", entry="H1: golden pocket works on 4h")
        self.assertIn("H1: golden pocket", self.call("read_notebook")["notebook"])
        self.assertEqual(self.call("rewrite_notebook", content="condensed")["notebook_chars"], 9)
        self.lab._notebook = "z" * 20000          # under notebook_max_chars, over MAX_RESULT_CHARS
        self.assertEqual(len(self.call("read_notebook")["notebook"]), 20000)


# -- cycles --------------------------------------------------------------------------------------

def research_script(sid: str) -> list:
    return [
        msg([text("Plan: test the golden pocket."), tool_use("get_lab_status", {}, "a"),
             tool_use("run_backtest", {"spec": {"preset": "golden_pocket_4h"}}, "b")], "tool_use"),
        msg([tool_use("validate_strategy", {"spec": {}}, "c"),
             tool_use("promote_to_paper", {"spec": {}, "name": "gp", "rationale": "robust"}, "d"),
             tool_use("append_notebook", {"entry": "Promoted gp."}, "e")], "tool_use"),
        msg([text("Summary: promoted gp; next test shorts.")], "end_turn"),
    ]


class TestCycles(TmpCase):
    def setUp(self):
        super().setUp()
        self.lab = FakeLab(self.cfg)

    def test_research_cycle(self):
        fc = FakeClient(scripts=[research_script(StrategySpec().spec_id())])
        before = int(time.time())
        res = research_cycle(self.lab, Brain(self.cfg, client=fc))
        self.assertTrue(res["ok"])
        self.assertEqual(self.lab.calls[0], ("begin_cycle", "research"))
        self.assertEqual(self.lab.calls[-1], ("end_cycle",))
        self.assertEqual(len(fc.runner_kwargs[0]["tools"]), 16)
        self.assertIn("RESEARCH cycle", fc.runner_kwargs[0]["messages"][0]["content"])
        self.assertIn('"validations_left":1', fc.runner_kwargs[0]["messages"][0]["content"])
        self.assertEqual(fc.runner_kwargs[0]["system"][0]["text"], SYSTEM_PROMPT)
        self.assertTrue(all(not err for _, _, err in fc.runners[0].results))
        self.assertEqual(len(res["promoted"]), 1)
        self.assertEqual(res["turns"], 3)
        j = self.lab.store.journal(kind="research")[0]
        self.assertEqual(j["id"], res["journal_id"])
        self.assertEqual(j["title"], "Research cycle: 1 promoted, 0 retired")
        self.assertIn("Summary: promoted gp", j["body"])
        self.assertIn("promote_to_paper x1", j["body"])
        self.assertIn("Promoted gp (", j["body"])
        self.assertIn("cache_read=150", j["body"])
        self.assertGreaterEqual(self.lab.store.get_kv("last_research_t"), before)
        self.assertIn("Promoted gp.", self.lab.notebook())

    def test_review_cycle(self):
        sid = "s1"
        self.lab.store.add_to_roster(sid, "old", StrategySpec().to_dict(), T_END, {}, "r")
        script = [msg([tool_use("retire_strategy", {"strategy_id": sid, "reason": "failing PF 0.7"})], "tool_use"),
                  msg([tool_use("run_optimization", {"base_spec": {}, "grid": {}})], "tool_use"),
                  msg([text("Retired old.")])]
        fc = FakeClient(scripts=[script])
        res = review_cycle(self.lab, Brain(self.cfg, client=fc))
        self.assertEqual(self.lab.calls[0], ("begin_cycle", "review"))
        names = [t.name for t in fc.runner_kwargs[0]["tools"]]
        self.assertNotIn("run_optimization", names)
        self.assertIn("REVIEW cycle", fc.runner_kwargs[0]["messages"][0]["content"])
        self.assertEqual(fc.runners[0].results[1][0], "run_optimization")
        self.assertTrue(fc.runners[0].results[1][2], "unknown tool -> error")
        self.assertEqual(res["retired"][0]["reason"], "failing PF 0.7")
        self.assertEqual(self.lab.store.journal(kind="review")[0]["title"], "Review cycle: 0 promoted, 1 retired")
        self.assertIsNotNone(self.lab.store.get_kv("last_review_t"))
        self.assertIsNone(self.lab.store.get_kv("last_research_t"))

    def test_brain_unavailable_is_journaled(self):
        fc = FakeClient(scripts=[[msg([tool_use("get_lab_status", {})], "tool_use"),
                                  api_error(anthropic.AuthenticationError, 401)]])
        res = research_cycle(self.lab, Brain(self.cfg, client=fc))
        self.assertFalse(res["ok"])
        self.assertIn("ANTHROPIC_API_KEY", res["error"])
        j = self.lab.store.journal()[0]
        self.assertEqual((j["kind"], j["title"]), ("research", "Research cycle failed: brain unavailable"))
        self.assertIn("get_lab_status x1", j["body"])
        self.assertEqual(self.lab.calls[-1], ("end_cycle",))
        self.assertIsNone(self.lab.store.get_kv("last_research_t"))

    def test_other_errors_propagate_after_end_cycle(self):
        fc = FakeClient(scripts=[[api_error(anthropic.BadRequestError, 400)]])
        with self.assertRaises(anthropic.BadRequestError):
            research_cycle(self.lab, Brain(self.cfg, client=fc))
        self.assertEqual(self.lab.calls[-1], ("end_cycle",))
        self.assertEqual(self.lab.store.journal(), [])

    def test_refusal_and_iteration_cap_titles(self):
        wrap = [msg([text("Wrapped up."), tool_use("append_notebook", {"entry": "cap hit"}, "w")], "tool_use")]
        fc = FakeClient(scripts=[[msg([text("no")], "refusal")],
                                 [msg([tool_use("get_lab_status", {})], "tool_use")] * 20, wrap])
        brain = Brain(self.cfg, client=fc)
        self.assertTrue(research_cycle(self.lab, brain)["refused"])
        self.assertEqual(self.lab.store.journal()[0]["title"], "Research cycle refused by the model")
        res = review_cycle(self.lab, brain)
        self.assertEqual(res["turns"], 13)            # the 12-turn cap plus one wrap-up reply
        self.assertTrue(res["wrapped_up"])
        self.assertEqual(self.lab.store.journal()[0]["title"],
                         "Review cycle: 0 promoted, 0 retired (stopped early: iteration cap)")
        self.assertIn("Wrapped up.", self.lab.store.journal()[0]["body"])
        self.assertIn("cap hit", self.lab.notebook())


# -- signal reviews ------------------------------------------------------------------------------

def make_signal(sid: str, sym: str, k: int, bars: Bars, side: int = 1) -> Signal:
    c, a = bars.c[k], 2.0
    anchor, extreme = c * 0.9, c * 1.1
    return Signal(sid, sym, "4h", side, f"L{bars.t[k - 30]}", bars.t[k], c * 0.98, c * 0.93, c * 1.1,
                  c * 1.2, {"anchor": anchor, "extreme": extreme, "atr": a, "rr": 2.1, "mode": "limit",
                            "anchor_t": bars.t[k - 30], "ext_t": bars.t[k - 5], "leg_size": extreme - anchor,
                            "zone": [c, c * 0.97]})


def review(decision="take", conf=0.8, reason="trend intact"):
    return msg([text(json.dumps({"decision": decision, "confidence": conf, "reason": reason}))])


class TestReviewSignals(TmpCase):
    def setUp(self):
        super().setUp()
        self.bars = make_bars("BTCUSDT", 300, seed=9)
        self.lab = FakeLab(cfg_(self.tmp.name, max_signal_reviews_per_day=3), {"BTCUSDT": self.bars})
        st = self.lab.store
        st.add_to_roster("live", "gp_live", PRESETS["golden_pocket_4h"].to_dict(), self.bars.t[200],
                         {"metrics": {"avg_r": 0.25}}, "r")
        st.add_to_roster("gone", "retired_one", StrategySpec().to_dict(), self.bars.t[200], {}, "r")
        st.set_strategy_status("gone", "retired", "bad")
        self.sigs = [make_signal("live", "BTCUSDT", k, self.bars) for k in (260, 280, 290, 295)]
        for s in self.sigs + [make_signal("gone", "BTCUSDT", 296, self.bars)]:
            st.add_signal(s)

    def test_off_mode_is_a_noop(self):
        self.lab.cfg = cfg_(self.tmp.name, signal_review="off")
        fc = FakeClient(replies=[review()] * 4)
        self.assertEqual(review_signals(self.lab, Brain(self.lab.cfg, client=fc)), [])
        self.assertEqual(review_signals(self.lab, Brain(self.lab.cfg, client=fc), signals=self.sigs), [])
        self.assertEqual(fc.create_kwargs, [])
        self.assertEqual(self.lab.store.journal(), [])

    def test_reviews_newest_roster_signals_within_quota(self):
        st = self.lab.store
        st.set_signal_review("live", "BTCUSDT", self.sigs[0].leg_id, "take", 0.5, "earlier")
        fc = FakeClient(replies=[review("skip", 1.7, "leg too small"), review("take", -0.2)])
        res = review_signals(self.lab, Brain(self.lab.cfg, client=fc))
        self.assertEqual([r["leg_id"] for r in res], [self.sigs[3].leg_id, self.sigs[2].leg_id])
        self.assertEqual([(r["decision"], r["confidence"]) for r in res], [("skip", 1.0), ("take", 0.0)])
        self.assertEqual(st.vetoed_legs("live"), {("BTCUSDT", self.sigs[3].leg_id)})
        self.assertEqual(len(st.signals(unreviewed=True)), 2)      # sigs[1] (quota) + the retired one
        self.assertEqual(fc.create_kwargs[0]["system"][0]["text"], SIGNAL_SYSTEM)
        self.assertEqual(fc.create_kwargs[0]["output_config"]["format"]["schema"], SIGNAL_SCHEMA)
        journal = st.journal(kind="signal")
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["title"], "Signal reviews (shadow): 1 take, 1 skip")
        self.assertIn("skip (1.00) leg too small", journal[0]["body"])
        self.assertIsNotNone(st.get_kv("last_signal_review_t"))
        self.assertEqual(review_signals(self.lab, Brain(self.lab.cfg, client=FakeClient())), [],
                         "daily quota used up")

    def test_prompt_is_causal(self):
        st = self.lab.store
        sig = self.sigs[0]
        k = self.bars.t.index(sig.t)
        for i, exit_t in enumerate((sig.t - H4, sig.t, sig.t + H4)):
            st.upsert_paper_trade("live", Trade("live", "BTCUSDT", "4h", 1, f"L{i}", 0, exit_t - 2 * H4,
                                                100.0, 95.0, exit_t, 110.0, "target", 0.1, 0.05,
                                                2.0, 2, {}), notional=1.0, pnl=0.1)
        fc = FakeClient(replies=[review()])
        res = review_signals(self.lab, Brain(self.lab.cfg, client=fc), signals=[sig])
        self.assertEqual(res[0]["decision"], "take")
        p = fc.create_kwargs[0]["messages"][0]["content"]
        blocks = {tag: json.loads(body) for tag, body in re.findall(r"<(\w+)>\n(.*?)\n</\1>", p, re.S)}
        market = blocks["market_context"]["market"]
        self.assertEqual(market["t"], iso(sig.t))
        self.assertEqual(market["n_bars"], k + 1)
        self.assertEqual(market["closes"][-1], round(self.bars.c[k], 4))
        setup = blocks["market_context"]["setup"]
        self.assertEqual(setup["side"], "long")
        self.assertAlmostEqual(setup["tp1_r"], (1.1 - 0.98) / (0.98 - 0.93), places=3)
        self.assertEqual((setup["bars_since_anchor"], setup["bars_since_extreme"]), (30, 5))
        self.assertAlmostEqual(setup["close_retracement"], 0.5, places=3)
        stats = blocks["strategy_stats"]
        self.assertEqual(stats["paper_closed_before_signal"]["n"], 2)   # exit_t <= signal t only
        self.assertEqual(stats["validation_avg_r"], 0.25)
        self.assertEqual(stats["your_review_record"]["reviewed"], 0)
        self.assertEqual(blocks["signal"]["strategy_name"], "gp_live")

    def test_failures(self):
        fc = FakeClient(replies=[msg([text("no")], "refusal"), review(decision="maybe"),
                                 api_error(anthropic.AuthenticationError, 401)])
        res = review_signals(self.lab, Brain(self.lab.cfg, client=fc))
        self.assertEqual(len(res), 3)
        self.assertTrue(all("error" in r for r in res))
        self.assertIn("brain unavailable", res[2]["error"])
        self.assertEqual(self.lab.store.count_signal_reviews_since(0), 0)
        self.assertEqual(self.lab.store.journal(kind="signal")[0]["title"],
                         "Signal reviews (shadow): 0 take, 0 skip, 3 failed")

    def test_unavailable_brain_stops_the_batch(self):
        fc = FakeClient(replies=[review(), api_error(anthropic.AuthenticationError, 401), review()])
        res = review_signals(self.lab, Brain(self.lab.cfg, client=fc))
        self.assertEqual([("decision" in r, "error" in r) for r in res], [(True, False), (False, True)])
        self.assertEqual(len(fc.create_kwargs), 2)

    def test_old_and_unknown_signals(self):
        old = make_signal("live", "BTCUSDT", 100, self.bars)
        self.assertLess(old.t, int(time.time()) - SIGNAL_MAX_AGE)
        self.lab.store.add_signal(old)
        ghost = make_signal("live", "BTCUSDT", 299, self.bars).to_dict()
        fc = FakeClient(replies=[review()] * 4)
        lab = FakeLab(cfg_(self.tmp.name, max_signal_reviews_per_day=10), {"BTCUSDT": self.bars})
        lab.store = self.lab.store
        res = review_signals(lab, Brain(lab.cfg, client=fc))
        self.assertNotIn(old.leg_id, [r["leg_id"] for r in res])
        self.assertEqual(review_signals(lab, Brain(lab.cfg, client=FakeClient(replies=[review()])),
                                        signals=[ghost])[0]["error"], "signal not found in the store")

    def test_setup_without_bars_or_meta(self):
        sig = make_signal("live", "BTCUSDT", 260, self.bars).to_dict()
        sig["meta"] = {}
        s = signal_setup(sig, None)
        self.assertIsNone(s["leg_atr"])
        self.assertNotIn("close", s)
        self.assertEqual(signal_stats(self.lab, self.lab.store.get_strategy("live"), sig["t"], {})
                         ["paper_closed_before_signal"]["n"], 0)


# -- real Lab integration ------------------------------------------------------------------------

class FakeCandles:
    def __init__(self, bars: dict[str, Bars]):
        self.bars = bars

    def get(self, symbol, tf):
        b = self.bars.get(symbol)
        return b if b is not None and b.tf == tf else Bars(symbol, tf)


class TestRealLab(TmpCase):
    def test_research_cycle_with_real_lab(self):
        from fibbot.lab import Lab
        bars = {s: make_bars(s, 2600, seed=i) for i, s in enumerate(("AAAUSDT", "BBBUSDT"), 11)}
        store = Store(os.path.join(self.tmp.name, "fibbot.sqlite"))
        self.addCleanup(store.close)
        lab = Lab(self.cfg, store=store, candles=FakeCandles(bars))
        spec = {"preset": "golden_pocket_4h", "trend_ema": 50, "min_leg_atr": 2.0, "min_rr": 1.0}
        script = [
            msg([tool_use("get_lab_status", {}, "1"), tool_use("describe_strategy_space", {}, "2")], "tool_use"),
            msg([tool_use("run_backtest", {"spec": spec}, "3"), tool_use("run_baseline", {"spec": spec}, "4"),
                 tool_use("get_market_context", {"symbol": "AAAUSDT"}, "5")], "tool_use"),
            msg([tool_use("list_experiments", {"limit": 5}, "6"), tool_use("validate_strategy", {"spec": spec}, "7"),
                 tool_use("run_backtest", {"spec": spec, "window": "validation"}, "8"),
                 tool_use("append_notebook", {"entry": "H1 tested."}, "9")], "tool_use"),
            msg([text("Tested H1.")]),
        ]
        fc = FakeClient(scripts=[script])
        with mock.patch("fibbot.lab.get_universe", return_value=list(bars)):
            res = research_cycle(lab, Brain(self.cfg, client=fc))
        results = fc.runners[0].results
        errors = {name: out for name, out, err in results if err}
        self.assertEqual(list(errors), ["run_backtest"], errors)     # only the validation-window call
        self.assertIn("only the research window", errors["run_backtest"])
        out = {name: json.loads(o) for name, o, err in results if not err and name != "run_backtest"}
        self.assertEqual(out["get_lab_status"]["cycle"], "research")
        self.assertEqual(out["get_lab_status"]["budget"]["validations_left"], 3)
        self.assertEqual(out["get_market_context"]["symbol"], "AAAUSDT")
        self.assertIn("passed", out["validate_strategy"])
        self.assertTrue(all(len(o) <= MAX_RESULT_CHARS for _, o, _ in results))
        self.assertTrue(res["ok"])
        self.assertIsNone(lab.cycle)
        sources = {e["source"] for e in store.list_experiments(limit=None)}
        self.assertEqual(sources, {"brain"})
        self.assertEqual(store.count_experiments(kind="validate"), 1)
        self.assertIn("H1 tested.", lab.notebook())
        self.assertIsNotNone(store.get_kv("last_research_t"))
        self.assertTrue(store.journal(kind="research")[0]["title"].startswith("Research cycle: 0 promoted"))


if __name__ == "__main__":
    unittest.main()
