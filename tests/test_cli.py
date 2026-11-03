"""Tests for fibbot.cli (and python -m fibbot): argument parsing, command dispatch, text and
--json output, spec input (preset, inline JSON, file, preset overrides, --tf), exit codes (0 ok,
1 user error without a traceback, 2 brain unavailable, 130 interrupted).

No network: Lab is patched to use synthetic candles (tests.test_lab), the universe and
CandleStore are fakes, the brain and scheduler are mocked.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fibbot import cli
from fibbot.config import DEFAULTS
from fibbot.lab import Lab
from fibbot.store import Store
from fibbot.strategy import PRESETS
from fibbot.types import Bars
from tests.test_lab import SYMS, T0, FakeCandles

ROOT = Path(__file__).resolve().parent.parent
SPEC = "deep_786_4h"
CONFIG = {"research": {"workers": 1, "validation_months": 3, "folds": 3, "min_trades": 5,
                       "validation_gate": {"min_deflated_sharpe": 0.0, "min_profit_factor": 1.0,
                                           "min_expectancy_r": 0.0, "min_validation_edge_z": None}},
          "home": "/nonexistent/config/home"}


class Candles(FakeCandles):
    updated = None

    def update_all(self, symbols, tfs):
        self.updated = (list(symbols), list(tfs))
        return {}


class FakeBrain:
    def __init__(self, ok=True, why="model x"):
        self.ok, self.why = ok, why

    def available(self):
        return self.ok, self.why


class CliCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / "home"
        self.config = Path(tmp.name) / "fibbot.json"
        self.config.write_text(json.dumps(CONFIG))
        self.lab_cfgs, self.candles = [], []
        for target, kw in (("fibbot.cli.Lab", {"side_effect": self.make_lab}),
                           ("fibbot.lab.get_universe", {"return_value": list(SYMS)}),
                           ("fibbot.paper.get_universe", {"return_value": list(SYMS)}),
                           ("fibbot.cli.get_universe", {"return_value": list(SYMS)})):
            p = mock.patch(target, **kw)
            p.start()
            self.addCleanup(p.stop)

    def make_lab(self, cfg):
        self.lab_cfgs.append(cfg)
        self.candles.append(Candles())
        return Lab(cfg, Store(Path(cfg["home"]) / "fibbot.sqlite"), self.candles[-1])

    def cli(self, *argv):
        """(exit code, stdout, stderr) of one command."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(["--config", str(self.config), "--home", str(self.home), *argv])
        return rc, out.getvalue(), err.getvalue()

    def ok(self, *argv) -> str:
        rc, out, err = self.cli(*argv)
        self.assertEqual(rc, 0, err)
        return out

    def json(self, *argv) -> dict:
        return json.loads(self.ok(*argv, "--json"))

    def user_error(self, *argv) -> str:
        rc, out, err = self.cli(*argv)
        self.assertEqual(rc, 1, (out, err))
        self.assertIn("error: ", err)
        self.assertNotIn("Traceback", err)
        return err

    def store(self) -> Store:
        s = Store(self.home / "fibbot.sqlite")
        self.addCleanup(s.close)
        return s


class TestParser(unittest.TestCase):
    def parse(self, *argv):
        return cli.build_parser().parse_args(list(argv))

    def test_commands_and_options(self):
        a = self.parse("--config", "c.json", "--home", "h", "backtest", "--preset", SPEC, "--window",
                       "validation", "--symbols", "AUSDT", "BUSDT", "--trades", "5", "--tf", "1h", "--json")
        self.assertEqual((a.config, a.home, a.command, a.preset, a.spec, a.window, a.symbols, a.trades,
                          a.tf, a.json), ("c.json", "h", "backtest", SPEC, None, "validation",
                                          ["AUSDT", "BUSDT"], 5, "1h", True))
        self.assertIs(a.func, cli.cmd_backtest)
        a = self.parse("optimize", "--spec", "{}", "--grid", '{"min_rr": [1, 2]}')
        self.assertEqual((a.spec, a.grid, a.json), ("{}", '{"min_rr": [1, 2]}', False))
        self.assertEqual(self.parse("baseline", "--preset", SPEC).runs, 10)
        a = self.parse("promote", "--preset", SPEC, "--name", "n", "--rationale", "r")
        self.assertEqual((a.name, a.rationale), ("n", "r"))
        a = self.parse("retire", "abc", "--reason", "why")
        self.assertEqual((a.strategy_id, a.reason), ("abc", "why"))
        a = self.parse("sweep", "--promote", "--presets", SPEC)
        self.assertEqual((a.promote, a.presets), (True, [SPEC]))
        a = self.parse("data", "--tf", "1h", "4h", "--symbols", "btcusdt")
        self.assertEqual((a.tf, a.symbols), (["1h", "4h"], ["btcusdt"]))
        a = self.parse("run", "--once", "--no-brain")
        self.assertEqual((a.once, a.no_brain), (True, True))
        self.assertEqual(self.parse("brain", "signals").action, "signals")
        self.assertTrue(self.parse("universe", "--refresh").refresh)
        for name in ("presets", "tick", "status", "report", "robustness", "validate"):
            argv = [name] + (["--preset", SPEC] if name in ("robustness", "validate") else [])
            self.assertEqual(self.parse(*argv).command, name)

    def test_json_only_where_the_contract_has_it(self):
        for name in ("backtest", "optimize", "baseline", "robustness", "validate", "status", "sweep"):
            parser = cli.build_parser()
            sub = parser._subparsers._group_actions[0].choices[name]
            self.assertIn("--json", sub._option_string_actions, name)
        for name in ("presets", "tick", "report", "universe", "data", "promote"):
            sub = cli.build_parser()._subparsers._group_actions[0].choices[name]
            self.assertNotIn("--json", sub._option_string_actions, name)

    def test_usage_errors_raise_user_errors(self):
        for argv in ([], ["bogus"], ["backtest"], ["backtest", "--preset", SPEC, "--spec", "{}"],
                     ["backtest", "--preset", "nope"], ["backtest", "--preset", SPEC, "--tf", "15m"],
                     ["promote", "--preset", SPEC], ["retire", "x"], ["brain", "dance"],
                     ["data", "--tf", "2h"], ["baseline", "--preset", SPEC, "--runs", "x"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(cli.UserError, msg=argv):
                self.parse(*argv)

    def test_help_exits_zero(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as cm:
            self.parse("--help")
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("No live trading", out.getvalue())


class TestConfigAndErrors(CliCase):
    def test_home_overrides_the_config_file(self):
        self.ok("status")
        cfg, = self.lab_cfgs
        self.assertEqual(cfg["home"], str(self.home.resolve()))
        self.assertEqual(cfg["research"]["folds"], 3)                  # the config file was read

    def test_bad_arguments_exit_one(self):
        rc, out, err = self.cli("backtest")
        self.assertEqual(rc, 1)
        self.assertIn("one of the arguments --preset --spec is required", err)
        self.assertIn("usage:", err)
        self.assertEqual(self.cli("bogus")[0], 1)

    def test_missing_config_file_is_a_user_error(self):
        self.config.unlink()
        self.user_error("presets")

    def test_keyboard_interrupt(self):
        with mock.patch("fibbot.scheduler.run_forever", side_effect=KeyboardInterrupt):
            rc, out, err = self.cli("run")
        self.assertEqual((rc, err.strip()), (130, "interrupted"))

    def test_key_errors_print_their_message(self):
        err = self.user_error("retire", "nope", "--reason", "x")
        self.assertIn("error: unknown strategy nope", err)


class TestSpecs(CliCase):
    def spec_id(self, *argv):
        return self.json("backtest", *argv)["spec_id"]

    def test_preset_inline_file_and_preset_overrides(self):
        base = PRESETS[SPEC]
        self.assertEqual(self.spec_id("--preset", SPEC), base.spec_id())
        self.assertEqual(self.spec_id("--spec", json.dumps(base.to_dict())), base.spec_id())
        over = base.with_overrides(min_rr=2.0)
        self.assertEqual(self.spec_id("--spec", '{"preset": "%s", "min_rr": 2}' % SPEC), over.spec_id())
        path = self.home.parent / "spec.json"
        path.write_text(json.dumps({"preset": SPEC, "min_rr": 2.0}))
        self.assertEqual(self.spec_id("--spec", str(path)), over.spec_id())

    def test_tf_overrides_the_spec_timeframe(self):
        err = self.user_error("backtest", "--preset", SPEC, "--tf", "1h")
        self.assertIn("no cached 1h candles", err)
        self.assertEqual(self.spec_id("--preset", SPEC, "--tf", "4h"), PRESETS[SPEC].spec_id())

    def test_bad_specs_are_user_errors(self):
        for spec, msg in (("{not json", "--spec: invalid JSON"), ("[1, 2]", "must be a JSON object"),
                          ("/no/such/spec.json", "no such file"), ('{"bogus": 1}', "bogus"),
                          ('{"min_rr": 99}', "min_rr"), ('{"preset": "nope"}', "unknown preset 'nope'"),
                          ('{"entry_zone_lo": 0.7, "entry_zone_hi": 0.65}', "entry_zone")):
            self.assertIn(msg, self.user_error("backtest", "--spec", spec), spec)
        self.assertEqual(self.store().count_experiments(), 0)


class TestResearchCommands(CliCase):
    def test_backtest_text_and_json(self):
        out = self.ok("backtest", "--preset", SPEC, "--trades", "3")
        sid = PRESETS[SPEC].spec_id()
        self.assertIn(f"Backtest: {SPEC} [{sid}] 4h long, research window", out)
        for part in ("mean R", "positive folds", "account: 10,000", "symbol", "year", "entry (UTC)",
                     "experiment #1"):
            self.assertIn(part, out)
        self.assertEqual(len([ln for ln in out.splitlines() if ln.startswith("S") and " long " in ln]), 3)
        d = self.json("backtest", "--preset", SPEC, "--window", "all", "--symbols", SYMS[0])
        self.assertEqual((d["window"], d["n_symbols"], d["experiment_id"]), ("all", 1, 2))
        self.assertEqual(self.store().count_experiments(kind="backtest"), 2)

    def test_optimize(self):
        grid = '{"min_rr": [1.0, 1.5], "stop_atr_buffer": [0.2, 9]}'
        out = self.ok("optimize", "--preset", SPEC, "--grid", grid)
        self.assertIn("Walk-forward optimization: 4 combos (2 evaluated, 2 invalid)", out)
        self.assertIn("invalid: stop_atr_buffer", out)
        self.assertIn("sensitivity", out)
        d = self.json("optimize", "--preset", SPEC, "--grid", grid)
        self.assertEqual((d["n_combos"], d["n_evaluated"]), (4, 2))
        self.assertIn("unknown parameters", self.user_error("optimize", "--preset", SPEC, "--grid", '{"x": [1]}'))
        self.assertIn("--grid", self.user_error("optimize", "--preset", SPEC, "--grid", "{bad"))

    def test_baseline_and_robustness(self):
        self.assertIn("Random-entry baseline", self.ok("baseline", "--preset", SPEC, "--runs", "3"))
        d = self.json("baseline", "--preset", SPEC, "--runs", "2")
        self.assertEqual((d["n_runs"], len(d["runs"])), (2, 2))
        self.assertIn("n_runs must be", self.user_error("baseline", "--preset", SPEC, "--runs", "0"))
        out = self.ok("robustness", "--preset", SPEC)
        self.assertIn("one-step neighbours", out)
        self.assertGreater(self.json("robustness", "--preset", SPEC)["n"], 0)

    def test_validate_promote_status_report_retire(self):
        self.assertIn("has no validation", self.user_error("promote", "--preset", SPEC, "--rationale", "r"))
        out = self.ok("validate", "--preset", SPEC)
        self.assertIn("gate PASSED", out)
        self.assertIn("min_deflated_sharpe", out)
        self.assertTrue(self.json("validate", "--preset", SPEC)["passed"])
        sid = PRESETS[SPEC].spec_id()
        # promotion also needs a research-window baseline with an edge over random entries
        self.assertIn("no research-window baseline",
                      self.user_error("promote", "--preset", SPEC, "--rationale", "r"))
        self.ok("baseline", "--preset", SPEC)
        out = self.ok("promote", "--preset", SPEC, "--rationale", "tested")
        self.assertIn(f"promoted {SPEC} [{sid}] 4h: paper trading starts", out)
        row = self.store().get_strategy(sid)
        self.assertEqual((row["name"], row["rationale"], row["status"]), (SPEC, "tested", "paper"))
        self.assertIn("already on the roster", self.user_error("promote", "--preset", SPEC, "--rationale", "x"))

        self.assertIn(f"{SPEC} [{sid}]", self.ok("status"))
        st = self.json("status")
        self.assertEqual([r["strategy_id"] for r in st["roster"]], [sid])
        out = self.ok("report")
        self.assertEqual(out.strip(), f"wrote {self.home.resolve() / 'REPORT.md'}")
        self.assertTrue((self.home / "REPORT.md").read_text().startswith("# fibbot report"))

        out = self.ok("tick")
        self.assertIn(sid, out)
        self.assertEqual(self.candles[-1].updated, (SYMS, ["4h"]))     # tick downloads first
        out = self.ok("retire", sid, "--reason", "done")
        self.assertIn(f"retired {SPEC} [{sid}]: done", out)
        self.assertEqual(self.store().get_strategy(sid)["status"], "retired")

    def test_promote_name_option(self):
        self.ok("validate", "--preset", SPEC)
        self.ok("baseline", "--preset", SPEC)
        self.ok("promote", "--preset", SPEC, "--name", "my label", "--rationale", "r")
        self.assertEqual(self.store().roster()[0]["name"], "my label")

    def test_sweep(self):
        d = self.json("sweep", "--presets", SPEC, "golden_pocket_4h")
        self.assertEqual(d["presets"], [SPEC, "golden_pocket_4h"])
        self.assertEqual(self.store().roster(), [])
        out = self.ok("sweep", "--presets", SPEC, "--promote")
        self.assertIn("Autopilot sweep", out)
        self.assertIn(f"promoted {SPEC}", out)
        self.assertEqual(len(self.store().roster()), 1)

    def test_tick_with_an_empty_roster(self):
        self.assertEqual(self.ok("tick").strip(), "no strategies on paper (promote a validated spec first)")

    def test_presets(self):
        out = self.ok("presets")
        for name, spec in PRESETS.items():
            self.assertIn(name, out)
            self.assertIn(spec.spec_id(), out)


class FakeCandleStore:
    """CandleStore stand-in: every update adds 2 bars; symbols in `bad` fail."""
    bad = {"BADUSDT"}

    def __init__(self, cfg):
        self.bars = {}

    def get(self, symbol, tf):
        return self.bars.get((symbol, tf), Bars(symbol, tf))

    def update(self, symbol, tf):
        if symbol in self.bad:
            raise RuntimeError("HTTP 400")
        old = self.get(symbol, tf)
        rows = [(t, 1, 1, 1, 1, 1) for t in old.t] + [(T0 + k * 3600, 1, 1, 1, 1, 1)
                                                       for k in range(len(old), len(old) + 2)]
        self.bars[(symbol, tf)] = Bars.from_rows(symbol, tf, rows)
        return self.bars[(symbol, tf)]


class TestDataAndUniverse(CliCase):
    def test_universe(self):
        out = self.ok("universe", "--refresh")
        self.assertIn("top 12 by market cap", out)
        self.assertEqual([ln.split()[1] for ln in out.splitlines()[1:]], SYMS)
        cli.get_universe.assert_called_with(mock.ANY, refresh=True)

    def test_data_progress_and_failures(self):
        with mock.patch("fibbot.cli.CandleStore", FakeCandleStore):
            out = self.ok("data", "--tf", "1h")
            lines = out.splitlines()
            self.assertEqual(len(lines), 1 + len(SYMS))
            self.assertEqual("[1/3] S0USDT 1h: +2 bars, 2 cached (2020-09-14..2020-09-14T02:00:00Z)", lines[1])
            rc, out, err = self.cli("data", "--symbols", "btcusdt", "badusdt")
        self.assertEqual(rc, 1)
        tfs = DEFAULTS["data"]["timeframes"]
        self.assertEqual(len(out.splitlines()), 1 + 2 * len(tfs))     # default timeframes
        self.assertIn("BTCUSDT 1h: +2 bars", out)
        self.assertIn("BADUSDT 4h: FAILED RuntimeError: HTTP 400", out)
        self.assertIn(f"{len(tfs)} of {2 * len(tfs)} updates failed", err)


class TestBrainCommands(CliCase):
    def setUp(self):
        super().setUp()
        self.brain = FakeBrain()
        p = mock.patch("fibbot.scheduler.make_brain", side_effect=lambda cfg: self.brain)
        p.start()
        self.addCleanup(p.stop)

    def test_unavailable_brain_exits_two(self):
        self.brain = FakeBrain(False, "no key here")
        for action in ("research", "review", "signals"):
            rc, out, err = self.cli("brain", action)
            self.assertEqual(rc, 2)
            self.assertIn("brain unavailable: no key here", err)
            self.assertIn("python -m fibbot sweep", err)
            self.assertNotIn("Traceback", err)

    def test_without_anthropic_or_a_key(self):
        from fibbot.brain.client import Brain
        self.brain = None
        with mock.patch("fibbot.scheduler.make_brain", side_effect=Brain), \
                mock.patch.dict(sys.modules, {"anthropic": None}):
            rc, out, err = self.cli("brain", "research")
        self.assertEqual(rc, 2)
        self.assertIn("anthropic", err)

    def test_cycles(self):
        res = {"kind": "research", "ok": True, "text": "Tried 3 ideas.", "stop_reason": "end_turn",
               "turns": 7, "refused": False, "promoted": [{}], "retired": [], "journal_id": 5,
               "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 1}}
        with mock.patch("fibbot.brain.cycles.research_cycle", return_value=res) as research, \
                mock.patch("fibbot.brain.cycles.review_cycle", return_value={**res, "kind": "review"}) as review:
            out = self.ok("brain", "research")
            self.assertIn("research cycle: 7 turns, stop end_turn, 1 promoted, 0 retired (journal #5)", out)
            self.assertIn("Tried 3 ideas.", out)
            self.assertIn("review cycle", self.ok("brain", "review"))
            self.assertEqual((research.call_count, review.call_count), (1, 1))
            self.assertIs(research.call_args[0][1], self.brain)
            research.return_value = {"kind": "research", "ok": False, "error": "API down", "actions": []}
            rc, out, err = self.cli("brain", "research")
        self.assertEqual(rc, 2)
        self.assertIn("brain unavailable: API down", err)

    def test_signal_reviews(self):
        results = [{"strategy_id": "s", "symbol": "AUSDT", "leg_id": "L1", "decision": "take",
                    "confidence": 0.8, "reason": "trend"},
                   {"strategy_id": "s", "symbol": "BUSDT", "leg_id": "L2", "error": "malformed"}]
        with mock.patch("fibbot.brain.cycles.review_signals", return_value=results) as rs:
            out = self.ok("brain", "signals")
            self.assertIn("AUSDT L1 [s]: take (0.80) trend", out)
            self.assertIn("BUSDT L2 [s]: ERROR malformed", out)
            rs.return_value = []
            self.assertIn("no unreviewed signals", self.ok("brain", "signals"))
            rs.return_value = [{**results[1], "error": "brain unavailable: 529"}]
            rc, out, err = self.cli("brain", "signals")
            self.assertEqual(rc, 2)
            self.assertIn("brain unavailable: 529", err)
            config = json.loads(self.config.read_text())
            self.config.write_text(json.dumps({**config, "brain": {"signal_review": "off"}}))
            rs.reset_mock()
            self.assertIn("signal_review is off", self.ok("brain", "signals"))
            rs.assert_not_called()


class TestRun(CliCase):
    def test_run_forever_and_once(self):
        with mock.patch("fibbot.scheduler.run_forever") as forever:
            self.ok("run")
            self.ok("run", "--no-brain")
        self.assertEqual([c.kwargs["use_brain"] for c in forever.call_args_list], [True, False])
        self.assertEqual(forever.call_args[0][0]["home"], str(self.home.resolve()))
        steps = [{"step": "paper", "status": "ok"}, {"step": "report", "status": "skipped"}]
        with mock.patch("fibbot.scheduler.run_once", return_value={"steps": steps}) as once, \
                mock.patch("fibbot.scheduler.setup_logging") as logs:
            self.ok("run", "--once")
            once.assert_called_once_with(mock.ANY, use_brain=True)
            logs.assert_called_once()
            once.return_value = {"steps": [*steps, {"step": "data", "status": "failed"}]}
            self.assertEqual(self.cli("run", "--once", "--no-brain")[0], 1)
            self.assertEqual(once.call_args.kwargs, {"use_brain": False})


class TestModuleEntryPoint(unittest.TestCase):
    def run_module(self, *argv):
        with tempfile.TemporaryDirectory() as home:
            env = {k: v for k, v in os.environ.items() if not k.startswith("FIBBOT_")}
            return subprocess.run([sys.executable, "-m", "fibbot", "--home", home, *argv], cwd=ROOT,
                                  env=env, capture_output=True, text=True, timeout=60)

    def test_python_m_fibbot(self):
        p = self.run_module("presets")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("golden_pocket_4h", p.stdout)
        p = self.run_module("backtest", "--spec", "{bad")
        self.assertEqual(p.returncode, 1)
        self.assertIn("error: --spec: invalid JSON", p.stderr)
        self.assertEqual(self.run_module().returncode, 1)


if __name__ == "__main__":
    unittest.main()
