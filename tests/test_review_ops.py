"""Regression tests for the "ops" review findings: deployment files (GitHub Actions workflow,
Dockerfile, requirements.txt), config validation and FIBBOT_HOME precedence, and the scheduler's
candle-stall detection, failure de-duplication, log rotation and brain-cycle stop/deadline.

No network and no Claude: the scheduler runs on the fakes of tests/test_scheduler.py.
"""
import ast
import contextlib
import importlib.metadata
import io
import json
import logging
import logging.handlers
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fibbot import cli, config, scheduler
from fibbot.config import DEFAULTS, OPTIONAL_KEYS, RULES, deep_merge, load_config, validate_config
from fibbot.scheduler import DATA_STALL_KEY, FAILURE_REPEAT_HOURS, STALE_BARS, cycle_stop
from fibbot.types import TF_SECONDS
from tests.test_scheduler import HOUR, NOW, FakeBrain, SchedulerCase, cfg_, open_trade

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is in the dev environment
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "fibbot-paper.yml"


def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def steps_by_name(job: dict) -> dict:
    return {s.get("id") or s.get("name") or s.get("uses"): s for s in job["steps"]}


def run_if(expr: str | None, outcome: str, job_cancelled: bool) -> bool:
    """Evaluate a step's `if:` the way Actions does for the expressions this workflow uses: no
    `if` means success(); a job that is being cancelled only runs always() steps."""
    if expr is None:
        return outcome == "success" and not job_cancelled
    s = expr.strip()
    if s.startswith("${{"):
        s = s[3:-2]
    status_fn = "always()" in s or "cancelled()" in s or "failure()" in s or "success()" in s
    py = (s.replace("always()", "True").replace("!cancelled()", f"(not {job_cancelled})")
          .replace("cancelled()", str(job_cancelled)).replace("steps.bot.outcome", repr(outcome))
          .replace("&&", " and ").replace("||", " or "))
    value = bool(eval(py, {"__builtins__": {}}))   # only literals and and/or/== remain
    if not status_fn:                               # implicit success() && ...
        value = value and not job_cancelled
    return value


def save_snippet() -> str:
    """The Python run by the save step (between `python - <<'EOF'` and `EOF`)."""
    run = steps_by_name(workflow()["jobs"]["tick"])["Save state to the fibbot-state branch (one squashed commit)"]["run"]
    return run.split("python - <<'EOF'\n", 1)[1].split("\nEOF\n", 1)[0]


@unittest.skipUnless(yaml, "needs PyYAML")
class TestWorkflow(unittest.TestCase):
    """F13, F35, F48: the Actions workflow keeps paid state, holds no token while the bot runs
    and refuses to push a state file GitHub would reject."""

    def setUp(self):
        self.job = workflow()["jobs"]["tick"]
        self.steps = steps_by_name(self.job)
        self.after_bot = [s for s in self.job["steps"][[s.get("id") for s in self.job["steps"]].index("bot") + 1:]]

    def test_bot_step_times_out_before_the_job(self):
        bot = self.steps["bot"]
        self.assertIn("timeout-minutes", bot, "a job timeout cancels the step and skips the saves")
        self.assertLessEqual(bot["timeout-minutes"], self.job["timeout-minutes"] - 10)

    def test_saves_run_after_success_failure_timeout_and_cancel_but_not_after_a_skip(self):
        self.assertEqual(len(self.after_bot), 3)
        for step in self.after_bot:
            name = step.get("name") or step.get("uses")
            for outcome, cancelled in (("success", False), ("failure", False),   # failure = step timeout
                                       ("cancelled", True)):                    # manual cancel / job timeout
                self.assertTrue(run_if(step.get("if"), outcome, cancelled), (name, outcome))
            for cancelled in (False, True):                                     # restore failed: no save
                self.assertFalse(run_if(step.get("if"), "skipped", cancelled), name)

    def test_checkout_does_not_persist_the_token(self):
        checkout = next(s for s in self.job["steps"] if str(s.get("uses", "")).startswith("actions/checkout"))
        self.assertIs((checkout.get("with") or {}).get("persist-credentials"), False)

    def test_only_the_git_steps_get_the_token(self):
        with_token = sorted(s.get("name") for s in self.job["steps"]
                            if "github.token" in json.dumps(s.get("env") or {}))
        self.assertEqual(with_token, ["Restore state from the fibbot-state branch",
                                      "Save state to the fibbot-state branch (one squashed commit)"])
        self.assertNotIn("github.token", json.dumps(self.steps["bot"]))
        for name in with_token:
            self.assertIn("GIT_CONFIG_VALUE_0", self.steps[name]["run"])

    def test_save_snippet_vacuums_checkpoints_and_saves_normal_state(self):
        with tempfile.TemporaryDirectory() as home:
            db = Path(home) / "fibbot.sqlite"
            con = sqlite3.connect(db)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("CREATE TABLE t (x)")
            con.executemany("INSERT INTO t VALUES (?)", [("x" * 1000,)] * 500)
            con.commit()
            con.execute("DELETE FROM t")
            con.commit()
            con.close()
            p = subprocess.run([sys.executable, "-c", save_snippet()], env={**os.environ, "FIBBOT_HOME": home},
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertLess(db.stat().st_size, 100_000)                 # VACUUM dropped the free pages
            wal = Path(home) / "fibbot.sqlite-wal"
            self.assertTrue(not wal.exists() or wal.stat().st_size == 0)

    def test_save_snippet_refuses_a_file_github_would_reject(self):
        code = save_snippet()
        self.assertIn("WARN_MB, MAX_MB = 50, 95", code)
        code = code.replace("WARN_MB, MAX_MB = 50, 95", "WARN_MB, MAX_MB = 0.001, 0.002")
        with tempfile.TemporaryDirectory() as home:
            (Path(home) / "BRAIN_NOTEBOOK.md").write_text("x" * 5000)
            (Path(home) / "candles").mkdir()
            (Path(home) / "candles" / "BTCUSDT_1h.csv.gz").write_bytes(b"x" * 5000)   # not committed
            (Path(home) / "fibbot.log.1").write_text("x" * 5000)                      # not committed
            p = subprocess.run([sys.executable, "-c", code], env={**os.environ, "FIBBOT_HOME": home},
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertEqual(p.stdout.count("::error::"), 1, p.stdout)
        self.assertIn("BRAIN_NOTEBOOK.md", p.stdout)
        self.assertIn("NOT saved", p.stdout)

    def test_rotated_logs_are_not_committed(self):
        run = self.steps["Save state to the fibbot-state branch (one squashed commit)"]["run"]
        self.assertIn("':(exclude)*.log.*'", run)


class TestDeployFiles(unittest.TestCase):
    def test_requirements_pin_the_sdk_the_tests_run_against(self):
        """F35: an unbounded 'anthropic>=' would install any new major on the next hourly run."""
        lines = [ln.strip() for ln in (ROOT / "requirements.txt").read_text().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        for ln in lines:
            self.assertRegex(ln, r"^[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.]*$", ln)
        pins = dict(ln.split("==") for ln in lines)
        self.assertEqual(pins.get("anthropic"), importlib.metadata.version("anthropic"))

    def test_docker_stop_gets_a_grace_period(self):
        """F49: docker's default 10 s SIGKILL would throw away a paid cycle in progress."""
        head = (ROOT / "Dockerfile").read_text().split("FROM ", 1)[0]
        self.assertRegex(head, r"docker run[^\n]*--init[^\n]*--stop-timeout \d+")


class TestConfigValidation(unittest.TestCase):
    """F38: a typo or a wrong type fails fast instead of being silently ignored."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for k in ("FIBBOT_HOME", "FIBBOT_CONFIG"):
            os.environ.pop(k, None)

    def load(self, **over):
        return load_config(overrides={"home": str(self.tmp), **over})

    def test_defaults_and_the_example_validate(self):
        validate_config(deep_merge(DEFAULTS, {"home": "/x"}))
        cfg = load_config(ROOT / "fibbot.example.json", {"home": str(self.tmp)})
        self.assertEqual(cfg["universe"]["top_n"], 10)

    def test_every_default_has_a_rule(self):
        def leaves(d, prefix=""):
            for k, v in d.items():
                yield from leaves(v, f"{prefix}{k}.") if isinstance(v, dict) else [prefix + k]
        self.assertEqual(sorted(leaves(DEFAULTS)), sorted(RULES))
        self.assertFalse(set(RULES) & set(OPTIONAL_KEYS))

    def test_bad_values_name_the_key(self):
        for over, msg in (({"brain": {"signal_review": False}}, "brain.signal_review: must be one of off|shadow|active, got False"),
                          ({"brain": {"signal_review": "Off"}}, "brain.signal_review"),
                          ({"brain": {"signal_review": "none"}}, "brain.signal_review"),
                          ({"brain": {"efort": "low"}}, "brain.efort: unknown key (did you mean 'effort'?)"),
                          ({"brain": {"enabled": "false"}}, "brain.enabled: must be true or false"),
                          ({"brain": {"effort": "extreme"}}, "brain.effort"),
                          ({"brain": {"review_effort": "High"}}, "brain.review_effort"),
                          ({"data": {"timeframes": ["2h"]}}, "data.timeframes"),
                          ({"data": {"timeframes": []}}, "data.timeframes: must not be empty"),
                          ({"data": {"history_start": "2020-13-01"}}, "data.history_start"),
                          ({"universe": {"extra_exclude": "TRX"}}, "universe.extra_exclude: must be a list"),
                          ({"universe": {"top_n": "10"}}, "universe.top_n"),
                          ({"portfolio": {"risk_per_trade": 5}}, "portfolio.risk_per_trade"),
                          ({"research": {"workers": True}}, "research.workers"),
                          ({"research": {"validation_gate": {"min_trade": 3}}},
                           "research.validation_gate.min_trade: unknown key (did you mean 'min_trades'?)"),
                          ({"scheduler": {"tick_every_seconds": 0}}, "scheduler.tick_every_seconds"),
                          ({"brain": "off"}, "brain: must be a JSON object"),
                          ({"papr": {}}, "papr: unknown key (did you mean 'paper'?)")):
            with self.assertRaises(ValueError, msg=over) as cm:
                self.load(**over)
            self.assertIn(msg, str(cm.exception), over)

    def test_every_problem_is_reported_at_once(self):
        with self.assertRaises(ValueError) as cm:
            self.load(brain={"efort": "low", "signal_review": False})
        self.assertIn("brain.efort", str(cm.exception))
        self.assertIn("brain.signal_review", str(cm.exception))

    def test_valid_optional_and_null_values(self):
        cfg = self.load(brain={"fallbacks": None, "review_effort": "low", "review_max_iterations": 7,
                               "max_retries": 0, "max_monthly_output_tokens": None, "max_cycle_minutes": 0},
                        research={"max_validations_per_week": None, "validation_universe": "current",
                                  "validation_gate": {"min_validation_edge_r": None}},
                        portfolio={"daily_loss_limit": 0, "starting_equity": 5000})
        self.assertEqual(cfg["brain"]["review_effort"], "low")
        self.load(brain={"fallbacks": [{"model": "claude-opus-4-8"}]})
        self.load(brain={"fallbacks": False})

    def test_values_are_not_normalised(self):
        self.assertEqual(self.load(brain={"signal_review": "off"})["brain"]["signal_review"], "off")

    def test_the_file_name_is_in_the_message_and_the_cli_exits_one(self):
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"brain": {"signal_review": False, "efort": "low"}}))
        with self.assertRaises(ValueError) as cm:
            load_config(bad, {"home": str(self.tmp)})
        self.assertTrue(str(cm.exception).startswith(f"{bad}: invalid config:"), cm.exception)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = cli.main(["--config", str(bad), "--home", str(self.tmp), "status"])
        self.assertEqual(rc, 1)
        self.assertIn("brain.signal_review", err.getvalue())
        self.assertIn("did you mean 'effort'", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_a_non_object_file_is_a_user_error(self):
        bad = self.tmp / "list.json"
        bad.write_text("[1, 2]")
        with self.assertRaisesRegex(ValueError, "must hold a JSON object"):
            load_config(bad)

    def test_run_forever_validates_a_config_built_in_code(self):
        cfg = cfg_(self.tmp, brain={"signal_review": "Active"})
        with mock.patch("fibbot.scheduler.Lab") as lab, mock.patch("fibbot.scheduler.setup_logging"), \
                mock.patch("fibbot.scheduler.run_once"), \
                mock.patch("fibbot.scheduler.sleep_until", side_effect=AssertionError("not validated")), \
                self.assertRaisesRegex(ValueError, "signal_review"):
            scheduler.run_forever(cfg)
        lab.assert_not_called()

    def test_every_config_key_the_code_reads_is_accepted(self):
        """A key read by some module but unknown to the validator would be rejected in a config
        file. Scans fibbot/ for cfg["section"]["key"] / .get("key") reads, directly or through a
        name bound to a section (bcfg = cfg["brain"], self.bcfg, gate_cfg, ...)."""
        known = set(config.known_keys())
        reads = {}
        for path in sorted((ROOT / "fibbot").rglob("*.py")):
            for key, line in config_reads(path.read_text()):
                reads.setdefault(key, f"{path.relative_to(ROOT)}:{line}")
        self.assertGreater(len(reads), 30)              # the scan does find the reads
        self.assertIn("brain.review_max_iterations", reads)
        self.assertIn("research.max_validations_per_week", reads)
        unknown = {k: where for k, where in reads.items() if k not in known}
        self.assertEqual(unknown, {}, "add these keys to config.DEFAULTS or config.OPTIONAL_KEYS")


SECTIONS = {k for k, v in DEFAULTS.items() if isinstance(v, dict)}
CONVENTION = {"bcfg": "brain", "rcfg": "research", "pcfg": "portfolio", "gate_cfg": "research.validation_gate"}


def _is_section(path: str) -> bool:
    node = DEFAULTS
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return isinstance(node, dict)


def config_reads(source: str):
    """(dotted key, line) for every config read config_reads can resolve in `source`."""
    tree = ast.parse(source)
    attr_alias = {}

    def path_of(e, alias):
        if isinstance(e, ast.Name):
            return "" if e.id == "cfg" else alias.get(e.id)
        if isinstance(e, ast.Attribute):
            return "" if e.attr == "cfg" else attr_alias.get(e.attr)
        key = None
        if isinstance(e, ast.Subscript) and isinstance(e.slice, ast.Constant) and isinstance(e.slice.value, str):
            base, key = e.value, e.slice.value
        elif (isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr == "get"
              and e.args and isinstance(e.args[0], ast.Constant) and isinstance(e.args[0].value, str)):
            base, key = e.func.value, e.args[0].value
        if key is None:
            return None
        p = path_of(base, alias)
        if p is None or (p == "" and key not in SECTIONS and key != "home") or (p and not _is_section(p)):
            return None
        return f"{p}.{key}" if p else key

    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    scopes = []
    for f in funcs:
        alias = {a.arg: CONVENTION[a.arg] for a in [*f.args.args, *f.args.kwonlyargs] if a.arg in CONVENTION}
        for n in ast.walk(f):
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                p = path_of(n.value, alias)
                if p and _is_section(p):
                    t = n.targets[0]
                    if isinstance(t, ast.Name):
                        alias[t.id] = p
                    elif isinstance(t, ast.Attribute):
                        attr_alias[t.attr] = p
        scopes.append((f, alias))
    for f, alias in scopes:
        for n in ast.walk(f):
            if isinstance(n, (ast.Subscript, ast.Call)):
                p = path_of(n, alias)
                if p:
                    yield p, n.lineno


class TestHomePrecedence(unittest.TestCase):
    """F53: $FIBBOT_HOME (Docker: /data) wins over a config file's "home"; --home wins over both."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.path = self.tmp / "fibbot.json"
        self.path.write_text(json.dumps({"home": str(self.tmp / "from-file")}))
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("FIBBOT_CONFIG", None)

    def test_env_beats_the_file(self):
        os.environ["FIBBOT_HOME"] = str(self.tmp / "data")
        with self.assertLogs("fibbot.config", "WARNING") as logs:
            cfg = load_config(self.path)
        self.assertEqual(cfg["home"], str((self.tmp / "data").resolve()))
        self.assertIn("takes precedence", logs.output[0])

    def test_home_override_beats_the_env(self):
        os.environ["FIBBOT_HOME"] = str(self.tmp / "data")
        cfg = load_config(self.path, {"home": str(self.tmp / "cli")})
        self.assertEqual(cfg["home"], str((self.tmp / "cli").resolve()))

    def test_the_file_is_used_without_the_env(self):
        os.environ.pop("FIBBOT_HOME", None)
        self.assertEqual(load_config(self.path)["home"], str((self.tmp / "from-file").resolve()))

    def test_a_non_string_home_is_rejected(self):
        self.path.write_text(json.dumps({"home": 5}))
        with self.assertRaisesRegex(ValueError, "home: must be a path string"):
            load_config(self.path)


class TestCandleStall(SchedulerCase):
    """F33: a total download failure or stale candles fail the data step (journaled once per
    stall, with a recovery entry), and `run --once` exits 1."""

    def stall_journal(self):
        return [j["title"] for j in self.system_journal()]

    def test_total_download_failure_fails_the_step_and_is_journaled_once(self):
        self.lab.candles.errors = "HTTP 451: Unavailable For Legal Reasons"
        for k in range(3):
            st = self.steps(self.go(now=NOW + k * HOUR))
            self.assertEqual(st["data"]["status"], "failed", k)
            self.assertIn("all 3 candle updates failed", st["data"]["detail"])
            self.assertIn("HTTP 451", st["data"]["detail"])
            self.assertIn(f"since {scheduler.iso_time(NOW)}", st["data"]["detail"])
        titles = self.stall_journal()
        self.assertEqual(len(titles), 1, titles)
        self.assertTrue(titles[0].startswith("Candle data stalled: all 3 candle updates failed"))
        self.assertEqual(self.store.get_kv(DATA_STALL_KEY)["since"], NOW)
        self.lab.candles.errors = None                             # the exchange answers again
        self.lab.candles.end = NOW + 3 * HOUR
        st = self.steps(self.go(now=NOW + 3 * HOUR))
        self.assertEqual(st["data"]["status"], "ok")
        self.assertEqual(self.stall_journal()[0], "Candle data recovered after 3.0 h")
        self.assertIsNone(self.store.get_kv(DATA_STALL_KEY))
        self.go(now=NOW + 4 * HOUR)
        self.assertEqual(len(self.stall_journal()), 2)

    def test_one_failing_symbol_is_not_a_stall(self):
        st = self.steps(self.go())
        self.assertEqual((st["data"]["status"], st["data"]["detail"]["failed"]), ("ok", 1))
        self.assertEqual(self.system_journal(), [])

    def test_stale_candles_of_an_open_position_fail_the_step(self):
        self.add("s1")
        self.store.upsert_paper_trade("s1", open_trade("s1", "S0USDT"))
        self.lab.candles.ends[("S0USDT", "4h")] = NOW - (STALE_BARS + 1) * TF_SECONDS["4h"]
        st = self.steps(self.go())
        self.assertEqual(st["data"]["status"], "failed")
        self.assertIn("stale candles for open paper positions: S0USDT 4h", st["data"]["detail"])
        self.lab.candles.ends[("S0USDT", "4h")] = NOW - STALE_BARS * TF_SECONDS["4h"]
        self.assertEqual(self.steps(self.go(now=NOW))["data"]["status"], "ok")

    def test_a_whole_timeframe_without_new_bars_fails_the_step(self):
        # every request "succeeds" but no bar arrives (e.g. an API change returning [])
        self.lab.candles.end = NOW - (STALE_BARS * TF_SECONDS["4h"] + 1)
        st = self.steps(self.go())
        self.assertEqual(st["data"]["status"], "failed")
        self.assertIn("no 4h bar of any symbol after", st["data"]["detail"])

    def test_the_review_cycle_waits_for_fresh_candles(self):
        self.add(start=NOW - 30 * 86400)
        self.lab.candles.errors = "HTTP 403"
        st = self.steps(self.go())
        self.assertEqual(st["review"]["detail"], "waiting for fresh candles (the data is stalled)")
        self.assertNotIn("review_cycle", self.lab.names())
        self.assertIn("research_cycle", self.lab.names())     # research uses the cached history

    def test_run_once_exits_one_on_a_stall(self):
        self.lab.candles.errors = "HTTP 451"
        with tempfile.TemporaryDirectory() as home, \
                mock.patch("fibbot.scheduler.Lab", return_value=self.lab), \
                mock.patch("fibbot.scheduler.setup_logging"), \
                mock.patch.dict(os.environ, {"FIBBOT_CONFIG": ""}):
            rc = cli.main(["--home", home, "run", "--once", "--no-brain"])
        self.assertEqual(rc, 1)


class TestTickExitCode(unittest.TestCase):
    """F33: `tick` exits 1 when the tick reported errors (e.g. every candle download failed)."""

    def test_tick_errors_exit_one(self):
        out = {"strategies": {}, "new_signals": [],
               "errors": [{"strategy_id": None, "symbol": "BTCUSDT", "error": "data update 4h: HTTP 451"}]}
        lab = mock.MagicMock()
        lab.paper.return_value.tick.return_value = out
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as home, mock.patch("fibbot.cli.Lab", return_value=lab), \
                mock.patch.dict(os.environ, {"FIBBOT_CONFIG": ""}), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = cli.main(["--home", home, "tick"])
        self.assertEqual(rc, 1)
        self.assertIn("error: strategy None symbol BTCUSDT: data update 4h: HTTP 451", stdout.getvalue())
        self.assertIn("1 errors during the tick", stderr.getvalue())
        lab.store.close.assert_called_once_with()


class TestFailureJournal(SchedulerCase):
    """F48: a persistent failure adds one journal entry a day, not one an hour."""

    def test_identical_failures_are_journaled_once_per_period(self):
        self.lab.fail = {"universe"}
        for k in range(FAILURE_REPEAT_HOURS):
            st = self.steps(self.go(now=NOW + k * HOUR))
            self.assertEqual(st["universe"]["status"], "failed")
        rows = self.system_journal()
        self.assertEqual(len(rows), 1, [r["title"] for r in rows])
        self.go(now=NOW + FAILURE_REPEAT_HOURS * HOUR)
        rows = self.system_journal()
        self.assertEqual(len(rows), 2)
        self.assertIn(f"also happened {FAILURE_REPEAT_HOURS - 1} more times", rows[0]["body"])
        self.assertIn("Traceback", rows[0]["body"])

    def test_a_different_failure_is_journaled_at_once(self):
        errors = iter(["DNS failure", "HTTP 500", "HTTP 500"])

        def universe():
            raise RuntimeError(next(errors))
        self.lab.universe = universe
        for k in range(3):
            self.go(now=NOW + k * HOUR)
        self.assertEqual([j["title"] for j in self.system_journal()],
                         ["Scheduler step universe failed: RuntimeError: HTTP 500",
                          "Scheduler step universe failed: RuntimeError: DNS failure"])
        self.lab.fail = {"write_report"}                      # another step: its own entry
        self.lab.universe = lambda: ["S0USDT"]
        self.go(now=NOW + 3 * HOUR)
        self.assertIn("step report", self.system_journal()[0]["title"])


class TestLogRotation(unittest.TestCase):
    """F48: fibbot.log (Docker keeps it in the volume) is rotated instead of growing forever."""

    def test_the_log_file_rotates(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.object(scheduler, "LOG_MAX_BYTES", 300):
            handlers = scheduler.setup_logging(cfg_(home), io.StringIO())
            logger = logging.getLogger("fibbot")
            try:
                file_handler, = [h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
                self.assertEqual(file_handler.backupCount, scheduler.LOG_BACKUPS)
                for k in range(40):
                    logging.getLogger("fibbot.x").info("line %d %s", k, "y" * 40)
            finally:
                for h in handlers:
                    logger.removeHandler(h)
                    h.close()
                scheduler._handlers.clear()
            names = sorted(p.name for p in Path(home).iterdir() if p.name.startswith("fibbot.log"))
        self.assertEqual(names, ["fibbot.log", "fibbot.log.1", "fibbot.log.2", "fibbot.log.3"])


class TestCycleStop(SchedulerCase):
    """F13 / F49: a brain cycle that accepts `stop` learns about the time limit
    (brain.max_cycle_minutes) and about a stop signal between its turns."""

    def fake_cycle(self, turns_done, minutes_per_turn=20.0, on_turn=None):
        clock = {"t": 0.0}

        def research_cycle(lab, brain, stop=None):
            lab.record("research_cycle")
            reason = ""
            for k in range(10):
                clock["t"] += minutes_per_turn * 60
                turns_done.append(k)
                if on_turn:
                    on_turn(k)
                reason = stop() if stop else ""
                if reason:
                    break
            return {**self.cycle_out, "kind": "research", "turns": len(turns_done),
                    "stop_reason": reason or "end_turn"}
        return research_cycle, clock

    def test_cycle_stop_reports_the_deadline_and_a_signal(self):
        t = {"now": 100.0}
        flag = {"stop": False}
        s = cycle_stop(cfg_("/x", brain={"max_cycle_minutes": 30}), lambda: flag["stop"], clock=lambda: t["now"])
        self.assertEqual(s(), "")
        t["now"] += 29 * 60
        self.assertEqual(s(), "")
        t["now"] += 60
        self.assertIn("brain.max_cycle_minutes (30 minutes)", s())
        flag["stop"] = True
        self.assertEqual(s(), "the bot is shutting down")
        self.assertEqual(cycle_stop(cfg_("/x", brain={"max_cycle_minutes": 0}), clock=lambda: 1e12)(), "")

    def test_a_cycle_stops_at_the_time_limit(self):
        self.cfg["brain"]["max_cycle_minutes"] = 60
        turns = []
        fn, clock = self.fake_cycle(turns)
        with mock.patch("fibbot.brain.cycles.research_cycle", fn), \
                mock.patch("fibbot.scheduler.time.monotonic", side_effect=lambda: clock["t"]):
            st = self.steps(self.go())
        self.assertEqual(len(turns), 3)                          # 20 min per turn, 60 min limit
        self.assertEqual(st["research"]["status"], "ok")
        self.assertIn("max_cycle_minutes", st["research"]["detail"]["stop_reason"])

    def test_a_stop_signal_reaches_a_running_cycle(self):
        turns, flag = [], {"stop": False}

        def on_turn(k):
            if k == 1:
                flag["stop"] = True                              # SIGTERM during the 2nd turn
        fn, _ = self.fake_cycle(turns, minutes_per_turn=0.0, on_turn=on_turn)
        with mock.patch("fibbot.brain.cycles.research_cycle", fn):
            out = self.go(stop=lambda: flag["stop"])
        self.assertEqual(len(turns), 2)
        self.assertEqual(self.steps(out)["research"]["detail"]["stop_reason"], "the bot is shutting down")
        self.assertTrue(out["stopped"])                          # and the pass ends after the step
        self.assertEqual(out["steps"][-1]["step"], "research")

    def test_signal_reviews_that_accept_stop_get_it(self):
        self.add()
        seen = []

        def review_signals(lab, brain, signals=None, stop=None):
            seen.append(stop)
            return []
        with mock.patch("fibbot.brain.cycles.review_signals", review_signals):
            self.go()
        self.assertTrue(callable(seen[0]))
        self.assertEqual(seen[0](), "")

    def test_cycles_without_a_stop_parameter_are_called_as_before(self):
        def research_cycle(lab, brain):
            lab.record("research_cycle")
            return {**self.cycle_out, "kind": "research"}
        with mock.patch("fibbot.brain.cycles.research_cycle", research_cycle):
            st = self.steps(self.go())
        self.assertEqual(st["research"]["status"], "ok")
        self.assertFalse(scheduler.accepts(mock.MagicMock(), "stop"))

    def test_the_cli_brain_command_passes_the_time_limit(self):
        seen = {}

        def research_cycle(lab, brain, stop=None):
            seen["stop"] = stop
            return {"kind": "research", "ok": True, "text": "", "stop_reason": "end_turn", "turns": 1,
                    "refused": False, "promoted": [], "retired": [], "journal_id": 1,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}}
        with tempfile.TemporaryDirectory() as home, \
                mock.patch("fibbot.scheduler.make_brain", return_value=FakeBrain()), \
                mock.patch("fibbot.brain.cycles.research_cycle", research_cycle), \
                mock.patch("fibbot.cli.Lab", return_value=mock.MagicMock()), \
                mock.patch.dict(os.environ, {"FIBBOT_CONFIG": ""}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--home", home, "brain", "research"]), 0)
        self.assertTrue(callable(seen["stop"]))
        self.assertEqual(seen["stop"](), "")


if __name__ == "__main__":
    unittest.main()
