"""Tests for fibbot.lab: data access, research calls (experiments + trials), cycle budgets, the
promote/retire gating, status, the signal-review value split and the brain notebook.

Synthetic data only (no network): get_universe is patched and candles come from FakeCandles.
tests/test_autopilot.py and tests/test_report.py reuse LabCase, swing_bars and the trade helpers.
"""
import json
import math
import random
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fibbot import lab as lab_mod
from fibbot import research
from fibbot.config import DEFAULTS, deep_merge
from fibbot.data.candles import CandleStore
from fibbot.lab import NOTEBOOK_HEADER, TRUNCATED, BudgetExceeded, Lab, iso_time
from fibbot.store import Store
from fibbot.strategy import ENGINE_VERSION, PRESETS
from fibbot.types import TF_SECONDS, Bars, Signal, Trade

T0 = 1_600_041_600            # 2020-09-14 00:00 UTC
H4 = TF_SECONDS["4h"]
SYMS = ["S0USDT", "S1USDT", "S2USDT"]
SPEC = PRESETS["deep_786_4h"]
OTHER = PRESETS["golden_pocket_4h"]


def swing_bars(symbol, n=2000, seed=0, tf="4h", start=100.0, t0=T0):
    """Seeded impulse / pullback walk without a strong drift: up-legs of 10-25% over 6-14 bars,
    then pullbacks of 0.55-1.35x the leg. Fib entries that hold beat random entries here, so some
    presets are eligible in a sweep, others are not."""
    rnd = random.Random(seed)
    closes, px = [], start
    while len(closes) < n:
        low = px
        n_up, up = rnd.randint(6, 14), rnd.uniform(0.10, 0.25)
        for _ in range(n_up):
            px *= (1 + up) ** (1 / n_up) * math.exp(rnd.gauss(0, 0.004))
            closes.append(px)
        n_dn = rnd.randint(4, 10)
        step = ((px - rnd.uniform(0.55, 1.35) * (px - low)) / px) ** (1 / n_dn)
        for _ in range(n_dn):
            px *= step * math.exp(rnd.gauss(0, 0.004))
            closes.append(px)
    rows, prev = [], start
    for i, c in enumerate(closes[:n]):
        w = abs(rnd.gauss(0, 0.004))
        rows.append((t0 + i * TF_SECONDS[tf], prev, max(prev, c) * (1 + w), min(prev, c) * (1 - w), c, 1.0))
        prev = c
    return Bars.from_rows(symbol, tf, rows)


BARS = {sym: swing_bars(sym, seed=i) for i, sym in enumerate(SYMS)}


class FakeCandles:
    """CandleStore stand-in serving fixed bars (empty Bars for anything else)."""

    def __init__(self, bars=None):
        self.bars = dict(BARS if bars is None else bars)

    def get(self, symbol, tf):
        b = self.bars.get(symbol)
        return b if b is not None and b.tf == tf else Bars(symbol, tf)


def make_cfg(home, **over):
    """Defaults for a 2000-bar synthetic history: serial, 3-month validation, 3 folds."""
    base = deep_merge(DEFAULTS, {"home": str(home), "research": {
        "workers": 1, "validation_months": 3, "folds": 3, "min_trades": 5}})
    return deep_merge(base, over)


def mk_trade(sid, symbol, leg, entry_t, r, closed=True, risk=0.02, side=1):
    """Hand-made trade with ret_pct = r * risk, closed 3 bars after entry (or open)."""
    ret = r * risk
    return Trade(sid, symbol, "4h", side, leg, entry_t - H4, entry_t, 100.0, 100.0 * (1 - side * risk),
                 entry_t + 3 * H4 if closed else None, 100.0 * (1 + side * ret),
                 ("target" if r > 0 else "stop") if closed else "open", ret, risk, r, 3,
                 {"tp1": 104.0, "tp2": 106.0})


def mk_signal(sid, symbol, leg, t):
    return Signal(sid, symbol, "4h", 1, leg, t, 100.0, 98.0, 104.0, 106.0, {"rr": 2.0})


def record_validation(store, spec, passed=True, avg_r=0.3, checks=(), now_t=None, baseline=True):
    """A validate experiment as Lab.validate would store it (independent of gate outcomes), by
    the current engine version; with `baseline` also the research-window baseline experiment
    (an edge over random entries) that Lab.promote requires."""
    if baseline:
        store.add_experiment("baseline", spec.to_dict(), spec.spec_id(), T0, T0 + 50 * H4,
                             {"edge_r": 0.2, "edge_regime_r": 0.1, "n_runs": 10,
                              "engine_version": ENGINE_VERSION}, now_t=now_t)
    return store.add_experiment("validate", spec.to_dict(), spec.spec_id(), T0, T0 + 100 * H4,
                                {"passed": passed, "checks": list(checks), "avg_r": avg_r, "n": 40,
                                 "profit_factor": 1.5, "deflated_sharpe": 0.97,
                                 "engine_version": ENGINE_VERSION}, now_t=now_t)


class LabCase(unittest.TestCase):
    """Temp home, temp Store, FakeCandles and a patched universe."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        patcher = mock.patch("fibbot.lab.get_universe", return_value=list(SYMS))
        self.get_universe = patcher.start()
        self.addCleanup(patcher.stop)
        self.store = Store(self.home / "lab.sqlite")
        self.addCleanup(self.store.close)
        self.candles = FakeCandles()
        self.lab = self.make_lab()

    def make_lab(self, candles=None, **over):
        self.cfg = make_cfg(self.home, **over)
        return Lab(self.cfg, self.store, self.candles if candles is None else candles)

    def research_bounds(self):
        return research.window_bounds(self.lab.windows("4h"), "research")


class TestData(LabCase):
    def test_default_store_and_candles_live_in_home(self):
        lab = Lab(self.cfg)
        self.addCleanup(lab.store.close)
        self.assertIsInstance(lab.candles, CandleStore)
        self.assertEqual(Path(lab.store.path), self.home / "fibbot.sqlite")
        self.assertTrue((self.home / "fibbot.sqlite").exists())

    def test_universe_is_memoized_and_refreshable(self):
        u = self.lab.universe()
        u.append("XUSDT")                       # callers get a copy
        self.assertEqual(self.lab.universe(), SYMS)
        self.get_universe.assert_called_once_with(self.cfg, refresh=False)
        self.lab.universe(refresh=True)
        self.get_universe.assert_called_with(self.cfg, refresh=True)
        with mock.patch("fibbot.lab.time.monotonic", return_value=1e12):   # memo expired
            self.lab.universe()
        self.assertEqual(self.get_universe.call_count, 3)

    def test_bars_only_symbols_with_data(self):
        self.get_universe.return_value = SYMS + ["NODATAUSDT"]
        self.assertEqual(list(self.lab.bars("4h")), SYMS)
        self.assertIs(self.lab.bars("4h")["S0USDT"], BARS["S0USDT"])
        self.assertEqual(list(self.lab.bars("4h", "s1usdt")), ["S1USDT"])     # a bare string is one symbol
        self.assertEqual(list(self.lab.bars("4h", ["S2USDT", "S2USDT", "NODATAUSDT"])), ["S2USDT"])
        self.assertEqual(self.lab.bars("1h"), {})
        with self.assertRaises(ValueError):
            self.lab.bars("5m")

    def test_windows_come_from_universe_bars(self):
        self.assertEqual(self.lab.windows("4h"), research.compute_windows(self.cfg, BARS))
        with self.assertRaisesRegex(ValueError, "no cached 1h candles"):
            self.lab.windows("1h")
        lab = self.make_lab(research={"validation_months": 24})
        with self.assertRaisesRegex(ValueError, "too short"):
            lab.windows("4h")


class TestResearchCalls(LabCase):
    def test_backtest_records_experiment_and_trial(self):
        out = self.lab.backtest(SPEC)
        res, extras = research.evaluate(SPEC, BARS, self.cfg, self.lab.windows("4h"), "research")
        self.assertEqual({k: v for k, v in out.items() if k not in ("window", "experiment_id")},
                         {**res.summary(), **extras})
        self.assertEqual(out["window"], "research")
        self.assertGreater(out["metrics"]["n"], 20)
        exp = self.store.get_experiment(out["experiment_id"])
        self.assertEqual((exp["kind"], exp["source"], exp["spec_id"]), ("backtest", "cli", SPEC.spec_id()))
        self.assertEqual((exp["window_start"], exp["window_end"]), self.research_bounds())
        self.assertEqual(exp["metrics"]["avg_r"], out["metrics"]["avg_r"])
        self.assertEqual(exp["metrics"]["fold_positive_frac"], out["fold_positive_frac"])
        self.assertEqual(exp["spec"], SPEC.to_dict())
        self.assertEqual(self.store.trial_stats()["n"], 1)
        json.dumps(out, allow_nan=False)

    def test_backtest_can_list_its_last_trades(self):
        self.assertNotIn("trades", self.lab.backtest(SPEC))
        out = self.lab.backtest(SPEC, trades=3)
        res, _ = research.evaluate(SPEC, BARS, self.cfg, self.lab.windows("4h"), "research")
        self.assertEqual([(t["symbol"], t["leg_id"]) for t in out["trades"]],
                         [(tr.symbol, tr.leg_id) for tr in res.trades[-3:]])
        last = res.trades[-1]
        self.assertEqual(out["trades"][-1]["entry_t"], iso_time(last.entry_t))
        self.assertEqual(out["trades"][-1]["exit_t"], iso_time(last.exit_t))
        self.assertAlmostEqual(out["trades"][-1]["r_multiple"], last.r_multiple, places=3)
        self.assertNotIn("trades", self.store.get_experiment(out["experiment_id"])["metrics"])
        json.dumps(out, allow_nan=False)

    def test_backtest_takes_dict_spec_and_symbol_subset(self):
        out = self.lab.backtest(SPEC.to_dict(), symbols=["S1USDT"])
        self.assertEqual((out["n_symbols"], list(out["per_symbol"])), (1, ["S1USDT"]))
        self.assertEqual(out["period"]["start"], iso_time(self.research_bounds()[0])[:10])
        self.assertIn("symbols=S1USDT", self.store.get_experiment(out["experiment_id"])["notes"])
        with self.assertRaisesRegex(ValueError, "no cached 4h candles"):
            self.lab.backtest(SPEC, symbols=["NODATAUSDT"])

    def test_non_research_windows_are_holdout_peeks(self):
        # A look at the held-out period is a trial (it used to be free: F16) and its numbers are
        # kept out of the experiment's top-level metrics, which the brain's tools list.
        for k, window in enumerate(("validation", "all"), 1):
            out = self.lab.backtest(SPEC, window=window)
            self.assertEqual((out["window"], out["holdout_looks"]), (window, k))
            exp = self.store.get_experiment(out["experiment_id"])
            self.assertTrue(exp["notes"].startswith(f"window={window}"))
            self.assertEqual(exp["source"], "cli-holdout")
            self.assertNotIn("avg_r", exp["metrics"])
            self.assertEqual(exp["metrics"]["holdout"]["result"]["avg_r"], out["metrics"]["avg_r"])
            self.assertEqual(self.store.trial_stats()["n"], k)        # the spec, then a 2nd distinct look
        self.assertEqual(self.store.trial_stats()["var_sharpe"], 0.0)
        with self.assertRaises(ValueError):
            self.lab.backtest(SPEC, window="future")

    def test_invalid_spec_records_nothing(self):
        for bad in ({"entry_zone_lo": 5.0}, {"no_such_param": 1}, {"timeframe": "1h"}):
            with self.assertRaises(ValueError):
                self.lab.backtest(bad)
        self.assertEqual(self.store.count_experiments(), 0)
        self.assertEqual(self.store.trial_stats()["n"], 0)

    def test_baseline_reuses_the_research_evaluation(self):
        with mock.patch("fibbot.research.backtest", wraps=research.backtest) as bt:
            self.lab.backtest(SPEC)
            out = self.lab.baseline(SPEC, n_runs=5)
        self.assertEqual(bt.call_count, 1)
        expected = research.random_baseline(SPEC, BARS, self.cfg, *self.research_bounds(), n_runs=5)
        self.assertEqual({k: v for k, v in out.items() if k not in ("spec_id", "experiment_id")}, expected)
        exp = self.store.get_experiment(out["experiment_id"])
        self.assertEqual((exp["kind"], exp["metrics"]["edge_r"]), ("baseline", out["edge_r"]))
        for n_runs in (0, lab_mod.MAX_BASELINE_RUNS + 1):
            with self.assertRaises(ValueError):
                self.lab.baseline(SPEC, n_runs=n_runs)

    def test_memo_keeps_only_recent_evaluations(self):
        specs = [SPEC, OTHER, SPEC.with_overrides(min_rr=1.0)]
        with mock.patch("fibbot.lab.MEMO_SIZE", 2), \
                mock.patch("fibbot.research.backtest", wraps=research.backtest) as bt:
            for spec in specs + [specs[2], specs[0]]:
                self.lab.backtest(spec)
        self.assertEqual(bt.call_count, 4)      # specs[2] was cached, specs[0] had been evicted

    def test_optimize_records_every_distinct_combo_as_a_trial(self):
        # F55: a combo whose trades duplicate an earlier one's is evaluated but is not another
        # trial; on these legs min_rr 1.0 and 1.5 trade identically
        grid = {"min_rr": [1.0, 1.5], "stop_atr_buffer": [0.1, 0.3]}
        out = self.lab.optimize(SPEC, grid)
        self.assertEqual((out["n_combos"], out["n_evaluated"], out["n_duplicates"]), (4, 4, 2))
        self.assertEqual(self.store.trial_stats()["n"], 4 - out["n_duplicates"])
        exp = self.store.get_experiment(out["experiment_id"])
        self.assertEqual((exp["kind"], exp["spec_id"]), ("optimize", SPEC.spec_id()))
        self.assertEqual(exp["metrics"]["avg_r"], out["oos"]["avg_r"])
        self.assertEqual(json.loads(exp["notes"].removeprefix("grid=")), grid)
        with self.assertRaises(ValueError):
            self.lab.optimize(SPEC, [["min_rr", [1.0]]])

    def test_robustness_evaluates_neighbours_as_trials(self):
        out = self.lab.robustness(SPEC)
        n = min(len(research.neighbours(SPEC)), lab_mod.ROBUSTNESS_NEIGHBOURS)
        self.assertEqual((out["n"], out["spec_id"]), (n, SPEC.spec_id()))
        self.assertEqual(n, len(research.neighbours(SPEC)))          # the whole neighbourhood
        self.assertEqual(self.store.trial_stats()["n"], out["n_distinct"])   # F55: no-op moves are not trials
        self.assertEqual(self.store.get_experiment(out["experiment_id"])["kind"], "robustness")

    def test_validate_records_gate_and_counts_itself_as_a_trial(self):
        out = self.lab.validate(SPEC)
        self.assertEqual(out["n_trials"], 1)                 # fresh store: the spec itself
        self.assertEqual(out["window"], "validation")
        self.assertEqual(out["passed"], out["gate"]["passed"])
        latest = self.store.latest_validation(SPEC.spec_id())
        self.assertEqual(latest["id"], out["experiment_id"])
        self.assertEqual(latest["metrics"]["passed"], out["passed"])
        self.assertEqual(latest["metrics"]["checks"], out["gate"]["checks"])
        self.assertEqual(latest["metrics"]["avg_r"], out["metrics"]["avg_r"])
        self.assertEqual((latest["window_start"], latest["window_end"]),
                         research.window_bounds(self.lab.windows("4h"), "validation"))
        self.lab.backtest(OTHER)
        self.assertEqual(self.lab.validate(SPEC)["n_trials"], 2)
        json.dumps(out, allow_nan=False)


class TestBudget(LabCase):
    def setUp(self):
        super().setUp()
        self.lab = self.make_lab(research={"max_backtest_units_per_cycle": 5, "max_validations_per_cycle": 1})

    def test_unbudgeted_outside_cycles(self):
        self.assertEqual(self.lab.budget(), {"backtest_units_left": None, "validations_left": None})
        self.assertEqual((self.lab.cycle, self.lab.source), (None, "cli"))
        for _ in range(7):
            self.lab.backtest(SPEC)

    def test_cycle_counts_units_and_validations(self):
        self.lab.begin_cycle("research")
        self.assertEqual(self.lab.budget(), {"backtest_units_left": 5, "validations_left": 1})
        out = self.lab.backtest(SPEC)
        self.assertEqual(self.store.get_experiment(out["experiment_id"])["source"], "brain")
        self.lab.baseline(SPEC)
        self.lab.validate(SPEC)
        self.assertEqual(self.lab.budget(), {"backtest_units_left": 3, "validations_left": 0})
        with self.assertRaisesRegex(BudgetExceeded, "validations"):
            self.lab.validate(OTHER)
        self.assertIsNone(self.store.latest_validation(OTHER.spec_id()))
        self.lab.begin_cycle("research")                     # a new cycle resets the counters
        self.assertEqual(self.lab.budget(), {"backtest_units_left": 5, "validations_left": 1})
        self.lab.end_cycle()
        self.assertEqual(self.lab.budget()["backtest_units_left"], None)

    def test_exceeding_the_budget_runs_nothing(self):
        self.lab.begin_cycle("research")
        n_exp = self.store.count_experiments()
        with self.assertRaisesRegex(BudgetExceeded, "needs 6 backtest units, 5 left"):
            self.lab.optimize(SPEC, {"min_rr": [1.0, 1.5, 2.0], "stop_atr_buffer": [0.1, 0.3]})
        self.assertEqual(self.store.count_experiments(), n_exp)
        self.assertEqual(self.store.trial_stats()["n"], 0)
        self.assertEqual(self.lab.budget()["backtest_units_left"], 5)
        for _ in range(5):
            self.lab.backtest(SPEC)
        with self.assertRaises(BudgetExceeded):
            self.lab.backtest(SPEC)
        self.assertIsInstance(BudgetExceeded("x"), RuntimeError)

    def test_failed_calls_are_refunded(self):
        self.lab.begin_cycle("research")
        with self.assertRaises(ValueError):
            self.lab.backtest(PRESETS["golden_pocket_1h"])          # no 1h data
        with self.assertRaises(ValueError):
            self.lab.optimize(SPEC, {"min_rr": "1.0"})               # not a list
        self.assertEqual(self.lab.budget()["backtest_units_left"], 5)

    def test_optimize_refunds_combos_it_did_not_evaluate(self):
        self.lab.begin_cycle("research")
        out = self.lab.optimize(SPEC, {"min_rr": [1.5, 1.5], "entry_zone_hi": [0.886, 0.95]})
        self.assertEqual((out["n_combos"], out["n_evaluated"], out["n_invalid"]), (4, 1, 2))
        self.assertEqual(self.lab.budget()["backtest_units_left"], 4)

    def test_robustness_costs_one_unit_per_neighbour(self):
        self.lab.begin_cycle("research")
        with self.assertRaises(BudgetExceeded):
            self.lab.robustness(SPEC)                                # > 5 neighbours
        lab = self.make_lab(research={"max_backtest_units_per_cycle": 40})
        lab.begin_cycle("research")
        out = lab.robustness(SPEC)
        # one unit per neighbour plus one for the base spec (evaluated to spot no-op moves)
        self.assertEqual(lab.budget()["backtest_units_left"], 40 - out["n"] - 1)

    def test_validation_window_is_locked_inside_a_cycle(self):
        self.lab.begin_cycle("research")
        for window in ("validation", "all"):
            with self.assertRaisesRegex(ValueError, "validate"):
                self.lab.backtest(SPEC, window=window)
        self.assertEqual(self.lab.budget()["backtest_units_left"], 5)
        self.lab.end_cycle()
        self.lab.backtest(SPEC, window="validation")

    def test_review_cycle_gets_a_smaller_budget(self):
        lab = self.make_lab(research={"max_backtest_units_per_cycle": 200, "max_validations_per_cycle": 3})
        lab.begin_cycle("review")
        self.assertEqual(lab.cycle, "review")
        self.assertEqual(lab.budget(), {"backtest_units_left": 50, "validations_left": 1})
        lab.begin_cycle("research")
        self.assertEqual(lab.budget(), {"backtest_units_left": 200, "validations_left": 3})


class TestPromoteRetire(LabCase):
    NOW = 1_700_000_123            # 2023-11-14 22:15:23 UTC, inside a 4h bar

    def promote(self, spec=SPEC, name="deep pullbacks", now_t=NOW):
        return self.lab.promote(spec, name, "beats random entries", now_t=now_t)

    def test_promote_needs_a_passing_latest_validation(self):
        with self.assertRaisesRegex(ValueError, "no validation"):
            self.promote()
        record_validation(self.store, SPEC, passed=True, now_t=1)
        record_validation(self.store, SPEC, passed=False, now_t=2,
                          checks=[{"name": "min_trades", "passed": False}, {"name": "max_drawdown", "passed": True}])
        with self.assertRaisesRegex(ValueError, "did not pass: min_trades$"):
            self.promote()
        self.store.add_experiment("validate", SPEC.to_dict(), SPEC.spec_id(), 0, 1, {"avg_r": 0.5}, now_t=3)
        with self.assertRaisesRegex(ValueError, "did not pass$"):         # no verdict is not a pass
            self.promote()
        self.assertEqual(self.store.roster(None), [])

    def test_promote_starts_paper_at_the_next_bar(self):
        eid = record_validation(self.store, SPEC, avg_r=0.31)
        out = self.promote()
        start = (self.NOW // H4 + 1) * H4
        self.assertEqual(out, {"strategy_id": SPEC.spec_id(), "name": "deep pullbacks", "timeframe": "4h",
                               "paper_start_t": start, "paper_start": iso_time(start),
                               "validation_experiment_id": eid, "roster_size": 1, "max_roster": 4})
        row = self.store.get_strategy(SPEC.spec_id())
        self.assertEqual((row["status"], row["paper_start_t"], row["rationale"]),
                         ("paper", start, "beats random entries"))
        self.assertEqual(row["spec"], SPEC.with_overrides(name="deep pullbacks").to_dict())
        self.assertEqual(row["validation"]["metrics"]["avg_r"], 0.31)
        self.assertEqual(row["validation"]["experiment_id"], eid)
        j = self.store.journal(kind="promotion")[0]
        self.assertIn("deep pullbacks", j["title"])
        self.assertIn("beats random entries", j["body"])
        record_validation(self.store, OTHER)
        self.assertEqual(self.lab.promote(OTHER, "gp", "x", now_t=start)["paper_start_t"], start + H4)

    def test_promote_rejects_duplicates_and_a_full_roster(self):
        self.lab = self.make_lab(paper={"max_roster": 1})
        record_validation(self.store, SPEC)
        record_validation(self.store, OTHER)
        self.promote()
        with self.assertRaisesRegex(ValueError, "already on the roster"):
            self.promote(name="again")
        with self.assertRaisesRegex(ValueError, r"roster is full \(1/1\)"):
            self.promote(OTHER, "gp")

    def test_retire_is_final(self):
        record_validation(self.store, SPEC)
        sid = self.promote()["strategy_id"]
        with self.assertRaises(KeyError):
            self.lab.retire("nope", "x")
        out = self.lab.retire(sid, "drawdown too deep")
        self.assertEqual((out["status"], out["reason"], out["account"]["status"]),
                         ("retired", "drawdown too deep", "retired"))
        self.assertEqual(self.store.get_strategy(sid)["status_reason"], "drawdown too deep")
        self.assertIn("drawdown too deep", self.store.journal(kind="retirement")[0]["body"])
        with self.assertRaisesRegex(ValueError, "already retired"):
            self.lab.retire(sid, "again")
        # A retired id cannot be re-promoted (it used to be re-activated with a fresh account,
        # erasing a losing paper record and re-drawing it; see store.add_to_roster).
        with self.assertRaisesRegex(ValueError, r"retired .*drawdown too deep.*final"):
            self.promote(now_t=self.NOW + 10 * H4)
        self.assertEqual(self.store.get_strategy(sid)["status"], "retired")

    def test_real_validation_then_promotion(self):
        lab = self.make_lab(research={"validation_gate": {"min_deflated_sharpe": 0.0, "min_profit_factor": 1.0,
                                                          "min_expectancy_r": 0.0,
                                                          "min_validation_edge_z": None}})
        v = lab.validate(SPEC)
        self.assertTrue(v["passed"], v["gate"])
        with self.assertRaisesRegex(ValueError, "no research-window baseline"):
            lab.promote(SPEC, "deep", "validated", now_t=self.NOW)
        lab.baseline(SPEC)
        self.assertEqual(lab.promote(SPEC, "deep", "validated", now_t=self.NOW)["validation_experiment_id"],
                         v["experiment_id"])


class TestStatus(LabCase):
    NOW = 1_700_000_123

    def setUp(self):
        super().setUp()
        record_validation(self.store, SPEC)
        self.sid = self.lab.promote(SPEC, "deep", "test", now_t=self.NOW)["strategy_id"]
        start = self.store.get_strategy(self.sid)["paper_start_t"]
        for k, r in enumerate((1.5, -1.0, 2.0)):
            self.store.upsert_paper_trade(self.sid, mk_trade(self.sid, SYMS[k], f"L{k}", start + k * H4, r), 100.0, r * 2.0)
        self.store.upsert_paper_trade(self.sid, mk_trade(self.sid, "S0USDT", "L9", start + 9 * H4, 0.4, closed=False),
                                      100.0, 0.8)
        self.now = start + 30 * 86400

    def test_status_is_json_safe_and_complete(self):
        self.lab.backtest(OTHER)
        self.store.set_kv("last_research_t", self.NOW)
        st = self.lab.status(now_t=self.now)
        json.dumps(st, allow_nan=False)
        self.assertEqual(st["as_of"], iso_time(self.now))
        self.assertEqual(st["universe"], SYMS)
        self.assertEqual(st["data"], {"4h": {"symbols": 3, "start": "2020-09-14",
                                             "end": iso_time(T0 + 2000 * H4),
                                             "oldest_end": iso_time(T0 + 2000 * H4),
                                             "oldest_symbol": "S0USDT",
                                             "oldest_lag_bars": round((self.now - T0) / H4 - 2000, 1)}})
        w = self.lab.windows("4h")
        self.assertEqual(st["windows"]["4h"]["validation"],
                         [iso_time(w["validation_start"])[:10], iso_time(w["validation_end"])[:10]])
        entry, = st["roster"]
        self.assertEqual((entry["strategy_id"], entry["name"], entry["timeframe"], entry["status"]),
                         (self.sid, "deep", "4h", "paper"))
        self.assertEqual(entry["spec"], self.store.get_strategy(self.sid)["spec"])
        acct = entry["account"]
        self.assertEqual((acct["n_closed"], acct["n_open"], acct["days"]), (3, 1, 30.0))
        self.assertAlmostEqual(acct["avg_r"], 2.5 / 3, places=3)
        grad = self.lab.graduation(self.sid, now_t=self.now)
        self.assertEqual(entry["graduation"]["status"], grad["status"])
        self.assertEqual(entry["graduation"]["failed"], [c["name"] for c in grad["checks"] if not c["passed"]])
        self.assertEqual(st["roster_slots"], {"paper": 1, "max": 4})
        self.assertEqual(st["experiments"]["backtest"], 1)
        self.assertEqual(st["experiments"]["validate"], 1)
        self.assertEqual(st["experiments"]["baseline"], 1)       # record_validation's research baseline
        self.assertEqual(st["experiments"]["total"], 3)
        self.assertEqual(st["trial_stats"]["n"], 1)
        self.assertEqual(st["last"], {"research": iso_time(self.NOW), "review": None, "sweep": None})
        self.assertEqual((st["cycle"], st["budget"]["backtest_units_left"]), (None, None))
        self.assertEqual(st["signal_review"], "shadow")
        self.assertLess(len(json.dumps(st)), 12000)

    def test_retired_strategies_leave_the_roster(self):
        self.lab.retire(self.sid, "enough")
        st = self.lab.status(now_t=self.now)
        self.assertEqual(st["roster"], [])
        self.assertEqual([(r["strategy_id"], r["reason"]) for r in st["retired"]], [(self.sid, "enough")])
        self.assertEqual(st["roster_slots"]["paper"], 0)

    def test_status_without_candles_or_with_short_history(self):
        st = self.make_lab(candles=FakeCandles({})).status(now_t=self.now)
        self.assertEqual((st["data"], st["windows"]), ({}, {}))
        json.dumps(st, allow_nan=False)
        st = self.make_lab(research={"validation_months": 24}).status(now_t=self.now)
        self.assertIn("too short", st["windows"]["4h"]["error"])


class TestSignalReviewStats(LabCase):
    def test_closed_trades_split_by_brain_decision(self):
        t = T0 + 1000 * H4
        cases = [("A", "S0USDT", "L1", 1.0, "take"), ("A", "S0USDT", "L2", -1.0, "take"),
                 ("A", "S1USDT", "L3", 2.0, "take"), ("A", "S1USDT", "L4", -1.0, "skip"),
                 ("A", "S2USDT", "L5", -0.5, "skip"), ("A", "S2USDT", "L6", 0.5, None),
                 ("B", "S0USDT", "L1", 3.0, None)]        # same leg, other strategy: its own signal
        for k, (sid, sym, leg, r, decision) in enumerate(cases):
            self.store.upsert_paper_trade(sid, mk_trade(sid, sym, leg, t + k * H4, r), 100.0, r)
            self.store.add_signal(mk_signal(sid, sym, leg, t + k * H4 - H4))
            if decision:
                self.store.set_signal_review(sid, sym, leg, decision, 0.7, "because")
        self.store.upsert_paper_trade("A", mk_trade("A", "S0USDT", "L7", t, 5.0, closed=False), 100.0, 5.0)
        self.store.add_signal(mk_signal("A", "S0USDT", "L7", t - H4))
        self.store.set_signal_review("A", "S0USDT", "L7", "take", 0.9, "open trades do not count")
        self.store.upsert_paper_trade("A", mk_trade("A", "S1USDT", "L8", t, -2.0), 100.0, -2.0)   # no signal

        out = self.lab.signal_review_stats()
        self.assertEqual(out["mode"], "shadow")
        self.assertEqual(out["reviewed"], 5)
        self.assertEqual(out["take"], {"n": 3, "total_r": 2.0, "avg_r": 0.6667, "win_rate": 0.6667,
                                       "profit_factor": 3.0})
        self.assertEqual((out["skip"]["n"], out["skip"]["avg_r"]), (2, -0.75))
        self.assertEqual((out["unreviewed"]["n"], out["unreviewed"]["total_r"]), (3, 1.5))
        self.assertAlmostEqual(out["skip_minus_take_avg_r"], -0.75 - 2 / 3, places=3)
        self.assertEqual((out["n_signals"], out["n_signals_reviewed"]), (8, 6))
        json.dumps(out, allow_nan=False)

    def test_no_difference_without_both_decisions(self):
        self.store.upsert_paper_trade("A", mk_trade("A", "S0USDT", "L1", T0, 1.0), 100.0, 1.0)
        out = self.lab.signal_review_stats()
        self.assertEqual((out["reviewed"], out["unreviewed"]["n"], out["skip_minus_take_avg_r"]), (0, 1, None))
        self.store.add_signal(mk_signal("A", "S0USDT", "L1", T0 - H4))
        self.store.set_signal_review("A", "S0USDT", "L1", "take", 0.8, "only takes so far")
        out = self.lab.signal_review_stats()
        self.assertEqual((out["reviewed"], out["take"]["n"], out["skip_minus_take_avg_r"]), (1, 1, None))


class TestNotebook(LabCase):
    def path(self):
        return self.home / "BRAIN_NOTEBOOK.md"

    def test_missing_notebook_is_empty(self):
        self.assertEqual(self.lab.notebook(), "")
        self.assertFalse(self.path().exists())

    def test_append_adds_timestamped_sections_under_the_header(self):
        self.lab.notebook_append("  first idea\n", now_t=T0)
        self.lab.notebook_append("second", now_t=T0 + 3600)
        expected = (f"{NOTEBOOK_HEADER}\n\n## 2020-09-14T00:00:00Z\nfirst idea\n\n"
                    "## 2020-09-14T01:00:00Z\nsecond\n")
        self.assertEqual(self.lab.notebook(), expected)
        self.assertEqual(self.path().read_text(), expected)

    def test_trimming_drops_whole_oldest_sections(self):
        self.lab = self.make_lab(brain={"notebook_max_chars": 150})
        for k in range(8):
            self.lab.notebook_append(f"entry {k} " + "x" * 20, now_t=T0 + k * 60)
        text = self.lab.notebook()
        self.assertLessEqual(len(text), 150)
        self.assertTrue(text.startswith(NOTEBOOK_HEADER + "\n"))
        self.assertIn("entry 7 ", text)
        self.assertNotIn("entry 0 ", text)
        kept = [int(k) for k in re.findall(r"^entry (\d) x{20}$", text, re.M)]   # whole entries only
        self.assertEqual(kept, list(range(8 - len(kept), 8)))
        self.assertEqual(text.count("\n## "), len(kept))

    def test_trimming_prefers_appended_sections_over_rewritten_ones(self):
        self.lab = self.make_lab(brain={"notebook_max_chars": 160})
        self.lab.notebook_rewrite("## Key findings\ndeep entries hold up\n")
        for k in range(6):
            self.lab.notebook_append(f"note {k} " + "y" * 20, now_t=T0 + k * 60)
        text = self.lab.notebook()
        self.assertLessEqual(len(text), 160)
        self.assertIn("## Key findings\ndeep entries hold up\n", text)
        self.assertIn("note 5 ", text)
        self.assertNotIn("note 0 ", text)

    def test_an_oversized_entry_is_truncated(self):
        self.lab = self.make_lab(brain={"notebook_max_chars": 200})
        self.lab.notebook_append("old", now_t=T0)
        self.lab.notebook_append("z" * 1000, now_t=T0 + 60)
        text = self.lab.notebook()
        self.assertLessEqual(len(text), 200)
        self.assertTrue(text.endswith(TRUNCATED + "\n"))
        self.assertNotIn("\nold\n", text)

    def test_a_full_notebook_refuses_the_entry_instead_of_dropping_condensed_text(self):
        # This used to write an empty truncated entry and leave the notebook over its limit; the
        # condensed preamble is never dropped, so the brain is asked to rewrite_notebook.
        self.lab = self.make_lab(brain={"notebook_max_chars": 100})
        self.lab.notebook_rewrite("p" * 80)                  # a preamble that leaves no room
        before = self.lab.notebook()
        with self.assertRaisesRegex(ValueError, "notebook full.*rewrite_notebook"):
            self.lab.notebook_append("hello", now_t=T0)
        self.assertEqual(self.lab.notebook(), before)

    def test_the_newest_entry_is_always_kept(self):
        self.lab = self.make_lab(brain={"notebook_max_chars": 120})
        for k in range(5):
            out = self.lab.notebook_append(f"entry {k} " + "q" * 40, now_t=T0 + k * 60)
            self.assertLessEqual(out["notebook_chars"], 120)
        text = self.lab.notebook()
        self.assertTrue(text.endswith(f"\n## {iso_time(T0 + 240)}\nentry 4 {'q' * 40}\n"), text)

    def test_rewrite_adds_the_header_and_enforces_the_limit(self):
        self.lab = self.make_lab(brain={"notebook_max_chars": 100})
        self.lab.notebook_rewrite("## Plan\ntest 1d presets next")
        self.assertEqual(self.lab.notebook(), f"{NOTEBOOK_HEADER}\n\n## Plan\ntest 1d presets next\n")
        self.lab.notebook_rewrite(f"{NOTEBOOK_HEADER}\nshort")
        self.assertEqual(self.lab.notebook(), f"{NOTEBOOK_HEADER}\nshort\n")
        with self.assertRaisesRegex(ValueError, "notebook_max_chars"):
            self.lab.notebook_rewrite("w" * 200)
        self.assertEqual(self.lab.notebook(), f"{NOTEBOOK_HEADER}\nshort\n")


if __name__ == "__main__":
    unittest.main()
