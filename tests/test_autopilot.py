"""Tests for fibbot.autopilot.sweep on synthetic data: ranking and eligibility, validation of the
top presets, optional promotion within max_roster, error isolation, records, and that it never
needs Claude (no API key, anthropic not importable)."""
import json
import os
import sys
import unittest
from unittest import mock

from fibbot import autopilot
from fibbot.autopilot import MIN_FOLD_POSITIVE, TOP_VALIDATE, sweep
from fibbot.strategy import PRESETS
from tests.test_lab import LabCase

PERMISSIVE_GATE = {"min_deflated_sharpe": 0.0, "min_profit_factor": 1.0, "min_expectancy_r": 0.0}
FOUR_H = [name for name, spec in PRESETS.items() if spec.timeframe == "4h"]


class SweepCase(LabCase):
    def setUp(self):
        super().setUp()
        self.lab = self.make_lab(research={"validation_gate": PERMISSIVE_GATE}, paper={"max_roster": 1})


class TestSweep(SweepCase):
    @classmethod
    def setUpClass(cls):
        cls.no_key = mock.patch.dict(os.environ)
        cls.no_key.start()
        os.environ.pop("ANTHROPIC_API_KEY", None)

    @classmethod
    def tearDownClass(cls):
        cls.no_key.stop()

    def run_sweep(self, **kw):
        with mock.patch.dict(sys.modules, {"anthropic": None}):      # importing anthropic would fail
            out = sweep(self.lab, **kw)
        json.dumps(out, allow_nan=False)
        return out

    def test_end_to_end_ranks_validates_and_promotes(self):
        out = self.run_sweep(promote=True)
        self.assertEqual(out["presets"], list(PRESETS))
        rows = out["ranking"]
        self.assertEqual(sorted(r["preset"] for r in rows), sorted(FOUR_H))   # 1h/1d have no data
        self.assertEqual({e["preset"] for e in out["errors"] if e["step"] == "evaluate"},
                         set(PRESETS) - set(FOUR_H))
        self.assertTrue(all("no cached" in e["error"] for e in out["errors"] if e["step"] == "evaluate"))
        for r in rows:
            self.assertEqual(set(r), {"preset", "spec_id", "timeframe", "n", "avg_r", "profit_factor",
                                      "fold_positive_frac", "edge_r", "eligible"})
            self.assertEqual(r["spec_id"], PRESETS[r["preset"]].spec_id())
            self.assertEqual(r["eligible"], r["n"] >= 5 and r["avg_r"] > 0
                             and r["fold_positive_frac"] >= MIN_FOLD_POSITIVE and r["edge_r"] > 0)
        flags = [r["eligible"] for r in rows]
        self.assertTrue(any(flags) and not all(flags), flags)          # the data makes both kinds
        self.assertEqual(flags, sorted(flags, reverse=True))            # eligible first
        for group in (True, False):
            rs = [r["avg_r"] for r in rows if r["eligible"] is group]
            self.assertEqual(rs, sorted(rs, reverse=True))

        eligible = [r["preset"] for r in rows if r["eligible"]]
        self.assertEqual([v["preset"] for v in out["validations"]], eligible[:TOP_VALIDATE])
        for v in out["validations"]:
            latest = self.store.latest_validation(v["spec_id"])
            self.assertEqual((latest["id"], latest["metrics"]["passed"]), (v["experiment_id"], v["passed"]))
            self.assertEqual(v["failed"], [c["name"] for c in latest["metrics"]["checks"] if not c["passed"]])
        passed = [v["preset"] for v in out["validations"] if v["passed"]]
        self.assertTrue(passed)
        self.assertEqual([p["preset"] for p in out["promoted"]], passed[:1])     # max_roster 1
        self.assertEqual([e["preset"] for e in out["errors"] if e["step"] == "promote"], passed[1:])
        row, = self.store.roster()
        self.assertEqual((row["name"], row["strategy_id"]), (passed[0], PRESETS[passed[0]].spec_id()))
        self.assertIn("Autopilot sweep", row["rationale"])

        exp = self.store.get_experiment(out["experiment_id"])
        self.assertEqual((exp["kind"], exp["source"]), ("sweep", "autopilot"))
        self.assertEqual(exp["metrics"]["ranking"], rows)
        j = self.store.journal(kind="research")[0]
        self.assertTrue(j["title"].startswith(f"Autopilot sweep: {len(eligible)}/{len(PRESETS)} presets eligible"))
        self.assertEqual(j["body"].count("\n- ") + 1, len(rows) + len(out["validations"])
                         + len(out["promoted"]) + len(out["errors"]))
        self.assertEqual(self.store.get_kv("last_sweep_t"), exp["created_t"])

    def test_without_promote_the_roster_stays_empty(self):
        out = self.run_sweep(presets=FOUR_H)
        self.assertEqual(out["promoted"], [])
        self.assertEqual(self.store.roster(None), [])
        self.assertFalse(any(e["step"] == "promote" for e in out["errors"]))

    def test_preset_subset_and_unknown_names(self):
        out = self.run_sweep(presets=["shallow_trend_4h", "shallow_trend_4h"])
        self.assertEqual(out["presets"], ["shallow_trend_4h"])
        self.assertEqual([r["preset"] for r in out["ranking"]], ["shallow_trend_4h"])
        n = self.store.count_experiments()
        with self.assertRaisesRegex(ValueError, "unknown presets"):
            sweep(self.lab, presets=["nope"])
        self.assertEqual(self.store.count_experiments(), n)

    def test_default_presets_follow_data_timeframes(self):
        self.lab = self.make_lab(data={"timeframes": ["4h"]})
        out = self.run_sweep()
        self.assertEqual(out["presets"], FOUR_H)
        self.assertEqual([e for e in out["errors"] if e["step"] == "evaluate"], [])
        out = self.run_sweep(presets=["golden_pocket_1d"])          # an explicit list is honoured
        self.assertEqual([e["preset"] for e in out["errors"]], ["golden_pocket_1d"])

    def test_only_eligible_presets_are_validated(self):
        with mock.patch.object(autopilot, "MIN_FOLD_POSITIVE", 1.01):    # nothing can be eligible
            out = self.run_sweep(presets=FOUR_H, promote=True)
        self.assertFalse(any(r["eligible"] for r in out["ranking"]))
        self.assertEqual((out["validations"], out["promoted"]), ([], []))
        self.assertEqual(self.store.count_experiments(kind="validate"), 0)

    def test_budget_errors_are_reported_per_preset(self):
        self.lab = self.make_lab(research={"max_backtest_units_per_cycle": 3})
        self.lab.begin_cycle("research")       # backtest + baseline for one preset, then one more unit
        out = self.run_sweep(presets=FOUR_H[:3])
        self.assertEqual(len(out["ranking"]), 1)
        self.assertEqual([(e["preset"], e["step"]) for e in out["errors"]],
                         [(FOUR_H[1], "evaluate"), (FOUR_H[2], "evaluate")])
        self.assertIn("budget", out["errors"][0]["error"])


class StubLab:
    """Lab stand-in with crafted research numbers per preset: (n, avg_r, fold_positive_frac, edge_r)."""

    def __init__(self, store, cfg, numbers, passing=()):
        self.store, self.cfg, self.numbers, self.passing = store, cfg, numbers, set(passing)
        self.validated, self.promoted = [], []

    def backtest(self, spec, window):
        n, avg_r, folds, _ = self.numbers[spec.name]
        return {"spec_id": spec.spec_id(), "timeframe": spec.timeframe, "fold_positive_frac": folds,
                "metrics": {"n": n, "avg_r": avg_r, "profit_factor": 1.2}}

    def baseline(self, spec):
        return {"edge_r": self.numbers[spec.name][3]}

    def validate(self, spec):
        self.validated.append(spec.name)
        ok = spec.name in self.passing
        return {"spec_id": spec.spec_id(), "passed": ok, "experiment_id": len(self.validated),
                "metrics": {"n": 40, "avg_r": 0.2, "profit_factor": 1.5},
                "gate": {"deflated_sharpe": 0.96, "checks": [{"name": "min_trades", "passed": ok}]}}

    def promote(self, spec, name, rationale):
        self.promoted.append(name)
        return {"strategy_id": spec.spec_id(), "paper_start": "2026-11-03T00:00:00Z"}


class TestRankingRules(LabCase):
    NUMBERS = {                                   # min_trades is 5 in these tests
        "golden_pocket_4h": (40, 0.50, 1.0, -0.1),    # loses to random entries
        "golden_pocket_1h": (4, 0.45, 1.0, 0.2),      # too few trades
        "golden_pocket_both_4h": (40, 0.40, 0.4, 0.2),  # unstable across folds
        "ote_confirm_1h": (40, 0.10, 0.5, 0.05),      # eligible at the fold boundary
        "shallow_trend_4h": (5, 0.30, 0.75, 0.01),    # eligible at the trade-count boundary
        "deep_786_4h": (40, 0.30, 1.0, 0.3),          # ties shallow_trend_4h on mean R
        "golden_pocket_1d": (40, 0.20, 1.0, 0.0),     # no edge is not beating the baseline
        "ext_runner_4h": (40, 0.25, 0.6, 0.1),
    }

    def test_negative_mean_r_is_never_eligible(self):
        numbers = {**self.NUMBERS, "deep_786_4h": (40, -0.05, 0.75, 0.3)}   # beats random, loses money
        stub = StubLab(self.store, self.cfg, numbers)
        out = sweep(stub)
        row = next(r for r in out["ranking"] if r["preset"] == "deep_786_4h")
        self.assertFalse(row["eligible"])
        self.assertNotIn("deep_786_4h", stub.validated)

    def test_eligibility_order_top_validations_and_promotion(self):
        stub = StubLab(self.store, self.cfg, self.NUMBERS, passing={"deep_786_4h", "ext_runner_4h"})
        out = sweep(stub, promote=True)
        self.assertEqual([(r["preset"], r["eligible"]) for r in out["ranking"]], [
            ("shallow_trend_4h", True), ("deep_786_4h", True), ("ext_runner_4h", True),
            ("ote_confirm_1h", True), ("golden_pocket_4h", False), ("golden_pocket_1h", False),
            ("golden_pocket_both_4h", False), ("golden_pocket_1d", False)])
        self.assertEqual(stub.validated, ["shallow_trend_4h", "deep_786_4h", "ext_runner_4h"])
        self.assertEqual(stub.promoted, ["deep_786_4h", "ext_runner_4h"])
        self.assertEqual([v["failed"] for v in out["validations"]], [["min_trades"], [], []])
        self.assertEqual(out["errors"], [])


if __name__ == "__main__":
    unittest.main()
