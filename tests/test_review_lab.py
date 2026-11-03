"""Regression tests for the lab-group review findings (fibbot/lab.py, autopilot.py, report.py).

F03 point-in-time validation universe + validation edge over random entries; F14 one look per
validation period, every validation must pass, weekly validation cap, sweep reuses verdicts;
F16 CLI holdout peeks are trials and hidden from the brain; F25 notebook trimming never drops
condensed sections; F27 roster entries carry validation and graduation numbers; F37 symbol-subset
research runs do not overwrite the universe Sharpe; F51 REPORT.md shows sweep outcomes, system
events and graduation changes. Synthetic data only (see tests/test_lab.py).
"""
import json
import time
import unittest
from unittest import mock

from fibbot import autopilot, research
from fibbot.autopilot import sweep
from fibbot.lab import BudgetExceeded, Lab
from fibbot.report import JOURNAL_BODY_CHARS, render_status, write_report
from fibbot.strategy import PRESETS
from fibbot.types import Bars
from tests.test_lab import BARS, H4, OTHER, SPEC, SYMS, T0, FakeCandles, LabCase, mk_trade, record_validation

PERMISSIVE_GATE = {"min_deflated_sharpe": 0.0, "min_profit_factor": 1.0, "min_expectancy_r": 0.0,
                   "min_validation_edge_z": None}   # synthetic random walks: no significant edge
GP_BOTH, SHALLOW, EXT = (PRESETS[n] for n in ("golden_pocket_both_4h", "shallow_trend_4h", "ext_runner_4h"))
DAY = 86400


def with_volume(b: Bars, v: float, symbol: str | None = None) -> Bars:
    """Copy of b with a constant volume v (and optionally another symbol name)."""
    return Bars(symbol or b.symbol, b.tf, list(b.t), list(b.o), list(b.h), list(b.l), list(b.c),
                [float(v)] * len(b))


class TruncCandles(FakeCandles):
    """FakeCandles whose history ends before cut_t (an older data_end)."""

    def __init__(self, cut_t, bars=None):
        super().__init__(bars)
        self.cut_t = cut_t

    def get(self, symbol, tf):
        b = super().get(symbol, tf)
        return b.slice(0, b.index_at_or_after(self.cut_t))


def trial_sharpe(store, spec_id):
    row = store._one("SELECT sharpe FROM trials WHERE spec_id = ?", (spec_id,))
    return None if row is None else row[0]


# -- F03: survivorship / selection in the validation window -----------------------------------

class TestPointInTimeValidationUniverse(LabCase):
    def setUp(self):
        super().setUp()
        self.get_universe.return_value = ["S0USDT", "S1USDT"]          # today's top 2
        self.candles = FakeCandles({"S0USDT": with_volume(BARS["S0USDT"], 1e3),
                                    "S1USDT": with_volume(BARS["S1USDT"], 1.0),       # small before vs
                                    "S2USDT": with_volume(BARS["S2USDT"], 1e6)})      # a former member
        self.store.set_kv("universe_seen", ["S2USDT"])
        self.lab = self.make_lab(universe={"top_n": 2}, research={"validation_gate": PERMISSIVE_GATE})

    def test_validation_symbols_are_ranked_before_validation_start(self):
        bars, windows, info = self.lab.validation_bars("4h")
        self.assertEqual(list(bars), ["S0USDT", "S2USDT"])          # S1 only made today's top 2
        self.assertEqual((info["dropped"], info["added"]), (["S1USDT"], ["S2USDT"]))
        self.assertEqual(windows, self.lab.windows("4h"))           # windows: today's universe bars
        out = self.lab.validate(SPEC)
        self.assertEqual(sorted(out["per_symbol"]), ["S0USDT", "S2USDT"])
        exp = self.store.latest_validation(SPEC.spec_id())
        self.assertEqual(exp["metrics"]["symbols"], ["S0USDT", "S2USDT"])
        self.assertEqual(exp["metrics"]["validation_universe"]["dropped"], ["S1USDT"])
        # the research window keeps today's universe
        self.assertEqual(sorted(self.lab.backtest(SPEC)["per_symbol"]), ["S0USDT", "S1USDT"])

    def test_a_coin_too_young_at_validation_start_is_left_out(self):
        young = BARS["S1USDT"].slice(1300, 2000)                   # listed ~25 days before vs
        self.candles.bars["S1USDT"] = with_volume(young, 1e9)
        self.store.set_kv("universe_seen", [])
        bars, _, info = self.lab.validation_bars("4h")
        self.assertEqual(list(bars), ["S0USDT"])
        self.assertEqual(info["dropped"], ["S1USDT"])

    def test_cached_former_members_join_the_pool(self):
        from fibbot.data.candles import CandleStore, _encode, atomic_write
        candles = CandleStore(self.make_lab().cfg)
        for sym, v in (("S0USDT", 1e3), ("S1USDT", 1.0), ("S2USDT", 1e6), ("USDCUSDT", 1e9)):
            atomic_write(candles.path(sym, "4h"), _encode(with_volume(BARS["S2USDT" if sym == "USDCUSDT" else sym],
                                                                      v, sym)))
        self.store.set_kv("universe_seen", [])
        lab = self.make_lab(candles=candles, universe={"top_n": 2})
        bars, _, info = lab.validation_bars("4h")
        self.assertEqual(list(bars), ["S0USDT", "S2USDT"])          # from the cache; the stablecoin is excluded
        self.assertEqual(info["added"], ["S2USDT"])

    def test_current_mode_and_validation_window_peek_use_the_same_symbols(self):
        out = self.lab.backtest(SPEC, window="validation")
        self.assertEqual(sorted(out["per_symbol"]), ["S0USDT", "S2USDT"])
        lab = self.make_lab(universe={"top_n": 2}, research={"validation_universe": "current"})
        self.assertEqual(list(lab.validation_bars("4h")[0]), ["S0USDT", "S1USDT"])


class TestValidationEdgeOverRandomEntries(LabCase):
    def setUp(self):
        super().setUp()
        self.lab = self.make_lab(research={"validation_gate": PERMISSIVE_GATE})

    def drift(self, edge):
        """random_baseline stand-in: random entries earn the strategy's mean R minus edge."""
        real = research.random_baseline

        def fake(spec, bars, cfg, start_t, end_t, n_runs=10, seed=1, result=None):
            out = real(spec, bars, cfg, start_t, end_t, n_runs=n_runs, seed=seed, result=result)
            self.calls.append((start_t, end_t))
            return {**out, "baseline_avg_r_mean": out["strategy_avg_r"] - edge, "edge_r": edge,
                    "baseline_regime_avg_r_mean": out["strategy_avg_r"] - edge, "edge_regime_r": edge}
        self.calls = []
        return mock.patch("fibbot.research.random_baseline", side_effect=fake)

    def test_a_spec_that_only_rides_the_drift_fails_validation(self):
        # The old gate passed this spec (see test_lab.test_real_validation_then_promotion); random
        # long entries on the same coins and window earn more, so the edge check fails it.
        with self.drift(-0.05):
            out = self.lab.validate(SPEC)
        check = next(c for c in out["gate"]["checks"] if c["name"] == "min_validation_edge_r")
        # (the gate measures the edge on the unrounded mean R: -0.05 up to rounding)
        self.assertAlmostEqual(check["value"], -0.05, places=3)
        self.assertEqual((check["threshold"], check["passed"]), (0.0, False))
        self.assertTrue(all(c["passed"] for c in out["gate"]["checks"] if c is not check))
        self.assertFalse(out["passed"])
        self.assertEqual(self.calls, [research.window_bounds(self.lab.windows("4h"), "validation")])
        self.assertEqual(out["baseline"]["edge_r"], -0.05)
        with self.assertRaisesRegex(ValueError, "did not pass: min_validation_edge_r"):
            self.lab.promote(SPEC, "drift", "x")

    def test_a_real_edge_passes_and_the_check_can_be_disabled(self):
        with self.drift(0.2):
            self.assertTrue(self.lab.validate(SPEC)["passed"])
        lab = self.make_lab(research={"validation_gate": {**PERMISSIVE_GATE, "min_validation_edge_r": None}})
        with self.drift(-1.0):
            out = lab.validate(OTHER)
        self.assertNotIn("min_validation_edge_r", [c["name"] for c in out["gate"]["checks"]])


# -- F14: re-validation, all-must-pass, weekly cap, sweep reuse --------------------------------

class TestOneLookPerValidationPeriod(LabCase):
    CUT = T0 + 1900 * H4

    def lab_at(self, cut_t, **over):
        return self.make_lab(candles=TruncCandles(cut_t), **over)

    def test_revalidating_on_a_slid_window_is_refused(self):
        lab = self.lab_at(self.CUT)
        first = lab.validate(SPEC)
        n_trials = self.store.trial_stats()["n"]
        later = self.lab_at(self.CUT + 7 * DAY)                    # data_end moved one week
        later.begin_cycle("research")
        with self.assertRaisesRegex(ValueError, rf"already validated on an overlapping window \(experiment"
                                                rf" {first['experiment_id']}"):
            later.validate(SPEC)
        self.assertEqual(later.budget()["validations_left"], 2)    # refunded (3 - 1 used this week)
        later.end_cycle()
        self.assertEqual(self.store.count_experiments(kind="validate"), 1)
        self.assertEqual(self.store.trial_stats()["n"], n_trials)
        self.assertEqual(later.prior_validation(SPEC)["id"], first["experiment_id"])

    def test_an_identical_window_is_a_replay_not_a_new_look(self):
        first = self.lab.validate(SPEC)
        again = self.make_lab().validate(SPEC)                      # same data, same symbols
        self.assertEqual((again["passed"], again["metrics"]), (first["passed"], first["metrics"]))
        self.assertEqual((again["holdout_looks"], again["replay"]), (1, True))
        self.assertEqual(again["n_trials"], first["n_trials"])

    def test_a_later_non_overlapping_window_is_a_new_look_and_a_new_trial(self):
        a = self.lab_at(T0 + 1300 * H4).validate(SPEC)
        b = self.lab_at(T0 + 2000 * H4).validate(SPEC)              # starts after a's window ended
        self.assertLess(self.store.get_experiment(a["experiment_id"])["window_end"],
                        self.store.get_experiment(b["experiment_id"])["window_start"] + 1)
        self.assertEqual((a["holdout_looks"], b["holdout_looks"]), (1, 2))
        self.assertEqual(b["n_trials"], a["n_trials"] + 1)          # "<spec_id>#look2"

    def test_one_pass_after_a_failed_validation_is_not_enough(self):
        strict = self.make_lab(research={"validation_gate": {"min_trades": 10 ** 6}})
        self.assertFalse(strict.validate(SPEC)["passed"])
        lenient = self.make_lab(research={"validation_gate": PERMISSIVE_GATE})
        with mock.patch("fibbot.research.random_baseline",
                        side_effect=lambda *a, **k: {"n_runs": 1, "baseline_avg_r_mean": -9.0,
                                                     "baseline_avg_r_std": 0.0, "beats_frac": 1.0,
                                                     "edge_r": 9.0}):
            self.assertTrue(lenient.validate(SPEC)["passed"])     # identical window: replay allowed
        lenient.baseline(SPEC)                                      # promote needs a research baseline
        with self.assertRaisesRegex(ValueError, "failed validation .* within the last 3 months"):
            lenient.promote(SPEC, "deep", "x")
        later = int(time.time()) + 100 * DAY                        # the failure is older than 3 months
        self.assertEqual(lenient.promote(SPEC, "deep", "x", now_t=later)["strategy_id"], SPEC.spec_id())


class TestWeeklyValidationCap(LabCase):
    def test_cycles_share_a_rolling_weekly_cap(self):
        for spec in (SPEC, OTHER, GP_BOTH):
            self.lab.begin_cycle("research")
            self.lab.validate(spec)
            self.lab.end_cycle()
        self.lab.begin_cycle("research")
        self.assertEqual(self.lab.budget()["validations_left"], 0)
        with self.assertRaisesRegex(BudgetExceeded, "max_validations_per_week"):
            self.lab.validate(SHALLOW)
        self.lab.end_cycle()
        self.assertEqual(self.lab.weekly_validations_left(), 0)
        self.assertEqual(self.lab.weekly_validations_left(now_t=int(time.time()) + 8 * DAY), 3)
        self.lab.validate(SHALLOW)                                  # the operator's CLI is not capped
        st = self.lab.status()
        self.assertEqual(st["validations_7d"], {"used": 4, "cap": 3})

    def test_cap_is_configurable(self):
        lab = self.make_lab(research={"max_validations_per_week": None})
        self.assertIsNone(lab.weekly_validations_left())
        lab.begin_cycle("research")
        self.assertEqual(lab.budget()["validations_left"], 3)


class TestSweepNeverRedrawsTheLockedWindow(LabCase):
    def setUp(self):
        super().setUp()
        self.lab = self.make_lab(research={"validation_gate": PERMISSIVE_GATE}, paper={"max_roster": 1},
                                 data={"timeframes": ["4h"]})

    def test_a_second_sweep_reuses_the_verdicts(self):
        first = sweep(self.lab)
        n_val = self.store.count_experiments(kind="validate")
        self.assertEqual(n_val, len(first["validations"]))
        self.assertTrue(first["validations"] and not any(v["reused"] for v in first["validations"]))
        second = sweep(self.lab, promote=True)
        self.assertEqual(self.store.count_experiments(kind="validate"), n_val)    # no new look
        self.assertEqual([(v["preset"], v["passed"], v["experiment_id"]) for v in second["validations"]],
                         [(v["preset"], v["passed"], v["experiment_id"]) for v in first["validations"]])
        self.assertTrue(all(v["reused"] for v in second["validations"]))
        passed = [v["preset"] for v in first["validations"] if v["passed"]]
        self.assertEqual([p["preset"] for p in second["promoted"]], passed[:1])   # a kept pass promotes
        body = self.store.journal(kind="research")[0]["body"]
        self.assertRegex(body, r"- validation \w+: (PASS|FAIL) \(kept from experiment \d+: n ")

    def test_retired_and_rostered_presets_are_not_validated(self):
        out = sweep(self.lab, promote=True)
        sid = out["promoted"][0]["strategy_id"]
        self.lab.retire(sid, "drawdown")
        n_val = self.store.count_experiments(kind="validate")
        again = sweep(self.lab, promote=True)
        name = out["promoted"][0]["preset"]
        self.assertIn({"preset": name, "reason": "on the roster (status retired): drawdown"}, again["skipped"])
        self.assertNotIn(name, [v["preset"] for v in again["validations"]])
        self.assertNotIn(name, [p["preset"] for p in again["promoted"]])
        self.assertGreaterEqual(self.store.count_experiments(kind="validate"), n_val)

    def test_weekly_cap_stops_new_validations(self):
        lab = self.make_lab(research={"validation_gate": PERMISSIVE_GATE, "max_validations_per_week": 1},
                            data={"timeframes": ["4h"]})
        out = sweep(lab)
        self.assertEqual(sum(not v["reused"] for v in out["validations"]), 1)
        self.assertTrue(any("max_validations_per_week" in s["reason"] for s in out["skipped"]))


# -- F16: CLI peeks at the locked window ---------------------------------------------------------

class TestHoldoutPeeks(LabCase):
    def test_peeks_count_as_trials_and_validate_sees_them(self):
        variants = [SPEC.with_overrides(stop_atr_buffer=b) for b in (0.1, 0.2, 0.3)]
        before = self.store.trial_stats()
        for k, v in enumerate(variants, 1):
            self.lab.backtest(v, window="validation")
            self.assertEqual(self.store.trial_stats()["n"], before["n"] + k)
        self.assertEqual(self.store.trial_stats()["var_sharpe"], before["var_sharpe"])
        out = self.lab.validate(variants[0])                        # the peek previewed this very draw
        self.assertEqual(out["n_trials"], len(variants))
        self.assertEqual((out["holdout_looks"], out["replay"]), (1, True))    # same draw: no new look
        self.assertEqual(self.lab.status()["holdout_peeks"], 3)
        self.assertEqual([lk["kind"] for lk in self.lab.holdout_looks(variants[0].spec_id())],
                         ["backtest", "validate"])

    def test_the_brain_cannot_read_peek_results(self):
        try:
            from fibbot.brain.tools import build_tools
        except ImportError:                                         # pragma: no cover
            self.skipTest("anthropic not installed")
        self.lab.backtest(SPEC, window="validation")
        self.lab.backtest(SPEC, window="all")
        self.lab.backtest(OTHER)
        tools = {t.name: t for t in build_tools(self.lab, "research")}
        for sort_by in ("recent", "avg_r"):
            rows = json.loads(tools["list_experiments"].call({"limit": 10, "sort_by": sort_by}))
            # F16: peek rows are never listed (their numbers are nested out of reach as well)
            self.assertFalse([r for r in rows if r["notes"].startswith(("window=validation", "window=all"))])
            self.assertEqual([r["spec_id"] for r in rows], [OTHER.spec_id()])
        looked_up = json.loads(tools["list_experiments"].call({"spec_id": SPEC.spec_id()}))
        self.assertEqual((looked_up["n_experiments"], looked_up["experiments"]), (0, []))


# -- F25: notebook trimming ----------------------------------------------------------------------

class TestNotebookKeepsCondensedSections(LabCase):
    def test_condensed_sections_survive_and_a_full_notebook_asks_for_a_rewrite(self):
        lab = self.make_lab(brain={"notebook_max_chars": 1000})
        lab.notebook_rewrite("## Conclusions\n" + "c" * 880)
        before = lab.notebook()
        with self.assertRaisesRegex(ValueError, "notebook full.*rewrite_notebook"):
            lab.notebook_append("r" * 300, now_t=T0)
        self.assertEqual(lab.notebook(), before)
        out = lab.notebook_append("short note", now_t=T0)
        self.assertIn("## Conclusions\n" + "c" * 880, lab.notebook())
        self.assertLessEqual(out["notebook_chars"], 1000)

    def test_markdown_headings_inside_entries_stay_inside_their_entry(self):
        lab = self.make_lab(brain={"notebook_max_chars": 2000})
        lab.notebook_rewrite("## Conclusions\nkeep me\n")
        trimmed = []
        for k in range(30):
            out = lab.notebook_append(f"## Results\ncycle {k} mean R +0.1\n## Next\ntry 1d {k}", now_t=T0 + k * 60)
            trimmed += out["trimmed"]
        text = lab.notebook()
        self.assertIn("## Conclusions\nkeep me\n", text)
        self.assertLessEqual(len(text), 2000)
        self.assertNotIn("\n## Results", text)
        self.assertIn("### Results\ncycle 29 mean R", text)
        self.assertTrue(text.rstrip().endswith("try 1d 29"))
        self.assertTrue(trimmed and all(t.startswith("## 2020-09-14T") for t in trimmed), trimmed)
        self.assertEqual((out["limit"], out["truncated"]), (2000, False))

    def test_rewrite_reports_the_room_left(self):
        lab = self.make_lab(brain={"notebook_max_chars": 1000})
        out = lab.notebook_rewrite("## Plan\n" + "x" * 900)
        self.assertEqual((out["limit"], out["free_chars"] < 250), (1000, True))
        self.assertIn("warning", out)
        self.assertNotIn("warning", lab.notebook_rewrite("## Plan\nshort"))


# -- F27: roster entries in the status -----------------------------------------------------------

class TestRosterEntryNumbers(LabCase):
    NOW = 1_700_000_123

    def test_roster_entry_has_validation_and_graduation_numbers(self):
        eid = record_validation(self.store, SPEC, avg_r=0.35)
        sid = self.lab.promote(SPEC, "deep", "x", now_t=self.NOW)["strategy_id"]
        start = self.store.get_strategy(sid)["paper_start_t"]
        for k, r in enumerate((1.0, -1.0, 0.5, 0.2)):
            self.store.upsert_paper_trade(sid, mk_trade(sid, SYMS[k % 3], f"L{k}", start + k * H4, r), 100.0, r)
        entry, = self.lab.status(now_t=start + 20 * DAY)["roster"]
        self.assertEqual({k: entry["validation"][k] for k in ("experiment_id", "n", "avg_r", "profit_factor",
                                                               "deflated_sharpe")},
                         {"experiment_id": eid, "n": 40, "avg_r": 0.35, "profit_factor": 1.5,
                          "deflated_sharpe": 0.97})
        self.assertEqual(len(entry["validation"]["window"]), 2)
        checks = {c[0]: c for c in entry["graduation"]["checks"]}
        name, value, threshold, passed = checks["max_shortfall_vs_validation_r"]
        # F52: value = validation mean R - paper mean R, passing when <= max_shortfall (0.25)
        self.assertAlmostEqual(value, 0.35 - entry["account"]["avg_r"], places=4)
        self.assertAlmostEqual(threshold, 0.25, places=4)
        self.assertEqual(passed, value <= threshold)
        self.assertEqual(len(entry["graduation"]["mean_r_ci"]), 2)
        self.assertEqual(entry["graduation"]["failed"], [c[0] for c in entry["graduation"]["checks"] if not c[3]])

    def test_a_full_roster_fits_the_cycle_prompt(self):
        from fibbot.brain.cycles import STATUS_MAX_CHARS
        from fibbot.brain.tools import to_json
        for k, spec in enumerate((SPEC, OTHER, GP_BOTH, SHALLOW)):
            record_validation(self.store, spec)
            self.lab.promote(spec, f"s{k}", "x", now_t=self.NOW)
        st = self.lab.status(now_t=self.NOW + 30 * DAY)
        self.assertEqual(len(st["roster"]), 4)
        self.assertLess(len(to_json(st, 10 ** 9)), STATUS_MAX_CHARS // 2)


# -- F37: symbol-subset research runs ------------------------------------------------------------

class TestSubsetRunsKeepTheUniverseSharpe(LabCase):
    def test_a_subset_run_counts_but_does_not_replace_the_sharpe(self):
        self.lab.backtest(SPEC)
        self.lab.backtest(OTHER)
        before, sharpe = self.store.trial_stats(), trial_sharpe(self.store, SPEC.spec_id())
        self.assertGreater(before["var_sharpe"], 0)
        self.lab.backtest(SPEC, symbols=["S1USDT"])
        self.assertEqual(self.store.trial_stats(), before)
        self.assertEqual(trial_sharpe(self.store, SPEC.spec_id()), sharpe)

    def test_a_spec_seen_only_on_a_subset_is_a_trial_without_a_sharpe(self):
        self.lab.backtest(SPEC)
        self.lab.backtest(OTHER)
        before = self.store.trial_stats()
        self.lab.backtest(GP_BOTH, symbols=["S0USDT"])
        after = self.store.trial_stats()
        self.assertEqual((after["n"], after["var_sharpe"], after["n_var"]),
                         (before["n"] + 1, before["var_sharpe"], before["n_var"]))
        self.assertIsNone(trial_sharpe(self.store, GP_BOTH.spec_id()))

    def test_the_full_symbol_list_is_a_universe_run(self):
        self.lab.backtest(SPEC, symbols=list(SYMS))
        self.assertIsNotNone(trial_sharpe(self.store, SPEC.spec_id()))


# -- F51: REPORT.md for an unattended bot --------------------------------------------------------

class TestReportShowsOutcomes(LabCase):
    def sweep_body(self):
        ranking = [{"preset": f"preset_{k}_with_a_long_name_4h", "timeframe": "4h", "n": 120 + k,
                    "avg_r": 0.05 * k, "profit_factor": 1.1, "fold_positive_frac": 0.75, "edge_r": 0.01 * k,
                    "eligible": k < 4} for k in range(8)]
        validations = [{"preset": "preset_0_with_a_long_name_4h", "passed": False, "n": 40, "avg_r": -0.1,
                        "profit_factor": 0.9, "deflated_sharpe": 0.1, "failed": ["min_profit_factor"]},
                       {"preset": "preset_1_with_a_long_name_4h", "passed": True, "n": 44, "avg_r": 0.2,
                        "profit_factor": 1.5, "deflated_sharpe": 0.95, "failed": []}]
        return autopilot._journal_body({"ranking": ranking, "validations": validations, "promoted": [],
                                        "errors": [], "skipped": []})

    def test_clipped_sweep_entry_still_shows_its_validations(self):
        body = self.sweep_body()
        self.assertGreater(len(body), JOURNAL_BODY_CHARS)
        self.store.add_journal("research", "Autopilot sweep: 4/8 presets eligible", body)
        text = write_report(self.lab, now_t=int(time.time())).read_text()
        self.assertIn("> - validation preset_0_with_a_long_name_4h: FAIL", text)
        self.assertIn("> - validation preset_1_with_a_long_name_4h: PASS", text)
        self.assertRegex(text, r"> \[\.\.\. \d+ more lines\]")
        self.assertNotRegex(text, r"> - \S+ \[\.\.\.\]")              # no line cut mid-word

    def test_system_events_and_graduation_changes_are_reported(self):
        now = int(time.time())
        self.store.add_journal("system", "Scheduler step data failed: TimeoutError: binance", "trace", now_t=now - 60)
        self.store.add_journal("system", "Scheduler step paper failed: old", "x", now_t=now - 30 * DAY)
        self.store.add_journal("graduation", "deep (abc) collecting -> promising", "checks", now_t=now - 10)
        text = write_report(self.lab, now_t=now).read_text()
        self.assertIn("Scheduler step data failed: TimeoutError: binance", text)
        self.assertIn("deep (abc) collecting -> promising", text)
        self.assertIn("**1 system event (errors, warnings) in the last 7 days**", text)
        self.assertIn("## Recent system events", text)
        status = render_status(self.lab, now_t=now)
        self.assertIn("system events (errors, warnings) last 7 days: 1; last", status)
        self.assertEqual(self.lab.status(now_t=now)["system_events_7d"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
