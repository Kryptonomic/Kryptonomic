"""Tests for fibbot.research: windows, folds, evaluation, the random-entry baseline, walk-forward
selection (on a scripted fake backtest), neighbours / robustness and the validation gate."""
import json
import unittest
from statistics import fmean, stdev
from unittest import mock

from fibbot import research as rs
from fibbot.backtest import as_spec, result_from_trades, rounded
from fibbot.config import DEFAULTS
from fibbot.metrics import deflated_sharpe, trade_stats
from fibbot.research import (compute_windows, evaluate, fold_bounds, fold_stats, neighbours,
                             random_baseline, robustness, validation_gate, walk_forward_optimize,
                             window_bounds)
from fibbot.strategy import PARAM_SPACE, FibEngine, StrategySpec, manage_position, warmup_bars
from fibbot.types import Bars
from tests.test_backtest import H4, SPEC, T0, cfg_, make_bars, make_universe, trade

DAY = 86400


def dumps(x):
    return json.dumps(x, allow_nan=False)


class TestWindows(unittest.TestCase):
    def test_compute_windows(self):
        bars = {"A": make_bars("A", n=1000), "B": make_bars("B", n=950, t0=T0 + 100 * H4),
                "E": Bars("E", "4h")}
        w = compute_windows(cfg_({"validation_months": 2}), bars)
        data_end = T0 + 1000 * H4
        vs = int(data_end - 2 * 30.44 * DAY) // DAY * DAY
        self.assertEqual(w, {"data_start": T0, "data_end": data_end, "research_start": T0,
                             "research_end": vs, "validation_start": vs, "validation_end": data_end})
        daily = {"D": make_bars("D", n=400, tf="1d")}
        w = compute_windows(cfg_({"validation_months": 12}), daily)
        self.assertEqual(w["data_end"], T0 + 400 * DAY)
        self.assertEqual(w["validation_start"], T0 + 400 * DAY - 366 * DAY)  # 365.28 days, floored
        for empty in ({}, {"E": Bars("E", "4h")}):
            with self.assertRaises(ValueError):
                compute_windows(cfg_(), empty)

    def test_window_bounds(self):
        w = {"data_start": 1, "data_end": 9, "research_start": 1, "research_end": 5,
             "validation_start": 5, "validation_end": 9}
        self.assertEqual([window_bounds(w, k) for k in ("research", "validation", "all")],
                         [(1, 5), (5, 9), (1, 9)])
        with self.assertRaises(ValueError):
            window_bounds(w, "train")


class TestFolds(unittest.TestCase):
    def test_fold_bounds(self):
        self.assertEqual(fold_bounds(0, 100, 4), [(0, 25), (25, 50), (50, 75), (75, 100)])
        self.assertEqual(fold_bounds(0, 10, 3), [(0, 3), (3, 6), (6, 10)])
        self.assertEqual(fold_bounds(T0, T0 + 7, 1), [(T0, T0 + 7)])
        for args in ((0, 100, 0), (5, 5, 2), (6, 5, 2)):
            with self.assertRaises(ValueError):
                fold_bounds(*args)

    def test_fold_stats_buckets_by_entry_time(self):
        trades = [trade(entry_t=t, r=r) for t, r in
                  ((-1, 9.0), (0, 1.0), (24, -1.0), (25, 2.0), (99, 0.5), (100, 9.0))]
        bounds = fold_bounds(0, 100, 4)
        got = fold_stats(trades, bounds)
        self.assertEqual([f["n"] for f in got], [2, 1, 0, 1])
        for f, sub in zip(got, ([trades[1], trades[2]], [trades[3]], [], [trades[4]])):
            st = trade_stats(sub)
            self.assertEqual(f, {"fold": f["fold"], "start": f["start"], "end": f["end"],
                                 **{k: st[k] for k in ("n", "avg_r", "total_r", "profit_factor")}})
        self.assertEqual([(f["fold"], f["start"], f["end"]) for f in got],
                         [(k, a, b) for k, (a, b) in enumerate(bounds)])


class Universe(unittest.TestCase):
    """Three synthetic symbols, 2000 4h bars, 2-month validation window."""
    @classmethod
    def setUpClass(cls):
        cls.bars = make_universe(3, n=2000)
        cls.cfg = cfg_({"validation_months": 2, "min_trades": 20})
        cls.w = compute_windows(cls.cfg, cls.bars)


class TestEvaluate(Universe):
    def test_windows_and_fold_extras(self):
        for window, (a, b) in (("research", ("research_start", "research_end")),
                               ("validation", ("validation_start", "validation_end")),
                               ("all", ("data_start", "data_end"))):
            res, ex = evaluate(SPEC, self.bars, self.cfg, self.w, window)
            self.assertEqual((res.start_t, res.end_t), (self.w[a], self.w[b]))
            self.assertTrue(all(self.w[a] <= tr.entry_t < self.w[b] for tr in res.trades))
            folds = ex["folds"]
            self.assertEqual(folds, rounded(fold_stats(res.closed_trades,
                                                       fold_bounds(self.w[a], self.w[b], 4))))
            self.assertEqual(sum(f["n"] for f in folds), res.metrics["n"])
            self.assertEqual(ex["fold_positive_frac"], sum(f["avg_r"] > 0 for f in folds) / 4)
            self.assertEqual(ex["min_fold_avg_r"], min(f["avg_r"] for f in folds))
            dumps(ex)
        with self.assertRaises(ValueError):
            evaluate(SPEC, self.bars, self.cfg, self.w, "oos")

    def test_empty_folds_are_not_positive_and_count_as_zero(self):
        fake = result_from_trades(SPEC, [trade(entry_t=T0 + 10, r=2.0)], ["AUSDT"], T0, T0 + 400,
                                  self.cfg)
        w = {"research_start": T0, "research_end": T0 + 400}
        with mock.patch.object(rs, "backtest", return_value=fake):
            _, ex = evaluate(SPEC, {}, self.cfg, w)
        self.assertEqual((ex["fold_positive_frac"], ex["min_fold_avg_r"]), (0.25, 0.0))


class TestRandomBaseline(Universe):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.start, cls.end = T0 + 700 * H4, cls.w["research_end"]    # 600 warmup bars before start
        cls.res = rs.backtest(SPEC, cls.bars, cls.start, cls.end, cls.cfg)

    def run_baseline(self, **kw):
        return random_baseline(SPEC, self.bars, self.cfg, self.start, self.end, result=self.res, **kw)

    def test_deterministic_per_seed(self):
        a, b, c = self.run_baseline(seed=3), self.run_baseline(seed=3), self.run_baseline(seed=4)
        self.assertEqual(a, b)
        self.assertNotEqual(a["runs"], c["runs"])
        self.assertEqual(random_baseline(SPEC, self.bars, self.cfg, self.start, self.end, seed=3), a)
        dumps(a)

    def test_statistics(self):
        b = self.run_baseline(n_runs=7, seed=2)
        runs = b["runs"]
        self.assertEqual((b["n_runs"], len(runs), b["seed"]), (7, 7, 2))
        self.assertEqual(b["n_trades"], self.res.metrics["n"])
        self.assertEqual(b["strategy_avg_r"], rounded(self.res.metrics["avg_r"]))
        self.assertAlmostEqual(b["baseline_avg_r_mean"], fmean(runs), places=3)
        self.assertAlmostEqual(b["baseline_avg_r_std"], stdev(runs), places=3)
        self.assertAlmostEqual(b["edge_r"], b["strategy_avg_r"] - b["baseline_avg_r_mean"], places=3)
        self.assertEqual(b["beats_frac"], sum(self.res.metrics["avg_r"] > r for r in runs) / 7)

    def test_random_entries_copy_side_risk_and_targets(self):
        calls = []

        def spy(*args, **kw):
            out = manage_position(*args, **kw)
            calls.append((args, out["r_multiple"]))
            return out

        with mock.patch.object(rs, "manage_position", side_effect=spy):
            b = self.run_baseline(n_runs=3, seed=5)
        trades = self.res.closed_trades
        self.assertEqual(len(calls), 3 * len(trades))
        slip = self.cfg["costs"]["slippage"]
        for j, ((bars, atr_w, e, side, px, stop, tp1, tp2, spec, costs), _) in enumerate(calls):
            tr = trades[j % len(trades)]
            w = self.bars[tr.symbol].window(self.start, self.end, warmup_bars(SPEC))
            self.assertEqual((bars.symbol, bars.t, len(atr_w)), (tr.symbol, w.t, len(w)))
            self.assertEqual(w.index_at_or_after(self.start), warmup_bars(SPEC))
            self.assertTrue(warmup_bars(SPEC) <= e < len(w) - 1)
            self.assertEqual(side, tr.side)
            self.assertAlmostEqual(px, w.o[e] * (1 + side * slip))
            self.assertAlmostEqual(abs(px - stop) / px, tr.risk_pct)
            self.assertEqual(side * (px - stop) > 0, True)
            risk = abs(tr.entry_price - tr.stop_price)
            for got, want in ((tp1, tr.meta["tp1"]), (tp2, tr.meta["tp2"])):
                self.assertAlmostEqual(side * (got - px) / abs(px - stop),
                                       side * (want - tr.entry_price) / risk)
            self.assertEqual(spec, SPEC)
        n = len(trades)
        for k in range(3):
            self.assertAlmostEqual(b["runs"][k], fmean(r for _, r in calls[k * n:(k + 1) * n]),
                                   places=3)

    def test_a_tie_does_not_beat_the_baseline(self):
        avg = self.res.metrics["avg_r"]
        with mock.patch.object(rs, "manage_position", return_value={"r_multiple": avg}):
            b = self.run_baseline(n_runs=4)
        self.assertEqual((b["beats_frac"], b["edge_r"], b["baseline_avg_r_std"]), (0.0, 0.0, 0.0))

    def test_backtests_when_no_result_and_handles_no_trades(self):
        with mock.patch.object(rs, "backtest", wraps=rs.backtest) as bt:
            random_baseline(SPEC, self.bars, self.cfg, self.start, self.end, n_runs=1)
        bt.assert_called_once()
        none = random_baseline(SPEC.with_overrides(min_rr=10.0), self.bars, self.cfg,
                               self.start, self.end)
        self.assertEqual(none, {"n_trades": 0, "n_runs": 10, "seed": 1, "strategy_avg_r": 0.0,
                                "baseline_avg_r_mean": 0.0, "baseline_avg_r_std": 0.0,
                                "beats_frac": 0.0, "runs": [], "edge_r": 0.0})


# Scripted walk-forward: fold k spans [T0 + k*D, T0 + (k+1)*D); trades of each grid value per fold.
D = 100 * DAY
PLAN = {
    0.5: [[2.0] * 6, [-1.0] * 6, [-1.0] * 6, [-1.0] * 6],      # A: great first fold, then bad
    0.55: [[0.5] * 12] * 4,                                     # B: steady, many trades
    0.6: [[2.0] * 3, [2.0] * 3, [-1.0] * 3, [1.0] * 3],        # C: few trades
}
WF_WINDOWS = {"research_start": T0, "research_end": T0 + 4 * D}


def fake_backtest(spec, bars, start, end, cfg, workers=None):
    trades = [trade("AUSDT", T0 + k * D + j * H4, r)
              for k, rs_ in enumerate(PLAN[spec.entry_zone_lo]) for j, r in enumerate(rs_)]
    return result_from_trades(spec, trades, ["AUSDT"], start, end, cfg)


class TestWalkForwardSelection(unittest.TestCase):
    GRID = {"entry_zone_lo": [0.5, 0.55, 0.6, 0.7]}           # 0.7 > entry_zone_hi: invalid

    def wf(self, objective="avg_r", research=None, **kw):
        cfg = cfg_({"min_trades": 20, "folds": 4, **(research or {})})
        with mock.patch.object(rs, "backtest", side_effect=fake_backtest):
            out = walk_forward_optimize(SPEC, self.GRID, {}, cfg, WF_WINDOWS, objective, **kw)
        dumps(out)
        return out

    def chosen(self, out):
        return [c["params"] and c["params"]["entry_zone_lo"] for c in out["chosen"]]

    def test_anchored_selection_by_avg_r(self):
        seen = []
        out = self.wf(on_combo=lambda spec, res: seen.append((spec.entry_zone_lo, res.metrics["n"])))
        self.assertEqual(seen, [(0.5, 24), (0.55, 48), (0.6, 12)])
        self.assertEqual((out["n_combos"], out["n_invalid"], out["n_evaluated"]), (4, 1, 3))
        self.assertIn("entry_zone_lo", out["invalid_reasons"][0])
        self.assertEqual((out["n_folds"], out["min_is_trades"]), (4, 5))
        # k=1: C has 3 < 5 in-sample trades -> A (2.0); k=2: C 2.0; k=3: C 1.0
        self.assertEqual(self.chosen(out), [0.5, 0.6, 0.6])
        self.assertEqual([c["is_avg_r"] for c in out["chosen"]], [2.0, 2.0, 1.0])
        self.assertEqual([c["is_n"] for c in out["chosen"]], [6, 6, 9])
        oos = [trade(r=r) for r in [-1.0] * 6 + [-1.0] * 3 + [1.0] * 3]
        self.assertEqual(out["oos"], rounded(trade_stats(oos)))
        self.assertEqual([(f["fold"], f["n"], f["total_r"]) for f in out["oos_by_fold"]],
                         [(1, 6, -6.0), (2, 3, -3.0), (3, 3, 3.0)])
        self.assertEqual(out["oos_by_fold"][0]["start"], T0 + D)

    def test_objectives_and_tie_breaks(self):
        # total_r k=2: B 12 (24 trades) ties C 12 (6 trades) -> more trades wins
        out = self.wf("total_r")
        self.assertEqual(self.chosen(out), [0.5, 0.55, 0.55])
        self.assertEqual((out["oos"]["n"], out["oos"]["total_r"]), (30, 6.0))
        # profit_factor: no-loss combos are capped at 99 -> the one with more trades
        out = self.wf("profit_factor")
        self.assertEqual(self.chosen(out), [0.55, 0.55, 0.55])
        self.assertEqual(out["oos"]["total_r"], 18.0)
        with self.assertRaises(ValueError):
            self.wf("sharpe")

    def test_ranking_best_top_and_sensitivity(self):
        out = self.wf()
        # min_trades 20: A (24, -0.25) and B (48, 0.5) are eligible, C (12, 1.0) is not
        self.assertEqual([t["params"]["entry_zone_lo"] for t in out["top"]], [0.55, 0.5, 0.6])
        best = out["best"]
        self.assertEqual((best["params"], best["n"], best["avg_r"], best["eligible"]),
                         ({"entry_zone_lo": 0.55}, 48, 0.5, True))
        self.assertEqual(best["spec"], SPEC.with_overrides(entry_zone_lo=0.55).to_dict())
        self.assertEqual(best["spec_id"], as_spec(best["spec"]).spec_id())
        self.assertEqual(set(out["top"][0]), {"params", "spec_id", "n", "avg_r", "profit_factor",
                                              "fold_positive_frac"})
        self.assertEqual(out["sensitivity"], {"entry_zone_lo": {"0.5": -0.25, "0.55": 0.5, "0.6": 1.0}})
        low = self.wf(research={"min_trades": 0})
        self.assertEqual(low["best"]["params"], {"entry_zone_lo": 0.6})

    def test_no_eligible_combo_in_a_fold(self):
        out = self.wf(research={"min_trades": 400})          # 400 // 4 = 100 in-sample trades
        self.assertEqual(self.chosen(out), [None, None, None])
        self.assertEqual(out["oos"]["n"], 0)
        self.assertEqual([f["n"] for f in out["oos_by_fold"]], [0, 0, 0])
        self.assertFalse(out["best"]["eligible"])

    def test_grid_errors_before_any_backtest(self):
        cases = [({"entry_zone_lo": [0.5, 0.55, 0.6, 0.62, 0.64]}, {"max_grid_size": 4}),
                 ({"nope": [1]}, {}), ({"name": ["x"]}, {}), ({"min_rr": []}, {}),
                 ({"min_rr": 2.0}, {}), ({"trend_filter": "ema"}, {}),
                 ({"timeframe": ["4h", "1h"]}, {}),
                 ({"min_rr": [1.0]}, {"folds": 1})]
        for grid, research in cases:
            with self.subTest(grid=grid), mock.patch.object(rs, "backtest") as bt:
                with self.assertRaises(ValueError):
                    walk_forward_optimize(SPEC, grid, {}, cfg_(research), WF_WINDOWS)
                bt.assert_not_called()

    def test_duplicate_and_all_invalid_combos(self):
        with mock.patch.object(rs, "backtest", side_effect=fake_backtest) as bt:
            out = walk_forward_optimize(SPEC, {"entry_zone_lo": [0.5, 0.5]}, {}, cfg_(), WF_WINDOWS)
            self.assertEqual((out["n_combos"], out["n_evaluated"], bt.call_count), (2, 1, 1))
            out = walk_forward_optimize(SPEC, {"min_rr": [-1.0, 11.0]}, {}, cfg_(), WF_WINDOWS)
        self.assertEqual((out["n_invalid"], out["n_evaluated"], out["best"], out["top"]),
                         (2, 0, None, []))
        self.assertEqual(out["sensitivity"], {})


class TestWalkForwardReal(Universe):
    def test_real_grid(self):
        grid = {"stop_ratio": [0.786, 1.0], "min_rr": [1.0, 2.0]}
        seen = []
        out = walk_forward_optimize(SPEC, grid, self.bars, self.cfg, self.w,
                                    on_combo=lambda s, r: seen.append(r.spec_id))
        self.assertEqual((out["n_combos"], out["n_evaluated"], len(seen)), (4, 4, 4))
        self.assertLess(len(dumps(out)), 12000)
        best = out["best"]
        self.assertEqual(as_spec(best["spec"]).spec_id(), best["spec_id"])
        self.assertIn(best["spec_id"], seen)
        res, _ = evaluate(as_spec(best["spec"]), self.bars, self.cfg, self.w)
        self.assertEqual(best["n"], res.metrics["n"])
        self.assertEqual(len(out["chosen"]), 3)
        self.assertEqual(out["oos"]["n"], sum(f["n"] for f in out["oos_by_fold"]))


INACTIVE_CASES = [
    (SPEC.with_overrides(name="limit_mid"), {"swing_mult", "trend_ema", "confirm_type", "confirm_window"}),
    (SPEC.with_overrides(name="confirm_lo_tp1", entry_mode="confirm", entry_at="lo", tp1_frac=1.0,
                         swing_atr_len=5, min_rr=0.5),
     {"swing_mult", "trend_ema", "entry_zone_hi", "tp2_ext", "trail_atr"}),
    (SPEC.with_overrides(name="limit_hi_atr", swing_mode="atr", entry_at="hi"),
     {"swing_pct", "trend_ema", "confirm_type", "confirm_window", "entry_zone_lo"}),
]


class TestNeighbours(unittest.TestCase):
    def moved(self, base, nbs):
        out = []
        for nb in nbs:
            self.assertIs(nb.validate(), nb)
            diff = [k for k, v in nb.to_dict().items() if v != getattr(base, k)]
            self.assertEqual(len(diff), 1, diff)
            out.append((diff[0], getattr(nb, diff[0])))
        return out

    def test_one_step_moves(self):
        base = StrategySpec()
        moves = self.moved(base, neighbours(base))
        params = {k for k, _ in moves}
        self.assertEqual(params, {"swing_mode", "swing_atr_len", "swing_mult", "min_leg_atr",
                                  "setup_expiry_bars", "trend_filter", "trend_ema", "adx_min",
                                  "entry_zone_lo", "entry_zone_hi", "entry_mode", "entry_at",
                                  "stop_ratio", "stop_atr_buffer", "tp1_ext", "tp1_frac", "tp2_ext",
                                  "trail_atr", "max_hold_bars", "min_rr"})
        for k, v in moves:
            p, old = PARAM_SPACE[k], getattr(base, k)
            if p["type"] == "choice":
                self.assertEqual(abs(p["choices"].index(v) - p["choices"].index(old)), 1)
            else:
                self.assertAlmostEqual(abs(v - old), p["step"])
        self.assertEqual(dict(moves)["adx_min"], 5.0)               # 0 - 5 is out of range
        self.assertEqual([v for k, v in moves if k == "entry_zone_lo"], [0.568])  # 0.668 > hi
        self.assertEqual([v for k, v in moves if k == "trend_filter"], ["none", "ema_slope"])
        self.assertEqual(len({nb.spec_id() for nb in neighbours(base)}), len(moves))

    def test_inactive_and_fixed_params_never_move(self):
        movable = {k for k, p in PARAM_SPACE.items() if p["type"] != "bool"} - {"timeframe", "direction"}
        moved = {}
        for base, inactive in INACTIVE_CASES:
            moved[base.name] = moves = self.moved(base, neighbours(base.to_dict()))
            self.assertEqual({k for k, _ in moves}, movable - inactive)
        confirm = moved["confirm_lo_tp1"]
        self.assertEqual([v for k, v in confirm if k == "swing_atr_len"], [7])   # 5 is the minimum
        self.assertTrue({"confirm_window", "entry_at", "entry_zone_lo"} <= {k for k, _ in confirm})

    def test_inactive_params_really_do_not_change_trades(self):
        bars = make_bars("S0USDT", n=1500, seed=0)

        def outcome(spec):
            r = FibEngine(spec).run(bars)
            return [(t.entry_t, t.exit_t, t.r_multiple) for t in r.trades]

        for base, inactive in INACTIVE_CASES:
            want = outcome(base)
            self.assertTrue(want)
            for k in inactive:
                p = PARAM_SPACE[k]
                for v in ([c for c in p["choices"] if c != getattr(base, k)] if p["type"] == "choice"
                          else (getattr(base, k) - p["step"], getattr(base, k) + p["step"])):
                    try:
                        moved = base.with_overrides(**{k: v}).validate()
                    except ValueError:
                        continue
                    with self.subTest(base=base.name, param=k, value=v):
                        self.assertEqual(outcome(moved), want)


class TestRobustness(Universe):
    def test_rows_and_aggregates(self):
        seen = []
        out = robustness(SPEC, self.bars, self.cfg, self.w, on_combo=lambda s, r: seen.append(s))
        cands = neighbours(SPEC)
        self.assertEqual((out["n"], out["n_candidates"]), (16, len(cands)))
        picked = [cands[i * len(cands) // 16] for i in range(16)]
        self.assertEqual(seen, picked)
        self.assertEqual([r["spec_id"] for r in out["rows"]], [s.spec_id() for s in picked])
        for row, nb in zip(out["rows"], picked):
            self.assertEqual(getattr(nb, row["param"]), row["value"])
        avg = [r["avg_r"] for r in out["rows"]]
        self.assertEqual(out["positive_frac"], rounded(sum(a > 0 for a in avg) / 16))
        self.assertEqual(out["worst_avg_r"], min(avg))
        res, ex = evaluate(picked[3], self.bars, self.cfg, self.w)
        self.assertEqual(out["rows"][3]["n"], res.metrics["n"])
        self.assertEqual(out["rows"][3]["fold_positive_frac"], ex["fold_positive_frac"])
        dumps(out)

    def test_subset_sizes(self):
        cands = neighbours(SPEC)
        out = robustness(SPEC, self.bars, self.cfg, self.w, max_neighbours=3)
        self.assertEqual([r["spec_id"] for r in out["rows"]],
                         [cands[i * len(cands) // 3].spec_id() for i in range(3)])
        out = robustness(SPEC, self.bars, self.cfg, self.w, max_neighbours=0)
        self.assertEqual((out["n"], out["positive_frac"], out["median_avg_r"], out["worst_avg_r"],
                          out["rows"]), (0, 0.0, 0.0, 0.0, []))


class TestValidationGate(unittest.TestCase):
    GATE = DEFAULTS["research"]["validation_gate"]

    def result(self, rs_=(1.5, -1.0, 2.0, 0.5), reps=10, symbols=5, extra=()):
        trades = [trade(f"S{i % symbols}USDT", T0 + i * H4, r)
                  for i, r in enumerate(list(rs_) * reps)] + list(extra)
        syms = sorted({t.symbol for t in trades}) + ["IDLEUSDT"]
        return result_from_trades(SPEC, trades, syms, T0, T0 + 200 * H4, cfg_())

    def checks(self, gate_out):
        return {c["name"]: c for c in gate_out["checks"]}

    def test_pass(self):
        res = self.result()
        out = validation_gate(res, self.GATE, n_trials=5, var_sr=0.02)
        dumps(out)
        self.assertTrue(out["passed"])
        c = self.checks(out)
        self.assertEqual(list(c), ["min_trades", "min_profit_factor", "min_expectancy_r",
                                   "min_deflated_sharpe", "max_drawdown", "min_symbol_win_frac"])
        rs_ = [t.r_multiple for t in res.closed_trades]
        self.assertEqual(out["deflated_sharpe"], rounded(deflated_sharpe(rs_, 5, 0.02)))
        self.assertEqual((c["min_trades"]["value"], c["min_profit_factor"]["value"],
                          c["min_expectancy_r"]["value"], c["min_symbol_win_frac"]["value"]),
                         (40, 4.0, 0.75, 1.0))
        self.assertEqual(c["max_drawdown"]["value"], rounded(res.metrics["max_drawdown"]))
        self.assertEqual({k: v["threshold"] for k, v in c.items()}, self.GATE)

    def test_each_check_can_fail_alone(self):
        res = self.result()
        for name, threshold in (("min_trades", 41), ("min_profit_factor", 4.5),
                                ("min_expectancy_r", 0.8), ("min_deflated_sharpe", 0.99999),
                                ("max_drawdown", 0.0001), ("min_symbol_win_frac", 1.01)):
            with self.subTest(name):
                out = validation_gate(res, {**self.GATE, name: threshold}, 1, None)
                self.assertFalse(out["passed"])
                self.assertEqual([c["name"] for c in out["checks"] if not c["passed"]], [name])

    def test_trials_deflate_and_symbols_without_trades_are_ignored(self):
        res = self.result()
        few = validation_gate(res, self.GATE, 1, None)["deflated_sharpe"]
        many = validation_gate(res, self.GATE, 10_000, 0.5)["deflated_sharpe"]
        self.assertLess(many, few)
        loser = self.result(extra=[trade("LOSEUSDT", T0 + 300 * H4, -2.0)])
        frac = self.checks(validation_gate(loser, {}, 1, None))["min_symbol_win_frac"]
        self.assertEqual((frac["value"], frac["threshold"]), (0.8333, 0.5))   # 5 of 6 traded

    def test_open_trades_do_not_count_and_empty_fails(self):
        res = self.result(extra=[trade("S0USDT", T0 + 500 * H4, 5.0, is_open=True)])
        self.assertEqual(self.checks(validation_gate(res, self.GATE, 1, None))["min_trades"]["value"], 40)
        empty = result_from_trades(SPEC, [], ["AUSDT"], T0, T0 + 10 * H4, cfg_())
        out = validation_gate(empty, self.GATE, 1, None)
        self.assertFalse(out["passed"])
        self.assertEqual([c["passed"] for c in out["checks"]], [False, False, False, False, True, False])
        self.assertEqual(out["deflated_sharpe"], 0.0)


if __name__ == "__main__":
    unittest.main()
