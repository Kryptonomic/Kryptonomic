"""Regression tests for the stats_research review findings (research.py, backtest.py,
portfolio.py): skill-vs-random validation check (F02), pinned validation window (F04),
mark-to-market drawdown (F19), baseline entry cost (F20), holdout deflation and the trial floor
(F22, F46), full-neighbourhood robustness (F36), no-op duplicates (F55), causal same-bar order
(F41), no leverage (F43), per-side breakdown (F30) and the regime-conditioned baseline (F40)."""
import json
import math
import random
import unittest
from types import SimpleNamespace
from unittest import mock

from fibbot import research as rs
from fibbot.backtest import backtest, costs_from_cfg, result_from_trades
from fibbot.config import DEFAULTS
from fibbot.indicators import atr, ema
from fibbot.metrics import expected_max_sharpe, probabilistic_sharpe, skew_kurt
from fibbot.portfolio import marked_curve, simulate_portfolio
from fibbot.strategy import PRESETS, _filter_ok, manage_position, warmup_bars
from fibbot.types import Bars, Trade
from tests.test_backtest import H4, SPEC, T0, cfg_, make_bars, make_universe, trade

DAY = 86400
GATE = DEFAULTS["research"]["validation_gate"]


def checks(out):
    return {c["name"]: c for c in out["checks"]}


def bars_from(symbol, ohlc, t0=T0, tf="4h", step=H4):
    return Bars.from_rows(symbol, tf, [(t0 + i * step, o, h, l, c, 1.0)
                                       for i, (o, h, l, c) in enumerate(ohlc)])


def hand_trade(symbol, entry_t, entry, stop, exit_t, fills, reason, side=1, fee=0.001, risk=None):
    """A closed trade whose ret_pct follows strategy._Position.result from its fills."""
    gross = sum(f * p for f, p, _, _ in fills)
    ret = gross / entry * (1 - fee) - (1 + fee) if side > 0 else (1 - fee) - gross / entry * (1 + fee)
    risk = risk if risk is not None else side * (entry - stop) / entry
    return Trade("sid", symbol, "4h", side, f"L{entry_t}", entry_t - H4, entry_t, entry, stop, exit_t,
                 gross, reason, ret, risk, ret / risk, (exit_t - entry_t) // H4,
                 {"tp1": entry * 1.1, "tp2": entry * 1.2, "fills": [list(f) for f in fills]})


def fake_result(rs_values, symbols=4, entry_step=DAY):
    """A minimal BacktestResult stand-in for validation_gate: only its DSR / edge checks matter
    (every other check passes)."""
    trades = [SimpleNamespace(r_multiple=r, entry_t=T0 + k * entry_step, symbol=f"S{k % symbols}")
              for k, r in enumerate(rs_values)]
    return SimpleNamespace(closed_trades=trades,
                           metrics={"n": len(trades), "profit_factor": 9.0, "expectancy_r": 9.0,
                                    "max_drawdown": 0.0},
                           per_symbol={f"S{i}": {"n": 1, "total_r": 1.0} for i in range(symbols)})


def two_point(n, wins, seed=0):
    """n R multiples of +2R wins and -1R losses, deterministic order."""
    xs = [2.0] * wins + [-1.0] * (n - wins)
    random.Random(seed).shuffle(xs)
    return xs


# -- F02: the validation gate must see an edge over random entries ------------------------------

def drift_bars(sym, n=1200, drift=0.0025, vol=0.012, seed=0):
    """A steadily rising market: any long entry tends to make money."""
    rnd, px, rows = random.Random(seed), 100.0, []
    for i in range(n):
        o = px
        px *= math.exp(drift + rnd.gauss(0, vol))
        rows.append((T0 + i * H4, o, max(o, px) * (1 + abs(rnd.gauss(0, 0.003))),
                     min(o, px) * (1 - abs(rnd.gauss(0, 0.003))), px, 1.0))
    return Bars.from_rows(sym, "4h", rows)


def picked_trades(bars, cfg, picker, seed, risk=0.03, k=25):
    """Long trades entered at the bars `picker` chooses, with SPEC's exits (fixed 3% stop, tp1 2R,
    tp2 3R) - the same template random_baseline copies."""
    costs, out = costs_from_cfg(cfg), []
    for j, (sym, b) in enumerate(sorted(bars.items())):
        a = atr(b.h, b.l, b.c, SPEC.swing_atr_len)
        for e in picker(b, random.Random(seed * 100 + j), k):
            px = b.o[e]
            stop, tp1, tp2 = px * (1 - risk), px * (1 + 2 * risk), px * (1 + 3 * risk)
            r = manage_position(b, a, e, 1, px, stop, tp1, tp2, SPEC, costs)
            out.append(Trade("sid", sym, "4h", 1, f"L{b.t[e]}", b.t[e] - H4, b.t[e], px, stop,
                             None if r["open"] else b.t[r["exit_idx"]], r["exit_price"],
                             r["exit_reason"], r["ret_pct"], r["risk_pct"], r["r_multiple"],
                             r["exit_idx"] - e, {"tp1": tp1, "tp2": tp2, "fills": r["fills"]}))
    return out


def random_pick(b, rnd, k):
    return sorted(rnd.randrange(200, len(b) - 100) for _ in range(k))


def skill_pick(b, rnd, k):
    """Planted timing skill: enter where the next 12 bars rise the most."""
    best = sorted(range(200, len(b) - 100), key=lambda e: -(b.c[e + 12] / b.o[e]))
    return sorted(rnd.sample(best[:len(best) // 6], k))


class TestValidationEdgeOverRandom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg_()
        cls.bars = {f"D{i}USDT": drift_bars(f"D{i}USDT", seed=i) for i in range(3)}
        cls.start, cls.end = T0 + 200 * H4, T0 + 1200 * H4

    def gate(self, picker, seed):
        trades = picked_trades(self.bars, self.cfg, picker, seed)
        res = result_from_trades(SPEC, trades, sorted(self.bars), self.start, self.end, self.cfg,
                                 bars_by_symbol=self.bars)
        base = rs.random_baseline(SPEC, self.bars, self.cfg, self.start, self.end, n_runs=50,
                                  seed=7, result=res)
        return (rs.validation_gate(res, GATE, 1, None),
                rs.validation_gate(res, GATE, 1, None, baseline=base), base)

    def test_random_entries_in_a_rising_market_fail_only_the_edge_check(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                plain, with_base, base = self.gate(random_pick, seed)
                self.assertTrue(plain["passed"])             # every absolute check passes on drift
                self.assertGreater(base["baseline_avg_r_mean"], 1.0)   # random longs earn > 1R
                c = checks(with_base)
                self.assertFalse(with_base["passed"])
                self.assertFalse(c["min_validation_edge_z"]["passed"])
                self.assertLessEqual({n for n, v in c.items() if not v["passed"]},
                                     {"min_validation_edge_r", "min_validation_edge_z"})
                self.assertEqual(c["min_validation_edge_z"]["threshold"], rs.MIN_VALIDATION_EDGE_Z)

    def test_planted_timing_skill_passes(self):
        for seed in range(2):
            with self.subTest(seed=seed):
                _, with_base, _ = self.gate(skill_pick, seed)
                self.assertTrue(with_base["passed"], with_base)
                self.assertGreater(with_base["edge"]["edge_z"], 3.0)

    def test_thresholds_from_the_gate_and_none_drops_a_check(self):
        plain, _, base = self.gate(random_pick, 0)
        res = result_from_trades(SPEC, picked_trades(self.bars, self.cfg, random_pick, 0),
                                 sorted(self.bars), self.start, self.end, self.cfg)
        out = rs.validation_gate(res, {**GATE, "min_validation_edge_z": None,
                                       "min_validation_edge_r": -5.0}, 1, None, baseline=base)
        c = checks(out)
        self.assertNotIn("min_validation_edge_z", c)
        self.assertEqual(c["min_validation_edge_r"]["threshold"], -5.0)
        self.assertTrue(out["passed"])
        self.assertNotIn("min_validation_edge_r", checks(plain))      # no baseline: no edge checks
        json.dumps(out, allow_nan=False)

    def test_the_regime_baseline_binds_when_it_is_stricter(self):
        res = fake_result([1.0, 0.5, 0.0, 1.5] * 10)
        base = {"n_runs": 50, "baseline_avg_r_mean": 0.0, "baseline_avg_r_std": 0.1,
                "regime_filtered": True, "baseline_regime_avg_r_mean": 0.6,
                "baseline_regime_avg_r_std": 0.1}
        e = checks(rs.validation_gate(res, GATE, 1, None, baseline=base))
        self.assertAlmostEqual(e["min_validation_edge_r"]["value"], 0.75 - 0.6, places=4)
        unfiltered = {**base, "regime_filtered": False}
        e = checks(rs.validation_gate(res, GATE, 1, None, baseline=unfiltered))
        self.assertAlmostEqual(e["min_validation_edge_r"]["value"], 0.75, places=4)

    def test_edge_stats_hand_computed(self):
        tr = [SimpleNamespace(r_multiple=r, entry_t=T0 + k * 8 * DAY) for k, r in enumerate([1, -1, 2, 0])]
        out = rs.edge_stats(tr, 0.1, 0.2, 4)
        se = math.sqrt(sum((r - 0.5) ** 2 for r in (1, -1, 2, 0)) / 3) / 2   # one trade per week
        self.assertAlmostEqual(out["strategy_se"], se)
        self.assertAlmostEqual(out["edge_r"], 0.4)
        self.assertAlmostEqual(out["edge_se"], math.sqrt(max(se, 0.2) ** 2 + 0.04 / 4))
        self.assertAlmostEqual(out["edge_z"], 0.4 / out["edge_se"])
        same_week = [SimpleNamespace(r_multiple=x.r_multiple, entry_t=T0) for x in tr]
        self.assertGreater(rs.edge_stats(same_week, 0.1, 0.2, 4)["strategy_se"], se)   # clustered
        self.assertEqual(rs.edge_stats([], 0.0, 0.0, 1)["edge_z"], 0.0)


# -- F04: a pinned validation window does not slide with new data ------------------------------

class TestPinnedValidationWindow(unittest.TestCase):
    def test_new_bars_do_not_move_a_pinned_window(self):
        cfg = cfg_({"validation_months": 2})
        short = {"A": make_bars("A", n=1000)}
        sliding = rs.compute_windows(cfg, short)
        epoch = sliding["data_end"]
        pinned = rs.compute_windows(cfg, short, validation_end=epoch)
        self.assertEqual(pinned, {**sliding, "pinned": True})          # pinned "now" == sliding now
        longer = {"A": make_bars("A", n=1300)}                         # 50 days later
        later = rs.compute_windows(cfg, longer, validation_end=epoch)
        for k in ("research_start", "research_end", "validation_start", "validation_end"):
            self.assertEqual(later[k], pinned[k], k)
        self.assertEqual(later["data_end"], T0 + 1300 * H4)
        self.assertGreater(rs.compute_windows(cfg, longer)["validation_start"],
                           pinned["validation_start"])                  # unpinned still slides
        # bars after the epoch are in neither window: forward / paper data only
        self.assertLessEqual(rs.window_bounds(later, "validation")[1], epoch)

    def test_an_epoch_past_the_data_is_clamped_but_keeps_its_start(self):
        cfg = cfg_({"validation_months": 2})
        bars = {"A": make_bars("A", n=1000)}
        epoch = T0 + 1100 * H4
        w = rs.compute_windows(cfg, bars, validation_end=epoch)
        self.assertEqual(w["validation_end"], T0 + 1000 * H4)
        vs = int(epoch - 2 * 30.44 * DAY) // DAY * DAY
        self.assertEqual((w["validation_start"], w["research_end"]), (vs, vs))


# -- F19: drawdown, Sharpe and Sortino see open-position marks ---------------------------------

class TestMarkToMarketEquity(unittest.TestCase):
    PCFG = {"risk_per_trade": 0.05}            # notional 2500 = the 25% cap

    def result(self, trades, bars, marked=True):
        cfg = cfg_(portfolio=self.PCFG)
        return result_from_trades(SPEC, trades, list(bars), T0, T0 + len(next(iter(bars.values()))) * H4,
                                  cfg, bars_by_symbol=bars if marked else None)

    def test_a_give_back_inside_a_winning_trade_is_a_drawdown(self):
        bars = {"AUSDT": bars_from("AUSDT", [(100, 101, 99, 100), (100, 100, 79, 80),
                                             (80, 106, 80, 104)])}
        tr = hand_trade("AUSDT", T0, 100.0, 90.0, T0 + 2 * H4, [(1.0, 105.0, "tp2", T0 + 2 * H4)],
                        "target")
        marked, realized = self.result([tr], bars), self.result([tr], bars, marked=False)
        self.assertEqual((marked.equity_basis, realized.equity_basis), ("marked", "realized"))
        self.assertEqual(realized.metrics["max_drawdown"], 0.0)       # only a winning exit
        low = 2500 * (80 * (1 - 0.0005) / 100 * 0.999 - 1.001)          # -20% marked with exit costs
        self.assertAlmostEqual(marked.metrics["max_drawdown"], -low / 10000, places=9)
        self.assertGreater(marked.metrics["max_drawdown"], 0.05)
        self.assertAlmostEqual(marked.equity_curve[-2][1], marked.portfolio.final_equity, places=9)
        self.assertEqual(marked.metrics["total_return"], realized.metrics["total_return"])

    def test_partial_fills_count_at_their_fill_time_and_the_peak_is_marked(self):
        fee, slip, be = 0.001, 0.0005, 100.0 * (1 + 2 * 0.001)
        bars = {"AUSDT": bars_from("AUSDT", [(100, 101.5, 99.5, 101), (101, 113, 101, 112),
                                             (112, 121, 111, 120), (120, 120, 99, 100)])}
        fills = [(0.5, 110.0, "tp1", T0 + H4), (0.5, be * (1 - slip), "breakeven", T0 + 3 * H4)]
        tr = hand_trade("AUSDT", T0, 100.0, 90.0, T0 + 3 * H4, fills, "breakeven")
        res = self.result([tr], bars)
        pts = dict(res.equity_curve[1:-1])
        bar1 = 2500 * ((0.5 * 110 + 0.5 * 112 * (1 - slip)) / 100 * (1 - fee) - (1 + fee))
        peak = 2500 * ((0.5 * 110 + 0.5 * 120 * (1 - slip)) / 100 * (1 - fee) - (1 + fee))
        self.assertAlmostEqual(pts[T0 + H4] - 10000, bar1, places=9)    # tp1 at 110, rest at 112
        self.assertAlmostEqual(pts[T0 + 2 * H4] - 10000, peak, places=9)
        final = res.portfolio.final_equity
        self.assertAlmostEqual(pts[T0 + 3 * H4], final, places=9)
        self.assertAlmostEqual(res.metrics["max_drawdown"], (10000 + peak - final) / (10000 + peak),
                               places=9)
        self.assertEqual(self.result([tr], bars, marked=False).metrics["max_drawdown"], 0.0)

    def test_marked_curve_ends_at_final_equity_in_a_real_backtest_with_open_trades(self):
        bars, cfg = make_universe(3), cfg_()
        start = T0 + 400 * H4
        full = backtest(SPEC, bars, start, T0 + 1400 * H4, cfg)
        cut = next(t for t in full.closed_trades if t.bars_held > 2).exit_t
        res = backtest(SPEC, bars, start, cut, cfg)
        self.assertGreater(res.metrics["n_open"], 0)
        c = costs_from_cfg(cfg)
        curve = marked_curve(res.portfolio, bars, c.fee_rate, c.slippage, cut)
        self.assertAlmostEqual(curve[-1][1], res.portfolio.final_equity, places=6)
        self.assertTrue(all(t < cut for t, _ in curve))
        self.assertGreaterEqual(res.metrics["max_drawdown"], 0.0)

    def test_skipped_and_unbarred_positions(self):
        pf = simulate_portfolio([trade("AUSDT", T0, 2.0), trade("BUSDT", T0, 1.0)],
                                {"max_open_positions": 1})
        self.assertEqual(len(pf.taken), 1)
        # no bars for the symbol: its pnl lands at exit_t like the realized curve
        self.assertEqual(marked_curve(pf, {}, 0.001, 0.0005),
                         [(T0 + H4, pf.start_equity + pf.taken[0]["pnl"])])
        self.assertEqual(marked_curve(simulate_portfolio([], {}), {}, 0.001, 0.0005), [])


# -- F20: the baseline pays the strategy's own entry cost --------------------------------------

class TestBaselineEntryCost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = make_universe(3, n=2000)
        cls.cfg = cfg_({"validation_months": 2})
        cls.w = rs.compute_windows(cls.cfg, cls.bars)

    def entries(self, spec, cfg):
        calls = []

        def spy(*args, **kw):
            calls.append((args[0].symbol, args[2], args[3], args[4], args[0].o[args[2]]))
            return manage_position(*args, **kw)

        res = backtest(spec, self.bars, self.w["research_start"], self.w["research_end"], cfg)
        with mock.patch.object(rs, "manage_position", side_effect=spy):
            rs.random_baseline(spec, self.bars, cfg, self.w["research_start"],
                               self.w["research_end"], n_runs=3, seed=4, result=res)
        self.assertTrue(calls)
        return calls

    def test_limit_entries_pay_no_slippage(self):
        dear = self.entries(SPEC, self.cfg)
        free = self.entries(SPEC, cfg_({"validation_months": 2}, costs={"slippage": 0.0}))
        self.assertEqual(dear, free)              # same bars, same price: slippage has no effect
        self.assertTrue(all(px == o for _, _, _, px, o in dear))

    def test_confirm_entries_pay_market_slippage(self):
        spec = SPEC.with_overrides(entry_mode="confirm", confirm_type="reclaim")
        slip = self.cfg["costs"]["slippage"]
        for _, _, side, px, o in self.entries(spec, self.cfg):
            self.assertAlmostEqual(px, o * (1 + side * slip))


# -- F22 / F46: holdout-based deflation and a floor on the trial count ---------------------------

class TestGateDeflation(unittest.TestCase):
    def dsr_check(self, values, *args, **kw):
        out = rs.validation_gate(fake_result(values), GATE, *args, **kw)
        return checks(out)["min_deflated_sharpe"], out

    def test_holdout_hurdle_falls_with_more_validation_trades_and_rises_with_looks(self):
        small = self.dsr_check(two_point(100, 40), 0, None, holdout_looks=50)[1]
        large = self.dsr_check(two_point(400, 160), 0, None, holdout_looks=50)[1]
        self.assertEqual((small["dsr_basis"], small["dsr_trials"]), ("holdout", 50))
        self.assertAlmostEqual(small["sr_star"], expected_max_sharpe(50, 1 / 100), places=4)
        self.assertLess(large["sr_star"], small["sr_star"])
        more = self.dsr_check(two_point(400, 160), 0, None, holdout_looks=500)[1]
        self.assertGreater(more["sr_star"], large["sr_star"])
        wide = self.dsr_check(two_point(400, 160), 0, None, holdout_looks=50, var_holdout_sr=0.01)[1]
        self.assertAlmostEqual(wide["sr_star"], expected_max_sharpe(50, 0.01), places=4)

    def test_a_strong_holdout_edge_passes_with_holdout_looks_but_not_research_bookkeeping(self):
        values = two_point(450, 212, seed=3)                  # per-trade Sharpe ~0.276
        research, _ = self.dsr_check(values, 8620, 0.0042)
        holdout, out = self.dsr_check(values, 8620, 0.0042, holdout_looks=260)
        self.assertFalse(research["passed"])
        self.assertTrue(holdout["passed"])
        self.assertEqual(out["dsr_basis"], "holdout")

    def test_zero_edge_holdout_pass_rate_stays_small_for_any_number_of_looks(self):
        rng = random.Random(11)
        for k in (1, 10, 100, 500):
            hits = sum(self.dsr_check([rng.choice((2.0, -1.0, -1.0)) for _ in range(150)], 0, None,
                                      holdout_looks=k)[0]["passed"] for _ in range(400))
            self.assertLessEqual(hits / 400, 0.06, k)

    def test_a_lone_first_look_is_not_a_ten_percent_test(self):
        # PSR(SR* = 0) = 0.917 passed the old N=1 gate (a one-sided 10% test)
        values = two_point(100, 40, seed=5)
        mu = sum(values) / 100
        sd = math.sqrt(sum((v - mu) ** 2 for v in values) / 99)
        skew, kurt = skew_kurt(values)
        self.assertGreater(probabilistic_sharpe(mu / sd, 100, skew, kurt, 0.0), 0.9)
        check, out = self.dsr_check(values, 1, None)
        self.assertFalse(check["passed"])
        self.assertEqual(out["dsr_trials"], rs.MIN_GATE_TRIALS)
        self.assertEqual(self.dsr_check(values, 1, None, holdout_looks=1)[1]["dsr_trials"],
                         rs.MIN_GATE_TRIALS)

    def test_zero_edge_false_pass_rate_with_one_trial(self):
        rng = random.Random(5)
        for n in (74, 150):
            hits = sum(self.dsr_check([rng.choice((2.0, -1.0, -1.0)) for _ in range(n)], 1, None)[0]["passed"]
                       for _ in range(1000))
            self.assertLessEqual(hits / 1000, 0.06, n)


# -- F36 / F55: the whole neighbourhood, no-op neighbours are not trials ------------------------

def fake_evaluate(dups=None):
    """research.evaluate stand-in: every spec trades differently (its R is a hash of its id),
    except specs for which dups(spec) returns a label: those trade like that label."""
    def run(spec, bars, cfg, windows, window="research"):
        label = (dups(spec) if dups else None) or spec.spec_id()
        r = int(rs.hashlib.sha1(label.encode()).hexdigest()[:6], 16) / 16 ** 6 - 0.3
        res = SimpleNamespace(spec_id=spec.spec_id(),
                              trades=[SimpleNamespace(symbol="X", side=1, entry_t=0, exit_t=1, r_multiple=r)],
                              metrics={"n": 1, "avg_r": r, "total_r": r, "profit_factor": 1.0})
        return res, {"fold_positive_frac": 0.5}
    return run


class TestNeighbourhood(unittest.TestCase):
    def test_every_preset_tests_its_whole_neighbourhood(self):
        with mock.patch.object(rs, "evaluate", side_effect=fake_evaluate()):
            for name, spec in PRESETS.items():
                with self.subTest(preset=name):
                    cands = rs.neighbours(spec)
                    self.assertLessEqual(len(cands), rs.MAX_NEIGHBOURS)
                    out = rs.robustness(spec, {}, cfg_(), {})
                    self.assertEqual(out["n"], out["n_candidates"])
                    self.assertEqual({(r["param"], r["value"]) for r in out["rows"]},
                                     {(rs._moved_param(spec, nb), getattr(nb, rs._moved_param(spec, nb)))
                                      for nb in cands})
                    self.assertEqual(out["params_untested"], [])
                    both = [p for p, v in out["coverage"].items() if len(v) == 2]
                    self.assertGreater(len(both), 5)            # both directions of most params

    def test_a_cap_takes_whole_parameters_spread_over_the_list(self):
        spec = PRESETS["golden_pocket_4h"]
        cands = rs.neighbours(spec)
        params = list(dict.fromkeys(rs._moved_param(spec, nb) for nb in cands))
        with mock.patch.object(rs, "evaluate", side_effect=fake_evaluate()):
            out = rs.robustness(spec, {}, cfg_(), {}, max_neighbours=16)
        self.assertLessEqual(out["n"], 16)
        self.assertGreaterEqual(out["n"], 14)
        for p in params:
            want = [getattr(nb, p) for nb in cands if rs._moved_param(spec, nb) == p]
            self.assertIn(out["coverage"].get(p, []), ([], want), p)
        tested = [params.index(p) for p in out["coverage"]]
        self.assertTrue(any(i < len(params) // 3 for i in tested))
        self.assertTrue(any(i >= 2 * len(params) // 3 for i in tested))
        self.assertEqual(sorted(out["params_untested"] + list(out["coverage"]), key=params.index), params)

    def test_no_op_neighbours_are_flagged_and_not_trials(self):
        spec = PRESETS["golden_pocket_4h"]

        def dups(nb):
            moved = rs._moved_param(spec, nb) if nb != spec else None
            if moved == "min_rr":
                return spec.spec_id()                        # trades like the base
            if moved == "tp1_frac":
                return "tp1_frac-any"                        # both moves trade alike
            return None

        seen = []
        with mock.patch.object(rs, "evaluate", side_effect=fake_evaluate(dups)):
            out = rs.robustness(spec, {}, cfg_(), {}, on_combo=lambda s, r: seen.append(s.spec_id()))
        rows = {(r["param"], r["value"]): r for r in out["rows"]}
        self.assertEqual({rows[k]["duplicate_of"] for k in rows if k[0] == "min_rr"}, {"base"})
        tp = [r for r in out["rows"] if r["param"] == "tp1_frac"]
        self.assertEqual([r["duplicate_of"] for r in tp], [None, tp[0]["spec_id"]])
        unique = [r for r in out["rows"] if r["duplicate_of"] is None]
        self.assertEqual(seen, [r["spec_id"] for r in unique])
        self.assertEqual((out["n_distinct"], out["n_duplicates"]), (len(unique), 3))
        avg = [r["avg_r"] for r in unique]
        self.assertEqual(out["worst_avg_r"], min(avg))
        self.assertAlmostEqual(out["positive_frac"], sum(a > 0 for a in avg) / len(avg), places=3)

    def test_optimize_records_one_trial_per_distinct_outcome(self):
        def fake_bt(spec, bars, start, end, cfg, workers=None):
            r = {0.5: 1.0, 0.55: 1.0, 0.6: -1.0}[spec.entry_zone_lo]   # 0.5 and 0.55 trade alike
            return result_from_trades(spec, [trade("AUSDT", T0 + j * H4, r) for j in range(8)],
                                      ["AUSDT"], start, end, cfg)
        seen = []
        with mock.patch.object(rs, "backtest", side_effect=fake_bt):
            out = rs.walk_forward_optimize(SPEC, {"entry_zone_lo": [0.5, 0.55, 0.6]}, {}, cfg_(),
                                           {"research_start": T0, "research_end": T0 + 40 * H4},
                                           on_combo=lambda s, r: seen.append(s.entry_zone_lo))
        self.assertEqual((out["n_evaluated"], out["n_duplicates"], seen), (3, 1, [0.5, 0.6]))


# -- F41: an entry at t sees only exits before t -------------------------------------------------

PCFG = {"starting_equity": 10000.0, "risk_per_trade": 0.01, "max_open_positions": 6,
        "max_total_risk": 0.5, "max_notional_frac": 0.5, "daily_loss_limit": 0.02,
        "max_total_notional": 10.0}


def mk(entry_t, exit_t, ret_pct, risk_pct=0.02, symbol="A"):
    return Trade("s1", symbol, "1h", 1, f"{symbol}{entry_t}", entry_t, entry_t, 100.0, 98.0, exit_t,
                 100.0, "open" if exit_t is None else "target", ret_pct, risk_pct, ret_pct / risk_pct, 1)


class TestCausalSameBarOrder(unittest.TestCase):
    def test_a_loss_inside_the_entry_bar_does_not_stop_that_entry(self):
        t = T0 + 3600
        loss = mk(T0, t, -0.04, symbol="A")                        # -200 = -2% exits during bar t
        same_bar = mk(t, t + 600, 0.0, symbol="B")
        next_bar = mk(t + 3600, t + 4000, 0.0, symbol="C")         # same UTC day, one bar later
        res = simulate_portfolio([loss, same_bar, next_bar], PCFG)
        self.assertEqual([r["trade"].symbol for r in res.taken], ["A", "B"])
        self.assertEqual([(s["trade"].symbol, s["reason"]) for s in res.skipped],
                         [("C", "daily_loss_limit")])

    def test_entries_do_not_depend_on_the_outcome_of_same_bar_exits(self):
        rng = random.Random(3)
        for _ in range(100):
            trades = []
            for k in range(rng.randrange(4, 14)):
                e = T0 + rng.randrange(0, 12) * 3600
                x = None if rng.random() < 0.1 else e + rng.randrange(0, 6) * 3600
                trades.append(mk(e, x, rng.uniform(-0.06, 0.06), rng.uniform(0.005, 0.05), f"S{k}"))
            pc = {**PCFG, "max_open_positions": rng.randrange(1, 5), "max_total_risk": 0.03}
            t = rng.choice([tr.entry_t for tr in trades])
            moved = [mk(tr.entry_t, tr.exit_t, -0.5 if tr.exit_t == t and tr.exit_t > tr.entry_t
                        else tr.ret_pct, tr.risk_pct, tr.symbol) for tr in trades]

            def decisions(ts):
                res = simulate_portfolio(ts, pc)
                taken = {r["trade"].symbol: round(r["notional"], 9) for r in res.taken
                         if r["trade"].entry_t <= t}
                return taken, sorted((s["trade"].symbol, s["reason"]) for s in res.skipped
                                     if s["trade"].entry_t <= t)

            self.assertEqual(decisions(trades), decisions(moved))


# -- F43: no leverage ----------------------------------------------------------------------------

class TestNoLeverage(unittest.TestCase):
    def overlapping(self, n=6):
        return [mk(T0 + i, T0 + DAY, 0.10, 0.02, f"S{i}") for i in range(n)]

    def test_open_notional_never_exceeds_equity_by_default(self):
        res = simulate_portfolio(self.overlapping(), DEFAULTS["portfolio"])
        self.assertEqual([r["notional"] for r in res.taken], [2500.0] * 4)
        self.assertEqual([(s["trade"].symbol, s["reason"]) for s in res.skipped],
                         [("S4", "max_notional"), ("S5", "max_notional")])
        self.assertAlmostEqual(res.final_equity, 11000.0)

    def test_the_cap_is_configurable(self):
        res = simulate_portfolio(self.overlapping(), {**DEFAULTS["portfolio"], "max_total_notional": 1.5})
        self.assertEqual((len(res.taken), res.skipped), (6, []))
        self.assertAlmostEqual(res.final_equity, 11500.0)

    def test_an_entry_is_shrunk_to_the_room_left(self):
        trades = self.overlapping(3) + [mk(T0 + 10, T0 + DAY, 0.10, 0.004, "BIG")]
        res = simulate_portfolio(trades, {**DEFAULTS["portfolio"], "max_notional_frac": 0.4})
        # 3 x 2500 open; BIG wants min(0.005 * 10000 / 0.004, 4000) = 4000 but only 2500 is left
        self.assertAlmostEqual(res.taken[-1]["notional"], 2500.0)
        self.assertAlmostEqual(sum(r["notional"] for r in res.taken), 10000.0)


# -- F30: per-side rows for direction "both" -----------------------------------------------------

class TestBySide(unittest.TestCase):
    def test_both_sides_are_split_and_add_up(self):
        bars, cfg = make_universe(3), cfg_()
        res = backtest(SPEC.with_overrides(direction="both"), bars, T0 + 400 * H4, T0 + 1400 * H4, cfg)
        m, side = res.metrics, res.by_side
        self.assertGreater(min(m["long_n"], m["short_n"]), 0)
        self.assertEqual((side["long"]["n"], side["short"]["n"]), (m["long_n"], m["short_n"]))
        self.assertAlmostEqual(side["long"]["total_r"] + side["short"]["total_r"], m["total_r"])
        s = res.summary()
        self.assertEqual(s["by_side"]["long"]["n"], m["long_n"])
        json.dumps(s, allow_nan=False)
        long_only = backtest(SPEC, bars, T0 + 400 * H4, T0 + 1400 * H4, cfg)
        self.assertIsNone(long_only.by_side)
        self.assertNotIn("by_side", long_only.summary())


# -- F40: a baseline drawn only where the strategy's filter lets it trade -----------------------

def up_down(sym, n=2400, seed=0):
    up = make_bars(sym, n=n // 2, seed=seed)
    rows = list(zip(up.t, up.o, up.h, up.l, up.c, up.v))
    rnd, px = random.Random(seed + 99), up.c[-1]
    for i in range(n // 2, n):
        o = px
        px *= math.exp(-0.004 + rnd.gauss(0, 0.01))
        rows.append((T0 + i * H4, o, max(o, px) * 1.002, min(o, px) * 0.998, px, 1.0))
    return Bars.from_rows(sym, "4h", rows)


class TestRegimeBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = {f"U{i}USDT": up_down(f"U{i}USDT", seed=i) for i in range(2)}
        cls.cfg = cfg_()
        cls.spec = SPEC.with_overrides(trend_filter="ema", trend_ema=50)
        cls.start, cls.end = T0 + 300 * H4, T0 + 2400 * H4
        cls.res = backtest(cls.spec, cls.bars, cls.start, cls.end, cls.cfg)

    def test_draws_only_where_the_filter_passed_and_the_edge_shrinks(self):
        calls = []

        def spy(*args, **kw):
            calls.append(args)
            return manage_position(*args, **kw)

        with mock.patch.object(rs, "manage_position", side_effect=spy):
            b = rs.random_baseline(self.spec, self.bars, self.cfg, self.start, self.end, n_runs=6,
                                   seed=2, result=self.res)
        n = len([t for t in self.res.closed_trades if t.risk_pct > 0])
        self.assertEqual(len(calls), 2 * 6 * n)
        for w, _, e, side, *_ in calls[6 * n:]:
            self.assertTrue(_filter_ok(self.spec, side, w.c, ema(w.c, 50), None, None)[e - 1])
        self.assertTrue(b["regime_filtered"])
        self.assertLess(b["edge_regime_r"], b["edge_r"])           # the up-trend half is easier
        again = rs.random_baseline(self.spec, self.bars, self.cfg, self.start, self.end, n_runs=6,
                                   seed=2, result=self.res)
        self.assertEqual(again, b)
        off = rs.random_baseline(self.spec, self.bars, self.cfg, self.start, self.end, n_runs=6,
                                 seed=2, result=self.res, regime=False)
        self.assertEqual({k: off[k] for k in ("runs", "edge_r")}, {k: b[k] for k in ("runs", "edge_r")})
        self.assertFalse(off["regime_filtered"])

    def test_an_unfiltered_spec_repeats_the_uniform_numbers(self):
        spec = self.spec.with_overrides(trend_filter="none")
        b = rs.random_baseline(spec, self.bars, self.cfg, self.start, self.end, n_runs=4, seed=2)
        self.assertFalse(b["regime_filtered"])
        self.assertEqual((b["baseline_regime_avg_r_mean"], b["edge_regime_r"]),
                         (b["baseline_avg_r_mean"], b["edge_r"]))


if __name__ == "__main__":
    unittest.main()
