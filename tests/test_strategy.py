"""Tests for fibbot.strategy: spec, parameter space, presets, hand-built engine scenarios,
the shared exit logic (manage_position) and causality / symmetry properties on random walks."""
import math
import random
import time
import unittest

from fibbot.fib import retrace_price, zigzag
from fibbot.indicators import atr, ema
from fibbot.strategy import (PARAM_SPACE, PRESETS, FibEngine, RunResult, StrategySpec,
                             manage_position, warmup_bars)
from fibbot.types import TF_SECONDS, Bars, Costs

T0, STEP = 1_700_006_400, 14400
FEE, SLIP = 0.001, 0.0005
COSTS = Costs(FEE, SLIP)


def t_(i):
    return T0 + i * STEP


def mk(rows, tf="4h", symbol="TESTUSDT"):
    """Bars from (o, h, l, c) rows, one 4h bar apart."""
    return Bars.from_rows(symbol, tf, [(T0 + i * STEP, *r, 1.0) for i, r in enumerate(rows)])


def mirror(rows, k=200.0):
    """Reflect (o, h, l, c) rows around k/2: highs become lows."""
    return [(k - o, k - l, k - h, k - c) for o, h, l, c in rows]


# Canonical long scenario, 5% pct pivots: H@0 (100.5, confirmed bar 2), anchor L@4 (90, confirmed
# bar 5), the leg runs to the extreme 110 at bar 8, H@8 confirmed at bar 9. With zone 0.618-0.65
# (limit at the mid ratio 0.634), stop 0.886 (no ATR buffer), tp1 1.0, tp2 1.618:
HEAD = [(100, 100.5, 99, 99.5), (99.5, 99.8, 96, 96.5), (96.5, 97, 93, 93.5), (93.5, 94, 91, 91.5),
        (91.5, 92, 90, 91), (91, 95, 91, 94.5), (94.5, 101, 94, 100.5), (100.5, 106, 100, 105.5),
        (105.5, 110, 105, 108), (108, 108.5, 99, 99.5)]
ANCHOR, EXTREME = 90.0, 110.0
ENTRY = EXTREME - 0.634 * 20          # 97.32
STOP = EXTREME - 0.886 * 20           # 92.28
TP1, TP2 = EXTREME, ANCHOR + 1.618 * 20   # 110, 122.36
BE = ENTRY * (1 + 2 * FEE)
FILL = (99.5, 100, 97, 97.8)          # bar 10 dips to 97 <= ENTRY: limit fills at ENTRY
UP11 = (97.8, 104, 97.5, 103.5)
TP1_BAR = (103.5, 111, 103, 110.5)    # h 111 >= TP1
CLEAN = HEAD + [FILL, UP11, TP1_BAR, (110.5, 116, 110, 115.5), (115.5, 123, 115, 122)]
BASE = dict(swing_mode="pct", swing_pct=0.05, swing_atr_len=5, min_leg_atr=1.0,
            trend_filter="none", stop_atr_buffer=0.0)
LEG = f"L{t_(4)}"


def spec(**kw):
    return StrategySpec(**{**BASE, **kw})


def run(rows, sp=None, costs=COSTS, **kw):
    return FibEngine(sp or spec(), costs).run(mk(rows), **kw)


def long_ret(fills, entry, fee=FEE):
    return sum(f * p for f, p in fills) / entry * (1 - fee) - (1 + fee)


def short_ret(fills, entry, fee=FEE):
    return (1 - fee) - sum(f * p for f, p in fills) / entry * (1 + fee)


def random_walk(n, seed, tf="4h", start=100.0):
    """Regime-drifting random walk with volatility clustering and occasional open gaps."""
    rnd = random.Random(seed)
    rows, px, vol, drift = [], start, 0.015, 0.0
    for i in range(n):
        if i % 150 == 0:
            drift = rnd.gauss(0, 0.003)
        vol = min(0.05, max(0.004, vol * math.exp(rnd.gauss(0, 0.1))))
        o = px * math.exp(rnd.gauss(0, 2 * vol)) if rnd.random() < 0.03 else px
        c = o * math.exp(drift + rnd.gauss(0, vol))
        h = max(o, c) * (1 + abs(rnd.gauss(0, vol / 2)))
        l = min(o, c) * (1 - abs(rnd.gauss(0, vol / 2)))
        rows.append((T0 + i * TF_SECONDS[tf], o, h, l, c, 1.0))
        px = c
    return Bars.from_rows("RWUSDT", tf, rows)


class FillsMixin:
    def assertFills(self, actual, expected):
        """expected: [(frac, price, label, bar_index)]."""
        self.assertEqual(len(actual), len(expected), actual)
        for (f, p, lab, t), (ef, ep, elab, ei) in zip(actual, expected):
            self.assertAlmostEqual(f, ef, places=12)
            self.assertAlmostEqual(p, ep, places=9)
            self.assertEqual((lab, t), (elab, t_(ei)))

    def one_trade(self, r):
        self.assertEqual(len(r.trades), 1, r.trades)
        return r.trades[0]


class TestStrategySpec(unittest.TestCase):
    def test_defaults_valid_and_validate_returns_self(self):
        s = StrategySpec()
        self.assertIs(s.validate(), s)
        self.assertEqual(s.name, "custom")

    def test_validation_errors_name_the_field(self):
        cases = [
            ("timeframe", {"timeframe": "2h"}), ("direction", {"direction": "up"}),
            ("swing_mode", {"swing_mode": "fixed"}), ("swing_atr_len", {"swing_atr_len": 4}),
            ("swing_atr_len", {"swing_atr_len": 51}), ("swing_atr_len", {"swing_atr_len": 14.5}),
            ("swing_atr_len", {"swing_atr_len": True}), ("swing_mult", {"swing_mult": 0.5}),
            ("swing_mult", {"swing_mult": float("nan")}), ("swing_mult", {"swing_mult": "3"}),
            ("swing_pct", {"swing_pct": 0.5}), ("min_leg_atr", {"min_leg_atr": -1}),
            ("setup_expiry_bars", {"setup_expiry_bars": 2}), ("trend_filter", {"trend_filter": "sma"}),
            ("trend_ema", {"trend_ema": 10}), ("adx_min", {"adx_min": 60}),
            ("require_structure", {"require_structure": 1}), ("entry_zone_lo", {"entry_zone_lo": 0.2}),
            ("entry_zone_hi", {"entry_zone_hi": 0.9}),
            ("entry_zone_hi", {"entry_zone_lo": 0.65, "entry_zone_hi": 0.618}),
            ("entry_mode", {"entry_mode": "market"}), ("entry_at", {"entry_at": "top"}),
            ("confirm_type", {"confirm_type": "hammer"}), ("confirm_window", {"confirm_window": 0}),
            ("stop_ratio", {"stop_ratio": 0.65}), ("stop_ratio", {"stop_ratio": 1.3}),
            ("stop_atr_buffer", {"stop_atr_buffer": 3.5}), ("tp1_ext", {"tp1_ext": 0.2}),
            ("tp1_frac", {"tp1_frac": 0.0}), ("tp1_frac", {"tp1_frac": 1.5}),
            ("tp2_ext", {"tp2_ext": 1.0}), ("tp2_ext", {"tp2_ext": 5.0}),
            ("move_stop_to_be", {"move_stop_to_be": "yes"}), ("trail_atr", {"trail_atr": -0.5}),
            ("max_hold_bars", {"max_hold_bars": 0}), ("min_rr", {"min_rr": 11}), ("name", {"name": ""}),
        ]
        for field, kw in cases:
            with self.subTest(kw=kw), self.assertRaises(ValueError) as cm:
                StrategySpec(**kw).validate()
            self.assertTrue(str(cm.exception).startswith(field + ":"), str(cm.exception))

    def test_boundaries_and_tp2_ignored_when_tp1_closes_all(self):
        for kw in ({"swing_atr_len": 5}, {"swing_atr_len": 50}, {"stop_ratio": 1.2},
                   {"tp2_ext": 4.236}, {"entry_zone_lo": 0.65, "entry_zone_hi": 0.65},
                   {"tp1_frac": 1.0, "tp2_ext": 0.5}, {"min_leg_atr": 0}, {"trail_atr": 10}):
            with self.subTest(kw=kw):
                StrategySpec(**kw).validate()

    def test_numeric_normalization(self):
        s = StrategySpec(swing_mult=3, swing_atr_len=14.0, entry_zone_lo=0.618 + 0.05, entry_zone_hi=0.7)
        self.assertIsInstance(s.swing_mult, float)
        self.assertIsInstance(s.swing_atr_len, int)
        self.assertEqual(s.entry_zone_lo, 0.668)
        self.assertEqual(StrategySpec(swing_mult=3).spec_id(), StrategySpec().spec_id())
        self.assertEqual(s.spec_id(), StrategySpec(entry_zone_lo=0.668, entry_zone_hi=0.7).spec_id())

    def test_spec_id(self):
        s = StrategySpec()
        self.assertEqual(s.spec_id(), "8450eeebe8")       # pinned: stored ids must stay stable
        self.assertRegex(s.spec_id(), r"^[0-9a-f]{10}$")
        self.assertEqual(s.with_overrides(name="other").spec_id(), s.spec_id())
        self.assertNotEqual(s.with_overrides(min_rr=2.0).spec_id(), s.spec_id())
        d = s.to_dict()
        shuffled = dict(reversed(list(d.items())))
        self.assertEqual(StrategySpec.from_dict(shuffled).spec_id(), s.spec_id())

    def test_dict_round_trip_and_errors(self):
        s = PRESETS["ote_confirm_1h"]
        d = s.to_dict()
        self.assertEqual(set(d), set(PARAM_SPACE) | {"name"})
        self.assertEqual(StrategySpec.from_dict(d), s)
        self.assertEqual(StrategySpec.from_dict({"min_rr": 2.0}), StrategySpec(min_rr=2.0))
        with self.assertRaises(ValueError) as cm:
            StrategySpec.from_dict({"min_rr": 2.0, "bogus": 1, "spec_id": "x"})
        self.assertIn("bogus", str(cm.exception))
        self.assertIn("spec_id", str(cm.exception))
        with self.assertRaises(ValueError):
            StrategySpec.from_dict([("min_rr", 2.0)])

    def test_with_overrides(self):
        s = StrategySpec()
        s2 = s.with_overrides(direction="both", swing_mult=4)
        self.assertEqual((s2.direction, s2.swing_mult), ("both", 4.0))
        self.assertEqual(s.direction, "long")
        with self.assertRaises(ValueError):
            s.with_overrides(nope=1)
        bad = s.with_overrides(swing_atr_len=1)       # not validated until asked
        with self.assertRaises(ValueError):
            bad.validate()

    def test_warmup_bars(self):
        self.assertEqual(warmup_bars(StrategySpec()), 600)
        self.assertEqual(warmup_bars(StrategySpec(trend_ema=20, swing_atr_len=50)), 300)
        self.assertEqual(warmup_bars(StrategySpec(trend_ema=150)), 450)


class TestParamSpaceAndPresets(unittest.TestCase):
    def test_param_space_covers_every_field(self):
        fields_ = set(StrategySpec().to_dict()) - {"name"}
        self.assertEqual(set(PARAM_SPACE), fields_)
        defaults = StrategySpec().to_dict()
        for k, p in PARAM_SPACE.items():
            with self.subTest(k=k):
                self.assertEqual(p["default"], defaults[k])
                self.assertTrue(isinstance(p["doc"], str) and p["doc"])
                if p["type"] in ("int", "float"):
                    self.assertLessEqual(p["min"], p["default"])
                    self.assertLessEqual(p["default"], p["max"])
                    self.assertGreater(p["step"], 0)
                elif p["type"] == "choice":
                    self.assertIn(p["default"], p["choices"])
                else:
                    self.assertEqual(p["type"], "bool")
                    self.assertIsInstance(p["default"], bool)

    def test_presets_listed_valid_and_distinct(self):
        names = {"golden_pocket_4h", "golden_pocket_1h", "golden_pocket_both_4h", "ote_confirm_1h",
                 "shallow_trend_4h", "deep_786_4h", "golden_pocket_1d", "ext_runner_4h"}
        self.assertLessEqual(names, set(PRESETS))
        for k, s in PRESETS.items():
            self.assertEqual(s.name, k)
            s.validate()
        self.assertEqual(len({s.spec_id() for s in PRESETS.values()}), len(PRESETS))

    def test_preset_parameters_follow_the_table(self):
        gp = PRESETS["golden_pocket_4h"]
        self.assertEqual(
            (gp.timeframe, gp.direction, gp.trend_filter, gp.trend_ema, gp.entry_zone_lo,
             gp.entry_zone_hi, gp.entry_mode, gp.entry_at, gp.stop_ratio, gp.stop_atr_buffer,
             gp.tp1_ext, gp.tp1_frac, gp.tp2_ext),
            ("4h", "long", "ema", 200, 0.618, 0.65, "limit", "mid", 0.886, 0.25, 1.0, 0.5, 1.618))
        self.assertEqual(PRESETS["golden_pocket_1h"], gp.with_overrides(name="golden_pocket_1h", timeframe="1h"))
        self.assertEqual(PRESETS["golden_pocket_both_4h"],
                         gp.with_overrides(name="golden_pocket_both_4h", direction="both"))
        o = PRESETS["ote_confirm_1h"]
        self.assertEqual((o.timeframe, o.entry_zone_lo, o.entry_zone_hi, o.entry_mode, o.confirm_type,
                          o.stop_ratio, o.stop_atr_buffer, o.tp1_ext, o.tp2_ext),
                         ("1h", 0.618, 0.786, "confirm", "reclaim", 1.0, 0.1, 1.0, 1.618))
        s = PRESETS["shallow_trend_4h"]
        self.assertEqual((s.trend_filter, s.adx_min, s.entry_zone_lo, s.entry_zone_hi, s.entry_mode,
                          s.entry_at, s.stop_ratio, s.tp1_ext, s.tp2_ext),
                         ("ema_stack", 20.0, 0.382, 0.5, "limit", "lo", 0.786, 1.0, 1.618))
        d = PRESETS["deep_786_4h"]
        self.assertEqual((d.trend_filter, d.entry_zone_lo, d.entry_zone_hi, d.entry_at, d.stop_ratio,
                          d.stop_atr_buffer, d.tp1_ext, d.tp2_ext),
                         ("none", 0.786, 0.886, "lo", 1.0, 0.3, 0.618, 1.0))
        g = PRESETS["golden_pocket_1d"]
        self.assertEqual((g.timeframe, g.entry_zone_lo, g.entry_zone_hi, g.entry_at, g.stop_ratio,
                          g.tp1_ext, g.tp2_ext), ("1d", 0.5, 0.65, "mid", 0.886, 1.0, 1.618))
        e = PRESETS["ext_runner_4h"]
        self.assertEqual((e.entry_zone_lo, e.entry_zone_hi, e.tp1_ext, e.tp1_frac, e.trail_atr, e.tp2_ext),
                         (0.618, 0.65, 1.0, 0.3, 2.5, 2.618))

    def test_every_preset_runs_on_its_timeframe(self):
        for name, s in PRESETS.items():
            with self.subTest(name=name):
                bars = random_walk(2500, 11, tf=s.timeframe)
                r = FibEngine(s, COSTS).run(bars)
                self.assertIsInstance(r, RunResult)
                self.assertEqual(r.n_bars, 2500)
                self.assertGreater(len(r.trades), 0)


class TestEngineBasics(unittest.TestCase):
    def test_strategy_id_and_invalid_inputs(self):
        s = spec()
        self.assertEqual(FibEngine(s).strategy_id, s.spec_id())
        self.assertEqual(FibEngine(s, strategy_id="abc").strategy_id, "abc")
        self.assertEqual(FibEngine(s).costs, Costs())
        with self.assertRaises(ValueError):
            FibEngine(StrategySpec(swing_atr_len=1))
        with self.assertRaises(ValueError):
            FibEngine(s).run(mk(CLEAN, tf="1h"))
        self.assertEqual(FibEngine(s).run(Bars("X", "4h")), RunResult([], None, [], 0))

    def test_scenario_pivots(self):
        b = mk(CLEAN)
        piv = zigzag(b, None, mode="pct", pct=0.05)
        self.assertEqual([(p.kind, p.idx, p.price, p.confirm_idx) for p in piv[:3]],
                         [("H", 0, 100.5, 2), ("L", 4, ANCHOR, 5), ("H", 8, EXTREME, 9)])

    def test_deterministic(self):
        bars = random_walk(1500, 5)
        eng = FibEngine(PRESETS["golden_pocket_both_4h"].with_overrides(trend_ema=50), COSTS)
        self.assertEqual(eng.run(bars), eng.run(bars))


class TestLongScenarios(FillsMixin, unittest.TestCase):
    def test_clean_long_limit_tp1_then_tp2(self):
        r = run(CLEAN)
        tr = self.one_trade(r)
        self.assertIsNone(r.open_trade)
        self.assertEqual((tr.side, tr.leg_id, tr.setup_t, tr.entry_t, tr.exit_t, tr.exit_reason, tr.bars_held),
                         (1, LEG, t_(5), t_(10), t_(14), "target", 4))
        self.assertEqual((tr.symbol, tr.tf, tr.strategy_id), ("TESTUSDT", "4h", spec().spec_id()))
        self.assertAlmostEqual(tr.entry_price, ENTRY, places=9)
        self.assertAlmostEqual(tr.stop_price, STOP, places=9)
        self.assertFills(tr.meta["fills"], [(0.5, TP1, "tp1", 12), (0.5, TP2, "tp2", 14)])
        ret = long_ret([(0.5, TP1), (0.5, TP2)], ENTRY)
        risk = (ENTRY - STOP) / ENTRY
        self.assertAlmostEqual(tr.ret_pct, ret, places=12)
        self.assertAlmostEqual(tr.risk_pct, risk, places=12)
        self.assertAlmostEqual(tr.r_multiple, ret / risk, places=10)
        self.assertAlmostEqual(tr.exit_price, 0.5 * TP1 + 0.5 * TP2, places=9)
        m = tr.meta
        self.assertEqual(set(m), {"anchor", "extreme", "leg_size", "atr", "entry_ratio", "tp1", "tp2",
                                  "rr", "fills", "entry_mode"})
        self.assertEqual((m["anchor"], m["extreme"], m["entry_mode"]), (ANCHOR, EXTREME, "limit"))
        self.assertAlmostEqual(m["leg_size"], 20.0)
        self.assertAlmostEqual(m["entry_ratio"], 0.634)
        self.assertAlmostEqual(m["rr"], (0.5 * (TP1 - ENTRY) + 0.5 * (TP2 - ENTRY)) / (ENTRY - STOP))
        self.assertAlmostEqual(m["atr"], atr(*[mk(CLEAN).h, mk(CLEAN).l, mk(CLEAN).c], 5)[9])
        # one signal: the leg first qualified at bar 5 (leg 90 -> 95)
        self.assertEqual(len(r.signals), 1)
        sg = r.signals[0]
        self.assertEqual((sg.side, sg.leg_id, sg.t, sg.symbol, sg.tf), (1, LEG, t_(5), "TESTUSDT", "4h"))
        self.assertAlmostEqual(sg.entry_price, 95 - 0.634 * 5)
        self.assertAlmostEqual(sg.stop_price, 95 - 0.886 * 5)
        self.assertAlmostEqual(sg.tp1, 95.0)
        self.assertAlmostEqual(sg.tp2, 90 + 1.618 * 5)
        self.assertEqual((sg.meta["anchor"], sg.meta["extreme"], sg.meta["mode"]), (ANCHOR, 95.0, "limit"))
        self.assertTrue({"atr", "rr"} <= set(sg.meta))

    def test_stop_out_with_slippage(self):
        tr = self.one_trade(run(HEAD + [FILL, (97.8, 98, 92, 92.5)]))
        self.assertEqual((tr.exit_reason, tr.exit_t, tr.bars_held), ("stop", t_(11), 1))
        px = STOP * (1 - SLIP)
        self.assertFills(tr.meta["fills"], [(1.0, px, "stop", 11)])
        self.assertAlmostEqual(tr.ret_pct, long_ret([(1.0, px)], ENTRY), places=12)
        self.assertLess(tr.r_multiple, -1.0)

    def test_gap_through_stop_fills_at_open(self):
        tr = self.one_trade(run(HEAD + [FILL, (91, 91.5, 90.5, 91.2)]))
        self.assertEqual(tr.exit_reason, "stop")
        self.assertFills(tr.meta["fills"], [(1.0, 91 * (1 - SLIP), "stop", 11)])

    def test_same_bar_stop_and_target_takes_the_stop(self):
        tr = self.one_trade(run(HEAD + [FILL, (97.8, 111, 92, 105)]))
        self.assertEqual(tr.exit_reason, "stop")
        self.assertFills(tr.meta["fills"], [(1.0, STOP * (1 - SLIP), "stop", 11)])

    def test_entry_bar_stop(self):
        tr = self.one_trade(run(HEAD + [(99.5, 100, 92, 93), UP11]))
        self.assertEqual((tr.entry_t, tr.exit_t, tr.bars_held, tr.exit_reason), (t_(10), t_(10), 0, "stop"))
        self.assertAlmostEqual(tr.entry_price, ENTRY, places=9)
        self.assertFills(tr.meta["fills"], [(1.0, STOP * (1 - SLIP), "stop", 10)])

    def test_entry_bar_targets_never_credited(self):
        # the entry bar reaches tp1 too, but targets only count from the next bar
        r = run(HEAD + [(99.5, 111, 97, 104)])
        self.assertEqual(r.trades, [])
        self.assertEqual(r.open_trade.meta["fills"][0][2], "open")

    def test_entry_bar_open_beyond_stop_exits_at_fill(self):
        tr = self.one_trade(run(HEAD + [(92, 93, 91.5, 92.5)]))
        self.assertEqual(tr.entry_price, 92.0)                  # limit gapped through: fill at open
        self.assertEqual((tr.exit_reason, tr.bars_held), ("stop", 0))
        self.assertFills(tr.meta["fills"], [(1.0, 92 * (1 - SLIP), "stop", 10)])
        self.assertAlmostEqual(tr.risk_pct, (STOP - 92) / 92, places=12)

    def test_breakeven_after_tp1_from_next_bar(self):
        # bar 12 fills tp1 and trades below the breakeven level (but above the stop): no exit yet
        rows = HEAD + [FILL, UP11, (103.5, 111, 97.4, 110.5), (110, 110.5, 97, 98)]
        tr = self.one_trade(run(rows))
        self.assertEqual((tr.exit_reason, tr.exit_t, tr.bars_held), ("breakeven", t_(13), 3))
        self.assertFills(tr.meta["fills"], [(0.5, TP1, "tp1", 12), (0.5, BE * (1 - SLIP), "breakeven", 13)])
        self.assertAlmostEqual(tr.ret_pct, long_ret([(0.5, TP1), (0.5, BE * (1 - SLIP))], ENTRY), places=12)
        # without breakeven the same bars keep the position open
        r = run(rows, spec(move_stop_to_be=False))
        self.assertEqual(r.trades, [])
        self.assertEqual(r.open_trade.exit_reason, "open")

    def test_time_stop(self):
        rows = HEAD + [FILL, UP11, (103.5, 108, 103, 107), (107, 109, 106, 108), (108, 109, 107, 108.5)]
        tr = self.one_trade(run(rows, spec(max_hold_bars=3)))
        self.assertEqual((tr.exit_reason, tr.exit_t, tr.bars_held), ("time", t_(13), 3))
        self.assertFills(tr.meta["fills"], [(1.0, 108 * (1 - SLIP), "time", 13)])

    def test_trailing_stop(self):
        rows = HEAD + [FILL, UP11, TP1_BAR, (110.5, 112, 104, 105)]
        b = mk(rows)
        trail = 110.5 - 1.0 * atr(b.h, b.l, b.c, 5)[12]
        self.assertGreater(trail, BE)
        self.assertGreater(104, STOP)
        tr = self.one_trade(run(rows, spec(trail_atr=1.0)))
        self.assertEqual((tr.exit_reason, tr.exit_t), ("trail", t_(13)))
        self.assertFills(tr.meta["fills"], [(0.5, TP1, "tp1", 12), (0.5, trail * (1 - SLIP), "trail", 13)])
        # before tp1 the stop does not trail: a dip below close - ATR keeps the position
        rows = HEAD + [FILL, UP11, (103.5, 105, 98, 99)]
        b = mk(rows)
        self.assertGreater(103.5 - atr(b.h, b.l, b.c, 5)[11], 98)
        r = run(rows, spec(trail_atr=1.0))
        self.assertEqual((r.trades, r.open_trade.exit_reason), ([], "open"))

    def test_tp1_full_close_and_gap_above_target(self):
        tr = self.one_trade(run(HEAD + [FILL, UP11, TP1_BAR], spec(tp1_frac=1.0)))
        self.assertEqual(tr.exit_reason, "target")
        self.assertFills(tr.meta["fills"], [(1.0, TP1, "tp1", 12)])
        # opening above tp1 fills it at the (better) open
        r = run(HEAD + [FILL, UP11, (112, 113, 111.5, 112.5)])
        self.assertFills(r.open_trade.meta["fills"],
                         [(0.5, 112, "tp1", 12), (0.5, 112.5 * (1 - SLIP), "open", 12)])
        tr = self.one_trade(run(HEAD + [FILL, UP11, TP1_BAR, (124, 125, 123.5, 124.5)]))
        self.assertFills(tr.meta["fills"], [(0.5, TP1, "tp1", 12), (0.5, 124, "tp2", 13)])
        tr = self.one_trade(run(HEAD + [FILL, UP11, (103.5, 123, 103, 122)]))
        self.assertFills(tr.meta["fills"], [(0.5, TP1, "tp1", 12), (0.5, TP2, "tp2", 12)])

    def test_open_trade_marked_at_last_close(self):
        r = run(HEAD + [FILL, UP11])
        self.assertEqual(r.trades, [])
        tr = r.open_trade
        self.assertEqual((tr.exit_t, tr.exit_reason, tr.bars_held, tr.entry_t), (None, "open", 1, t_(10)))
        px = 103.5 * (1 - SLIP)
        self.assertAlmostEqual(tr.exit_price, px, places=9)
        self.assertFills(tr.meta["fills"], [(1.0, px, "open", 11)])
        self.assertAlmostEqual(tr.ret_pct, long_ret([(1.0, px)], ENTRY), places=12)
        self.assertTrue(tr.is_open)

    def test_limit_only_placed_when_close_is_beyond_entry(self):
        # bar 8 sets the 110 extreme but closes at 97, below the new entry (its own low stays above
        # the previous bar's lower limit), so no limit rests for bar 9 although bar 9 trades through
        # the entry; bar 9 closes above it and bar 10 fills
        rows = HEAD[:8] + [(105.5, 110, 96.5, 97.0), (97.0, 98, 96.8, 97.9), (97.9, 99, 97.0, 98.5)]
        r = run(rows)
        self.assertEqual((r.trades, r.open_trade.entry_t), ([], t_(10)))
        self.assertAlmostEqual(r.open_trade.entry_price, ENTRY, places=9)

    def test_leg_dies_when_low_reaches_stop_while_unfilled(self):
        sp = spec(stop_ratio=0.7)
        stop = retrace_price(ANCHOR, EXTREME, 0.7)
        for low9, alive in ((stop, False), (stop + 0.1, True)):
            with self.subTest(low=low9):
                # bar 8 closes below the entry (no order for bar 9); bar 9's low reaches the stop
                rows = (HEAD[:5] + [(91.6, 95, 91.6, 94.5)] + HEAD[6:8]
                        + [(105.5, 110, 96.5, 97.0), (97.0, 97.2, low9, 96.5), (96.5, 99, 96.4, 98.5),
                           (98.5, 99, 97.0, 98)])
                r = run(rows, sp)
                self.assertEqual(r.trades, [])
                self.assertEqual(r.open_trade is not None, alive)
                if alive:
                    self.assertEqual(r.open_trade.entry_t, t_(11))

    def test_leg_dies_when_price_trades_below_the_anchor(self):
        # stop 1.2 sits below the anchor; bar 6 dips under the anchor (90) without reaching it
        sp = spec(stop_ratio=1.2)
        for low6, alive in ((89.8, False), (90.2, True)):
            with self.subTest(low=low6):
                rows = HEAD[:5] + [(91, 95, 91, 91.5), (91.5, 95.5, low6, 94), (94, 106, 93.5, 105.5),
                                   (105.5, 110, 105, 108), (108, 108.5, 99, 99.5), FILL]
                r = run(rows, sp)
                self.assertEqual(r.trades, [])
                self.assertEqual(r.open_trade is not None, alive)
                if alive:
                    self.assertEqual((r.open_trade.entry_t, r.open_trade.leg_id), (t_(10), LEG))

    def test_expiry_counts_from_first_occurrence_of_the_extreme(self):
        # bar 9 retests the 110 high exactly; expiry still counts from bar 8
        rows = HEAD[:9] + [(108, 110, 106, 107), (107, 108, 105.5, 106), (106, 107, 105, 106),
                           (106, 106.5, 105, 105.5), (105.5, 106, 96, 97)]
        r = run(rows, spec(setup_expiry_bars=3))
        self.assertEqual((r.trades, r.open_trade), ([], None))
        self.assertEqual(run(rows, spec(setup_expiry_bars=4)).open_trade.entry_t, t_(13))

    def test_setup_expiry(self):
        # after the extreme (bar 8) price drifts above the zone; bar 13 finally dips into it
        rows = HEAD + [(99.5, 101, 99, 100), (100, 101.5, 99.2, 101), (101, 102, 99.5, 100.5),
                       (100.5, 101, 96, 97)]
        r = run(rows, spec(setup_expiry_bars=3))           # dead at bar 12 (12 - 8 > 3)
        self.assertEqual((r.trades, r.open_trade), ([], None))
        r = run(rows, spec(setup_expiry_bars=4))
        self.assertEqual((r.open_trade.leg_id, r.open_trade.entry_t), (LEG, t_(13)))
        self.assertAlmostEqual(r.open_trade.entry_price, ENTRY, places=9)

    def test_reward_risk_filter(self):
        rr = (0.5 * 0.634 + 0.5 * (1.618 - 0.366)) / (0.886 - 0.634)      # 3.742, scale free here
        r = run(CLEAN, spec(min_rr=round(rr, 1) + 0.1))
        self.assertEqual((r.trades, r.signals, r.open_trade), ([], [], None))
        self.assertEqual(len(run(CLEAN, spec(min_rr=round(rr, 1) - 0.1)).trades), 1)

    def test_min_leg_atr_filter(self):
        r = run(CLEAN, spec(min_leg_atr=30.0))
        self.assertEqual((r.trades, r.signals), ([], []))

    def test_trend_filter_blocks_and_passes(self):
        down = [(300 - 6.6 * k, 300 - 6.6 * k + 0.3, 300 - 6.6 * (k + 1) - 0.3, 300 - 6.6 * (k + 1))
                for k in range(30)]
        up = [(50 + 1.7 * k, 50 + 1.7 * (k + 1) + 0.3, 50 + 1.7 * k - 0.3, 50 + 1.7 * (k + 1))
              for k in range(30)]
        ext_idx = 30 + 8
        for prefix, above in ((down, True), (up, False)):
            rows = prefix + CLEAN
            b = mk(rows)
            self.assertEqual(ema(b.c, 20)[ext_idx] > b.c[ext_idx], above)
            self.assertEqual(len(run(rows).trades), 1)
            trades = run(rows, spec(trend_filter="ema", trend_ema=20)).trades
            self.assertEqual(len(trades), 0 if above else 1)
        # nan EMA (not enough bars yet) fails the filter
        self.assertEqual(run(CLEAN, spec(trend_filter="ema", trend_ema=20)).trades, [])

    def test_require_structure(self):
        # the anchor low is the first confirmed low: there is no higher-low structure yet
        r = run(CLEAN, spec(require_structure=True))
        self.assertEqual((r.trades, r.signals), ([], []))

    def test_skip_legs(self):
        r = run(CLEAN, skip_legs={LEG})
        self.assertEqual((r.trades, r.open_trade), ([], None))
        self.assertNotIn(LEG, {s.leg_id for s in r.signals})
        self.assertEqual(len(run(CLEAN, skip_legs=frozenset({"L123"})).trades), 1)

    def test_trade_start_t_gating(self):
        r = run(CLEAN, trade_start_t=t_(10))                  # fill bar allowed
        self.assertEqual(len(r.trades), 1)
        self.assertEqual(r.signals, [])                        # armed at t5 < t10 - 4h
        r = run(CLEAN, trade_start_t=t_(6))
        self.assertEqual([s.t for s in r.signals], [t_(5)])    # t5 >= t6 - 4h
        r = run(CLEAN, trade_start_t=t_(10) + 1)               # the fill bar is before the start
        self.assertEqual((r.trades, r.open_trade), ([], None))

    def test_both_directions_fill_the_order_closest_to_the_open(self):
        # at bar 10 a long limit (97.32) and a short limit (~105.97, leg 110 -> 99) are both live;
        # an outside bar reaches both, the one priced nearer the open fills
        for o10, side, leg, entry in ((99.5, 1, LEG, ENTRY), (104, -1, f"S{t_(8)}", 99 + 0.634 * 11)):
            with self.subTest(open=o10):
                r = run(HEAD + [(o10, 106.5, 97.0, 100)], spec(direction="both"))
                ot = r.open_trade
                self.assertEqual((ot.side, ot.leg_id, ot.entry_t), (side, leg, t_(10)))
                self.assertAlmostEqual(ot.entry_price, entry, places=9)


class TestConfirmAndShort(FillsMixin, unittest.TestCase):
    def test_confirm_reclaim_enters_at_next_open(self):
        # bar 10 touches the zone (low 97 <= zone top 97.64) but closes red; bar 11 reclaims
        rows = HEAD + [FILL, (97.8, 99, 97.2, 98.9), (99.0, 104, 98.5, 103.5)]
        r = run(rows, spec(entry_mode="confirm", confirm_type="reclaim"))
        self.assertEqual(r.trades, [])
        tr = r.open_trade
        self.assertEqual((tr.leg_id, tr.entry_t, tr.meta["entry_mode"]), (LEG, t_(12), "confirm"))
        self.assertAlmostEqual(tr.entry_price, 99.0 * (1 + SLIP), places=9)
        self.assertAlmostEqual(tr.stop_price, STOP, places=9)
        self.assertAlmostEqual(tr.meta["entry_ratio"], (EXTREME - 98.9) / 20)
        self.assertAlmostEqual(tr.meta["rr"], (0.5 * (TP1 - 98.9) + 0.5 * (TP2 - 98.9)) / (98.9 - STOP))
        self.assertEqual(r.signals[0].meta["mode"], "confirm")

    def test_confirm_window_expiry(self):
        rows = HEAD + [FILL, (98, 98.5, 97.5, 97.7), (97.7, 99, 97.7, 98.9), (99, 100, 98.8, 99.5)]
        r = run(rows, spec(entry_mode="confirm", confirm_window=1))
        self.assertEqual((r.trades, r.open_trade), ([], None))
        r = run(rows, spec(entry_mode="confirm", confirm_window=2))
        self.assertEqual(r.open_trade.entry_t, t_(13))

    def test_confirm_engulf(self):
        # bar 10 is red and touches the zone; bar 11 opens at bar 10's close and engulfs it
        sp = spec(entry_mode="confirm", confirm_type="engulf")
        r = run(HEAD + [FILL, (97.8, 100, 97.6, 99.8), (99.8, 101, 99.5, 100.5)], sp)
        self.assertEqual(r.open_trade.entry_t, t_(12))
        self.assertAlmostEqual(r.open_trade.entry_price, 99.8 * (1 + SLIP), places=9)
        # opening above the prior close is not an engulfing bar
        r = run(HEAD + [FILL, (97.9, 100, 97.6, 99.8), (99.8, 101, 99.5, 100.5)], sp)
        self.assertEqual((r.trades, r.open_trade), ([], None))

    def test_short_mirror_scenario(self):
        # ATR thresholds are reflection invariant, so the mirrored bars give the mirrored pivots
        sp = dict(swing_mode="atr", swing_mult=1.5)
        long_tr = self.one_trade(run(CLEAN, spec(**sp)))
        self.assertEqual((long_tr.leg_id, long_tr.exit_reason), (LEG, "target"))
        r = run(mirror(CLEAN), spec(direction="short", **sp))
        tr = self.one_trade(r)
        e, st, t1, t2 = 200 - ENTRY, 200 - STOP, 200 - TP1, 200 - TP2
        self.assertEqual((tr.side, tr.leg_id, tr.entry_t, tr.exit_t, tr.exit_reason),
                         (-1, f"S{t_(4)}", t_(10), t_(14), "target"))
        self.assertAlmostEqual(tr.entry_price, e, places=9)
        self.assertAlmostEqual(tr.stop_price, st, places=9)
        self.assertFills(tr.meta["fills"], [(0.5, t1, "tp1", 12), (0.5, t2, "tp2", 14)])
        ret = short_ret([(0.5, t1), (0.5, t2)], e)
        self.assertAlmostEqual(tr.ret_pct, ret, places=12)
        self.assertAlmostEqual(tr.r_multiple, ret / ((st - e) / e), places=10)
        self.assertEqual(r.signals[0].side, -1)
        # short stop exits pay slippage upwards; short breakeven sits below the entry
        tr = self.one_trade(run(mirror(HEAD + [FILL, (97.8, 98, 92, 92.5)]), spec(direction="short", **sp)))
        self.assertFills(tr.meta["fills"], [(1.0, st * (1 + SLIP), "stop", 11)])
        rows = mirror(HEAD + [FILL, UP11, (103.5, 111, 97.4, 110.5), (110, 110.5, 97, 98)])
        tr = self.one_trade(run(rows, spec(direction="short", **sp)))
        be = e * (1 - 2 * FEE)
        self.assertFills(tr.meta["fills"], [(0.5, t1, "tp1", 12), (0.5, be * (1 + SLIP), "breakeven", 13)])


class TestManagePosition(FillsMixin, unittest.TestCase):
    ROWS = [(100, 101, 99, 100), (100, 100.5, 95, 96), (96, 104, 95.8, 103), (103, 112, 102, 111),
            (111, 121, 110, 120), (120, 121, 119, 120)]

    def setUp(self):
        self.b = mk(self.ROWS)
        self.a = atr(self.b.h, self.b.l, self.b.c, 5)
        self.sp = spec()

    def test_entry_bar_check_and_start_mgmt_idx(self):
        m = manage_position(self.b, self.a, 1, 1, 100.0, 95.5, 110.0, 120.0, self.sp, COSTS)
        self.assertEqual((m["exit_idx"], m["exit_reason"], m["open"]), (1, "stop", False))
        self.assertFills(m["fills"], [(1.0, 95.5 * (1 - SLIP), "stop", 1)])
        # starting management at bar 2 skips the entry bar: tp1 at bar 3, tp2 at bar 4
        m = manage_position(self.b, self.a, 1, 1, 100.0, 95.5, 110.0, 120.0, self.sp, COSTS, start_mgmt_idx=2)
        self.assertEqual((m["exit_idx"], m["exit_reason"]), (4, "target"))
        self.assertFills(m["fills"], [(0.5, 110, "tp1", 3), (0.5, 120, "tp2", 4)])
        ret = long_ret([(0.5, 110), (0.5, 120)], 100.0)
        self.assertAlmostEqual(m["ret_pct"], ret, places=12)
        self.assertAlmostEqual(m["risk_pct"], 0.045, places=12)
        self.assertAlmostEqual(m["r_multiple"], ret / 0.045, places=10)
        self.assertAlmostEqual(m["exit_price"], 115.0)

    def test_fill_beyond_stop_open_and_time(self):
        m = manage_position(self.b, self.a, 1, 1, 95.0, 95.5, 110.0, 120.0, self.sp, COSTS)
        self.assertFills(m["fills"], [(1.0, 95 * (1 - SLIP), "stop", 1)])
        # a market fill lifted above the stop only by slippage: the open (100, long) / 96 (short)
        # is at or beyond the stop, so it exits there with exit slippage, not at the stop
        for side, i, stop in ((1, 1, 100.02), (-1, 2, 95.97)):
            with self.subTest(side=side):
                o = self.ROWS[i][0]
                m = manage_position(self.b, self.a, i, side, o * (1 + side * SLIP), stop, 0.0, 0.0,
                                    self.sp, COSTS)
                self.assertFills(m["fills"], [(1.0, o * (1 - side * SLIP), "stop", i)])
        # a short that is never stopped and never reaches its target stays open at the end
        m = manage_position(self.b, self.a, 2, -1, 96.0, 130.0, 50.0, 40.0, self.sp, COSTS)
        self.assertEqual((m["open"], m["exit_reason"], m["exit_idx"]), (True, "open", 5))
        self.assertFills(m["fills"], [(1.0, 120 * (1 + SLIP), "open", 5)])
        self.assertAlmostEqual(m["ret_pct"], short_ret([(1.0, 120 * (1 + SLIP))], 96.0), places=12)
        m = manage_position(self.b, self.a, 2, 1, 96.0, 90.0, 200.0, 300.0, spec(max_hold_bars=2), COSTS)
        self.assertEqual((m["exit_idx"], m["exit_reason"]), (4, "time"))
        self.assertFills(m["fills"], [(1.0, 120 * (1 - SLIP), "time", 4)])

    def test_bad_arguments(self):
        for kw in ({"side": 0}, {"entry_idx": 6}, {"entry_idx": -1}, {"start_mgmt_idx": 0}):
            args = dict(entry_idx=1, side=1, start_mgmt_idx=None)
            args.update(kw)
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                manage_position(self.b, self.a, args["entry_idx"], args["side"], 100.0, 95.5, 110.0,
                                120.0, self.sp, COSTS, start_mgmt_idx=args["start_mgmt_idx"])

    def test_engine_and_manage_position_share_exit_logic(self):
        b = mk(CLEAN)
        tr = run(CLEAN).trades[0]
        m = manage_position(b, atr(b.h, b.l, b.c, 5), 10, 1, tr.entry_price, tr.stop_price,
                            tr.meta["tp1"], tr.meta["tp2"], spec(), COSTS)
        self.assertEqual((m["fills"], m["ret_pct"], m["r_multiple"], m["exit_reason"]),
                         (tr.meta["fills"], tr.ret_pct, tr.r_multiple, tr.exit_reason))


CAUSAL_SPECS = {
    "limit_both": PRESETS["golden_pocket_both_4h"].with_overrides(trend_ema=50),
    "confirm_reclaim_both": PRESETS["ote_confirm_1h"].with_overrides(timeframe="4h", direction="both",
                                                                     trend_ema=50),
    "confirm_engulf_short": StrategySpec(direction="short", trend_filter="none", entry_mode="confirm",
                                         confirm_type="engulf", entry_zone_hi=0.786, stop_ratio=1.0,
                                         min_leg_atr=3.0),
    "confirm_rsi_both": StrategySpec(direction="both", trend_filter="none", entry_mode="confirm",
                                     confirm_type="rsi", confirm_window=10, entry_zone_hi=0.786,
                                     stop_ratio=1.0, min_leg_atr=3.0),
    "trail_runner_both": PRESETS["ext_runner_4h"].with_overrides(direction="both", trend_ema=50),
    "pct_structure_slope": StrategySpec(direction="both", swing_mode="pct", swing_pct=0.04,
                                        trend_filter="ema_slope", trend_ema=30, require_structure=True,
                                        entry_at="hi", max_hold_bars=15, min_leg_atr=2.0),
    "stack_adx_long": PRESETS["shallow_trend_4h"].with_overrides(trend_ema=100, tp1_ext=1.272, tp1_frac=1.0),
}


class TestCausalityAndProperties(unittest.TestCase):
    N = 1200

    def full_and_bars(self, name, seed, start_i):
        bars = random_walk(self.N, seed)
        eng = FibEngine(CAUSAL_SPECS[name], COSTS)
        start = None if start_i is None else bars.t[start_i]
        return eng, bars, start, eng.run(bars, trade_start_t=start)

    def test_prefix_runs_reproduce_history(self):
        for name in CAUSAL_SPECS:
            for seed, start_i in ((1, None), (5, 400), (6, None)):
                with self.subTest(spec=name, seed=seed):
                    self._check_prefixes(*self.full_and_bars(name, seed, start_i), seed)

    def _check_prefixes(self, eng, bars, start, full, seed):
        idx = {t: i for i, t in enumerate(bars.t)}
        everything = full.trades + ([full.open_trade] if full.open_trade else [])
        self.assertGreaterEqual(len(full.trades), 3)
        ks = set(random.Random(seed).sample(range(1, self.N + 1), 25))
        for tr in everything:
            e = idx[tr.entry_t]
            x = self.N - 1 if tr.exit_t is None else idx[tr.exit_t]
            ks |= {e, e + 1, x, x + 1}
        for k in sorted(k for k in ks if 1 <= k <= self.N):
            last_t = bars.t[k - 1]
            sub = eng.run(bars.slice(0, k), trade_start_t=start)
            self.assertEqual(sub.trades, [tr for tr in full.trades if tr.exit_t <= last_t], k)
            self.assertEqual(sub.signals, [s for s in full.signals if s.t <= last_t], k)
            live = [tr for tr in everything
                    if tr.entry_t <= last_t and (tr.exit_t is None or tr.exit_t > last_t)]
            if not live:
                self.assertIsNone(sub.open_trade, k)
                continue
            ot, ref = sub.open_trade, live[0]
            self.assertEqual((ot.leg_id, ot.side, ot.entry_t, ot.entry_price, ot.stop_price, ot.setup_t),
                             (ref.leg_id, ref.side, ref.entry_t, ref.entry_price, ref.stop_price,
                              ref.setup_t), k)

    def test_long_short_mirror_symmetry_without_costs(self):
        zero = Costs(0.0, 0.0)
        for name in ("limit_both", "confirm_reclaim_both", "confirm_rsi_both", "trail_runner_both"):
            for seed in (4, 5):
                with self.subTest(spec=name, seed=seed):
                    bars = random_walk(self.N, seed)
                    k = 2 * max(bars.h)
                    mbars = Bars.from_rows(bars.symbol, bars.tf, [
                        (t, k - o, k - l, k - h, k - c, 1.0)
                        for t, o, h, l, c in zip(bars.t, bars.o, bars.h, bars.l, bars.c)])
                    s = CAUSAL_SPECS[name]
                    longs = FibEngine(s.with_overrides(direction="long"), zero).run(bars).trades
                    shorts = FibEngine(s.with_overrides(direction="short"), zero).run(mbars).trades
                    self.assertGreaterEqual(len(longs), 3)
                    self.assertEqual(len(longs), len(shorts))
                    for a, b in zip(longs, shorts):
                        self.assertEqual((a.entry_t, a.exit_t, a.exit_reason, a.bars_held, a.leg_id[1:]),
                                         (b.entry_t, b.exit_t, b.exit_reason, b.bars_held, b.leg_id[1:]))
                        self.assertEqual((a.side, b.side, a.leg_id[0], b.leg_id[0]), (1, -1, "L", "S"))
                        self.assertAlmostEqual(a.entry_price, k - b.entry_price, places=6)
                        self.assertAlmostEqual(a.stop_price, k - b.stop_price, places=6)
                        for fa, fb in zip(a.meta["fills"], b.meta["fills"], strict=True):
                            self.assertEqual((fa[0], fa[2], fa[3]), (fb[0], fb[2], fb[3]))
                            self.assertAlmostEqual(fa[1], k - fb[1], places=6)

    def test_invariants(self):
        for name in CAUSAL_SPECS:
            with self.subTest(spec=name):
                eng, bars, _, r = self.full_and_bars(name, 6, None)
                a = atr(bars.h, bars.l, bars.c, eng.spec.swing_atr_len)
                idx = {t: i for i, t in enumerate(bars.t)}
                sig_t = {s.leg_id: s.t for s in r.signals}
                self.assertEqual(len(sig_t), len(r.signals))              # one signal per leg
                everything = r.trades + ([r.open_trade] if r.open_trade else [])
                for prev, nxt in zip(everything, everything[1:]):
                    self.assertGreater(nxt.entry_t, prev.exit_t)           # one position at a time
                for tr in everything:
                    self.assertEqual(sig_t[tr.leg_id], tr.setup_t)
                    fills = tr.meta["fills"]
                    self.assertAlmostEqual(sum(f for f, *_ in fills), 1.0, places=12)
                    self.assertAlmostEqual(tr.exit_price, sum(f * p for f, p, *_ in fills), places=9)
                    self.assertAlmostEqual(tr.risk_pct, abs(tr.entry_price - tr.stop_price) / tr.entry_price)
                    self.assertAlmostEqual(tr.r_multiple, tr.ret_pct / tr.risk_pct)
                    ret = (long_ret if tr.side > 0 else short_ret)([(f, p) for f, p, *_ in fills], tr.entry_price)
                    self.assertAlmostEqual(tr.ret_pct, ret, places=12)
                    self.assertIn(tr.exit_reason, {"stop", "breakeven", "trail", "target", "time", "open"})
                    self.assertLessEqual(tr.bars_held, eng.spec.max_hold_bars)
                    # the shared exit logic reproduces every trade from its entry
                    e = idx[tr.entry_t]
                    m = manage_position(bars, a, e, tr.side, tr.entry_price, tr.stop_price,
                                        tr.meta["tp1"], tr.meta["tp2"], eng.spec, COSTS)
                    self.assertEqual((m["fills"], m["ret_pct"], m["exit_reason"], m["open"]),
                                     (fills, tr.ret_pct, tr.exit_reason, tr.exit_t is None))

    def test_trade_start_gating_on_random_walk(self):
        for name in ("limit_both", "confirm_reclaim_both"):
            with self.subTest(spec=name):
                _, bars, start, r = self.full_and_bars(name, 7, 600)
                self.assertGreater(len(r.trades), 0)
                for tr in r.trades + ([r.open_trade] if r.open_trade else []):
                    self.assertGreaterEqual(tr.entry_t, start)
                self.assertTrue(all(s.t >= start - STEP for s in r.signals))
                self.assertGreater(len(r.signals), 0)

    def test_speed_15k_bars(self):
        bars = random_walk(15000, 8)
        for name in ("golden_pocket_both_4h", "ext_runner_4h"):
            eng = FibEngine(PRESETS[name], COSTS)
            t0 = time.perf_counter()
            eng.run(bars)
            self.assertLess(time.perf_counter() - t0, 5.0, name)


if __name__ == "__main__":
    unittest.main()
