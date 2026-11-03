"""Regression tests for the engine review findings (fibbot/strategy.py, fibbot/types.py):

F39  the entry bar credits a target its close is beyond, AT the target (a resting bracket fills
     there), instead of the next bar crediting it at its open;
F18  a resting limit entry needs a trade-through of Costs.limit_through, a bare touch never fills;
F21  StrategySpec.min_stop_cost_mult: minimum stop distance in round-trip costs;
F44  a fill at or beyond the stop measures R against the order's planned risk.

Scenarios reuse the canonical hand-built leg of test_strategy_contract (1h bars, anchor low 80,
leg high 120: limit 100, stop 90, tp1 120, tp2 140)."""
from __future__ import annotations

import math
import random
import unittest

from fibbot import indicators
from fibbot.backtest import costs_from_cfg
from fibbot.config import DEFAULTS
from fibbot.strategy import (ENGINE_VERSION, PARAM_SPACE, PRESETS, FibEngine, StrategySpec,
                             limit_fills, manage_position, round_trip_cost)
from fibbot.types import Costs
from tests.test_strategy_contract import (BE_LONG, CSETUP, FEE, LEG, NO_FILL, SHORT_LEG, SLIP,
                                          EngineCase, long_ret, make_bars, mirror, pivots,
                                          random_bars, run, setup_rows, short_ret, spec, t, A, J)
from tests.test_strategy import random_walk

COSTS = Costs(FEE, SLIP)
LT = Costs().limit_through


# --------------------------------------------------------------------------- F39 entry-bar targets

class TestEntryBarTargets(EngineCase):
    def test_entry_bar_closing_beyond_both_targets_fills_them_at_their_levels(self):
        """Fill at 100 (stop 90), the bar closes at 145 > tp2 140 with its low above the stop:
        tp1 and tp2 fill at 120 / 140 on the entry bar, not at the next bar's open (145)."""
        rows = setup_rows() + [NO_FILL, (104.0, 146.0, 98.0, 145.0), (145.0, 147.0, 144.0, 146.0)]
        tr = self.only_trade(run(rows))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 34), (0.5, 140.0, 34)], reason="target")
        self.assertEqual(tr.bars_held, 0)

    def test_entry_bar_closing_beyond_tp1_only(self):
        """Close 125 (beyond tp1 only): tp1 is credited at 120 on the entry bar; the next bar
        opening at 125 does not refill it and tp2 then follows the normal rules."""
        rows = setup_rows() + [NO_FILL, (104.0, 126.0, 98.0, 125.0)]
        tr = self.only_open(run(rows + [(125.0, 130.0, 124.0, 128.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 34), (0.5, 128.0 * (1 - SLIP), 35)], reason="open")
        self.assertEqual([f[2] for f in tr.meta["fills"]], ["tp1", "open"])
        tr = self.only_trade(run(rows + [(125.0, 130.0, 124.0, 128.0), (128.0, 141.0, 127.0, 139.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 34), (0.5, 140.0, 36)], reason="target")

    def test_breakeven_after_entry_bar_tp1_acts_from_the_next_bar(self):
        """tp1 credited at the entry bar's close moves the stop to breakeven for the next bar:
        a dip to 100.1 (< 100.2) on bar 35 exits there, reason "breakeven". (The old rule filled
        tp1 at bar 35's open 125 and kept the rest open.)"""
        rows = setup_rows() + [NO_FILL, (104.0, 126.0, 98.0, 125.0), (125.0, 126.0, 100.1, 101.0)]
        tr = self.only_trade(run(rows, move_stop_to_be=True))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 34), (0.5, BE_LONG * (1 - SLIP), 35)],
                         reason="breakeven")

    def test_tp1_frac_one_closes_on_the_entry_bar(self):
        rows = setup_rows() + [NO_FILL, (104.0, 126.0, 98.0, 125.0), (125.0, 130.0, 124.0, 128.0)]
        tr = self.only_trade(run(rows, tp1_frac=1.0))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 120.0, 34)], reason="target")

    def test_close_at_the_target_or_high_only_is_not_credited(self):
        """Only a close strictly beyond the target proves it was crossed after the fill: a close
        exactly at tp1, or a high above it with a close below, credits nothing on the entry bar."""
        for bar in ((104.0, 121.0, 98.0, 120.0), (104.0, 125.0, 98.0, 110.0)):
            with self.subTest(bar=bar):
                tr = self.only_open(run(setup_rows() + [NO_FILL, bar]))
                self.assertEqual([f[2] for f in tr.meta["fills"]], ["open"])

    def test_true_gap_after_entry_bar_still_fills_at_the_open(self):
        """Entry bar closes below tp1 (101); the next bar gaps open at 125: that is a real gap,
        filled at the (better) open."""
        rows = setup_rows() + [NO_FILL, (104.0, 105.0, 98.0, 101.0), (125.0, 130.0, 124.0, 128.0)]
        tr = self.only_open(run(rows))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 125.0, 35), (0.5, 128.0 * (1 - SLIP), 35)], reason="open")

    def test_short_mirror(self):
        rows = mirror(setup_rows() + [NO_FILL, (104.0, 146.0, 98.0, 145.0),
                                      (145.0, 147.0, 144.0, 146.0)])
        tr = self.only_trade(run(rows, direction="short"))
        self.check_trade(tr, side=-1, leg=SHORT_LEG, entry_i=34, entry=100.0, stop=110.0,
                         fills=[(0.5, 80.0, 34), (0.5, 60.0, 34)], reason="target")

    def test_confirm_mode_market_entry_bar(self):
        """Market entry at o[35]*(1+slip); bar 35 closes at 145 > tp2: both targets at their levels."""
        rows = CSETUP + [NO_FILL, (101.0, 104.0, 99.0, 103.0), (103.0, 146.0, 102.0, 145.0),
                         (145.0, 147.0, 144.0, 146.0)]
        tr = self.only_trade(run(rows, entry_mode="confirm"))
        self.check_trade(tr, side=1, leg=LEG, entry_i=35, entry=103.0 * (1 + SLIP), stop=90.0,
                         fills=[(0.5, 120.0, 35), (0.5, 140.0, 35)], reason="target")

    def test_trailing_starts_at_the_entry_bar_close(self):
        """After an entry-bar tp1 the close-of-bar trail applies on that same close."""
        rows = setup_rows() + [NO_FILL, (104.0, 126.0, 98.0, 125.0)]
        b = make_bars(rows + [(125.0, 125.5, 110.0, 112.0)])
        trail = 125.0 - 1.0 * indicators.atr(b.h, b.l, b.c, 5)[34]
        self.assertTrue(110.0 < trail < 125.0)
        tr = self.only_trade(run(rows + [(125.0, 125.5, 110.0, 112.0)], trail_atr=1.0))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 34), (0.5, trail * (1 - SLIP), 35)], reason="trail")

    def test_manage_position_parity(self):
        """manage_position runs the same entry-bar rule (shared _Position code)."""
        for bars_rows, kw in (([(104.0, 146.0, 98.0, 145.0), (145.0, 147.0, 144.0, 146.0)], {}),
                              ([(104.0, 126.0, 98.0, 125.0), (125.0, 126.0, 100.1, 101.0)],
                               {"move_stop_to_be": True})):
            with self.subTest(rows=bars_rows):
                rows = setup_rows() + [NO_FILL] + bars_rows
                tr = self.only_trade(run(rows, **kw))
                b = make_bars(rows)
                out = manage_position(b, indicators.atr(b.h, b.l, b.c, 5), 34, 1, 100.0, 90.0,
                                      120.0, 140.0, spec(**kw), COSTS)
                self.assertEqual(out["fills"], tr.meta["fills"])
                self.assertEqual((out["exit_reason"], out["ret_pct"], out["r_multiple"]),
                                 (tr.exit_reason, tr.ret_pct, tr.r_multiple))

    def test_no_target_fill_beyond_its_level_without_a_true_gap(self):
        """Property on random walks (with open gaps) and tight targets: a target never fills at a
        price beyond its level unless that bar opened beyond it after a close on the near side
        (a true gap), and never beyond its level on the entry bar."""
        specs = [
            spec(swing_pct=0.05, entry_at="mid", tp1_ext=0.5, tp2_ext=1.0, move_stop_to_be=True,
                 min_rr=0.0, max_hold_bars=30),
            spec(direction="both", swing_pct=0.04, tp1_ext=0.3, tp1_frac=1.0, min_rr=0.0),
            spec(direction="both", swing_pct=0.04, entry_mode="confirm", entry_zone_lo=0.382,
                 entry_zone_hi=0.786, stop_ratio=1.0, tp1_ext=0.5, tp2_ext=0.8, min_rr=0.0),
        ]
        crossed = gaps = 0
        for seed in range(6):
            bars = random_bars(seed, n=600, vol=0.03)
            idx = {x: i for i, x in enumerate(bars.t)}
            for sp in specs:
                res = FibEngine(sp, COSTS).run(bars)
                for tr in res.trades + ([res.open_trade] if res.open_trade else []):
                    e, s = idx[tr.entry_t], tr.side
                    for frac, px, label, ft in tr.meta["fills"]:
                        if label not in ("tp1", "tp2"):
                            continue
                        level, k = tr.meta[label], idx[ft]
                        crossed += k == e
                        if s * (px - level) > 1e-12:
                            gaps += 1
                            self.assertGreater(k, e, "a target beyond its level on the entry bar")
                            self.assertEqual(px, bars.o[k])
                            self.assertLessEqual(s * (bars.c[k - 1] - level), 0,
                                                 f"{label} filled at the open {px} after a close "
                                                 f"{bars.c[k - 1]} already beyond {level}")
        self.assertGreater(crossed, 0)      # the entry-bar rule was exercised
        self.assertGreater(gaps, 0)         # and so were true gaps


# --------------------------------------------------------------------------- F18 trade-through

class TestLimitTradeThrough(EngineCase):
    def test_default_is_positive_and_predicate(self):
        self.assertGreater(LT, 0)
        self.assertEqual(Costs(FEE, SLIP), Costs(FEE, SLIP, LT))
        self.assertFalse(limit_fills(1, 100.0, 105.0, 100.0, LT))
        self.assertFalse(limit_fills(1, 100.0, 105.0, 100.0 * (1 - LT / 2), LT))
        self.assertTrue(limit_fills(1, 100.0, 105.0, 100.0 * (1 - LT), LT))
        self.assertTrue(limit_fills(1, 100.0, 105.0, 100.0, 0.0))        # 0: a touch fills
        self.assertFalse(limit_fills(-1, 100.0, 100.0, 95.0, LT))
        self.assertTrue(limit_fills(-1, 100.0, 100.0 * (1 + LT), 95.0, LT))
        self.assertTrue(limit_fills(-1, 100.0, 100.0, 95.0, 0.0))

    def test_costs_rejects_a_negative_or_absurd_margin(self):
        """A negative margin would fill limits before price reaches them."""
        Costs(limit_through=0.0)
        Costs(limit_through=0.05)
        for bad in (-0.0001, 0.06, float("nan"), "0.0002", True, None):
            with self.subTest(value=bad), self.assertRaises(ValueError) as cm:
                Costs(limit_through=bad)
            self.assertTrue(str(cm.exception).startswith("limit_through"))

    def test_touch_does_not_fill_a_later_trade_through_does(self):
        """Bar 34 tags the 100 limit exactly (or inside the margin) and closes above it: no fill;
        the order rests again and bar 35 trading through fills at 100."""
        for low in (100.0, 100.0 * (1 - LT / 2)):
            with self.subTest(low=low):
                rows = setup_rows() + [NO_FILL, (104.0, 105.0, low, 101.0),
                                       (101.0, 102.0, 99.0, 100.5)]
                tr = self.only_open(run(rows))
                self.assertEqual(tr.entry_t, t(35))
                self.close(tr.entry_price, 100.0)
                touch = run(rows, costs=Costs(FEE, SLIP, 0.0))      # the old touch rule
                self.assertEqual(self.only_open(touch).entry_t, t(34))

    def test_short_touch_does_not_fill(self):
        rows = mirror(setup_rows() + [NO_FILL, (104.0, 105.0, 100.0, 101.0),
                                      (101.0, 102.0, 99.0, 100.5)])
        tr = self.only_open(run(rows, direction="short"))
        self.assertEqual((tr.side, tr.entry_t), (-1, t(35)))
        self.close(tr.entry_price, 100.0)

    def test_gap_through_fills_at_the_open(self):
        """Opening below the limit still fills at the open, provided the bar trades through by
        the margin; an open and low both inside the margin do not fill."""
        tr = self.only_open(run(setup_rows() + [NO_FILL, (98.0, 99.0, 95.0, 97.0)]))
        self.close(tr.entry_price, 98.0)
        tr = self.only_open(run(setup_rows() + [NO_FILL, (99.99, 101.0, 99.0, 100.5)]))
        self.close(tr.entry_price, 99.99)
        self.assert_flat(run(setup_rows() + [NO_FILL, (99.995, 101.0, 99.99, 100.5)]))

    def test_both_directions_only_the_order_traded_through_fills(self):
        """Both limits are reached, the short one (110.5, nearer the open 108) only touched: the
        long, which the bar traded through, fills."""
        rows = setup_rows(a1=(81.0, 90.0, 80.5, 89.0), j=(91.0, 120.0, 91.0, 118.0)) + [
            (118.0, 118.5, 108.0, 110.0), (110.0, 110.5, 101.0, 103.0), (108.0, 110.5, 99.5, 105.0)]
        self.assertEqual(pivots(rows, 0.15), [("L", A, J), ("H", J, 34)])
        tr = self.only_open(run(rows, direction="both", swing_pct=0.15))
        self.assertEqual((tr.side, tr.leg_id, tr.entry_t, tr.entry_price), (1, LEG, t(35), 100.0))
        tr = self.only_open(run(rows, costs=Costs(FEE, SLIP, 0.0), direction="both", swing_pct=0.15))
        self.assertEqual(tr.side, -1)

    def test_market_and_target_fills_are_unaffected(self):
        """limit_through only gates resting ENTRY limits: targets still fill on a touch."""
        rows = setup_rows() + [NO_FILL, (104.0, 105.0, 98.0, 101.0), (102.0, 120.0, 101.0, 115.0)]
        tr = self.only_open(run(rows))
        self.close(tr.exit_price, 0.5 * 120.0 + 0.5 * 115.0 * (1 - SLIP))

    def test_backtest_and_paper_costs_agree_by_default(self):
        """Paper builds Costs(**cfg["costs"]) and the backtest costs_from_cfg(cfg): with the
        default config both trade under the same (positive) trade-through."""
        a, b = costs_from_cfg(DEFAULTS), Costs(**DEFAULTS["costs"])
        self.assertEqual(a, b)
        self.assertGreater(a.limit_through, 0)


# --------------------------------------------------------------------------- F21 min_stop_cost_mult

# entry 100 (zone 0.5-0.5), stop ratio 0.51 -> 99.6: a 0.4% stop, RR 75; neither the extreme bar's
# low 99.7 nor the fill bar's low 99.9 (which trades through the limit) reaches the stop
TIGHT = dict(entry_zone_hi=0.5, stop_ratio=0.51)
TIGHT_ROWS = setup_rows(j=(99.0, 120.0, 99.7, 118.0)) + [NO_FILL, (104.0, 105.0, 99.9, 101.0)]


class TestMinStopCostMult(EngineCase):
    def test_param_space_and_validation(self):
        p = PARAM_SPACE["min_stop_cost_mult"]
        self.assertEqual((p["type"], p["min"], p["max"], p["default"]), ("float", 0.0, 20.0, 0.0))
        self.assertGreater(p["step"], 0)
        StrategySpec(min_stop_cost_mult=20.0).validate()
        for bad in (-0.5, 20.5, "3", True, float("nan")):
            with self.subTest(value=bad), self.assertRaises(ValueError) as cm:
                StrategySpec(min_stop_cost_mult=bad).validate()
            self.assertTrue(str(cm.exception).startswith("min_stop_cost_mult"))

    def test_existing_spec_ids_are_unchanged(self):
        """The field is left out of the spec_id blob at its default 0: stored ids stay valid."""
        want = {"golden_pocket_4h": "8450eeebe8", "golden_pocket_1h": "aaa3bbf571",
                "golden_pocket_both_4h": "e9f296e83e", "ote_confirm_1h": "97f60ed19c",
                "shallow_trend_4h": "d25afc4741", "deep_786_4h": "c343ee318f",
                "golden_pocket_1d": "94a858f2fa", "ext_runner_4h": "b8fe9b21e4"}
        self.assertEqual({k: PRESETS[k].spec_id() for k in want}, want)
        old = {k: v for k, v in StrategySpec().to_dict().items() if k != "min_stop_cost_mult"}
        self.assertEqual(StrategySpec.from_dict(old).spec_id(), "8450eeebe8")
        self.assertNotEqual(StrategySpec(min_stop_cost_mult=3).spec_id(), "8450eeebe8")
        self.assertEqual(StrategySpec(min_stop_cost_mult=3).spec_id(),
                         StrategySpec(min_stop_cost_mult=3.0).spec_id())

    def test_round_trip_cost(self):
        self.assertAlmostEqual(round_trip_cost(COSTS, "limit"), 2 * FEE + SLIP)
        self.assertAlmostEqual(round_trip_cost(COSTS, "confirm"), 2 * FEE + 2 * SLIP)

    def test_tight_stop_leg_is_skipped_when_set(self):
        """Stop 0.4% from the entry; limit-mode round trip 0.25%: k=1.5 (0.375%) qualifies and
        trades, k=2 (0.5%) never qualifies (no signal, no order); 0 disables the check."""
        res = run(TIGHT_ROWS, **TIGHT)
        self.assertEqual(len(res.signals), 1)
        self.only_open(res)
        res = run(TIGHT_ROWS, min_stop_cost_mult=1.5, **TIGHT)
        self.assertEqual(len(res.signals), 1)
        self.close(self.only_open(res).entry_price, 100.0)
        res = run(TIGHT_ROWS, min_stop_cost_mult=2.0, **TIGHT)
        self.assertEqual(res.signals, [])
        self.assert_flat(res)

    def test_threshold_scales_with_costs(self):
        """Doubling the costs doubles the required stop distance (k=1.5 -> 0.75% > 0.4%)."""
        res = run(TIGHT_ROWS, costs=Costs(2 * FEE, 2 * SLIP), min_stop_cost_mult=1.5, **TIGHT)
        self.assertEqual(res.signals, [])
        self.assert_flat(res)

    def test_confirm_mode_uses_its_own_round_trip(self):
        """Confirm mode pays entry slippage too (0.30%): at k=1.5 (0.45% > 0.4% at the nominal
        entry) the leg never arms although the limit-mode threshold (0.375%) would pass; k=1.25
        (0.375%) arms, and the market entry at c[k] keeps a stop distance above the threshold."""
        rows = setup_rows(j=(101.0, 120.0, 100.5, 118.0)) + [
            NO_FILL, (101.0, 104.0, 99.7, 103.0), (103.0, 104.0, 102.5, 103.5)]
        res = run(rows, entry_mode="confirm", min_stop_cost_mult=1.5, **TIGHT)
        self.assertEqual(res.signals, [])
        self.assert_flat(res)
        res = run(rows, entry_mode="confirm", min_stop_cost_mult=1.25, **TIGHT)
        self.assertEqual(len(res.signals), 1)
        tr = self.only_open(res)
        self.assertEqual(tr.entry_t, t(35))
        self.assertGreaterEqual((103.0 - tr.stop_price) / 103.0, 1.25 * 0.003)

    def test_on_random_walks_every_trade_respects_the_threshold(self):
        bars = random_walk(2000, 3, tf="1h")
        base = PRESETS["golden_pocket_1h"].with_overrides(trend_ema=50, min_leg_atr=2.0)
        plain = FibEngine(base, COSTS).run(bars).trades
        k = 3.0
        sp = base.with_overrides(min_stop_cost_mult=k)
        kept = FibEngine(sp, COSTS).run(bars).trades
        floor = k * round_trip_cost(COSTS, "limit")
        self.assertTrue(any((tr.meta["extreme"] - tr.meta["entry_ratio"] * tr.meta["leg_size"]
                             - tr.stop_price) / tr.entry_price < floor for tr in plain))
        self.assertTrue(kept)
        for tr in kept:
            m = tr.meta
            plan = m["extreme"] - m["entry_ratio"] * (m["extreme"] - m["anchor"])
            self.assertGreaterEqual((plan - tr.stop_price) / plan, floor - 1e-12)


# --------------------------------------------------------------------------- F44 R beyond the stop

class TestRiskWhenFilledBeyondTheStop(EngineCase):
    def test_limit_gap_below_stop(self):
        """Long limit 100, stop 90, bar opens at 89.99: immediate stop-out at the open with
        slippage; ret is fees + slippage and R uses the planned 10% risk (not 0.011%)."""
        tr = self.only_trade(run(setup_rows() + [NO_FILL, (89.99, 91.0, 85.0, 86.0)]))
        ret = long_ret([(1.0, 89.99 * (1 - SLIP))], 89.99)
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=89.99, stop=90.0,
                         fills=[(1.0, 89.99 * (1 - SLIP), 34)], reason="stop", risk=0.1)
        self.close(tr.r_multiple, ret / 0.1)
        self.assertGreater(tr.r_multiple, -0.05)

    def test_short_limit_gap_above_stop(self):
        rows = mirror(setup_rows() + [NO_FILL, (89.99, 91.0, 85.0, 86.0)])
        tr = self.only_trade(run(rows, direction="short"))
        ret = short_ret([(1.0, 110.01 * (1 + SLIP))], 110.01)
        self.check_trade(tr, side=-1, leg=SHORT_LEG, entry_i=34, entry=110.01, stop=110.0,
                         fills=[(1.0, 110.01 * (1 + SLIP), 34)], reason="stop", risk=0.1)
        self.close(tr.r_multiple, ret / 0.1)

    def test_confirm_market_fill_lifted_above_the_stop_only_by_slippage(self):
        """o = 89.99 < stop 90 but the fill o*(1+slip) = 90.035 is above it: the immediate-exit
        rule fires on the pre-slippage price and R uses the order's planned risk at c[k]=103
        (|P - stop| / P = 0.04% would make R about -8)."""
        rows = CSETUP + [NO_FILL, (101.0, 104.0, 99.0, 103.0), (89.99, 91.0, 85.0, 86.0)]
        tr = self.only_trade(run(rows, entry_mode="confirm"))
        self.assertGreater(tr.entry_price, 90.0)
        self.check_trade(tr, side=1, leg=LEG, entry_i=35, entry=89.99 * (1 + SLIP), stop=90.0,
                         fills=[(1.0, 89.99 * (1 - SLIP), 35)], reason="stop",
                         risk=13.0 / 103.0)
        self.assertGreater(tr.r_multiple, -0.05)

    def test_normal_fills_keep_the_fill_based_risk(self):
        """A fill on the right side of the stop keeps risk_pct = (P - stop) / P, also when the
        limit gapped in at a better price than planned."""
        tr = self.only_open(run(setup_rows() + [NO_FILL, (98.0, 99.0, 95.0, 97.0)]))
        self.close(tr.risk_pct, (98.0 - 90.0) / 98.0)

    def test_manage_position_plan_px(self):
        b = make_bars([(100.0, 101.0, 99.0, 100.0), (94.99, 96.0, 94.0, 95.5)])
        a = [1.0, 1.0]
        out = manage_position(b, a, 1, 1, 94.99, 95.0, 110.0, 120.0, spec(), COSTS, plan_px=100.0)
        self.assertEqual((out["exit_reason"], out["exit_idx"]), ("stop", 1))
        self.close(out["risk_pct"], 0.05)
        self.close(out["r_multiple"], out["ret_pct"] / 0.05)
        self.close(out["ret_pct"], long_ret([(1.0, 94.99 * (1 - SLIP))], 94.99))
        # without the planned price there is no meaningful basis: the legacy |P - stop| / P
        legacy = manage_position(b, a, 1, 1, 94.99, 95.0, 110.0, 120.0, spec(), COSTS)
        self.close(legacy["risk_pct"], 0.01 / 94.99)
        self.assertEqual(legacy["fills"], out["fills"])

    def test_random_walk_gap_stop_outs_cost_a_fraction_of_r(self):
        """On a walk with frequent large opening gaps, every entry that fills at or beyond its
        stop is an immediate stop-out whose R is its cost over the PLANNED risk: far above -1R
        (with the fill-to-stop distance it ranged from about -0.01R to below -20R)."""
        rnd, rows, px = random.Random(1), [], 100.0
        for _ in range(3000):
            o = px * math.exp(rnd.gauss(0, 0.03)) if rnd.random() < 0.3 else px
            c = o * math.exp(rnd.gauss(0, 0.02))
            rows.append((o, max(o, c) * (1 + abs(rnd.gauss(0, 0.007))),
                         min(o, c) * (1 - abs(rnd.gauss(0, 0.007))), c))
            px = c
        bars = make_bars(rows, "GAPUSDT")
        sp = spec(direction="both", swing_pct=0.06, stop_ratio=0.65, entry_zone_hi=0.618,
                  min_rr=0.0)
        res = FibEngine(sp, COSTS).run(bars)
        idx = {x: i for i, x in enumerate(bars.t)}
        hit = 0
        for tr in res.trades:
            e = idx[tr.entry_t]
            x = min(bars.o[e], tr.entry_price) if tr.side > 0 else max(bars.o[e], tr.entry_price)
            if tr.side * (x - tr.stop_price) <= 0:
                hit += 1
                m = tr.meta
                plan = m["extreme"] - m["entry_ratio"] * (m["extreme"] - m["anchor"])
                self.assertEqual((tr.exit_reason, tr.bars_held), ("stop", 0))
                self.assertAlmostEqual(tr.risk_pct, tr.side * (plan - tr.stop_price) / plan)
                self.assertTrue(-1.0 < tr.r_multiple < 0.0, tr.r_multiple)
        self.assertGreater(hit, 5)


class TestEngineVersion(unittest.TestCase):
    def test_engine_version_marks_the_new_rules(self):
        self.assertIsInstance(ENGINE_VERSION, int)
        self.assertGreaterEqual(ENGINE_VERSION, 2)


if __name__ == "__main__":
    unittest.main()
