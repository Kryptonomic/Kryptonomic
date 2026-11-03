import dataclasses
import math
import random
import unittest

from fibbot.fib import (EXTENSION_LEVELS, RETRACE_LEVELS, Pivot, PivotTracker, extension_price,
                        fib_table, retrace_price, zigzag)
from fibbot.indicators import atr
from fibbot.types import Bars

nan = float("nan")


def make_bars(hl, t0=1_700_000_000, step=3600):
    """Bars from (high, low) pairs; open/close at the bar midpoint."""
    rows = [(t0 + i * step, (h + l) / 2, h, l, (h + l) / 2, 1.0) for i, (h, l) in enumerate(hl)]
    return Bars.from_rows("TESTUSDT", "1h", rows)


def random_walk_bars(n, seed, vol=0.02):
    rnd = random.Random(seed)
    rows, px = [], 100.0
    for i in range(n):
        o = px
        c = o * math.exp(rnd.gauss(0, vol))
        h = max(o, c) * (1 + abs(rnd.gauss(0, vol / 2)))
        l = min(o, c) * (1 - abs(rnd.gauss(0, vol / 2)))
        rows.append((i * 3600, o, h, l, c, 1.0))
        px = c
    return Bars.from_rows("RWUSDT", "1h", rows)


def ref_zigzag(bars, atr_values, mult, mode, pct):
    """Independent, literal transcription of the DESIGN.md PivotTracker algorithm."""
    out, d = [], 0
    hi, hi_i, lo, lo_i = -math.inf, -1, math.inf, -1
    for i in range(len(bars)):
        H, L = bars.h[i], bars.l[i]
        a = atr_values[i]
        if mode == "atr":
            ok = not math.isnan(a) and mult * a > 0
            th_h = th_l = mult * a
        if d == 0:
            if H > hi:
                hi, hi_i = H, i
            if L < lo:
                lo, lo_i = L, i
            if mode == "pct":
                ok, th_h, th_l = True, pct * hi, pct * lo
            if not ok:
                continue
            c_h = hi - L >= th_h and hi_i < i
            c_l = H - lo >= th_l and lo_i < i
            if c_h and c_l:
                c_h = hi_i <= lo_i
                c_l = not c_h
            if c_h:
                out.append((hi_i, hi, "H", i))
                d, lo, lo_i = -1, L, i
            elif c_l:
                out.append((lo_i, lo, "L", i))
                d, hi, hi_i = 1, H, i
        elif d == 1:
            if H > hi:
                hi, hi_i = H, i
            else:
                if mode == "pct":
                    ok, th_h = True, pct * hi
                if ok and hi - L >= th_h:
                    out.append((hi_i, hi, "H", i))
                    d, lo, lo_i = -1, L, i
        else:
            if L < lo:
                lo, lo_i = L, i
            else:
                if mode == "pct":
                    ok, th_l = True, pct * lo
                if ok and H - lo >= th_l:
                    out.append((lo_i, lo, "L", i))
                    d, hi, hi_i = 1, H, i
    return out


def key(p):
    return (p.idx, p.price, p.kind, p.confirm_idx)


class TestFibLevels(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(RETRACE_LEVELS, (0.236, 0.382, 0.5, 0.618, 0.65, 0.705, 0.786, 0.886))
        self.assertEqual(EXTENSION_LEVELS, (1.272, 1.414, 1.618, 2.0, 2.618))

    def test_retrace_up_leg(self):
        # anchor = swing low 100, extreme = running high 200
        self.assertAlmostEqual(retrace_price(100, 200, 0.618), 138.2)
        self.assertAlmostEqual(retrace_price(100, 200, 0.5), 150.0)
        self.assertEqual(retrace_price(100, 200, 0.0), 200)
        self.assertEqual(retrace_price(100, 200, 1.0), 100)

    def test_retrace_down_leg(self):
        # anchor = swing high 200, extreme = running low 100: retracements are above the low
        self.assertAlmostEqual(retrace_price(200, 100, 0.618), 161.8)
        self.assertAlmostEqual(retrace_price(200, 100, 0.236), 123.6)
        self.assertEqual(retrace_price(200, 100, 0.0), 100)
        self.assertEqual(retrace_price(200, 100, 1.0), 200)

    def test_extension_both_legs(self):
        self.assertAlmostEqual(extension_price(100, 200, 1.272), 227.2)
        self.assertAlmostEqual(extension_price(100, 200, 1.618), 261.8)
        self.assertEqual(extension_price(100, 200, 1.0), 200)
        self.assertEqual(extension_price(100, 200, 0.0), 100)
        self.assertAlmostEqual(extension_price(200, 100, 1.272), 72.8)
        self.assertAlmostEqual(extension_price(200, 100, 2.618), -61.8)

    def test_extension_is_one_minus_retrace(self):
        for a, x in ((100, 200), (200, 100), (0.0123, 0.0456)):
            for e in (0.0, 0.5, 1.0, 1.272, 2.618):
                self.assertAlmostEqual(extension_price(a, x, e), retrace_price(a, x, 1 - e), places=12)

    def test_fib_table(self):
        tab = fib_table(100, 200)
        self.assertEqual(list(tab), ["0.236", "0.382", "0.5", "0.618", "0.65", "0.705", "0.786",
                                     "0.886", "1.272", "1.414", "1.618", "2.0", "2.618"])
        self.assertAlmostEqual(tab["0.618"], 138.2)
        self.assertAlmostEqual(tab["1.272"], 227.2)
        self.assertAlmostEqual(tab["2.0"], 300.0)
        down = fib_table(200, 100)
        self.assertAlmostEqual(down["0.618"], 161.8)
        self.assertAlmostEqual(down["1.272"], 72.8)


class TestPivotTracker(unittest.TestCase):
    def test_pivot_is_frozen(self):
        p = Pivot(1, 2, 3.0, "H", 4)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.price = 5.0

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            PivotTracker(mode="fixed")
        with self.assertRaises(ValueError):
            PivotTracker(mult=0)
        with self.assertRaises(ValueError):
            PivotTracker(mode="pct", pct=-0.1)
        b = make_bars([(10, 9), (11, 10)])
        with self.assertRaises(ValueError):
            zigzag(b, [1.0], mode="atr")
        with self.assertRaises(ValueError):
            zigzag(b, None, mode="atr")

    def test_atr_mode_alternating_and_no_confirm_on_extending_bar(self):
        b = make_bars([
            (10, 9),       # 0
            (13, 12),      # 1 dir 0: 13 - 9 >= 3 -> L@0 confirmed at 1 (this bar is a new high)
            (15, 14),      # 2 extend hi
            (16, 11),      # 3 new high 16 AND 16 - 11 >= 3: extends, must NOT confirm
            (15.5, 13),    # 4 16 - 13 == 3 (threshold is inclusive) -> H@3 confirmed at 4
            (13, 12),      # 5 extend lo
            (15, 12.5),    # 6 15 - 12 >= 3 -> L@5
        ])
        pivots = zigzag(b, [1.0] * len(b), mult=3.0)
        self.assertEqual([key(p) for p in pivots],
                         [(0, 9, "L", 1), (3, 16, "H", 4), (5, 12, "L", 6)])
        self.assertEqual([p.t for p in pivots], [b.t[0], b.t[3], b.t[5]])

    def test_new_low_with_spike_never_confirms(self):
        b = make_bars([
            (10, 9), (13, 12),  # L@0 confirmed at 1
            (16, 15),           # 2 extend hi
            (15, 12),           # 3 H@2 confirmed at 3
            (14, 11),           # 4 extend lo
            (16, 10.5),         # 5 new low 10.5 AND 16 - 11 >= 3: extends, must NOT confirm
            (14, 13),           # 6 14 - 10.5 >= 3 -> L@5
        ])
        pivots = zigzag(b, [1.0] * len(b), mult=3.0)
        self.assertEqual([key(p) for p in pivots],
                         [(0, 9, "L", 1), (2, 16, "H", 3), (5, 10.5, "L", 6)])

    def test_update_returns_pivot_and_tracks_state(self):
        tr = PivotTracker(mult=3.0)
        self.assertIsNone(tr.update(0, 100, 10, 9, 1.0))
        p = tr.update(1, 200, 13, 12, 1.0)
        self.assertEqual(p, Pivot(0, 100, 9, "L", 1))
        self.assertIs(tr.last, p)
        self.assertEqual((tr.direction, tr.hi, tr.hi_idx, tr.hi_t), (1, 13, 1, 200))
        self.assertIsNone(tr.update(2, 300, 15, 14, 1.0))
        self.assertEqual((tr.hi, tr.hi_idx), (15, 2))

    def test_pct_mode(self):
        b = make_bars([
            (100, 99),   # 0
            (101, 95),   # 1 both extremes on this bar -> nothing
            (98, 90),    # 2 101 - 90 = 11 >= 10.1 -> H@1
            (92, 85),    # 3 extend lo
            (93, 86),    # 4 93 - 85 = 8 < 8.5
            (94, 88),    # 5 94 - 85 = 9 >= 8.5 -> L@3
            (100, 93),   # 6 extend hi
            (99, 91),    # 7 100 - 91 = 9 < 10
            (95, 89.5),  # 8 100 - 89.5 = 10.5 >= 10 -> H@6
        ])
        want = [(1, 101, "H", 2), (3, 85, "L", 5), (6, 100, "H", 8)]
        self.assertEqual([key(p) for p in zigzag(b, None, mode="pct", pct=0.10)], want)
        # ATR values (even nan) are ignored in pct mode.
        self.assertEqual([key(p) for p in zigzag(b, [nan] * len(b), mode="pct", pct=0.10)], want)

    def test_nan_atr_warmup(self):
        b = make_bars([(20, 19), (12, 11), (11, 10), (10.5, 9.5), (12, 11), (13.5, 12.5)])
        a = [nan, nan, nan, nan, 1.0, 1.0]
        # Huge swings during warmup confirm nothing; the first valid bar (4) confirms the H.
        pivots = zigzag(b, a, mult=3.0)
        self.assertEqual([key(p) for p in pivots], [(0, 20, "H", 4)])
        self.assertEqual(zigzag(b, [nan] * len(b), mult=3.0), [])
        self.assertEqual(zigzag(b, [0.0] * len(b), mult=3.0), [])   # zero threshold is invalid

    def test_invalid_threshold_after_first_pivot(self):
        b = make_bars([(10, 9), (13, 12), (12, 5), (12, 5), (12, 9.5), (20, 9.6), (13, 10)])
        a = [1.0, 1.0, 0.0, nan, 1.0, 0.0, 1.0]
        self.assertEqual([key(p) for p in zigzag(b, a, mult=3.0)],
                         [(0, 9, "L", 1), (1, 13, "H", 4), (4, 9.5, "L", 6)])

    def test_both_conditions_confirm_older_extreme(self):
        # L (idx 0) older than H (idx 1)
        b = make_bars([(10, 9), (20, 19), (12, 11), (12, 11)])
        a = [nan, nan, nan, 1.0]
        self.assertEqual([key(p) for p in zigzag(b, a)], [(0, 9, "L", 3)])
        # H (idx 0) older than L (idx 1)
        b = make_bars([(20, 19), (12, 11), (15, 14)])
        a = [nan, nan, 1.0]
        self.assertEqual([key(p) for p in zigzag(b, a)], [(0, 20, "H", 2)])
        # Same-bar extremes (a tie) confirm the H.
        b = make_bars([(20, 10), (16, 14)])
        self.assertEqual([key(p) for p in zigzag(b, [nan, 1.0])], [(0, 20, "H", 1)])

    def test_matches_reference_on_random_walks(self):
        for seed in range(6):
            b = random_walk_bars(700, seed)
            av = atr(b.h, b.l, b.c, 14)
            for mode, mult, pct in (("atr", 2.0, 0.05), ("atr", 3.5, 0.05), ("pct", 3.0, 0.04)):
                got = [key(p) for p in zigzag(b, av, mult=mult, mode=mode, pct=pct)]
                self.assertEqual(got, ref_zigzag(b, av, mult, mode, pct), (seed, mode))
                self.assertGreater(len(got), 5)

    def test_structural_properties(self):
        for seed in range(5):
            b = random_walk_bars(1000, seed + 100)
            av = atr(b.h, b.l, b.c, 14)
            for kw in ({"mult": 2.5}, {"mode": "pct", "pct": 0.05}):
                ps = zigzag(b, av, **kw)
                self.assertGreater(len(ps), 10)
                for p in ps:
                    self.assertGreater(p.confirm_idx, p.idx)
                    self.assertEqual(p.t, b.t[p.idx])
                    self.assertEqual(p.price, b.h[p.idx] if p.kind == "H" else b.l[p.idx])
                for p, q in zip(ps, ps[1:]):
                    self.assertNotEqual(p.kind, q.kind)              # strict alternation
                    self.assertGreaterEqual(q.idx, p.confirm_idx)    # next extreme is after confirmation
                    self.assertGreater(q.confirm_idx, p.confirm_idx)
                    if p.kind == "H":
                        self.assertLess(q.price, p.price)
                    else:
                        self.assertGreater(q.price, p.price)

    def test_causality_prefix_property(self):
        """zigzag(bars[:k]) == pivots of zigzag(bars) with confirm_idx < k, for every k."""
        for seed, kw in ((7, {"mult": 2.0}), (8, {"mult": 3.0}), (9, {"mode": "pct", "pct": 0.04})):
            b = random_walk_bars(400, seed)
            full = zigzag(b, atr(b.h, b.l, b.c, 14), **kw)
            self.assertGreater(len(full), 5)
            for k in range(0, len(b) + 1):
                part = b.slice(0, k)
                got = zigzag(part, atr(part.h, part.l, part.c, 14), **kw)
                self.assertEqual(got, [p for p in full if p.confirm_idx < k], (seed, k))


if __name__ == "__main__":
    unittest.main()
