import math
import random
import unittest

from fibbot.indicators import adx, atr, ema, rsi, sma, true_range

nan = float("nan")


def random_ohlc(n, seed=1, start=100.0):
    """Seeded random-walk OHLC lists (h >= max(o, c), l <= min(o, c))."""
    rnd = random.Random(seed)
    o, h, l, c = [], [], [], []
    px = start
    for _ in range(n):
        op = px
        cl = op * math.exp(rnd.gauss(0, 0.02))
        o.append(op)
        c.append(cl)
        h.append(max(op, cl) * (1 + abs(rnd.gauss(0, 0.01))))
        l.append(min(op, cl) * (1 - abs(rnd.gauss(0, 0.01))))
        px = cl
    return o, h, l, c


# ---- straightforward textbook reference implementations (index based, no shortcuts) ----

def ref_sma(x, n):
    return [sum(x[i - n + 1:i + 1]) / n if i >= n - 1 else nan for i in range(len(x))]


def ref_ema(x, n):
    out = [nan] * len(x)
    if len(x) < n:
        return out
    a = 2 / (n + 1)
    out[n - 1] = sum(x[:n]) / n
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def ref_tr(h, l, c):
    return [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
                            for i in range(1, len(h))]


def ref_atr(h, l, c, n):
    tr = ref_tr(h, l, c)
    out = [nan] * len(h)
    if len(h) < n:
        return out
    out[n - 1] = sum(tr[:n]) / n
    for i in range(n, len(h)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def ref_rsi(c, n):
    """Wilder (1978): averages of the first n changes, then smoothed; RSI = 100 - 100/(1+RS)."""
    out = [nan] * len(c)
    if len(c) <= n:
        return out
    ag = al = None
    for i in range(1, len(c)):
        ch = c[i] - c[i - 1]
        g, q = max(ch, 0.0), max(-ch, 0.0)
        if i < n:
            continue
        if i == n:
            ag = sum(max(c[j] - c[j - 1], 0.0) for j in range(1, n + 1)) / n
            al = sum(max(c[j - 1] - c[j], 0.0) for j in range(1, n + 1)) / n
        else:
            ag = (ag * (n - 1) + g) / n
            al = (al * (n - 1) + q) / n
        if al == 0:
            out[i] = 50.0 if ag == 0 else 100.0
        else:
            out[i] = 100 - 100 / (1 + ag / al)
    return out


def ref_adx(h, l, c, n):
    """Wilder (1978) with running sums: TR14/+DM14/-DM14 -> DI -> DX -> ADX."""
    m = len(h)
    out = [nan] * m
    if m < 2 * n:
        return out
    tr, pdm, mdm = [0.0] * m, [0.0] * m, [0.0] * m
    for i in range(1, m):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    s_tr, s_p, s_m = sum(tr[1:n + 1]), sum(pdm[1:n + 1]), sum(mdm[1:n + 1])
    dxs, a = [], None
    for i in range(n, m):
        if i > n:
            s_tr = s_tr - s_tr / n + tr[i]
            s_p = s_p - s_p / n + pdm[i]
            s_m = s_m - s_m / n + mdm[i]
        pdi = 100 * s_p / s_tr if s_tr else 0.0
        mdi = 100 * s_m / s_tr if s_tr else 0.0
        dx = 100 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0
        dxs.append(dx)
        if len(dxs) == n:
            a = sum(dxs) / n
        elif len(dxs) > n:
            a = (a * (n - 1) + dx) / n
        if a is not None:
            out[i] = a
    return out


class Base(unittest.TestCase):
    def assertSeries(self, got, want, places=9):
        self.assertEqual(len(got), len(want))
        for i, (g, w) in enumerate(zip(got, want)):
            if math.isnan(w):
                self.assertTrue(math.isnan(g), f"index {i}: expected nan, got {g}")
            else:
                self.assertFalse(math.isnan(g), f"index {i}: unexpected nan")
                self.assertAlmostEqual(g, w, places=places, msg=f"index {i}")

    def first_valid(self, xs):
        return next(i for i, v in enumerate(xs) if not math.isnan(v))


class TestSmaEma(Base):
    def test_sma_hand(self):
        self.assertSeries(sma([1, 2, 3, 4, 5], 2), [nan, 1.5, 2.5, 3.5, 4.5])
        self.assertSeries(sma([3, 1, 2, 6], 3), [nan, nan, 2.0, 3.0])
        self.assertSeries(sma([4, 5], 1), [4.0, 5.0])

    def test_ema_hand(self):
        # alpha = 0.5, seed = mean(1,2,3) = 2 at index 2
        self.assertSeries(ema([1, 2, 3, 4, 5], 3), [nan, nan, 2.0, 3.0, 4.0])
        # alpha = 2/3, seed = 3 ; 2/3*6+1/3*3 = 5 ; 7 ; 9 ; 2/3*3+1/3*9 = 5
        self.assertSeries(ema([2, 4, 6, 8, 10, 3], 2), [nan, 3.0, 5.0, 7.0, 9.0, 5.0])
        self.assertSeries(ema([4, 7, 1], 1), [4.0, 7.0, 1.0])

    def test_short_and_empty(self):
        for f in (sma, ema):
            self.assertSeries(f([1, 2], 3), [nan, nan])
            self.assertEqual(f([], 3), [])
            with self.assertRaises(ValueError):
                f([1, 2, 3], 0)

    def test_against_reference(self):
        _, _, _, c = random_ohlc(600, seed=3)
        for n in (1, 2, 5, 14, 50, 200):
            self.assertSeries(sma(c, n), ref_sma(c, n), places=8)
            self.assertSeries(ema(c, n), ref_ema(c, n), places=8)

    def test_sma_long_series_no_drift(self):
        x = [60000.0 + (i % 97) * 13.7 for i in range(60000)]
        got = sma(x, 200)
        for i in (199, 30000, 59999):
            self.assertAlmostEqual(got[i], sum(x[i - 199:i + 1]) / 200, places=6)


class TestTrueRangeAtr(Base):
    H = [10, 12, 11, 15]
    L = [8, 9, 7, 12]
    C = [9, 11, 8, 14]

    def test_true_range_hand(self):
        # tr0 = 2 ; max(3, |12-9|, |9-9|) = 3 ; max(4, |11-11|, |7-11|) = 4 ; max(3, |15-8|, |12-8|) = 7
        self.assertSeries(true_range(self.H, self.L, self.C), [2, 3, 4, 7])
        self.assertEqual(true_range([], [], []), [])

    def test_atr_hand(self):
        # n=2: (2+3)/2 = 2.5 ; (2.5*1+4)/2 = 3.25 ; (3.25+7)/2 = 5.125
        self.assertSeries(atr(self.H, self.L, self.C, 2), [nan, 2.5, 3.25, 5.125])
        # n=3: mean(2,3,4) = 3 at index 2 ; (3*2+7)/3
        self.assertSeries(atr(self.H, self.L, self.C, 3), [nan, nan, 3.0, 13 / 3])
        self.assertSeries(atr(self.H, self.L, self.C, 5), [nan] * 4)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            true_range([1, 2], [1], [1, 2])
        with self.assertRaises(ValueError):
            adx([1, 2], [1, 2], [1], 2)

    def test_atr_against_reference(self):
        _, h, l, c = random_ohlc(500, seed=5)
        for n in (1, 3, 14, 30):
            self.assertSeries(atr(h, l, c, n), ref_atr(h, l, c, n))
            self.assertEqual(self.first_valid(atr(h, l, c, n)), n - 1)


class TestRsi(Base):
    def test_hand(self):
        # changes +1, -1, +1 ; n=2: ag=(1+0)/2, al=(0+1)/2 -> 50 ; ag=(0.5+1)/2=.75, al=.25 -> 75
        self.assertSeries(rsi([1, 2, 1, 2], 2), [nan, nan, 50.0, 75.0])

    def test_edge_rules(self):
        self.assertSeries(rsi([1, 2, 3, 4, 5], 3), [nan, nan, nan, 100.0, 100.0])  # no losses
        self.assertSeries(rsi([5, 5, 5, 5], 2), [nan, nan, 50.0, 50.0])            # no movement
        self.assertSeries(rsi([5, 4, 3, 2], 2), [nan, nan, 0.0, 0.0])              # no gains
        self.assertSeries(rsi([1, 2, 3], 3), [nan, nan, nan])                      # too short
        self.assertEqual(rsi([], 14), [])

    def test_against_reference(self):
        _, _, _, c = random_ohlc(800, seed=11)
        for n in (2, 7, 14, 30):
            got = rsi(c, n)
            self.assertSeries(got, ref_rsi(c, n))
            self.assertEqual(self.first_valid(got), n)
            self.assertTrue(all(0 <= v <= 100 for v in got[n:]))

    def test_default_period(self):
        _, _, _, c = random_ohlc(100, seed=2)
        self.assertSeries(rsi(c), ref_rsi(c, 14))


class TestAdx(Base):
    def test_pure_trends_are_100(self):
        n = 2
        h = [10, 11, 12, 13, 14, 15]
        l = [9, 10, 11, 12, 13, 14]
        c = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
        self.assertSeries(adx(h, l, c, n), [nan, nan, nan, 100.0, 100.0, 100.0])
        hd, ld, cd = h[::-1], l[::-1], c[::-1]
        self.assertSeries(adx(hd, ld, cd, n), [nan, nan, nan, 100.0, 100.0, 100.0])

    def test_flat_is_zero(self):
        self.assertSeries(adx([5] * 6, [5] * 6, [5] * 6, 2), [nan, nan, nan, 0.0, 0.0, 0.0])

    def test_hand_mixed(self):
        # n=2. +DM/-DM for bars 1..4: (2,0) (0,3) (1,0) (0,0) ; inside bar at 4.
        h = [10, 12, 11, 12, 11.5]
        l = [9, 10, 7, 8, 9]
        c = [9.5, 11, 8, 11, 10]
        # smoothed (avg) +DM: idx2 = 1, idx3 = (1+1)/2 = 1, idx4 = (1+0)/2 = .5
        # smoothed (avg) -DM: idx2 = 1.5, idx3 = .75, idx4 = .375
        # DX: idx2 = 100*.5/2.5 = 20 ; idx3 = 100*.25/1.75 ; idx4 = 100*.125/.875
        dx2, dx3, dx4 = 20.0, 100 * .25 / 1.75, 100 * .125 / .875
        a3 = (dx2 + dx3) / 2
        a4 = (a3 + dx4) / 2
        self.assertSeries(adx(h, l, c, 2), [nan, nan, nan, a3, a4])
        self.assertSeries(adx(h, l, c, 2), ref_adx(h, l, c, 2))

    def test_against_reference(self):
        _, h, l, c = random_ohlc(900, seed=17)
        for n in (2, 5, 14, 28):
            got = adx(h, l, c, n)
            self.assertSeries(got, ref_adx(h, l, c, n), places=8)
            self.assertEqual(self.first_valid(got), 2 * n - 1)
            self.assertTrue(all(0 <= v <= 100 for v in got[2 * n - 1:]))
        self.assertSeries(adx(h, l, c), ref_adx(h, l, c, 14), places=8)

    def test_short(self):
        _, h, l, c = random_ohlc(27, seed=1)
        self.assertTrue(all(math.isnan(v) for v in adx(h, l, c, 14)))
        self.assertEqual(adx([], [], [], 14), [])
        with self.assertRaises(ValueError):
            adx(h, l, c, 0)


class TestCausality(Base):
    def test_prefix_equals_full_prefix(self):
        _, h, l, c = random_ohlc(300, seed=23)
        full = {
            "sma": sma(c, 20), "ema": ema(c, 20), "tr": true_range(h, l, c),
            "atr": atr(h, l, c, 14), "rsi": rsi(c, 14), "adx": adx(h, l, c, 14),
        }
        for k in (1, 13, 14, 27, 28, 29, 150, 299):
            part = {
                "sma": sma(c[:k], 20), "ema": ema(c[:k], 20), "tr": true_range(h[:k], l[:k], c[:k]),
                "atr": atr(h[:k], l[:k], c[:k], 14), "rsi": rsi(c[:k], 14),
                "adx": adx(h[:k], l[:k], c[:k], 14),
            }
            for name, series in part.items():
                self.assertSeries(series, full[name][:k], places=12)


if __name__ == "__main__":
    unittest.main()
