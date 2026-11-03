import math
import random
import statistics
import unittest

from fibbot.metrics import (CAP, bootstrap_mean_ci, daily_closes, deflated_sharpe, equity_stats,
                            expected_max_sharpe, max_drawdown, monthly_returns, probabilistic_sharpe,
                            skew_kurt, t_stat_mean, trade_stats)
from fibbot.types import Trade

DAY = 86400
D0 = 1_704_067_200          # 2024-01-01 00:00:00 UTC
G = 0.5772156649


def phi(z):
    """Standard normal CDF via erf (independent of statistics.NormalDist)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def phi_inv(p):
    """Inverse standard normal CDF by bisection on erf."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


def gauss_sample(n, mu, sd, seed):
    rnd = random.Random(seed)
    return [rnd.gauss(mu, sd) for _ in range(n)]


def mk(r, i=0, risk=0.01, side=1, reason="target", bars=5, ret=None):
    """Closed trade with r_multiple r (ret_pct = r * risk unless given)."""
    ret = r * risk if ret is None else ret
    return Trade("s1", "BTCUSDT", "4h", side, f"L{i}", D0 + i * 3600, D0 + i * 3600, 100.0, 99.0,
                 D0 + (i + 1) * 3600, 100.0 * (1 + ret), reason, ret, risk, r, bars)


class TradeStatsTest(unittest.TestCase):
    RS = [2.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0, 0.5]

    def trades(self):
        return [mk(r, i, side=1 if i % 3 else -1, reason="target" if r > 0 else "stop", bars=i + 1)
                for i, r in enumerate(self.RS)]

    def test_empty(self):
        s = trade_stats([])
        self.assertEqual(s["exit_reasons"], {})
        for k, v in s.items():
            if k != "exit_reasons":
                self.assertEqual(v, 0, k)
        self.assertIn("expectancy_r", s)

    def test_known_sample(self):
        s = trade_stats(self.trades())
        self.assertEqual(s["n"], 8)
        self.assertEqual(s["wins"], 3)
        self.assertAlmostEqual(s["win_rate"], 3 / 8)
        self.assertAlmostEqual(s["avg_r"], 0.5 / 8)
        self.assertEqual(s["avg_r"], s["expectancy_r"])
        self.assertAlmostEqual(s["median_r"], -1.0)
        self.assertAlmostEqual(s["avg_win_r"], 5.5 / 3)
        self.assertAlmostEqual(s["avg_loss_r"], -1.0)
        self.assertAlmostEqual(s["total_r"], 0.5)
        self.assertAlmostEqual(s["profit_factor"], 0.055 / 0.05)
        self.assertAlmostEqual(s["avg_ret_pct"], 0.005 / 8)
        self.assertAlmostEqual(s["avg_bars_held"], 4.5)
        self.assertEqual(s["max_consec_losses"], 3)
        self.assertEqual((s["long_n"], s["short_n"]), (5, 3))
        self.assertEqual(s["exit_reasons"], {"target": 3, "stop": 5})
        mean = sum(self.RS) / 8
        sd = math.sqrt(sum((r - mean) ** 2 for r in self.RS) / 7)
        self.assertAlmostEqual(s["sharpe_per_trade"], mean / sd)

    def test_streak_uses_entry_order(self):
        shuffled = self.trades()
        random.Random(3).shuffle(shuffled)
        self.assertEqual(trade_stats(shuffled)["max_consec_losses"], 3)
        self.assertEqual(trade_stats(shuffled), trade_stats(self.trades()))

    def test_profit_factor_edges(self):
        self.assertEqual(trade_stats([mk(1.0, 0), mk(2.0, 1)])["profit_factor"], 99.0)
        self.assertEqual(trade_stats([mk(-1.0, 0), mk(-0.5, 1)])["profit_factor"], 0.0)
        self.assertEqual(trade_stats([mk(5.0, 0), mk(-1e-6, 1)])["profit_factor"], CAP)

    def test_zero_return_is_a_loss_for_streaks_not_for_pf(self):
        s = trade_stats([mk(1.0, 0), mk(0.0, 1), mk(-1.0, 2)])
        self.assertEqual(s["wins"], 1)
        self.assertEqual(s["max_consec_losses"], 2)
        self.assertAlmostEqual(s["profit_factor"], 1.0)
        self.assertAlmostEqual(s["avg_loss_r"], -0.5)

    def test_sharpe_needs_two_trades(self):
        self.assertEqual(trade_stats([mk(1.0)])["sharpe_per_trade"], 0.0)
        self.assertEqual(trade_stats([mk(1.0, 0), mk(1.0, 1)])["sharpe_per_trade"], CAP)


class DrawdownTest(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(max_drawdown([100, 120, 90, 130, 117]), 0.25)
        self.assertEqual(max_drawdown([1, 2, 3]), 0.0)
        self.assertEqual(max_drawdown([]), 0.0)
        self.assertAlmostEqual(max_drawdown([100, 50, 25]), 0.75)


class EquityStatsTest(unittest.TestCase):
    CURVE = [(D0, 100.0), (D0 + 3600, 101.0), (D0 + 80000, 99.0), (D0 + 2 * DAY + 10, 102.0)]

    def test_daily_resampling_last_point_and_forward_fill(self):
        self.assertEqual(daily_closes(self.CURVE), [99.0, 99.0, 102.0])
        self.assertEqual(daily_closes(list(reversed(self.CURVE))), [99.0, 99.0, 102.0])
        self.assertEqual(daily_closes([]), [])
        same_t = [(D0, 1.0), (D0, 2.0), (D0, 3.0)]
        self.assertEqual(daily_closes(same_t), [3.0])

    def test_hand_computed(self):
        s = equity_stats(self.CURVE, 100.0)
        rets = [-0.01, 0.0, 102.0 / 99.0 - 1]
        mean = sum(rets) / 3
        sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / 2)
        down = math.sqrt(0.01 ** 2 / 3)
        cagr = 1.02 ** (365 / 3) - 1
        mdd = (101 - 99) / 101       # raw intraday points count, not just daily closes
        self.assertEqual(s["n_days"], 3)
        self.assertAlmostEqual(s["total_return"], 0.02)
        self.assertAlmostEqual(s["cagr"], cagr)
        self.assertAlmostEqual(s["max_drawdown"], mdd)
        self.assertAlmostEqual(s["sharpe"], mean / sd * math.sqrt(365))
        self.assertAlmostEqual(s["sortino"], mean / down * math.sqrt(365))
        self.assertEqual(s["calmar"], min(CAP, cagr / mdd))     # short span -> huge CAGR, capped

    def test_empty_and_degenerate(self):
        zero = {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "calmar": 0.0, "n_days": 0}
        self.assertEqual(equity_stats([], 100.0), zero)
        self.assertEqual(equity_stats(self.CURVE, 0.0), zero)
        one_day = equity_stats([(D0, 100.0), (D0 + 60, 110.0)], 100.0)
        self.assertEqual(one_day["n_days"], 1)
        self.assertEqual(one_day["sharpe"], 0.0)
        self.assertAlmostEqual(one_day["total_return"], 0.1)

    def test_only_gains_caps_ratios(self):
        s = equity_stats([(D0, 100.0), (D0 + DAY, 101.0), (D0 + 2 * DAY, 103.0)], 100.0)
        self.assertEqual(s["max_drawdown"], 0.0)
        self.assertEqual(s["sortino"], CAP)
        self.assertEqual(s["calmar"], CAP)
        self.assertGreater(s["sharpe"], 0)

    def test_annual_curve(self):
        curve = [(D0 + d * DAY, 100.0 * 1.001 ** d) for d in range(365)]
        s = equity_stats(curve, 100.0)
        self.assertEqual(s["n_days"], 365)
        self.assertAlmostEqual(s["cagr"], s["total_return"], places=9)

    def test_ruin(self):
        s = equity_stats([(D0, 100.0), (D0 + DAY, 0.0)], 100.0)
        self.assertEqual(s["cagr"], -1.0)
        self.assertAlmostEqual(s["max_drawdown"], 1.0)


class MonthlyReturnsTest(unittest.TestCase):
    def test_gap_month_forward_filled(self):
        curve = [(D0 + 14 * DAY, 105.0), (D0 + 30 * DAY, 110.0), (D0 + 69 * DAY, 99.0)]
        out = monthly_returns(curve, 100.0)
        self.assertEqual([m for m, _ in out], ["2024-01", "2024-02", "2024-03"])
        self.assertAlmostEqual(out[0][1], 0.10)
        self.assertEqual(out[1][1], 0.0)
        self.assertAlmostEqual(out[2][1], 99.0 / 110.0 - 1)

    def test_year_rollover(self):
        dec = 1_733_011_200      # 2024-12-01 UTC
        out = monthly_returns([(dec, 100.0), (dec + 40 * DAY, 120.0)], 100.0)
        self.assertEqual([m for m, _ in out], ["2024-12", "2025-01"])
        self.assertEqual(out[0][1], 0.0)
        self.assertAlmostEqual(out[1][1], 0.2)

    def test_empty(self):
        self.assertEqual(monthly_returns([], 100.0), [])


class BootstrapTest(unittest.TestCase):
    VALUES = gauss_sample(120, 0.2, 1.0, seed=11)

    def test_deterministic(self):
        a = bootstrap_mean_ci(self.VALUES)
        self.assertEqual(a, bootstrap_mean_ci(list(self.VALUES)))
        self.assertEqual(bootstrap_mean_ci(self.VALUES, seed=3), bootstrap_mean_ci(self.VALUES, seed=3))
        self.assertNotEqual(a, bootstrap_mean_ci(self.VALUES, seed=8))

    def test_covers_mean_and_matches_normal_theory(self):
        lo, hi = bootstrap_mean_ci(self.VALUES, n_boot=4000)
        mean = statistics.fmean(self.VALUES)
        half = 1.96 * statistics.stdev(self.VALUES) / math.sqrt(len(self.VALUES))
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        self.assertAlmostEqual(lo, mean - half, delta=0.25 * half)
        self.assertAlmostEqual(hi, mean + half, delta=0.25 * half)

    def test_level_widens(self):
        lo90, hi90 = bootstrap_mean_ci(self.VALUES, level=0.90)
        lo99, hi99 = bootstrap_mean_ci(self.VALUES, level=0.99)
        self.assertLess(lo99, lo90)
        self.assertGreater(hi99, hi90)

    def test_edges(self):
        self.assertEqual(bootstrap_mean_ci([]), (0.0, 0.0))
        self.assertEqual(bootstrap_mean_ci([0.5]), (0.5, 0.5))
        self.assertEqual(bootstrap_mean_ci([2.0] * 10), (2.0, 2.0))


class MomentsTest(unittest.TestCase):
    def test_known_samples(self):
        sk, ku = skew_kurt([1, 2, 3])
        self.assertAlmostEqual(sk, 0.0)
        self.assertAlmostEqual(ku, 1.5)
        sk, ku = skew_kurt([0, 0, 0, 1])          # Bernoulli(0.25)
        self.assertAlmostEqual(sk, 2 / math.sqrt(3))
        self.assertAlmostEqual(ku, 7 / 3)
        sk, ku = skew_kurt([0, 1, 1, 1])
        self.assertAlmostEqual(sk, -2 / math.sqrt(3))
        self.assertAlmostEqual(ku, 7 / 3)

    def test_degenerate(self):
        self.assertEqual(skew_kurt([]), (0.0, 3.0))
        self.assertEqual(skew_kurt([4.0]), (0.0, 3.0))
        self.assertEqual(skew_kurt([0.1, 0.1, 0.1]), (0.0, 3.0))

    def test_normal_sample(self):
        rnd = random.Random(5)
        sk, ku = skew_kurt([rnd.gauss(0, 1) for _ in range(20000)])
        self.assertAlmostEqual(sk, 0.0, delta=0.06)
        self.assertAlmostEqual(ku, 3.0, delta=0.12)

    def test_t_stat(self):
        self.assertAlmostEqual(t_stat_mean([1, 2, 3, 4, 5]), 3 * math.sqrt(2))
        self.assertEqual(t_stat_mean([1.0]), 0.0)
        self.assertEqual(t_stat_mean([]), 0.0)


class SharpeInferenceTest(unittest.TestCase):
    """PSR / DSR against hand-computed Bailey & Lopez de Prado values."""

    E10 = (1 - G) * phi_inv(0.9) + G * phi_inv(1 - 1 / (10 * math.e))

    def test_psr_hand_computed(self):
        # skew 0, kurt 1 -> radicand 1 ; z = 0.1 * sqrt(100) = 1
        self.assertAlmostEqual(probabilistic_sharpe(0.1, 101, 0.0, 1.0), 0.8413447460685429, places=12)
        # radicand = 1 + 0.5*0.2 + (5-1)/4*0.04 = 1.14 ; z = 0.15*7/sqrt(1.14)
        z = 0.15 * 7 / math.sqrt(1.14)
        self.assertAlmostEqual(probabilistic_sharpe(0.2, 50, -0.5, 5.0, sr_star=0.05), phi(z), places=12)
        self.assertAlmostEqual(probabilistic_sharpe(0.3, 40, 0.0, 3.0, sr_star=0.3), 0.5)

    def test_psr_guards(self):
        # radicand 1 - 2*2 + 0 < 0 -> 1e-12 guard -> |z| huge
        self.assertEqual(probabilistic_sharpe(2.0, 10, 2.0, 1.0), 1.0)
        self.assertEqual(probabilistic_sharpe(-2.0, 10, -2.0, 1.0), 0.0)
        self.assertEqual(probabilistic_sharpe(0.5, 1, 0.0, 3.0), 0.5)

    def test_expected_max_sharpe(self):
        self.assertEqual(expected_max_sharpe(1, 1.0), 0.0)
        self.assertEqual(expected_max_sharpe(0, 1.0), 0.0)
        self.assertEqual(expected_max_sharpe(10, 0.0), 0.0)
        self.assertAlmostEqual(expected_max_sharpe(10, 1.0), self.E10, places=9)
        self.assertAlmostEqual(expected_max_sharpe(10, 4.0), 2 * self.E10, places=9)
        # close to the exact expected maximum of N standard normals
        self.assertAlmostEqual(expected_max_sharpe(10, 1.0), 1.538753, delta=0.05)
        self.assertAlmostEqual(expected_max_sharpe(100, 1.0), 2.507594, delta=0.05)

    def test_deflated_sharpe_hand_computed(self):
        xs = [2.0, 0.0, 2.0, 0.0]       # SR = 1/sqrt(4/3) = sqrt(3)/2, skew 0, kurt 1, n = 4
        sr = math.sqrt(3) / 2
        self.assertAlmostEqual(deflated_sharpe(xs, n_trials=1), phi(1.5), places=12)
        sr0 = math.sqrt(0.25) * self.E10        # default var_sr = 1/n
        self.assertAlmostEqual(deflated_sharpe(xs, n_trials=10), phi((sr - sr0) * math.sqrt(3)), places=9)
        sr0 = 0.1 * self.E10
        self.assertAlmostEqual(deflated_sharpe(xs, 10, var_sr=0.01), phi((sr - sr0) * math.sqrt(3)), places=9)
        self.assertEqual(deflated_sharpe(xs, 10, var_sr=0.0), deflated_sharpe(xs, 10))

    def test_deflated_sharpe_behaviour(self):
        self.assertEqual(deflated_sharpe([1.0, 2.0], 5), 0.0)
        rnd = random.Random(2)
        xs = [rnd.gauss(0.15, 1.0) for _ in range(200)]
        dsr = [deflated_sharpe(xs, n) for n in (1, 10, 100, 1000)]
        self.assertEqual(dsr, sorted(dsr, reverse=True))
        sk, ku = skew_kurt(xs)
        sr = statistics.fmean(xs) / statistics.stdev(xs)
        self.assertAlmostEqual(dsr[0], probabilistic_sharpe(sr, 200, sk, ku))


if __name__ == "__main__":
    unittest.main()
