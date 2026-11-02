"""Tests for fibbot.graduation: each check, each status and the validation lookup, on
constructed paper trade sets."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fibbot.config import DEFAULTS, deep_merge
from fibbot.graduation import evaluate, monthly_r, validation_mean_r
from fibbot.metrics import bootstrap_mean_ci, deflated_sharpe
from fibbot.store import Store
from fibbot.types import Trade

DAY = 86400
P = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())   # paper_start_t
SID = "g1"
START = DEFAULTS["paper"]["starting_equity"]
CHECKS = ["min_days", "min_trades", "min_profit_factor", "mean_r_ci_lower", "max_drawdown",
          "min_positive_month_frac", "min_deflated_sharpe", "max_shortfall_vs_validation_r"]
WINNING = [2.0, -1.0, 1.5, 2.0, -1.0, 1.0]          # mean R 0.75, PF 3.25
DEEP = [2.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0]     # one win, then five straight losses


def trade(i, r, exit_t, risk_pct=0.02, symbol="AAAUSDT"):
    ret = r * risk_pct
    return Trade(SID, symbol, "4h", 1, f"L{exit_t - 2 * DAY + i}", exit_t - 3 * DAY, exit_t - 2 * DAY,
                 100.0, 98.0, exit_t, 100.0 * (1 + ret), "target" if r > 0 else "stop", ret,
                 risk_pct, r, 12, {})


class GraduationTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(Path(tmp.name) / "fib.sqlite")
        self.addCleanup(self.store.close)
        self.cfg = deep_merge(DEFAULTS, {"home": tmp.name})

    def add(self, validation=None, sid=SID):
        self.store.add_to_roster(sid, "grad test", {"timeframe": "4h"}, P,
                                 {"metrics": {"avg_r": 0.6}} if validation is None else validation, "")

    def put(self, rs, days=None, notional=1000.0, curve=True, exits=None, risks=None):
        """Closed trades with R multiples rs (risk_pct risks[i], default 2%) exiting at `exits`
        or spread evenly over `days` from P + 1 day, plus the matching equity curve."""
        exits = exits or [int(P + DAY + i * days * DAY / len(rs)) for i in range(len(rs))]
        equity, points = START, [(P, START)]
        for i, r in enumerate(rs):
            tr = trade(i, r, exits[i], risk_pct=risks[i] if risks else 0.02)
            pnl = notional * tr.ret_pct
            self.store.upsert_paper_trade(SID, tr, notional, pnl)
            equity += pnl
            points.append((tr.exit_t, equity))
        if curve:
            self.store.set_equity_curve(SID, points)

    def evaluate(self, days):
        return evaluate(self.store, self.cfg, SID, now_t=P + days * DAY)

    def checks(self, res):
        return {c["name"]: c for c in res["checks"]}

    def failed(self, res):
        return {c["name"] for c in res["checks"] if not c["passed"]}


class TestStatuses(GraduationTestCase):
    def test_graduated_when_every_check_passes(self):
        self.add()
        self.put(WINNING * 10, days=85)
        res = self.evaluate(90)
        self.assertEqual(res["status"], "graduated", res["summary"])
        self.assertEqual([c["name"] for c in res["checks"]], CHECKS)
        self.assertEqual(self.failed(res), set())
        self.assertTrue(res["summary"].startswith("graduated: 8/8 checks passed; 60 closed trades"))
        c = self.checks(res)
        self.assertEqual((c["min_days"]["value"], c["min_days"]["threshold"]), (90.0, 60))
        self.assertEqual((c["min_trades"]["value"], c["min_trades"]["threshold"]), (60, 40))
        self.assertAlmostEqual(c["min_profit_factor"]["value"], 3.25, places=4)
        self.assertEqual(c["min_positive_month_frac"]["value"], 1.0)
        self.assertAlmostEqual(c["max_shortfall_vs_validation_r"]["value"], 0.75, places=6)
        self.assertAlmostEqual(c["max_shortfall_vs_validation_r"]["threshold"], 0.35, places=6)
        lo, hi = res["mean_r_ci"]
        self.assertGreater(lo, 0)
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.75)

    def test_collecting_when_days_or_trades_are_short(self):
        self.add()
        self.put(WINNING * 2, days=10)
        res = self.evaluate(12)
        self.assertEqual(res["status"], "collecting")
        self.assertTrue({"min_days", "min_trades"} <= self.failed(res))
        res = self.evaluate(200)                  # long enough, still too few trades
        self.assertEqual(res["status"], "collecting")
        self.assertEqual(self.failed(res) & {"min_days", "min_trades"}, {"min_trades"})

    def test_failing_on_losing_trades_once_half_the_trades_are_in(self):
        self.add()
        rs = [-1.0, -1.0, 0.5, -1.0, -0.8] * 4        # 20 trades = min_trades / 2
        self.put(rs, days=20, notional=100.0)
        res = self.evaluate(21)
        self.assertEqual(res["status"], "failing", res["summary"])
        self.assertLess(res["mean_r_ci"][1], 0)
        self.assertLess(self.checks(res)["min_profit_factor"]["value"], 1.0)

    def test_a_low_profit_factor_with_a_wide_ci_is_not_failing(self):
        self.add()
        self.put([2.5, -1.0, -1.0, -1.0] * 5, days=20, notional=100.0)   # PF 0.83, CI (-0.8, 0.6)
        res = self.evaluate(21)
        self.assertLess(self.checks(res)["min_profit_factor"]["value"], 1.0)
        self.assertGreater(res["mean_r_ci"][1], 0)
        self.assertEqual(res["status"], "collecting")

    def test_losing_but_too_few_trades_is_still_collecting(self):
        self.add()
        self.put([-1.0, -1.0, 0.5, -1.0, -0.8] * 3 + [-1.0] * 4, days=20, notional=100.0)   # 19 trades
        self.assertEqual(self.evaluate(21)["status"], "collecting")

    def test_failing_on_deep_drawdown_even_while_collecting(self):
        self.add()
        self.put(DEEP, days=10, notional=30000.0)
        # 600 per R (2% risk of a 30k notional): 10000 -> 11200 -> ... -> 8200 -> 8800; DD 0.268 > 0.225
        res = self.evaluate(12)
        dd = self.checks(res)["max_drawdown"]["value"]
        self.assertAlmostEqual(dd, (11200 - 8800 + 600) / 11200, places=5)
        self.assertGreater(dd, 1.5 * self.cfg["graduation"]["max_drawdown"])
        self.assertEqual(res["status"], "failing")

    def test_promising_when_mature_but_not_proven(self):
        self.add()
        noisy = [3.0, -1.0, -1.0, -1.0, 2.5, -1.0, -1.0, 1.0, -1.0, 0.8]   # mean 0.03, CI spans 0
        self.put(noisy * 5, days=85)
        res = self.evaluate(90)
        self.assertEqual(res["status"], "promising", res["summary"])
        self.assertIn("mean_r_ci_lower", self.failed(res))
        self.assertNotIn("min_trades", self.failed(res))
        self.assertIn("not yet: ", res["summary"])


class TestSingleCheckFailures(GraduationTestCase):
    """A graduating record with exactly one check broken is "promising"."""

    def assertOnlyFails(self, name, days=90):
        res = self.evaluate(days)
        self.assertEqual(self.failed(res), {name}, res["summary"])
        self.assertEqual(res["status"], "promising")

    def test_profit_factor(self):
        # same R multiples, but the losers risked 3% of the notional and the winners 1%: PF 1.08
        self.add()
        rs = WINNING * 10
        self.put(rs, days=85, risks=[0.01 if r > 0 else 0.03 for r in rs])
        self.assertOnlyFails("min_profit_factor")

    def test_positive_months(self):
        self.add()
        day = lambda m, d: int(datetime(2026, m, d, tzinfo=timezone.utc).timestamp())
        rs = WINNING * 5 + [-0.5, -0.5] + WINNING * 5 + [-0.5]
        exits = ([day(1, 2) + i * DAY for i in range(30)] + [day(2, 10), day(2, 12)]
                 + [day(3, 1) + i * DAY for i in range(30)] + [day(4, 5)])
        self.put(rs, exits=exits)
        self.assertOnlyFails("min_positive_month_frac", days=100)
        self.assertEqual(self.checks(self.evaluate(100))["min_positive_month_frac"]["value"], 0.5)

    def test_deflated_sharpe(self):
        # the same record is no longer convincing once 200 strategies have been paper traded
        self.add()
        self.put(WINNING * 10, days=85)
        for k in range(199):
            self.add(sid=f"other{k}")
        self.assertOnlyFails("min_deflated_sharpe")


class TestChecks(GraduationTestCase):
    def test_validation_shortfall(self):
        self.add(validation={"avg_r": 1.5, "metrics": {"n": 40}})
        self.put(WINNING * 10, days=85)
        res = self.evaluate(90)
        c = self.checks(res)["max_shortfall_vs_validation_r"]
        self.assertFalse(c["passed"])
        self.assertAlmostEqual(c["threshold"], 1.25, places=6)
        self.assertEqual(res["status"], "promising")

    def test_missing_validation_mean_passes_with_a_note(self):
        self.add(validation={"passed": True})
        self.put(WINNING * 10, days=85)
        res = self.evaluate(90)
        c = self.checks(res)["max_shortfall_vs_validation_r"]
        self.assertEqual((c["value"], c["threshold"], c["passed"]), (None, None, True))
        self.assertIn("no validation mean R", c["note"])
        self.assertEqual(res["status"], "graduated")

    def test_validation_mean_r_lookup(self):
        self.assertEqual(validation_mean_r({"metrics": {"avg_r": 0.3}, "avg_r": 0.9}), 0.3)
        self.assertEqual(validation_mean_r({"metrics": {"n": 3}, "avg_r": 1}), 1.0)
        self.assertEqual(validation_mean_r({"metrics": None, "avg_r": 0.2}), 0.2)
        for bad in (None, {}, {"avg_r": float("nan")}, {"avg_r": True}, {"metrics": {"avg_r": "x"}},
                    {"avg_r": float("inf")}, [0.3]):
            self.assertIsNone(validation_mean_r(bad), bad)

    def test_positive_months_count_only_months_with_trades(self):
        self.add()
        jan, mar, apr = (int(datetime(2026, m, 10, tzinfo=timezone.utc).timestamp()) for m in (1, 3, 4))
        for i, (r, t) in enumerate([(1.0, jan), (0.5, jan), (-2.0, mar), (1.5, mar), (0.5, apr)]):
            self.store.upsert_paper_trade(SID, trade(i, r, t), 100.0, r)
        self.assertEqual(monthly_r(self.store.trade_from_row(r) for r in self.store.paper_trades(SID)),
                         {"2026-01": 1.5, "2026-03": -0.5, "2026-04": 0.5})
        self.assertAlmostEqual(self.checks(self.evaluate(120))["min_positive_month_frac"]["value"],
                               0.666667, places=6)

    def test_drawdown_falls_back_to_cumulative_pnl_without_a_curve(self):
        self.add()
        self.put(DEEP, days=10, notional=30000.0, curve=False)
        self.assertEqual(self.store.equity_curve(SID), [])
        dd = self.checks(self.evaluate(12))["max_drawdown"]["value"]
        self.assertAlmostEqual(dd, (11200 - 8800 + 600) / 11200, places=5)

    def test_open_positions_count_in_the_drawdown_but_not_in_trade_checks(self):
        self.add()
        self.put(WINNING * 10, days=85)
        open_tr = Trade(SID, "BBBUSDT", "4h", 1, "Lopen", P, P + 86 * DAY, 100.0, 98.0, None, 80.0,
                        "open", -0.2, 0.02, -10.0, 5, {})
        self.store.upsert_paper_trade(SID, open_tr, 10000.0, -2000.0)
        res = self.evaluate(90)
        c = self.checks(res)
        self.assertEqual(c["min_trades"]["value"], 60)
        self.assertGreater(c["max_drawdown"]["value"], 0.15)
        self.assertEqual(res["status"], "promising")

    def test_deflated_sharpe_uses_every_strategy_ever_on_paper(self):
        self.add()
        self.put(WINNING * 10, days=85)
        rs = WINNING * 10
        alone = self.checks(self.evaluate(90))["min_deflated_sharpe"]["value"]
        self.assertAlmostEqual(alone, deflated_sharpe(rs, n_trials=1), places=5)
        for k in range(30):
            self.add(sid=f"other{k}")
        crowded = self.checks(self.evaluate(90))["min_deflated_sharpe"]["value"]
        self.assertAlmostEqual(crowded, deflated_sharpe(rs, n_trials=31), places=5)
        self.assertLess(crowded, alone)

    def test_ci_level_comes_from_config(self):
        self.add()
        self.put(WINNING * 10, days=85)
        self.cfg["graduation"]["ci_level"] = 0.5
        res = self.evaluate(90)
        lo, hi = bootstrap_mean_ci(WINNING * 10, level=0.5)
        self.assertAlmostEqual(res["mean_r_ci"][0], lo, places=5)
        self.assertAlmostEqual(res["mean_r_ci"][1], hi, places=5)

    def test_no_trades_and_unknown_strategy(self):
        self.add()
        res = self.evaluate(0)
        self.assertEqual(res["status"], "collecting")
        months = self.checks(res)["min_positive_month_frac"]
        self.assertEqual((months["value"], months["passed"]), (0.0, False))
        self.assertEqual(res["mean_r_ci"], [0.0, 0.0])
        self.assertEqual(self.checks(res)["max_drawdown"]["value"], 0.0)
        with self.assertRaises(KeyError):
            evaluate(self.store, self.cfg, "nope", now_t=P)


if __name__ == "__main__":
    unittest.main()
