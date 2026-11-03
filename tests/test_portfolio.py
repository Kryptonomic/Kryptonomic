import unittest

from fibbot.config import DEFAULTS
from fibbot.portfolio import PortfolioResult, simulate_portfolio
from fibbot.types import Trade

DAY = 86400
D0 = 1_704_067_200          # 2024-01-01 00:00:00 UTC
# max_total_notional is raised above what these fixtures can reach (3 x 0.5): the tests below
# are about the other rules; the no-leverage cap (default 1x) has its own tests
PCFG = {"starting_equity": 10000.0, "risk_per_trade": 0.01, "max_open_positions": 3,
        "max_total_risk": 0.05, "max_notional_frac": 0.5, "daily_loss_limit": 0.02,
        "max_total_notional": 2.0}


def cfg(**kw):
    return {**PCFG, **kw}


def mk(entry_t, exit_t, ret_pct, risk_pct=0.02, symbol="BTCUSDT", side=1):
    """Trade with only the fields the portfolio uses set meaningfully."""
    return Trade("s1", symbol, "1h", side, f"L{entry_t}", entry_t, entry_t, 100.0, 98.0, exit_t,
                 100.0 * (1 + ret_pct), "open" if exit_t is None else "target", ret_pct, risk_pct,
                 ret_pct / risk_pct, 1)


def reasons(res):
    return [(s["trade"].symbol, s["reason"]) for s in res.skipped]


class SizingTest(unittest.TestCase):
    def test_risk_based_notional(self):
        res = simulate_portfolio([mk(D0, D0 + 3600, 0.05, risk_pct=0.04)], PCFG)
        rec = res.taken[0]
        self.assertAlmostEqual(rec["notional"], 100.0 / 0.04)       # 1% of 10k at risk / 4% stop
        self.assertAlmostEqual(rec["pnl"], 2500.0 * 0.05)
        self.assertAlmostEqual(res.final_equity, 10125.0)
        self.assertAlmostEqual(res.realized_equity, 10125.0)

    def test_capped_by_max_notional_frac(self):
        res = simulate_portfolio([mk(D0, D0 + 3600, 0.01, risk_pct=0.005)], PCFG)
        self.assertAlmostEqual(res.taken[0]["notional"], 5000.0)    # 100/0.005 = 20k > 50% of 10k

    def test_sized_from_realized_equity(self):
        trades = [mk(D0, D0 + 3600, 0.10, symbol="A"),             # +500 on 5000 notional
                  mk(D0 + 2 * 3600, D0 + 3 * 3600, 0.0, risk_pct=0.04, symbol="B")]
        res = simulate_portfolio(trades, PCFG)
        self.assertAlmostEqual(res.taken[1]["notional"], 0.01 * 10500.0 / 0.04)

    def test_start_equity_default_and_override(self):
        trades = [mk(D0, D0 + 3600, 0.02, risk_pct=0.04)]
        self.assertEqual(simulate_portfolio(trades, PCFG).start_equity, 10000.0)
        res = simulate_portfolio(trades, PCFG, start_equity=2000.0)
        self.assertEqual(res.start_equity, 2000.0)
        self.assertAlmostEqual(res.taken[0]["notional"], 20.0 / 0.04)

    def test_partial_config_uses_defaults(self):
        res = simulate_portfolio([mk(D0, D0 + 3600, 0.01, risk_pct=0.04)], {"risk_per_trade": 0.02})
        eq = DEFAULTS["portfolio"]["starting_equity"]
        self.assertAlmostEqual(res.taken[0]["notional"], min(0.02 * eq / 0.04, 0.25 * eq))


class LimitsTest(unittest.TestCase):
    def test_max_open_positions(self):
        trades = [mk(D0 + i, D0 + 100, 0.01, risk_pct=0.04, symbol=s) for i, s in enumerate("ABCD")]
        trades.append(mk(D0 + 200, D0 + 300, 0.01, risk_pct=0.04, symbol="E"))   # after the exits
        res = simulate_portfolio(trades, PCFG)
        self.assertEqual([r["trade"].symbol for r in res.taken], ["A", "B", "C", "E"])
        self.assertEqual(reasons(res), [("D", "max_open")])

    def test_max_total_risk(self):
        pc = cfg(max_total_risk=0.025, max_open_positions=10, max_total_notional=10.0)
        trades = [mk(D0, D0 + 100, 0.0, symbol="A"), mk(D0 + 1, D0 + 100, 0.0, symbol="B"),
                  mk(D0 + 2, D0 + 100, 0.0, symbol="C"),                     # 3 x 100 > 250
                  mk(D0 + 3, D0 + 100, 0.0, risk_pct=0.001, symbol="D")]     # capped: 5000*0.001 = 5
        res = simulate_portfolio(trades, pc)
        self.assertEqual(reasons(res), [("C", "max_total_risk")])
        self.assertEqual([r["trade"].symbol for r in res.taken], ["A", "B", "D"])

    def test_total_risk_boundary_is_inclusive(self):
        pc = cfg(max_total_risk=0.03, max_open_positions=10)
        trades = [mk(D0 + i, D0 + 100, 0.0, risk_pct=0.04, symbol=s) for i, s in enumerate("ABC")]
        self.assertEqual(len(simulate_portfolio(trades, pc).taken), 3)       # exactly 300 == 3%

    def test_daily_loss_limit(self):
        loss = mk(D0 + 100, D0 + 200, -0.04, symbol="A")                     # -200 = -2% of 10k
        same_day = mk(D0 + 300, D0 + 400, 0.01, symbol="B")
        next_day = mk(D0 + DAY + 10, D0 + DAY + 20, 0.01, symbol="C")
        res = simulate_portfolio([loss, same_day, next_day], PCFG)
        self.assertEqual(reasons(res), [("B", "daily_loss_limit")])
        self.assertEqual([r["trade"].symbol for r in res.taken], ["A", "C"])
        smaller = mk(D0 + 100, D0 + 200, -0.0398, symbol="A")                # -199 > -200
        self.assertEqual(simulate_portfolio([smaller, same_day], PCFG).skipped, [])
        self.assertEqual(simulate_portfolio([loss, same_day], cfg(daily_loss_limit=0)).skipped, [])

    def test_daily_loss_resets_at_utc_midnight(self):
        late_loss = mk(D0 + DAY - 500, D0 + DAY - 10, -0.06, symbol="A")
        after_midnight = mk(D0 + DAY + 10, D0 + DAY + 20, 0.01, symbol="B")
        res = simulate_portfolio([late_loss, after_midnight], PCFG)
        self.assertEqual(res.skipped, [])

    def test_daily_limit_uses_equity_at_days_first_event(self):
        # day 1 opens at 12000 equity: -200 is only -1.67%, so trading continues
        win = mk(D0, D0 + 100, 0.40, symbol="A")                              # +2000
        loss = mk(D0 + DAY + 10, D0 + DAY + 20, -200 / 6000, symbol="B")      # notional 6000 -> -200
        after = mk(D0 + DAY + 30, D0 + DAY + 40, 0.0, symbol="C")
        res = simulate_portfolio([win, loss, after], PCFG)
        self.assertAlmostEqual(res.taken[1]["pnl"], -200.0)
        self.assertEqual(res.skipped, [])

    def test_no_equity(self):
        res = simulate_portfolio([mk(D0, D0 + 100, 0.01)], PCFG, start_equity=0.0)
        self.assertEqual(reasons(res), [("BTCUSDT", "no_equity")])
        wipeout = mk(D0, D0 + 100, -2.5, symbol="A")                          # 5000 * -2.5 = -12500
        res = simulate_portfolio([wipeout, mk(D0 + DAY, D0 + DAY + 5, 0.01, symbol="B")], PCFG)
        self.assertLess(res.realized_equity, 0)
        self.assertEqual(reasons(res), [("B", "no_equity")])


class OrderingTest(unittest.TestCase):
    def test_entries_before_exits_at_equal_timestamps(self):
        # F41: B's order was placed at the close of the bar before D0+3600, when A still held the
        # only slot; A's exit happened somewhere inside B's fill bar. B cannot use A's slot or
        # A's profit (the old exits-first order did both).
        a = mk(D0, D0 + 3600, 0.10, symbol="A")
        b = mk(D0 + 3600, D0 + 7200, 0.0, risk_pct=0.04, symbol="B")
        res = simulate_portfolio([b, a], cfg(max_open_positions=1))
        self.assertEqual(reasons(res), [("B", "max_open")])
        res = simulate_portfolio([b, a], cfg(max_open_positions=2))
        self.assertEqual(res.skipped, [])
        self.assertAlmostEqual(res.taken[1]["notional"], 0.01 * 10000.0 / 0.04)  # before A's +500
        c = mk(D0 + 7200, D0 + 9000, 0.0, risk_pct=0.04, symbol="C")           # the bar after
        res = simulate_portfolio([b, a, c], cfg(max_open_positions=1))
        self.assertEqual([r["trade"].symbol for r in res.taken], ["A", "C"])
        self.assertAlmostEqual(res.taken[1]["notional"], 0.01 * 10500.0 / 0.04)

    def test_same_bar_exit_holds_slot_during_that_timestamp(self):
        pc = cfg(max_open_positions=1)
        a = mk(D0, D0, -0.01, symbol="A")              # stopped out on its entry bar
        b = mk(D0, D0 + 100, 0.01, symbol="B")         # same entry time -> slot still taken by A
        c = mk(D0 + 50, D0 + 100, 0.01, symbol="C")    # later -> A is closed
        res = simulate_portfolio([c, b, a], pc)
        self.assertEqual(reasons(res), [("B", "max_open")])
        self.assertEqual([r["trade"].symbol for r in res.taken], ["A", "C"])
        self.assertEqual(res.curve[:2], [(D0, 10000.0), (D0, 9950.0)])

    def test_entries_at_equal_time_ordered_by_symbol(self):
        pc = cfg(max_open_positions=1)
        res = simulate_portfolio([mk(D0, D0 + 10, 0.0, symbol="ETHUSDT"),
                                  mk(D0, D0 + 10, 0.0, symbol="BTCUSDT")], pc)
        self.assertEqual(res.taken[0]["trade"].symbol, "BTCUSDT")
        self.assertEqual(reasons(res), [("ETHUSDT", "max_open")])

    def test_input_order_does_not_matter(self):
        trades = [mk(D0 + i * 900, D0 + i * 900 + 5000, (-1) ** i * 0.02, symbol=f"S{i}")
                  for i in range(12)]
        a = simulate_portfolio(trades, PCFG)
        b = simulate_portfolio(list(reversed(trades)), PCFG)
        self.assertEqual(a.curve, b.curve)
        self.assertEqual(a.final_equity, b.final_equity)
        self.assertEqual(reasons(a), reasons(b))


class AccountingTest(unittest.TestCase):
    def test_open_trades_marked_not_realized(self):
        closed = mk(D0, D0 + 100, 0.02, risk_pct=0.04, symbol="A")         # 2500 notional, +50
        still_open = mk(D0 + 50, None, 0.04, risk_pct=0.04, symbol="B")    # 2500 notional, +100 marked
        res = simulate_portfolio([closed, still_open], PCFG)
        self.assertAlmostEqual(res.realized_equity, 10050.0)
        self.assertAlmostEqual(res.final_equity, 10150.0)
        self.assertAlmostEqual(res.taken[1]["pnl"], 100.0)
        self.assertEqual(res.curve, [(D0, 10000.0), (D0 + 100, 10050.0)])

    def test_open_trade_keeps_its_slot(self):
        pc = cfg(max_open_positions=1)
        res = simulate_portfolio([mk(D0, None, -0.01, symbol="A"),
                                  mk(D0 + 10 * DAY, D0 + 11 * DAY, 0.01, symbol="B")], pc)
        self.assertEqual(reasons(res), [("B", "max_open")])
        self.assertAlmostEqual(res.realized_equity, 10000.0)
        self.assertAlmostEqual(res.final_equity, 9950.0)

    def test_curve_and_empty(self):
        trades = [mk(D0 + 10, D0 + 100, 0.02, symbol="A"), mk(D0 + 20, D0 + 200, -0.02, symbol="B")]
        res = simulate_portfolio(trades, PCFG)
        self.assertEqual(res.curve, [(D0 + 10, 10000.0), (D0 + 100, 10100.0), (D0 + 200, 10000.0)])
        empty = simulate_portfolio([], PCFG)
        self.assertIsInstance(empty, PortfolioResult)
        self.assertEqual((empty.curve, empty.taken, empty.skipped), ([], [], []))
        self.assertEqual(empty.final_equity, empty.realized_equity)
        self.assertEqual(empty.final_equity, 10000.0)

    def test_short_trades_use_signed_ret_pct(self):
        res = simulate_portfolio([mk(D0, D0 + 100, 0.03, side=-1)], PCFG)
        self.assertAlmostEqual(res.realized_equity, 10000.0 + 5000.0 * 0.03)


if __name__ == "__main__":
    unittest.main()
