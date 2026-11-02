"""Tests for fibbot.backtest: multi-symbol runs (serial == process pool), result assembly,
portfolio / metrics wiring, per-symbol and per-year breakdowns, and the compact JSON summary."""
import json
import math
import random
import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest import mock

from fibbot import backtest as bt
from fibbot.backtest import (as_spec, backtest, costs_from_cfg, data_period, iso_day,
                             resolve_workers, result_from_trades, rounded, run_strategy)
from fibbot.config import DEFAULTS, deep_merge
from fibbot.metrics import equity_stats, trade_stats
from fibbot.portfolio import simulate_portfolio
from fibbot.strategy import FibEngine, StrategySpec, warmup_bars
from fibbot.types import TF_SECONDS, Bars, Costs, Trade

T0 = 1_600_041_600            # 2020-09-14 00:00 UTC
H4 = TF_SECONDS["4h"]
SPEC = StrategySpec(name="synthetic", swing_mode="pct", swing_pct=0.05, min_leg_atr=2.0,
                    trend_filter="none", stop_atr_buffer=0.0, max_hold_bars=80, min_rr=1.0)


def make_bars(symbol, n=1500, seed=0, tf="4h", start=100.0, t0=T0):
    """Seeded impulse / pullback walk: up-legs of 10-25% over 6-14 bars, then pullbacks of
    0.3-1.3x the leg (some fill the 0.618-0.65 zone and win, some run through the stop)."""
    rnd = random.Random(seed)
    closes, px = [], start
    while len(closes) < n:
        low = px
        n_up, up = rnd.randint(6, 14), rnd.uniform(0.10, 0.25)
        for _ in range(n_up):
            px *= (1 + up) ** (1 / n_up) * math.exp(rnd.gauss(0, 0.004))
            closes.append(px)
        n_dn = rnd.randint(4, 10)
        step = ((px - rnd.uniform(0.3, 1.3) * (px - low)) / px) ** (1 / n_dn)
        for _ in range(n_dn):
            px *= step * math.exp(rnd.gauss(0, 0.004))
            closes.append(px)
    rows, prev = [], start
    for i, c in enumerate(closes[:n]):
        w = abs(rnd.gauss(0, 0.004))
        rows.append((t0 + i * TF_SECONDS[tf], prev, max(prev, c) * (1 + w), min(prev, c) * (1 - w), c, 1.0))
        prev = c
    return Bars.from_rows(symbol, tf, rows)


def make_universe(k=3, n=1500):
    return {f"S{i}USDT": make_bars(f"S{i}USDT", n=n, seed=i) for i in range(k)}


def cfg_(research=None, **sections):
    """Defaults with serial research and optional section overrides."""
    return deep_merge(DEFAULTS, {"research": {"workers": 1, **(research or {})}, **sections})


def trade(symbol="AUSDT", entry_t=T0, r=1.0, risk_pct=0.02, side=1, is_open=False, leg=None):
    """Hand-made closed (one bar later) or open trade with ret_pct = r * risk_pct; tp1/tp2 at 2R/3R."""
    entry = 100.0
    return Trade("sid", symbol, "4h", side, leg or f"L{entry_t}", entry_t - H4, entry_t, entry,
                 entry * (1 - side * risk_pct), None if is_open else entry_t + H4,
                 entry * (1 + side * r * risk_pct), "open" if is_open else "stop" if r <= 0 else "target",
                 r * risk_pct, risk_pct, r, 1,
                 {"tp1": entry * (1 + side * 2 * risk_pct), "tp2": entry * (1 + side * 3 * risk_pct)})


def all_trades(runs):
    return [tr for r in runs.values() for tr in r.trades + ([r.open_trade] if r.open_trade else [])]


class TestRunStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = make_universe(3)
        cls.start, cls.end = T0 + 500 * H4, T0 + 1300 * H4
        cls.costs = Costs(0.001, 0.0005)
        cls.serial = run_strategy(SPEC, cls.bars, cls.start, cls.end, cls.costs)

    def test_each_symbol_runs_the_engine_on_its_warmup_window(self):
        self.assertEqual(list(self.serial), list(self.bars))
        for sym, b in self.bars.items():
            w = b.window(self.start, self.end, warmup_bars(SPEC))
            expected = FibEngine(SPEC, self.costs).run(w, trade_start_t=self.start)
            got = self.serial[sym]
            self.assertEqual(got, expected)
            self.assertEqual(got.n_bars, min(500, warmup_bars(SPEC)) + 800)
            self.assertTrue(got.trades)
            for tr in got.trades:
                self.assertGreaterEqual(tr.entry_t, self.start)
                self.assertLess(tr.exit_t, self.end)
                self.assertEqual(tr.strategy_id, SPEC.spec_id())

    def test_process_pool_gives_identical_results(self):
        sym = next(iter(self.bars))
        skip = {sym: {self.serial[sym].trades[0].leg_id}}
        for kw in ({}, {"skip_legs": skip, "strategy_id": "paper-1"}):
            serial = run_strategy(SPEC, self.bars, self.start, self.end, self.costs, workers=1, **kw)
            pooled = run_strategy(SPEC, self.bars, self.start, self.end, self.costs, workers=2, **kw)
            self.assertEqual(list(pooled), list(serial))
            self.assertEqual(pooled, serial)

    def test_pool_only_for_several_workers_and_symbols(self):
        one = dict(list(self.bars.items())[:1])
        with mock.patch.object(bt, "ProcessPoolExecutor", side_effect=AssertionError("no pool")):
            run_strategy(SPEC, self.bars, self.start, self.end, self.costs, workers=1)
            run_strategy(SPEC, one, self.start, self.end, self.costs, workers=4)
        with mock.patch.object(bt, "ProcessPoolExecutor", wraps=ProcessPoolExecutor) as pool:
            run_strategy(SPEC, self.bars, self.start, self.end, self.costs, workers=8)
        pool.assert_called_once_with(3)

    def test_skip_legs_and_strategy_id(self):
        sym, other = list(self.bars)[:2]
        leg = self.serial[sym].trades[0].leg_id
        r = run_strategy(SPEC, self.bars, self.start, self.end, self.costs,
                         skip_legs={sym: {leg}}, strategy_id="sid-x")
        self.assertNotIn(leg, [tr.leg_id for tr in r[sym].trades])
        self.assertNotIn(leg, [s.leg_id for s in r[sym].signals])
        self.assertEqual([tr.leg_id for tr in r[other].trades],
                         [tr.leg_id for tr in self.serial[other].trades])
        self.assertEqual({tr.strategy_id for res in r.values() for tr in res.trades}, {"sid-x"})

    def test_dict_spec_none_window_and_invalid_spec(self):
        self.assertEqual(run_strategy(SPEC.to_dict(), self.bars, self.start, self.end, self.costs),
                         self.serial)
        full = run_strategy(SPEC, self.bars, None, None, self.costs)
        self.assertEqual({s: r.n_bars for s, r in full.items()}, {s: 1500 for s in self.bars})
        with self.assertRaises(ValueError):
            run_strategy(SPEC.with_overrides(entry_zone_hi=0.5), self.bars, None, None, self.costs)
        with self.assertRaises(ValueError):
            run_strategy({"bogus": 1}, self.bars, None, None, self.costs)


class TestHelpers(unittest.TestCase):
    def test_resolve_workers(self):
        with mock.patch("os.cpu_count", return_value=8):
            self.assertEqual([resolve_workers(n) for n in (0, None, -3, 1, 5)], [7, 7, 7, 1, 5])
        for cpus, want in ((1, 1), (None, 1), (2, 1)):
            with mock.patch("os.cpu_count", return_value=cpus):
                self.assertEqual(resolve_workers(0), want)

    def test_costs_from_cfg(self):
        self.assertEqual(costs_from_cfg(cfg_(costs={"fee_rate": 0.002, "slippage": 0.001})),
                         Costs(0.002, 0.001))
        self.assertEqual(costs_from_cfg({"costs": {"fee_rate": 0.0}}), Costs(0.0, Costs().slippage))
        self.assertEqual(costs_from_cfg({}), Costs())

    def test_as_spec(self):
        self.assertIs(as_spec(SPEC), SPEC)
        self.assertEqual(as_spec(SPEC.to_dict()), SPEC)
        with self.assertRaises(ValueError):
            as_spec({"min_rr": -1})

    def test_rounded(self):
        x = {"a": 1234.56789, "b": 0.000123456, "c": [float("nan"), float("inf"), -0.5],
             "d": (1, True, None, "s"), "e": {"f": -98.123456}}
        self.assertEqual(rounded(x), {"a": 1234.5679, "b": 0.0001235, "c": [None, None, -0.5],
                                      "d": [1, True, None, "s"], "e": {"f": -98.1235}})
        self.assertEqual(rounded(0.123456789, 2), 0.12)
        self.assertIs(rounded(True), True)
        json.dumps(rounded(x), allow_nan=False)

    def test_iso_day_and_data_period(self):
        self.assertEqual(iso_day(T0 + 3600), "2020-09-14")
        self.assertIsNone(iso_day(None))
        bars = {"A": make_bars("A", n=10), "B": make_bars("B", n=20, t0=T0 + 5 * H4),
                "E": Bars("E", "4h")}
        self.assertEqual(data_period(bars, None, None), (T0, T0 + 25 * H4))
        self.assertEqual(data_period(bars, T0 + H4, T0 + 3 * H4), (T0 + H4, T0 + 3 * H4))
        self.assertEqual(data_period(bars, T0 - H4, T0 + 99 * H4), (T0, T0 + 25 * H4))
        self.assertEqual(data_period({}, 5, None), (5, None))


class TestBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = make_universe(3)
        cls.cfg = cfg_()
        cls.start, cls.end = T0 + 400 * H4, T0 + 1400 * H4
        cls.res = backtest(SPEC, cls.bars, cls.start, cls.end, cls.cfg)

    def test_trades_are_every_run_trade_sorted(self):
        runs = run_strategy(SPEC, self.bars, self.start, self.end, costs_from_cfg(self.cfg))
        want = sorted(all_trades(runs), key=lambda tr: (tr.entry_t, tr.symbol, tr.leg_id))
        self.assertEqual(self.res.trades, want)
        self.assertGreater(len(want), 30)
        self.assertEqual(self.res.closed_trades, [tr for tr in want if tr.exit_t is not None])
        self.assertEqual((self.res.spec, self.res.spec_id), (SPEC.to_dict(), SPEC.spec_id()))
        self.assertEqual((self.res.start_t, self.res.end_t, self.res.symbols),
                         (self.start, self.end, list(self.bars)))

    def test_metrics_portfolio_and_equity_curve(self):
        r = self.res
        closed = r.closed_trades
        pf = simulate_portfolio(r.trades, self.cfg["portfolio"])
        self.assertEqual(r.portfolio, pf)
        self.assertEqual(r.period, (self.start, self.end))
        curve = [(self.start, pf.start_equity)] + pf.curve + [(self.end, pf.final_equity)]
        want = {**trade_stats(closed), **equity_stats(curve, pf.start_equity),
                "n_open": len(r.trades) - len(closed)}
        self.assertEqual(r.metrics, want)
        self.assertEqual(r.metrics["n_days"], self.end // 86400 - self.start // 86400 + 1)

    def test_per_symbol_and_per_year(self):
        r = self.res
        self.assertEqual(list(r.per_symbol), list(self.bars))
        for sym, row in r.per_symbol.items():
            st = trade_stats([tr for tr in r.closed_trades if tr.symbol == sym])
            self.assertEqual(row, {k: st[k] for k in bt.LITE_KEYS})
        self.assertEqual(list(r.per_year), ["2020", "2021"])
        self.assertEqual(sum(y["n"] for y in r.per_year.values()), r.metrics["n"])
        self.assertAlmostEqual(sum(y["total_r"] for y in r.per_year.values()), r.metrics["total_r"])
        self.assertEqual(set(r.per_year["2021"]), {"n", "total_r", "avg_r"})

    def test_open_trade_kept_marked_but_not_in_trade_stats(self):
        closed = self.res.closed_trades
        tr = next(t for t in closed if t.bars_held > 2)
        r = backtest(SPEC, self.bars, self.start, tr.exit_t, self.cfg)
        open_ = [t for t in r.trades if t.exit_t is None]
        self.assertIn(tr.leg_id, [t.leg_id for t in open_ if t.symbol == tr.symbol])
        self.assertEqual(r.metrics["n_open"], len(open_))
        self.assertEqual(r.metrics["n"], len(r.trades) - len(open_))
        marked = sum(rec["pnl"] for rec in r.portfolio.taken if rec["trade"].exit_t is None)
        self.assertAlmostEqual(r.portfolio.final_equity, r.portfolio.realized_equity + marked)

    def test_workers_come_from_cfg_unless_given(self):
        runs = run_strategy(SPEC, self.bars, self.start, self.end, costs_from_cfg(self.cfg))
        no_research = {k: v for k, v in self.cfg.items() if k != "research"}
        with mock.patch.object(bt, "run_strategy", return_value=runs) as rs, \
                mock.patch("os.cpu_count", return_value=8):
            backtest(SPEC, self.bars, self.start, self.end, cfg_({"workers": 0}))
            backtest(SPEC, self.bars, self.start, self.end, cfg_({"workers": 3}))
            backtest(SPEC, self.bars, self.start, self.end, cfg_({"workers": 3}), workers=1)
            got = backtest(SPEC, self.bars, self.start, self.end, no_research)
        self.assertEqual([c.kwargs["workers"] for c in rs.call_args_list], [7, 3, 1, 7])
        self.assertEqual(got.trades, self.res.trades)

    def test_costs_come_from_cfg(self):
        with mock.patch.object(bt, "run_strategy", wraps=run_strategy) as rs:
            dear = backtest(SPEC, self.bars, self.start, self.end,
                            cfg_(costs={"fee_rate": 0.004, "slippage": 0.002}))
        self.assertEqual(rs.call_args.args[4], Costs(0.004, 0.002))
        self.assertLess(dear.metrics["avg_ret_pct"], self.res.metrics["avg_ret_pct"])

    def test_no_trades(self):
        r = backtest(SPEC.with_overrides(min_rr=10.0), self.bars, self.start, self.end, self.cfg)
        self.assertEqual((r.trades, r.metrics["n"], r.metrics["n_open"]), ([], 0, 0))
        self.assertEqual((r.metrics["total_return"], r.metrics["max_drawdown"]), (0.0, 0.0))
        self.assertEqual(r.per_year, {})
        self.assertEqual(r.summary()["per_symbol_omitted"], 0)
        empty = backtest(SPEC, {"X": Bars("X", "4h")}, None, None, self.cfg)
        self.assertEqual((empty.period, empty.metrics["n_days"]), ((None, None), 0))
        json.dumps(empty.summary(), allow_nan=False)


class TestResultFromTrades(unittest.TestCase):
    def test_years_symbols_open_trades_and_period_curve(self):
        new_year = 1_609_459_200                     # 2021-01-01 00:00 UTC
        trades = [trade("BUSDT", new_year - 3600, 2.0), trade("AUSDT", new_year, -1.0),
                  trade("AUSDT", new_year + 86400, 0.5, is_open=True),
                  trade("AUSDT", new_year - 7200, 1.0, side=-1)]
        start, end = new_year - 10 * 86400, new_year + 10 * 86400
        r = result_from_trades(SPEC, trades, ["AUSDT", "BUSDT", "CUSDT"], start, end, cfg_())
        self.assertEqual([t.entry_t for t in r.trades], sorted(t.entry_t for t in trades))
        self.assertEqual(r.per_year, {"2020": {"n": 2, "total_r": 3.0, "avg_r": 1.5},
                                      "2021": {"n": 1, "total_r": -1.0, "avg_r": -1.0}})
        self.assertEqual(r.per_symbol["CUSDT"], dict.fromkeys(bt.LITE_KEYS, 0))
        self.assertEqual(r.per_symbol["AUSDT"]["n"], 2)
        self.assertEqual((r.metrics["n"], r.metrics["n_open"], r.metrics["short_n"]), (3, 1, 1))
        self.assertEqual(len(r.portfolio.taken), 4)
        self.assertEqual(r.metrics["n_days"], 21)
        self.assertEqual(r.period, (start, end))
        explicit = result_from_trades(SPEC, trades, ["AUSDT"], None, None, cfg_(), period=(start, end))
        self.assertEqual(explicit.metrics["n_days"], 21)
        bare = result_from_trades(SPEC, trades, ["AUSDT"], None, None, cfg_())
        self.assertEqual(bare.metrics["n_days"], 2)   # only the portfolio's own points


class TestSummary(unittest.TestCase):
    def setUp(self):
        # 7 symbols with total R 6, 5, ..., 0
        self.trades = [trade(f"S{k}USDT", T0 + (10 * k + j) * H4, 1.0) for k in range(7)
                       for j in range(6 - k)] + [trade("S6USDT", T0, -0.5)]
        self.res = result_from_trades(SPEC, self.trades, [f"S{k}USDT" for k in range(7)],
                                      T0, T0 + 200 * H4,
                                      cfg_(portfolio={"max_open_positions": 1}))

    def test_compact_json_safe_overview(self):
        s = self.res.summary()
        text = json.dumps(s, allow_nan=False)
        self.assertLess(len(text), 12000)
        self.assertNotIn("trades", s)
        self.assertEqual(s["spec_id"], SPEC.spec_id())
        self.assertEqual((s["name"], s["timeframe"], s["direction"]), ("synthetic", "4h", "long"))
        self.assertEqual(s["period"], {"start": "2020-09-14", "end": "2020-10-17"})
        self.assertEqual((s["n_symbols"], s["n_open"]), (7, 0))
        self.assertEqual(s["metrics"], rounded(self.res.metrics))
        self.assertEqual(s["per_year"], rounded(self.res.per_year))
        pf = s["portfolio"]
        self.assertEqual(pf["n_taken"] + sum(pf["skipped"].values()), len(self.trades))
        self.assertEqual(set(pf["skipped"]), {"max_open"})
        self.assertEqual(pf["final_equity"], rounded(self.res.portfolio.final_equity))

    def test_per_symbol_lists_best_and_worst_when_many(self):
        s = self.res.summary()
        self.assertEqual(list(s["per_symbol"]), [f"S{k}USDT" for k in range(7)])
        self.assertEqual(s["per_symbol_omitted"], 0)
        s = self.res.summary(max_symbols=4)
        self.assertEqual(list(s["per_symbol"]), ["S0USDT", "S1USDT", "S5USDT", "S6USDT"])
        self.assertEqual(s["per_symbol_omitted"], 3)
        s = self.res.summary(max_symbols=3)
        self.assertEqual(list(s["per_symbol"]), ["S0USDT", "S1USDT", "S6USDT"])
        self.assertEqual(s["per_symbol_omitted"], 4)
        s = self.res.summary(max_symbols=0)
        self.assertEqual((s["per_symbol"], s["per_symbol_omitted"]), ({}, 7))

    def test_real_backtest_summary(self):
        r = backtest(SPEC, make_universe(3), T0 + 400 * H4, None, cfg_())
        s = r.summary()
        json.dumps(s, allow_nan=False)
        self.assertEqual(s["metrics"]["n"], r.metrics["n"])
        self.assertEqual(s["period"]["end"], iso_day(T0 + 1500 * H4))


if __name__ == "__main__":
    unittest.main()
