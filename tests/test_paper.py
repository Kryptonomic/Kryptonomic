"""Tests for fibbot.paper: replay parity with the engine, incremental ticks, signals, vetoes,
universe changes, error isolation, portfolio sizing, the equity curve and account()."""
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fibbot.config import DEFAULTS, deep_merge
from fibbot.paper import UNIVERSE_KEY, PaperTrader, account_drawdown, paper_days, rounded
from fibbot.store import Store
from fibbot.strategy import FibEngine, StrategySpec, warmup_bars
from fibbot.types import TF_SECONDS, Bars, Costs, Signal

T0, STEP = 1_700_006_400, TF_SECONDS["4h"]
N_BARS = 1200
P = T0 + 500 * STEP                        # paper_start_t
SID = "s1"
SPEC = StrategySpec(name="paper_test", trend_filter="none", trend_ema=20, swing_mult=2.0,
                    swing_atr_len=10, min_leg_atr=2.0, min_rr=1.0, direction="both",
                    max_hold_bars=40)
SYMS = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
END = T0 + N_BARS * STEP                   # a time at which every bar has closed


def random_walk(n, seed, symbol, tf="4h", start=100.0):
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
    return Bars.from_rows(symbol, tf, rows)


BARS = {sym: random_walk(N_BARS, seed, sym) for seed, sym in enumerate(SYMS, start=1)}


def make_cfg(home, **over):
    cfg = deep_merge(DEFAULTS, over)
    cfg["home"] = str(home)
    return cfg


def engine_run(sym, trade_start_t=P, until=None, bars=None):
    """Reference: FibEngine on the paper window of `sym` (bars with t < until)."""
    b = (bars or BARS[sym]).window(P - warmup_bars(SPEC) * STEP, until)
    return FibEngine(SPEC, Costs(**DEFAULTS["costs"]), strategy_id=SID).run(b, trade_start_t=trade_start_t)


def engine_trades(sym, **kw):
    r = engine_run(sym, **kw)
    return r.trades + ([r.open_trade] if r.open_trade else [])


def now_after(i):
    """A tick time at which bars 0..i have closed (bar i+1 is forming)."""
    return T0 + (i + 1) * STEP + 90


class FakeCandles:
    """CandleStore stand-in: serves bars with t < until, records update_all calls."""

    def __init__(self, bars=None, until=None):
        self.bars = dict(BARS if bars is None else bars)
        self.until = until
        self.calls = []
        self.last_errors = {}
        self.fail = set()

    def get(self, symbol, tf):
        if symbol in self.fail:
            raise RuntimeError(f"cache unreadable for {symbol}")
        b = self.bars.get(symbol)
        return b.window(None, self.until) if b is not None and b.tf == tf else Bars(symbol, tf)

    def update_all(self, symbols, tfs):
        self.calls.append((list(symbols), list(tfs)))
        return {}


def trade_key(d):
    return d["symbol"], d["leg_id"]


def comparable(row):
    """A paper_trades row without the derived status, for exact comparisons."""
    return {k: v for k, v in row.items() if k != "status"}


class PaperTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.n_stores = 0
        self.cfg = make_cfg(self.home)
        self.store = self.new_store()
        self.candles = FakeCandles()

    def new_store(self, sid=SID, add=True):
        self.n_stores += 1
        store = Store(self.home / f"fib{self.n_stores}.sqlite")
        self.addCleanup(store.close)
        if add:
            store.add_to_roster(sid, "paper test", SPEC.to_dict(), P, {"metrics": {"avg_r": 0.1}}, "test")
        return store

    def trader(self, store=None, candles=None, cfg=None):
        return PaperTrader(cfg or self.cfg, store or self.store, candles or self.candles)

    def tick(self, store=None, candles=None, cfg=None, now_t=END, universe=SYMS, **kw):
        return self.trader(store, candles, cfg).tick(now_t=now_t, universe=universe, **kw)

    def rows(self, store=None, status=None):
        return {trade_key(r): r for r in (store or self.store).paper_trades(SID, status=status)}


class TestTickReplay(PaperTestCase):
    def test_empty_roster_does_nothing(self):
        store = self.new_store(add=False)
        with mock.patch("fibbot.paper.get_universe", side_effect=AssertionError("not needed")):
            out = self.trader(store).tick()
        self.assertEqual(out, {"strategies": {}, "new_signals": [], "errors": []})
        self.assertEqual(self.candles.calls, [])

    def test_trades_match_the_engine_and_start_at_paper_start(self):
        out = self.tick()
        self.assertEqual(out["errors"], [])
        rows = self.rows()
        expected = {(tr.symbol, tr.leg_id): tr for sym in SYMS for tr in engine_trades(sym)}
        self.assertGreater(len(expected), 40)
        self.assertEqual(set(rows), set(expected))
        for k, tr in expected.items():
            self.assertEqual(self.store.trade_from_row(rows[k]).to_dict(), tr.to_dict())
        self.assertTrue(all(r["entry_t"] >= P for r in rows.values()))
        # the engine would have traded before paper_start_t: those trades are not paper trades
        self.assertTrue(any(tr.entry_t < P for tr in engine_trades("AAAUSDT", trade_start_t=None)))
        sigs = self.store.signals(SID)
        self.assertEqual(len(sigs), sum(len(engine_run(s).signals) for s in SYMS))
        self.assertTrue(all(s["t"] >= P - STEP for s in sigs))

    def test_now_t_limits_the_replay_to_closed_bars(self):
        self.tick(now_t=now_after(699))
        other = self.new_store()
        self.tick(store=other, candles=FakeCandles(until=T0 + 700 * STEP), now_t=None)
        self.assertEqual({k: comparable(r) for k, r in self.rows().items()},
                         {k: comparable(r) for k, r in self.rows(other).items()})
        self.assertTrue(all(r["entry_t"] <= T0 + 699 * STEP for r in self.rows().values()))
        self.assertEqual(len(self.rows()), sum(len(engine_trades(s, until=T0 + 700 * STEP))
                                               for s in SYMS))

    def test_incremental_ticks_never_change_closed_trades(self):
        closed_before: dict = {}
        entries, exits = [], []
        for k in (505, 560, 640, 700, 760, 850, 990, 1100, N_BARS):
            self.candles.until = T0 + k * STEP
            res = self.tick(now_t=None)["strategies"][SID]
            entries += [trade_key(d) for d in res["new_entries"]]
            exits += [trade_key(d) for d in res["new_exits"]]
            rows = self.rows()
            for key, row in closed_before.items():
                self.assertEqual(comparable(rows[key]), row, key)
            closed_before = {key: comparable(r) for key, r in rows.items() if r["status"] == "closed"}
        # every trade is reported once as an entry and, once closed, once as an exit
        self.assertEqual(sorted(entries), sorted(set(entries)))
        self.assertEqual(set(entries), set(self.rows()))
        self.assertEqual(sorted(exits), sorted(set(exits)))
        self.assertEqual(set(exits), set(self.rows(status="closed")))
        # incremental ticks == one tick over all the data (same trades, sizing and curve)
        single = self.new_store()
        self.tick(store=single)
        self.assertEqual({k: comparable(r) for k, r in self.rows().items()},
                         {k: comparable(r) for k, r in self.rows(single).items()})
        self.assertEqual(self.store.equity_curve(SID), single.equity_curve(SID))
        strip = lambda s: {k: v for k, v in s.items() if k != "created_t"}
        self.assertEqual([strip(s) for s in self.store.signals(SID)],
                         [strip(s) for s in single.signals(SID)])

    def test_new_entry_then_new_exit(self):
        tr = next(tr for tr in engine_trades("AAAUSDT") if tr.exit_t is not None and tr.bars_held >= 2)
        entry_i = (tr.entry_t - T0) // STEP
        first = self.tick(now_t=now_after(entry_i))["strategies"][SID]
        key = (tr.symbol, tr.leg_id)
        self.assertIn(key, [trade_key(d) for d in first["new_entries"]])
        self.assertNotIn(key, [trade_key(d) for d in first["new_exits"]])
        self.assertEqual(self.rows()[key]["status"], "open")
        second = self.tick()["strategies"][SID]
        self.assertNotIn(key, [trade_key(d) for d in second["new_entries"]])
        exit_d = next(d for d in second["new_exits"] if trade_key(d) == key)
        self.assertEqual(exit_d["exit_t"], tr.exit_t)
        self.assertIn("notional", exit_d)
        self.assertIn("pnl", exit_d)

    def test_signals_are_recorded_once(self):
        first = self.tick()
        n = len(self.store.signals(SID))
        self.assertGreater(n, 0)
        self.assertEqual(len(first["new_signals"]), n)
        self.assertTrue(all(isinstance(s, Signal) for s in first["new_signals"]))
        self.assertEqual(first["strategies"][SID]["new_signals"],
                         [s.to_dict() for s in first["new_signals"]])
        second = self.tick()
        self.assertEqual(second["new_signals"], [])
        self.assertEqual(second["strategies"][SID]["new_signals"], [])
        self.assertEqual(second["strategies"][SID]["new_entries"], [])
        self.assertEqual(second["strategies"][SID]["new_exits"], [])
        self.assertEqual(len(self.store.signals(SID)), n)

    def test_update_data_calls_update_all_once(self):
        self.store.add_to_roster("h1", "hourly", SPEC.with_overrides(timeframe="1h").to_dict(), P, {}, "")
        with mock.patch("fibbot.paper.get_universe", return_value=["BBBUSDT", "AAAUSDT"]) as gu:
            out = self.trader().tick(now_t=END)
        gu.assert_called_once_with(self.cfg)
        self.assertEqual(self.candles.calls, [(["AAAUSDT", "BBBUSDT"], ["1h", "4h"])])
        self.assertEqual(set(out["strategies"]), {SID, "h1"})
        self.assertEqual(out["strategies"]["h1"]["new_entries"], [])   # no 1h data
        self.tick(update_data=False)
        self.assertEqual(len(self.candles.calls), 1)

    def test_data_update_errors_are_reported_not_fatal(self):
        self.candles.last_errors = {("BBBUSDT", "4h"): "HTTP 500"}
        out = self.tick()
        self.assertIn({"strategy_id": None, "symbol": "BBBUSDT", "error": "data update 4h: HTTP 500"},
                      out["errors"])
        self.assertTrue(any(k[0] == "BBBUSDT" for k in self.rows()))
        broken = FakeCandles()
        broken.update_all = mock.Mock(side_effect=OSError("disk full"))
        out = self.tick(store=self.new_store(), candles=broken)
        self.assertEqual(out["errors"][0]["error"], "data update failed: disk full")
        self.assertIn(SID, out["strategies"])

    def test_symbol_and_strategy_errors_do_not_abort_the_tick(self):
        self.store.add_to_roster("bad", "bad", {"timeframe": "7m"}, P, {}, "")
        self.candles.fail = {"BBBUSDT"}
        out = self.tick()
        errs = {(e["strategy_id"], e["symbol"]) for e in out["errors"]}
        self.assertEqual(errs, {("bad", None), (SID, "BBBUSDT")})
        self.assertIn("timeframe", next(e["error"] for e in out["errors"] if e["strategy_id"] == "bad"))
        self.assertEqual({k[0] for k in self.rows()}, {"AAAUSDT", "CCCUSDT"})
        self.assertNotIn("bad", out["strategies"])


class TestVetoes(PaperTestCase):
    def leg_with_signal(self):
        """A trade whose signal is in the paper signal window, before its entry."""
        return next(tr for tr in engine_trades("AAAUSDT") if tr.setup_t >= P)

    def run_with_veto(self, mode):
        cfg = make_cfg(self.home, brain={"signal_review": mode})
        store = self.new_store()
        tr = self.leg_with_signal()
        self.tick(store=store, cfg=cfg, candles=FakeCandles(until=tr.setup_t + STEP))
        self.assertNotIn((tr.symbol, tr.leg_id), self.rows(store))
        store.set_signal_review(SID, tr.symbol, tr.leg_id, "skip", 0.9, "looks weak")
        self.tick(store=store, cfg=cfg)
        return tr, self.rows(store)

    def test_vetoed_leg_is_skipped_in_active_mode(self):
        tr, rows = self.run_with_veto("active")
        self.assertNotIn((tr.symbol, tr.leg_id), rows)
        self.assertTrue(any(k[0] == "AAAUSDT" for k in rows))

    def test_vetoed_leg_is_traded_in_shadow_and_off_modes(self):
        for mode in ("shadow", "off"):
            tr, rows = self.run_with_veto(mode)
            self.assertIn((tr.symbol, tr.leg_id), rows, mode)

    def test_a_veto_never_erases_a_recorded_trade(self):
        cfg = make_cfg(self.home, brain={"signal_review": "active"})
        cut = T0 + 1000 * STEP
        self.tick(cfg=cfg, candles=FakeCandles(until=cut))
        rows = self.rows()
        signal_legs = {(s["symbol"], s["leg_id"]) for s in self.store.signals(SID)}
        vetoed = [k for k in rows if k in signal_legs]
        self.assertTrue(any(rows[k]["status"] == "open" for k in vetoed) or len(vetoed) > 5)
        for sym, leg in vetoed:
            self.store.set_signal_review(SID, sym, leg, "skip", 1.0, "hindsight")
        self.tick(cfg=cfg, candles=FakeCandles(until=cut))
        after = self.rows()
        for k in vetoed:
            self.assertEqual(comparable(after[k]), comparable(rows[k]))


class TestUniverseChanges(PaperTestCase):
    def test_a_symbol_joining_later_is_not_back_filled(self):
        self.tick(now_t=now_after(599), universe=["AAAUSDT", "BBBUSDT"])
        self.assertEqual(self.store.get_kv(UNIVERSE_KEY.format(SID)),
                         {"paper_start_t": P, "symbols": {"AAAUSDT": P, "BBBUSDT": P}})
        join = T0 + 800 * STEP
        self.tick(now_t=now_after(799), universe=SYMS)
        self.tick()
        ccc = {k: r for k, r in self.rows().items() if k[0] == "CCCUSDT"}
        self.assertTrue(any(tr.entry_t < join for tr in engine_trades("CCCUSDT")))
        expected = engine_trades("CCCUSDT", trade_start_t=join)
        self.assertGreater(len(expected), 0)
        self.assertEqual(set(ccc), {(tr.symbol, tr.leg_id) for tr in expected})
        self.assertTrue(all(r["entry_t"] >= join for r in ccc.values()))
        self.assertTrue(all(s["t"] >= join - STEP for s in self.store.signals(SID)
                            if s["symbol"] == "CCCUSDT"))
        self.assertEqual(self.store.get_kv(UNIVERSE_KEY.format(SID))["symbols"]["CCCUSDT"], join)

    def test_a_dropped_symbol_only_manages_its_open_position(self):
        tr = next(t for t in engine_trades("BBBUSDT") if t.exit_t is not None
                  and t.exit_t < T0 + 1100 * STEP and t.bars_held >= 2)
        entry_i = (tr.entry_t - T0) // STEP
        self.tick(now_t=now_after(entry_i))
        key = (tr.symbol, tr.leg_id)
        self.assertEqual(self.rows()[key]["status"], "open")
        before = {k for k in self.rows() if k[0] == "BBBUSDT"}
        n_signals = len([s for s in self.store.signals(SID) if s["symbol"] == "BBBUSDT"])
        dropped = ["AAAUSDT", "CCCUSDT"]
        out = self.tick(universe=dropped)
        self.assertEqual(out["errors"], [])
        rows = self.rows()
        self.assertEqual({k for k in rows if k[0] == "BBBUSDT"}, before)
        self.assertEqual(self.store.trade_from_row(rows[key]).to_dict(), tr.to_dict())
        self.assertTrue(any(t.entry_t > tr.exit_t for t in engine_trades("BBBUSDT")))
        self.assertEqual(len([s for s in self.store.signals(SID) if s["symbol"] == "BBBUSDT"]),
                         n_signals)
        self.assertIn("BBBUSDT", self.store.get_kv(UNIVERSE_KEY.format(SID))["symbols"])
        self.tick(universe=dropped)    # nothing held any more: the symbol leaves the paper universe
        self.assertNotIn("BBBUSDT", self.store.get_kv(UNIVERSE_KEY.format(SID))["symbols"])


def scaled(bars, k):
    """The same series with every price multiplied by k: same pivots and legs, other prices."""
    return Bars(bars.symbol, bars.tf, list(bars.t),
                *[[x * k for x in col] for col in (bars.o, bars.h, bars.l, bars.c)], list(bars.v))


def open_cut(sym):
    """A bar index after which `sym` holds an open position (from the full-run intervals)."""
    return next(i for i in range(900, N_BARS) for tr in engine_trades(sym)
                if tr.entry_t <= T0 + i * STEP and (tr.exit_t is None or tr.exit_t > T0 + i * STEP))


class TestRecordIntegrity(PaperTestCase):
    def test_closed_trades_are_final_open_trades_follow_the_replay(self):
        sym, cut = "AAAUSDT", open_cut("AAAUSDT")
        self.tick(now_t=now_after(cut))
        rows = self.rows()
        revised = scaled(BARS[sym], 1.01)     # a data revision that keeps every leg
        self.assertEqual({(t.symbol, t.leg_id) for t in engine_trades(sym, until=T0 + (cut + 1) * STEP)},
                         {(t.symbol, t.leg_id) for t in engine_trades(sym, bars=revised,
                                                                      until=T0 + (cut + 1) * STEP)})
        out = self.tick(candles=FakeCandles(bars={**BARS, sym: revised}), now_t=now_after(cut))
        self.assertEqual(out["errors"], [])
        after = self.rows()
        self.assertEqual(set(after), set(rows))
        n_closed = 0
        for k, r in rows.items():
            if k[0] == sym and r["status"] == "closed":
                n_closed += 1
                self.assertEqual(self.store.trade_from_row(after[k]).to_dict(),
                                 self.store.trade_from_row(r).to_dict())
            elif k[0] == sym:
                self.assertAlmostEqual(after[k]["entry_price"], r["entry_price"] * 1.01, places=9)
        self.assertGreater(n_closed, 5)

    def test_a_failing_symbol_keeps_its_recorded_trades(self):
        sym = "BBBUSDT"
        self.tick(now_t=now_after(open_cut(sym)))
        rows = {k: comparable(r) for k, r in self.rows().items() if k[0] == sym}
        self.assertIn("open", {self.rows()[k]["status"] for k in rows})
        self.candles.fail = {sym}
        out = self.tick()
        self.assertEqual([(e["symbol"], e["error"]) for e in out["errors"]],
                         [(sym, f"RuntimeError: cache unreadable for {sym}")])
        self.assertEqual({k: comparable(r) for k, r in self.rows().items() if k[0] == sym}, rows)

    def test_an_open_trade_the_replay_loses_is_kept_and_reported(self):
        sym = "AAAUSDT"
        self.tick(now_t=now_after(open_cut(sym)))
        rows = self.rows()
        open_key = next(k for k, r in rows.items() if k[0] == sym and r["status"] == "open")
        out = self.tick(candles=FakeCandles(bars={**BARS, sym: random_walk(N_BARS, 99, sym)}))
        self.assertIn({"strategy_id": SID, "symbol": sym,
                       "error": f"open trade {open_key[1]} was not reproduced by the replay;"
                                " kept as recorded"}, out["errors"])
        after = self.rows()
        self.assertEqual(after[open_key]["status"], "open")
        for k, r in rows.items():
            if r["status"] == "closed":
                self.assertEqual(comparable(after[k])["exit_t"], r["exit_t"])


class TestBookkeeping(PaperTestCase):
    def test_sizing_equity_curve_and_result(self):
        out = self.tick()["strategies"][SID]
        rows = list(self.store.paper_trades(SID))
        start = DEFAULTS["paper"]["starting_equity"]
        for r in rows:
            self.assertAlmostEqual(r["pnl"], r["notional"] * r["ret_pct"], places=9)
            self.assertGreaterEqual(r["notional"], 0.0)
        skipped = [r for r in rows if r["notional"] == 0.0]
        self.assertEqual(sum(r["pnl"] for r in skipped), 0.0)
        curve = self.store.equity_curve(SID)
        self.assertEqual(curve[0], (P, start))
        taken_closed = sorted((r for r in rows if r["status"] == "closed" and r["notional"] > 0),
                              key=lambda r: r["exit_t"])
        # F19: the stored curve is marked to market at every bar close a position is open (the
        # basis of backtest()'s drawdown), so it holds every realized exit and more points
        times = [t for t, _ in curve]
        self.assertEqual(times, sorted(times))
        self.assertGreater(len(curve), 1 + len(taken_closed))
        self.assertTrue({r["exit_t"] for r in taken_closed} <= set(times))
        self.assertAlmostEqual(out["equity"], start + sum(r["pnl"] for r in rows), places=6)
        self.assertAlmostEqual(curve[-1][1], out["equity"], places=6)     # the last mark is the equity
        self.assertEqual(out["open"], sum(r["status"] == "open" and r["notional"] > 0 for r in rows))

    def test_marked_drawdown_matches_the_backtest(self):
        # F19: paper drawdown is measured on the same mark-to-market curve as backtest()
        from fibbot.backtest import backtest
        self.tick()
        acc = self.trader().account(SID, now_t=END)
        bt = backtest(SPEC, {s: BARS[s] for s in SYMS}, P, END, self.cfg)
        self.assertEqual(bt.equity_basis, "marked")
        self.assertAlmostEqual(acc["equity"], bt.portfolio.final_equity, delta=0.01)   # rounded
        self.assertGreater(bt.metrics["max_drawdown"], 0)
        self.assertAlmostEqual(acc["max_drawdown"], bt.metrics["max_drawdown"], places=5)

    def test_portfolio_limits_skip_trades(self):
        iv = {s: [(tr.entry_t, tr.exit_t or math.inf) for tr in engine_trades(s)] for s in SYMS}
        cut = next(i for i in range(600, N_BARS)
                   if sum(any(e <= T0 + i * STEP < x for e, x in iv[s]) for s in SYMS) >= 2)
        cfg = make_cfg(self.home, portfolio={"max_open_positions": 1})
        res = self.tick(cfg=cfg, now_t=now_after(cut))["strategies"][SID]
        rows = list(self.store.paper_trades(SID))
        self.assertTrue(any(r["notional"] == 0.0 and r["pnl"] == 0.0 for r in rows))
        self.assertTrue(any(r["notional"] > 0.0 for r in rows))
        open_rows = [r for r in rows if r["status"] == "open"]
        self.assertGreaterEqual(len(open_rows), 2)
        held = [r for r in open_rows if r["notional"] > 0]
        self.assertLessEqual(len(held), 1)
        self.assertEqual(res["open"], len(held))
        acc = PaperTrader(cfg, self.store, self.candles).account(SID, now_t=END)
        self.assertEqual(acc["n_open"], len(held))
        self.assertEqual([p["leg_id"] for p in acc["open_positions"]], [r["leg_id"] for r in held])
        self.assertEqual(acc["n_closed"], sum(r["status"] == "closed" for r in rows))

    def test_curve_without_trades_is_the_start_point(self):
        self.tick(now_t=now_after(400))
        self.assertEqual(self.store.equity_curve(SID), [(P, DEFAULTS["paper"]["starting_equity"])])
        self.assertEqual(self.rows(), {})

    def test_account(self):
        self.tick(now_t=now_after(1000))
        now = P + 30 * 86400
        acc = self.trader().account(SID, now_t=now)
        rows = list(self.store.paper_trades(SID))
        start = DEFAULTS["paper"]["starting_equity"]
        closed = [r for r in rows if r["status"] == "closed"]
        held = [r for r in rows if r["status"] == "open" and r["notional"] > 0]
        equity = start + sum(r["pnl"] for r in rows)
        self.assertEqual((acc["strategy_id"], acc["name"], acc["status"], acc["timeframe"]),
                         (SID, "paper test", "paper", "4h"))
        self.assertEqual((acc["paper_start_t"], acc["days"]), (P, 30.0))
        self.assertAlmostEqual(acc["equity"], equity, delta=equity * 1e-5)
        self.assertAlmostEqual(acc["realized_equity"], start + sum(r["pnl"] for r in closed),
                               delta=equity * 1e-5)
        self.assertAlmostEqual(acc["return"], equity / start - 1, places=5)
        self.assertEqual((acc["n_closed"], acc["n_open"], acc["stats"]["n"]),
                         (len(closed), len(held), len(closed)))
        self.assertEqual(acc["max_drawdown"], rounded(account_drawdown(
            start, self.store.equity_curve(SID), rows)))
        self.assertEqual([p["leg_id"] for p in acc["open_positions"]], [r["leg_id"] for r in held])
        if held:
            p, r = acc["open_positions"][0], held[0]
            self.assertEqual(p["tp1"], rounded(r["meta"]["tp1"]))
            self.assertEqual(p["mark_price"], rounded(r["exit_price"]))
        with self.assertRaises(KeyError):
            self.trader().account("nope")


class TestHelpers(unittest.TestCase):
    @staticmethod
    def row(exit_t, pnl):
        return {"exit_t": exit_t, "pnl": pnl}

    def test_account_drawdown_uses_curve_then_marked_equity(self):
        curve = [(1, 100.0), (2, 120.0), (3, 90.0), (4, 110.0)]
        self.assertAlmostEqual(account_drawdown(100.0, curve, []), 0.25)
        # an open position marked -40 below the realized 110 deepens the drawdown
        rows = [self.row(4, 10.0), self.row(None, -40.0)]
        self.assertAlmostEqual(account_drawdown(100.0, [(1, 100.0), (4, 110.0)], rows), 40 / 110)

    def test_account_drawdown_falls_back_to_cumulative_pnl(self):
        rows = [self.row(3, -30.0), self.row(1, 20.0), self.row(None, None), self.row(5, 5.0)]
        # exits in time order: 100 -> 120 -> 90 -> 95
        self.assertAlmostEqual(account_drawdown(100.0, [], rows), 30 / 120)
        self.assertEqual(account_drawdown(100.0, [], []), 0.0)

    def test_paper_days_and_rounded(self):
        self.assertEqual(paper_days({"paper_start_t": 1000}, 1000 + 86400 * 3), 3.0)
        self.assertEqual(paper_days({"paper_start_t": 1000}, 10), 0.0)
        self.assertEqual(rounded({"a": [1.23456789, 2], "b": (0.000123456789,), "c": "x"}),
                         {"a": [1.23457, 2], "b": [0.000123457], "c": "x"})
        self.assertTrue(math.isnan(rounded(float("nan"))))
        self.assertIs(rounded(True), True)


if __name__ == "__main__":
    unittest.main()
