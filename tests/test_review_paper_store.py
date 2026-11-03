"""Regression tests for the review findings fixed in fibbot/store.py and fibbot/paper.py.

F01  a few-trade trial (|Sharpe| up to the CAP) blew up trial_stats' var_sharpe and the gate
F06  zero-trade / duplicate trials shrank var_sharpe, so more search lowered the DSR hurdle
F08  re-promoting a retired spec deleted its losing paper record and was a free re-draw
F31  config edits / engine changes rewrote a running paper account (re-sized, back-filled)
F32  an open trade the replay stops reproducing, or whose bars stop, stayed open forever
F45  closed trades re-sized after a per-symbol outage were not reported
"""
import math
import random
import sqlite3
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fibbot.config import DEFAULTS
from fibbot.lab import Lab
from fibbot.metrics import CAP, deflated_sharpe, expected_max_sharpe
from fibbot.paper import (ACCOUNT_KEY, DATA_GAP_DAYS, VOID_AFTER_TICKS, account_settings, code_digest,
                          engine_fingerprint, force_close)
from fibbot.store import Store
from fibbot.strategy import ENGINE_VERSION, FibEngine, warmup_bars
from fibbot.types import Costs
from tests.test_paper import (BARS, END, N_BARS, P, SID, SPEC, STEP, SYMS, T0, FakeCandles,
                              PaperTestCase, comparable, engine_run, engine_trades, make_cfg,
                              now_after, open_cut, random_walk)

DAY = 86400
COSTS = Costs(**DEFAULTS["costs"])


def sr0(stats):
    return expected_max_sharpe(stats["n"], stats["var_sharpe"])


def r_sample(n, sharpe, seed):
    """n R multiples with per-trade Sharpe exactly `sharpe` (standardised Gaussian draws)."""
    rnd = random.Random(seed)
    xs = [rnd.gauss(0, 1) for _ in range(n)]
    m, sd = statistics.fmean(xs), statistics.stdev(xs)
    return [(x - m) / sd + sharpe for x in xs]


def realistic_trials(store, k=30, seed=7, n=100):
    """k universe-wide research trials with per-trade Sharpes ~ N(0.05, 0.07)."""
    rnd = random.Random(seed)
    vals = [rnd.gauss(0.05, 0.07) for _ in range(k)]
    for i, v in enumerate(vals):
        store.record_trial(f"t{i}", v, n=n)
    return vals


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)


class TestF01FewTradeTrials(StoreCase):
    def test_a_capped_two_trade_trial_leaves_the_variance_alone(self):
        vals = realistic_trials(self.store)
        base = self.store.trial_stats()
        self.assertAlmostEqual(base["var_sharpe"], statistics.pvariance(vals), places=12)
        self.store.record_trial("x", 99.0, n=2)              # 2 near-equal R values -> CAP
        self.store.record_trial("btc", -81.94, n=2)          # uncapped but 2 trades
        self.store.record_trial("eth", -63.87, n=3)
        st = self.store.trial_stats()
        self.assertEqual(st["n"], 33)                        # every trial still counts
        self.assertAlmostEqual(st["var_sharpe"], base["var_sharpe"], delta=0.1 * base["var_sharpe"])
        self.assertLess(sr0(st), 0.3)                        # was ~34 per trade before the fix

    def test_capped_values_are_excluded_whatever_the_trade_count(self):
        realistic_trials(self.store)
        base = self.store.trial_stats()
        self.store.record_trial("x", CAP, n=500)
        self.store.record_trial("y", -CAP)
        st = self.store.trial_stats()
        self.assertEqual((st["n"], st["n_var"]), (base["n"] + 2, base["n_var"]))
        self.assertEqual(st["var_sharpe"], base["var_sharpe"])

    def test_an_outlier_of_unknown_trade_count_is_bounded(self):
        realistic_trials(self.store)
        base = self.store.trial_stats()
        self.store.record_trial("legacy", -81.94)            # e.g. recorded before n was stored
        st = self.store.trial_stats()
        self.assertLess(st["var_sharpe"], 3 * base["var_sharpe"])
        self.assertLess(sr0(st), 2 * sr0(base))

    def test_a_strong_validation_still_passes_the_dsr_after_a_few_trade_trial(self):
        realistic_trials(self.store, k=37)
        sample = r_sample(150, 0.45, seed=3)
        st = self.store.trial_stats()
        self.assertGreater(deflated_sharpe(sample, st["n"], st["var_sharpe"]), 0.99)
        self.store.record_trial("x", 99.0, n=2)
        self.store.record_trial("y", -81.94)
        st = self.store.trial_stats()
        self.assertGreater(deflated_sharpe(sample, st["n"], st["var_sharpe"]), 0.99)

    def test_a_narrower_run_never_overwrites_a_wider_estimate(self):
        self.store.record_trial("a", 0.1, n=150)
        self.store.record_trial("a", -81.94, n=2)            # symbol-subset run of the same spec
        self.store.record_trial("b", 0.3, n=150)
        self.assertAlmostEqual(self.store.trial_stats()["var_sharpe"], 0.01, places=12)
        self.store.record_trial("a", 0.2, n=160)             # wider (or later, same size) replaces
        self.assertAlmostEqual(self.store.trial_stats()["var_sharpe"], 0.0025, places=12)
        self.store.record_trial("a", None, n=999)            # a count without an estimate: no-op
        self.assertAlmostEqual(self.store.trial_stats()["var_sharpe"], 0.0025, places=12)
        self.assertEqual(self.store.trial_stats()["n"], 2)

    def test_min_trades_threshold(self):
        self.store.record_trial("a", 0.1, n=10)
        self.store.record_trial("b", 0.3, n=10)
        self.store.record_trial("c", 0.2, n=DEFAULTS["research"]["min_trades"])
        self.assertEqual(self.store.trial_stats()["n_var"], 1)            # default research.min_trades
        st = self.store.trial_stats(min_trades=5)
        self.assertEqual((st["n"], st["n_var"]), (3, 3))
        self.assertAlmostEqual(st["var_sharpe"], statistics.pvariance([0.1, 0.3, 0.2]), places=12)

    def test_old_databases_are_migrated_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE trials (spec_id TEXT PRIMARY KEY, sharpe REAL)")
            con.executemany("INSERT INTO trials VALUES (?, ?)", [("a", 0.1), ("b", 0.3), ("c", None)])
            con.commit()
            con.close()
            with Store(path) as s:
                # F39: sharpes written before engine versions were recorded are engine 1: they
                # still count as trials, but are not comparable estimates of the current engine
                self.assertEqual(s.trial_stats(), {"n": 3, "var_sharpe": 0.0, "n_var": 0})
                s.record_trial("d", 0.2, n=200)
                s.record_trial("a", 0.4, n=200)                  # re-evaluated by this engine
                st = s.trial_stats()
                self.assertEqual((st["n"], st["n_var"]), (4, 2))
                self.assertAlmostEqual(st["var_sharpe"], 0.01, places=12)
                s.record_trial("a", 0.1, n=50)                   # narrower run of the same engine
                self.assertAlmostEqual(s.trial_stats()["var_sharpe"], 0.01, places=12)
            with Store(path) as s:                               # reopening is idempotent
                self.assertEqual(s.trial_stats()["n"], 4)


class TestF06MoreSearchNeverLowersTheBar(StoreCase):
    def test_zero_trade_trials_count_but_do_not_shrink_the_variance(self):
        realistic_trials(self.store)
        base = self.store.trial_stats()
        for i in range(64):
            self.store.record_trial(f"strict{i}", 0.0, n=0)
        st = self.store.trial_stats()
        self.assertEqual(st["n"], base["n"] + 64)
        self.assertEqual(st["var_sharpe"], base["var_sharpe"])
        self.assertGreater(sr0(st), sr0(base))

    def test_padding_with_zero_or_duplicate_trials_never_lowers_sr0(self):
        vals = realistic_trials(self.store, k=50)
        last = sr0(self.store.trial_stats())
        rnd = random.Random(11)
        for i in range(200):
            kind = i % 3
            if kind == 0:
                self.store.record_trial(f"zero{i}", 0.0, n=0)            # no trades
            elif kind == 1:
                self.store.record_trial(f"zero_unknown{i}", 0.0)         # count not passed
            else:
                self.store.record_trial(f"noop{i}", rnd.choice(vals), n=100)   # no-op variant
            now = sr0(self.store.trial_stats())
            if i > 1:                    # the first unknown-count 0.0 is one new value
                self.assertGreaterEqual(now, last - 1e-15, i)
            last = now

    def test_count_experiment_specs_counts_distinct_validated_specs(self):
        self.assertEqual(self.store.count_experiment_specs("validate"), 0)
        for sid in ("a", "b", "a"):
            self.store.add_experiment("validate", {}, sid, 0, 1, {"passed": False})
        self.store.add_experiment("backtest", {}, "c", 0, 1, {})
        self.store.add_experiment("sweep", None, None, 0, 1, {})
        self.assertEqual(self.store.count_experiment_specs("validate"), 2)
        self.assertEqual(self.store.count_experiment_specs(), 3)

    def test_a_fixed_validation_sample_cannot_flip_to_pass_by_padding(self):
        realistic_trials(self.store, k=50)
        sample = r_sample(74, 0.25, seed=5)
        st = self.store.trial_stats()
        before = deflated_sharpe(sample, st["n"], st["var_sharpe"])
        for i in range(128):
            self.store.record_trial(f"zero{i}", 0.0, n=0)
            self.store.record_trial(f"dup{i}", 0.05, n=100)
        st = self.store.trial_stats()
        self.assertLessEqual(deflated_sharpe(sample, st["n"], st["var_sharpe"]), before)


class TestF08RetirementIsFinal(PaperTestCase):
    def test_promote_after_retire_is_refused_and_the_record_stands(self):
        store = self.new_store(add=False)
        lab = Lab(self.cfg, store, self.candles)
        store.add_experiment("baseline", SPEC.to_dict(), SPEC.spec_id(), T0, P,
                             {"edge_r": 0.1, "edge_regime_r": 0.1, "engine_version": ENGINE_VERSION},
                             now_t=P - 20)
        store.add_experiment("validate", SPEC.to_dict(), SPEC.spec_id(), T0, P,
                             {"passed": True, "checks": [], "n": 40, "avg_r": 0.3,
                              "engine_version": ENGINE_VERSION}, now_t=P - 10)
        sid = lab.promote(SPEC, "gp", "validated", now_t=P - STEP)["strategy_id"]
        lab.paper().tick(now_t=END, universe=SYMS)
        trades, curve = store.paper_trades(sid), store.equity_curve(sid)
        self.assertGreater(len(trades), 10)
        lab.retire(sid, "losing on paper")
        with self.assertRaisesRegex(ValueError, "retired .*losing on paper.*final"):
            lab.promote(SPEC, "gp", "try again", now_t=END)
        self.assertEqual(store.get_strategy(sid)["status"], "retired")
        self.assertEqual(store.paper_trades(sid), trades)
        self.assertEqual(store.equity_curve(sid), curve)
        self.assertEqual(store.count_ever_on_paper(), 1)


class TestF31FrozenAccount(PaperTestCase):
    def test_config_edits_do_not_touch_a_running_account(self):
        cut = 900
        self.tick(now_t=now_after(cut))
        before = self.rows()
        curve1 = self.store.equity_curve(SID)
        edited = make_cfg(self.home, portfolio={"risk_per_trade": 0.0025, "max_open_positions": 2},
                          costs={"fee_rate": 0.0005, "slippage": 0.0001},
                          paper={"starting_equity": 5000.0})
        self.tick(cfg=edited)
        after = self.rows()
        closed = [k for k, r in before.items() if r["status"] == "closed"]
        self.assertGreater(len(closed), 20)
        for k in closed:
            self.assertEqual(comparable(after[k]), comparable(before[k]), k)
        self.assertEqual(self.store.equity_curve(SID)[:len(curve1)], curve1)
        # the whole account equals one that never saw the edit (= a backtest with its settings)
        ref = self.new_store()
        self.tick(store=ref)
        self.assertEqual({k: comparable(r) for k, r in after.items()},
                         {k: comparable(r) for k, r in self.rows(ref).items()})
        self.assertEqual(self.store.equity_curve(SID), ref.equity_curve(SID))
        acc = self.trader(cfg=edited).account(SID, now_t=END)
        self.assertAlmostEqual(acc["return"], acc["equity"] / DEFAULTS["paper"]["starting_equity"] - 1,
                               places=4)
        # a strategy promoted after the edit is booked with the new settings
        self.store.add_to_roster("s2", "later", SPEC.to_dict(), P, {}, "")
        self.tick(cfg=edited)
        s2 = account_settings(self.store, edited, self.store.get_strategy("s2"))
        self.assertEqual((s2["costs"]["fee_rate"], s2["starting_equity"]), (0.0005, 5000.0))
        s1 = self.store.get_kv(ACCOUNT_KEY.format(SID))
        self.assertEqual((s1["costs"]["fee_rate"], s1["portfolio"]["risk_per_trade"]),
                         (DEFAULTS["costs"]["fee_rate"], DEFAULTS["portfolio"]["risk_per_trade"]))

    def test_a_tick_that_fails_while_booking_does_not_advance_the_history_marks(self):
        cut = 900
        with mock.patch.object(self.store, "upsert_paper_trade", side_effect=OSError("disk full")):
            out = self.tick(now_t=now_after(cut))
        self.assertIn("disk full", out["errors"][0]["error"])
        self.assertEqual(self.rows(), {})
        out = self.tick()                             # the trades of the failed tick are not lost
        self.assertEqual(out["strategies"][SID]["refused"], [])
        ref = self.new_store()
        self.tick(store=ref)
        self.assertEqual({k: comparable(r) for k, r in self.rows().items()},
                         {k: comparable(r) for k, r in self.rows(ref).items()})

    def test_an_engine_change_is_reported_and_cannot_rewrite_history(self):
        cut = 900
        mark = T0 + cut * STEP
        self.tick(now_t=now_after(cut))
        before = self.rows()
        sigs_before = {(s["symbol"], s["leg_id"]) for s in self.store.signals(SID)}

        class Changed(FibEngine):
            def __init__(self, spec, costs=None, strategy_id=None):
                super().__init__(spec.with_overrides(swing_mult=spec.swing_mult * 1.15), costs,
                                 strategy_id)

        changed_trades = []
        for s in SYMS:
            r = Changed(SPEC, COSTS, SID).run(BARS[s].window(P - warmup_bars(SPEC) * STEP, None),
                                              trade_start_t=P)
            changed_trades += r.trades + ([r.open_trade] if r.open_trade else [])
        backfill = [tr for tr in changed_trades if (tr.symbol, tr.leg_id) not in before and tr.entry_t <= mark]
        self.assertGreater(len(backfill), 3)         # the new code would back-fill these
        with mock.patch("fibbot.paper.FibEngine", Changed), \
                mock.patch("fibbot.paper.engine_fingerprint", return_value="changed00000"):
            out = self.tick()
            res = out["strategies"][SID]
            self.assertTrue(res["engine_changed"])
            self.assertTrue(any("engine code changed" in e["error"] for e in out["errors"]))
            self.assertTrue(res["refused"])
            after = self.rows()
            for k in set(after) - set(before):
                self.assertGreater(after[k]["entry_t"], mark, k)
            for k, r in before.items():
                if r["status"] == "closed":
                    self.assertEqual(comparable(after[k]), comparable(r))
            for sym in SYMS:                         # still one position at a time per symbol
                ivs = sorted((r["entry_t"], r["exit_t"] if r["exit_t"] is not None else math.inf)
                             for k, r in after.items() if k[0] == sym)
                for (e1, x1), (e2, _) in zip(ivs, ivs[1:]):
                    self.assertLessEqual(x1, e2, sym)
            new_sigs = [s for s in self.store.signals(SID) if (s["symbol"], s["leg_id"]) not in sigs_before]
            self.assertTrue(all(s["t"] > mark for s in new_sigs))
            journal = self.store.journal(kind="system")
            self.assertTrue(any("engine changed" in j["title"] and "refused" in j["title"] for j in journal))
            again = self.tick()["strategies"][SID]   # reported once
            self.assertFalse(again["engine_changed"])
            self.assertEqual(again["refused"], [])
        self.assertEqual(len(self.store.get_kv(ACCOUNT_KEY.format(SID))["engine_changes"]), 1)


class TestEngineFingerprint(unittest.TestCase):
    def test_comments_and_docstrings_do_not_count_code_does(self):
        src = 'def f(x):\n    """Doc."""\n    return x * 2  # double\n'
        same = '# header\n\n\ndef f(x):\n    """Other doc."""\n    # note\n    return x * 2\n'
        other = 'def f(x):\n    """Doc."""\n    return x * 3\n'
        self.assertEqual(code_digest(src), code_digest(same))
        self.assertNotEqual(code_digest(src), code_digest(other))
        self.assertEqual(code_digest('class A:\n    "doc"\n'), code_digest("class A:\n    pass\n"))

    def test_fingerprint_is_stable(self):
        fp = engine_fingerprint()
        self.assertRegex(fp, r"^[0-9a-f]{12}$")
        engine_fingerprint.cache_clear()
        self.assertEqual(engine_fingerprint(), fp)


def truncated(bars, until):
    return bars.window(None, until)


class TestF32StrandedOpenTrades(PaperTestCase):
    def open_trade(self, sym, cut):
        k = next(k for k, r in self.rows().items() if k[0] == sym and r["status"] == "open")
        return k, self.rows()[k]

    def test_a_held_symbol_whose_bars_stop_is_closed_as_data_gap(self):
        sym = "BBBUSDT"
        cut = open_cut(sym)
        self.tick(now_t=now_after(cut))
        key, row = self.open_trade(sym, cut)
        stop_at = T0 + (cut + 1) * STEP
        candles = FakeCandles(bars={**BARS, sym: truncated(BARS[sym], stop_at)})
        others = [s for s in SYMS if s != sym]           # it also left the universe
        lag = DATA_GAP_DAYS * DAY // STEP + 6
        expected = engine_run(sym, until=stop_at).open_trade
        self.assertEqual((expected.symbol, expected.leg_id), key)
        for i in range(VOID_AFTER_TICKS):
            out = self.tick(candles=candles, now_t=now_after(cut + lag + i), universe=others)
            r = self.rows()[key]
            if i < VOID_AFTER_TICKS - 1:
                self.assertEqual(r["status"], "open")
                self.assertTrue(any(e["symbol"] == sym and "data_gap" in e["error"] for e in out["errors"]))
        self.assertEqual(r["status"], "closed")
        self.assertEqual((r["exit_reason"], r["exit_t"]), ("data_gap", T0 + cut * STEP))
        self.assertAlmostEqual(r["exit_price"], expected.exit_price, places=9)   # last close, net
        self.assertAlmostEqual(r["ret_pct"], expected.ret_pct, places=12)
        self.assertAlmostEqual(r["r_multiple"], expected.r_multiple, places=9)
        res = out["strategies"][SID]
        self.assertIn(key, [(d["symbol"], d["leg_id"]) for d in res["new_exits"]])
        self.tick(candles=candles, now_t=now_after(cut + lag + VOID_AFTER_TICKS), universe=others)
        self.assertNotIn(sym, self.store.get_kv("paper_universe:" + SID)["symbols"])   # not replayed
        self.assertEqual(comparable(self.rows()[key]), comparable(r))                  # final
        self.assertTrue(any("closed as data_gap" in j["body"] for j in self.store.journal(kind="system")))
        acc = self.trader().account(SID, now_t=END)
        self.assertNotIn(key[1], [p["leg_id"] for p in acc["open_positions"]])

    def test_an_open_trade_the_replay_no_longer_reproduces_is_voided(self):
        sym = "AAAUSDT"
        cut = open_cut(sym)
        self.tick(now_t=now_after(cut))
        key, row = self.open_trade(sym, cut)
        new = random_walk(N_BARS, 99, sym)
        candles = FakeCandles(bars={**BARS, sym: new})
        for i in range(VOID_AFTER_TICKS):
            out = self.tick(candles=candles)
            r = self.rows()[key]
            if i < VOID_AFTER_TICKS - 1:
                self.assertEqual(r["status"], "open")
                self.assertIn({"strategy_id": SID, "symbol": sym,
                               "error": f"open trade {key[1]} was not reproduced by the replay;"
                                        " kept as recorded"}, out["errors"])
        self.assertEqual((r["status"], r["exit_reason"], r["exit_t"]), ("closed", "void", new.t[-1]))
        # at the latest close, with the engine's exit slippage and fees
        s, fee, slip = row["side"], COSTS.fee_rate, COSTS.slippage
        done = [f for f in row["meta"]["fills"] if f[2] != "open"]
        rem = 1.0 - sum(f[0] for f in done)
        gross = sum(f[0] * f[1] for f in done) + rem * new.c[-1] * (1 - s * slip)
        p0 = row["entry_price"]
        ret = gross / p0 * (1 - fee) - (1 + fee) if s > 0 else (1 - fee) - gross / p0 * (1 + fee)
        self.assertAlmostEqual(r["exit_price"], gross, places=9)
        self.assertAlmostEqual(r["ret_pct"], ret, places=12)
        self.assertAlmostEqual(r["r_multiple"], ret / row["risk_pct"], places=9)
        self.assertEqual((r["entry_t"], r["entry_price"]), (row["entry_t"], row["entry_price"]))
        # no overlapping position on the symbol came in while the stranded one was held
        ivs = sorted((x["entry_t"], x["exit_t"] if x["exit_t"] is not None else math.inf)
                     for k, x in self.rows().items() if k[0] == sym)
        for (e1, x1), (e2, _) in zip(ivs, ivs[1:]):
            self.assertLessEqual(x1, e2)
        out = self.tick(candles=candles)              # closed is final: nothing more happens
        self.assertEqual(comparable(self.rows()[key]), comparable(r))
        self.assertFalse(any(key[1] in e["error"] for e in out["errors"]))

    def test_a_short_outage_keeps_the_trade_and_parity(self):
        sym = "BBBUSDT"
        cut = open_cut(sym)
        self.tick(now_t=now_after(cut))
        key, _ = self.open_trade(sym, cut)
        frozen = FakeCandles(bars={**BARS, sym: truncated(BARS[sym], T0 + (cut + 1) * STEP)})
        long_lag = DATA_GAP_DAYS * DAY // STEP + 6
        for i in range(VOID_AFTER_TICKS - 1):           # long lag, but too few ticks
            self.tick(candles=frozen, now_t=now_after(cut + long_lag + i))
        self.assertEqual(self.rows()[key]["status"], "open")
        reference = self.new_store()                     # one tick over everything (= backtest)
        self.tick(store=reference)
        self.tick()                                     # data is back
        self.assertEqual({k: comparable(r) for k, r in self.rows().items()},
                         {k: comparable(r) for k, r in self.rows(reference).items()})
        self.assertEqual(self.store.equity_curve(SID), reference.equity_curve(SID))

    def test_many_ticks_within_the_gap_allowance_keep_the_trade(self):
        sym = "BBBUSDT"
        cut = open_cut(sym)
        self.tick(now_t=now_after(cut))
        key, _ = self.open_trade(sym, cut)
        frozen = FakeCandles(bars={**BARS, sym: truncated(BARS[sym], T0 + (cut + 1) * STEP)})
        short_lag = DATA_GAP_DAYS * DAY // STEP - 2
        for i in range(2 * VOID_AFTER_TICKS):
            self.tick(candles=frozen, now_t=now_after(cut + short_lag) + i * 600)
        self.assertEqual(self.rows()[key]["status"], "open")

    def test_force_close_at_the_mark_reproduces_the_engine_accounting(self):
        n = 0
        for sym in SYMS:
            for cut in range(560, N_BARS, 13):
                res = engine_run(sym, until=T0 + cut * STEP)
                tr = res.open_trade
                if tr is None:
                    continue
                bars = BARS[sym].window(None, T0 + cut * STEP)
                for px in (bars.c[-1], None):
                    c = force_close(tr, bars.t[-1], "void", COSTS, STEP, close_px=px)
                    self.assertEqual((c.exit_t, c.exit_reason), (bars.t[-1], "void"))
                    self.assertAlmostEqual(c.exit_price, tr.exit_price, places=9)
                    self.assertAlmostEqual(c.ret_pct, tr.ret_pct, places=12)
                    self.assertAlmostEqual(c.r_multiple, tr.r_multiple, places=9)
                    self.assertEqual(c.bars_held, tr.bars_held)
                n += 1
        self.assertGreater(n, 20)


class TestF45ResizedTradesAreReported(PaperTestCase):
    def outage_case(self):
        """(frozen symbol X, freeze bar a, tick bar b): a trade of X enters after bar a and closes
        before a taken, closed trade of another symbol enters, which closes by bar b."""
        full = {s: engine_trades(s) for s in SYMS}
        for x in SYMS:
            for tx in full[x]:
                if tx.exit_t is None or tx.exit_t <= tx.entry_t or tx.entry_t < P + 20 * STEP:
                    continue
                a = (tx.entry_t - T0) // STEP - 1
                for y in SYMS:
                    if y == x:
                        continue
                    for ty in full[y]:
                        if ty.entry_t > tx.exit_t and ty.exit_t is not None \
                                and ty.exit_t - T0 < (a + 60) * STEP:
                            return x, a, (ty.exit_t - T0) // STEP + 1
        self.fail("no outage case in the synthetic data")

    def test_a_late_trade_re_sizes_other_closed_trades_and_it_is_reported(self):
        x, a, b = self.outage_case()
        self.tick(now_t=now_after(a))
        frozen = FakeCandles(bars={**BARS, x: truncated(BARS[x], T0 + (a + 1) * STEP)})
        self.tick(candles=frozen, now_t=now_after(b))
        sized = {k: (r["notional"], r["pnl"]) for k, r in self.rows().items() if r["status"] == "closed"}
        res = self.tick(now_t=now_after(b))["strategies"][SID]     # X's bars arrive late
        self.assertTrue(res["resized"])
        for d in res["resized"]:
            k = (d["symbol"], d["leg_id"])
            self.assertNotEqual(d["symbol"], x)
            self.assertEqual((d["notional_before"], d["pnl_before"]), sized[k])
            self.assertEqual((d["notional"], d["pnl"]), (self.rows()[k]["notional"], self.rows()[k]["pnl"]))
        self.assertTrue(any("re-sized" in j["title"] for j in self.store.journal(kind="system")))
        self.assertEqual(res["refused"], [])            # late bars are not a history rewrite
        reference = self.new_store()                    # paper == backtest sizing, exactly
        self.tick(store=reference, now_t=now_after(b))
        self.assertEqual({k: comparable(r) for k, r in self.rows().items()},
                         {k: comparable(r) for k, r in self.rows(reference).items()})
        self.assertEqual(self.store.equity_curve(SID), reference.equity_curve(SID))
        again = self.tick(now_t=now_after(b))["strategies"][SID]
        self.assertEqual(again["resized"], [])


if __name__ == "__main__":
    unittest.main()
