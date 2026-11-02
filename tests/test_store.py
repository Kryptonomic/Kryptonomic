import math
import sqlite3
import statistics
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fibbot.store import EXPERIMENT_KINDS, JOURNAL_KINDS, Store
from fibbot.types import Signal, Trade

T0 = 1_700_000_000


def make_trade(symbol="BTCUSDT", leg_id="L100", entry_t=T0, exit_t=None, exit_price=101.0,
               exit_reason="open", ret_pct=0.01, strategy_id="s1", **kw):
    d = dict(strategy_id=strategy_id, symbol=symbol, tf="4h", side=1, leg_id=leg_id,
             setup_t=entry_t - 14400, entry_t=entry_t, entry_price=100.0, stop_price=95.0,
             exit_t=exit_t, exit_price=exit_price, exit_reason=exit_reason, ret_pct=ret_pct,
             risk_pct=0.05, r_multiple=ret_pct / 0.05, bars_held=3,
             meta={"anchor": 90.0, "fills": [[0.5, 105.0, "tp1", entry_t + 14400]], "entry_mode": "limit"})
    d.update(kw)
    return Trade(**d)


def make_signal(symbol="BTCUSDT", leg_id="L100", t=T0, strategy_id="s1", **kw):
    d = dict(strategy_id=strategy_id, symbol=symbol, tf="4h", side=1, leg_id=leg_id, t=t,
             entry_price=100.0, stop_price=95.0, tp1=110.0, tp2=116.18,
             meta={"anchor": 90.0, "extreme": 110.0, "rr": 2.1, "mode": "limit"})
    d.update(kw)
    return Signal(**d)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)


class TestOpenAndPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "nested" / "dir" / "fibbot.sqlite"

    def test_creates_parent_dirs_wal_and_schema(self):
        with Store(self.path) as s:
            mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
            tables = {r[0] for r in s._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue(self.path.exists())
        self.assertEqual(mode, "wal")
        self.assertTrue({"experiments", "trials", "roster", "paper_trades", "signals", "equity",
                         "journal", "kv"} <= tables)

    def test_reopen_keeps_data_and_schema_is_idempotent(self):
        s = Store(str(self.path))
        eid = s.add_experiment("backtest", {"a": 1}, "abc", 1, 2, {"avg_r": 0.2}, now_t=T0)
        s.set_kv("k", {"x": [1, 2]})
        s.add_to_roster("s1", "gp", {"timeframe": "4h"}, T0, {"passed": True}, "why", now_t=T0)
        s.close()
        s2 = Store(self.path)
        self.addCleanup(s2.close)
        self.assertEqual(s2.get_experiment(eid)["metrics"], {"avg_r": 0.2})
        self.assertEqual(s2.get_kv("k"), {"x": [1, 2]})
        self.assertEqual(s2.get_strategy("s1")["validation"], {"passed": True})

    def test_two_connections_see_each_others_writes(self):
        a, b = Store(self.path), Store(self.path)
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        a.add_journal("system", "hello", "from a", now_t=T0)
        self.assertEqual(b.journal()[0]["title"], "hello")

    def test_usable_from_another_thread(self):
        with Store(self.path) as s:
            errors = []

            def work():
                try:
                    for i in range(20):
                        s.add_journal("system", f"t{i}", "", now_t=T0 + i)
                except Exception as e:  # pragma: no cover - reported via assert below
                    errors.append(e)

            threads = [threading.Thread(target=work) for _ in range(3)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.assertEqual(errors, [])
            self.assertEqual(len(s.journal(limit=None)), 60)

    def test_close_is_idempotent_and_blocks_use(self):
        s = Store(self.path)
        s.close()
        s.close()
        with self.assertRaises(sqlite3.ProgrammingError):
            s.get_kv("x")


class TestExperiments(StoreCase):
    def test_add_and_get_round_trip(self):
        spec = {"timeframe": "4h", "entry_zone_lo": 0.618, "trend_filter": "ema"}
        metrics = {"n": 42, "avg_r": 0.21, "per_year": {"2023": {"n": 3}}, "tags": ["x", "y"]}
        eid = self.store.add_experiment("backtest", spec, "abc123", 100, 200, metrics,
                                        source="brain", notes="first", now_t=T0)
        row = self.store.get_experiment(eid)
        self.assertEqual(row, {"id": eid, "created_t": T0, "kind": "backtest", "spec_id": "abc123",
                               "spec": spec, "window_start": 100, "window_end": 200,
                               "metrics": metrics, "source": "brain", "notes": "first"})

    def test_defaults_none_spec_and_time(self):
        before = int(time.time())
        eid = self.store.add_experiment("sweep", None, None, None, None, {"rows": 3})
        row = self.store.get_experiment(eid)
        self.assertIsNone(row["spec"])
        self.assertIsNone(row["spec_id"])
        self.assertIsNone(row["window_start"])
        self.assertEqual(row["source"], "cli")
        self.assertEqual(row["notes"], "")
        self.assertTrue(before <= row["created_t"] <= int(time.time()))

    def test_nan_and_non_json_metrics_survive(self):
        eid = self.store.add_experiment("backtest", {}, "a", None, None,
                                        {"sharpe": float("nan"), "path": Path("/x"), "tup": (1, 2)}, now_t=T0)
        m = self.store.get_experiment(eid)["metrics"]
        self.assertTrue(math.isnan(m["sharpe"]))
        self.assertEqual(m["path"], "/x")
        self.assertEqual(m["tup"], [1, 2])

    def test_ids_increase_and_missing_is_none(self):
        a = self.store.add_experiment("backtest", {}, "a", None, None, {}, now_t=T0)
        b = self.store.add_experiment("backtest", {}, "b", None, None, {}, now_t=T0)
        self.assertGreater(b, a)
        self.assertIsNone(self.store.get_experiment(b + 100))

    def test_all_contract_kinds_accepted_and_unknown_rejected(self):
        for k in sorted(EXPERIMENT_KINDS):
            self.store.add_experiment(k, None, None, None, None, {}, now_t=T0)
        self.assertEqual(self.store.count_experiments(), 6)
        with self.assertRaises(ValueError):
            self.store.add_experiment("walkforward", None, None, None, None, {})

    def test_list_recent_order_kind_filter_and_limit(self):
        ids = [self.store.add_experiment(k, {}, f"s{i}", None, None, {}, now_t=T0 + t)
               for i, (k, t) in enumerate([("backtest", 5), ("optimize", 9), ("backtest", 9), ("backtest", 1)])]
        rows = self.store.list_experiments()
        # created_t DESC, then id DESC for equal timestamps
        self.assertEqual([r["id"] for r in rows], [ids[2], ids[1], ids[0], ids[3]])
        self.assertEqual([r["id"] for r in self.store.list_experiments(kind="backtest")],
                         [ids[2], ids[0], ids[3]])
        self.assertEqual(len(self.store.list_experiments(limit=2)), 2)
        self.assertEqual(len(self.store.list_experiments(limit=None)), 4)
        self.assertEqual(self.store.list_experiments(kind="baseline"), [])

    def test_sort_by_metric_missing_last(self):
        def add(metrics, t):
            return self.store.add_experiment("backtest", {}, "x", None, None, metrics, now_t=T0 + t)
        a = add({"avg_r": 0.1, "profit_factor": 3.0}, 1)
        b = add({"avg_r": 0.5, "profit_factor": 1.1}, 2)
        c = add({"profit_factor": 2.0}, 3)                       # avg_r missing
        d = add({"avg_r": float("nan"), "profit_factor": 0.5}, 4)  # nan counts as missing
        e = add({"avg_r": -0.2}, 5)
        f = add({"avg_r": 0.1}, 6)                               # ties a, newer first
        g = add({"avg_r": "high"}, 7)                            # non-numeric counts as missing
        self.assertEqual([r["id"] for r in self.store.list_experiments(limit=None, sort_by="avg_r")],
                         [b, f, a, e, g, d, c])
        self.assertEqual([r["id"] for r in self.store.list_experiments(limit=3, sort_by="profit_factor")],
                         [a, c, b])
        self.assertEqual([r["id"] for r in self.store.list_experiments(limit=2, sort_by="avg_r",
                                                                       kind="backtest")], [b, f])
        with self.assertRaises(ValueError):
            self.store.list_experiments(sort_by="sharpe")

    def test_latest_validation(self):
        self.assertIsNone(self.store.latest_validation("abc"))
        self.store.add_experiment("validate", {}, "abc", None, None, {"passed": False}, now_t=T0)
        newest = self.store.add_experiment("validate", {}, "abc", None, None, {"passed": True}, now_t=T0 + 10)
        self.store.add_experiment("backtest", {}, "abc", None, None, {"passed": "no"}, now_t=T0 + 20)
        self.store.add_experiment("validate", {}, "other", None, None, {"passed": False}, now_t=T0 + 30)
        row = self.store.latest_validation("abc")
        self.assertEqual(row["id"], newest)
        self.assertEqual(row["metrics"], {"passed": True})

    def test_latest_validation_tie_breaks_on_id(self):
        self.store.add_experiment("validate", {}, "abc", None, None, {"v": 1}, now_t=T0)
        second = self.store.add_experiment("validate", {}, "abc", None, None, {"v": 2}, now_t=T0)
        self.assertEqual(self.store.latest_validation("abc")["id"], second)

    def test_count_experiments(self):
        for k, t in [("backtest", 0), ("backtest", 10), ("validate", 10), ("optimize", 20)]:
            self.store.add_experiment(k, None, None, None, None, {}, now_t=T0 + t)
        self.assertEqual(self.store.count_experiments(), 4)
        self.assertEqual(self.store.count_experiments(kind="backtest"), 2)
        self.assertEqual(self.store.count_experiments(since_t=T0 + 10), 3)
        self.assertEqual(self.store.count_experiments(kind="backtest", since_t=T0 + 10), 1)
        self.assertEqual(self.store.count_experiments(kind="sweep"), 0)


class TestTrials(StoreCase):
    def test_empty(self):
        self.assertEqual(self.store.trial_stats(), {"n": 0, "var_sharpe": 0.0})

    def test_upsert_keeps_latest_and_population_variance(self):
        self.store.record_trial("a", 0.5)
        self.assertEqual(self.store.trial_stats(), {"n": 1, "var_sharpe": 0.0})
        self.store.record_trial("b", 1.5)
        self.store.record_trial("c", None)
        self.store.record_trial("a", 1.0)      # latest sharpe replaces 0.5
        self.store.record_trial("d", -0.25)
        st = self.store.trial_stats()
        self.assertEqual(st["n"], 4)
        self.assertAlmostEqual(st["var_sharpe"], statistics.pvariance([1.0, 1.5, -0.25]), places=12)

    def test_null_or_nan_does_not_erase_known_sharpe(self):
        self.store.record_trial("a", 0.0)
        self.store.record_trial("b", 2.0)
        self.store.record_trial("a", None)
        self.store.record_trial("b", float("nan"))
        self.assertEqual(self.store.trial_stats(), {"n": 2, "var_sharpe": 1.0})

    def test_non_finite_and_null_sharpes_excluded_from_variance(self):
        self.store.record_trial("a", 1.0)
        self.store.record_trial("b", float("inf"))
        self.store.record_trial("c", None)
        self.assertEqual(self.store.trial_stats(), {"n": 3, "var_sharpe": 0.0})
        self.store.record_trial("d", 3.0)
        self.assertEqual(self.store.trial_stats(), {"n": 4, "var_sharpe": 1.0})


class TestRoster(StoreCase):
    SPEC = {"name": "gp", "timeframe": "4h", "entry_zone_lo": 0.618}
    VALIDATION = {"passed": True, "checks": [{"name": "min_trades", "value": 40, "passed": True}]}

    def add(self, sid="s1", start=T0, now=T0, **kw):
        args = dict(name="gp", spec=self.SPEC, paper_start_t=start, validation=self.VALIDATION,
                    rationale="solid", now_t=now)
        args.update(kw)
        self.store.add_to_roster(sid, **args)

    def test_add_and_get(self):
        self.add()
        row = self.store.get_strategy("s1")
        self.assertEqual(row["status"], "paper")
        self.assertEqual(row["spec"], self.SPEC)
        self.assertEqual(row["validation"], self.VALIDATION)
        self.assertEqual(row["paper_start_t"], T0)
        self.assertEqual(row["name"], "gp")
        self.assertEqual(row["rationale"], "solid")
        self.assertEqual(row["added_t"], T0)
        self.assertEqual(row["status_reason"], "")
        self.assertIsNone(self.store.get_strategy("nope"))

    def test_roster_filters_and_order(self):
        self.add("b", start=T0 + 100)
        self.add("a", start=T0 + 200)
        self.add("c", start=T0)
        self.store.set_strategy_status("a", "retired", "drawdown", now_t=T0 + 300)
        self.assertEqual([r["strategy_id"] for r in self.store.roster()], ["c", "b"])
        self.assertEqual([r["strategy_id"] for r in self.store.roster(None)], ["c", "b", "a"])
        self.assertEqual([r["strategy_id"] for r in self.store.roster("retired")], ["a"])
        self.assertEqual(self.store.roster("graduated"), [])
        with self.assertRaises(ValueError):
            self.store.roster("live")

    def test_set_status(self):
        self.add()
        self.store.set_strategy_status("s1", "graduated", "all checks", now_t=T0 + 50)
        row = self.store.get_strategy("s1")
        self.assertEqual((row["status"], row["status_reason"], row["status_t"]),
                         ("graduated", "all checks", T0 + 50))
        with self.assertRaises(ValueError):
            self.store.set_strategy_status("s1", "live")
        with self.assertRaises(KeyError):
            self.store.set_strategy_status("missing", "retired")

    def test_readding_active_strategy_raises(self):
        self.add()
        with self.assertRaises(ValueError):
            self.add(start=T0 + 999)
        self.store.set_strategy_status("s1", "graduated")
        with self.assertRaises(ValueError):
            self.add(start=T0 + 999)
        self.assertEqual(self.store.get_strategy("s1")["paper_start_t"], T0)

    def test_readding_retired_reactivates_with_fresh_account(self):
        self.add()
        self.store.upsert_paper_trade("s1", make_trade(exit_t=T0 + 1, exit_reason="stop"), 1000.0, -50.0)
        self.store.set_equity_curve("s1", [(T0, 10000.0), (T0 + 1, 9950.0)])
        self.store.add_signal(make_signal())
        self.store.set_signal_review("s1", "BTCUSDT", "L100", "skip", 0.7, "late", now_t=T0)
        self.store.set_strategy_status("s1", "retired", "bad", now_t=T0 + 10)
        # other strategies' data must be untouched
        self.add("s2")
        self.store.upsert_paper_trade("s2", make_trade(strategy_id="s2"))
        self.store.set_equity_curve("s2", [(T0, 10000.0)])

        new_spec = dict(self.SPEC, entry_zone_lo=0.5)
        self.add(start=T0 + 5000, now=T0 + 4000, spec=new_spec, validation={"passed": True},
                 rationale="retry", name="gp2")
        row = self.store.get_strategy("s1")
        self.assertEqual(row["status"], "paper")
        self.assertEqual(row["paper_start_t"], T0 + 5000)
        self.assertEqual(row["spec"], new_spec)
        self.assertEqual((row["name"], row["rationale"], row["status_reason"]), ("gp2", "retry", ""))
        self.assertEqual(row["validation"], {"passed": True})
        self.assertEqual(self.store.paper_trades("s1"), [])
        self.assertEqual(self.store.equity_curve("s1"), [])
        self.assertEqual(self.store.vetoed_legs("s1"), {("BTCUSDT", "L100")})
        self.assertEqual(len(self.store.paper_trades("s2")), 1)
        self.assertEqual(self.store.equity_curve("s2"), [(T0, 10000.0)])
        self.assertEqual(self.store.count_ever_on_paper(), 2)

    def test_count_ever_on_paper(self):
        self.assertEqual(self.store.count_ever_on_paper(), 0)
        self.add("a")
        self.add("b")
        self.store.set_strategy_status("a", "retired")
        self.assertEqual(self.store.count_ever_on_paper(), 2)
        self.add("a", start=T0 + 1)
        self.assertEqual(self.store.count_ever_on_paper(), 2)


class TestPaperTrades(StoreCase):
    def test_new_unchanged_updated(self):
        t = make_trade()
        self.assertEqual(self.store.upsert_paper_trade("s1", t), "new")
        self.assertEqual(self.store.upsert_paper_trade("s1", t), "unchanged")
        # marked-to-market open trade moves
        t2 = make_trade(exit_price=102.0, ret_pct=0.02)
        self.assertEqual(self.store.upsert_paper_trade("s1", t2), "updated")
        # it closes
        t3 = make_trade(exit_t=T0 + 3 * 14400, exit_price=105.0, exit_reason="target", ret_pct=0.048)
        self.assertEqual(self.store.upsert_paper_trade("s1", t3), "updated")
        self.assertEqual(self.store.upsert_paper_trade("s1", t3), "unchanged")
        rows = self.store.paper_trades()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "closed")
        self.assertEqual(rows[0]["exit_reason"], "target")

    def test_notional_and_pnl_compare_and_none_keeps_stored(self):
        t = make_trade()
        self.store.upsert_paper_trade("s1", t)
        self.assertIsNone(self.store.paper_trades()[0]["notional"])
        self.assertEqual(self.store.upsert_paper_trade("s1", t, 1000.0, 10.0), "updated")
        self.assertEqual(self.store.upsert_paper_trade("s1", t, 1000.0, 10.0), "unchanged")
        self.assertEqual(self.store.upsert_paper_trade("s1", t), "unchanged")   # None keeps 1000/10
        row = self.store.paper_trades()[0]
        self.assertEqual((row["notional"], row["pnl"]), (1000.0, 10.0))
        self.assertEqual(self.store.upsert_paper_trade("s1", t, pnl=12.5), "updated")
        self.assertEqual(self.store.upsert_paper_trade("s1", t, notional=0.0, pnl=0.0), "updated")
        row = self.store.paper_trades()[0]
        self.assertEqual((row["notional"], row["pnl"]), (0.0, 0.0))

    def test_other_field_changes_count_as_updated(self):
        self.store.upsert_paper_trade("s1", make_trade())
        self.assertEqual(self.store.upsert_paper_trade("s1", make_trade(bars_held=4)), "updated")
        meta = {"anchor": 90.0, "fills": [[0.5, 105.0, "tp1", T0], [0.5, 110.0, "tp2", T0]]}
        self.assertEqual(self.store.upsert_paper_trade("s1", make_trade(bars_held=4, meta=meta)), "updated")
        self.assertEqual(self.store.paper_trades()[0]["meta"], meta)

    def test_row_contains_trade_fields_and_round_trips(self):
        t = make_trade(exit_t=T0 + 14400, exit_reason="stop", ret_pct=-0.051, side=-1)
        self.store.upsert_paper_trade("s1", t, notional=2500.0, pnl=-127.5)
        row = self.store.paper_trades()[0]
        for k, v in t.to_dict().items():
            self.assertEqual(row[k], v, k)
        self.assertEqual((row["notional"], row["pnl"], row["status"]), (2500.0, -127.5, "closed"))
        self.assertEqual(Store.trade_from_row(row), t)
        self.assertEqual(self.store.trade_from_row(row), t)

    def test_nan_fields_round_trip_as_nan(self):
        t = make_trade(r_multiple=float("nan"), risk_pct=float("nan"))
        self.store.upsert_paper_trade("s1", t, notional=float("nan"))
        self.assertEqual(self.store.upsert_paper_trade("s1", t), "unchanged")
        row = self.store.paper_trades()[0]
        self.assertTrue(math.isnan(row["r_multiple"]) and math.isnan(row["risk_pct"]))
        self.assertIsNone(row["notional"])
        self.assertTrue(math.isnan(Store.trade_from_row(row).r_multiple))

    def test_uniqueness_key(self):
        self.assertEqual(self.store.upsert_paper_trade("s1", make_trade()), "new")
        self.assertEqual(self.store.upsert_paper_trade("s1", make_trade(symbol="ETHUSDT")), "new")
        self.assertEqual(self.store.upsert_paper_trade("s1", make_trade(leg_id="L200")), "new")
        self.assertEqual(self.store.upsert_paper_trade("s2", make_trade(strategy_id="s2")), "new")
        self.assertEqual(self.store.upsert_paper_trade("s1", make_trade(entry_t=T0 + 5)), "updated")
        self.assertEqual(len(self.store.paper_trades()), 4)

    def test_strategy_id_argument_is_the_key(self):
        self.store.upsert_paper_trade("roster-id", make_trade(strategy_id="spec-id"))
        row = self.store.paper_trades("roster-id")[0]
        self.assertEqual(row["strategy_id"], "roster-id")
        self.assertEqual(self.store.paper_trades("spec-id"), [])

    def test_filters_and_newest_entry_first(self):
        self.store.upsert_paper_trade("s1", make_trade(symbol="A", leg_id="L1", entry_t=T0 + 10))
        self.store.upsert_paper_trade("s1", make_trade(symbol="B", leg_id="L2", entry_t=T0 + 30,
                                                       exit_t=T0 + 40, exit_reason="stop"))
        self.store.upsert_paper_trade("s1", make_trade(symbol="A", leg_id="L3", entry_t=T0 + 20,
                                                       exit_t=T0 + 25, exit_reason="target"))
        self.store.upsert_paper_trade("s2", make_trade(symbol="C", leg_id="L4", entry_t=T0 + 50))
        legs = lambda rows: [r["leg_id"] for r in rows]
        self.assertEqual(legs(self.store.paper_trades()), ["L4", "L2", "L3", "L1"])
        self.assertEqual(legs(self.store.paper_trades("s1")), ["L2", "L3", "L1"])
        self.assertEqual(legs(self.store.paper_trades("s1", status="closed")), ["L2", "L3"])
        self.assertEqual(legs(self.store.paper_trades("s1", status="open")), ["L1"])
        self.assertEqual(legs(self.store.paper_trades(status="open")), ["L4", "L1"])
        self.assertEqual(legs(self.store.paper_trades("s1", limit=2)), ["L2", "L3"])
        self.assertEqual({r["status"] for r in self.store.paper_trades(status="open")}, {"open"})
        self.assertEqual(self.store.paper_trades("nobody"), [])
        with self.assertRaises(ValueError):
            self.store.paper_trades(status="pending")


class TestSignals(StoreCase):
    def test_add_unique(self):
        self.assertTrue(self.store.add_signal(make_signal(), now_t=T0))
        self.assertFalse(self.store.add_signal(make_signal(entry_price=99.0), now_t=T0 + 1))
        self.assertTrue(self.store.add_signal(make_signal(symbol="ETHUSDT")))
        self.assertTrue(self.store.add_signal(make_signal(leg_id="L200")))
        self.assertTrue(self.store.add_signal(make_signal(strategy_id="s2")))
        self.assertEqual(len(self.store.signals()), 4)
        first = self.store.signals("s1", since_t=T0)[-1]
        self.assertEqual(first["entry_price"], 100.0)   # the duplicate did not overwrite

    def test_row_fields_and_round_trip(self):
        sig = make_signal()
        self.store.add_signal(sig, now_t=T0 + 7)
        row = self.store.signals()[0]
        for k, v in sig.to_dict().items():
            self.assertEqual(row[k], v, k)
        self.assertEqual(row["created_t"], T0 + 7)
        self.assertEqual((row["decision"], row["confidence"], row["reason"], row["reviewed_t"]),
                         (None, None, None, None))
        self.assertEqual(Signal.from_dict(row), sig)

    def test_filters_and_order(self):
        self.store.add_signal(make_signal(symbol="A", leg_id="L1", t=T0 + 10))
        self.store.add_signal(make_signal(symbol="B", leg_id="L2", t=T0 + 30))
        self.store.add_signal(make_signal(symbol="A", leg_id="L3", t=T0 + 20))
        self.store.add_signal(make_signal(symbol="C", leg_id="L4", t=T0 + 40, strategy_id="s2"))
        self.store.set_signal_review("s1", "B", "L2", "take", 0.8, "trend", now_t=T0 + 31)
        legs = lambda rows: [r["leg_id"] for r in rows]
        self.assertEqual(legs(self.store.signals()), ["L4", "L2", "L3", "L1"])
        self.assertEqual(legs(self.store.signals("s1")), ["L2", "L3", "L1"])
        self.assertEqual(legs(self.store.signals("s1", unreviewed=True)), ["L3", "L1"])
        self.assertEqual(legs(self.store.signals(since_t=T0 + 20)), ["L4", "L2", "L3"])
        self.assertEqual(legs(self.store.signals("s1", unreviewed=True, since_t=T0 + 15)), ["L3"])
        self.assertEqual(legs(self.store.signals(limit=1)), ["L4"])

    def test_review_update_and_validation(self):
        self.store.add_signal(make_signal())
        self.store.set_signal_review("s1", "BTCUSDT", "L100", "take", 0.6, "ok", now_t=T0 + 1)
        self.store.set_signal_review("s1", "BTCUSDT", "L100", "skip", 0.9, "changed mind", now_t=T0 + 2)
        row = self.store.signals()[0]
        self.assertEqual((row["decision"], row["confidence"], row["reason"], row["reviewed_t"]),
                         ("skip", 0.9, "changed mind", T0 + 2))
        with self.assertRaises(ValueError):
            self.store.set_signal_review("s1", "BTCUSDT", "L100", "maybe", 0.5, "")
        with self.assertRaises(KeyError):
            self.store.set_signal_review("s1", "BTCUSDT", "L999", "skip", 0.5, "")

    def test_vetoed_legs(self):
        for sym, leg, sid in [("A", "L1", "s1"), ("A", "L2", "s1"), ("B", "L1", "s1"), ("A", "L1", "s2")]:
            self.store.add_signal(make_signal(symbol=sym, leg_id=leg, strategy_id=sid))
        self.assertEqual(self.store.vetoed_legs("s1"), set())
        self.store.set_signal_review("s1", "A", "L1", "skip", 0.7, "", now_t=T0)
        self.store.set_signal_review("s1", "A", "L2", "take", 0.7, "", now_t=T0)
        self.store.set_signal_review("s1", "B", "L1", "skip", 0.7, "", now_t=T0)
        self.store.set_signal_review("s2", "A", "L1", "skip", 0.7, "", now_t=T0)
        self.assertEqual(self.store.vetoed_legs("s1"), {("A", "L1"), ("B", "L1")})
        self.assertEqual(self.store.vetoed_legs("s2"), {("A", "L1")})
        self.assertEqual(self.store.vetoed_legs("s3"), set())

    def test_count_signal_reviews_since(self):
        for i in range(4):
            self.store.add_signal(make_signal(leg_id=f"L{i}"))
        self.assertEqual(self.store.count_signal_reviews_since(0), 0)
        for i, t in enumerate([T0, T0 + 100, T0 + 200]):
            self.store.set_signal_review("s1", "BTCUSDT", f"L{i}", "take", 0.5, "", now_t=t)
        self.assertEqual(self.store.count_signal_reviews_since(T0), 3)
        self.assertEqual(self.store.count_signal_reviews_since(T0 + 100), 2)
        self.assertEqual(self.store.count_signal_reviews_since(T0 + 201), 0)


class TestEquity(StoreCase):
    def test_set_replace_and_isolation(self):
        self.assertEqual(self.store.equity_curve("s1"), [])
        pts = [(T0, 10000.0), (T0 + 10, 10050.5), (T0 + 10, 10020.0), (T0 + 20, 9990.25)]
        self.store.set_equity_curve("s1", pts)
        self.store.set_equity_curve("s2", [(T0, 5.0)])
        self.assertEqual(self.store.equity_curve("s1"), pts)
        self.store.set_equity_curve("s1", [(T0, 10000.0), (T0 + 5, 10001.0)])
        self.assertEqual(self.store.equity_curve("s1"), [(T0, 10000.0), (T0 + 5, 10001.0)])
        self.assertEqual(self.store.equity_curve("s2"), [(T0, 5.0)])
        self.store.set_equity_curve("s1", [])
        self.assertEqual(self.store.equity_curve("s1"), [])

    def test_types(self):
        self.store.set_equity_curve("s1", [(T0 + 0.0, 100)])
        (t, e), = self.store.equity_curve("s1")
        self.assertIsInstance(t, int)
        self.assertIsInstance(e, float)


class TestJournal(StoreCase):
    def test_add_and_list(self):
        a = self.store.add_journal("research", "r1", "body1", now_t=T0)
        b = self.store.add_journal("system", "s1", "body2", now_t=T0 + 5)
        c = self.store.add_journal("research", "r2", "body3", now_t=T0 + 5)
        self.assertEqual([r["id"] for r in self.store.journal()], [c, b, a])
        self.assertEqual(self.store.journal(limit=1),
                         [{"id": c, "t": T0 + 5, "kind": "research", "title": "r2", "body": "body3"}])
        self.assertEqual([r["id"] for r in self.store.journal(kind="research")], [c, a])
        self.assertEqual(self.store.journal(kind="promotion"), [])

    def test_kinds(self):
        for k in sorted(JOURNAL_KINDS):
            self.store.add_journal(k, k, "")
        self.assertEqual(len(self.store.journal(limit=None)), len(JOURNAL_KINDS))
        with self.assertRaises(ValueError):
            self.store.add_journal("misc", "x", "y")

    def test_default_time(self):
        before = int(time.time())
        self.store.add_journal("system", "x", "y")
        self.assertTrue(before <= self.store.journal()[0]["t"] <= int(time.time()))


class TestKV(StoreCase):
    def test_round_trip_default_and_overwrite(self):
        self.assertIsNone(self.store.get_kv("missing"))
        self.assertEqual(self.store.get_kv("missing", 7), 7)
        value = {"t": T0, "list": [1, 2.5, "x", None, True], "nested": {"a": {"b": []}}}
        self.store.set_kv("state", value)
        self.assertEqual(self.store.get_kv("state"), value)
        self.store.set_kv("state", 3)
        self.assertEqual(self.store.get_kv("state"), 3)
        for v in ["text", 1.25, [1, 2], True, ""]:
            self.store.set_kv("k", v)
            self.assertEqual(self.store.get_kv("k", "default"), v)

    def test_stored_null_is_not_default(self):
        self.store.set_kv("k", None)
        self.assertIsNone(self.store.get_kv("k", "default"))


if __name__ == "__main__":
    unittest.main()
