"""Regression tests for the review findings of the "stats_gate" group (fibbot/metrics.py and
fibbot/graduation.py):

F07  graduation is re-evaluated on every scheduler pass, so its evidence check must be valid under
     continuous monitoring (anytime-valid confidence sequence instead of a fixed-sample CI).
F09  trades on correlated coins are clustered in time: the CI, the deflated Sharpe and the
     graduation evidence use the week-clustered variance / effective trade count.
F05  profit_factor is gross winning R / gross losing R (matches risk-based sizing), not ret_pct.
F23  clearly losing mature paper records are "failing" (time-out, validation floor) or "lagging",
     never "promising".
F52  graduation checks follow the max_* / min_* direction convention (shortfall reported as such).

Synthetic data only; the Monte Carlo tests are seeded and deterministic.
"""
import math
import random
import statistics
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fibbot.config import DEFAULTS, deep_merge
from fibbot.graduation import STATUSES, TIMEOUT_FACTOR, evaluate
from fibbot.metrics import (bootstrap_mean_ci, clustered_t_stat, confidence_sequence, deflated_sharpe,
                            design_effect, effective_n, entry_week, probabilistic_sharpe, skew_kurt,
                            t_stat_mean, trade_stats)
from fibbot.portfolio import simulate_portfolio
from fibbot.store import Store
from fibbot.types import Trade

DAY = 86400
WEEK = 7 * DAY
T0 = 1_700_000_000 - 1_700_000_000 % WEEK          # a UTC week boundary
P = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
GRAD = DEFAULTS["graduation"]
CFG = {"graduation": GRAD, "paper": {"starting_equity": 10000.0}, "costs": DEFAULTS["costs"],
       "portfolio": DEFAULTS["portfolio"]}


def mk(r, i=0, risk=0.01, entry_t=None, exit_t=None, symbol="AAAUSDT", sid="s1"):
    """Closed trade with r_multiple r and ret_pct = r * risk."""
    entry_t = T0 + i * 3600 if entry_t is None else int(entry_t)
    exit_t = entry_t + 3600 if exit_t is None else int(exit_t)
    return Trade(sid, symbol, "4h", 1, f"L{i}", entry_t, entry_t, 100.0, 100.0 * (1 - risk), exit_t,
                 100.0 * (1 + r * risk), "target" if r > 0 else "stop", r * risk, risk, r, 3, {})


# -- a minimal store for Monte Carlo runs of the REAL graduation.evaluate -----------------------

class FakeStore:
    """Just what graduation.evaluate reads; rows carry their prebuilt Trade."""

    def __init__(self, rows, val_r=0.3, n_paper=1, start=T0):
        self.rows, self.val_r, self.n_paper, self.start = rows, val_r, n_paper, start

    def get_strategy(self, sid):
        return {"strategy_id": sid, "paper_start_t": self.start,
                "validation": {"metrics": {"avg_r": self.val_r}}}

    def paper_trades(self, sid):
        return self.rows

    @staticmethod
    def get_kv(key, default=None):       # no frozen paper account settings
        return default

    @staticmethod
    def trade_from_row(row):
        return row["trade"]

    def equity_curve(self, sid):
        return []           # account_drawdown falls back to the cumulative pnl of closed trades

    def count_ever_on_paper(self):
        return self.n_paper


def stream(rng, mean_r, days=365, rate=0.66):
    """Paper trades of one strategy: Poisson entries (rate per day), holding 0.2-4 days,
    R = +2 with probability (1 + mean_r) / 3 else -1 (mean mean_r), $50 risk (0.5% of $10k)."""
    rows, t, p = [], float(T0), (1.0 + mean_r) / 3.0
    while True:
        t += rng.expovariate(rate) * DAY
        if t > T0 + days * DAY:
            return sorted(rows, key=lambda row: row["exit_t"])
        r = 2.0 if rng.random() < p else -1.0
        tr = mk(r, len(rows), risk=0.03, entry_t=t, exit_t=t + rng.uniform(0.2, 4.0) * DAY,
                symbol=f"S{rng.randrange(8)}USDT", sid="x")
        rows.append({"exit_t": tr.exit_t, "pnl": 50.0 * r, "trade": tr})


def ever_graduated(rows, every=3, days=365):
    """Monitor like the scheduler: re-evaluate after closed trades arrive (every `every`-th close)
    and return True if any evaluation said "graduated". Evaluations that cannot graduate (too
    early, too few trades, total R <= 0) are skipped; they cannot change the answer."""
    min_t = T0 + GRAD["min_days"] * DAY
    for k in range(GRAD["min_trades"], len(rows) + 1):
        now = max(rows[k - 1]["exit_t"], min_t)
        if now > T0 + days * DAY:
            break
        if k < len(rows) and rows[k]["exit_t"] <= now:
            continue                      # more trades closed by then: evaluate with all of them
        if (k - GRAD["min_trades"]) % every and k != len(rows):
            continue
        closed = rows[:k]
        if math.fsum(row["trade"].r_multiple for row in closed) <= 0:
            continue
        if evaluate(FakeStore(closed), CFG, "x", now_t=now)["status"] == "graduated":
            return True
    return False


class TestF07ContinuousMonitoring(unittest.TestCase):
    def test_zero_edge_rarely_graduates_when_checked_after_every_few_trades(self):
        # Before the fix (fixed-sample iid bootstrap CI re-tested on every pass) 15 of 100 such
        # zero-edge strategies graduated at some point within the year (1-2% at a single look).
        rng = random.Random(20261103)
        paths = 80
        hits = sum(ever_graduated(stream(rng, 0.0)) for _ in range(paths))
        self.assertLessEqual(hits / paths, 0.03, f"{hits} of {paths} zero-edge strategies graduated")

    def test_a_real_edge_still_graduates(self):
        # the gate stays passable: a +0.3R edge at 0.66 trades/day graduates within a year in about
        # 55-65% of the paths (+0.25R: ~40%; +0.2R within two years: ~50%)
        rng = random.Random(7)
        paths = 40
        hits = sum(ever_graduated(stream(rng, 0.3)) for _ in range(paths))
        self.assertGreaterEqual(hits / paths, 0.35, f"only {hits} of {paths} +0.3R strategies graduated")

    def test_the_mean_r_interval_is_anytime_valid(self):
        # A zero-mean stream checked after every 4th value from 40 to 400: a fixed 95% CI excludes 0
        # at some point several times more often than its nominal 2.5%; the confidence sequence
        # stays below that nominal rate.
        rng = random.Random(11)
        fixed = anytime = 0
        paths = 150
        for _ in range(paths):
            xs = [2.0 if rng.random() < 1 / 3 else -1.0 for _ in range(400)]
            f = a = False
            for n in range(40, 401, 4):
                head = xs[:n]
                mean, s2 = math.fsum(head) / n, math.fsum(x * x for x in head) / n
                f = f or mean - 1.96 * math.sqrt((s2 - mean * mean) / (n - 1)) > 0
                a = a or confidence_sequence(head)[0] > 0
            fixed += f
            anytime += a
        self.assertGreater(fixed / paths, 0.06)
        self.assertLessEqual(anytime / paths, 0.025)

    def test_a_record_that_only_clears_a_fixed_ci_does_not_graduate(self):
        # 64 trades with mean R 0.5: the fixed-sample bootstrap CI excludes 0, which used to be
        # enough; the anytime-valid interval does not, because the gate is re-checked every hour.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / "fib.sqlite")
        self.addCleanup(store.close)
        cfg = deep_merge(DEFAULTS, {"home": tmp.name})
        store.add_to_roster("g", "g", {"timeframe": "4h"}, P, {"metrics": {"avg_r": 0.3}}, "")
        rs = [2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 2.0] * 8                          # mean 0.5
        for i, r in enumerate(rs):
            tr = mk(r, i, risk=0.02, entry_t=P + DAY + i * 1.5 * DAY, exit_t=P + 2 * DAY + i * 1.5 * DAY,
                    symbol=f"S{i % 6}USDT", sid="g")
            store.upsert_paper_trade("g", tr, 250.0, 250.0 * tr.ret_pct)
        res = evaluate(store, cfg, "g", now_t=P + 100 * DAY)
        self.assertGreater(bootstrap_mean_ci(rs)[0], 0)              # the old check passed
        checks = {c["name"]: c for c in res["checks"]}
        self.assertFalse(checks["mean_r_ci_lower"]["passed"])
        self.assertLessEqual(res["mean_r_ci"][0], 0)
        self.assertEqual({c["name"] for c in res["checks"] if not c["passed"]}, {"mean_r_ci_lower"})
        self.assertEqual(res["status"], "promising")

    def test_the_sequence_is_wider_than_a_fixed_ci_and_shrinks_with_n(self):
        rng = random.Random(3)
        xs = [rng.gauss(0.1, 1.0) for _ in range(1000)]
        widths = []
        for n in (40, 100, 200, 500, 1000):
            lo, hi = confidence_sequence(xs[:n])
            mean, sd = statistics.fmean(xs[:n]), statistics.stdev(xs[:n])
            self.assertAlmostEqual((lo + hi) / 2, mean, places=9)
            self.assertGreater(hi - lo, 2 * 1.96 * sd / math.sqrt(n))
            widths.append(hi - lo)
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertLess(confidence_sequence(xs[:200], level=0.5)[1] - confidence_sequence(xs[:200], level=0.5)[0],
                        widths[2])

    def test_confidence_sequence_edges(self):
        self.assertEqual(confidence_sequence([]), (0.0, 0.0))
        self.assertEqual(confidence_sequence([0.4]), (-99.0, 99.0))
        lo, hi = confidence_sequence([0.4, 0.4, 0.4])
        self.assertAlmostEqual(lo, 0.4)
        self.assertAlmostEqual(hi, 0.4)
        self.assertEqual(confidence_sequence([1.0, 2.0], level=1.0), (-99.0, 99.0))
        with self.assertRaises(ValueError):
            confidence_sequence([1.0, 2.0], groups=[1])


# -- F09: clustered trades ------------------------------------------------------------------------

def clustered_sample(rng, weeks=13, per_week=4):
    """Zero-mean trades, per_week per week: r = f_week + e with var(f) = var(e) (ICC 0.5)."""
    xs, gs = [], []
    for w in range(weeks):
        f = rng.gauss(0.0, 1.0)
        for _ in range(per_week):
            xs.append(f + rng.gauss(0.0, 1.0))
            gs.append(w)
    return xs, gs


class TestF09Clustering(unittest.TestCase):
    def test_design_effect_hand_computed(self):
        self.assertEqual(design_effect([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)       # singletons
        self.assertAlmostEqual(design_effect([1, 2, 3, 4], [1, 1, 2, 2]), 2.4)  # 2/1 * 8 * 3 / (4 * 5)
        self.assertEqual(design_effect([1, 2, 3, 4], [7, 7, 7, 7]), 4.0)        # one cluster = one draw
        self.assertEqual(design_effect([1, 2, 1, 2], [1, 1, 2, 2]), 1.0)        # clipped at 1
        self.assertEqual(design_effect([3.0] * 4, [1, 1, 2, 2]), 1.0)
        self.assertEqual(design_effect([], []), 1.0)
        self.assertEqual(effective_n([], []), 0.0)
        self.assertAlmostEqual(effective_n([1, 2, 3, 4], [1, 1, 2, 2]), 4 / 2.4)
        self.assertAlmostEqual(clustered_t_stat([1, 2, 3, 4], [1, 1, 2, 2]),
                               t_stat_mean([1, 2, 3, 4]) / math.sqrt(2.4))
        with self.assertRaises(ValueError):
            design_effect([1.0, 2.0], [1])
        self.assertEqual(entry_week(T0 + WEEK - 1), entry_week(T0))
        self.assertEqual(entry_week(T0 + WEEK), entry_week(T0) + 1)

    def test_copies_of_the_same_outcomes_add_no_evidence(self):
        # 30 independent weekly trades, then each copied 3x in its week (correlated coins taking
        # the same setup together). The iid statistics treat the copies as new evidence; the
        # clustered ones do not.
        rng = random.Random(5)
        xs = [rng.choice([2.0, -1.0, -1.0, 1.5]) for _ in range(30)]
        weeks = list(range(30))
        dup = [x for x in xs for _ in range(3)]
        dup_weeks = [w for w in weeks for _ in range(3)]
        # iid: narrower CI and a higher PSR from zero new information (the bug)
        lo, hi = bootstrap_mean_ci(xs)
        dlo, dhi = bootstrap_mean_ci(dup)
        self.assertLess(dhi - dlo, 0.7 * (hi - lo))
        self.assertGreater(deflated_sharpe(dup, 1), deflated_sharpe(xs, 1) + 0.03)
        # clustered: unchanged
        self.assertEqual(bootstrap_mean_ci(dup, groups=dup_weeks), bootstrap_mean_ci(xs, groups=weeks))
        self.assertAlmostEqual(effective_n(dup, dup_weeks), 30, delta=1.0)
        self.assertAlmostEqual(deflated_sharpe(dup, 1, n_eff=effective_n(dup, dup_weeks)),
                               deflated_sharpe(xs, 1, n_eff=effective_n(xs, weeks)), delta=0.01)
        clo, chi = confidence_sequence(xs, groups=weeks)
        dclo, dchi = confidence_sequence(dup, groups=dup_weeks)
        self.assertAlmostEqual(dchi - dclo, chi - clo, delta=0.02 * (chi - clo))

    def test_size_of_the_clustered_ci_and_deflated_sharpe(self):
        # zero edge, 4 trades a week for 13 weeks with a common weekly factor (design effect 2.5):
        # the iid statistics reject far above their nominal rates, the clustered ones near them
        rng = random.Random(42)
        seeds = 300
        ci = dsr = dsr_iid = 0
        for s in range(seeds):
            xs, gs = clustered_sample(rng)
            ci += bootstrap_mean_ci(xs, n_boot=400, seed=s, groups=gs)[0] > 0
            dsr += deflated_sharpe(xs, 1, n_eff=effective_n(xs, gs)) >= 0.95
            dsr_iid += deflated_sharpe(xs, 1) >= 0.95
        self.assertLessEqual(ci / seeds, 0.06)          # nominal 0.025 (percentile bootstrap, 13 clusters)
        self.assertLessEqual(dsr / seeds, 0.08)         # nominal 0.05
        self.assertGreater(dsr_iid / seeds, 0.10)       # the uncorrected version

    def test_no_power_is_lost_on_independent_trades(self):
        # one trade per week: design effect exactly 1, so every corrected statistic equals the iid one
        rng = random.Random(9)
        xs = [rng.gauss(0.2, 1.0) for _ in range(80)]
        weeks = list(range(80))
        self.assertEqual(design_effect(xs, weeks), 1.0)
        self.assertEqual(effective_n(xs, weeks), 80.0)
        self.assertEqual(deflated_sharpe(xs, 5, n_eff=effective_n(xs, weeks)), deflated_sharpe(xs, 5))
        self.assertAlmostEqual(clustered_t_stat(xs, weeks), t_stat_mean(xs))
        lo, hi = bootstrap_mean_ci(xs, groups=weeks)
        ilo, ihi = bootstrap_mean_ci(xs)
        self.assertAlmostEqual(lo, ilo, places=12)
        self.assertAlmostEqual(hi, ihi, places=12)
        self.assertEqual(confidence_sequence(xs, groups=weeks), confidence_sequence(xs))

    def test_deflated_sharpe_with_n_eff_is_the_psr_at_n_eff(self):
        rng = random.Random(4)
        xs = [rng.gauss(0.15, 1.0) for _ in range(120)]
        sk, ku = skew_kurt(xs)
        sr = statistics.fmean(xs) / statistics.stdev(xs)
        self.assertAlmostEqual(deflated_sharpe(xs, 1, n_eff=50), probabilistic_sharpe(sr, 50, sk, ku))
        self.assertLess(deflated_sharpe(xs, 1, n_eff=50), deflated_sharpe(xs, 1))
        self.assertEqual(deflated_sharpe(xs, 1, n_eff=1e9), deflated_sharpe(xs, 1))    # clipped to n
        self.assertAlmostEqual(deflated_sharpe(xs, 1, n_eff=0.2), 0.5)                   # clipped to 1

    def test_graduation_uses_week_clusters(self):
        # the same 60 winning outcomes, all entered in 3 weeks instead of spread over 12: same
        # mean, but far less independent evidence
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / "fib.sqlite")
        self.addCleanup(store.close)
        cfg = deep_merge(DEFAULTS, {"home": tmp.name})
        rs = [2.0, -1.0, 1.5, 2.0, -1.0, 1.0] * 10
        out = {}
        for sid, span in (("spread", 84), ("bunched", 20)):
            store.add_to_roster(sid, sid, {"timeframe": "4h"}, P, {"metrics": {"avg_r": 0.6}}, "")
            for i, r in enumerate(rs):
                # bunched: the winners of a week all fall in one week, the losers in another
                entry = P + DAY + (i * span * DAY) // len(rs) if sid == "spread" else \
                    P + DAY + (0 if r > 0 else 7 * DAY) + (i % 7) * 3600 + (14 * DAY if i >= 30 else 0)
                tr = mk(r, i, risk=0.02, entry_t=entry, exit_t=entry + DAY, symbol=f"S{i % 6}USDT", sid=sid)
                store.upsert_paper_trade(sid, tr, 250.0, 250.0 * tr.ret_pct)
            out[sid] = evaluate(store, cfg, sid, now_t=P + 90 * DAY)
        self.assertGreater(out["bunched"]["evidence"]["design_effect"], 5)
        self.assertLess(out["bunched"]["evidence"]["n_eff"], 12)
        self.assertLess(out["bunched"]["mean_r_ci"][0], out["spread"]["mean_r_ci"][0])
        dsr = {sid: next(c["value"] for c in r["checks"] if c["name"] == "min_deflated_sharpe")
               for sid, r in out.items()}
        self.assertLess(dsr["bunched"], dsr["spread"])
        self.assertIn("n_eff", out["bunched"]["summary"])
        self.assertEqual(out["spread"]["status"], "graduated", out["spread"]["summary"])
        self.assertNotEqual(out["bunched"]["status"], "graduated")


# -- F05: profit factor on R --------------------------------------------------------------------

class TestF05ProfitFactor(unittest.TestCase):
    def test_wide_stop_winner_and_tight_stop_losers(self):
        # +1R on an 8% stop, two -1.1R on 2.5% stops: the account loses (risk sizing), so PF < 1
        trades = [mk(1.0, 0, risk=0.08), mk(-1.1, 1, risk=0.025), mk(-1.1, 2, risk=0.025)]
        s = trade_stats(trades)
        self.assertAlmostEqual(s["profit_factor"], 1 / 2.2)       # was 0.08 / 0.055 = 1.45
        self.assertLess(s["total_r"], 0)

    def test_profit_factor_above_one_iff_total_r_positive(self):
        rng = random.Random(1)
        for _ in range(200):
            trades = [mk(rng.choice([2.0, 1.0, -1.0, -1.1, 0.4]), i, risk=rng.uniform(0.005, 0.1))
                      for i in range(rng.randrange(2, 15))]
            s = trade_stats(trades)
            if abs(s["total_r"]) > 1e-9:
                self.assertEqual(s["profit_factor"] > 1, s["total_r"] > 0, s)

    def test_profit_factor_matches_the_risk_sized_account(self):
        # uncapped, non-overlapping trades: every P&L is risk_per_trade * equity * R, so the
        # account's gross profit / gross loss equals the R profit factor (up to compounding)
        rng = random.Random(2)
        trades = [mk(rng.choice([2.0, 1.0, -1.0, -1.1, 0.4]), i, risk=rng.uniform(0.005, 0.1),
                     entry_t=T0 + i * DAY, exit_t=T0 + i * DAY + 3600, symbol=f"S{i % 5}USDT")
                  for i in range(60)]
        pcfg = {"risk_per_trade": 1e-5, "max_notional_frac": 1e6, "max_open_positions": 100,
                "max_total_risk": 1.0, "daily_loss_limit": 0.0}
        taken = simulate_portfolio(trades, pcfg).taken
        self.assertEqual(len(taken), 60)
        pnl = [rec["pnl"] for rec in taken]
        account_pf = math.fsum(p for p in pnl if p > 0) / -math.fsum(p for p in pnl if p < 0)
        self.assertAlmostEqual(trade_stats(trades)["profit_factor"], account_pf, delta=1e-3)


# -- graduation statuses and check conventions (F23, F52) --------------------------------------

class GradCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(Path(tmp.name) / "fib.sqlite")
        self.addCleanup(self.store.close)
        self.cfg = deep_merge(DEFAULTS, {"home": tmp.name})

    def record(self, rs, days, val_r=0.25, sid="g", per_trade_usd=50.0):
        """Closed trades with R multiples rs spread over `days` from P + 1 day, 6 symbols."""
        self.store.add_to_roster(sid, sid, {"timeframe": "4h"}, P,
                                 {"metrics": {"avg_r": val_r}} if val_r is not None else {}, "")
        for i, r in enumerate(rs):
            entry = P + DAY + (i * (days - 2) * DAY) // len(rs)
            tr = mk(r, i, risk=0.02, entry_t=entry, exit_t=entry + DAY, symbol=f"S{i % 6}USDT", sid=sid)
            self.store.upsert_paper_trade(sid, tr, per_trade_usd / 0.02, per_trade_usd * r)

    def evaluate(self, days, sid="g"):
        return evaluate(self.store, self.cfg, sid, now_t=P + days * DAY)


def losing_mix(n, rng_seed=0):
    """n R multiples, mean about -0.07R: 31% at +2R, 69% at -1R, shuffled deterministically."""
    k = round(0.31 * n)
    rs = [2.0] * k + [-1.0] * (n - k)
    random.Random(rng_seed).shuffle(rs)
    return rs


class TestF23LosersAreNotPromising(GradCase):
    def test_a_long_losing_record_is_failing(self):
        rs = losing_mix(200)
        self.record(rs, days=400)
        res = self.evaluate(400)
        self.assertLess(statistics.fmean(rs), 0)
        self.assertGreater(res["mean_r_ci"][1], 0)          # not a "significant" loss ...
        self.assertEqual(res["status"], "failing", res["summary"])   # ... but it timed out
        self.assertIn("mean R <= 0 after 400 days", res["summary"])

    def test_a_mature_negative_record_is_lagging_before_the_time_out(self):
        self.record(losing_mix(80), days=100)
        res = self.evaluate(100)
        self.assertLess(100, TIMEOUT_FACTOR * self.cfg["graduation"]["min_days"])
        self.assertEqual(res["status"], "lagging", res["summary"])
        self.assertIn("lagging", STATUSES)

    def test_a_positive_but_unproven_record_is_not_failing(self):
        rs = ([2.0] * 23 + [-1.0] * 37)            # 60 trades, mean +0.15R, wide interval
        random.Random(1).shuffle(rs)
        self.record(rs, days=90, val_r=0.6)
        res = self.evaluate(90)
        self.assertGreater(res["mean_r_ci"][1], 0.35)
        self.assertEqual(res["status"], "promising", res["summary"])
        res = self.evaluate(400)                   # same record, long after: positive, not timed out
        self.assertEqual(res["status"], "promising", res["summary"])

    def test_far_below_the_validation_floor_is_failing(self):
        # mature record whose whole mean-R interval lies below validation mean R - 0.25
        rs = [0.2, -0.1, 0.1, 0.0] * 30             # mean +0.05, tight
        self.record(rs, days=90, val_r=1.0)
        res = self.evaluate(90)
        self.assertLess(res["mean_r_ci"][1], 0.75)
        self.assertEqual(res["status"], "failing", res["summary"])
        self.assertIn("validation floor", res["summary"])
        # without a validation mean R the same (small but steady) edge graduates
        self.record(rs, days=90, val_r=None, sid="h")
        self.assertEqual(self.evaluate(90, sid="h")["status"], "graduated")


class TestF52CheckConventions(GradCase):
    def assertConventions(self, res):
        for c in res["checks"]:
            if c["value"] is None:
                continue
            if c["name"].startswith("max_"):
                self.assertEqual(c["passed"], c["value"] <= c["threshold"], c)
            elif c["name"].startswith("min_"):
                self.assertEqual(c["passed"], c["value"] >= c["threshold"], c)

    def test_every_check_follows_its_name(self):
        for k, (rs, days, val) in enumerate([([2.0, -1.0, 1.5, 2.0, -1.0, 1.0] * 10, 90, 0.6),
                                             ([2.0, -1.0, 1.5, 2.0, -1.0, 1.0] * 10, 90, 1.5),
                                             (losing_mix(80), 100, 0.3), ([1.0, -1.0] * 5, 20, 0.1)]):
            self.record(rs, days=days, val_r=val, sid=f"s{k}")
            self.assertConventions(self.evaluate(days, sid=f"s{k}"))

    def test_shortfall_flips_at_the_floor(self):
        base = [2.0, -1.0, 1.5, 2.0, -1.0, 1.0] * 10          # mean R 0.75
        for sid, val_r, passed in (("above", 0.99, True), ("below", 1.01, False)):
            self.record(base, days=90, val_r=val_r, sid=sid)
            c = next(c for c in self.evaluate(90, sid=sid)["checks"]
                     if c["name"] == "max_shortfall_vs_validation_r")
            self.assertAlmostEqual(c["value"], val_r - 0.75, places=6)
            self.assertEqual(c["threshold"], 0.25)
            self.assertEqual(c["passed"], passed, c)


if __name__ == "__main__":
    unittest.main()
