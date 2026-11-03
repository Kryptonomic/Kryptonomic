"""Regression tests for the data-layer review findings (fibbot/data/candles.py, universe.py):

F34  rate limits and regional blocks must not burn scheduled-run minutes: 401/403/418/451 are
     never retried, a Retry-After longer than the caller's max_delay gives up at once, the
     CoinGecko ranking call makes at most 2 tries with <= 10 s of waiting, and a failed ranking
     is recorded in universe.json and not retried for RANK_RETRY_SECONDS (refresh=True overrides);
F50  a Binance bar counts as closed only CLOSED_MARGIN_MS before the local clock, and
     CandleStore.update re-fetches the newest cached bar so one stored while still forming (a
     local clock ahead of the exchange) is corrected; the OKX fallback never revises it.

No network: urlopen / http_json are mocked and time.time is patched."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.parse
from unittest import mock

from fibbot.data import candles
from fibbot.data.candles import (CLOSED_MARGIN_MS, CandleStore, FetchError, fetch_klines,
                                 http_json, parse_day)
from fibbot.data.universe import (GECKO_MAX_DELAY, RANK_RETRY_SECONDS, _params, get_universe)
from fibbot.types import TF_SECONDS, Bars
from tests.test_data import (FORMING, NOW, U_NOW, ClockMixin, FakeHttp, FakeResp, UniverseCase,
                             failing, http_error, make_cfg, okx_handler)

H = 3600
D = 86400
MARKETS = "/api/v3/coins/markets"


class SleepMixin:
    def start_sleep(self):
        p = mock.patch("fibbot.data.candles.time.sleep")
        self.sleep = p.start()
        self.addCleanup(p.stop)

    def slept(self) -> float:
        return sum(c.args[0] for c in self.sleep.call_args_list)


class UrlopenRouter:
    """Stands in for urllib.request.urlopen: routes by URL path suffix to handler(params) whose
    return value is served as JSON (or which raises, e.g. an HTTPError). Records every URL."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.urls: list[str] = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.urls.append(url)
        parts = urllib.parse.urlsplit(url)
        params = {k: v[-1] for k, v in urllib.parse.parse_qs(parts.query).items()}
        for path, fn in self.routes.items():
            if parts.path.endswith(path):
                return FakeResp(json.dumps(fn(params)).encode())
        raise AssertionError(f"unexpected url {url}")

    def count(self, path: str) -> int:
        return sum(urllib.parse.urlsplit(u).path.endswith(path) for u in self.urls)


# ---------------------------------------------------------------------------------------------
# F34: http_json retry policy


class TestNoRetryStatus(SleepMixin, unittest.TestCase):
    def setUp(self):
        self.start_sleep()

    def test_blocked_and_banned_codes_fail_at_once(self):
        for code in (401, 403, 418, 451):
            with mock.patch("urllib.request.urlopen", side_effect=http_error(code, b"blocked")) as uo:
                with self.assertRaises(FetchError) as cm:
                    http_json("https://h/p")
            self.assertEqual(uo.call_count, 1, code)
            self.assertEqual(cm.exception.status, code)
        self.sleep.assert_not_called()

    def test_retry_after_beyond_max_delay_gives_up_at_once(self):
        with mock.patch("urllib.request.urlopen", side_effect=http_error(429, retry_after="60")) as uo:
            with self.assertRaises(FetchError) as cm:
                http_json("https://h/p", max_delay=10)
        self.assertEqual((uo.call_count, cm.exception.status), (1, 429))
        self.assertIn("Retry-After", str(cm.exception))
        # the default limit (BACKOFF_MAX = 60 s) too: retrying before Retry-After only earns
        # another refusal (on Binance, eventually an IP ban)
        with mock.patch("urllib.request.urlopen", side_effect=http_error(503, retry_after="3600")) as uo:
            with self.assertRaises(FetchError):
                http_json("https://h/p")
        self.assertEqual(uo.call_count, 1)
        self.sleep.assert_not_called()

    def test_retry_after_within_max_delay_is_honoured(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=[http_error(429, retry_after="8"), FakeResp(b"[1]")]) as uo:
            self.assertEqual(http_json("https://h/p", max_delay=10), [1])
        self.assertEqual(uo.call_count, 2)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [8.0])

    def test_max_delay_caps_exponential_backoff(self):
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("slow")) as uo:
            with self.assertRaises(FetchError):
                http_json("https://h/p", tries=5, max_delay=3)
        self.assertEqual(uo.call_count, 5)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1.0, 2.0, 3.0, 3.0])


# ---------------------------------------------------------------------------------------------
# F34: the CoinGecko ranking call is cheap to fail


class TestRankingRateLimit(SleepMixin, ClockMixin, unittest.TestCase):
    now = U_NOW

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.start_clock()
        self.start_sleep()
        self.cfg = make_cfg(self.tmp.name, universe={"top_n": 3})
        self.cached = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        with open(os.path.join(self.tmp.name, "universe.json"), "w") as f:   # 8 days old: stale
            json.dump({"as_of": U_NOW - 8 * D, "symbols": self.cached, "params": _params(self.cfg)}, f)

    def rank_once(self, error) -> UrlopenRouter:
        def markets(p):
            raise error
        router = UrlopenRouter({MARKETS: markets})
        with mock.patch("urllib.request.urlopen", router), \
                self.assertLogs("fibbot.data.universe", "WARNING"):
            self.assertEqual(get_universe(self.cfg), self.cached)
        return router

    def test_429_with_long_retry_after_costs_no_sleep(self):
        router = self.rank_once(http_error(429, retry_after="60"))
        self.assertEqual(router.count(MARKETS), 1)
        self.assertLessEqual(self.slept(), GECKO_MAX_DELAY)
        self.assertEqual(self.slept(), 0)

    def test_429_without_retry_after_tries_twice(self):
        router = self.rank_once(http_error(429))
        self.assertEqual(router.count(MARKETS), 2)
        self.assertLessEqual(self.slept(), GECKO_MAX_DELAY)

    def test_gateway_errors_bounded_too(self):
        router = self.rank_once(http_error(502, retry_after="9"))
        self.assertEqual(router.count(MARKETS), 2)
        self.assertEqual(self.slept(), 9.0)


# ---------------------------------------------------------------------------------------------
# F34: a failed ranking is remembered across processes


class TestRankingFailureBackoff(UniverseCase):
    def path(self):
        return os.path.join(self.tmp.name, "universe.json")

    def saved(self) -> dict:
        with open(self.path()) as f:
            return json.load(f)

    def markets_calls(self) -> int:
        return len(self.fake.params(MARKETS))

    def fail_ranking(self, cfg, **kw) -> list[str]:
        with self.assertLogs("fibbot.data.universe", "WARNING"):
            return get_universe(cfg, **kw)

    def test_failed_ranking_backs_off(self):
        cfg = self.cfg(top_n=3)
        listed = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.assertEqual(get_universe(cfg), listed)
        ranked = self.saved()
        self.now += 8 * D                                       # stale: refresh_days is 7
        self.fake.routes[MARKETS] = failing
        self.assertEqual(self.fail_ranking(cfg), listed)
        rec = self.saved()
        for key in ("as_of", "symbols", "params"):               # staleness stays visible
            self.assertEqual(rec[key], ranked[key], key)
        self.assertEqual(rec["failed_t"], self.now)
        self.assertIn("503", rec["failed_error"])
        failed_at, n, calls = self.now, self.markets_calls(), len(self.fake.calls)

        self.now = failed_at + RANK_RETRY_SECONDS - 60          # a later run, inside the window
        self.assertEqual(get_universe(cfg), listed)
        self.assertEqual(len(self.fake.calls), calls)            # no network at all

        self.now = failed_at + RANK_RETRY_SECONDS + 60          # window over: retried (fails again)
        self.assertEqual(self.fail_ranking(cfg), listed)
        self.assertEqual(self.markets_calls(), n + 1)
        self.assertEqual(self.saved()["failed_t"], self.now)
        self.assertEqual(self.saved()["as_of"], ranked["as_of"])

        self.assertEqual(self.fail_ranking(cfg, refresh=True), listed)   # refresh ignores the backoff
        self.assertEqual(self.markets_calls(), n + 2)

        self.fake.routes[MARKETS] = lambda p: self.markets      # CoinGecko is back
        self.assertEqual(get_universe(cfg), listed)
        self.assertEqual(self.markets_calls(), n + 2)            # still inside the window
        self.assertEqual(get_universe(cfg, refresh=True), listed)
        self.assertEqual(self.markets_calls(), n + 3)
        clean = self.saved()
        self.assertEqual(clean["as_of"], self.now)
        self.assertFalse({"failed_t", "failed_params", "failed_error"} & set(clean))

    def test_clock_before_failure_time_retries(self):
        cfg = self.cfg(top_n=3)
        get_universe(cfg)
        self.now += 8 * D
        self.fake.routes[MARKETS] = failing
        self.fail_ranking(cfg)
        n = self.markets_calls()
        self.now -= H                                           # clock stepped back
        self.fail_ranking(cfg)
        self.assertEqual(self.markets_calls(), n + 1)

    def test_settings_change_retries_once_then_backs_off(self):
        get_universe(self.cfg(top_n=3))
        self.fake.routes[MARKETS] = failing
        self.fail_ranking(self.cfg(top_n=3), refresh=True)
        n = self.markets_calls()
        cfg2 = self.cfg(top_n=2)
        self.assertEqual(self.fail_ranking(cfg2), ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(self.markets_calls(), n + 1)            # new settings: tried once
        self.assertEqual(get_universe(cfg2), ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(self.markets_calls(), n + 1)            # then backed off

    def test_failure_without_cache_backs_off(self):
        self.fake.routes[MARKETS] = failing
        cfg = self.cfg(top_n=2)
        self.assertEqual(self.fail_ranking(cfg), ["BTCUSDT", "ETHUSDT"])   # static fallback
        calls = len(self.fake.calls)
        self.now += H
        self.assertEqual(get_universe(cfg), ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(len(self.fake.calls), calls)

    def test_unwritable_failure_record_never_raises(self):
        self.fake.routes[MARKETS] = failing
        with mock.patch("fibbot.data.candles.atomic_write", side_effect=OSError("read-only")), \
                self.assertLogs("fibbot.data.universe", "WARNING") as logs:
            self.assertEqual(get_universe(self.cfg(top_n=2)), ["BTCUSDT", "ETHUSDT"])
        self.assertTrue(any("could not record" in m for m in logs.output))


# ---------------------------------------------------------------------------------------------
# F34: a geo-blocked Binance costs no sleeps in a candle pass


class TestGeoBlockedCandlePass(SleepMixin, ClockMixin, unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.start_clock()
        self.start_sleep()
        self.store = CandleStore(make_cfg(self.tmp.name))
        rows = [(t, 1.0, 2.0, 0.5, 1.5, 3.0) for t in range(FORMING - 50 * H, FORMING - 10 * H, H)]
        for sym in ("BTCUSDT", "ETHUSDT"):
            candles.atomic_write(self.store.path(sym, "1h"), candles._encode(Bars.from_rows(sym, "1h", rows)))

    def test_451_falls_through_to_okx_without_retries(self):
        def blocked(p):
            raise http_error(451, b'{"msg": "restricted location"}')
        router = UrlopenRouter({"/api/v3/klines": blocked,
                                "/api/v5/market/history-candles": okx_handler(FORMING - 400 * H, H)})
        with mock.patch("urllib.request.urlopen", router), \
                self.assertLogs("fibbot.data.candles", "WARNING"):
            res = self.store.update_all(["BTCUSDT", "ETHUSDT"], ["1h"])
        self.assertEqual(res, {("BTCUSDT", "1h"): 10, ("ETHUSDT", "1h"): 10})
        self.assertEqual(self.store.last_errors, {})
        self.assertEqual(router.count("/api/v3/klines"), 2)      # one refused request per pair
        self.sleep.assert_not_called()


# ---------------------------------------------------------------------------------------------
# F50: forming bars


class ExchangeKlines:
    """Binance klines as the exchange sees them at its own clock `self.true_now` (s): every bar
    from `first_t` up to the one containing true_now. Every bar opens at 100 and closes at
    o + 5, high o + 10, volume 12; the forming bar has only the traded fraction of that move."""

    def __init__(self, first_t: int, true_now: int):
        self.first_t, self.true_now = first_t, true_now

    def __call__(self, p):
        tf_s = TF_SECONDS[p["interval"]]
        start = p["startTime"] // 1000
        t = max(start + (-start) % tf_s, self.first_t)
        rows = []
        while t <= self.true_now and t * 1000 <= p["endTime"] and len(rows) < p["limit"]:
            frac = min(1.0, (self.true_now - t) / tf_s)
            rows.append([t * 1000, "100", str(100 + 10 * frac), "99", str(100 + 5 * frac),
                         str(12 * frac), (t + tf_s) * 1000 - 1, "0", 3, "0", "0", "0"])
            t += tf_s
        return rows


class TestFormingBar(ClockMixin, unittest.TestCase):
    B = FORMING                                                 # the bar at issue opens here

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = make_cfg(self.tmp.name, data={"history_start": "2024-01-10"})
        self.start_clock()
        self.exchange = ExchangeKlines(parse_day("2024-01-10"), self.B)
        self.fake = FakeHttp({"/api/v3/klines": self.exchange,
                              "/api/v5/market/history-candles": failing})
        p = mock.patch("fibbot.data.candles.http_json", self.fake)
        p.start()
        self.addCleanup(p.stop)
        self.store = CandleStore(self.cfg)

    def at(self, true_now: int, skew: int = 0):
        """Exchange clock at true_now, local clock `skew` seconds ahead of it."""
        self.exchange.true_now, self.now = true_now, true_now + skew

    def test_bar_stored_while_forming_is_corrected_on_the_next_update(self):
        close = self.B + H
        self.at(close - 60, skew=120)                           # local clock 2 min fast
        first = self.store.update("BTCUSDT", "1h")
        self.assertEqual(first.last_t, self.B)                  # stored although still forming
        self.assertLess(first.h[-1], 110.0)

        self.at(close + 1800)                                   # next run, correct clock
        with self.assertLogs("fibbot.data.candles", "WARNING") as logs:
            res = self.store.update_all(["BTCUSDT"], ["1h"])
        self.assertTrue(any("differs from the cached copy" in m for m in logs.output))
        self.assertEqual(res, {("BTCUSDT", "1h"): 0})          # the re-fetch adds no bar
        self.assertEqual(self.fake.params("/api/v3/klines")[-1]["startTime"], self.B * 1000)
        for bars in (self.store.get("BTCUSDT", "1h"), CandleStore(self.cfg).get("BTCUSDT", "1h")):
            self.assertEqual(bars.t, first.t)
            self.assertEqual((bars.h[-1], bars.c[-1], bars.v[-1]), (110.0, 105.0, 12.0))
            self.assertEqual(bars.c[:-1], first.c[:-1])         # older bars untouched

    def test_bar_closing_within_the_margin_is_not_stored(self):
        close = self.B + H
        self.at(close - 1, skew=2)                              # 1 s after the close by the local clock
        self.assertLess(1000, CLOSED_MARGIN_MS)
        bars = self.store.update("BTCUSDT", "1h")
        self.assertEqual(bars.last_t, self.B - H)
        self.at(close + 90)
        bars = self.store.update("BTCUSDT", "1h")
        self.assertEqual((bars.last_t, bars.h[-1], bars.c[-1]), (self.B, 110.0, 105.0))

    def test_unchanged_refetch_rewrites_nothing(self):
        self.at(self.B + H + 90)
        bars = self.store.update("BTCUSDT", "1h")
        st = self.store.path("BTCUSDT", "1h").stat()
        self.at(self.B + H + 900)
        self.assertIs(self.store.update("BTCUSDT", "1h"), bars)
        self.assertEqual(self.fake.params("/api/v3/klines")[-1]["startTime"], self.B * 1000)
        self.assertEqual(self.store.path("BTCUSDT", "1h").stat().st_mtime_ns, st.st_mtime_ns)

    def test_okx_fallback_never_revises_the_cached_bar(self):
        self.at(self.B + H + 90)
        cached = self.store.update("BTCUSDT", "1h")
        self.assertEqual(cached.last_t, self.B)
        self.at(self.B + 4 * H + 90)
        self.fake.routes["/api/v3/klines"] = failing
        self.fake.routes["/api/v5/market/history-candles"] = okx_handler(self.B - 10 * H, H)
        with self.assertLogs("fibbot.data.candles", "WARNING"):
            bars = self.store.update("BTCUSDT", "1h")
        self.assertEqual(bars.t, cached.t + [self.B + H, self.B + 2 * H, self.B + 3 * H])
        i = bars.t.index(self.B)
        self.assertEqual((bars.o[i], bars.h[i], bars.c[i]), (100.0, 110.0, 105.0))   # Binance's copy
        self.assertEqual(bars.o[i + 1], 1000.0 + ((self.B + H) // H) % 11)          # OKX appended

    def test_fetch_klines_fallback_start(self):
        self.fake.routes["/api/v3/klines"] = failing
        self.fake.routes["/api/v5/market/history-candles"] = okx_handler(FORMING - 400 * H, H)
        self.now = NOW
        with self.assertLogs("fibbot.data.candles", "WARNING"):
            rows = fetch_klines(self.cfg, "BTCUSDT", "1h", FORMING - 5 * H, fallback_start_t=FORMING - 3 * H)
        self.assertEqual([r[0] for r in rows], [FORMING - 3 * H, FORMING - 2 * H, FORMING - H])
        with self.assertLogs("fibbot.data.candles", "WARNING"):    # never earlier than start_t
            rows = fetch_klines(self.cfg, "BTCUSDT", "1h", FORMING - 2 * H, fallback_start_t=0)
        self.assertEqual([r[0] for r in rows], [FORMING - 2 * H, FORMING - H])
        with self.assertRaises(FetchError):                      # eligibility uses the fallback start
            fetch_klines(self.cfg, "BTCUSDT", "1h", FORMING - 5000 * H, fallback_start_t=FORMING - 5000 * H)


if __name__ == "__main__":
    unittest.main()
