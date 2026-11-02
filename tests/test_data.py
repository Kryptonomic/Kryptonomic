"""Tests for fibbot.data (universe + candles). No network: http_json / urlopen are mocked."""
from __future__ import annotations

import email.message
import gzip
import io
import json
import os
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

from fibbot.config import load_config
from fibbot.data.candles import CandleStore, FetchError, fetch_klines, http_json, parse_day, resample
from fibbot.data.universe import binance_spot_symbols, get_universe, rank_universe
from fibbot.types import TF_SECONDS, Bars

H = 3600
D = 86400
LIST_T = parse_day("2024-01-01")                 # fake listing date of every kline symbol
FORMING = parse_day("2024-01-10") + 5 * H        # open of the 1h bar still forming at NOW
NOW = FORMING + 1800


def kline(t: int, tf_s: int) -> list:
    """A Binance kline row (prices as strings, like the real API)."""
    o = 100.0 + (t // tf_s) % 37
    return [t * 1000, str(o), str(o + 2.5), str(o - 1.25), str(o + 0.5), "12.5",
            (t + tf_s) * 1000 - 1, "0", 3, "0", "0", "0"]


def forming_open(tf_s: int) -> int:
    now = int(time.time())
    return now - now % tf_s


class FakeHttp:
    """Stands in for candles.http_json: routes by URL suffix to handler(params); records calls."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, params=None, **kw):
        self.calls.append((url, dict(params or {})))
        for path, fn in self.routes.items():
            if url.endswith(path):
                return fn(dict(params or {}))
        raise AssertionError(f"unexpected url {url}")

    def params(self, path: str) -> list[dict]:
        return [p for u, p in self.calls if u.endswith(path)]


class FakeBinance:
    """Serves klines from `first_t` up to (and including) the bar forming at time.time()."""

    def __init__(self, first_t: int = LIST_T, fail_symbols=()):
        self.first_t, self.fail_symbols = first_t, set(fail_symbols)

    def klines(self, p):
        if p["symbol"] in self.fail_symbols:
            raise FetchError("HTTP 400: Invalid symbol.", 400)
        tf_s = TF_SECONDS[p["interval"]]
        start = p["startTime"] // 1000
        t = max(start + (-start) % tf_s, self.first_t)
        last = min(p.get("endTime", 10 ** 16) // 1000, forming_open(tf_s))
        rows = []
        while t <= last and len(rows) < p["limit"]:
            rows.append(kline(t, tf_s))
            t += tf_s
        return rows


def okx_handler(first_t: int, tf_s: int):
    """OKX history-candles: rows with ts < after, newest first, forming bar has confirm "0"."""
    def fn(p):
        after_s = (int(p["after"]) - 1) // 1000
        t = min(after_s - after_s % tf_s, forming_open(tf_s))
        rows = []
        while t >= first_t and len(rows) < int(p["limit"]):
            o = 1000.0 + (t // tf_s) % 11
            rows.append([str(t * 1000), str(o), str(o + 3), str(o - 2), str(o + 1), "7.5", "1", "1",
                         "0" if t == forming_open(tf_s) else "1"])
            t -= tf_s
        return {"code": "0", "msg": "", "data": rows}
    return fn


def failing(p):
    raise FetchError("HTTP 503 after 5 tries", 503)


class ClockMixin:
    """Patches time.time to self.now (advance by assigning self.now)."""

    now = NOW

    def start_clock(self):
        patcher = mock.patch("time.time", side_effect=lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)


def make_cfg(tmp: str, **overrides) -> dict:
    return load_config(overrides={"home": tmp, **overrides})


# ---------------------------------------------------------------------------------------------


class FakeResp:
    def __init__(self, body: bytes, headers=None):
        self.body, self.headers = body, headers or {}

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code: int, body: bytes = b"", retry_after: str | None = None):
    hdrs = email.message.Message()
    if retry_after:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x", code, "err", hdrs, io.BytesIO(body))


class TestHttpJson(unittest.TestCase):
    def setUp(self):
        p = mock.patch("fibbot.data.candles.time.sleep")
        self.sleep = p.start()
        self.addCleanup(p.stop)

    def test_success_url_and_headers(self):
        with mock.patch("urllib.request.urlopen", return_value=FakeResp(b'{"a": [1, 2]}')) as uo:
            self.assertEqual(http_json("https://h/p", {"x": 1, "y": "z"}), {"a": [1, 2]})
        req = uo.call_args.args[0]
        self.assertEqual(req.full_url, "https://h/p?x=1&y=z")
        self.assertIn("fibbot", req.get_header("User-agent"))
        self.assertEqual(uo.call_args.kwargs["timeout"], 30.0)
        self.sleep.assert_not_called()

    def test_gzip_body(self):
        resp = FakeResp(gzip.compress(b'[1, 2, 3]'), {"Content-Encoding": "gzip"})
        with mock.patch("urllib.request.urlopen", return_value=resp):
            self.assertEqual(http_json("https://h/p"), [1, 2, 3])

    def test_retries_with_exponential_backoff(self):
        seq = [urllib.error.URLError("down"), http_error(503), FakeResp(b"not json"), FakeResp(b'{"ok": 1}')]
        with mock.patch("urllib.request.urlopen", side_effect=seq) as uo:
            self.assertEqual(http_json("https://h/p"), {"ok": 1})
        self.assertEqual(uo.call_count, 4)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1.0, 2.0, 4.0])

    def test_retry_after_is_honoured(self):
        with mock.patch("urllib.request.urlopen", side_effect=[http_error(429, retry_after="7"), FakeResp(b"1")]):
            self.assertEqual(http_json("https://h/p"), 1)
        self.assertEqual(self.sleep.call_args.args[0], 7.0)

    def test_404_and_400_not_retried(self):
        for code in (400, 404):
            with mock.patch("urllib.request.urlopen", side_effect=http_error(code, b'{"msg":"Invalid symbol."}')) as uo:
                with self.assertRaises(FetchError) as cm:
                    http_json("https://h/p")
            self.assertEqual(uo.call_count, 1)
            self.assertEqual(cm.exception.status, code)
            self.assertIn("Invalid symbol", str(cm.exception))
        self.sleep.assert_not_called()

    def test_gives_up_after_five_tries(self):
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("slow")) as uo:
            with self.assertRaises(FetchError) as cm:
                http_json("https://h/p")
        self.assertEqual(uo.call_count, 5)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1.0, 2.0, 4.0, 8.0])
        self.assertIsInstance(cm.exception, RuntimeError)


# ---------------------------------------------------------------------------------------------


class TestFetchKlines(ClockMixin, unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = make_cfg(self.tmp.name)
        self.start_clock()

    def patch_http(self, routes) -> FakeHttp:
        fake = FakeHttp(routes)
        p = mock.patch("fibbot.data.candles.http_json", fake)
        p.start()
        self.addCleanup(p.stop)
        return fake

    def test_pagination_and_closed_bars_only(self):
        start = FORMING - 2500 * H
        fake = self.patch_http({"/api/v3/klines": FakeBinance(start).klines})
        rows = fetch_klines(self.cfg, "BTCUSDT", "1h", start)
        self.assertEqual(len(rows), 2500)                       # the forming bar is dropped
        self.assertEqual([r[0] for r in rows], list(range(start, FORMING, H)))
        self.assertEqual(rows[0], (start, *[float(x) for x in kline(start, H)[1:6]]))
        starts = [p["startTime"] for p in fake.params("/api/v3/klines")]
        self.assertEqual(starts, [start * 1000, (start + 1000 * H) * 1000, (start + 2000 * H) * 1000])
        self.assertTrue(all(p["limit"] == 1000 and p["interval"] == "1h" for p in fake.params("/api/v3/klines")))

    def test_drops_bar_whose_close_time_is_not_past(self):
        rows = [kline(FORMING - H, H), kline(FORMING, H)]
        rows[1][6] = NOW * 1000                                  # closeTime == now -> not closed
        self.patch_http({"/api/v3/klines": lambda p: rows})
        self.assertEqual([r[0] for r in fetch_klines(self.cfg, "BTCUSDT", "1h", FORMING - H)], [FORMING - H])

    def test_end_t_is_exclusive(self):
        start = FORMING - 2000 * H
        fake = self.patch_http({"/api/v3/klines": FakeBinance(start).klines})
        rows = fetch_klines(self.cfg, "BTCUSDT", "1h", start, end_t=start + 1500 * H)
        self.assertEqual(len(rows), 1500)
        self.assertEqual(rows[-1][0], start + 1499 * H)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fetch_klines(self.cfg, "BTCUSDT", "1h", start, end_t=start), [])

    def test_okx_fallback_for_recent_range(self):
        start = FORMING - 150 * H
        fake = self.patch_http({"/api/v3/klines": failing,
                                "/api/v5/market/history-candles": okx_handler(FORMING - 400 * H, H)})
        with self.assertLogs("fibbot.data.candles", "WARNING"):
            rows = fetch_klines(self.cfg, "ETHUSDT", "1h", start)
        self.assertEqual([r[0] for r in rows], list(range(start, FORMING, H)))
        o = 1000.0 + (start // H) % 11
        self.assertEqual(rows[0], (start, o, o + 3, o - 2, o + 1, 7.5))
        okx = fake.params("/api/v5/market/history-candles")
        self.assertEqual(okx[0]["instId"], "ETH-USDT")
        self.assertEqual(okx[0]["bar"], "1H")
        self.assertEqual(okx[0]["after"], NOW * 1000)
        self.assertEqual(len(okx), 2)                            # pages of 100 rows until start
        self.assertEqual(okx[1]["after"], (FORMING - 99 * H) * 1000)

    def test_okx_daily_bar_name(self):
        start = parse_day("2024-01-05")
        fake = self.patch_http({"/api/v3/klines": failing,
                                "/api/v5/market/history-candles": okx_handler(LIST_T, D)})
        with self.assertLogs("fibbot.data.candles", "WARNING"):
            rows = fetch_klines(self.cfg, "BTCUSDT", "1d", start)
        self.assertEqual([r[0] for r in rows], list(range(start, parse_day("2024-01-10"), D)))
        self.assertEqual(fake.params("/api/v5/market/history-candles")[0]["bar"], "1Dutc")

    def test_no_okx_fallback_for_old_range(self):
        fake = self.patch_http({"/api/v3/klines": failing})
        with self.assertRaises(FetchError):
            fetch_klines(self.cfg, "BTCUSDT", "1h", FORMING - 5000 * H)
        self.assertEqual(fake.params("/api/v5/market/history-candles"), [])

    def test_both_sources_failing_raises(self):
        self.patch_http({"/api/v3/klines": failing, "/api/v5/market/history-candles": failing})
        with self.assertLogs("fibbot.data.candles", "WARNING"), self.assertRaises(FetchError) as cm:
            fetch_klines(self.cfg, "BTCUSDT", "1h", FORMING - 10 * H)
        self.assertIn("OKX", str(cm.exception))

    def test_unknown_timeframe(self):
        with self.assertRaises(ValueError):
            fetch_klines(self.cfg, "BTCUSDT", "2h", FORMING - 10 * H)


# ---------------------------------------------------------------------------------------------


class TestCandleStore(ClockMixin, unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = make_cfg(self.tmp.name, data={"history_start": "2023-06-01"})
        self.start_clock()
        self.fake = FakeHttp({"/api/v3/klines": FakeBinance(fail_symbols={"BADUSDT"}).klines,
                              "/api/v5/market/history-candles": failing})
        p = mock.patch("fibbot.data.candles.http_json", self.fake)
        p.start()
        self.addCleanup(p.stop)
        self.store = CandleStore(self.cfg)

    def test_get_without_cache_is_empty_and_offline(self):
        bars = self.store.get("BTCUSDT", "1h")
        self.assertEqual((len(bars), bars.symbol, bars.tf), (0, "BTCUSDT", "1h"))
        self.assertEqual(self.fake.calls, [])

    def test_update_from_history_start_and_round_trip(self):
        bars = self.store.update("BTCUSDT", "1h")
        self.assertEqual(self.fake.params("/api/v3/klines")[0]["startTime"], parse_day("2023-06-01") * 1000)
        self.assertEqual(bars.t, list(range(LIST_T, FORMING, H)))
        path = self.store.path("BTCUSDT", "1h")
        self.assertEqual(path.name, "BTCUSDT_1h.csv.gz")
        self.assertEqual(path.parent.name, "candles")
        with gzip.open(path, "rt") as f:
            self.assertEqual(f.readline().strip(), "t,o,h,l,c,v")
        self.assertEqual(os.listdir(path.parent), ["BTCUSDT_1h.csv.gz"])   # no temp files left
        again = CandleStore(self.cfg).get("BTCUSDT", "1h")
        self.assertEqual(again, bars)

    def test_incremental_update(self):
        first = self.store.update("BTCUSDT", "4h")
        self.now += 9 * H                                       # two more 4h bars close
        second = self.store.update("BTCUSDT", "4h")
        self.assertEqual(self.fake.params("/api/v3/klines")[1]["startTime"], (first.last_t + 4 * H) * 1000)
        self.assertEqual(len(second), len(first) + 2)
        self.assertEqual(second.t, list(range(LIST_T, forming_open(4 * H), 4 * H)))
        self.assertEqual(len(first), len(CandleStore(self.cfg).get("BTCUSDT", "4h")) - 2)

    def test_merge_overlapping_rows_without_duplicates(self):
        old = [(LIST_T + i * H, 1.0, 2.0, 0.5, 1.5, 10.0) for i in range(5)]
        new = [(LIST_T + i * H, 9.0, 9.0, 9.0, 9.0, 9.0) for i in range(3, 8)]
        with mock.patch("fibbot.data.candles.fetch_klines", side_effect=[old, new]):
            self.store.update("BTCUSDT", "1h")
            merged = self.store.update("BTCUSDT", "1h")
        self.assertEqual(merged.t, [LIST_T + i * H for i in range(8)])
        self.assertEqual(merged.c, [1.5, 1.5, 1.5, 9.0, 9.0, 9.0, 9.0, 9.0])   # newest rows win

    def test_update_without_new_bars_keeps_file(self):
        bars = self.store.update("BTCUSDT", "1d")
        st = self.store.path("BTCUSDT", "1d").stat()
        self.assertIs(self.store.update("BTCUSDT", "1d"), bars)
        self.assertEqual(self.store.path("BTCUSDT", "1d").stat().st_mtime_ns, st.st_mtime_ns)

    def test_memory_cache_and_invalidation(self):
        self.store.update("BTCUSDT", "1h")
        a = self.store.get("BTCUSDT", "1h")
        self.assertIs(self.store.get("BTCUSDT", "1h"), a)
        self.now += 2 * H
        CandleStore(self.cfg).update("BTCUSDT", "1h")           # another process appends bars
        b = self.store.get("BTCUSDT", "1h")
        self.assertEqual(len(b), len(a) + 2)

    def test_update_all_counts_and_isolates_errors(self):
        with self.assertLogs("fibbot.data.candles", "WARNING") as logs:
            res = self.store.update_all(["BTCUSDT", "BADUSDT", "ETHUSDT"], ["1h", "4h"])
        n1h, n4h = len(range(LIST_T, FORMING, H)), len(range(LIST_T, forming_open(4 * H), 4 * H))
        self.assertEqual(res, {("BTCUSDT", "1h"): n1h, ("BTCUSDT", "4h"): n4h,
                               ("ETHUSDT", "1h"): n1h, ("ETHUSDT", "4h"): n4h})
        self.assertEqual(set(self.store.last_errors), {("BADUSDT", "1h"), ("BADUSDT", "4h")})
        self.assertTrue(any("BADUSDT" in m for m in logs.output))
        self.now += H
        self.assertEqual(self.store.update_all(["BTCUSDT"], ["1h"]), {("BTCUSDT", "1h"): 1})
        self.assertEqual(self.store.last_errors, {})

    def test_load_many_skips_empty(self):
        self.store.update("BTCUSDT", "4h")
        got = self.store.load_many(["BTCUSDT", "ETHUSDT"], "4h")
        self.assertEqual(list(got), ["BTCUSDT"])

    def test_corrupt_file_reads_as_empty(self):
        path = self.store.path("BTCUSDT", "1h")
        path.write_bytes(b"definitely not gzip")
        with self.assertLogs("fibbot.data.candles", "ERROR"):
            self.assertEqual(len(self.store.get("BTCUSDT", "1h")), 0)

    def test_rejects_bad_symbol_or_tf(self):
        for sym, tf in (("../etc", "1h"), ("btcusdt", "1h"), ("BTCUSDT", "2h")):
            with self.assertRaises(ValueError):
                self.store.get(sym, tf)


# ---------------------------------------------------------------------------------------------


def hourly(start: int, n: int, skip=()) -> Bars:
    rows = []
    for i in range(n):
        t = start + i * H
        if t in skip:
            continue
        rows.append((t, 10.0 + i, 12.0 + i + (i % 5), 9.0 + i - (i % 3), 11.0 + i, 1.0 + i))
    return Bars.from_rows("BTCUSDT", "1h", rows)


def brute_force(bars: Bars, tf: str) -> dict[int, tuple]:
    dst, groups = TF_SECONDS[tf], {}
    for i, t in enumerate(bars.t):
        groups.setdefault(t - t % dst, []).append(i)
    return {b: (bars.o[ix[0]], max(bars.h[i] for i in ix), min(bars.l[i] for i in ix), bars.c[ix[-1]],
                sum(bars.v[i] for i in ix)) for b, ix in groups.items()}


class TestResample(unittest.TestCase):
    def test_explicit_bucket(self):
        day = parse_day("2024-03-01")
        rows = [(day, 1, 5, 0.5, 2, 1), (day + H, 2, 9, 0.4, 3, 2), (day + 2 * H, 3, 6, 0.9, 4, 3),
                (day + 3 * H, 4, 7, 0.8, 4.5, 4)]
        out = resample(Bars.from_rows("X", "1h", rows), "4h")
        self.assertEqual((out.symbol, out.tf, out.t, out.o, out.h, out.l, out.c, out.v),
                         ("X", "4h", [day], [1.0], [9.0], [0.4], [4.5], [10.0]))

    def test_drops_partial_edges_keeps_interior_gaps(self):
        day = parse_day("2024-03-01")
        gap = day + 5 * H
        bars = hourly(day + H, 26, skip={gap})                  # 01:00 day 1 .. 02:00 day 2
        out = resample(bars, "4h")
        self.assertEqual(out.t, [day + k * 4 * H for k in range(1, 6)])   # 04:00 .. 20:00 day 1
        ref = brute_force(bars, "4h")
        for k, t in enumerate(out.t):
            self.assertEqual((out.o[k], out.h[k], out.l[k], out.c[k], out.v[k]), ref[t])

    def test_daily_from_hourly_and_4h(self):
        day = parse_day("2024-03-01")
        bars = hourly(day, 24 * 3 + 7)
        out = resample(bars, "1d")
        self.assertEqual(out.t, [day, day + D, day + 2 * D])
        ref = brute_force(bars, "1d")
        self.assertEqual([(o, h, l, c, v) for o, h, l, c, v in zip(out.o, out.h, out.l, out.c, out.v)],
                         [ref[t] for t in out.t])
        via_4h = resample(resample(bars, "4h"), "1d")
        self.assertEqual((via_4h.t, via_4h.o, via_4h.h, via_4h.l, via_4h.c, via_4h.v),
                         (out.t, out.o, out.h, out.l, out.c, out.v))

    def test_complete_trailing_bucket_kept(self):
        day = parse_day("2024-03-01")
        self.assertEqual(resample(hourly(day, 8), "4h").t, [day, day + 4 * H])

    def test_identity_empty_and_errors(self):
        bars = hourly(parse_day("2024-03-01"), 10)
        self.assertEqual(resample(bars, "1h"), bars)
        self.assertEqual(len(resample(Bars("X", "1h"), "4h")), 0)
        with self.assertRaises(ValueError):
            resample(resample(bars, "4h"), "1h")
        with self.assertRaises(ValueError):
            resample(bars, "2h")


# ---------------------------------------------------------------------------------------------

U_NOW = parse_day("2025-06-01") + 12345
OLD, YOUNG = U_NOW - 1000 * D, U_NOW - 100 * D
MARKETS = [{"id": c, "symbol": c} for c in
           ("btc", "usdt", "eth", "wbtc", "steth", "xyz", "sol", "bnb", "eth", "doge", "trx", "ada")]
EXCHANGE = {"symbols": [{"symbol": s, "status": st} for s, st in (
    ("BTCUSDT", "TRADING"), ("ETHUSDT", "TRADING"), ("WBTCUSDT", "TRADING"), ("SOLUSDT", "TRADING"),
    ("BNBUSDT", "TRADING"), ("DOGEUSDT", "TRADING"), ("TRXUSDT", "TRADING"), ("ADAUSDT", "BREAK"),
    ("XYZBTC", "TRADING"))]}


class UniverseCase(ClockMixin, unittest.TestCase):
    now = U_NOW

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.start_clock()
        self.markets = MARKETS
        self.fake = FakeHttp({
            "/api/v3/coins/markets": lambda p: self.markets,
            "/api/v3/exchangeInfo": lambda p: EXCHANGE,
            "/api/v3/klines": lambda p: [kline(YOUNG if p["symbol"] == "SOLUSDT" else OLD, D)],
        })
        p = mock.patch("fibbot.data.candles.http_json", self.fake)
        p.start()
        self.addCleanup(p.stop)

    def cfg(self, **u):
        return make_cfg(self.tmp.name, universe=u)


class TestRankUniverse(UniverseCase):
    def test_rank_filters_and_preserves_order(self):
        self.assertEqual(rank_universe(self.cfg(top_n=3)), ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        gecko = self.fake.params("/api/v3/coins/markets")[0]
        self.assertEqual(gecko, {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 100, "page": 1})
        self.assertEqual(self.fake.params("/api/v3/exchangeInfo"), [{"permissions": "SPOT"}])
        checked = [p["symbol"] for p in self.fake.params("/api/v3/klines")]
        self.assertEqual(checked, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])   # stops at top_n, no dup
        self.assertTrue(all(p["interval"] == "1d" and p["startTime"] == 0 and p["limit"] == 1
                            for p in self.fake.params("/api/v3/klines")))

    def test_top_n_larger_than_available_and_extra_exclude(self):
        got = rank_universe(self.cfg(top_n=10, extra_exclude=["trx"]))
        self.assertEqual(got, ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT"])   # ADA not TRADING

    def test_min_history_days(self):
        got = rank_universe(self.cfg(top_n=10, min_history_days=50))
        self.assertEqual(got, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "TRXUSDT"])

    def test_exchange_symbols_cached_for_a_day(self):
        cfg = self.cfg()
        self.assertIn("BTCUSDT", binance_spot_symbols(cfg))
        self.assertNotIn("ADAUSDT", binance_spot_symbols(cfg))
        self.assertEqual(len(self.fake.params("/api/v3/exchangeInfo")), 1)
        with open(os.path.join(self.tmp.name, "exchange_symbols.json")) as f:
            cached = json.load(f)
        self.assertEqual(cached["as_of"], U_NOW)
        self.now += D + 1
        binance_spot_symbols(cfg)
        self.assertEqual(len(self.fake.params("/api/v3/exchangeInfo")), 2)

    def test_network_failure_raises_runtime_error(self):
        self.fake.routes["/api/v3/coins/markets"] = failing
        with self.assertRaises(RuntimeError):
            rank_universe(self.cfg())
        self.fake.routes["/api/v3/coins/markets"] = lambda p: {"status": {"error_code": 429}}
        with self.assertRaises(RuntimeError):
            rank_universe(self.cfg())


class TestGetUniverse(UniverseCase):
    def path(self):
        return os.path.join(self.tmp.name, "universe.json")

    def test_ranks_then_serves_cache(self):
        cfg = self.cfg(top_n=3)
        self.assertEqual(get_universe(cfg), ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        with open(self.path()) as f:
            saved = json.load(f)
        self.assertEqual((saved["as_of"], saved["symbols"]), (U_NOW, ["BTCUSDT", "ETHUSDT", "BNBUSDT"]))
        n = len(self.fake.calls)
        self.now += 6 * D
        self.assertEqual(get_universe(cfg), ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        self.assertEqual(len(self.fake.calls), n)                # fresh cache: no network

    def test_rerank_when_stale_refreshed_or_settings_change(self):
        cfg = self.cfg(top_n=3)
        get_universe(cfg)
        n = len(self.fake.params("/api/v3/coins/markets"))
        get_universe(cfg, refresh=True)
        self.now += 7 * D + 1
        get_universe(cfg)
        self.assertEqual(get_universe(self.cfg(top_n=2)), ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(len(self.fake.params("/api/v3/coins/markets")), n + 3)

    def test_failure_falls_back_to_cached_list(self):
        get_universe(self.cfg(top_n=10, min_history_days=50))
        self.fake.routes["/api/v3/coins/markets"] = failing
        with self.assertLogs("fibbot.data.universe", "WARNING"):
            got = get_universe(self.cfg(top_n=3, extra_exclude=["ETH"]), refresh=True)
        self.assertEqual(got, ["BTCUSDT", "SOLUSDT", "BNBUSDT"])

    def test_failure_without_cache_uses_static_fallback(self):
        self.fake.routes["/api/v3/exchangeInfo"] = failing
        with self.assertLogs("fibbot.data.universe", "WARNING"):
            got = get_universe(self.cfg(top_n=4, extra_exclude=["ETH"]))
        self.assertEqual(got, ["BTCUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT"])
        self.assertFalse(os.path.exists(self.path()))

    def test_empty_ranking_falls_back(self):
        self.markets = [{"symbol": "usdt"}, {"symbol": "xyz"}]
        with self.assertLogs("fibbot.data.universe", "WARNING"):
            got = get_universe(self.cfg(top_n=2))
        self.assertEqual(got, ["BTCUSDT", "ETHUSDT"])


if __name__ == "__main__":
    unittest.main()
