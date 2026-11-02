"""Closed-candle download (Binance spot klines, OKX fallback) and the local CSV.gz candle cache.

All network access goes through `http_json`, the single patch point for tests. Times are Unix
seconds, `t` is a bar's OPEN time, and only CLOSED bars are ever returned or stored.
"""
from __future__ import annotations

import contextlib
import csv
import gzip
import http.client
import io
import json
import logging
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from ..config import home_path
from ..types import TF_SECONDS, Bars

log = logging.getLogger(__name__)

USER_AGENT = "fibbot/0.1 (paper-trading research; +stdlib urllib)"
HTTP_TIMEOUT = 30.0
HTTP_TRIES = 5
BACKOFF_BASE = 1.0               # seconds before the 2nd try; doubles after every failure
BACKOFF_MAX = 60.0
NO_RETRY_STATUS = frozenset({400, 404})
BINANCE_LIMIT = 1000             # max klines per Binance request
OKX_LIMIT = 100                  # max candles per OKX history-candles request
OKX_BARS = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1Dutc"}
OKX_FALLBACK_MAX_BARS = 1000     # OKX is only tried for ranges starting at most this many bars ago
CSV_HEADER = ("t", "o", "h", "l", "c", "v")
_SYMBOL_RE = re.compile(r"[A-Z0-9]{2,30}")


class FetchError(RuntimeError):
    """A request failed (after all retries, or at once for HTTP 400/404). `status`: HTTP code or None."""

    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def http_json(url: str, params: dict | None = None, timeout: float = HTTP_TIMEOUT,
              tries: int = HTTP_TRIES):
    """GET `url` (+ query `params`) and decode the JSON body.

    Transient failures (connection errors, timeouts, 5xx, 418/429, bad JSON) are retried with
    exponential backoff that honours Retry-After; HTTP 400/404 fail immediately. Raises FetchError.
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip"})
    last, status = None, None
    for attempt in range(tries):
        delay = BACKOFF_BASE * 2 ** attempt
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                    body = gzip.decompress(body)
            return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code in NO_RETRY_STATUS:
                raise FetchError(f"HTTP {e.code} for {url}: {_error_body(e)}", e.code) from e
            last, status = e, e.code
            delay = max(delay, _retry_after(e))
        except (OSError, ValueError, EOFError, http.client.HTTPException) as e:
            last, status = e, None
        if attempt + 1 < tries:
            log.info("retrying %s in %.0fs after: %s", url, min(delay, BACKOFF_MAX), last)
            time.sleep(min(delay, BACKOFF_MAX))
    raise FetchError(f"{url} failed after {tries} tries: {last}", status) from last


def _error_body(e: urllib.error.HTTPError) -> str:
    try:
        return e.read(300).decode("utf-8", "replace")
    except (OSError, AttributeError, ValueError):
        return ""


def _retry_after(e: urllib.error.HTTPError) -> float:
    try:
        return float((e.headers or {}).get("Retry-After") or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_day(s: str) -> int:
    """'YYYY-MM-DD' -> Unix seconds at 00:00 UTC of that day."""
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to a temp file next to `path`, then os.replace it (readers never see partial files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _tf_seconds(tf: str) -> int:
    if tf not in TF_SECONDS:
        raise ValueError(f"unknown timeframe {tf!r}; expected one of {sorted(TF_SECONDS)}")
    return TF_SECONDS[tf]


# ---------------------------------------------------------------------------------------------
# Downloads


def fetch_klines(cfg: dict, symbol: str, tf: str, start_t: int, end_t: int | None = None) -> list[tuple]:
    """Closed bars with start_t <= t < end_t (default: up to now) as [(t, o, h, l, c, v), ...], t in seconds.

    Binance spot klines, paginated by startTime in pages of 1000. If Binance fails and the range
    starts at most OKX_FALLBACK_MAX_BARS bars ago, OKX history-candles is tried instead.
    """
    tf_s = _tf_seconds(tf)
    now_ms = int(time.time() * 1000)
    start_ms = int(start_t) * 1000
    end_ms = now_ms if end_t is None else min(int(end_t) * 1000, now_ms)
    if start_ms >= end_ms:
        return []
    try:
        return _binance_klines(cfg, symbol, tf, start_ms, end_ms, now_ms)
    except FetchError as e:
        if start_ms < now_ms - OKX_FALLBACK_MAX_BARS * tf_s * 1000:
            raise
        log.warning("Binance klines failed for %s %s (%s); trying OKX", symbol, tf, e)
        try:
            return _okx_candles(cfg, symbol, tf, start_ms, end_ms, now_ms)
        except FetchError as e2:
            raise FetchError(f"Binance: {e}; OKX: {e2}", e2.status) from e2


def _binance_klines(cfg, symbol, tf, start_ms, end_ms, now_ms) -> list[tuple]:
    """Rows: [openTime ms, open, high, low, close, volume, closeTime ms, ...]; forming bars dropped."""
    url = f"{cfg['data']['binance_base']}/api/v3/klines"
    step = TF_SECONDS[tf] * 1000
    out, cursor = [], start_ms
    while cursor < end_ms:
        page = http_json(url, {"symbol": symbol, "interval": tf, "startTime": cursor,
                               "endTime": end_ms - 1, "limit": BINANCE_LIMIT})
        if not isinstance(page, list):
            raise FetchError(f"unexpected Binance klines response for {symbol}: {str(page)[:200]}")
        for r in page:
            t_ms = int(r[0])
            if int(r[6]) < now_ms and cursor <= t_ms < end_ms:
                out.append((t_ms // 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        if len(page) < BINANCE_LIMIT:
            break
        cursor = max(cursor, int(page[-1][0])) + step
    return out


def _okx_candles(cfg, symbol, tf, start_ms, end_ms, now_ms) -> list[tuple]:
    """OKX history-candles, paged backwards with `after` (rows newest first; confirm "1" = closed)."""
    quote = cfg["universe"]["quote"].upper()
    if not symbol.endswith(quote) or symbol == quote:
        raise FetchError(f"cannot map {symbol} to an OKX {quote} instrument")
    params = {"instId": f"{symbol[:-len(quote)]}-{quote}", "bar": OKX_BARS[tf], "limit": OKX_LIMIT}
    url = f"{cfg['data']['okx_base']}/api/v5/market/history-candles"
    step = TF_SECONDS[tf] * 1000
    rows_by_t: dict[int, tuple] = {}
    after = end_ms
    while after > start_ms:
        res = http_json(url, {**params, "after": after})
        if not isinstance(res, dict) or str(res.get("code")) != "0" or not isinstance(res.get("data"), list):
            raise FetchError(f"unexpected OKX response for {symbol}: {str(res)[:200]}")
        page = res["data"]
        if not page:
            break
        for r in page:
            t_ms = int(r[0])
            if (str(r[8]) == "1" and t_ms + step <= now_ms and start_ms <= t_ms < end_ms
                    and t_ms % step == 0):
                rows_by_t[t_ms] = (t_ms // 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                                   float(r[5]))
        oldest = min(int(r[0]) for r in page)
        if oldest >= after:
            break
        after = oldest
    return [rows_by_t[t] for t in sorted(rows_by_t)]


# ---------------------------------------------------------------------------------------------
# Cache


def _encode(bars: Bars) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_HEADER)
    w.writerows(zip(bars.t, bars.o, bars.h, bars.l, bars.c, bars.v))
    return gzip.compress(buf.getvalue().encode("ascii"), compresslevel=6, mtime=0)


def _read_rows(path: Path) -> list[tuple]:
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        if tuple(next(reader, ())) != CSV_HEADER:
            raise ValueError(f"{path}: missing header {','.join(CSV_HEADER)}")
        return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                for r in reader if r]


class CandleStore:
    """Local cache of closed candles: home/candles/{SYMBOL}_{tf}.csv.gz with header t,o,h,l,c,v.

    `get` serves from an in-memory cache keyed by (symbol, tf) that is invalidated when the file on
    disk changes. Returned Bars are shared objects: treat them as read-only.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._mem: dict[tuple[str, str], tuple[tuple, Bars]] = {}
        self.last_errors: dict[tuple[str, str], str] = {}

    def path(self, symbol: str, tf: str) -> Path:
        """Cache file of (symbol, tf); validates both (symbols are upper-case alphanumerics)."""
        if not _SYMBOL_RE.fullmatch(symbol or ""):
            raise ValueError(f"invalid symbol {symbol!r}")
        _tf_seconds(tf)
        return home_path(self.cfg, "candles", f"{symbol}_{tf}.csv.gz")

    def get(self, symbol: str, tf: str) -> Bars:
        """Cached bars from disk (no network); empty Bars if there is no (readable) cache file."""
        path = self.path(symbol, tf)
        try:
            st = path.stat()
        except FileNotFoundError:
            self._mem.pop((symbol, tf), None)
            return Bars(symbol, tf)
        sig = (st.st_ino, st.st_mtime_ns, st.st_size)
        hit = self._mem.get((symbol, tf))
        if hit and hit[0] == sig:
            return hit[1]
        try:
            bars = Bars.from_rows(symbol, tf, _read_rows(path))
        except (OSError, EOFError, ValueError, IndexError) as e:
            log.error("unreadable candle cache %s (%s); treating as empty", path, e)
            return Bars(symbol, tf)
        self._mem[(symbol, tf)] = (sig, bars)
        return bars

    def update(self, symbol: str, tf: str) -> Bars:
        """Fetch bars after the last cached one (or from data.history_start), merge, save, return all."""
        old = self.get(symbol, tf)
        start = old.last_t + TF_SECONDS[tf] if len(old) else parse_day(self.cfg["data"]["history_start"])
        rows = fetch_klines(self.cfg, symbol, tf, start)
        if not rows:
            return old
        merged = Bars.from_rows(symbol, tf, chain(zip(old.t, old.o, old.h, old.l, old.c, old.v), rows))
        path = self.path(symbol, tf)
        atomic_write(path, _encode(merged))
        st = path.stat()
        self._mem[(symbol, tf)] = ((st.st_ino, st.st_mtime_ns, st.st_size), merged)
        return merged

    def update_all(self, symbols, tfs) -> dict[tuple[str, str], int]:
        """update() every (symbol, tf); returns {(sym, tf): n_new_bars} for the ones that succeeded.

        Failures are logged and recorded in `self.last_errors` {(sym, tf): message}, never raised.
        """
        out: dict[tuple[str, str], int] = {}
        self.last_errors = {}
        for sym in symbols:
            for tf in tfs:
                try:
                    before = len(self.get(sym, tf))
                    out[(sym, tf)] = len(self.update(sym, tf)) - before
                except Exception as e:  # one bad symbol must not stop the others
                    log.warning("candle update failed for %s %s: %s", sym, tf, e)
                    self.last_errors[(sym, tf)] = str(e)
        return out

    def load_many(self, symbols, tf: str) -> dict[str, Bars]:
        """{symbol: get(symbol, tf)} for symbols with at least one cached bar."""
        out = {}
        for sym in symbols:
            bars = self.get(sym, tf)
            if len(bars):
                out[sym] = bars
        return out


def resample(bars: Bars, tf: str) -> Bars:
    """Aggregate to the coarser `tf` with buckets aligned to UTC multiples of TF_SECONDS[tf].

    o = first, h = max, l = min, c = last, v = sum. A trailing bucket whose last source bar does
    not end at the bucket end is dropped (incomplete), as is a leading bucket the series starts
    inside of; interior buckets with missing source bars (exchange gaps) are kept.
    """
    src, dst = _tf_seconds(bars.tf), _tf_seconds(tf)
    if dst < src or dst % src:
        raise ValueError(f"cannot resample {bars.tf} -> {tf}")
    out = Bars(bars.symbol, tf)
    n, i = len(bars), 0
    while i < n:
        b0 = bars.t[i] - bars.t[i] % dst
        j = i
        while j < n and bars.t[j] < b0 + dst:
            j += 1
        complete_start = i > 0 or bars.t[i] == b0
        complete_end = j < n or bars.t[j - 1] + src == b0 + dst
        if complete_start and complete_end:
            out.append(b0, bars.o[i], max(bars.h[i:j]), min(bars.l[i:j]), bars.c[j - 1], sum(bars.v[i:j]))
        i = j
    return out
