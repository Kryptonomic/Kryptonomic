"""Pure technical indicators on plain float lists.

Every function returns a new list aligned index-for-index with its input; values that are not
yet defined (warmup) are float('nan'). Inputs are expected to be finite numbers (closed bars),
and every output value at index i depends only on inputs at indices <= i (causal).
"""
from __future__ import annotations

import math
from itertools import islice
from typing import Sequence

NAN = float("nan")


def _check_period(n: int) -> None:
    if n < 1:
        raise ValueError(f"period must be >= 1, got {n}")


def _check_hlc(h: Sequence[float], l: Sequence[float], c: Sequence[float]) -> int:
    if not len(h) == len(l) == len(c):
        raise ValueError(f"h/l/c lengths differ: {len(h)}/{len(l)}/{len(c)}")
    return len(h)


def _wilder(x: Sequence[float], n: int, start: int) -> list[float]:
    """Wilder smoothing of x[start:]: mean of the first n values (at index start+n-1),
    then (prev*(n-1) + x[i]) / n."""
    m = len(x)
    first = start + n - 1
    if m <= first:
        return [NAN] * m
    v = math.fsum(x[start:first + 1]) / n
    out = [NAN] * first + [v]
    append = out.append
    k = n - 1
    for xi in islice(x, first + 1, None):
        v = (v * k + xi) / n
        append(v)
    return out


def sma(x: Sequence[float], n: int) -> list[float]:
    """Simple moving average of the last n values; first value at index n-1."""
    _check_period(n)
    m = len(x)
    if m < n:
        return [NAN] * m
    s = math.fsum(x[:n])
    out = [NAN] * (n - 1) + [s / n]
    append = out.append
    for new, old in zip(islice(x, n, None), x):
        s += new - old
        append(s / n)
    return out


def ema(x: Sequence[float], n: int) -> list[float]:
    """EMA with alpha = 2/(n+1), seeded with the SMA of the first n values at index n-1."""
    _check_period(n)
    m = len(x)
    if m < n:
        return [NAN] * m
    a = 2.0 / (n + 1)
    b = 1.0 - a
    v = math.fsum(x[:n]) / n
    out = [NAN] * (n - 1) + [v]
    append = out.append
    for xi in islice(x, n, None):
        v = a * xi + b * v
        append(v)
    return out


def true_range(h: Sequence[float], l: Sequence[float], c: Sequence[float]) -> list[float]:
    """tr[0] = h0 - l0; tr[i] = max(h-l, |h - c[i-1]|, |l - c[i-1]|)."""
    if _check_hlc(h, l, c) == 0:
        return []
    out = [h[0] - l[0]]
    append = out.append
    for hi, lo, pc in zip(islice(h, 1, None), islice(l, 1, None), c):
        append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
    return out


def atr(h: Sequence[float], l: Sequence[float], c: Sequence[float], n: int) -> list[float]:
    """Wilder ATR: first value at index n-1 = mean(tr[0..n-1]); then (prev*(n-1) + tr)/n."""
    _check_period(n)
    return _wilder(true_range(h, l, c), n, 0)


def rsi(c: Sequence[float], n: int = 14) -> list[float]:
    """Wilder RSI; first value at index n. avg_loss == 0 -> 100 (both averages 0 -> 50).

    Uses the identity 100 - 100/(1 + g/l) == 100*g/(g + l)."""
    _check_period(n)
    m = len(c)
    diffs = [b - a for a, b in zip(c, islice(c, 1, None))]
    gains = [0.0] + [d if d > 0 else 0.0 for d in diffs]
    losses = [0.0] + [-d if d < 0 else 0.0 for d in diffs]
    out = [NAN] * min(n, m)
    out.extend(100.0 * g / (g + q) if g + q > 0 else 50.0
               for g, q in zip(islice(_wilder(gains, n, 1), n, None),
                               islice(_wilder(losses, n, 1), n, None)))
    return out


def adx(h: Sequence[float], l: Sequence[float], c: Sequence[float], n: int = 14) -> list[float]:
    """Wilder's ADX; first value at index 2n-1.

    +DM/-DM are Wilder-smoothed from bar 1 (first at index n), DX = 100*|+DI - -DI|/(+DI + -DI)
    from index n, and ADX = Wilder smoothing of DX. The smoothed true range divides both DIs, so
    it cancels in DX: DX = 100*|+DM_s - -DM_s|/(+DM_s + -DM_s) (0 when both are 0)."""
    _check_period(n)
    m = _check_hlc(h, l, c)
    pdm = [0.0] * m
    mdm = [0.0] * m
    for i, (h0, h1, l0, l1) in enumerate(zip(h, islice(h, 1, None), l, islice(l, 1, None)), 1):
        up = h1 - h0
        dn = l0 - l1
        if up > dn and up > 0:
            pdm[i] = up
        elif dn > up and dn > 0:
            mdm[i] = dn
    dx = [NAN] * min(n, m)
    dx.extend(100.0 * abs(p - q) / (p + q) if p + q > 0 else 0.0
              for p, q in zip(islice(_wilder(pdm, n, 1), n, None),
                              islice(_wilder(mdm, n, 1), n, None)))
    return _wilder(dx, n, n)
