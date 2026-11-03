"""Fibonacci price levels and a causal ZigZag swing-pivot detector (pure)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import repeat
from typing import Sequence

from .types import Bars

RETRACE_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.65, 0.705, 0.786, 0.886)
EXTENSION_LEVELS = (1.272, 1.414, 1.618, 2.0, 2.618)


@dataclass(frozen=True)
class Pivot:
    """A confirmed swing extreme."""
    idx: int          # bar index of the extreme
    t: int            # bar open time of the extreme
    price: float      # the high (kind 'H') or low (kind 'L')
    kind: str         # 'H' or 'L'
    confirm_idx: int  # bar index at whose CLOSE the pivot became known (> idx)


class PivotTracker:
    """Causal ZigZag. mode 'atr': reversal threshold = mult * atr_i ; mode 'pct': pct * extreme price.

    Feed bars in order with update(); pivots strictly alternate H/L. A threshold that is nan or
    <= 0 (e.g. ATR warmup) is invalid and confirms nothing. Public state, read-only for callers:
    direction (0 before the first pivot, +1 tracking a running high, -1 a running low),
    hi/hi_idx/hi_t and lo/lo_idx/lo_t (the running extreme for the current direction; both while
    direction == 0), and last (the most recent confirmed Pivot or None).
    """

    __slots__ = ("mult", "mode", "pct", "_atr_mode", "direction",
                 "hi", "hi_idx", "hi_t", "lo", "lo_idx", "lo_t", "last")

    def __init__(self, mult: float = 3.0, mode: str = "atr", pct: float = 0.05):
        if mode not in ("atr", "pct"):
            raise ValueError(f"mode must be 'atr' or 'pct', got {mode!r}")
        if mode == "atr" and not mult > 0:
            raise ValueError(f"mult must be > 0, got {mult}")
        if mode == "pct" and not pct > 0:
            raise ValueError(f"pct must be > 0, got {pct}")
        self.mult, self.mode, self.pct = mult, mode, pct
        self._atr_mode = mode == "atr"
        self.direction = 0
        self.hi, self.hi_idx, self.hi_t = -math.inf, -1, 0
        self.lo, self.lo_idx, self.lo_t = math.inf, -1, 0
        self.last: Pivot | None = None

    def update(self, i: int, t: int, high: float, low: float, atr_i: float) -> Pivot | None:
        """Process bar i at its close; return the pivot confirmed at this bar, if any."""
        d = self.direction
        if d == 1:
            if high > self.hi:
                self.hi, self.hi_idx, self.hi_t = high, i, t
                return None
            thr = self.mult * atr_i if self._atr_mode else self.pct * self.hi
            if thr > 0 and self.hi - low >= thr:
                return self._confirm_high(i, t, low)
            return None
        if d == -1:
            if low < self.lo:
                self.lo, self.lo_idx, self.lo_t = low, i, t
                return None
            thr = self.mult * atr_i if self._atr_mode else self.pct * self.lo
            if thr > 0 and high - self.lo >= thr:
                return self._confirm_low(i, t, high)
            return None
        if high > self.hi:
            self.hi, self.hi_idx, self.hi_t = high, i, t
        if low < self.lo:
            self.lo, self.lo_idx, self.lo_t = low, i, t
        if self._atr_mode:
            thr_h = thr_l = self.mult * atr_i
        else:
            thr_h, thr_l = self.pct * self.hi, self.pct * self.lo
        h_ok = thr_h > 0 and self.hi_idx < i and self.hi - low >= thr_h
        l_ok = thr_l > 0 and self.lo_idx < i and high - self.lo >= thr_l
        if h_ok and not (l_ok and self.lo_idx < self.hi_idx):   # both: the older extreme wins
            return self._confirm_high(i, t, low)
        if l_ok:
            return self._confirm_low(i, t, high)
        return None

    def _confirm_high(self, i: int, t: int, low: float) -> Pivot:
        p = Pivot(self.hi_idx, self.hi_t, self.hi, "H", i)
        self.direction = -1
        self.lo, self.lo_idx, self.lo_t = low, i, t
        self.last = p
        return p

    def _confirm_low(self, i: int, t: int, high: float) -> Pivot:
        p = Pivot(self.lo_idx, self.lo_t, self.lo, "L", i)
        self.direction = 1
        self.hi, self.hi_idx, self.hi_t = high, i, t
        self.last = p
        return p


def zigzag(bars: Bars, atr_values: Sequence[float] | None, mult: float = 3.0, mode: str = "atr",
           pct: float = 0.05) -> list[Pivot]:
    """All pivots from feeding every bar to one PivotTracker (atr_values may be None in pct mode)."""
    tracker = PivotTracker(mult, mode, pct)
    if atr_values is None:
        if mode == "atr":
            raise ValueError("atr mode needs atr_values")
        atr_values = repeat(math.nan)
    elif len(atr_values) != len(bars):
        raise ValueError(f"atr_values length {len(atr_values)} != bars length {len(bars)}")
    update = tracker.update
    out = []
    for i, (t, hi, lo, a) in enumerate(zip(bars.t, bars.h, bars.l, atr_values)):
        p = update(i, t, hi, lo, a)
        if p is not None:
            out.append(p)
    return out


def retrace_price(anchor: float, extreme: float, r: float) -> float:
    """Price retracing ratio r of the leg anchor -> extreme: r=0 -> extreme, r=1 -> anchor.

    Up-leg: anchor = swing low, extreme = running high; down-leg: anchor = swing high,
    extreme = running low."""
    return extreme - r * (extreme - anchor)


def extension_price(anchor: float, extreme: float, e: float) -> float:
    """Extension e of the leg measured from the anchor: e=1 -> extreme, e>1 beyond it.

    extension_price(a, x, e) == retrace_price(a, x, 1 - e)."""
    return anchor + e * (extreme - anchor)


def fib_table(anchor: float, extreme: float) -> dict[str, float]:
    """{"0.236": price, ..., "0.886": price, "1.272": price, ..., "2.618": price} for display."""
    table = {str(r): retrace_price(anchor, extreme, r) for r in RETRACE_LEVELS}
    table.update({str(e): extension_price(anchor, extreme, e) for e in EXTENSION_LEVELS})
    return table
