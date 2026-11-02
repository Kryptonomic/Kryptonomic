"""Core data types shared by every fibbot module.

Conventions (see fibbot/DESIGN.md):
- Time is integer Unix seconds, UTC. A bar's `t` is its OPEN time.
- Only CLOSED bars are ever stored or traded on.
- side: +1 = long, -1 = short.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field, asdict
from typing import Any

TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
BARS_PER_YEAR = {tf: 365 * 86400 / s for tf, s in TF_SECONDS.items()}


@dataclass
class Bars:
    """Columnar OHLCV series for one symbol and timeframe, sorted by time, no duplicates."""
    symbol: str
    tf: str
    t: list[int] = field(default_factory=list)
    o: list[float] = field(default_factory=list)
    h: list[float] = field(default_factory=list)
    l: list[float] = field(default_factory=list)
    c: list[float] = field(default_factory=list)
    v: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t)

    def slice(self, i0: int, i1: int) -> "Bars":
        """Bars with indices [i0, i1)."""
        return Bars(self.symbol, self.tf, self.t[i0:i1], self.o[i0:i1], self.h[i0:i1],
                    self.l[i0:i1], self.c[i0:i1], self.v[i0:i1])

    def index_at_or_after(self, ts: int) -> int:
        """First index whose t >= ts (len(self) if none)."""
        return bisect.bisect_left(self.t, ts)

    def window(self, start_t: int | None = None, end_t: int | None = None, warmup: int = 0) -> "Bars":
        """Bars with start_t <= t < end_t, plus `warmup` extra bars before start_t."""
        i0 = 0 if start_t is None else self.index_at_or_after(start_t)
        i1 = len(self) if end_t is None else self.index_at_or_after(end_t)
        return self.slice(max(0, i0 - warmup), i1)

    def append(self, t: int, o: float, h: float, l: float, c: float, v: float) -> None:
        if self.t and t <= self.t[-1]:
            raise ValueError(f"{self.symbol} {self.tf}: bar {t} not after last bar {self.t[-1]}")
        self.t.append(t); self.o.append(o); self.h.append(h)
        self.l.append(l); self.c.append(c); self.v.append(v)

    @property
    def last_t(self) -> int | None:
        return self.t[-1] if self.t else None

    @classmethod
    def from_rows(cls, symbol: str, tf: str, rows) -> "Bars":
        """rows: iterable of (t, o, h, l, c, v); sorted and de-duplicated (last wins)."""
        uniq = {}
        for r in rows:
            uniq[int(r[0])] = r
        b = cls(symbol, tf)
        for t in sorted(uniq):
            r = uniq[t]
            b.append(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
        return b


@dataclass
class Trade:
    """One round-trip trade produced by the strategy engine (single symbol, single position).

    ret_pct is the NET return on the entry notional, after fees and slippage, signed
    (+0.03 = +3% of notional). risk_pct = |entry_price - stop_price| / entry_price using the
    INITIAL stop. r_multiple = ret_pct / risk_pct. An open trade has exit_t=None,
    exit_reason="open", and exit_price/ret_pct marked at the last close.
    """
    strategy_id: str
    symbol: str
    tf: str
    side: int
    leg_id: str
    setup_t: int
    entry_t: int
    entry_price: float
    stop_price: float
    exit_t: int | None
    exit_price: float
    exit_reason: str
    ret_pct: float
    risk_pct: float
    r_multiple: float
    bars_held: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.exit_t is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Signal:
    """A setup that the engine ARMED: an entry order was placed (limit) or a confirmation watch began.

    Emitted once per (symbol, leg_id) - the first time the leg qualifies. Prices are as of
    arming time (limit orders may later be re-priced as the leg's running extreme moves).
    """
    strategy_id: str
    symbol: str
    tf: str
    side: int
    leg_id: str
    t: int                  # time of the bar whose close armed the setup
    entry_price: float
    stop_price: float
    tp1: float
    tp2: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Signal":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Costs:
    """Trading frictions. fee_rate per side on notional; slippage applied to market and stop fills."""
    fee_rate: float = 0.001
    slippage: float = 0.0005
