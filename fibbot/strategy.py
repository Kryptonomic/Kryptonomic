"""Strategy spec, parameter space, presets and the bar-by-bar Fibonacci retracement engine (pure).

StrategySpec describes one family of Fibonacci-retracement setups. FibEngine replays a spec over the
closed bars of one symbol and returns its trades and signals; it is deterministic and causal (what
happens at bar i depends only on bars <= i). manage_position exposes the same per-bar exit logic
(the _Position class) for the random baseline. The binding rules are in fibbot/DESIGN.md.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace

from . import indicators as ind
from .fib import Pivot, PivotTracker, extension_price, retrace_price
from .types import Bars, Costs, Signal, Trade, TF_SECONDS


@dataclass
class StrategySpec:
    """One Fibonacci-retracement strategy. Ratios are retracements of the leg anchor -> extreme
    (0 = extreme, 1 = anchor); *_ext are extensions from the anchor (1 = extreme).

    On construction ints given for float fields become floats, floats are rounded to 10 decimals
    and integral floats given for int fields become ints, so equal specs share one spec_id."""
    name: str = "custom"
    timeframe: str = "4h"
    direction: str = "long"
    swing_mode: str = "atr"
    swing_atr_len: int = 14
    swing_mult: float = 3.0
    swing_pct: float = 0.05
    min_leg_atr: float = 4.0
    setup_expiry_bars: int = 40
    trend_filter: str = "ema"
    trend_ema: int = 200
    adx_min: float = 0.0
    require_structure: bool = False
    entry_zone_lo: float = 0.618
    entry_zone_hi: float = 0.65
    entry_mode: str = "limit"
    entry_at: str = "mid"
    confirm_type: str = "reclaim"
    confirm_window: int = 6
    stop_ratio: float = 0.886
    stop_atr_buffer: float = 0.25
    tp1_ext: float = 1.0
    tp1_frac: float = 0.5
    tp2_ext: float = 1.618
    move_stop_to_be: bool = True
    trail_atr: float = 0.0
    max_hold_bars: int = 120
    min_rr: float = 1.5

    def __post_init__(self):
        for k, p in PARAM_SPACE.items():
            v = getattr(self, k)
            if isinstance(v, bool):
                continue
            if p["type"] == "float" and isinstance(v, (int, float)):
                setattr(self, k, round(float(v), 10))
            elif p["type"] == "int" and isinstance(v, float) and v.is_integer():
                setattr(self, k, int(v))

    def validate(self) -> "StrategySpec":
        """Check types, ranges (PARAM_SPACE) and cross-field rules; raise ValueError("field: reason")."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name: must be a non-empty string, got {self.name!r}")
        for k, p in PARAM_SPACE.items():
            v, kind = getattr(self, k), p["type"]
            if kind == "choice":
                if not isinstance(v, str) or v not in p["choices"]:
                    raise ValueError(f"{k}: must be one of {p['choices']}, got {v!r}")
            elif kind == "bool":
                if not isinstance(v, bool):
                    raise ValueError(f"{k}: must be true or false, got {v!r}")
            else:
                ok = isinstance(v, int) if kind == "int" else isinstance(v, (int, float))
                if isinstance(v, bool) or not ok or not p["min"] <= v <= p["max"]:
                    what = "an integer" if kind == "int" else "a number"
                    raise ValueError(f"{k}: must be {what} in [{p['min']}, {p['max']}], got {v!r}")
        if self.entry_zone_hi < self.entry_zone_lo:
            raise ValueError(f"entry_zone_hi: must be >= entry_zone_lo ({self.entry_zone_lo}), "
                             f"got {self.entry_zone_hi}")
        if self.stop_ratio <= self.entry_zone_hi:
            raise ValueError(f"stop_ratio: must be > entry_zone_hi ({self.entry_zone_hi}), "
                             f"got {self.stop_ratio}")
        if self.tp1_frac < 1.0 and self.tp2_ext <= self.tp1_ext:
            raise ValueError(f"tp2_ext: must be > tp1_ext ({self.tp1_ext}) unless tp1_frac == 1, "
                             f"got {self.tp2_ext}")
        return self

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "StrategySpec":
        """Build from a dict (missing keys take defaults). Unknown keys raise ValueError; the
        result is not validated - call validate()."""
        if not isinstance(d, dict):
            raise ValueError(f"spec must be a dict, got {type(d).__name__}")
        _check_keys(d)
        return cls(**d)

    def spec_id(self) -> str:
        """10 hex chars: sha1 of the canonical JSON of to_dict() without "name"."""
        d = self.to_dict()
        del d["name"]
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def with_overrides(self, **kw) -> "StrategySpec":
        """Copy with some fields replaced (unknown names raise ValueError; not validated)."""
        _check_keys(kw)
        return replace(self, **kw)


_FIELD_NAMES = frozenset(f.name for f in fields(StrategySpec))


def _check_keys(d: dict) -> None:
    unknown = sorted(str(k) for k in set(d) - _FIELD_NAMES)
    if unknown:
        raise ValueError(f"unknown spec keys: {', '.join(unknown)}")


def _num(kind, lo, hi, step, doc):
    return {"type": kind, "min": lo, "max": hi, "step": step, "doc": doc}


def _choice(choices, doc):
    return {"type": "choice", "choices": list(choices), "doc": doc}


PARAM_SPACE: dict[str, dict] = {
    "timeframe": _choice(("1h", "4h", "1d"), "bar timeframe"),
    "direction": _choice(("long", "short", "both"), "which legs to trade"),
    "swing_mode": _choice(("atr", "pct"), "pivot reversal threshold: swing_mult*ATR or swing_pct*price"),
    "swing_atr_len": _num("int", 5, 50, 2, "ATR period (pivots, leg size, stop buffer, trailing)"),
    "swing_mult": _num("float", 1.0, 10.0, 0.5, "ATR multiple that confirms a swing pivot (atr mode)"),
    "swing_pct": _num("float", 0.01, 0.30, 0.01, "price fraction that confirms a swing pivot (pct mode)"),
    "min_leg_atr": _num("float", 0.0, 30.0, 1.0, "minimum leg size (anchor to extreme) in ATRs"),
    "setup_expiry_bars": _num("int", 3, 500, 10, "an unfilled leg dies when its extreme is older than this"),
    "trend_filter": _choice(("none", "ema", "ema_slope", "ema_stack"),
                            "trend condition, evaluated at the leg's extreme bar"),
    "trend_ema": _num("int", 20, 400, 25, "trend EMA period"),
    "adx_min": _num("float", 0.0, 50.0, 5.0, "minimum ADX(14) at the extreme bar (0 disables)"),
    "require_structure": {"type": "bool", "doc": "long: anchor low above the previous swing low "
                                                 "(short: anchor high below the previous high)"},
    "entry_zone_lo": _num("float", 0.236, 0.886, 0.05, "shallow edge of the entry zone (retracement)"),
    "entry_zone_hi": _num("float", 0.236, 0.886, 0.05, "deep edge of the entry zone (>= entry_zone_lo)"),
    "entry_mode": _choice(("limit", "confirm"), "resting limit in the zone, or market entry after a "
                                                "confirmation candle"),
    "entry_at": _choice(("lo", "mid", "hi"), "limit price within the zone (limit mode)"),
    "confirm_type": _choice(("reclaim", "engulf", "rsi"), "confirmation pattern (confirm mode)"),
    "confirm_window": _num("int", 1, 50, 2, "bars after a zone touch to wait for a confirmation"),
    "stop_ratio": _num("float", 0.25, 1.2, 0.05, "stop retracement ratio (> entry_zone_hi; 1.0 = anchor)"),
    "stop_atr_buffer": _num("float", 0.0, 3.0, 0.1, "extra ATRs beyond the stop level"),
    "tp1_ext": _num("float", 0.3, 3.0, 0.1, "target 1 as a leg extension (1.0 = the extreme)"),
    "tp1_frac": _num("float", 0.05, 1.0, 0.1, "fraction of the position closed at target 1"),
    "tp2_ext": _num("float", 0.3, 4.236, 0.25,
                    "final target extension (> tp1_ext; unused if tp1_frac == 1)"),
    "move_stop_to_be": {"type": "bool", "doc": "after target 1 the stop moves to breakeven incl. fees "
                                               "(active from the next bar)"},
    "trail_atr": _num("float", 0.0, 10.0, 0.5,
                      "after target 1 trail the stop at close -/+ k*ATR (0 disables)"),
    "max_hold_bars": _num("int", 1, 2000, 20, "time stop in bars"),
    "min_rr": _num("float", 0.0, 10.0, 0.25, "minimum planned (blended) reward/risk"),
}
for _k, _p in PARAM_SPACE.items():
    _p["default"] = StrategySpec.__dataclass_fields__[_k].default

PRESETS: dict[str, StrategySpec] = {s.name: s.validate() for s in (
    StrategySpec(name="golden_pocket_4h"),
    StrategySpec(name="golden_pocket_1h", timeframe="1h"),
    StrategySpec(name="golden_pocket_both_4h", direction="both"),
    StrategySpec(name="ote_confirm_1h", timeframe="1h", entry_zone_lo=0.618, entry_zone_hi=0.786,
                 entry_mode="confirm", confirm_type="reclaim", stop_ratio=1.0, stop_atr_buffer=0.1,
                 tp1_ext=1.0, tp2_ext=1.618),
    StrategySpec(name="shallow_trend_4h", trend_filter="ema_stack", adx_min=20.0,
                 entry_zone_lo=0.382, entry_zone_hi=0.5, entry_at="lo", stop_ratio=0.786,
                 stop_atr_buffer=0.0, tp1_ext=1.0, tp2_ext=1.618),
    StrategySpec(name="deep_786_4h", trend_filter="none", entry_zone_lo=0.786, entry_zone_hi=0.886,
                 entry_at="lo", stop_ratio=1.0, stop_atr_buffer=0.3, tp1_ext=0.618, tp2_ext=1.0),
    StrategySpec(name="golden_pocket_1d", timeframe="1d", entry_zone_lo=0.5, entry_zone_hi=0.65,
                 stop_ratio=0.886, stop_atr_buffer=0.0, tp1_ext=1.0, tp2_ext=1.618),
    StrategySpec(name="ext_runner_4h", tp1_ext=1.0, tp1_frac=0.3, trail_atr=2.5, tp2_ext=2.618),
)}


def warmup_bars(spec: StrategySpec) -> int:
    """Bars of history to prepend before a trading window so indicators and pivots are settled."""
    return max(3 * spec.trend_ema, 5 * spec.swing_atr_len, 4 * 14, 300)


@dataclass
class RunResult:
    """Output of FibEngine.run for one symbol."""
    trades: list[Trade]          # CLOSED trades, entry_t >= trade_start_t, in entry order
    open_trade: Trade | None     # position still open at the last bar (marked at the last close)
    signals: list[Signal]        # legs armed at bars with t >= trade_start_t - TF_SECONDS[tf]
    n_bars: int


class _Position:
    """One open position and the per-bar exit rules. FibEngine and manage_position both drive it
    through intrabar() (fills inside a bar) and at_close() (close-of-bar rules)."""

    __slots__ = ("side", "entry_idx", "entry_px", "init_stop", "stop", "stop_kind", "tp1", "tp2",
                 "tp1_frac", "be_px", "trail_atr", "max_hold", "slip", "fee", "tp1_hit", "rem",
                 "fills", "exit_idx", "exit_reason")

    def __init__(self, side: int, entry_idx: int, entry_px: float, stop_px: float, tp1: float,
                 tp2: float, spec: StrategySpec, costs: Costs):
        self.side, self.entry_idx, self.entry_px = side, entry_idx, entry_px
        self.init_stop = self.stop = stop_px
        self.stop_kind = "stop"               # "stop" -> "breakeven" -> "trail" as the stop moves
        self.tp1, self.tp2, self.tp1_frac = tp1, tp2, spec.tp1_frac
        self.fee, self.slip = costs.fee_rate, costs.slippage
        self.be_px = entry_px * (1 + side * 2 * self.fee) if spec.move_stop_to_be else None
        self.trail_atr, self.max_hold = spec.trail_atr, spec.max_hold_bars
        self.tp1_hit = False
        self.rem = 1.0
        self.fills: list[list] = []           # [fraction, price, label, t]
        self.exit_idx: int | None = None
        self.exit_reason: str | None = None

    def _exit(self, i: int, t: int, price: float, reason: str, label: str | None = None) -> None:
        self.fills.append([self.rem, price, label or reason, t])
        self.rem = 0.0
        self.exit_idx, self.exit_reason = i, reason

    def intrabar(self, i: int, t: int, o: float, h: float, l: float) -> None:
        """Fills inside bar i. The entry bar only checks the stop (worst case, targets never
        credited); later bars: gap through stop, stop, then tp1 / tp2 (stop before targets)."""
        s, stop = self.side, self.stop
        adverse = l if s > 0 else h
        if i == self.entry_idx:
            # the worse of the open and the fill is the pre-slippage price of the fill
            x = min(o, self.entry_px) if s > 0 else max(o, self.entry_px)
            if s * (x - stop) <= 0:
                self._exit(i, t, x * (1 - s * self.slip), "stop")
            elif s * (adverse - stop) <= 0:
                self._exit(i, t, stop * (1 - s * self.slip), "stop")
            return
        if s * (o - stop) <= 0:
            self._exit(i, t, o * (1 - s * self.slip), self.stop_kind)
            return
        if s * (adverse - stop) <= 0:
            self._exit(i, t, stop * (1 - s * self.slip), self.stop_kind)
            return
        fav = h if s > 0 else l
        if not self.tp1_hit:
            if s * (fav - self.tp1) < 0:
                return
            px = o if s * (o - self.tp1) > 0 else self.tp1
            if self.tp1_frac >= 1.0:
                self._exit(i, t, px, "target", "tp1")
                return
            self.fills.append([self.tp1_frac, px, "tp1", t])
            self.rem = 1.0 - self.tp1_frac
            self.tp1_hit = True
            # this bar's stop check already ran, so the breakeven stop acts from the next bar
            if self.be_px is not None and s * (self.be_px - self.stop) > 0:
                self.stop, self.stop_kind = self.be_px, "breakeven"
        if s * (fav - self.tp2) >= 0:
            self._exit(i, t, o if s * (o - self.tp2) > 0 else self.tp2, "target", "tp2")

    def at_close(self, i: int, t: int, c: float, atr_i: float) -> None:
        """Close of bar i: trail the stop after tp1, then the time stop."""
        s = self.side
        if self.tp1_hit and self.trail_atr > 0 and not math.isnan(atr_i):
            level = c - s * self.trail_atr * atr_i
            if s * (level - self.stop) > 0:
                self.stop, self.stop_kind = level, "trail"
        if i - self.entry_idx >= self.max_hold:
            self._exit(i, t, c * (1 - s * self.slip), "time")

    def result(self, t: list[int], c: list[float]) -> dict:
        """Accounting; a position still open is marked at the last close with exit costs."""
        s, fills = self.side, [list(f) for f in self.fills]
        exit_idx, reason = self.exit_idx, self.exit_reason
        if exit_idx is None:
            exit_idx, reason = len(c) - 1, "open"
            fills.append([self.rem, c[-1] * (1 - s * self.slip), "open", t[-1]])
        gross = sum(f * p for f, p, _, _ in fills)
        p0, fee = self.entry_px, self.fee
        ret = gross / p0 * (1 - fee) - (1 + fee) if s > 0 else (1 - fee) - gross / p0 * (1 + fee)
        risk = abs(p0 - self.init_stop) / p0
        return {"exit_idx": exit_idx, "exit_reason": reason, "fills": fills, "exit_price": gross,
                "ret_pct": ret, "risk_pct": risk, "r_multiple": ret / risk if risk > 0 else 0.0,
                "open": self.exit_idx is None}


def manage_position(bars: Bars, atr: list[float], entry_idx: int, side: int, entry_px: float,
                    stop_px: float, tp1: float, tp2: float, spec: StrategySpec, costs: Costs,
                    start_mgmt_idx: int | None = None) -> dict:
    """Manage a position filled at entry_px (incl. slippage) during bar entry_idx with the exact
    exit rules FibEngine uses, from bar start_mgmt_idx (default entry_idx) to the exit or the last
    bar. The entry bar gets the entry-bar stop check; a later start_mgmt_idx skips the bars before
    it. Returns {"exit_idx", "exit_reason", "fills", "exit_price", "ret_pct", "risk_pct",
    "r_multiple", "open"}; an open position is marked at the last close."""
    if side not in (1, -1):
        raise ValueError(f"side must be +1 or -1, got {side!r}")
    if not 0 <= entry_idx < len(bars):
        raise ValueError(f"entry_idx {entry_idx} outside bars [0, {len(bars)})")
    first = entry_idx if start_mgmt_idx is None else start_mgmt_idx
    if first < entry_idx:
        raise ValueError(f"start_mgmt_idx {first} < entry_idx {entry_idx}")
    pos = _Position(side, entry_idx, entry_px, stop_px, tp1, tp2, spec, costs)
    t, o, h, l, c = bars.t, bars.o, bars.h, bars.l, bars.c
    for i in range(first, len(bars)):
        pos.intrabar(i, t[i], o[i], h[i], l[i])
        if pos.exit_idx is None:
            pos.at_close(i, t[i], c[i], atr[i])
        if pos.exit_idx is not None:
            break
    return pos.result(t, c)


def _rr(s: int, entry: float, stop: float, tp1: float, tp2: float, frac: float) -> float:
    """Planned blended reward/risk of an entry; -inf unless stop < entry < tp1 (long sense)."""
    risk, r1 = s * (entry - stop), s * (tp1 - entry)
    if risk <= 0 or r1 <= 0:
        return -math.inf
    return (frac * r1 + (1.0 - frac) * s * (tp2 - entry)) / risk


class _Order:
    """Entry order placed at a bar close for the next bar only (limit at px, or market)."""
    __slots__ = ("side", "market", "px", "stop", "tp1", "tp2", "leg_id", "setup_t", "meta")

    def __init__(self, side, market, px, stop, tp1, tp2, meta):
        self.side, self.market, self.px = side, market, px
        self.stop, self.tp1, self.tp2, self.meta = stop, tp1, tp2, meta
        self.leg_id, self.setup_t = side.leg_id, side.armed_t


class _Side:
    """The current leg of one direction: anchor pivot, running extreme, liveness, confirm touch."""

    __slots__ = ("s", "kind", "tag", "spec", "entry_ratio", "t", "o", "c", "fav", "adv", "atr",
                 "filt", "rsi", "anchor", "leg_id", "ext", "ext_idx", "worst", "struct_ok",
                 "active", "armed_t", "touch")

    def __init__(self, s: int, spec: StrategySpec, entry_ratio: float, bars: Bars,
                 atr: list[float], filt: list[bool], rsi: list[float] | None):
        self.s, self.spec, self.entry_ratio = s, spec, entry_ratio
        self.kind, self.tag = ("L", "L") if s > 0 else ("H", "S")
        self.t, self.o, self.c = bars.t, bars.o, bars.c
        self.fav, self.adv = (bars.h, bars.l) if s > 0 else (bars.l, bars.h)
        self.atr, self.filt, self.rsi = atr, filt, rsi
        self.anchor: Pivot | None = None
        self.active = False

    def set_anchor(self, piv: Pivot, i: int, skip_legs) -> None:
        """New anchor pivot confirmed at bar i: scan (piv.idx, i] once for the extreme and the
        adverse extreme; afterwards advance() keeps them up to date bar by bar."""
        s, fav, adv = self.s, self.fav, self.adv
        prev = self.anchor
        self.anchor = piv
        self.leg_id = f"{self.tag}{piv.t}"
        j = piv.idx + 1
        ext, ext_idx, worst = fav[j], j, adv[j]
        for j in range(j + 1, i + 1):
            if s * (fav[j] - ext) > 0:
                ext, ext_idx = fav[j], j
            if s * (adv[j] - worst) < 0:
                worst = adv[j]
        self.ext, self.ext_idx, self.worst = ext, ext_idx, worst
        self.struct_ok = (not self.spec.require_structure
                          or (prev is not None and s * (piv.price - prev.price) > 0))
        self.active = self.leg_id not in skip_legs
        self.armed_t = self.touch = None

    def advance(self, i: int) -> None:
        """Extend the running extreme (first occurrence) and the adverse extreme with bar i.
        A new extreme moves the zone, so it clears a confirm-mode touch."""
        s = self.s
        x = self.fav[i]
        if s * (x - self.ext) > 0:
            self.ext, self.ext_idx, self.touch = x, i, None
        y = self.adv[i]
        if s * (y - self.worst) < 0:
            self.worst = y

    def on_close(self, i: int):
        """Evaluate the live leg at the close of bar i. Returns (signal, order): signal is
        (entry, stop, tp1, tp2, meta) the first time the leg qualifies, order a candidate _Order."""
        s, sp, a, ext = self.s, self.spec, self.anchor.price, self.ext
        if s * (self.worst - a) < 0 or i - self.ext_idx > sp.setup_expiry_bars:
            self.active = False
            return None, None
        atr_i = self.atr[i]
        stop = retrace_price(a, ext, sp.stop_ratio) - s * sp.stop_atr_buffer * atr_i
        if s * (self.adv[i] - stop) <= 0:            # price ran through the stop while unfilled
            self.active = False
            return None, None
        confirm = sp.entry_mode == "confirm"
        if confirm and self.touch is not None and i - self.touch > sp.confirm_window:
            self.touch = None
        leg = s * (ext - a)
        if not (leg >= sp.min_leg_atr * atr_i and self.filt[self.ext_idx] and self.struct_ok):
            return None, None
        tp1 = extension_price(a, ext, sp.tp1_ext)
        tp2 = extension_price(a, ext, sp.tp2_ext)
        entry = retrace_price(a, ext, self.entry_ratio)
        rr = _rr(s, entry, stop, tp1, tp2, sp.tp1_frac)
        ok = rr >= sp.min_rr
        zone_top = retrace_price(a, ext, sp.entry_zone_lo)
        signal = order = None
        if ok and self.armed_t is None:
            self.armed_t = self.t[i]
            signal = (entry, stop, tp1, tp2, {
                "anchor": a, "anchor_t": self.anchor.t, "extreme": ext, "ext_t": self.t[self.ext_idx],
                "leg_size": leg, "atr": atr_i, "rr": rr, "mode": sp.entry_mode,
                "zone": [zone_top, retrace_price(a, ext, sp.entry_zone_hi)]})
        c_i = self.c[i]
        if not confirm:
            if ok and s * (c_i - entry) > 0:
                order = _Order(self, False, entry, stop, tp1, tp2,
                               self._meta(ext, leg, atr_i, self.entry_ratio, tp1, tp2, rr))
            return signal, order
        # A touch must come after the extreme: on the extreme bar the low may precede the high.
        # A live leg's adverse extreme is above its stop (checked above), so a zone touch only
        # needs the bar to reach zone_top.
        if self.touch is None and ok and i > self.ext_idx and s * (zone_top - self.adv[i]) >= 0:
            self.touch = i
        if self.touch is not None and self._confirmed(i, zone_top):
            rr_c = _rr(s, c_i, stop, tp1, tp2, sp.tp1_frac)
            if rr_c >= sp.min_rr:
                ratio = (ext - c_i) / (ext - a)
                order = _Order(self, True, c_i, stop, tp1, tp2,
                               self._meta(ext, leg, atr_i, ratio, tp1, tp2, rr_c))
        return signal, order

    def _confirmed(self, k: int, zone_top: float) -> bool:
        """Confirmation pattern (spec.confirm_type) at the close of bar k."""
        s, o, c, kind = self.s, self.o, self.c, self.spec.confirm_type
        if kind == "reclaim":
            return s * (c[k] - zone_top) > 0 and s * (c[k] - o[k]) > 0
        if k == 0:
            return False
        if kind == "engulf":
            return (s * (c[k] - o[k]) > 0 and s * (c[k - 1] - o[k - 1]) < 0
                    and s * (c[k] - o[k - 1]) >= 0 and s * (o[k] - c[k - 1]) <= 0)
        r0, r1 = self.rsi[k - 1], self.rsi[k]
        return r0 < 40 and r1 > r0 if s > 0 else r0 > 60 and r1 < r0

    def _meta(self, ext, leg, atr_i, ratio, tp1, tp2, rr) -> dict:
        """Trade meta frozen into an order (fills are added when the trade is built)."""
        return {"anchor": self.anchor.price, "extreme": ext, "leg_size": leg, "atr": atr_i,
                "entry_ratio": ratio, "tp1": tp1, "tp2": tp2, "rr": rr,
                "entry_mode": self.spec.entry_mode}


def _filter_ok(spec: StrategySpec, s: int, c: list[float], ema_t, ema50, adx_v) -> list[bool]:
    """Per-bar trend/ADX filter for side s (nan fails); legs read it at their extreme bar."""
    kind = spec.trend_filter
    if kind == "none":
        ok = [True] * len(c)
    elif kind == "ema":
        ok = [s * (ci - e) > 0 for ci, e in zip(c, ema_t)]
    elif kind == "ema_slope":
        ok = [k >= 10 and s * (c[k] - ema_t[k]) > 0 and s * (ema_t[k] - ema_t[k - 10]) > 0
              for k in range(len(c))]
    else:
        ok = [s * (e50 - e) > 0 and s * (ci - e50) > 0 for ci, e50, e in zip(c, ema50, ema_t)]
    if spec.adx_min > 0:
        ok = [k and a >= spec.adx_min for k, a in zip(ok, adx_v)]
    return ok


def _pick_fill(orders: list[_Order], o: float, h: float, l: float, slip: float):
    """The order that fills in this bar and its fill price: limits fill when the range reaches px
    (at the open if it gapped through); if several fill, the one priced closest to the open wins."""
    best, best_px, best_d = None, 0.0, math.inf
    for od in orders:
        s = od.side.s
        if od.market:
            px, d = o * (1 + s * slip), 0.0
        elif s * ((l if s > 0 else h) - od.px) <= 0:
            px, d = (od.px if s * (o - od.px) > 0 else o), abs(od.px - o)
        else:
            continue
        if d < best_d:
            best, best_px, best_d = od, px, d
    return best, best_px


class FibEngine:
    """Bar-by-bar state machine for one StrategySpec on one symbol (see DESIGN.md)."""

    def __init__(self, spec: StrategySpec, costs: Costs | None = None, strategy_id: str | None = None):
        self.spec = spec.validate()
        self.costs = costs or Costs()
        self.strategy_id = strategy_id or spec.spec_id()

    def run(self, bars: Bars, trade_start_t: int | None = None,
            skip_legs: frozenset[str] | set[str] = frozenset()) -> RunResult:
        """Replay the strategy over bars. Orders fill only on bars with t >= trade_start_t;
        legs whose leg_id is in skip_legs are ignored entirely (no signal, no order)."""
        sp, n = self.spec, len(bars)
        skip_legs = skip_legs or frozenset()
        if bars.tf != sp.timeframe:
            raise ValueError(f"bars timeframe {bars.tf!r} != spec timeframe {sp.timeframe!r}")
        if n == 0:
            return RunResult([], None, [], 0)
        t, o, h, l, c = bars.t, bars.o, bars.h, bars.l, bars.c
        atr = ind.atr(h, l, c, sp.swing_atr_len)
        ema_t = ind.ema(c, sp.trend_ema) if sp.trend_filter != "none" else None
        ema50 = ind.ema(c, 50) if sp.trend_filter == "ema_stack" else None
        adx_v = ind.adx(h, l, c, 14) if sp.adx_min > 0 else None
        rsi = ind.rsi(c, 14) if sp.entry_mode == "confirm" and sp.confirm_type == "rsi" else None
        ratio = {"lo": sp.entry_zone_lo, "hi": sp.entry_zone_hi,
                 "mid": (sp.entry_zone_lo + sp.entry_zone_hi) / 2}[sp.entry_at]
        sides = [_Side(s, sp, ratio, bars, atr, _filter_ok(sp, s, c, ema_t, ema50, adx_v), rsi)
                 for s in (1, -1) if sp.direction in ("both", "long" if s > 0 else "short")]
        tracker = PivotTracker(sp.swing_mult, sp.swing_mode, sp.swing_pct)
        start = -math.inf if trade_start_t is None else trade_start_t
        signal_from = start - TF_SECONDS[sp.timeframe]
        slip = self.costs.slippage
        trades: list[Trade] = []
        signals: list[Signal] = []
        orders: list[_Order] = []
        pos: _Position | None = None
        pos_order: _Order | None = None
        for i in range(n):
            ti = t[i]
            # 1. fills during bar i (orders were placed at the close of bar i-1)
            if ti >= start:
                if pos is not None:
                    pos.intrabar(i, ti, o[i], h[i], l[i])
                elif orders:
                    pos_order, px = _pick_fill(orders, o[i], h[i], l[i], slip)
                    if pos_order is not None:
                        pos_order.side.active = False          # the leg is used
                        pos = _Position(pos_order.side.s, i, px, pos_order.stop, pos_order.tp1,
                                        pos_order.tp2, sp, self.costs)
                        pos.intrabar(i, ti, o[i], h[i], l[i])
                if pos is not None and pos.exit_idx is not None:
                    trades.append(self._trade(bars, pos, pos_order))
                    pos = None
            # 2. close of bar i: pivots, setups, position close rules, then orders for bar i+1
            piv = tracker.update(i, ti, h[i], l[i], atr[i])
            candidates = []
            for sd in sides:
                if piv is not None and piv.kind == sd.kind:
                    sd.set_anchor(piv, i, skip_legs)
                elif sd.active:
                    sd.advance(i)
                if not sd.active:
                    continue
                signal, order = sd.on_close(i)
                if signal is not None and ti >= signal_from:
                    signals.append(Signal(self.strategy_id, bars.symbol, bars.tf, sd.s, sd.leg_id,
                                          ti, *signal))
                if order is not None:
                    candidates.append(order)
            if pos is not None:
                pos.at_close(i, ti, c[i], atr[i])
                if pos.exit_idx is not None:
                    trades.append(self._trade(bars, pos, pos_order))
                    pos = None
            orders = candidates if pos is None else []
        open_trade = self._trade(bars, pos, pos_order) if pos is not None else None
        return RunResult(trades, open_trade, signals, n)

    def _trade(self, bars: Bars, pos: _Position, order: _Order) -> Trade:
        r = pos.result(bars.t, bars.c)
        return Trade(self.strategy_id, bars.symbol, bars.tf, pos.side, order.leg_id, order.setup_t,
                     bars.t[pos.entry_idx], pos.entry_px, pos.init_stop,
                     None if r["open"] else bars.t[r["exit_idx"]], r["exit_price"], r["exit_reason"],
                     r["ret_pct"], r["risk_pct"], r["r_multiple"], r["exit_idx"] - pos.entry_idx,
                     dict(order.meta, fills=r["fills"]))
