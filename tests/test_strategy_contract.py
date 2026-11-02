"""Black-box contract tests for fibbot/strategy.py, written from fibbot/DESIGN.md only.

Every scenario is built by hand on 1h bars and traded with swing_mode="pct" (swing_pct=0.30 unless
noted), trend_filter="none", min_leg_atr=0, stop_atr_buffer=0 so pivots and levels are easy to
reason about; scenarios assert their expected pivots with fib.zigzag as a precondition, so a
failure there is a test-construction error, not an engine bug.

Canonical long setup (`setup_rows`, indices with the default pre_n=30):
  0..29  flat bars o=c=85, h=86, l=84
  A=30   anchor low 80                         (L pivot at A, confirmed at J)
  31     rally bar to 100 (low 80.5)
  J=32   leg high 120 (o=99, l=98, c=118)
Leg A->J: entry (ratio 0.5, entry_at="lo") 100, stop (ratio 0.75) 90, tp1 (ext 1.0) 120,
tp2 (ext 1.5) 140, blended reward/risk 3.0.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import unittest

from fibbot import fib, indicators
from fibbot.strategy import (PARAM_SPACE, PRESETS, FibEngine, RunResult, StrategySpec,
                             manage_position, warmup_bars)
from fibbot.types import Bars, Costs, TF_SECONDS

T0 = 1_699_999_200
STEP = TF_SECONDS["1h"]
FEE, SLIP = 0.001, 0.0005
COSTS = Costs(FEE, SLIP)
A, J = 30, 32
LEG = f"L{T0 + A * STEP}"
BE_LONG = 100.0 * (1 + 2 * FEE)
BE_SHORT = 100.0 * (1 - 2 * FEE)

BASE = dict(name="t", timeframe="1h", direction="long", swing_mode="pct", swing_atr_len=5,
            swing_pct=0.3, min_leg_atr=0.0, setup_expiry_bars=40, trend_filter="none",
            adx_min=0.0, require_structure=False, entry_zone_lo=0.5, entry_zone_hi=0.618,
            entry_mode="limit", entry_at="lo", confirm_type="reclaim", confirm_window=6,
            stop_ratio=0.75, stop_atr_buffer=0.0, tp1_ext=1.0, tp1_frac=0.5, tp2_ext=1.5,
            move_stop_to_be=False, trail_atr=0.0, max_hold_bars=100, min_rr=1.0)

NO_FILL = (117.0, 117.5, 105.0, 108.0)   # J+1: stays above the 100 limit
FILL = (104.0, 105.0, 98.0, 101.0)       # J+2: limit 100 fills at 100, low 98 > stop 90
TP1 = (102.0, 121.0, 101.0, 119.0)       # J+3: tp1 120 (the extreme moves to 121)
TP2 = (119.0, 141.0, 118.0, 139.0)       # J+4: tp2 140


def t(i):
    return T0 + i * STEP


def make_bars(rows, symbol="TESTUSDT"):
    return Bars.from_rows(symbol, "1h", [(t(i), o, h, l, c, 1.0)
                                         for i, (o, h, l, c) in enumerate(rows)])


def setup_rows(pre_n=30, pre=85.0, a1=(81.0, 100.0, 80.5, 99.0), j=(99.0, 120.0, 98.0, 118.0)):
    """Flat pre bars, anchor low 80 at index pre_n, rally bar, leg high 120 at pre_n + 2."""
    return [(pre, pre + 1, pre - 1, pre)] * pre_n + [(pre, pre + 0.5, 80.0, 81.0), a1, j]


ENTERED = setup_rows() + [NO_FILL, FILL]   # long entered at 100 on bar 34


def mirror(rows, axis=200.0):
    """Price mirror p -> axis - p (highs and lows swap): turns the long setup into a short one."""
    return [(axis - o, axis - l, axis - h, axis - c) for o, h, l, c in rows]


def spec(**kw):
    s = StrategySpec(**{**BASE, **kw})
    s.validate()
    return s


def run(rows, costs=COSTS, trade_start_t=None, skip_legs=frozenset(), strategy_id=None, **kw):
    return FibEngine(spec(**kw), costs, strategy_id).run(
        make_bars(rows), trade_start_t=trade_start_t, skip_legs=skip_legs)


def pivots(rows, pct=0.3):
    return [(p.kind, p.idx, p.confirm_idx)
            for p in fib.zigzag(make_bars(rows), None, mode="pct", pct=pct)]


def long_ret(fills, entry, fee=FEE):
    return sum(f * p for f, p in fills) / entry * (1 - fee) - (1 + fee)


def short_ret(fills, entry, fee=FEE):
    return (1 - fee) - sum(f * p for f, p in fills) / entry * (1 + fee)


def canon(x):
    """nan-safe structural form for exact equality of trades/signals."""
    if isinstance(x, float):
        return "nan" if math.isnan(x) else x
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    return x


def random_bars(seed, n=200, vol=0.02):
    """Deterministic random walk with occasional opening gaps."""
    rnd = random.Random(seed)
    rows, px = [], 100.0
    for _ in range(n):
        o = px * (1 + rnd.gauss(0, vol / 4)) if rnd.random() < 0.2 else px
        c = o * math.exp(rnd.gauss(0, vol))
        rows.append((o, max(o, c) * (1 + abs(rnd.gauss(0, vol / 3))),
                     min(o, c) * (1 - abs(rnd.gauss(0, vol / 3))), c))
        px = c
    return make_bars(rows, "RNDUSDT")


class EngineCase(unittest.TestCase):
    def close(self, a, b, msg=None):
        self.assertAlmostEqual(a, b, delta=1e-9, msg=msg)

    def check_trade(self, tr, *, side, leg, entry_i, entry, stop, fills, reason):
        """fills: [(frac, price, bar_idx)] exit fills; for an open trade the last one is the mark."""
        fp = [(f, p) for f, p, _ in fills]
        ret = (long_ret if side == 1 else short_ret)(fp, entry)
        risk = abs(entry - stop) / entry
        last_i = fills[-1][2]
        self.assertEqual((tr.side, tr.leg_id, tr.symbol, tr.tf), (side, leg, "TESTUSDT", "1h"))
        self.assertEqual(tr.entry_t, t(entry_i))
        self.assertLess(tr.setup_t, tr.entry_t)
        self.assertEqual(tr.exit_reason, reason)
        self.assertEqual(tr.bars_held, last_i - entry_i)
        self.close(tr.entry_price, entry, "entry_price")
        self.close(tr.stop_price, stop, "stop_price")
        self.close(tr.exit_price, sum(f * p for f, p in fp), "exit_price")
        self.close(tr.ret_pct, ret, "ret_pct")
        self.close(tr.risk_pct, risk, "risk_pct")
        self.close(tr.r_multiple, ret / risk, "r_multiple")
        if reason == "open":
            self.assertIsNone(tr.exit_t)
            return
        self.assertEqual(tr.exit_t, t(last_i))
        got = tr.meta["fills"]
        self.assertEqual(len(got), len(fills), got)
        for g, (f, p, i) in zip(got, fills):
            self.close(g[0], f, "fill frac")
            self.close(g[1], p, "fill price")
            self.assertEqual(g[3], t(i))

    def only_trade(self, res):
        self.assertIsNone(res.open_trade)
        self.assertEqual(len(res.trades), 1, res.trades)
        return res.trades[0]

    def only_open(self, res):
        self.assertEqual(res.trades, [])
        self.assertIsNotNone(res.open_trade)
        return res.open_trade

    def assert_flat(self, res):
        self.assertEqual(res.trades, [])
        self.assertIsNone(res.open_trade)


# --------------------------------------------------------------------------- StrategySpec

DESIGN_DEFAULTS = dict(
    name="custom", timeframe="4h", direction="long", swing_mode="atr", swing_atr_len=14,
    swing_mult=3.0, swing_pct=0.05, min_leg_atr=4.0, setup_expiry_bars=40, trend_filter="ema",
    trend_ema=200, adx_min=0.0, require_structure=False, entry_zone_lo=0.618, entry_zone_hi=0.65,
    entry_mode="limit", entry_at="mid", confirm_type="reclaim", confirm_window=6,
    stop_ratio=0.886, stop_atr_buffer=0.25, tp1_ext=1.0, tp1_frac=0.5, tp2_ext=1.618,
    move_stop_to_be=True, trail_atr=0.0, max_hold_bars=120, min_rr=1.5)

CHOICES = {"timeframe": ["1h", "4h", "1d"], "direction": ["long", "short", "both"],
           "swing_mode": ["atr", "pct"], "trend_filter": ["none", "ema", "ema_slope", "ema_stack"],
           "entry_mode": ["limit", "confirm"], "entry_at": ["lo", "mid", "hi"],
           "confirm_type": ["reclaim", "engulf", "rsi"]}

# (field, valid edge values, invalid values) for single-field ranges stated in DESIGN.md.
BOUNDS = [
    ("swing_atr_len", [5, 50], [4, 51]),
    ("swing_mult", [1.0, 10.0], [0.99, 10.01]),
    ("swing_pct", [0.01, 0.30], [0.009, 0.31]),
    ("min_leg_atr", [0.0, 30.0], [-0.01, 30.01]),
    ("setup_expiry_bars", [3, 500], [2, 501]),
    ("trend_ema", [20, 400], [19, 401]),
    ("adx_min", [0.0, 50.0], [-0.1, 50.1]),
    ("stop_atr_buffer", [0.0, 3.0], [-0.01, 3.01]),
    ("tp1_frac", [0.05, 1.0], [0.04, 1.01]),
    ("confirm_window", [1, 50], [0, 51]),
    ("max_hold_bars", [1, 2000], [0, 2001]),
    ("min_rr", [0.0, 10.0], [-0.01, 10.01]),
]


class TestStrategySpec(unittest.TestCase):
    def assertInvalid(self, field=None, **kw):
        with self.assertRaises(ValueError) as cm:
            StrategySpec(**kw).validate()
        if field:
            self.assertTrue(str(cm.exception).startswith(field), str(cm.exception))

    def test_fields_and_defaults_match_design(self):
        """StrategySpec: exactly the documented fields, with the documented defaults."""
        self.assertEqual({f.name for f in dataclasses.fields(StrategySpec)}, set(DESIGN_DEFAULTS))
        s = StrategySpec()
        for k, v in DESIGN_DEFAULTS.items():
            self.assertEqual(getattr(s, k), v, k)

    def test_validate_returns_self(self):
        """validate() -> "StrategySpec": returns self."""
        s = StrategySpec()
        self.assertIs(s.validate(), s)

    def test_single_field_bounds(self):
        """Field comments give inclusive ranges; out of range raises ValueError("field: reason")."""
        for field, good, bad in BOUNDS:
            for v in good:
                with self.subTest(field=field, value=v):
                    StrategySpec(**{field: v}).validate()
            for v in bad:
                with self.subTest(field=field, value=v):
                    self.assertInvalid(field, **{field: v})

    def test_choice_fields(self):
        """Choice fields accept exactly the documented options ("15m" is not a spec timeframe)."""
        for field, options in CHOICES.items():
            for v in options:
                StrategySpec(**{field: v}).validate()
            with self.subTest(field=field):
                self.assertInvalid(field, **{field: "bogus"})
        self.assertInvalid("timeframe", timeframe="15m")

    def test_entry_zone_bounds(self):
        """entry_zone_lo 0.236..0.886; entry_zone_lo <= entry_zone_hi <= 0.886."""
        StrategySpec(entry_zone_lo=0.236).validate()
        StrategySpec(entry_zone_lo=0.5, entry_zone_hi=0.5, stop_ratio=0.886).validate()
        StrategySpec(entry_zone_lo=0.886, entry_zone_hi=0.886, stop_ratio=1.0).validate()
        self.assertInvalid("entry_zone_lo", entry_zone_lo=0.235)
        self.assertInvalid(entry_zone_lo=0.887, entry_zone_hi=0.887, stop_ratio=1.0)
        self.assertInvalid(entry_zone_lo=0.65, entry_zone_hi=0.618)
        self.assertInvalid(entry_zone_lo=0.5, entry_zone_hi=0.887, stop_ratio=1.0)

    def test_stop_ratio_bounds(self):
        """stop_ratio > entry_zone_hi and <= 1.2."""
        StrategySpec(stop_ratio=0.651).validate()
        StrategySpec(stop_ratio=1.2).validate()
        self.assertInvalid(stop_ratio=0.65)            # == entry_zone_hi (default 0.65)
        self.assertInvalid(stop_ratio=0.6)
        self.assertInvalid("stop_ratio", stop_ratio=1.21)

    def test_target_bounds(self):
        """tp1_ext 0.3..3.0; tp2_ext > tp1_ext and <= 4.236, ignored when tp1_frac == 1."""
        StrategySpec(tp1_ext=0.3).validate()
        StrategySpec(tp1_ext=3.0, tp2_ext=4.236).validate()
        self.assertInvalid("tp1_ext", tp1_ext=0.29)
        self.assertInvalid(tp1_ext=3.01, tp2_ext=4.236)
        self.assertInvalid("tp2_ext", tp2_ext=4.24)
        self.assertInvalid(tp1_ext=1.0, tp2_ext=1.0)
        self.assertInvalid(tp1_ext=1.2, tp2_ext=1.0)
        StrategySpec(tp1_ext=1.0, tp2_ext=1.0, tp1_frac=1.0).validate()

    def test_dict_round_trip(self):
        """to_dict()/from_dict(): round trip; unknown keys -> ValueError; missing -> defaults."""
        s = StrategySpec(name="x", direction="both", swing_pct=0.1, require_structure=True)
        d = s.to_dict()
        self.assertEqual(set(d), set(DESIGN_DEFAULTS))
        json.dumps(d)
        self.assertEqual(StrategySpec.from_dict(d), s)
        self.assertEqual(StrategySpec.from_dict({}), StrategySpec())
        part = StrategySpec.from_dict({"swing_pct": 0.1})
        self.assertEqual(part, StrategySpec(swing_pct=0.1))
        with self.assertRaises(ValueError):
            StrategySpec.from_dict({"bogus": 1})

    def test_spec_id(self):
        """spec_id(): 10 hex chars, sha1 of canonical JSON of to_dict() minus "name"."""
        s = StrategySpec()
        sid = s.spec_id()
        self.assertRegex(sid, r"^[0-9a-f]{10}$")
        self.assertEqual(s.with_overrides(name="other").spec_id(), sid)
        self.assertNotEqual(s.with_overrides(min_rr=2.0).spec_id(), sid)
        self.assertEqual(StrategySpec.from_dict(s.to_dict()).spec_id(), sid)
        d = {k: v for k, v in s.to_dict().items() if k != "name"}
        candidates = {hashlib.sha1(json.dumps(d, sort_keys=True, separators=sep).encode())
                      .hexdigest()[:10] for sep in (None, (",", ":"))}
        self.assertIn(sid, candidates)

    def test_with_overrides(self):
        """with_overrides(**kw) returns a new spec; the original is unchanged."""
        s = StrategySpec()
        s2 = s.with_overrides(direction="short", trail_atr=1.5)
        self.assertIsNot(s2, s)
        self.assertEqual((s2.direction, s2.trail_atr), ("short", 1.5))
        self.assertEqual((s.direction, s.trail_atr), ("long", 0.0))
        self.assertEqual(dataclasses.replace(s2, direction="long", trail_atr=0.0), s)

    def test_param_space(self):
        """PARAM_SPACE: every field except name, with type/min/max/step or choices, default, doc."""
        self.assertEqual(set(PARAM_SPACE), set(DESIGN_DEFAULTS) - {"name"})
        ints = {"swing_atr_len", "setup_expiry_bars", "trend_ema", "confirm_window", "max_hold_bars"}
        bools = {"require_structure", "move_stop_to_be"}
        for k, p in PARAM_SPACE.items():
            with self.subTest(param=k):
                want = ("choice" if k in CHOICES else "bool" if k in bools
                        else "int" if k in ints else "float")
                self.assertEqual(p["type"], want)
                self.assertEqual(p["default"], DESIGN_DEFAULTS[k])
                self.assertIsInstance(p["doc"], str)
                if want == "choice":
                    self.assertEqual(sorted(p["choices"]), sorted(CHOICES[k]))
                elif want in ("int", "float"):
                    self.assertLessEqual(p["min"], p["default"])
                    self.assertLessEqual(p["default"], p["max"])
                    self.assertGreater(p["step"], 0)
        for field, good, _ in BOUNDS:
            self.assertEqual((PARAM_SPACE[field]["min"], PARAM_SPACE[field]["max"]),
                             (good[0], good[1]), field)
        for field, hi in (("entry_zone_lo", 0.886), ("entry_zone_hi", 0.886),
                          ("stop_ratio", 1.2), ("tp1_ext", 3.0), ("tp2_ext", 4.236)):
            self.assertAlmostEqual(PARAM_SPACE[field]["max"], hi, msg=field)
        self.assertAlmostEqual(PARAM_SPACE["entry_zone_lo"]["min"], 0.236)
        self.assertAlmostEqual(PARAM_SPACE["tp1_ext"]["min"], 0.3)


# --------------------------------------------------------------------------- presets / warmup

GP = dict(entry_zone_lo=0.618, entry_zone_hi=0.65)
PRESET_TABLE = {
    "golden_pocket_4h": dict(GP, timeframe="4h", direction="long", trend_filter="ema",
                             trend_ema=200, entry_mode="limit", entry_at="mid", stop_ratio=0.886,
                             stop_atr_buffer=0.25, tp1_ext=1.0, tp1_frac=0.5, tp2_ext=1.618),
    "golden_pocket_1h": dict(GP, timeframe="1h", direction="long", trend_filter="ema",
                             trend_ema=200, entry_mode="limit", entry_at="mid", stop_ratio=0.886,
                             stop_atr_buffer=0.25, tp1_ext=1.0, tp1_frac=0.5, tp2_ext=1.618),
    "golden_pocket_both_4h": dict(GP, timeframe="4h", direction="both"),
    "ote_confirm_1h": dict(timeframe="1h", direction="long", entry_zone_lo=0.618,
                           entry_zone_hi=0.786, entry_mode="confirm", confirm_type="reclaim",
                           stop_ratio=1.0, stop_atr_buffer=0.1, tp1_ext=1.0, tp2_ext=1.618),
    "shallow_trend_4h": dict(timeframe="4h", direction="long", trend_filter="ema_stack",
                             adx_min=20.0, entry_zone_lo=0.382, entry_zone_hi=0.5,
                             entry_mode="limit", entry_at="lo", stop_ratio=0.786, tp1_ext=1.0,
                             tp2_ext=1.618),
    "deep_786_4h": dict(timeframe="4h", direction="long", trend_filter="none",
                        entry_zone_lo=0.786, entry_zone_hi=0.886, entry_mode="limit",
                        entry_at="lo", stop_ratio=1.0, stop_atr_buffer=0.3, tp1_ext=0.618,
                        tp2_ext=1.0),
    "golden_pocket_1d": dict(timeframe="1d", direction="long", entry_zone_lo=0.5,
                             entry_zone_hi=0.65, entry_mode="limit", entry_at="mid",
                             stop_ratio=0.886, tp1_ext=1.0, tp2_ext=1.618),
    "ext_runner_4h": dict(GP, timeframe="4h", direction="long", tp1_ext=1.0, tp1_frac=0.3,
                          trail_atr=2.5, tp2_ext=2.618),
}


class TestPresetsAndWarmup(unittest.TestCase):
    def test_presets_match_table(self):
        """PRESETS: at least the eight documented presets, valid, with the tabled parameters."""
        for name, want in PRESET_TABLE.items():
            with self.subTest(preset=name):
                p = PRESETS[name]
                self.assertIsInstance(p, StrategySpec)
                self.assertIs(p.validate(), p)
                for k, v in want.items():
                    self.assertEqual(getattr(p, k), v, k)

    def test_presets_are_distinct(self):
        """Each preset is a different strategy (distinct spec_id)."""
        ids = [p.spec_id() for p in PRESETS.values()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_warmup_bars(self):
        """warmup_bars(spec) = max(3 * trend_ema, 5 * swing_atr_len, 4 * 14, 300)."""
        cases = [({}, 600), ({"trend_ema": 20, "swing_atr_len": 5}, 300), ({"trend_ema": 150}, 450),
                 ({"trend_ema": 50, "swing_atr_len": 50}, 300), ({"trend_ema": 400}, 1200),
                 ({"trend_ema": 101}, 303)]
        for kw, want in cases:
            self.assertEqual(warmup_bars(StrategySpec(**kw)), want, kw)
        for p in PRESETS.values():
            self.assertEqual(warmup_bars(p), max(3 * p.trend_ema, 5 * p.swing_atr_len, 56, 300))


# --------------------------------------------------------------------------- FibEngine: limit long

class TestLimitLong(EngineCase):
    def test_scenario_preconditions(self):
        """Scenario sanity (fib.zigzag, pct 0.3): only the L pivot at A, confirmed at J."""
        self.assertEqual(pivots(setup_rows() + [NO_FILL, FILL, TP1, TP2]), [("L", A, J)])

    def test_full_cycle_tp1_then_tp2(self):
        """Limit fill at min(o, px); tp1 fills tp1_frac at max(o, tp1); tp2 closes the rest
        ("target"); accounting: ret_pct = sum(f*p)/P*(1-f) - (1+f), risk_pct = |P - stop|/P,
        r_multiple = ret/risk, exit_price = sum(f*p), bars_held = exit_idx - entry_idx."""
        res = run(setup_rows() + [NO_FILL, FILL, TP1, TP2])
        self.assertIsInstance(res, RunResult)
        self.assertEqual(res.n_bars, 37)
        tr = self.only_trade(res)
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 35), (0.5, 140.0, 36)], reason="target")
        self.assertEqual(tr.strategy_id, spec().spec_id())
        self.close(tr.ret_pct, 1.3 * 0.999 - 1.001)

    def test_trade_meta(self):
        """Trade meta: anchor, extreme, leg_size, atr, entry_ratio, tp1, tp2, rr, fills, entry_mode."""
        tr = self.only_trade(run(setup_rows() + [NO_FILL, FILL, TP1, TP2]))
        m = tr.meta
        for k in ("anchor", "extreme", "leg_size", "atr", "entry_ratio", "tp1", "tp2", "rr",
                  "fills", "entry_mode"):
            self.assertIn(k, m)
        for k, v in (("anchor", 80.0), ("extreme", 120.0), ("leg_size", 40.0),
                     ("entry_ratio", 0.5), ("tp1", 120.0), ("tp2", 140.0)):
            self.close(m[k], v, k)
        self.assertEqual(m["entry_mode"], "limit")

    def test_signal_once_per_leg_with_fields(self):
        """The first bar at which a leg qualifies emits one Signal (entry_px, stop_px, tp1, tp2,
        meta anchor/extreme/atr/rr/mode); it is not re-emitted while the leg keeps qualifying."""
        rows = setup_rows() + [NO_FILL, FILL, TP1, TP2]
        res = run(rows)
        self.assertEqual(len(res.signals), 1)
        s = res.signals[0]
        self.assertEqual((s.strategy_id, s.symbol, s.tf, s.side, s.leg_id, s.t),
                         (spec().spec_id(), "TESTUSDT", "1h", 1, LEG, t(J)))
        for got, want in ((s.entry_price, 100.0), (s.stop_price, 90.0), (s.tp1, 120.0),
                          (s.tp2, 140.0), (s.meta["anchor"], 80.0), (s.meta["extreme"], 120.0),
                          (s.meta["rr"], 3.0)):
            self.close(got, want)
        b = make_bars(rows)
        self.close(s.meta["atr"], indicators.atr(b.h, b.l, b.c, 5)[J])
        self.assertEqual(s.meta["mode"], "limit")

    def test_strategy_id_and_default_costs(self):
        """strategy_id defaults to spec.spec_id(); costs default to Costs()."""
        rows = setup_rows() + [NO_FILL, FILL, TP1, TP2]
        res = FibEngine(spec(), None, "my-id").run(make_bars(rows))
        self.assertEqual(res.trades[0].strategy_id, "my-id")
        self.assertEqual(res.signals[0].strategy_id, "my-id")
        a = FibEngine(spec()).run(make_bars(rows))
        b = FibEngine(spec(), Costs()).run(make_bars(rows))
        self.assertEqual(canon([x.to_dict() for x in a.trades]), canon([x.to_dict() for x in b.trades]))
        self.assertEqual(a.trades[0].strategy_id, spec().spec_id())

    def test_targets_frozen_at_fill(self):
        """Stop/targets are frozen into the position at fill time: after the extreme moves to 121,
        tp2 stays 140 (a re-computed tp2 would be 141.5 and bar 36's high 141 would miss it)."""
        tr = self.only_trade(run(setup_rows() + [NO_FILL, FILL, TP1, TP2]))
        self.close(tr.meta["fills"][-1][1], 140.0)
        self.assertEqual(tr.exit_t, t(36))

    def test_used_leg_is_not_reentered(self):
        """Mark the leg used on fill: price returning to the zone later does not re-enter it."""
        rows = setup_rows() + [NO_FILL, FILL, TP1, TP2, (139.0, 139.5, 110.0, 112.0),
                               (112.0, 113.0, 99.5, 101.0)]
        self.assertEqual(pivots(rows), [("L", A, J)])
        self.only_trade(run(rows))

    def test_limit_needs_close_strictly_above_entry(self):
        """Limit placed for bar i+1 only if c[i] > entry_px: c[J] == 100 places nothing, so bar 33
        (low 99.5) cannot fill; the order from bar 33's close misses bar 34 (low 100.2)."""
        rows = setup_rows(j=(99.0, 120.0, 98.0, 100.0)) + [(100.0, 101.0, 99.5, 100.8),
                                                           (100.8, 101.2, 100.2, 101.0)]
        self.assert_flat(run(rows))

    def test_limit_reprices_with_moving_extreme(self):
        """Orders are re-computed every bar: a new high 130 re-prices entry to 105 / stop 92.5 /
        tp1 130 / tp2 155, while the Signal keeps its arming-time prices."""
        rows = setup_rows() + [(117.0, 130.0, 116.0, 128.0), (128.0, 128.5, 104.0, 106.0)]
        self.assertEqual(pivots(rows), [("L", A, J)])
        res = run(rows)
        tr = self.only_open(res)
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=105.0, stop=92.5,
                         fills=[(1.0, 106.0 * (1 - SLIP), 34)], reason="open")
        self.close(tr.meta["tp1"], 130.0)
        self.close(tr.meta["tp2"], 155.0)
        self.close(tr.meta["extreme"], 130.0)
        s = res.signals[0]
        self.assertEqual(len(res.signals), 1)
        self.close(s.entry_price, 100.0)
        self.close(s.stop_price, 90.0)

    def test_limit_gap_fill_at_open(self):
        """Long limit fills at min(o[i], px): opening at 98 below the 100 limit fills at 98."""
        tr = self.only_open(run(setup_rows() + [NO_FILL, (98.0, 99.0, 95.0, 97.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=98.0, stop=90.0,
                         fills=[(1.0, 97.0 * (1 - SLIP), 34)], reason="open")

    def test_limit_touch_boundary(self):
        """Long limit fills if l[i] <= px: low == 100 fills, low 100.01 does not."""
        tr = self.only_open(run(setup_rows() + [NO_FILL, (104.0, 105.0, 100.0, 101.0)]))
        self.close(tr.entry_price, 100.0)
        self.assert_flat(run(setup_rows() + [NO_FILL, (104.0, 105.0, 100.01, 101.0)]))

    def test_open_trade_marked_at_last_close(self):
        """open_trade: exit_t None, exit_reason "open", marked at c[-1] with exit slippage and
        fees; it is not in trades."""
        tr = self.only_open(run(ENTERED + [(101.0, 103.0, 99.0, 102.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 102.0 * (1 - SLIP), 35)], reason="open")

    def test_open_trade_after_partial_tp1(self):
        """An open position after tp1 is marked as tp1_frac at tp1 plus the rest at c[-1]."""
        tr = self.only_open(run(ENTERED + [TP1]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 35), (0.5, 119.0 * (1 - SLIP), 35)], reason="open")


# --------------------------------------------------------------------------- entry bar rules

class TestEntryBar(EngineCase):
    def test_fill_beyond_stop_exits_immediately(self):
        """Entry bar: fill price already beyond the stop (gap open 88 < stop 90) -> exit at the
        fill price with slippage, reason "stop", bars_held 0."""
        rows = ENTERED[:-1] + [(88.0, 89.0, 85.0, 86.0)]
        self.assertEqual(pivots(rows), [("L", A, J)])
        tr = self.only_trade(run(rows))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=88.0, stop=90.0,
                         fills=[(1.0, 88.0 * (1 - SLIP), 34)], reason="stop")

    def test_entry_bar_range_hits_stop(self):
        """Entry bar: l[i] <= stop -> exit at stop*(1-slippage), reason "stop", same bar."""
        tr = self.only_trade(run(ENTERED[:-1] + [(104.0, 105.0, 89.0, 92.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 90.0 * (1 - SLIP), 34)], reason="stop")

    def test_market_fill_beyond_stop_exits_at_open(self):
        """Entry bar: the stop is checked at the worse of o[i] and the fill price, i.e. the market
        fill's pre-slippage price o[i]; it exits at o*(1-slip) (not o*(1+slip)*(1-slip)), also
        when only the slippage lifts the fill above the stop."""
        touch = CSETUP + [NO_FILL, (101.0, 104.0, 99.0, 103.0)]     # reclaim at 34 -> market for 35
        for o in (88.0, 89.97):
            with self.subTest(open=o):
                self.assertEqual(o * (1 + SLIP) > 90.0, o == 89.97)
                tr = self.only_trade(run(touch + [(o, 91.0, 85.0, 86.0)], entry_mode="confirm"))
                self.check_trade(tr, side=1, leg=LEG, entry_i=35, entry=o * (1 + SLIP), stop=90.0,
                                 fills=[(1.0, o * (1 - SLIP), 35)], reason="stop")

    def test_targets_never_credited_on_entry_bar(self):
        """Targets are never credited on the entry bar (high 125 >= tp1 120 on the fill bar)."""
        tr = self.only_open(run(ENTERED[:-1] + [(101.0, 125.0, 99.0, 110.0),
                                                (110.0, 115.0, 105.0, 112.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 112.0 * (1 - SLIP), 35)], reason="open")


# --------------------------------------------------------------------------- position management

class TestManagement(EngineCase):
    def test_stop_gap_exits_at_open(self):
        """Rule 1: o <= stop -> exit the remainder at o*(1-slippage)."""
        tr = self.only_trade(run(ENTERED + [(88.0, 89.0, 85.0, 86.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 88.0 * (1 - SLIP), 35)], reason="stop")

    def test_stop_touch_boundary(self):
        """Rule 2: l <= stop exits at stop*(1-slippage); low 90.01 does not."""
        tr = self.only_trade(run(ENTERED + [(99.0, 100.0, 90.0, 91.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 90.0 * (1 - SLIP), 35)], reason="stop")
        self.only_open(run(ENTERED + [(99.0, 100.0, 90.01, 91.0)]))

    def test_stop_before_targets_same_bar(self):
        """Worst case: a bar reaching both the stop and tp1 is a full stop-out, no tp1 fill."""
        tr = self.only_trade(run(ENTERED + [(101.0, 125.0, 89.0, 110.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 90.0 * (1 - SLIP), 35)], reason="stop")

    def test_tp1_touch_boundary(self):
        """Rule 3: h >= tp1 fills tp1_frac at max(o, tp1) (high exactly 120 counts)."""
        tr = self.only_open(run(ENTERED + [(102.0, 120.0, 101.0, 115.0)]))
        self.close(tr.exit_price, 0.5 * 120.0 + 0.5 * 115.0 * (1 - SLIP))

    def test_target_gaps_fill_at_open(self):
        """Targets are limits: tp1 at max(o, tp1) and tp2 at max(o, tp2), no slippage."""
        tr = self.only_trade(run(ENTERED + [(125.0, 130.0, 124.0, 128.0),
                                            (150.0, 155.0, 149.0, 152.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 125.0, 35), (0.5, 150.0, 36)], reason="target")

    def test_tp1_and_tp2_same_bar(self):
        """Rule 3: after tp1, "then if h >= tp2 close the remainder" on the same bar."""
        tr = self.only_trade(run(ENTERED + [(102.0, 145.0, 101.0, 143.0)]))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 35), (0.5, 140.0, 35)], reason="target")

    def test_tp1_frac_one_closes_at_tp1(self):
        """tp1_frac == 1: tp1 closes the whole trade, reason "target"."""
        tr = self.only_trade(run(ENTERED + [TP1], tp1_frac=1.0))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 120.0, 35)], reason="target")

    def test_breakeven_from_next_bar(self):
        """move_stop_to_be: stop -> max(stop, entry*(1+2*fee)) from the NEXT bar: the tp1 bar's
        low 99 (< 100.2) does not exit; the next bar's low 99.5 exits at 100.2*(1-slip),
        reason "breakeven"."""
        rows = ENTERED + [(105.0, 121.0, 99.0, 110.0), (108.0, 109.0, 99.5, 101.0)]
        self.assertEqual(pivots(rows), [("L", A, J)])
        tr = self.only_trade(run(rows, move_stop_to_be=True))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 35), (0.5, BE_LONG * (1 - SLIP), 36)],
                         reason="breakeven")

    def test_breakeven_gap_exits_at_open(self):
        """Rule 1 with a moved stop: open 99 <= breakeven 100.2 -> exit at 99*(1-slip),
        reason "breakeven"."""
        tr = self.only_trade(run(ENTERED + [(105.0, 121.0, 99.0, 110.0), (99.0, 100.0, 97.0, 98.0)],
                                 move_stop_to_be=True))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 35), (0.5, 99.0 * (1 - SLIP), 36)],
                         reason="breakeven")

    def test_without_breakeven_initial_stop_stays(self):
        """move_stop_to_be False: after tp1 the stop stays at the initial 90 (reason "stop")."""
        rows = ENTERED + [(105.0, 121.0, 99.0, 110.0), (108.0, 109.0, 99.5, 101.0),
                          (100.0, 101.0, 89.0, 91.0)]
        self.assertEqual(pivots(rows), [("L", A, J)])
        tr = self.only_trade(run(rows))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 35), (0.5, 90.0 * (1 - SLIP), 37)], reason="stop")

    def test_time_stop(self):
        """Rule 5: at the close, i - entry_idx >= max_hold_bars exits at c*(1-slip), "time"."""
        rows = ENTERED + [(101.0, 103.0, 99.0, 102.0), (102.0, 104.0, 100.0, 103.0),
                          (103.0, 105.0, 101.0, 104.0), (104.0, 141.0, 103.0, 139.0)]
        tr = self.only_trade(run(rows, max_hold_bars=3))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 104.0 * (1 - SLIP), 37)], reason="time")

    def test_time_stop_after_partial(self):
        """Time stop closes only the remainder after a tp1 partial."""
        rows = ENTERED + [TP1, (119.0, 125.0, 115.0, 118.0)]
        tr = self.only_trade(run(rows, max_hold_bars=2))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 35), (0.5, 118.0 * (1 - SLIP), 36)], reason="time")

    def test_trailing_stop(self):
        """Rule 5: trail_atr > 0 and tp1 hit -> at the close stop = max(stop, c - k*ATR); a
        trailed stop above breakeven exits with reason "trail" (ATR = Wilder ATR(swing_atr_len))."""
        rows = ENTERED + [TP1, (118.0, 119.0, 106.0, 108.0)]
        b = make_bars(rows)
        trail = 119.0 - 1.0 * indicators.atr(b.h, b.l, b.c, 5)[35]
        self.assertTrue(106.0 < trail < 118.0 and trail > BE_LONG)
        for be in (True, False):            # "trail" also when the stop never went to breakeven
            with self.subTest(move_stop_to_be=be):
                tr = self.only_trade(run(rows, trail_atr=1.0, move_stop_to_be=be))
                self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                                 fills=[(0.5, 120.0, 35), (0.5, trail * (1 - SLIP), 36)],
                                 reason="trail")

    def test_trail_below_breakeven_keeps_breakeven(self):
        """stop = max(stop, c - k*ATR): a trail level below breakeven leaves the breakeven stop,
        so the exit reason is "breakeven"."""
        rows = ENTERED + [TP1, (108.0, 109.0, 99.5, 101.0)]
        b = make_bars(rows)
        self.assertLess(119.0 - 2.0 * indicators.atr(b.h, b.l, b.c, 5)[35], BE_LONG)
        tr = self.only_trade(run(rows, trail_atr=2.0, move_stop_to_be=True))
        self.check_trade(tr, side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(0.5, 120.0, 35), (0.5, BE_LONG * (1 - SLIP), 36)],
                         reason="breakeven")


# --------------------------------------------------------------------------- setup lifecycle

EXPIRY_BARS = [NO_FILL, (108.0, 110.0, 103.0, 104.0), (104.0, 106.0, 101.0, 105.0)]


class TestSetupLifecycle(EngineCase):
    def test_expiry_boundary_fill(self):
        """Leg dies when i - ext_idx > setup_expiry_bars: with 3, the order placed at the close of
        bar J+3 (3 > 3 is false) still fills on bar J+4."""
        rows = setup_rows() + EXPIRY_BARS + [(105.0, 106.0, 99.0, 101.0)]
        tr = self.only_open(run(rows, setup_expiry_bars=3))
        self.assertEqual(tr.entry_t, t(36))

    def test_expiry_boundary_dead(self):
        """With setup_expiry_bars=3 the leg is dead at the close of J+4, so J+5 cannot fill."""
        rows = setup_rows() + EXPIRY_BARS + [(105.0, 107.0, 102.0, 106.0), (106.0, 107.0, 99.0, 101.0)]
        self.assert_flat(run(rows, setup_expiry_bars=3))
        self.only_open(run(rows, setup_expiry_bars=4))

    def test_extreme_index_is_first_occurrence(self):
        """ext_idx is the first occurrence of the highest high: an equal high 120 at J+2 does not
        extend the setup's life."""
        rows = setup_rows() + [NO_FILL, (108.0, 120.0, 107.0, 119.0), (119.0, 119.5, 106.0, 107.0),
                               (107.0, 108.0, 102.0, 106.0), (106.0, 107.0, 99.0, 101.0)]
        self.assertEqual(pivots(rows), [("L", A, J)])
        self.assert_flat(run(rows, setup_expiry_bars=3))

    def _dead_stop_rows(self, low):
        return setup_rows(j=(99.0, 120.0, 98.0, 99.9)) + [
            (99.9, 100.0, low, 92.0), (92.0, 102.0, 91.5, 101.0), (101.0, 101.5, 99.0, 100.5)]

    def test_low_through_stop_kills_unfilled_leg(self):
        """Dead leg: the low of bar i <= the leg's stop while unfilled (89.5 and exactly 90 kill
        it); the later return to the zone does not trade."""
        for low in (89.5, 90.0):
            rows = self._dead_stop_rows(low)
            self.assertEqual(pivots(rows), [("L", A, J)])
            self.assert_flat(run(rows))

    def test_low_above_stop_keeps_leg(self):
        """Control: low 90.5 > stop keeps the leg alive; it fills on the return to 100."""
        tr = self.only_open(run(self._dead_stop_rows(90.5)))
        self.assertEqual((tr.entry_t, tr.entry_price), (t(35), 100.0))

    def test_new_extreme_bar_low_still_kills(self):
        """The dead check also applies on the bar that sets a new extreme (intrabar order unknown,
        resolved against the trade): high 125 moves the stop to 91.25 and that bar's low 90.5
        kills the leg; with low 92 the leg lives and fills on the later return to 102.5. (c[J]
        below the 100 entry, so no limit rests on bar 33.)"""
        for low, alive in ((90.5, False), (92.0, True)):
            with self.subTest(low=low):
                rows = setup_rows(j=(101.0, 120.0, 100.5, 99.9)) + [
                    (99.9, 125.0, low, 100.0), (100.0, 104.0, 99.5, 103.0), (103.0, 104.0, 102.0, 103.5)]
                self.assertEqual(pivots(rows), [("L", A, J)])
                res = run(rows)
                self.assertEqual(res.trades, [])
                self.assertEqual(res.open_trade is not None, alive)
                if alive:
                    self.assertEqual((res.open_trade.entry_t, res.open_trade.entry_price), (t(35), 102.5))

    def test_signal_emitted_on_qualify_even_without_order(self):
        """The first bar at which a leg qualifies emits a Signal, even when c[i] <= entry_px means
        no limit is placed (c[J] = 99.9)."""
        res = run(self._dead_stop_rows(89.5))
        self.assertEqual([(s.leg_id, s.t) for s in res.signals], [(LEG, t(J))])

    def _anchor_rows(self, low):
        return setup_rows(j=(99.0, 120.0, 98.0, 99.9)) + [
            (99.9, 100.0, low, 85.0), (85.0, 102.0, 84.0, 101.0), (101.0, 101.5, 99.0, 100.5)]

    def test_trading_below_anchor_kills_leg(self):
        """Dead leg: min low over (anchor.idx, i] < anchor.price (79 < 80) even with the stop
        (ratio 1.2 -> 72) not reached."""
        rows = self._anchor_rows(79.0)
        self.assertEqual(pivots(rows), [("L", A, J), ("H", J, 33)])
        self.assert_flat(run(rows, stop_ratio=1.2))

    def test_anchor_comparison_is_strict(self):
        """"traded below the anchor" is strict: lows of exactly 80 or 80.5 keep the leg alive."""
        for low in (80.0, 80.5):
            rows = self._anchor_rows(low)
            self.assertEqual(pivots(rows), [("L", A, J), ("H", J, 33)])
            tr = self.only_open(run(rows, stop_ratio=1.2))
            self.assertEqual(tr.entry_t, t(35))
            self.close(tr.stop_price, fib.retrace_price(80.0, 120.0, 1.2))

    def test_min_leg_atr(self):
        """Qualifies only when extreme - anchor >= min_leg_atr * atr[i] (ATR(swing_atr_len));
        the order placed at J's close fills on bar 33 only if the leg qualified at J."""
        rows = setup_rows() + [(117.0, 117.5, 99.0, 101.0)]
        b = make_bars(rows)
        k = 40.0 / indicators.atr(b.h, b.l, b.c, 5)[J]
        res = run(rows, min_leg_atr=k * (1 - 1e-9))
        self.assertEqual([s.t for s in res.signals], [t(J)])
        self.assertEqual(self.only_open(res).entry_t, t(33))
        res = run(rows, min_leg_atr=k * (1 + 1e-6))
        self.assertEqual(res.signals, [])
        self.assert_flat(res)
        res = run(rows, min_leg_atr=30.0)
        self.assertEqual(res.signals, [])
        self.assert_flat(res)

    def test_min_rr_boundary(self):
        """RR = (tp1_frac*(tp1-entry) + (1-tp1_frac)*(tp2-entry)) / (entry-stop) >= min_rr; here
        exactly 3.0."""
        rows = setup_rows() + [NO_FILL, FILL, TP1, TP2]
        self.assertEqual(len(run(rows, min_rr=3.0).trades), 1)
        res = run(rows, min_rr=3.01)
        self.assertEqual(res.signals, [])
        self.assert_flat(res)

    def test_rr_uses_tp1_only_when_full_exit_at_tp1(self):
        """If tp1_frac == 1, RR uses tp1 only: (120-100)/(100-90) = 2.0."""
        rows = setup_rows() + [NO_FILL, FILL, TP1]
        res = run(rows, tp1_frac=1.0, min_rr=2.0)
        self.close(res.signals[0].meta["rr"], 2.0)
        self.assertEqual(len(res.trades), 1)
        res = run(rows, tp1_frac=1.0, min_rr=2.01)
        self.assertEqual(res.signals, [])
        self.assert_flat(res)

    def test_tp1_must_exceed_entry(self):
        """Qualification requires tp1 > entry: tp1_ext 0.4 puts tp1 at 96 < entry 100, even though
        the blended RR (with tp2 at 200) would pass min_rr 0."""
        res = run(setup_rows() + [NO_FILL, FILL, TP1], tp1_ext=0.4, tp2_ext=3.0, min_rr=0.0)
        self.assertEqual(res.signals, [])
        self.assert_flat(res)


# --------------------------------------------------------------------------- filters

FILTER_TAIL = EXPIRY_BARS + [(105.0, 106.0, 99.0, 101.0)]      # no new high after J; fill at 36


class TestFilters(EngineCase):
    def test_ema_filter_passes(self):
        """trend "ema": c > EMA(trend_ema) at ext_idx passes -> same trade as "none"."""
        rows = setup_rows() + [NO_FILL, FILL, TP1, TP2]
        b = make_bars(rows)
        self.assertGreater(b.c[J], indicators.ema(b.c, 20)[J])
        got = run(rows, strategy_id="x", trend_filter="ema", trend_ema=20)
        want = run(rows, strategy_id="x")
        self.assertEqual(canon([x.to_dict() for x in got.trades]),
                         canon([x.to_dict() for x in want.trades]))

    def test_ema_nan_fails(self):
        """Trend filter: nan fails (EMA(400) undefined on 37 bars)."""
        res = run(setup_rows() + [NO_FILL, FILL, TP1, TP2], trend_filter="ema", trend_ema=400)
        self.assertEqual(res.signals, [])
        self.assert_flat(res)

    def test_trend_filter_evaluated_at_extreme_bar(self):
        """The trend filter is evaluated at ext_idx, not at bar i: c[J] > EMA20[J] passes even
        though c[33] < EMA20[33] when the re-placed order (filled on bar 34) is made."""
        rows = setup_rows(pre=108.0) + [(117.0, 117.5, 100.5, 101.0), (101.0, 102.0, 99.5, 101.0)]
        self.assertEqual(pivots(rows), [("L", A, J)])
        b = make_bars(rows)
        e = indicators.ema(b.c, 20)
        self.assertTrue(b.c[J] > e[J] and b.c[33] < e[33])
        tr = self.only_open(run(rows, trend_filter="ema", trend_ema=20))
        self.assertEqual((tr.entry_t, tr.entry_price), (t(34), 100.0))

    def test_ema_slope(self):
        """ema_slope: c > EMA and EMA[ext_idx] > EMA[ext_idx-10]; a falling EMA fails."""
        for pre, ok in ((85.0, True), (108.0, False)):
            rows = setup_rows(pre=pre) + FILTER_TAIL
            self.assertEqual(pivots(rows), [("L", A, J)])
            b = make_bars(rows)
            e = indicators.ema(b.c, 20)
            self.assertGreater(b.c[J], e[J])
            self.assertEqual(e[J] > e[J - 10], ok)
            res = run(rows, trend_filter="ema_slope", trend_ema=20)
            self.assertEqual(len(res.signals), int(ok))
            self.assertEqual(res.open_trade is not None, ok)

    def test_ema_stack(self):
        """ema_stack: EMA(50) > EMA(trend_ema) and c > EMA(50) at ext_idx; nan EMA(50) fails."""
        rows = setup_rows(pre_n=110) + [NO_FILL, FILL]
        jj = 112
        b = make_bars(rows)
        e50, e100, e20 = (indicators.ema(b.c, n)[jj] for n in (50, 100, 20))
        self.assertTrue(e50 > e100 and b.c[jj] > e50 and e50 < e20)
        tr = self.only_open(run(rows, trend_filter="ema_stack", trend_ema=100))
        self.assertEqual(tr.entry_t, t(jj + 2))
        res = run(rows, trend_filter="ema_stack", trend_ema=20)
        self.assertEqual(res.signals, [])
        self.assert_flat(res)
        res = run(setup_rows() + [NO_FILL, FILL], trend_filter="ema_stack", trend_ema=20)
        self.assertEqual(res.signals, [])

    def test_adx_min(self):
        """adx[ext_idx] >= adx_min (ADX(14)) when adx_min > 0."""
        rows = setup_rows() + FILTER_TAIL
        b = make_bars(rows)
        a = indicators.adx(b.h, b.l, b.c, 14)[J]
        self.assertTrue(0 < a < 50)
        tr = self.only_open(run(rows, adx_min=a * (1 - 1e-6)))
        self.assertEqual(tr.entry_t, t(36))
        res = run(rows, adx_min=a * (1 + 1e-6))
        self.assertEqual(res.signals, [])
        self.assert_flat(res)


# --------------------------------------------------------------------------- multiple legs

HIGHER_LOW = [(118.0, 118.5, 100.5, 101.0), (101.0, 102.0, 83.5, 85.0), (85.0, 100.0, 84.0, 99.0),
              (99.0, 124.0, 98.0, 122.0), (120.0, 121.0, 103.0, 105.0)]
LOWER_LOW = [(118.0, 118.5, 100.5, 101.0), (101.0, 102.0, 78.0, 79.0), (79.0, 95.0, 78.5, 94.0),
             (94.0, 124.0, 93.0, 122.0), (120.0, 121.0, 100.0, 104.0)]
LEG2 = f"L{T0 + 34 * STEP}"


class TestLegs(EngineCase):
    def test_two_legs(self):
        """Leg 1 fills on bar 34 and stops out on its entry bar; the next confirmed low starts a
        new leg (leg_id L{anchor.t}) with its own signal and trade."""
        rows = setup_rows() + HIGHER_LOW
        self.assertEqual(pivots(rows), [("L", A, J), ("H", J, 34), ("L", 34, 36)])
        res = run(rows)
        self.assertEqual(len(res.trades), 1)
        self.check_trade(res.trades[0], side=1, leg=LEG, entry_i=34, entry=100.0, stop=90.0,
                         fills=[(1.0, 90.0 * (1 - SLIP), 34)], reason="stop")
        self.check_trade(res.open_trade, side=1, leg=LEG2, entry_i=37, entry=103.75,
                         stop=93.625, fills=[(1.0, 105.0 * (1 - SLIP), 37)], reason="open")
        self.assertEqual([(s.leg_id, s.t) for s in res.signals], [(LEG, t(J)), (LEG2, t(36))])
        s2 = res.signals[1]
        for got, want in ((s2.entry_price, 103.75), (s2.stop_price, 93.625), (s2.tp1, 124.0),
                          (s2.tp2, 144.25)):
            self.close(got, want)

    def test_signal_emitted_while_a_position_is_open(self):
        """Signals do not depend on being flat: leg 2 qualifies at bar 36 while leg 1's position
        (stop 1.2 -> 72) is open, so it signals but never orders."""
        res = run(setup_rows() + HIGHER_LOW, stop_ratio=1.2)
        self.assertEqual(res.trades, [])
        self.assertEqual((res.open_trade.leg_id, res.open_trade.entry_t), (LEG, t(34)))
        self.assertEqual([(s.leg_id, s.t) for s in res.signals], [(LEG, t(J)), (LEG2, t(36))])

    def test_structure_higher_low_passes(self):
        """require_structure (long): anchor low 83.5 > previous confirmed low 80 qualifies."""
        res = run(setup_rows() + HIGHER_LOW, require_structure=True, skip_legs={LEG})
        tr = self.only_open(res)
        self.assertEqual((tr.leg_id, tr.entry_t), (LEG2, t(37)))
        self.close(tr.entry_price, 103.75)

    def test_structure_lower_low_fails(self):
        """require_structure: anchor low 78 < previous low 80 never qualifies (control without
        the filter trades it at 101)."""
        rows = setup_rows() + LOWER_LOW
        self.assertEqual(pivots(rows), [("L", A, J), ("H", J, 34), ("L", 34, 36)])
        res = run(rows, require_structure=True, skip_legs={LEG})
        self.assertEqual(res.signals, [])
        self.assert_flat(res)
        tr = self.only_open(run(rows, skip_legs={LEG}))
        self.assertEqual((tr.leg_id, tr.entry_t), (LEG2, t(37)))
        self.close(tr.entry_price, 101.0)
        self.close(tr.stop_price, 89.5)


# --------------------------------------------------------------------------- gating / skip_legs

class TestGating(EngineCase):
    ROWS = setup_rows() + [NO_FILL, FILL, TP1, TP2]

    def test_signal_window_includes_bar_before_start(self):
        """Signals: armed at a bar with t >= trade_start_t - TF; trades unchanged by the gate."""
        res = run(self.ROWS, trade_start_t=t(33))
        self.assertEqual([s.t for s in res.signals], [t(J)])
        self.assertEqual(canon([x.to_dict() for x in res.trades]),
                         canon([x.to_dict() for x in run(self.ROWS).trades]))

    def test_signal_window_boundary(self):
        """A leg first armed before trade_start_t - TF never emits a signal, but may still trade
        on bars with t >= trade_start_t."""
        for start in (t(33) + 1, t(34)):
            res = run(self.ROWS, trade_start_t=start)
            self.assertEqual(res.signals, [])
            self.assertEqual(self.only_trade(res).entry_t, t(34))

    def test_no_fills_before_trade_start(self):
        """Fills on bar i only if t[i] >= trade_start_t: the bar-34 fill is suppressed and the
        re-placed order never fills later."""
        res = run(self.ROWS, trade_start_t=t(35))
        self.assert_flat(res)
        self.assertEqual(res.signals, [])

    def test_skip_legs(self):
        """leg_id in skip_legs -> the leg is dead: no signal, no trade; unrelated ids no effect;
        the caller's set is not mutated."""
        skip = {LEG}
        res = run(self.ROWS, skip_legs=skip)
        self.assertEqual(skip, {LEG})
        self.assertEqual(res.signals, [])
        self.assert_flat(res)
        res = run(self.ROWS, skip_legs=frozenset({"L123", f"S{t(A)}"}))
        self.assertEqual(len(res.trades), 1)


# --------------------------------------------------------------------------- short / both

SHORT_LEG = f"S{T0 + A * STEP}"


class TestShortAndBoth(EngineCase):
    def test_short_mirror_full_cycle(self):
        """Short mirrors every comparison: anchor = last confirmed HIGH (leg_id S{t}), limit fills
        if h >= px at max(o, px), targets at min(o, tp), ret = (1-f) - sum(f*p)/P*(1+f)."""
        rows = mirror(setup_rows() + [NO_FILL, FILL, TP1, TP2])
        self.assertEqual(pivots(rows), [("H", A, J)])
        res = run(rows, direction="short")
        tr = self.only_trade(res)
        self.check_trade(tr, side=-1, leg=SHORT_LEG, entry_i=34, entry=100.0, stop=110.0,
                         fills=[(0.5, 80.0, 35), (0.5, 60.0, 36)], reason="target")
        self.close(tr.ret_pct, 0.999 - 0.7 * 1.001)
        s = res.signals[0]
        self.assertEqual((len(res.signals), s.side, s.leg_id, s.t), (1, -1, SHORT_LEG, t(J)))
        for got, want in ((s.entry_price, 100.0), (s.stop_price, 110.0), (s.tp1, 80.0),
                          (s.tp2, 60.0)):
            self.close(got, want)

    def test_direction_restricts_side(self):
        """direction "long" never trades the short setup and vice versa."""
        long_rows = setup_rows() + [NO_FILL, FILL, TP1, TP2]
        for rows, direction in ((mirror(long_rows), "long"), (long_rows, "short")):
            res = run(rows, direction=direction)
            self.assertEqual(res.signals, [])
            self.assert_flat(res)

    def test_short_entry_bar_stop(self):
        """Short entry bar: h >= stop exits at stop*(1+slippage), reason "stop"."""
        rows = mirror(setup_rows() + [NO_FILL, (104.0, 105.0, 89.0, 92.0)])
        tr = self.only_trade(run(rows, direction="short"))
        self.check_trade(tr, side=-1, leg=SHORT_LEG, entry_i=34, entry=100.0, stop=110.0,
                         fills=[(1.0, 110.0 * (1 + SLIP), 34)], reason="stop")

    def test_short_stop_gap(self):
        """Short rule 1: o >= stop exits at o*(1+slippage)."""
        rows = mirror(ENTERED + [(88.0, 89.0, 85.0, 86.0)])
        tr = self.only_trade(run(rows, direction="short"))
        self.check_trade(tr, side=-1, leg=SHORT_LEG, entry_i=34, entry=100.0, stop=110.0,
                         fills=[(1.0, 112.0 * (1 + SLIP), 35)], reason="stop")

    def test_short_breakeven(self):
        """Short breakeven: stop = min(stop, entry*(1-2*fee)) from the next bar."""
        rows = mirror(ENTERED + [(105.0, 121.0, 99.0, 110.0), (108.0, 109.0, 99.5, 101.0)])
        self.assertEqual(pivots(rows), [("H", A, J)])
        tr = self.only_trade(run(rows, direction="short", move_stop_to_be=True))
        self.check_trade(tr, side=-1, leg=SHORT_LEG, entry_i=34, entry=100.0, stop=110.0,
                         fills=[(0.5, 80.0, 35), (0.5, BE_SHORT * (1 + SLIP), 36)],
                         reason="breakeven")

    def _conflict_rows(self, conflict_open):
        return setup_rows(a1=(81.0, 90.0, 80.5, 89.0), j=(91.0, 120.0, 91.0, 118.0)) + [
            (118.0, 118.5, 108.0, 110.0), (110.0, 110.5, 101.0, 103.0),
            (conflict_open, 111.0, 99.5, 105.0)]

    def test_both_conflict_short_closer_to_open(self):
        """Both limits would fill: fill the one whose price is closer to o[i] (short 110.5 vs
        long 100 with o=108), cancel the other."""
        rows = self._conflict_rows(108.0)
        self.assertEqual(pivots(rows, 0.15), [("L", A, J), ("H", J, 34)])
        res = run(rows, direction="both", swing_pct=0.15)
        tr = self.only_open(res)
        self.assertEqual((tr.side, tr.leg_id, tr.entry_t), (-1, f"S{t(J)}", t(35)))
        self.close(tr.entry_price, fib.retrace_price(120.0, 101.0, 0.5))
        self.close(tr.stop_price, fib.retrace_price(120.0, 101.0, 0.75))
        self.assertEqual([(s.side, s.leg_id, s.t) for s in res.signals],
                         [(1, LEG, t(J)), (-1, f"S{t(J)}", t(34))])

    def test_both_conflict_long_closer_to_open(self):
        """Conflict mirror case: with o=101 the long limit (100) is closer and wins."""
        tr = self.only_open(run(self._conflict_rows(101.0), direction="both", swing_pct=0.15))
        self.assertEqual((tr.side, tr.leg_id, tr.entry_t, tr.entry_price), (1, LEG, t(35), 100.0))


# --------------------------------------------------------------------------- confirm mode

# Same leg as setup_rows(), but bar J opens at 101 with low 100.5 > zone top 100, so the leg's own
# extreme bar is not a touch and every window below starts at the touch bar named in the test.
CSETUP = setup_rows(j=(101.0, 120.0, 100.5, 118.0))


def window_rows(after):
    """Touch at 34 (red, close < zone top), two red bars inside the zone top, then `after`."""
    return CSETUP + [NO_FILL, (104.0, 105.0, 99.0, 99.5), (100.4, 101.0, 100.2, 100.3),
                           (100.6, 100.9, 100.3, 100.5)] + after


def rsi_rows():
    """Slow 24-bar grind from 118 down into the zone (touch at 53), first up-close at 57."""
    rows, c = list(CSETUP), 118.0       # a copy: CSETUP is shared by other tests
    for _ in range(24):
        o, c = c, c - 0.875
        rows.append((o, o + 0.1, c - 0.1, c))
    return rows + [(c, c + 1.2, c - 0.1, c + 1.0), (98.1, 98.5, 97.9, 98.3)]


class TestConfirmMode(EngineCase):
    def test_reclaim_after_touch(self):
        """confirm/reclaim: touch (l <= zone_top, l > stop), then c[k] > zone_top and c[k] > o[k]
        places a market order filling at o[k+1]*(1+slippage); stop/targets unchanged."""
        rows = CSETUP + [NO_FILL, (104.0, 105.0, 99.0, 102.0), (101.0, 104.0, 100.5, 103.0),
                               (103.5, 105.0, 102.0, 104.0), (104.0, 121.0, 103.0, 119.0), TP2]
        res = run(rows, entry_mode="confirm")
        tr = self.only_trade(res)
        self.check_trade(tr, side=1, leg=LEG, entry_i=36, entry=103.5 * (1 + SLIP), stop=90.0,
                         fills=[(0.5, 120.0, 37), (0.5, 140.0, 38)], reason="target")
        self.assertEqual(tr.meta["entry_mode"], "confirm")
        self.assertEqual([(s.leg_id, s.t, s.meta["mode"]) for s in res.signals],
                         [(LEG, t(J), "confirm")])

    def test_confirmation_on_touch_bar(self):
        """The window is j..j+window, so the touch bar itself can confirm."""
        rows = CSETUP + [NO_FILL, (101.0, 104.0, 99.0, 103.0), (103.5, 105.0, 102.0, 104.0)]
        tr = self.only_open(run(rows, entry_mode="confirm"))
        self.assertEqual(tr.entry_t, t(35))
        self.close(tr.entry_price, 103.5 * (1 + SLIP))

    def test_extreme_bar_is_never_a_touch(self):
        """A touch needs j > ext_idx: bar 33 makes the new high 125 (zone top 102.5) and its low
        102 is in the zone, but the low may precede the high, so the green bar 34 (low 102.8,
        above the zone top) has nothing to confirm. Control: a bar-34 low of 102.4 is a touch
        and reclaims at once."""
        for low34, trades in ((102.8, False), (102.4, True)):
            with self.subTest(low34=low34):
                rows = CSETUP + [(118.0, 125.0, 102.0, 103.0), (103.0, 104.5, low34, 104.0),
                                 (104.0, 105.0, 103.5, 104.5)]
                self.assertEqual(pivots(rows), [("L", A, J)])
                res = run(rows, entry_mode="confirm")
                if not trades:
                    self.assert_flat(res)
                    continue
                tr = self.only_open(res)
                self.assertEqual(tr.entry_t, t(35))
                self.close(tr.entry_price, 104.0 * (1 + SLIP))
                self.close(tr.stop_price, 91.25)

    def test_new_extreme_clears_touch(self):
        """A new extreme moves the zone and clears an earlier touch: touch at 34, new high 121 on
        the red bar 35, so the reclaim-shaped bar 36 (low 100.6 above the new zone top 100.5)
        does not enter. Control: without bar 35's new high the touch stands and bar 36 confirms."""
        for high35, trades in ((121.0, False), (119.0, True)):
            with self.subTest(high35=high35):
                rows = CSETUP + [NO_FILL, (104.0, 105.0, 99.0, 99.5), (100.3, high35, 99.8, 100.2),
                                 (100.2, 102.0, 100.6, 101.8), (101.8, 103.0, 101.5, 102.5)]
                self.assertEqual(pivots(rows), [("L", A, J)])
                res = run(rows, entry_mode="confirm")
                if not trades:
                    self.assert_flat(res)
                    continue
                tr = self.only_open(res)
                self.assertEqual(tr.entry_t, t(37))
                self.close(tr.entry_price, 101.8 * (1 + SLIP))

    def test_confirm_window_boundary(self):
        """Confirmation allowed through bar j+window inclusive: touch at 34, reclaim at 37 trades
        with window 3 but not with window 2."""
        rows = window_rows([(100.5, 104.0, 100.4, 103.5), (103.5, 106.0, 103.0, 105.0)])
        self.assert_flat(run(rows, entry_mode="confirm", confirm_window=2))
        tr = self.only_open(run(rows, entry_mode="confirm", confirm_window=3))
        self.assertEqual(tr.entry_t, t(38))
        self.close(tr.entry_price, 103.5 * (1 + SLIP))

    def test_retouch_after_window(self):
        """No confirmation in the window -> untouched again; a later re-touch can confirm."""
        rows = window_rows([(100.7, 100.9, 100.4, 100.5), (100.5, 100.6, 99.6, 99.9),
                            (99.9, 104.0, 99.8, 103.5), (103.5, 104.0, 103.0, 103.8)])
        tr = self.only_open(run(rows, entry_mode="confirm", confirm_window=2))
        self.assertEqual(tr.entry_t, t(40))
        self.close(tr.entry_price, 103.5 * (1 + SLIP))

    def test_confirm_requires_rr_with_close_as_entry(self):
        """A confirmation orders only if the leg still qualifies with entry = c[k]
        (RR 27/13 = 2.077 at c=103)."""
        rows = CSETUP + [NO_FILL, (101.0, 104.0, 99.0, 103.0), (103.5, 105.0, 102.0, 104.0)]
        res = run(rows, entry_mode="confirm", min_rr=2.1)
        self.assertEqual(len(res.signals), 1)
        self.assert_flat(res)
        self.assertEqual(self.only_open(run(rows, entry_mode="confirm", min_rr=2.0)).entry_t, t(35))

    def test_engulf(self):
        """engulf: c[k] > o[k], c[k-1] < o[k-1], c[k] >= o[k-1], o[k] <= c[k-1]."""
        touch = (104.0, 105.0, 99.0, 99.5)
        after = (104.6, 106.0, 104.0, 105.0)
        rows = CSETUP + [NO_FILL, touch, (99.5, 105.0, 99.2, 104.5), after]
        tr = self.only_open(run(rows, entry_mode="confirm", confirm_type="engulf"))
        self.assertEqual(tr.entry_t, t(36))
        self.close(tr.entry_price, 104.6 * (1 + SLIP))
        rows = CSETUP + [NO_FILL, touch, (99.6, 105.0, 99.2, 104.5), after]
        self.assert_flat(run(rows, entry_mode="confirm", confirm_type="engulf"))

    def test_rsi(self):
        """rsi: rsi[k-1] < 40 and rsi[k] > rsi[k-1] (RSI(14)) within the touch window."""
        rows = rsi_rows()
        self.assertEqual(pivots(rows), [("L", A, J)])
        self.assertEqual(pivots(CSETUP + [NO_FILL]), [("L", A, J)])
        b = make_bars(rows)
        r = indicators.rsi(b.c, 14)
        self.assertTrue(b.l[52] > 100.0 >= b.l[53])
        self.assertTrue(r[56] < 40 and r[57] > r[56])
        self.assertTrue(all(r[k] < r[k - 1] for k in range(53, 57)))
        tr = self.only_open(run(rows, entry_mode="confirm", confirm_type="rsi"))
        self.assertEqual(tr.entry_t, t(58))
        self.close(tr.entry_price, 98.1 * (1 + SLIP))


# --------------------------------------------------------------------------- manage_position

def mp_bars(rows):
    return make_bars([(100.0, 101.0, 99.0, 100.0)] + rows)


class TestManagePosition(EngineCase):
    def mp(self, rows, side=1, stop=90.0, tp1=120.0, tp2=140.0, atr=2.0, **kw):
        b = mp_bars(rows)
        return b, manage_position(b, [atr] * len(b), 0, side, 100.0, stop, tp1, tp2, spec(**kw),
                                  COSTS)

    def check(self, out, side, stop, fills, reason, exit_i, is_open=False):
        for k in ("exit_idx", "exit_reason", "fills", "ret_pct", "risk_pct", "r_multiple", "open"):
            self.assertIn(k, out)
        fp = [(f, p) for f, p in fills]
        ret = (long_ret if side == 1 else short_ret)(fp, 100.0)
        risk = abs(100.0 - stop) / 100.0
        self.assertEqual((out["exit_idx"], out["exit_reason"], out["open"]),
                         (exit_i, reason, is_open))
        self.close(out["ret_pct"], ret)
        self.close(out["risk_pct"], risk)
        self.close(out["r_multiple"], ret / risk)
        if not is_open:
            self.assertEqual(len(out["fills"]), len(fills))
            for g, (f, p) in zip(out["fills"], fills):
                self.close(g[0], f)
                self.close(g[1], p)

    def test_trailing_ratchets(self):
        """Trail after tp1: stop = max(stop, breakeven, c - k*atr) at each close, using the atr
        list passed in; exit at the trailed stop, reason "trail"."""
        b, out = self.mp([(100.0, 121.0, 99.5, 119.0), (119.0, 125.0, 118.0, 124.0),
                          (123.0, 123.5, 121.0, 121.5)], trail_atr=1.0, move_stop_to_be=True)
        self.check(out, 1, 90.0, [(0.5, 120.0), (0.5, 122.0 * (1 - SLIP))], "trail", 3)
        self.assertEqual([g[3] for g in out["fills"]], [b.t[1], b.t[3]])

    def test_breakeven(self):
        """move_stop_to_be: breakeven entry*(1+2*fee) active from the bar after tp1."""
        b, out = self.mp([(100.0, 121.0, 99.5, 119.0), (110.0, 111.0, 100.1, 105.0)],
                         move_stop_to_be=True)
        self.check(out, 1, 90.0, [(0.5, 120.0), (0.5, BE_LONG * (1 - SLIP))], "breakeven", 2)

    def test_short_trailing(self):
        """Short mirror: tp1 at min(o, tp1); trail stop = min(stop, c + k*atr); exit at
        stop*(1+slippage)."""
        b, out = self.mp([(100.0, 100.5, 79.0, 81.0), (81.0, 82.0, 75.0, 76.0),
                          (77.0, 78.5, 76.5, 78.2)], side=-1, stop=110.0, tp1=80.0, tp2=60.0,
                         trail_atr=1.0, move_stop_to_be=True, direction="short")
        self.check(out, -1, 110.0, [(0.5, 80.0), (0.5, 78.0 * (1 + SLIP))], "trail", 3)

    def test_time_stop(self):
        """Time stop at the close of entry_idx + max_hold_bars at c*(1-slippage)."""
        b, out = self.mp([(100.0, 102.0, 99.0, 101.0), (101.0, 103.0, 100.0, 102.0),
                          (102.0, 150.0, 101.0, 149.0)], max_hold_bars=2)
        self.check(out, 1, 90.0, [(1.0, 102.0 * (1 - SLIP))], "time", 2)

    def test_stop_rules(self):
        """Rule 1 gap (o <= stop -> o*(1-slip)) and rule 2 worst case (stop before targets)."""
        b, out = self.mp([(85.0, 86.0, 80.0, 82.0)])
        self.check(out, 1, 90.0, [(1.0, 85.0 * (1 - SLIP))], "stop", 1)
        b, out = self.mp([(101.0, 125.0, 89.0, 110.0)])
        self.check(out, 1, 90.0, [(1.0, 90.0 * (1 - SLIP))], "stop", 1)

    def test_open_and_no_entry_bar_target(self):
        """No target credit on the entry bar; still open at the last bar -> "open": true,
        marked at c[-1]*(1-slippage)."""
        b = make_bars([(100.0, 125.0, 99.0, 110.0), (110.0, 115.0, 105.0, 112.0)])
        out = manage_position(b, [2.0, 2.0], 0, 1, 100.0, 90.0, 120.0, 140.0, spec(), COSTS)
        self.assertTrue(out["open"])
        self.close(out["ret_pct"], long_ret([(1.0, 112.0 * (1 - SLIP))], 100.0))
        self.close(out["risk_pct"], 0.1)


# --------------------------------------------------------------------------- determinism / causality

PROP_SPECS = [
    dict(swing_pct=0.05, entry_at="mid", stop_ratio=0.786, move_stop_to_be=True,
         max_hold_bars=30, setup_expiry_bars=30),
    dict(direction="both", swing_mode="atr", swing_atr_len=7, swing_mult=2.5, min_leg_atr=1.0,
         entry_zone_hi=0.65, entry_at="hi", stop_ratio=0.886, stop_atr_buffer=0.25,
         tp1_frac=0.3, tp2_ext=2.0, trail_atr=1.5, move_stop_to_be=True, max_hold_bars=40),
    dict(direction="both", swing_pct=0.04, entry_mode="confirm", confirm_type="reclaim",
         confirm_window=5, entry_zone_lo=0.382, entry_at="mid", stop_ratio=0.886,
         move_stop_to_be=True, min_rr=0.8),
    dict(swing_pct=0.04, entry_mode="confirm", confirm_type="engulf", confirm_window=8,
         entry_zone_lo=0.382, entry_zone_hi=0.786, stop_ratio=1.0, min_rr=0.5),
    dict(direction="both", swing_pct=0.04, entry_mode="confirm", confirm_type="rsi",
         confirm_window=10, entry_zone_lo=0.382, entry_zone_hi=0.786, stop_ratio=1.0, min_rr=0.5),
    dict(direction="both", swing_pct=0.04, trend_filter="ema", trend_ema=20, adx_min=10.0,
         entry_at="mid", stop_ratio=0.886, tp1_frac=1.0, min_rr=0.8),
]


def dicts(items):
    return canon([x.to_dict() for x in items])


class TestDeterminismAndCausality(EngineCase):
    def test_determinism_and_purity(self):
        """Determinism: two runs (same or reused engine) are identical and the input bars are not
        mutated."""
        for i, kw in enumerate(PROP_SPECS):
            with self.subTest(spec=i):
                bars = random_bars(11 + i)
                before = canon(dataclasses.asdict(bars))
                eng = FibEngine(spec(**kw), COSTS)
                r1 = eng.run(bars)
                eng.run(random_bars(99))
                r2 = eng.run(bars)
                r3 = FibEngine(spec(**kw), COSTS).run(random_bars(11 + i))
                self.assertEqual(canon(dataclasses.asdict(bars)), before)
                for r in (r2, r3):
                    self.assertEqual(dicts(r.trades), dicts(r1.trades))
                    self.assertEqual(dicts(r.signals), dicts(r1.signals))
                    self.assertEqual(canon(r.open_trade and r.open_trade.to_dict()),
                                     canon(r1.open_trade and r1.open_trade.to_dict()))

    def check_prefixes(self, bars, eng, start=None):
        full = eng.run(bars, trade_start_t=start)
        allt = full.trades + ([full.open_trade] if full.open_trade else [])
        self.assertGreaterEqual(len(full.trades), 1, "scenario must produce trades")
        self.assertGreaterEqual(len(full.signals), 1, "scenario must produce signals")
        self.assertEqual([x.entry_t for x in full.trades], sorted(x.entry_t for x in full.trades))
        for k in range(1, len(bars) + 1):
            part = eng.run(bars.slice(0, k), trade_start_t=start)
            last = bars.t[k - 1]
            msg = f"prefix k={k}"
            self.assertEqual(part.n_bars, k)
            self.assertEqual(dicts(part.trades),
                             dicts([x for x in full.trades if x.exit_t <= last]), msg)
            self.assertEqual(dicts(part.signals),
                             dicts([s for s in full.signals if s.t <= last]), msg)
            live = [x for x in allt
                    if x.entry_t <= last and (x.exit_t is None or x.exit_t > last)]
            self.assertLessEqual(len(live), 1, msg)
            if live:
                self.assertIsNotNone(part.open_trade, msg)
                key = ("leg_id", "side", "entry_t", "entry_price", "stop_price", "setup_t")
                self.assertEqual([getattr(part.open_trade, a) for a in key],
                                 [getattr(live[0], a) for a in key], msg)
            else:
                self.assertIsNone(part.open_trade, msg)

    def test_prefix_causality(self):
        """For every k, run(bars.slice(0, k)) yields exactly the trades of run(bars) closed at or
        before bar k-1 and the signals with t <= t[k-1]; open positions agree."""
        for i, kw in enumerate(PROP_SPECS):
            with self.subTest(spec=i):
                self.check_prefixes(random_bars(11 + i), FibEngine(spec(**kw), COSTS))

    def test_prefix_causality_with_trade_start(self):
        """Causality also holds with trade_start_t; nothing enters before it and no signal is
        older than trade_start_t - TF."""
        bars = random_bars(12)
        start = bars.t[80]
        eng = FibEngine(spec(**PROP_SPECS[1]), COSTS)
        self.check_prefixes(bars, eng, start)
        res = eng.run(bars, trade_start_t=start)
        for x in res.trades + ([res.open_trade] if res.open_trade else []):
            self.assertGreaterEqual(x.entry_t, start)
        for s in res.signals:
            self.assertGreaterEqual(s.t, start - STEP)

    def test_engine_exits_match_manage_position(self):
        """FibEngine must use the same per-bar exit logic as manage_position (shared code)."""
        bars = random_bars(12)
        sp = spec(**PROP_SPECS[1])
        res = FibEngine(sp, COSTS).run(bars)
        atr = indicators.atr(bars.h, bars.l, bars.c, sp.swing_atr_len)
        checked = 0
        for tr in res.trades + ([res.open_trade] if res.open_trade else []):
            if tr.bars_held == 0:
                continue
            ei = bars.index_at_or_after(tr.entry_t)
            out = manage_position(bars, atr, ei, tr.side, tr.entry_price, tr.stop_price,
                                  tr.meta["tp1"], tr.meta["tp2"], sp, COSTS)
            self.assertEqual(out["open"], tr.exit_t is None)
            if tr.exit_t is not None:
                self.assertEqual(bars.t[out["exit_idx"]], tr.exit_t)
                self.assertEqual(out["exit_reason"], tr.exit_reason)
            self.close(out["ret_pct"], tr.ret_pct)
            self.close(out["r_multiple"], tr.r_multiple)
            checked += 1
        self.assertGreaterEqual(checked, 2)

    def test_empty_and_tiny_inputs(self):
        """Edge inputs: empty or very short bars produce an empty result without raising."""
        for n in (0, 1, 3):
            res = FibEngine(spec(), COSTS).run(make_bars(setup_rows()[:n]))
            self.assertEqual((res.trades, res.open_trade, res.signals, res.n_bars), ([], None, [], n))


if __name__ == "__main__":
    unittest.main()
