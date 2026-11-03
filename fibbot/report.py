"""Human-readable reports (see DESIGN.md, "fibbot/report.py").

render_status() is compact terminal text; write_report() writes home/REPORT.md, Markdown that reads
well on GitHub and on a phone. Both show the same lab state: roster and paper accounts, graduation
checks, recent paper trades, the value of the brain's signal reviews, the latest research and
review journal entries (clipped at line boundaries), system events (journal kind "system":
scheduler step failures, paper-integrity warnings) and graduation changes (kind "graduation"), so an
unattended bot's failures are visible, and the head of the brain notebook.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import home_path
from .data.candles import atomic_write
from .lab import NOTEBOOK_HEADER, Lab, iso_time

DISCLAIMER = "Paper trading only: fibbot never places real orders. Nothing here is financial advice."
RECENT_TRADES = 15
JOURNAL_ENTRIES = 3
JOURNAL_BODY_CHARS = 800
EVENT_ENTRIES = 5              # newest system events (errors, warnings) / graduation changes listed
STALE_BARS = 3                 # candles more than this many bars old are stale (as scheduler.STALE_BARS)
DB_WARN_BYTES = 50_000_000     # GitHub rejects files over 100 MB; the Actions save step refuses over 95 MB
EVENT_BODY_CHARS = 300
NOTEBOOK_LINES = 60
REVIEW_GROUPS = ("take", "skip", "unreviewed")


def _when(t: int | None) -> str:
    """'YYYY-MM-DD HH:MM' UTC, or '-'."""
    return "-" if t is None else datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _f(x, fmt: str = "{:.4g}") -> str:
    """Format a number, '-' for None."""
    return "-" if x is None else fmt.format(x)


def _cell(x) -> str:
    """Markdown table cell text: pipes escaped, newlines flattened."""
    return str(x).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list], right: set[int] = frozenset()) -> list[str]:
    """Markdown table lines; columns whose index is in `right` are right-aligned."""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("--:" if i in right else "---" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def _clip(text: str, n: int) -> str:
    """text cut to about n chars at a line boundary, with "[... k more lines]" for dropped
    lines; a first line longer than n is cut with " [...]"."""
    text = text.strip()
    if len(text) <= n:
        return text
    lines, keep, used = text.splitlines(), [], 0
    for ln in lines:
        if used + len(ln) > n:
            break
        keep.append(ln)
        used += len(ln) + 1
    if not keep:
        return text[:n].rstrip() + " [...]"
    rest = len(lines) - len(keep)
    return "\n".join(keep) + f"\n[... {rest} more line{'' if rest == 1 else 's'}]"


def _events(lab: Lab, kind: str) -> list[dict]:
    """Newest EVENT_ENTRIES journal entries of kind (e.g. "system", "graduation")."""
    return lab.store.journal(EVENT_ENTRIES, kind)


def _db_bytes(lab: Lab) -> int:
    """Size of the SQLite database (plus its WAL), 0 for an in-memory store."""
    path = getattr(lab.store, "path", None)
    total = 0
    for p in ([path, f"{path}-wal"] if path and path != ":memory:" else []):
        try:
            total += Path(p).stat().st_size
        except OSError:
            pass
    return total


def _alerts(lab: Lab, st: dict) -> list[str]:
    """Problems an unattended run must show first: a candle stall (kv data_stall, written by the
    scheduler), a timeframe a roster strategy trades whose oldest per-symbol candles lag by more
    than STALE_BARS bars, and a database approaching GitHub's 100 MB file limit."""
    out = []
    stall = st.get("data_stall")
    if isinstance(stall, dict) and stall.get("since") is not None:
        out.append(f"STALE DATA since {_iso(stall['since'])}: {stall.get('reason', '')}")
    used = {e["timeframe"] for e in st.get("roster", [])}
    for tf, d in sorted((st.get("data") or {}).items()):
        lag = d.get("oldest_lag_bars")
        if tf in used and isinstance(lag, (int, float)) and lag > STALE_BARS:
            out.append(f"STALE CANDLES: {d.get('oldest_symbol')} {tf} ends {d.get('oldest_end')}"
                       f" ({lag:.0f} bars behind)")
    size = _db_bytes(lab)
    if size > DB_WARN_BYTES:
        out.append(f"DATABASE {size / 1e6:.0f} MB: GitHub rejects files over 100 MB and the Actions"
                   " save step refuses over 95 MB (the scheduler compacts old detail daily)")
    return out


def _iso(t) -> str:
    try:
        return iso_time(int(t))
    except (TypeError, ValueError):
        return str(t)


def _errors_line(st: dict) -> str:
    e = st.get("system_events_7d") or {"n": 0, "last": None, "last_t": None}
    return (f"system events (errors, warnings) last 7 days: {e['n']}"
            + (f"; last {e['last_t']}: {e['last']}" if e["n"] else ""))


def _notebook_head(lab: Lab) -> list[str]:
    """First NOTEBOOK_LINES lines of the notebook after its "# Brain notebook" header."""
    lines = lab.notebook().splitlines()
    if lines and lines[0].strip() == NOTEBOOK_HEADER:
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return lines[:NOTEBOOK_LINES]


def _research_journal(lab: Lab) -> list[dict]:
    """Newest JOURNAL_ENTRIES journal entries of kind research or review."""
    rows = lab.store.journal(JOURNAL_ENTRIES, "research") + lab.store.journal(JOURNAL_ENTRIES, "review")
    return sorted(rows, key=lambda r: (r["t"], r["id"]), reverse=True)[:JOURNAL_ENTRIES]


def _names(lab: Lab) -> dict[str, str]:
    return {r["strategy_id"]: r["name"] for r in lab.store.roster(None)}


def _account_cells(e: dict) -> list:
    a = e["account"]
    return [_f(a["days"], "{:.1f}"), f"{a['n_closed']} / {a['n_open']}", _f(a["avg_r"], "{:+.2f}"),
            _f(a["profit_factor"], "{:.2f}"), _f(a["return"], "{:+.1%}"), _f(a["max_drawdown"], "{:.1%}")]


def _trade_cells(r: dict, names: dict[str, str]) -> list:
    is_open = r["exit_t"] is None
    pnl = ("-" if r["notional"] is None else "not taken" if r["notional"] == 0   # 0: risk limits
           else _f(r["pnl"], "{:+.2f}"))
    return [_when(r["entry_t"]), names.get(r["strategy_id"], r["strategy_id"]), r["symbol"],
            "long" if r["side"] > 0 else "short", _f(r["entry_price"], "{:.6g}"),
            _f(r["exit_price"], "{:.6g}") + (" (mark)" if is_open else ""),
            "open" if is_open else r["exit_reason"], _f(r["r_multiple"], "{:+.2f}"), pnl]


def _check_line(c: dict) -> str:
    """GitHub task-list item: [x] passed, [ ] not yet."""
    detail = c.get("note") or f"{_f(c['value'])} (threshold {_f(c['threshold'])})"
    return f"- [{'x' if c['passed'] else ' '}] {c['name']}: {detail}"


def _data_lines(st: dict) -> list[str]:
    """Universe, candle ranges and research windows per timeframe."""
    lines = [f"universe ({len(st['universe'])}): {', '.join(st['universe']) or '-'}"]
    for tf, d in st["data"].items():
        w = st["windows"].get(tf, {})
        win = (w["error"] if "error" in w else
               f"research {w['research'][0]}..{w['research'][1]}, validation {w['validation'][0]}..{w['validation'][1]}")
        lines.append(f"{tf} candles: {d['symbols']} symbols {d['start']}..{d['end']}; {win}")
    if not st["data"]:
        lines.append("no candles cached yet (python -m fibbot data)")
    return lines


def _budget_line(st: dict) -> str:
    b = st["budget"]
    if st["cycle"] is None:
        return "budget: unlimited (no brain cycle running)"
    return (f"budget ({st['cycle']} cycle): {b['backtest_units_left']} backtest units,"
            f" {b['validations_left']} validations left")


def _activity_line(st: dict) -> str:
    ts, ex, last = st["trial_stats"], st["experiments"], st["last"]
    kinds = ", ".join(f"{k} {v}" for k, v in ex.items() if k != "total" and v)
    v7 = st.get("validations_7d")
    return (f"trials {ts['n']} (var Sharpe {_f(ts['var_sharpe'])}); experiments {ex['total']}"
            + (f" ({kinds})" if kinds else "")
            + (f"; validations last 7 days {v7['used']}/{'-' if v7['cap'] is None else v7['cap']}"
               if v7 else "")
            + (f"; CLI holdout peeks {st['holdout_peeks']}" if st.get("holdout_peeks") else "")
            + f"; last research {last['research'] or '-'}, review {last['review'] or '-'},"
              f" sweep {last['sweep'] or '-'}")


def render_status(lab: Lab, now_t: int | None = None) -> str:
    """Compact plain-text status for the terminal."""
    st = lab.status(now_t=now_t)
    slots = st["roster_slots"]
    lines = [f"fibbot status {st['as_of']} | signal review: {st['signal_review']}", DISCLAIMER,
             *(f"!! {a}" for a in _alerts(lab, st)),
             *_data_lines(st), f"roster {slots['paper']}/{slots['max']} paper slots:"]
    for e in st["roster"]:
        a, g = e["account"], e["graduation"]
        lines.append(f"  {e['name']} [{e['strategy_id']}] {e['timeframe']} {e['status']}:"
                     f" {a['days']:.1f}d, closed {a['n_closed']} open {a['n_open']},"
                     f" mean R {a['avg_r']:+.2f}, PF {a['profit_factor']:.2f}, return {a['return']:+.1%},"
                     f" DD {a['max_drawdown']:.1%} -> {g['status']}"
                     + (f" (not yet: {', '.join(g['failed'])})" if g["failed"] else ""))
    if not st["roster"]:
        lines.append("  (empty)")
    lines += [f"  retired {r['name']} [{r['strategy_id']}] {r['since']}: {r['reason']}" for r in st["retired"]]
    names = _names(lab)
    trades = lab.store.paper_trades(limit=5)
    if trades:
        lines.append("recent paper trades:")
        lines += ["  " + " ".join(str(c) for c in _trade_cells(r, names)) for r in trades]
    rv = lab.signal_review_stats()
    lines.append(f"signal review ({rv['mode']}): "
                 + " | ".join(f"{g} n {rv[g]['n']}" + (f" mean R {rv[g]['avg_r']:+.2f}" if rv[g]["n"] else "")
                              for g in REVIEW_GROUPS)
                 + f" | skip-take {_f(rv['skip_minus_take_avg_r'], '{:+.2f}')}")
    lines += [_errors_line(st), _activity_line(st), _budget_line(st)]
    lines += [f"{_when(j['t'])} {j['kind']}: {j['title']}" for j in _research_journal(lab)[:1]]
    return "\n".join(lines)


def _markdown(lab: Lab, st: dict, now_t: int | None) -> str:
    slots, names = st["roster_slots"], _names(lab)
    md = ["# fibbot report", "",
          f"**As of** {st['as_of']} · {len(st['universe'])} coins · signal review: {st['signal_review']}",
          "", f"> **{DISCLAIMER}**", ""]
    for alert in _alerts(lab, st):
        md += [f"**{_cell(alert)}**", ""]
    errs = st.get("system_events_7d") or {}
    if errs.get("n"):
        md += [f"**{errs['n']} system event{'' if errs['n'] == 1 else 's'} (errors, warnings) in the last"
               f" 7 days** (newest {errs['last_t']}: {_cell(errs['last'])}); see Recent system events.", ""]
    md += [f"## Paper roster ({slots['paper']}/{slots['max']} slots)", ""]
    if st["roster"]:
        md += _table(["Strategy", "TF", "Days", "Closed / open", "Mean R", "PF", "Return", "Max DD", "Graduation"],
                     [[f"{e['name']} `{e['strategy_id']}`", e["timeframe"], *_account_cells(e),
                       e["graduation"]["status"]] for e in st["roster"]], right={2, 3, 4, 5, 6, 7})
    else:
        md.append("_No strategies on paper yet._")
    if st["retired"]:
        md += ["", "Retired: " + "; ".join(f"{r['name']} `{r['strategy_id']}` ({r['since']}): {r['reason']}"
                                           for r in st["retired"])]

    md += ["", "## Graduation checks", "",
           "Graduation is a statistical verdict on paper results. It never enables live trading."]
    for e in st["roster"]:
        g = lab.graduation(e["strategy_id"], now_t=now_t)
        md += ["", f"### {e['name']} · {g['status']}", "", g["summary"], "",
               *(_check_line(c) for c in g["checks"])]
    changes = _events(lab, "graduation")
    md += ["", f"### Graduation changes (last {EVENT_ENTRIES})", ""]
    md += ([f"- {_when(j['t'])} UTC · {_cell(j['title'])}" for j in changes] if changes
           else ["_No graduation changes yet._"])

    trades = lab.store.paper_trades(limit=RECENT_TRADES)
    md += ["", f"## Recent paper trades (last {RECENT_TRADES})", ""]
    md += (_table(["Entry (UTC)", "Strategy", "Symbol", "Side", "Entry", "Exit", "Reason", "R", "PnL"],
                  [_trade_cells(r, names) for r in trades], right={4, 5, 7, 8})
           if trades else ["_No paper trades yet._"])

    rv = lab.signal_review_stats()
    md += ["", f"## Brain signal review (mode: {rv['mode']})", "",
           f"{rv['n_signals_reviewed']} of {rv['n_signals']} signals reviewed. Closed paper trades by"
           " the brain's decision on their setup:", ""]
    md += _table(["Brain decision", "Closed trades", "Mean R", "Total R", "Win rate", "PF"],
                 [[g, rv[g]["n"], *(["-"] * 4 if not rv[g]["n"] else    # no trades: no statistics
                                    [_f(rv[g]["avg_r"], "{:+.2f}"), _f(rv[g]["total_r"], "{:+.2f}"),
                                     _f(rv[g]["win_rate"], "{:.0%}"), _f(rv[g]["profit_factor"], "{:.2f}")])]
                  for g in REVIEW_GROUPS], right={1, 2, 3, 4, 5})
    md += ["", f"Skip minus take mean R: {_f(rv['skip_minus_take_avg_r'], '{:+.3f}')} (above 0 means the"
               " setups the brain would skip did better than the ones it would take)."]

    md += ["", "## Latest research and reviews"]
    entries = _research_journal(lab)
    for j in entries:
        md += ["", f"### {_when(j['t'])} UTC · {j['kind']} · {j['title']}", "",
               *(f"> {line}" for line in _clip(j["body"], JOURNAL_BODY_CHARS).splitlines())]
    if not entries:
        md += ["", "_No research or review entries yet._"]

    errors = _events(lab, "system")
    md += ["", f"## Recent system events (last {EVENT_ENTRIES})", "", _errors_line(st)]
    for j in errors:
        md += ["", f"- **{_when(j['t'])} UTC** · {_cell(j['title'])}",
               *(f"  > {line}" for line in _clip(j["body"], EVENT_BODY_CHARS).splitlines())]

    md += ["", f"## Brain notebook (first {NOTEBOOK_LINES} lines)", ""]
    md += [f"> {ln}" for ln in _notebook_head(lab)] or ["_Empty._"]

    md += ["", "## Data and research activity", "",
           *(f"- {ln}" for ln in (*_data_lines(st), _activity_line(st), _budget_line(st))), ""]
    return "\n".join(md)


def write_report(lab: Lab, path=None, now_t: int | None = None) -> Path:
    """Write the Markdown report to `path` (default home/REPORT.md) atomically; returns the path."""
    p = Path(path) if path is not None else home_path(lab.cfg, "REPORT.md")
    atomic_write(p, _markdown(lab, lab.status(now_t=now_t), now_t).encode("utf-8"))
    return p
